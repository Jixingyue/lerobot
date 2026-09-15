#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""进程内执行器，运行标注阶段。

执行器按依赖顺序运行**六个阶段**：

    阶段 1：``plan`` 模块（计划 + 子任务 + 记忆）
    阶段 2：``interjections`` 模块（插入语 + 语音）
    阶段 3：``plan`` 计划更新阶段——在阶段 2 产生的每个插入语时间戳处
             重新运行计划发射
    阶段 4：``vqa`` 模块（VQA）
    阶段 5：验证器
    阶段 6：写入器

阶段 3 解释了为什么 ``plan`` 模块必须在 ``interjections`` 模块之后重新进入——
以在插入语时间戳处刷新 ``plan`` 行。

分布式执行由 Hugging Face Jobs 提供（参见
``lerobot.jobs.annotate``，通过 ``--job.target=<flavor>`` 访问）；
作业内的 pod 调用 ``lerobot-annotate``，后者使用此进程内执行器。
回合级并发由 ``ExecutorConfig.episode_parallelism`` 控制。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AnnotationPipelineConfig
from .reader import EpisodeRecord, iter_episodes
from .staging import EpisodeStaging
from .validator import StagingValidator
from .writer import LanguageColumnsWriter

logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    """跨所有回合的一个流水线阶段的摘要。"""

    name: str
    episodes_processed: int
    episodes_skipped: int


@dataclass
class PipelineRunSummary:
    """由 :meth:`Executor.run` 返回的聚合结果。"""

    phases: list[PhaseResult]
    written_paths: list[Path]
    validation_report: Any  # ValidationReport，保持 Any 以避免导入循环


@dataclass
class Executor:
    """在数据集根目录上进程内运行所有六个阶段。

    回合级并发来自 ``ExecutorConfig.episode_parallelism``（线程池）；
    集群级并发来自在 Hugging Face Job 内运行此执行器。
    测试直接使用存根模块构造执行器。
    """

    config: AnnotationPipelineConfig
    plan: Any  # PlanSubtasksMemoryModule
    interjections: Any  # InterjectionsAndSpeechModule
    vqa: Any  # GeneralVqaModule
    writer: LanguageColumnsWriter
    validator: StagingValidator

    def run(self, root: Path) -> PipelineRunSummary:
        records = list(iter_episodes(root, only_episodes=self.config.only_episodes))
        n = len(records)
        if n == 0:
            raise ValueError(f"No episodes found under {root}/data/")

        print(f"[annotate] {n} episodes total", flush=True)

        staging_dir = self.config.resolved_staging_dir(root)
        staging_dir.mkdir(parents=True, exist_ok=True)

        phases: list[PhaseResult] = []

        # 阶段 1：``plan`` 模块（计划 + 子任务 + 记忆）
        phases.append(self._run_module_phase("plan", records, staging_dir, self.plan))
        # 阶段 2：``interjections`` 模块（插入语 + 语音）。它从同一暂存树
        # 读取 ``plan`` 模块的子任务行，以便将插入语提示词基于正确的本地子任务。
        phases.append(self._run_module_phase("interjections", records, staging_dir, self.interjections))
        # 阶段 3：在插入语时间戳处的 ``plan`` 计划更新阶段。
        phases.append(self._run_plan_update_phase(records, staging_dir))
        # 阶段 4：``vqa`` 模块（VQA）
        phases.append(self._run_module_phase("vqa", records, staging_dir, self.vqa))

        print("[annotate] running validator...", flush=True)
        report = self.validator.validate(records, staging_dir)
        if not report.ok and not self.config.skip_validation:
            raise RuntimeError(f"Staging validation failed: {report.summary()}")
        print(f"[annotate] validator: {report.summary()}", flush=True)

        print(f"[annotate] writing parquet shards into {root}/data/...", flush=True)
        written = self.writer.write_all(records, staging_dir, root)
        print(f"[annotate] wrote {len(written)} shard(s); pipeline complete", flush=True)

        # 保持 meta/info.json 与我们刚写入的 parquet 模式一致。
        # 幂等且累加：现有的用户元数据被保留。
        self._ensure_annotation_metadata_in_info(root)

        return PipelineRunSummary(phases=phases, written_paths=written, validation_report=report)

    @staticmethod
    def _ensure_annotation_metadata_in_info(root: Path) -> None:
        """将语言特征和规范工具写入 ``meta/info.json``。

        ``LanguageColumnsWriter`` 向 parquet 分片添加 ``language_persistent`` 和
        ``language_events``。元数据也必须通告这些列，否则非流式的 ``LeRobotDataset``
        加载会针对旧模式进行转换，并在额外的 parquet 列上失败。
        """
        from lerobot.datasets.io_utils import load_info, write_info  # noqa: PLC0415
        from lerobot.datasets.language import SAY_TOOL_SCHEMA, language_feature_info  # noqa: PLC0415

        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            return
        try:
            info = load_info(root)
        except Exception as exc:  # noqa: BLE001
            print(f"[annotate] could not read {info_path}: {exc}", flush=True)
            return

        changed = False

        merged_features = {**info.features, **language_feature_info()}
        if merged_features != info.features:
            info.features = merged_features
            changed = True

        existing = info.tools or []
        names = {(t.get("function") or {}).get("name") for t in existing if isinstance(t, dict)}
        if SAY_TOOL_SCHEMA["function"]["name"] not in names:
            info.tools = [*existing, SAY_TOOL_SCHEMA]
            changed = True

        if changed:
            write_info(info, root)
            print(
                "[annotate] meta/info.json: "
                f"language_features={list(language_feature_info())}, "
                f"tools={[t['function']['name'] for t in (info.tools or [])]}",
                flush=True,
            )

    def _run_module_phase(
        self,
        name: str,
        records: list[EpisodeRecord],
        staging_dir: Path,
        module: Any,
    ) -> PhaseResult:
        if not module.enabled:
            print(f"[annotate] phase={name} skipped (module disabled)", flush=True)
            return PhaseResult(name=name, episodes_processed=0, episodes_skipped=len(records))
        n = len(records)
        parallelism = max(1, min(self.config.executor.episode_parallelism, n))
        print(
            f"[annotate] phase={name} starting on {n} episode(s) (parallelism={parallelism})",
            flush=True,
        )
        t0 = time.time()

        def _do(idx_record: tuple[int, EpisodeRecord]) -> tuple[int, int, float]:
            i, record = idx_record
            ep_start = time.time()
            staging = EpisodeStaging(staging_dir, record.episode_index)
            module.run_episode(record, staging)
            return i, record.episode_index, time.time() - ep_start

        processed = 0
        if parallelism == 1:
            for i, record in enumerate(records, 1):
                _, ep_idx, elapsed = _do((i, record))
                processed += 1
                print(
                    f"[annotate]   {name} episode {i}/{n} (idx={ep_idx}) done in {elapsed:.1f}s",
                    flush=True,
                )
        else:
            with ThreadPoolExecutor(max_workers=parallelism) as pool:
                futures = [pool.submit(_do, (i, r)) for i, r in enumerate(records, 1)]
                for fut in as_completed(futures):
                    i, ep_idx, elapsed = fut.result()
                    processed += 1
                    print(
                        f"[annotate]   {name} episode {processed}/{n} "
                        f"(idx={ep_idx}, submit_order={i}) done in {elapsed:.1f}s",
                        flush=True,
                    )
        total = time.time() - t0
        print(f"[annotate] phase={name} complete: {processed}/{n} in {total:.1f}s", flush=True)
        return PhaseResult(name=name, episodes_processed=processed, episodes_skipped=0)

    def _run_plan_update_phase(  # noqa: PLR0915
        self, records: list[EpisodeRecord], staging_dir: Path
    ) -> PhaseResult:
        """在 ``interjections`` 模块产生的每个时间戳处重新发射 ``plan`` 行。

        ``plan`` 模块拥有提示词；``interjections`` 模块产生了时间戳。
        因此此阶段使用插入语时间戳回调到 ``plan`` 模块，以便复用其现有的提示词路径。
        """
        if not self.plan.enabled or not self.interjections.enabled:
            return PhaseResult(name="plan_update", episodes_processed=0, episodes_skipped=len(records))
        processed = 0
        for record in records:
            staging = EpisodeStaging(staging_dir, record.episode_index)
            interjection_rows = [
                row for row in staging.read("interjections") if row.get("style") == "interjection"
            ]
            interjection_times = [float(row["timestamp"]) for row in interjection_rows]
            interjection_texts = [str(row.get("content") or "") for row in interjection_rows]
            if interjection_times:
                self.plan.run_plan_updates(record, staging, interjection_times, interjection_texts)
                processed += 1
        # 没有任何插入语的回合被跳过（不需要计划刷新）；
        # 计数它们以便摘要的 processed+skipped == total。
        return PhaseResult(
            name="plan_update",
            episodes_processed=processed,
            episodes_skipped=len(records) - processed,
        )
