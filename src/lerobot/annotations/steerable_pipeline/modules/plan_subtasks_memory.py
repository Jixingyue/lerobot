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
"""``plan`` 模块：子任务分解 + 计划 + 记忆（PERSISTENT 风格）。"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import PlanConfig
from ..frames import (
    FrameProvider,
    null_provider,
    to_contact_sheet_blocks,
)
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord, reconstruct_subtask_spans, snap_to_frame
from ..staging import EpisodeStaging
from ..vlm_client import VlmClient

logger = logging.getLogger(__name__)


# 添加到每个 describe / segment 提示词的前面，让 VLM 知道这些图像是
# 带时间戳的联络印样（contact-sheet）网格，而不是单个视频，并在选择
# 边界时读取烧录在每个图块上的时间戳。
def _contact_sheet_preamble(columns: int) -> str:
    return (
        "CONTACT SHEETS — how to read the images below:\n"
        f"- Each image is a grid of sampled video frames, {columns} per row, "
        "with time running left-to-right then top-to-bottom (row-major).\n"
        "- Each frame has its timestamp burned into the top-left corner, e.g. "
        '"012.50s". Use that printed timestamp (not the tile position) when you '
        "choose start/end times; boundaries should land on or near a printed "
        "timestamp.\n"
        "- Frames continue across grids: an action may span the end of one sheet "
        "and the start of the next, so do not place a boundary just because a new "
        "image begins.\n\n"
    )


# 追加到每个 describe（以及 segment）提示词的末尾。这是一个关于一个事件
# 在哪里结束、下一个事件从哪里开始的视觉化、因果性定义——改编自
# macrodata/refiner——用于锐化切分点，同时由现有提示词继续负责祈使句式
# 的措辞。
_CAUSAL_BOUNDARY_RULES = (
    "EVENT BOUNDARIES — where one event ends and the next begins:\n"
    "- Start a new event whenever the world state changes: an object becomes "
    "held (the gripper closes on it), an object is released (the gripper opens "
    "and it stays put), an object reaches a new location, a lid/door/drawer "
    "changes open/closed state, a tool starts or stops affecting a surface, or "
    "contents visibly move (e.g. poured).\n"
    "- If a single action changes the same state gradually and continuously, "
    "keep it as ONE event — do not split it.\n"
    "- If the same action repeats on different objects or target locations, "
    "treat each repetition as a separate event.\n"
    "- Do NOT create boundaries for idle time, camera motion, hesitation, or "
    "tiny hand adjustments."
)


@dataclass
class PlanSubtasksMemoryModule:
    """生成子任务区间、计划（plan）和记忆（memory）行。

    所有输出都是持久化的（存放在 ``language_persistent`` 中）：

    - ``subtask`` 行：每个区间一行，打上该区间*开始*时间戳
      （对齐到精确帧）。
    - ``plan`` 行：在 ``t=0`` 发射；通过 :meth:`run_plan_updates`
      在每个插话时间戳刷新（由执行器在 ``interjections`` 模块
      完成后调用）。
    - ``memory`` 行：在每个子任务边界（= 从第二个子任务起的子任务
      开始时间戳）发射。
    """

    vlm: VlmClient
    config: PlanConfig
    frame_provider: FrameProvider = field(default_factory=null_provider)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        rows: list[dict[str, Any]] = []
        # 驱动所有 plan 模块提示词的任务：规范的 episode_task，或者当它
        # 为空/占位符时从视频推导出的任务（参见 derive_task_*）。
        effective_task = self._resolve_effective_task(record)
        # t=0 处的 task_aug 行：渲染器轮换 ${task} 所用的各种措辞。
        # 要么是结构化的 5 轴分类法（task_aug_axes.enabled），要么是
        # 自由形式的 n_task_rephrasings；有效任务总是最先发射，
        # 以保证轮换覆盖事实来源的措辞。
        t0 = float(record.frame_timestamps[0]) if record.frame_timestamps else 0.0
        variants: list[str] | None = None
        if self.config.task_aug_axes.enabled and effective_task:
            variants = self._generate_task_aug_by_axes(effective_task, self.config.task_aug_axes)
        elif self.config.n_task_rephrasings > 0 and effective_task:
            variants = self._generate_task_rephrasings(effective_task, n=self.config.n_task_rephrasings)
        if variants is not None:
            rows.extend(self._task_aug_rows([effective_task, *variants], t0))

        subtask_spans = self._generate_subtasks(record, task=effective_task)
        if self.config.subtask_seeded_relabel and subtask_spans:
            subtask_spans = self._seeded_relabel(record, subtask_spans, effective_task)

        # subtask 行
        for span in subtask_spans:
            rows.append(
                {
                    "role": "assistant",
                    "content": span["text"],
                    "style": "subtask",
                    "timestamp": snap_to_frame(span["start"], record.frame_timestamps),
                    "tool_calls": None,
                }
            )
        # 在每个子任务边界（包括 t=0）发射 Plan 行。计划是一个
        # 尚未完成的子任务的编号列表，因此在每个边界重新发射
        # 会让它随着工作推进而收缩——帧 t 处的 ${plan} 恰好就是
        # 剩余要做的事情。
        if self.config.emit_plan:
            for span in subtask_spans:
                boundary_t = snap_to_frame(span["start"], record.frame_timestamps)
                plan_text = self._generate_plan(
                    record, subtask_spans, refresh_t=boundary_t, task=effective_task
                )
                if plan_text is not None:
                    rows.append(
                        {
                            "role": "assistant",
                            "content": plan_text,
                            "style": "plan",
                            "timestamp": float(boundary_t),
                            "tool_calls": None,
                        }
                    )
        # 在除第一个开始之外的每个子任务边界发射 memory 行；
        # 当 ``emit_memory`` 为 False 时完全跳过（仅子任务 / 仅计划）。
        prior_memory = ""
        memory_boundaries = enumerate(subtask_spans[1:], start=1) if self.config.emit_memory else []
        for i, span in memory_boundaries:
            completed = subtask_spans[i - 1]["text"]
            remaining = [s["text"] for s in subtask_spans[i:]]
            mem_text = self._generate_memory(record, prior_memory, completed, remaining, task=effective_task)
            if mem_text:
                ts = snap_to_frame(span["start"], record.frame_timestamps)
                rows.append(
                    {
                        "role": "assistant",
                        "content": mem_text,
                        "style": "memory",
                        "timestamp": ts,
                        "tool_calls": None,
                    }
                )
                prior_memory = mem_text
        staging.write("plan", rows)

    # ------------------------------------------------------------------
    # 任务推导 + 改写
    # ------------------------------------------------------------------

    _PLACEHOLDER_TASKS: frozenset[str] = frozenset(
        {
            "debug",
            "test",
            "tbd",
            "todo",
            "n/a",
            "na",
            "untitled",
            "unnamed",
            "default",
            "placeholder",
        }
    )

    def _resolve_effective_task(self, record: EpisodeRecord) -> str:
        """决定本片段中驱动 ``plan`` 模块的任务字符串。

        除非 ``derive_task_from_video`` 另有指示（参见配置的
        docstring），否则返回用户提供的 ``record.episode_task``。
        如果视频推导失败，则优雅地回退到规范任务。
        """
        canonical = (record.episode_task or "").strip()
        mode = (self.config.derive_task_from_video or "off").strip().lower()
        if mode == "always":
            derived = self._derive_task_from_video(record)
            return derived or canonical
        if mode == "if_short" and self._task_seems_bad(canonical):
            derived = self._derive_task_from_video(record)
            if derived:
                return derived
        return canonical

    def _task_seems_bad(self, task: str) -> bool:
        if not task:
            return True
        if len(task.split()) < int(self.config.derive_task_min_words):
            return True
        return task.lower() in self._PLACEHOLDER_TASKS

    @staticmethod
    def _task_aug_rows(phrasings: Sequence[str], t0: float) -> list[dict[str, Any]]:
        """在 ``t0`` 构建去重后的 ``task_aug`` 行（role=user）。"""
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for phrasing in phrasings:
            key = phrasing.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(
                {"role": "user", "content": key, "style": "task_aug", "timestamp": t0, "tool_calls": None}
            )
        return rows

    # ------------------------------------------------------------------
    # VLM 调用辅助函数——每个 plan 模块提示词都遵循相同的形态：
    # 构建 messages → 单次 VLM 调用 → 提取一个命名字段。
    # ------------------------------------------------------------------

    def _vlm_field(self, messages: list[dict[str, Any]], field: str) -> Any:
        """执行单次 VLM 调用并返回 ``result[field]`` 或 ``None``。

        集中处理每个提示词调用点都需要的
        ``vlm.generate_json([m])[0]`` + ``isinstance(dict)`` 这套流程。
        """
        result = self.vlm.generate_json([messages])[0]
        if isinstance(result, dict):
            return result.get(field)
        return None

    @staticmethod
    def _text_message(text: str) -> list[dict[str, Any]]:
        """为 ``generate_json`` 包装的一次性纯文本用户消息。"""
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    def _video_message(
        self,
        record: EpisodeRecord,
        prompt: str,
        window: tuple[float, float] | None = None,
    ) -> list[dict[str, Any]]:
        """将（可选加窗的）联络印样与 ``prompt`` 组合成的用户消息。

        提示词总是带有一段关于如何阅读带时间戳网格的简短说明
        作为前缀，使模型将它们视为一个有序的帧序列，而不是
        互不相关的图像。
        """
        prompt = _contact_sheet_preamble(self.config.contact_sheet_columns) + prompt
        content = [*self._episode_video_block(record, window=window), {"type": "text", "text": prompt}]
        return [{"role": "user", "content": content}]

    def _derive_task_from_video(self, record: EpisodeRecord) -> str | None:
        """在完全不提供任务提示的情况下询问 VLM"这个视频是关于什么的"。"""
        text = self._vlm_field(self._video_message(record, load_prompt("plan_video_task")), "task")
        return text.strip() if isinstance(text, str) and text.strip() else None

    def _generate_task_rephrasings(self, base_task: str, *, n: int) -> list[str]:
        """生成 ``base_task`` 的 ``n`` 个纯文本改写。"""
        if n <= 0 or not base_task:
            return []
        prompt = load_prompt("plan_task_rephrasings").format(base_task=base_task, n=n)
        raw = self._vlm_field(self._text_message(prompt), "rephrasings")
        if not isinstance(raw, list):
            return []
        out = [item.strip().strip('"').strip("'") for item in raw if isinstance(item, str)]
        return [s for s in out if s][:n]

    # ------------------------------------------------------------------
    # 结构化 5 轴任务增强（EgoMimic 风格的分类法）
    # ------------------------------------------------------------------

    def _generate_task_aug_by_axes(self, base_task: str, axes_cfg: Any) -> list[str]:
        """一次 VLM 调用 → 沿 5 轴分类法生成变体。

        所有轴的变体被展平为一个列表（下游流水线不需要知道
        按轴分桶的情况——每个变体都会成为一个 ``task_aug`` 行）。
        为了可复现性保持顺序：synonym_paraphrase 在前，
        然后是 omit_arm，接着是 omit_orientation，再是
        omit_grasp_method，最后是 combined_omissions。
        """
        if not base_task:
            return []
        prompt = load_prompt("plan_task_aug_axes").format(
            base_task=base_task,
            n_synonym=axes_cfg.synonym_paraphrase,
            n_omit_arm=axes_cfg.omit_arm,
            n_omit_orientation=axes_cfg.omit_orientation,
            n_omit_grasp_method=axes_cfg.omit_grasp_method,
            n_combined=axes_cfg.combined_omissions,
        )
        result = self.vlm.generate_json([self._text_message(prompt)])[0]
        if not isinstance(result, dict):
            return []
        ordered_axes = (
            "synonym_paraphrase",
            "omit_arm",
            "omit_orientation",
            "omit_grasp_method",
            "combined_omissions",
        )
        flat: list[str] = []
        seen: set[str] = set()
        for axis in ordered_axes:
            entries = result.get(axis)
            if not isinstance(entries, list):
                continue
            for item in entries:
                if not isinstance(item, str):
                    continue
                key = item.strip().strip('"').strip("'")
                if not key or key in seen:
                    continue
                seen.add(key)
                flat.append(key)
        return flat

    def _episode_video_block(
        self, record: EpisodeRecord, window: tuple[float, float] | None = None
    ) -> list[dict[str, Any]]:
        """用于 describe / segmentation 提示词的带时间戳联络印样。

        总是将（可选加窗的）片段渲染为联络印样：按
        ``frames_per_second`` 采样的帧被打包成带时间戳的 JPEG
        网格。``max_frames_per_prompt`` 限制帧数；超出该限制的
        完整片段会在上游的 :meth:`_generate_subtasks` 中加窗，
        使每次调用都保持在预算之内，同时完整片段保持其采样
        密度。

        当给定 ``window=(w0, w1)`` 时，徽标是相对于窗口的
        （``ts - w0``），以匹配 segmentation 提示词所使用的
        窗口相对时间框架（之后区间会被偏移回绝对时间）。
        """
        if not record.frame_timestamps:
            return []
        if window is not None:
            w0, w1 = float(window[0]), float(window[1])
            dur = max(0.0, w1 - w0)
            n = max(1, int(round(dur * self.config.frames_per_second)) + 1)
            n = min(n, self.config.max_frames_per_prompt)
            if n <= 1 or dur <= 0.0:
                timestamps = [0.5 * (w0 + w1)]
            else:
                step = dur / (n - 1)
                timestamps = [w0 + i * step for i in range(n)]
            frames = self.frame_provider.frames_at(record, timestamps)
            rel = [ts - w0 for ts in timestamps[: len(frames)]]
            return self._contact_sheet_blocks(frames, rel)
        episode_duration = record.frame_timestamps[-1] - record.frame_timestamps[0]
        n = max(1, int(round(episode_duration * self.config.frames_per_second)) + 1)
        n = min(n, self.config.max_frames_per_prompt)
        timestamps = self._uniform_episode_timestamps(record, n)
        frames = self.frame_provider.frames_at(record, timestamps)
        return self._contact_sheet_blocks(frames, timestamps[: len(frames)])

    @staticmethod
    def _uniform_episode_timestamps(record: EpisodeRecord, n: int) -> list[float]:
        """均匀覆盖 ``[t0, t_last]`` 的 ``n`` 个片段相对时间戳。"""
        ts = record.frame_timestamps
        if n >= len(ts):
            return [float(t) for t in ts]
        t0, t_last = float(ts[0]), float(ts[-1])
        if t_last <= t0 or n <= 1:
            return [t0] * max(1, n)
        step = (t_last - t0) / (n - 1)
        return [t0 + i * step for i in range(n)]

    def _contact_sheet_blocks(self, frames: list[Any], timestamps: list[float]) -> list[dict[str, Any]]:
        """从解码后的帧构建带时间戳的联络印样图像块。"""
        return to_contact_sheet_blocks(
            frames,
            timestamps,
            columns=self.config.contact_sheet_columns,
            frames_per_sheet=self.config.contact_sheet_frames_per_sheet,
            frame_width=self.config.contact_sheet_frame_width,
            quality=self.config.contact_sheet_quality,
        )

    def run_plan_updates(
        self,
        record: EpisodeRecord,
        staging: EpisodeStaging,
        interjection_times: Sequence[float],
        interjection_texts: Sequence[str] | None = None,
    ) -> None:
        """在每个插话时间戳追加额外的 ``plan`` 行。

        计划仅在用户插话时刷新（事件驱动）。插话文本会被
        转发到提示词中，使刷新后的计划反映用户的纠正。
        """
        if not self.config.emit_plan:
            return
        existing = staging.read("plan")
        # 传入最后一帧的时间戳，使最后一个区间是闭合的（否则其
        # end == start，持续时间为零，其内部的刷新会被遗漏）。
        episode_end_t = float(record.frame_timestamps[-1]) if record.frame_timestamps else None
        spans = reconstruct_subtask_spans(existing, episode_end_t=episode_end_t)
        already_planned: set[float] = {float(r["timestamp"]) for r in existing if r.get("style") == "plan"}
        new_rows = list(existing)

        texts: list[str | None] = (
            [None] * len(interjection_times)
            if interjection_texts is None
            else [str(t) if t else None for t in interjection_texts]
        )
        for raw_t, inter_text in zip(interjection_times, texts, strict=True):
            t = snap_to_frame(raw_t, record.frame_timestamps)
            if t in already_planned:
                continue
            already_planned.add(t)
            plan_text = self._generate_plan(record, spans, refresh_t=t, interjection=inter_text)
            if plan_text is not None:
                new_rows.append(
                    {
                        "role": "assistant",
                        "content": plan_text,
                        "style": "plan",
                        "timestamp": t,
                        "tool_calls": None,
                    }
                )
        staging.write("plan", new_rows)

    def _generate_subtasks(self, record: EpisodeRecord, *, task: str | None = None) -> list[dict[str, Any]]:
        """生成子任务区间，可选通过多次调用的质量链。

        单次调用（默认）：观看视频 → 发射子任务 JSON。

        多次调用（可选启用，质量更高，VLM 调用更多）：
          1. ``subtask_describe_first`` —— 一个落地（grounding）过程，
             仅叙述可见的内容（此时尚不对子任务做 JSON 承诺）；
             其描述会被注入到 segmentation 提示词中，使模型对
             自己有依据的观察进行切分，而不是对任务文本做
             模式匹配。
          2. segmentation —— 发射子任务 JSON（与之前相同）。
        """
        if record.row_count == 0 or not record.frame_timestamps:
            return []
        episode_duration = record.frame_timestamps[-1] - record.frame_timestamps[0]
        effective_task = task if task is not None else record.episode_task

        # ---- 自动加窗（保持完整采样密度）--------------------------
        # 联络印样很便宜，但按 ``frames_per_second`` 采样的整个长
        # 片段仍可能超过 ``max_frames_per_prompt``。当超过时，
        # 切分为恰好那么多帧的连续窗口（每个窗口一次
        # describe→segment 调用，仍保持完整采样密度），然后
        # 合并 + 拼接——这样任意长度的片段都能以完整密度被覆盖，
        # 而不是被降采样成一次稀疏调用。
        fps = max(1e-6, float(self.config.frames_per_second))
        n_whole = int(round(episode_duration * fps)) + 1
        if n_whole > self.config.max_frames_per_prompt:
            window_s = self.config.max_frames_per_prompt / fps
            return self._generate_subtasks_windowed(record, effective_task, window_s)

        # ---- 过程 1（可选）：落地描述 ------------------------------
        observation_block = ""
        if getattr(self.config, "subtask_describe_first", False):
            description = self._describe_episode(record, effective_task)
            if description:
                observation_block = (
                    "You watched this video and described, chronologically, "
                    "ONLY what the robot actually does:\n"
                    f'"""{description}"""\n\n'
                    "Segment THAT grounded description (cross-checked against "
                    "the video) into atomic subtasks. Do not introduce any "
                    "action that is not in your description above.\n\n"
                )

        # ---- 过程 2：segmentation ------------------------------------
        prompt = self._with_causal_rules(
            load_prompt("plan_subtasks").format(
                episode_task=effective_task,
                min_subtask_seconds=self.config.min_subtask_seconds,
                max_steps=self.config.plan_max_steps,
                episode_duration=f"{episode_duration:.3f}",
                observation_block=observation_block,
            )
        )
        spans = self._vlm_field(self._video_message(record, prompt), "subtasks")
        cleaned = self._clean_spans(spans, record)
        if not cleaned:
            return []

        # ---- 完整片段覆盖拼接 --------------------------------------
        # VLM 可能从 t0 之后才开始，或者留下空隙，导致某些帧没有
        # 有效的子任务。总是拼接成连续的 [t0, t_last] 覆盖。
        cleaned = self._stitch_full_coverage(cleaned, record)

        return cleaned

    def _seeded_relabel(
        self, record: EpisodeRecord, spans: list[dict[str, Any]], task: str
    ) -> list[dict[str, Any]]:
        """使用 prev/current/next 区段的联络印样重新标注每个区间。

        边界保持固定；只精炼 ``text``。原始（"种子"）标签作为强
        先验传入，使模型对其进行验证并做最小限度的纠正，而不是
        从头重新描述——即 macrodata 的种子重标注步骤。每个区间
        一次 VLM 调用。
        """
        n = len(spans)
        out: list[dict[str, Any]] = []
        for i, span in enumerate(spans):
            content: list[dict[str, Any]] = []
            if i > 0:
                content += self._segment_sheet(record, spans[i - 1])
            content += self._segment_sheet(record, span)
            if i < n - 1:
                content += self._segment_sheet(record, spans[i + 1])
            prompt = load_prompt("plan_subtask_relabel").format(
                episode_task=task,
                seed_label=span["text"],
                segment_index=i + 1,
                segment_count=n,
                start=float(span["start"]),
                end=float(span["end"]),
            )
            content.append({"type": "text", "text": prompt})
            label = self._vlm_field([{"role": "user", "content": content}], "label")
            text = label.strip() if isinstance(label, str) and label.strip() else span["text"]
            out.append({**span, "text": text})
        return out

    def _segment_sheet(self, record: EpisodeRecord, span: dict[str, Any]) -> list[dict[str, Any]]:
        """单个区间的联络印样块：最多均匀采样 N 帧。"""
        s, e = float(span["start"]), float(span["end"])
        n = max(1, int(self.config.subtask_relabel_frames))
        if e <= s or n == 1:
            timestamps = [s]
        else:
            step = (e - s) / (n - 1)
            timestamps = [s + i * step for i in range(n)]
        frames = self.frame_provider.frames_at(record, timestamps)
        return self._contact_sheet_blocks(frames, timestamps[: len(frames)])

    def _generate_subtasks_windowed(
        self, record: EpisodeRecord, task: str, window_s: float
    ) -> list[dict[str, Any]]:
        """以固定 fps 在固定长度窗口内进行子任务生成。

        将 ``[t0, t_last]`` 切分为 ``window_s`` 秒的连续窗口，
        在每个窗口自己的帧上（按 ``frames_per_second`` 采样）
        运行 describe -> segment 链，将每个窗口的区间偏移回
        绝对的片段时间，然后合并 + 拼接成连续的整片段覆盖。
        """
        t0 = float(record.frame_timestamps[0])
        t_last = float(record.frame_timestamps[-1])
        all_spans: list[dict[str, Any]] = []
        w0 = t0
        n_windows = 0
        while w0 < t_last - 1e-6:
            w1 = min(w0 + window_s, t_last)
            all_spans.extend(self._subtasks_for_window(record, task, w0, w1))
            n_windows += 1
            w0 = w1
        logger.info(
            "episode %d: windowed subtask gen over %d window(s) of %.1fs -> %d raw spans",
            record.episode_index,
            n_windows,
            window_s,
            len(all_spans),
        )
        # 跨窗口合并：钳制到绝对片段范围内，排序，并将开始时间
        # 对齐到互不相同的帧（处理任何边界碰撞）。
        cleaned = self._clean_spans(all_spans, record)
        if not cleaned:
            return []
        return self._stitch_full_coverage(cleaned, record)

    def _subtasks_for_window(
        self, record: EpisodeRecord, task: str, w0: float, w1: float
    ) -> list[dict[str, Any]]:
        """在一个 ``[w0, w1]`` 窗口上运行 describe -> segment。

        模型在窗口相对时间 ``[0, L]`` 内工作（它将窗口感知为
        从 0 开始的片段）；区间在返回前被偏移回绝对的
        ``[w0, w1]``。
        """
        window = (w0, w1)
        win_len = max(0.0, w1 - w0)

        observation_block = ""
        if getattr(self.config, "subtask_describe_first", False):
            description = self._describe_episode(record, task, window=window)
            if description:
                observation_block = (
                    "You watched this video clip and described, chronologically, "
                    "ONLY what the robot actually does:\n"
                    f'"""{description}"""\n\n'
                    "Segment THAT grounded description (cross-checked against "
                    "the clip) into atomic subtasks. Do not introduce any "
                    "action that is not in your description above.\n\n"
                )

        prompt = self._with_causal_rules(
            load_prompt("plan_subtasks").format(
                episode_task=task,
                min_subtask_seconds=self.config.min_subtask_seconds,
                max_steps=self.config.plan_max_steps,
                episode_duration=f"{win_len:.3f}",
                observation_block=observation_block,
            )
        )
        spans = self._vlm_field(self._video_message(record, prompt, window=window), "subtasks")
        # 窗口相对钳制；此时尚不做帧对齐去重（在合并后的绝对
        # 时间集合上统一进行）。
        cleaned = self._clean_spans(spans, record, bounds=(0.0, win_len), dedupe=False)
        if not cleaned:
            return []

        # 将窗口相对的区间偏移回绝对的片段时间。
        for s in cleaned:
            s["start"] = w0 + float(s["start"])
            s["end"] = w0 + float(s["end"])
        return cleaned

    def _stitch_full_coverage(
        self, spans: list[dict[str, Any]], record: EpisodeRecord
    ) -> list[dict[str, Any]]:
        """使子任务区间无缝铺满整个片段。

        * 第一个子任务从片段的第一帧 ``t0`` 开始（第一个有标签
          的动作之前的任何空闲/接近过程都被并入其中），因此
          每个早期帧都有一个有效的子任务。
        * 每个子任务的 ``end`` 对齐到下一个子任务的 ``start``
          （区间之间的空隙被闭合），最后一个子任务的 ``end``
          延伸到最后一帧 ``t_last``。

        开始时间在其他方面保持 VLM 产生的（已对齐到帧且互不相同
        的）值——只有第一个开始时间被拉回 ``t0``，这不会与后面
        的区间冲突，因为它本来就是最早的。完全确定性；在 VLM
        过程之后运行。
        """
        if not spans or not record.frame_timestamps:
            return spans
        t0 = float(record.frame_timestamps[0])
        t_last = float(record.frame_timestamps[-1])
        spans = sorted(spans, key=lambda s: float(s["start"]))
        spans[0]["start"] = t0
        for i in range(len(spans) - 1):
            spans[i]["end"] = float(spans[i + 1]["start"])
        spans[-1]["end"] = t_last
        for s in spans:
            if float(s["end"]) < float(s["start"]):
                s["end"] = float(s["start"])
        return spans

    @staticmethod
    def _with_causal_rules(prompt: str) -> str:
        """将因果事件边界规则追加到 describe/segment 提示词中。"""
        return f"{prompt}\n\n{_CAUSAL_BOUNDARY_RULES}"

    def _clean_spans(
        self,
        spans: Any,
        record: EpisodeRecord,
        bounds: tuple[float, float] | None = None,
        dedupe: bool = True,
    ) -> list[dict[str, Any]]:
        """将原始 VLM 子任务区间钳制 / 排序 /（可选）去重为有效行。

        ``bounds`` 覆盖钳制范围——在清理窗口相对区间时传入窗口
        的 ``(w_lo, w_hi)``，或者留 ``None`` 以钳制到整个片段
        ``[t0, t_last]``。``dedupe`` 执行帧对齐的不同开始时间
        步骤；对窗口相对区间跳过它（帧对齐在合并后的绝对时间
        集合上只做一次）。
        """
        if not spans:
            return []
        if bounds is not None:
            lo, hi = float(bounds[0]), float(bounds[1])
        else:
            lo = record.frame_timestamps[0]
            hi = record.frame_timestamps[-1]
        cleaned: list[dict[str, Any]] = []
        for span in spans:
            try:
                start = float(span["start"])
                end = float(span["end"])
                text = str(span["text"]).strip()
            except (KeyError, ValueError, TypeError):
                continue
            start = max(lo, min(start, hi))
            end = max(lo, min(end, hi))
            if end < start:
                start, end = end, start
            if not text:
                continue
            cleaned.append({"text": text, "start": start, "end": end})
        cleaned.sort(key=lambda s: s["start"])
        if dedupe:
            return self._dedupe_starts_to_distinct_frames(cleaned, record)
        return cleaned

    def _describe_episode(
        self, record: EpisodeRecord, task: str, window: tuple[float, float] | None = None
    ) -> str:
        """落地过程：对（加窗的）视频进行自由形式的按时间顺序描述。"""
        prompt = self._with_causal_rules(load_prompt("plan_subtask_describe").format(episode_task=task))
        text = self._vlm_field(self._video_message(record, prompt, window=window), "description")
        return text.strip() if isinstance(text, str) and text.strip() else ""

    @staticmethod
    def _dedupe_starts_to_distinct_frames(
        spans: list[dict[str, Any]], record: EpisodeRecord
    ) -> list[dict[str, Any]]:
        """将落在同一帧上的子任务开始时间挪到互不相同的帧上。

        两个连续的 VLM 区间，如果其 ``start``（经过
        :func:`snap_to_frame` 之后）舍入到同一个源帧，就会在
        相同的持久化时间戳上发射两行 ``style=subtask``。
        训练时渲染器的 ``active_at(t, style=subtask)`` 解析器
        无法消歧，会抛出 ``Ambiguous resolver for style='subtask'``。

        遍历（按开始时间排序的）区间，将每个区间对齐到其帧，
        如果对齐的帧已被占用，就把该区间推到下一个未使用的帧，
        使两个子任务都能以不同的时间戳保留下来。如果在找到空闲
        帧之前片段就结束了，则丢弃末尾的区间并发出警告——这比
        污染渲染要好。
        """
        if not spans:
            return spans
        frames = record.frame_timestamps
        if not frames:
            return spans
        used: set[float] = set()
        out: list[dict[str, Any]] = []
        for span in spans:
            ts = snap_to_frame(span["start"], frames)
            if ts in used:
                next_ts = next((f for f in frames if f > ts and f not in used), None)
                if next_ts is None:
                    logger.warning(
                        "episode %d: subtask %r snapped to occupied frame "
                        "%.3f and no free later frame exists — dropping",
                        record.episode_index,
                        span.get("text"),
                        ts,
                    )
                    continue
                ts = next_ts
            used.add(ts)
            new_span = {**span, "start": ts}
            if float(new_span.get("end", ts)) < ts:
                new_span["end"] = ts
            out.append(new_span)
        return out

    def _generate_plan(
        self,
        record: EpisodeRecord,  # noqa: ARG002  (kept for signature stability)
        subtask_spans: Sequence[dict[str, Any]],
        *,
        refresh_t: float | None = None,
        interjection: str | None = None,  # noqa: ARG002
        task: str | None = None,  # noqa: ARG002
    ) -> str | None:
        """确定性计划 = *尚未完成*的子任务的编号列表。

        不调用 VLM：简单的编号列表使计划与即将到来的子任务
        保持一致（旧的 VLM"紧凑分层计划"提示词每个片段/每次
        刷新都要一次往返，而且可能产生偏差）。

            1. <subtask 1>
            2. <subtask 2>

        在 ``refresh_t`` 处刷新时（来自 ``run_plan_updates`` 在
        插话时的调用，以及 ``run_episode`` 在每个边界的调用），
        只包含在 ``refresh_t`` 或之后开始的子任务——因此它总是
        描述剩余要做的事情。
        """
        if not subtask_spans:
            return None
        remaining = [
            s for s in subtask_spans if refresh_t is None or float(s.get("start", 0.0)) >= float(refresh_t)
        ]
        if not remaining:
            # 在一次较晚的刷新中已越过最后一个子任务边界——没有
            # 剩余可计划的内容；返回 None 让调用方跳过该行。
            return None
        return "\n".join(f"{i}. {span.get('text', '').strip()}" for i, span in enumerate(remaining, start=1))

    def _generate_memory(
        self,
        record: EpisodeRecord,
        prior_memory: str,
        completed: str,
        remaining: Sequence[str],
        *,
        task: str | None = None,
    ) -> str:
        prompt = load_prompt("plan_memory").format(
            episode_task=(task if task is not None else record.episode_task),
            prior_memory=prior_memory or "(none)",
            completed_subtask=completed,
            remaining_subtasks=", ".join(remaining) if remaining else "(none)",
        )
        memory = self._vlm_field(self._text_message(prompt), "memory")
        return memory.strip() if isinstance(memory, str) else ""
