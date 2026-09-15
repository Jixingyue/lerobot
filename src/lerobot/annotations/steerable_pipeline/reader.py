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
"""Datatrove 形状的读取器。

读取器遍历 ``data/chunk-*/file-*.parquet`` 并为每个回合产生一条记录，包含：

- ``episode_index``：int
- ``frame_timestamps``：tuple[float, ...]
- ``frame_indices``：tuple[int, ...]
- ``episode_task``：str（来自 ``meta/tasks.parquet`` 的规范任务）
- ``data_path``：源 parquet 分片的 pathlib.Path
- ``frames_df``：回合的 pandas.DataFrame 切片（仅按需加载）

这种形状让每个模块可以按回合操作，而无需一次将所有 parquet 行加载到内存中。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from lerobot.datasets.io_utils import load_tasks
from lerobot.datasets.utils import DEFAULT_TASKS_PATH


@dataclass
class EpisodeRecord:
    """读取器产生的按回合记录。"""

    episode_index: int
    episode_task: str
    frame_timestamps: tuple[float, ...]
    frame_indices: tuple[int, ...]
    data_path: Path
    row_offset: int  # 此回合在 parquet 文件中开始的行偏移量
    row_count: int  # 此回合的行数

    # 记忆化的 parquet 切片——在第一次 ``frames_df()`` 调用时填充，
    # 以便来自不同模块的重复查询不会重新读取整个分片。
    _frames_df_cache: Any = field(default=None, init=False, repr=False, compare=False)

    def frames_df(self):  # type: ignore[no-untyped-def]
        """懒加载此回合的 pandas 切片（记忆化）。"""
        if self._frames_df_cache is None:
            import pandas as pd  # noqa: PLC0415  - 延迟导入可选的 dataset 扩展

            table = pq.read_table(self.data_path)
            df: pd.DataFrame = table.to_pandas()
            self._frames_df_cache = df.iloc[self.row_offset : self.row_offset + self.row_count].reset_index(
                drop=True
            )
        return self._frames_df_cache


def reconstruct_subtask_spans(
    rows: Sequence[dict[str, Any]],
    *,
    episode_end_t: float | None = None,
) -> list[dict[str, Any]]:
    """将 ``style="subtask"`` 行转换为 ``{text, start, end}`` 跨度。

    每个跨度的 ``end`` 是下一个跨度的 ``start``。最后一个跨度的
    ``end`` 默认为其自身的 ``start``（零持续时间）——改为传递
    ``episode_end_t`` 以将其扩展到回合的最后一帧，
    这是下游消费者（记忆、插入语边界选择）所期望的。

    由 ``plan`` 模块（计划更新阶段）和 ``interjections`` 模块（插入语锚定）使用，
    它们都需要相同的跨度形状。
    """
    sorted_rows = sorted(
        (r for r in rows if r.get("style") == "subtask"),
        key=lambda r: float(r["timestamp"]),
    )
    spans: list[dict[str, Any]] = []
    for r in sorted_rows:
        t = float(r["timestamp"])
        if spans:
            spans[-1]["end"] = t
        spans.append({"text": r.get("content") or "", "start": t, "end": t})
    if spans and episode_end_t is not None and float(episode_end_t) > spans[-1]["start"]:
        spans[-1]["end"] = float(episode_end_t)
    return spans


def snap_to_frame(t: float, frame_timestamps: Sequence[float]) -> float:
    """将任意浮点数对齐到最近的确切源帧时间戳。

    模块在发射事件样式行时使用此函数，以便行的时间戳与真实的 parquet 帧匹配：
    事件行必须落在确切的帧上，否则写入器执行的逐帧事件查找永远不会匹配它们。
    """
    if not frame_timestamps:
        return float(t)
    nearest = min(frame_timestamps, key=lambda f: abs(f - t))
    return float(nearest)


def _load_tasks_lookup(root: Path) -> dict[int, str]:
    """从 ``meta/tasks.parquet`` 映射 ``task_index -> task``。

    当文件不存在时返回空字典——如果需要，任务描述稍后从视频推导。
    重用库级别的 :func:`lerobot.datasets.io_utils.load_tasks`，
    它返回由任务字符串索引的任务帧，带有 ``task_index`` 列。
    """
    if not (root / DEFAULT_TASKS_PATH).exists():
        return {}
    tasks = load_tasks(root)
    return {int(idx): str(task) for task, idx in zip(tasks.index, tasks["task_index"], strict=True)}


def iter_episodes(root: Path, *, only_episodes: tuple[int, ...] | None = None) -> Iterator[EpisodeRecord]:
    """为 ``root/data/`` 下的每个回合产生 :class:`EpisodeRecord`。

    回合按 ``episode_index`` 升序产生。读取器不假设特定的分块/文件布局：
    它扫描 ``data/`` 下的每个 ``*.parquet`` 并按 ``episode_index`` 分组。
    """
    tasks = _load_tasks_lookup(root)
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))

    only_set = set(only_episodes) if only_episodes is not None else None

    for path in parquet_files:
        yield from _iter_one_path(path, tasks, only_set)


def _iter_one_path(path: Path, tasks: dict[int, str], only_set: set[int] | None) -> Iterator[EpisodeRecord]:
    table = pq.read_table(path)
    names = table.column_names
    if "episode_index" not in names:
        return
    episode_col = table.column("episode_index").to_pylist()
    timestamp_col = (
        table.column("timestamp").to_pylist() if "timestamp" in names else [0.0] * len(episode_col)
    )
    frame_col = (
        table.column("frame_index").to_pylist() if "frame_index" in names else list(range(len(episode_col)))
    )
    task_col = table.column("task_index").to_pylist() if "task_index" in names else None

    def _build(
        ep: int,
        start: int,
        end: int,
        task_idx: int | None,
        ts_buf: list[float],
        fi_buf: list[int],
    ) -> EpisodeRecord | None:
        if only_set is not None and ep not in only_set:
            return None
        task = tasks.get(task_idx, "") if task_idx is not None else ""
        return EpisodeRecord(
            episode_index=ep,
            episode_task=task,
            frame_timestamps=tuple(ts_buf),
            frame_indices=tuple(fi_buf),
            data_path=path,
            row_offset=start,
            row_count=end - start,
        )

    cur_ep: int | None = None
    start_offset = 0
    ts_buf: list[float] = []
    fi_buf: list[int] = []
    cur_task_idx: int | None = None

    for i, ep in enumerate(episode_col):
        if cur_ep is None:
            cur_ep = ep
            start_offset = i
            ts_buf = [timestamp_col[i]]
            fi_buf = [frame_col[i]]
            cur_task_idx = task_col[i] if task_col is not None else None
            continue
        if ep != cur_ep:
            rec = _build(cur_ep, start_offset, i, cur_task_idx, ts_buf, fi_buf)
            if rec is not None:
                yield rec
            cur_ep = ep
            start_offset = i
            ts_buf = [timestamp_col[i]]
            fi_buf = [frame_col[i]]
            cur_task_idx = task_col[i] if task_col is not None else None
        else:
            ts_buf.append(timestamp_col[i])
            fi_buf.append(frame_col[i])

    if cur_ep is not None:
        rec = _build(cur_ep, start_offset, len(episode_col), cur_task_idx, ts_buf, fi_buf)
        if rec is not None:
            yield rec
