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
"""最终的 parquet 重写。

对于每个回合，写入器：

1. 读取暂存的模块输出，
2. 将它们分区为持久切片（PERSISTENT_STYLES）和事件切片
   （EVENT_ONLY_STYLES + style=None 工具调用原子），
3. 确定性地排序每个切片，
4. 将持久切片广播到回合中的每一帧，
5. 对于每一帧，具体化时间戳恰好等于该帧时间戳的事件行的子列表，
6. 丢弃遗留的 ``subtask_index`` 列，
7. 将 parquet 分片原地写回。

写入器不会添加数据集级别的 ``tools`` 列。工具*调用*通过
v3.1 行结构体上现有的 ``tool_calls`` 字段为每个语音原子按行发射。
工具*模式*（``say`` 函数及其参数的描述）是一个固定的代码常量——
下面的 ``SAY_TOOL_SCHEMA``——下游聊天模板消费者直接导入它，
而不是读取冗余的逐行列。

这里强制执行的不变量（并由验证器重新检查）：

- 按回合的持久切片在每一帧上都是字节相同的；
- 帧上的 ``language_events`` 行都具有 ``timestamp == frame_ts``
  （时间戳直接来自源 parquet——从不重新计算）；
- 每行都通过 ``column_for_style(style)``。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
from lerobot.datasets.language import (
    EVENT_ONLY_STYLES,
    PERSISTENT_STYLES,
    column_for_style,
    validate_camera_field,
)
from lerobot.utils.constants import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT

from .reader import EpisodeRecord
from .staging import EpisodeStaging

logger = logging.getLogger(__name__)


# 工具模式常量位于 lerobot.datasets.language —— 单一真实来源。
# 在此重新导出，以便现有的导入
# （``from lerobot.annotations.steerable_pipeline.writer import SAY_TOOL_SCHEMA``）
# 继续工作。
from lerobot.datasets.language import DEFAULT_TOOLS, SAY_TOOL_SCHEMA  # noqa: F401, E402


def _row_persistent_sort_key(row: dict[str, Any]) -> tuple:
    return (float(row["timestamp"]), row.get("style") or "", row.get("role") or "")


def _row_event_sort_key(row: dict[str, Any]) -> tuple:
    # 事件按帧分桶，但在一帧内我们仍然想要确定性
    return (
        row.get("style") or "",
        row.get("role") or "",
        row.get("camera") or "",
    )


def _normalize_row(row: dict[str, Any], style: str | None, *, with_timestamp: bool) -> dict[str, Any]:
    """将暂存行强制转换为语言列结构体形状。

    键顺序匹配 ``PERSISTENT_ROW_FIELDS`` / ``EVENT_ROW_FIELDS``——
    写入器从插入顺序推断 parquet 结构体模式，因此
    ``timestamp``（仅持久行）位于 ``style`` 和 ``camera`` 之间。
    """
    camera = row.get("camera")
    validate_camera_field(style, camera)
    out: dict[str, Any] = {
        "role": str(row["role"]),
        "content": None if row.get("content") is None else str(row["content"]),
        "style": style,
    }
    if with_timestamp:
        out["timestamp"] = float(row["timestamp"])
    out["camera"] = None if camera is None else str(camera)
    out["tool_calls"] = _normalize_tool_calls(row.get("tool_calls"))
    return out


def _normalize_persistent_row(row: dict[str, Any]) -> dict[str, Any]:
    """将暂存行强制转换为持久列的结构体形状。"""
    style = row.get("style")
    if style not in PERSISTENT_STYLES:
        raise ValueError(
            f"persistent slice contains row with non-persistent style {style!r}; "
            "row would be misrouted under column_for_style()"
        )
    if "timestamp" not in row:
        raise ValueError(f"persistent row missing timestamp: {row!r}")
    if "role" not in row:
        # 来自写入器的友好错误，而不是下面的原始 KeyError；
        # 验证器尚未检查 ``role``。
        raise ValueError(f"persistent row missing role: {row!r}")
    return _normalize_row(row, style, with_timestamp=True)


def _normalize_event_row(row: dict[str, Any]) -> dict[str, Any]:
    """将暂存行强制转换为事件列的结构体形状（无时间戳）。"""
    style = row.get("style")
    if style is not None and style not in EVENT_ONLY_STYLES:
        raise ValueError(
            f"event slice contains row with style {style!r}; expected None or one of {EVENT_ONLY_STYLES}"
        )
    if column_for_style(style) != LANGUAGE_EVENTS:
        raise ValueError(f"event row with style {style!r} would not route to language_events")
    if "role" not in row:
        raise ValueError(f"event row missing role: {row!r}")
    return _normalize_row(row, style, with_timestamp=False)


def _normalize_tool_calls(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"tool_calls must be a list or None, got {type(value).__name__}")
    return list(value)


def _validate_atom_invariants(row: dict[str, Any]) -> None:
    """content/tool_calls 至少一个；style=None 隐含 tool_calls。"""
    has_content = row.get("content") is not None
    has_tools = row.get("tool_calls") is not None
    if not (has_content or has_tools):
        raise ValueError(f"row has neither content nor tool_calls: {row!r}")
    if row.get("style") is None and not has_tools:
        raise ValueError(f"style=None requires tool_calls: {row!r}")


def _validate_speech_atom(row: dict[str, Any]) -> None:
    """语音原子：role=assistant，style=None，content=None，say 工具调用。"""
    if row.get("style") is not None:
        return  # 不是语音原子
    if row.get("role") != "assistant":
        raise ValueError(f"speech atom must have role=assistant: {row!r}")
    if row.get("content") is not None:
        raise ValueError(f"speech atom must have content=null: {row!r}")
    tool_calls = row.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        raise ValueError(f"speech atom must have non-empty tool_calls list: {row!r}")
    first = tool_calls[0]
    if not isinstance(first, dict):
        raise ValueError(f"speech atom tool_calls[0] must be a dict: {row!r}")
    if first.get("type") != "function":
        raise ValueError(f"speech atom tool_calls[0].type must be 'function': {row!r}")
    fn = first.get("function") or {}
    if fn.get("name") != "say":
        raise ValueError(f"speech atom tool_calls[0].function.name must be 'say': {row!r}")
    args = fn.get("arguments") or {}
    if not isinstance(args, dict) or "text" not in args or not isinstance(args["text"], str):
        raise ValueError(f"speech atom must carry 'text' string in arguments: {row!r}")


@dataclass
class LanguageColumnsWriter:
    """用两个语言列重写 ``data/chunk-*/file-*.parquet``。"""

    drop_existing_subtask_index: bool = True

    def write_all(
        self,
        records: Sequence[EpisodeRecord],
        staging_dir: Path,
        root: Path,
    ) -> list[Path]:
        episodes_by_path: dict[Path, list[EpisodeRecord]] = defaultdict(list)
        for record in records:
            episodes_by_path[record.data_path].append(record)

        written: list[Path] = []
        for path, eps in episodes_by_path.items():
            self._rewrite_one(path, eps, staging_dir, root)
            written.append(path)
        return written

    def _rewrite_one(
        self,
        path: Path,
        episodes: Sequence[EpisodeRecord],
        staging_dir: Path,
        root: Path,
    ) -> None:
        table = pq.read_table(path)
        n_rows = table.num_rows

        # 确保我们覆盖文件中的每个回合。没有暂存产物的回合
        # 以空标注列表通过——这保持写入器幂等且对部分重运行安全。
        staged_per_ep: dict[int, dict[str, list[dict[str, Any]]]] = {}
        for record in episodes:
            staging = EpisodeStaging(staging_dir, record.episode_index)
            staged_per_ep[record.episode_index] = staging.read_all()

        persistent_by_ep: dict[int, list[dict[str, Any]]] = {}
        events_by_ep_ts: dict[int, dict[float, list[dict[str, Any]]]] = {}

        for ep_index, ep_staged in staged_per_ep.items():
            persistent_rows: list[dict[str, Any]] = []
            event_rows: list[dict[str, Any]] = []  # 携带时间戳直到分桶
            for _module_name, rows in ep_staged.items():
                for row in rows:
                    style = row.get("style")
                    if column_for_style(style) == LANGUAGE_PERSISTENT:
                        persistent_rows.append(row)
                    else:
                        event_rows.append(row)

            persistent_rows.sort(key=_row_persistent_sort_key)
            normalized_persistent = []
            for r in persistent_rows:
                _validate_atom_invariants(r)
                _validate_speech_atom(r)
                normalized_persistent.append(_normalize_persistent_row(r))
            persistent_by_ep[ep_index] = normalized_persistent

            buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for r in event_rows:
                _validate_atom_invariants(r)
                _validate_speech_atom(r)
                ts = float(r["timestamp"])
                buckets[ts].append(_normalize_event_row(r))
            for ts in list(buckets.keys()):
                buckets[ts].sort(key=_row_event_sort_key)
            events_by_ep_ts[ep_index] = buckets

        episode_col = (
            table.column("episode_index").to_pylist() if "episode_index" in table.column_names else None
        )
        ts_col = table.column("timestamp").to_pylist() if "timestamp" in table.column_names else None
        if episode_col is None or ts_col is None:
            raise ValueError(f"{path} is missing 'episode_index' or 'timestamp' — required by the writer.")

        per_row_persistent: list[list[dict[str, Any]]] = []
        per_row_events: list[list[dict[str, Any]]] = []
        for i in range(n_rows):
            ep = episode_col[i]
            ts = float(ts_col[i])
            per_row_persistent.append(persistent_by_ep.get(ep, []))
            buckets = events_by_ep_ts.get(ep, {})
            per_row_events.append(buckets.get(ts, []))

        new_table = self._materialize_table(
            table, per_row_persistent, per_row_events, drop_old=self.drop_existing_subtask_index
        )
        # 每个回合重新发射一个行组（批量 pq.write_table 会将它们折叠成一个）。
        # 写入兄弟 tmp 路径并原子重命名，以便写入中途崩溃不会留下半写的分片。
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        write_table_one_row_group_per_episode(new_table, tmp_path)
        tmp_path.replace(path)

    def _materialize_table(
        self,
        table: pa.Table,
        persistent: list[list[dict[str, Any]]],
        events: list[list[dict[str, Any]]],
        *,
        drop_old: bool,
    ) -> pa.Table:
        cols = []
        names = []
        for name in table.column_names:
            if drop_old and name == "subtask_index":
                continue
            if name in (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS):
                continue  # 我们将重新添加规范版本
            # 去除旧写入器以前发射的任何遗留 ``tools`` 列——
            # 模式不再使用它（常量位于 SAY_TOOL_SCHEMA / DEFAULT_TOOLS）。
            if name == "tools":
                continue
            cols.append(table.column(name))
            names.append(name)

        # 我们让 pyarrow 推断结构体/列表模式，而不是直接从
        # `lerobot.datasets.language` 传递规范类型：该类型
        # 对 `tool_calls` 元素类型使用 `pa.json_()`，
        # 在当前 pyarrow 版本上 `pa.array(..., type=...)` 无法从 Python 列表具体化它。
        # 推断的模式通过 parquet 和 `LeRobotDataset` 正确往返——
        # `tests/datasets/test_language.py` 练习了相同的流程。
        persistent_arr = pa.array(persistent)
        events_arr = pa.array(events)

        cols.extend([persistent_arr, events_arr])
        names.extend([LANGUAGE_PERSISTENT, LANGUAGE_EVENTS])

        return pa.Table.from_arrays(cols, names=names)


def speech_atom(timestamp: float, text: str) -> dict[str, Any]:
    """为事件列构建规范的语音工具调用原子。"""
    return {
        "role": "assistant",
        "content": None,
        "style": None,
        "timestamp": float(timestamp),
        "camera": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "say",
                    "arguments": {"text": text},
                },
            }
        ],
    }
