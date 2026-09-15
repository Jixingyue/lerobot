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

from __future__ import annotations

from typing import Literal

import datasets
import pyarrow as pa

from lerobot.utils.constants import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT

LANGUAGE_COLUMNS = (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS)
PERSISTENT_ROW_FIELDS = ("role", "content", "style", "timestamp", "camera", "tool_calls")
EVENT_ROW_FIELDS = ("role", "content", "style", "camera", "tool_calls")

CORE_STYLES = {
    "subtask",
    "plan",
    "memory",
    "motion",
    "interjection",
    "vqa",
    "trace",
    "task_aug",
}
# 项目本地的风格可以在导入时注册，方法是在调用 ``column_for_style`` 之前
# 将其追加到 ``EXTENDED_STYLES`` 中。此处添加的任何内容都会与
# ``CORE_STYLES`` 一样被视为已知风格，供解析器验证使用。
# 默认为空——由同时扩展了 ``PERSISTENT_STYLES`` 或 ``EVENT_ONLY_STYLES``
# 的下游模块填充，以声明新风格所属的列。
EXTENDED_STYLES: set[str] = set()
STYLE_REGISTRY = CORE_STYLES | EXTENDED_STYLES

PERSISTENT_STYLES = {"subtask", "plan", "memory", "motion", "task_aug"}
EVENT_ONLY_STYLES = {"interjection", "vqa", "trace"}

# 这些风格的 ``content`` 基于特定的相机视角。这些风格的行必须
# 携带非空的 ``camera``，其值引用某个 ``observation.images.*``
# 特征键。所有其他风格的行必须满足 ``camera=None``。``motion``
# 有意不包含在此集合中：运动原语是用机器人坐标系
# （关节/笛卡尔）术语描述的，而不是像素空间，因此它们
# 与相机无关。``trace`` 是像素轨迹事件风格，确实
# 依赖于视角。不过 ``camera`` 字段同样存在于
# ``PERSISTENT_ROW_FIELDS`` 中，以便 schema、验证器和解析器
# 在两列上的行为保持对称；目前实践中持久化行
# 始终满足 ``camera=None``。
VIEW_DEPENDENT_STYLES = {"vqa", "trace"}

LanguageColumn = Literal["language_persistent", "language_events"]


def _json_arrow_type() -> pa.DataType:
    """返回 Arrow 的 JSON 类型，在较旧的 pyarrow 上回退为 ``string``。"""
    return pa.json_() if hasattr(pa, "json_") else pa.string()


def _json_feature() -> object:
    """返回 HF ``datasets`` 的 JSON 特征，回退为字符串值。"""
    return datasets.Json() if hasattr(datasets, "Json") else datasets.Value("string")


def language_persistent_row_arrow_type() -> pa.StructType:
    """返回单个持久化语言行的 Arrow struct 类型。

    持久化行携带自己的 ``timestamp``，因为它们表示在特定时刻
    生效并持续有效直到被取代的状态。
    ``timestamp`` 为 ``float32``，与 LeRobotDataset
    用于帧数据的时间戳 dtype 一致。
    """
    return pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=True),
            pa.field("style", pa.string(), nullable=True),
            pa.field("timestamp", pa.float32(), nullable=False),
            pa.field("camera", pa.string(), nullable=True),
            pa.field("tool_calls", pa.list_(_json_arrow_type()), nullable=True),
        ]
    )


def language_event_row_arrow_type() -> pa.StructType:
    """返回单个事件语言行的 Arrow struct 类型。

    事件行没有 ``timestamp`` 字段：每个事件都存储在
    帧时间戳等于该事件触发时间的数据集行上。
    """
    return pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=True),
            pa.field("style", pa.string(), nullable=True),
            pa.field("camera", pa.string(), nullable=True),
            pa.field("tool_calls", pa.list_(_json_arrow_type()), nullable=True),
        ]
    )


def language_persistent_arrow_type() -> pa.ListType:
    """返回 ``language_persistent`` 列的 Arrow list 类型。"""
    return pa.list_(language_persistent_row_arrow_type())


def language_events_arrow_type() -> pa.ListType:
    """返回 ``language_events`` 列的 Arrow list 类型。"""
    return pa.list_(language_event_row_arrow_type())


def language_persistent_row_feature() -> dict[str, object]:
    """返回持久化语言行的 HF ``datasets`` 特征映射。"""
    return {
        "role": datasets.Value("string"),
        "content": datasets.Value("string"),
        "style": datasets.Value("string"),
        "timestamp": datasets.Value("float32"),
        "camera": datasets.Value("string"),
        "tool_calls": datasets.List(_json_feature()),
    }


def language_event_row_feature() -> dict[str, object]:
    """返回事件语言行的 HF ``datasets`` 特征映射。"""
    return {
        "role": datasets.Value("string"),
        "content": datasets.Value("string"),
        "style": datasets.Value("string"),
        "camera": datasets.Value("string"),
        "tool_calls": datasets.List(_json_feature()),
    }


def language_persistent_column_feature() -> datasets.List:
    """返回 ``language_persistent`` 列的 HF ``datasets`` 特征。"""
    return datasets.List(language_persistent_row_feature())


def language_events_column_feature() -> datasets.List:
    """返回 ``language_events`` 列的 HF ``datasets`` 特征。"""
    return datasets.List(language_event_row_feature())


def language_feature_info() -> dict[str, dict]:
    """返回两个语言列的 ``info["features"]`` 条目。"""
    return {
        LANGUAGE_PERSISTENT: {"dtype": "language", "shape": (1,), "names": None},
        LANGUAGE_EVENTS: {"dtype": "language", "shape": (1,), "names": None},
    }


def is_language_column(key: str) -> bool:
    """如果 ``key`` 是数据集的语言列名之一，则返回 ``True``。"""
    return key in LANGUAGE_COLUMNS


def is_view_dependent_style(style: str | None) -> bool:
    """如果 ``style`` 的行必须标记 ``camera`` 键，则返回 ``True``。"""
    return style in VIEW_DEPENDENT_STYLES


def validate_camera_field(style: str | None, camera: str | None) -> None:
    """强制 ``camera`` 不变式：当且仅当 ``style`` 依赖视角时才必须提供。

    如果依赖视角的风格缺少 ``camera``，或者非依赖视角的风格
    携带了 ``camera``，则抛出 ``ValueError``。流水线写入器和验证器
    应对每个输出的行调用此函数。
    """
    if is_view_dependent_style(style):
        if not camera:
            raise ValueError(
                f"Rows of view-dependent style {style!r} require a non-empty 'camera' "
                f"field referencing an 'observation.images.*' feature key."
            )
    elif camera is not None:
        raise ValueError(f"Rows of style {style!r} must have camera=None; got camera={camera!r}.")


# --- Tool registry --------------------------------------------------------
# 数据集上声明的工具以 OpenAI 风格函数 schema 列表的形式
# 存放在 ``meta/info.json["tools"]`` 中。运行时/训练栈通过
# :class:`LeRobotDatasetMetadata.tools` 读取它们
# （当数据集未声明任何工具时，以这些常量作为
# 回退）。实现位于 :mod:`lerobot.tools` 下
# （每个工具一个文件）；编写指南参见
# ``docs/source/tools.mdx``。

SAY_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "say",
        "description": "Speak a short utterance to the user via the TTS executor.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The verbatim text to speak.",
                }
            },
            "required": ["text"],
        },
    },
}
"""由可控标注流水线（PR 2 Module 2）输出的 ``say`` 工具的
标准 schema。单一事实来源——PR 2 的
写入器、PR 3 的运行时工具注册表以及数据集可视化工具都
导入此常量，而不是复制该字典。"""

DEFAULT_TOOLS: list[dict] = [SAY_TOOL_SCHEMA]
"""回退工具列表。当 ``meta/info.json["tools"]`` 未设置时，
由 ``LeRobotDatasetMetadata.tools`` 返回，使未标注的数据集和
chat-template 使用方（``apply_chat_template(messages, tools=...)``）
开箱即用。"""


def column_for_style(style: str | None) -> LanguageColumn:
    """将语言风格映射到存储该风格行的列。

    :data:`PERSISTENT_STYLES` 中的风格路由到 :data:`LANGUAGE_PERSISTENT`。
    :data:`EVENT_ONLY_STYLES` 中的风格以及隐式的 ``None`` 风格路由
    到 :data:`LANGUAGE_EVENTS`。
    """
    if style is None:
        return LANGUAGE_EVENTS
    if style in PERSISTENT_STYLES:
        return LANGUAGE_PERSISTENT
    if style in EVENT_ONLY_STYLES:
        return LANGUAGE_EVENTS
    raise ValueError(f"Unknown language style: {style!r}")
