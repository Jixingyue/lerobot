#!/usr/bin/env python

# Copyright 2025 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""GR00T N1.7 策略共用的、无副作用的工具函数。

这些辅助函数同时被配置层（检查点附属文件检查）和处理器层（统计量展平、动作解码、
语言与图像打包）使用。它们是不包含任何 GR00T 特有状态的纯函数，因此可以单独进行
单元测试，并且无需导入更重的 config/processor 模块即可复用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def read_json(path: Path) -> dict[str, Any]:
    """从 ``path`` 读取一个 JSON 对象，遇到任何读取/解析错误时返回 ``{}``。"""
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def as_int_pair(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        return None


def as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().reshape(-1).float().tolist()
    if isinstance(values, np.ndarray):
        return values.reshape(-1).astype(np.float32).tolist()
    if isinstance(values, (list, tuple)):
        flattened: list[float] = []
        for value in values:
            flattened.extend(as_float_list(value))
        return flattened
    return [float(values)]


def config_value(value: Any) -> str:
    if hasattr(value, "value"):
        value = value.value
    text = str(value).lower()
    return {
        "relative": "relative",
        "absolute": "absolute",
        "delta": "delta",
        "eef": "eef",
        "non_eef": "non_eef",
        "default": "default",
        "xyz_rot6d": "xyz+rot6d",
        "xyz+rot6d": "xyz+rot6d",
        "xyz_rotvec": "xyz+rotvec",
        "xyz+rotvec": "xyz+rotvec",
    }.get(text, text)


def has_modality_stats(stats: dict[str, dict[str, Any]] | None) -> bool:
    if not stats:
        return False
    return any(bool(modality_stats) for modality_stats in stats.values())


def stat_dim_from_entry(entry: dict[str, Any]) -> int:
    for stat_name in ("mean", "q01", "min", "max", "std"):
        value = entry.get(stat_name)
        if isinstance(value, torch.Tensor):
            return int(value.shape[-1]) if value.ndim > 0 else 1
        if isinstance(value, np.ndarray):
            return int(value.shape[-1]) if value.ndim > 0 else 1
        if isinstance(value, list) and len(value) > 0:
            first = value[0]
            if isinstance(first, (list, tuple)) and len(first) > 0:
                return len(first)
            return len(value)
    return 0


def flatten_n1_7_modality_stats(
    *,
    embodiment_stats: dict[str, Any],
    embodiment_config: dict[str, Any],
    modality: str,
    use_percentiles: bool,
    use_relative_action: bool,
) -> dict[str, list[float]]:
    """按照检查点中的顺序展平 N1.7 单个模态的分组统计量。

    当检查点要求使用百分位归一化时，对常规分组用 q01/q99 替换 min/max。
    相对动作分组则从 ``relative_action`` 统计量中读取并保留 min/max，与
    Isaac-GR00T 的处理器覆写行为保持一致。
    """

    source_stats = embodiment_stats.get(modality, {})
    modality_config = embodiment_config.get(modality, {})
    if not isinstance(source_stats, dict) or not isinstance(modality_config, dict):
        return {}
    modality_keys = modality_config.get("modality_keys", [])
    if not isinstance(modality_keys, list):
        return {}

    flattened: dict[str, list[float]] = {}
    action_configs = modality_config.get("action_configs", []) if modality == "action" else []
    if not isinstance(action_configs, list):
        action_configs = []
    relative_stats = embodiment_stats.get("relative_action", {})
    if not isinstance(relative_stats, dict):
        relative_stats = {}

    for stat_name in ("min", "max", "mean", "std"):
        values: list[float] = []
        source_stat_name = stat_name
        if use_percentiles and stat_name == "min":
            source_stat_name = "q01"
        elif use_percentiles and stat_name == "max":
            source_stat_name = "q99"

        for idx, modality_key in enumerate(modality_keys):
            if not isinstance(modality_key, str):
                continue
            key_source_stats = source_stats
            key_stat_name = source_stat_name
            if modality == "action" and use_relative_action and idx < len(action_configs):
                action_config = action_configs[idx]
                if isinstance(action_config, dict) and config_value(action_config.get("rep")) == "relative":
                    key_source_stats = relative_stats
                    key_stat_name = stat_name
            key_stats = key_source_stats.get(modality_key, {})
            if not isinstance(key_stats, dict):
                raise KeyError(f"Missing statistics for {modality}.{modality_key}")
            raw_values = key_stats.get(key_stat_name)
            if raw_values is None:
                raise KeyError(f"Missing '{key_stat_name}' statistics for {modality}.{modality_key}")
            values.extend(as_float_list(raw_values))
        if values:
            flattened[stat_name] = values

    return flattened


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rows = rot6d.reshape(2, 3).astype(np.float64)
    row1 = rows[0] / np.linalg.norm(rows[0])
    row2 = rows[1] - np.dot(row1, rows[1]) * row1
    row2 = row2 / np.linalg.norm(row2)
    row3 = np.cross(row1, row2)
    return np.vstack([row1, row2, row3])


def xyz_rot6d_to_homogeneous(xyz_rot6d: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot6d_to_matrix(xyz_rot6d[3:])
    transform[:3, 3] = xyz_rot6d[:3]
    return transform


def homogeneous_to_xyz_rot6d(transform: np.ndarray) -> np.ndarray:
    return np.concatenate([transform[:3, 3], transform[:2, :3].reshape(-1)], axis=0)


def relative_eef_to_absolute(action: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
    """将 xyz+rot6d 格式的相对 EEF 增量转换为绝对 EEF 位姿。"""

    out = np.empty_like(action, dtype=np.float64)
    for batch_idx in range(action.shape[0]):
        reference = xyz_rot6d_to_homogeneous(reference_state[batch_idx])
        for timestep in range(action.shape[1]):
            relative = xyz_rot6d_to_homogeneous(action[batch_idx, timestep])
            out[batch_idx, timestep] = homogeneous_to_xyz_rot6d(reference @ relative)
    return out.astype(np.float32)


def infer_n1_7_batch_size_and_device(
    obs: dict[str, Any], action: torch.Tensor | None
) -> tuple[int, torch.device]:
    for value in list(obs.values()) + [action]:
        if isinstance(value, torch.Tensor):
            return value.shape[0], value.device
    video = obs.get("video")
    if isinstance(video, np.ndarray):
        return video.shape[0], torch.device("cpu")
    return 1, torch.device("cpu")


def prepare_n1_7_language_batch(
    language: Any,
    batch_size: int,
    *,
    formalize_language: bool,
) -> list[str]:
    default_language = "Perform the task."
    if language is None or (isinstance(language, str) and language == ""):
        languages = [default_language] * batch_size
    elif isinstance(language, str):
        languages = [language] * batch_size
    elif isinstance(language, (list, tuple)):
        languages = list(language)
        if len(languages) == 0:
            languages = [default_language] * batch_size
        elif len(languages) == 1 and batch_size > 1:
            languages = languages * batch_size
        elif len(languages) != batch_size:
            raise ValueError(
                f"language batch has {len(languages)} entries, but GR00T N1.7 input batch has {batch_size}."
            )
    else:
        languages = [str(language)] * batch_size

    formatted = []
    for item in languages:
        text = str(item) if item else default_language
        if formalize_language:
            text = text.lower()
            text = "".join(ch for ch in text if ch.isalnum() or ch.isspace() or ch == "_")
        formatted.append(text)
    return formatted
