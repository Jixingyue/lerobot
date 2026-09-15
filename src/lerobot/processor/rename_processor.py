#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PipelineFeatureType, PolicyFeature

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="rename_observations_processor")
class RenameObservationsProcessorStep(ObservationProcessorStep):
    """
    重命名观测字典中键的处理器步骤。

    本步骤通过将环境格式的键映射为 LeRobot 策略或
    其他下游组件期望的格式，来创建标准化的数据接口。

    Attributes:
        rename_map: 将旧键名映射到新键名的字典。
                    观测中存在但不在此映射中的键将
                    保留其原始名称。
    """

    rename_map: dict[str, str] = field(default_factory=dict)

    def observation(self, observation):
        processed_obs = {}
        for key, value in observation.items():
            processed_obs[_rename_key_with_metadata(key, self.rename_map)] = value

        return processed_obs

    def get_config(self) -> dict[str, Any]:
        return {"rename_map": self.rename_map}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """变换规则：
        - 观测中出现在 `rename_map` 中的每个键都会被重命名为其对应的值。
        - 不在 `rename_map` 中的键保持不变。
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = features.copy()
        new_features[PipelineFeatureType.OBSERVATION] = {
            self.rename_map.get(k, k): v for k, v in features[PipelineFeatureType.OBSERVATION].items()
        }
        return new_features


def rename_stats(stats: dict[str, dict[str, Any]], rename_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    """
    使用提供的映射重命名统计字典中的顶层键。

    这是一个辅助函数，通常用于使归一化统计量与
    重命名后的观测或动作特征保持一致。它会进行防御性的
    深拷贝，以避免修改原始 `stats` 字典。

    Args:
        stats: 嵌套的统计字典，其顶层键为
               特征名（例如 `{"observation.state": {"mean": 0.5}}`）。
        rename_map: 将旧特征名映射到新特征名的字典。

    Returns:
        顶层键已重命名的新统计字典。如果输入的
        `stats` 为空，则返回空字典。
    """
    if not stats:
        return {}
    renamed: dict[str, dict[str, Any]] = {}
    for old_key, sub_stats in stats.items():
        new_key = rename_map.get(old_key, old_key)
        renamed[new_key] = deepcopy(sub_stats) if sub_stats is not None else {}
    return renamed


def _rename_key_with_metadata(key: str, rename_map: dict[str, str]) -> str:
    """重命名特征键，同时保留时间采样元数据后缀。"""
    if key in rename_map:
        return rename_map[key]
    for suffix in ("_is_pad", "_padding_mask"):
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            if base in rename_map:
                return f"{rename_map[base]}{suffix}"
    return key


def rename_batch_keys(batch: dict[str, Any], rename_map: dict[str, str] | None) -> dict[str, Any]:
    """在原始数据集键被分组为 transition 之前对其进行规范化。"""
    if not rename_map:
        return batch
    return {_rename_key_with_metadata(key, rename_map): value for key, value in batch.items()}
