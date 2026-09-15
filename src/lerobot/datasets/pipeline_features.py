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

import re
from collections.abc import Sequence
from typing import Any

from lerobot.configs import PipelineFeatureType
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import DataProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features


def create_initial_features(
    action: RobotAction | None = None, observation: RobotObservation | None = None
) -> dict[PipelineFeatureType, dict[str, Any]]:
    """
    根据动作和观测的规格创建数据集的初始特征字典。

    Args:
        action: 动作特征名称到其类型/形状的字典。
        observation: 观测特征名称到其类型/形状的字典。

    Returns:
        按 PipelineFeatureType 组织的初始特征字典。
    """
    features = {PipelineFeatureType.ACTION: {}, PipelineFeatureType.OBSERVATION: {}}
    if action:
        features[PipelineFeatureType.ACTION] = action
    if observation:
        features[PipelineFeatureType.OBSERVATION] = observation
    return features


# 辅助函数：根据已编译的正则表达式模式过滤 state/action 键。
def should_keep(key: str, patterns: tuple[re.Pattern] | None) -> bool:
    if patterns is None:
        return True
    return any(pat.search(key) for pat in patterns)


def strip_prefix(key: str, prefixes_to_strip: tuple[str]) -> str:
    for prefix in prefixes_to_strip:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


# 定义要从特征键中去除的前缀，以获得干净的名称。
# 同时处理完整限定形式（例如 "action.state"）和简短形式（例如 "state"）。
PREFIXES_TO_STRIP = tuple(
    f"{token}." for const in (ACTION, OBS_STATE, OBS_IMAGES) for token in (const, const.split(".")[-1])
)


def aggregate_pipeline_dataset_features(
    pipeline: DataProcessorPipeline,
    initial_features: dict[PipelineFeatureType, dict[str, Any]],
    *,
    use_videos: bool = True,
    exclude_images: bool = False,
    patterns: Sequence[str] | None = None,
) -> dict[str, dict]:
    """
    聚合并过滤流水线特征，以创建可用于数据集的特征字典。

    该函数使用流水线转换初始特征，将其归类为动作或观测
    （图像或状态），根据 `exclude_images` 和 `patterns` 进行过滤，最后
    将其格式化为可供 Hugging Face LeRobot Dataset 使用的形式。

    Args:
        pipeline: 要应用的 DataProcessorPipeline。
        initial_features: 动作和观测的原始特征规格字典。
        use_videos: 控制图像特征的存储 dtype。如果为 True，图像存储为 "video"；如果为 False，则存储为 "image"。
        exclude_images: 如果为 True，图像特征将从输出中完全丢弃。
        patterns: 用于过滤动作和状态特征的正则表达式模式序列。
                  图像特征不受此过滤器影响。

    Returns:
        为 Hugging Face LeRobot Dataset 格式化好的特征字典。
    """
    compiled_patterns = tuple(re.compile(p) for p in patterns) if patterns is not None else None

    all_features = pipeline.transform_features(initial_features)

    # 用于存放已分类和已过滤特征的中间存储。
    processed_features: dict[str, dict[str, Any]] = {
        ACTION: {},
        OBS_STR: {},
    }
    images_token = OBS_IMAGES.split(".")[-1]

    # 遍历流水线转换后的所有特征。
    for ptype, feats in all_features.items():
        if ptype not in [PipelineFeatureType.ACTION, PipelineFeatureType.OBSERVATION]:
            continue

        for key, value in feats.items():
            # 1. 对特征进行分类。
            is_action = ptype == PipelineFeatureType.ACTION
            # 如果观测的键匹配与图像相关的词元，或者特征的形状为 3 维，则将其归类为图像。
            # 所有其他观测都被视为状态。
            is_image = not is_action and (
                (isinstance(value, tuple) and len(value) == 3)
                or (
                    key.startswith(f"{OBS_IMAGES}.")
                    or key.startswith(f"{images_token}.")
                    or f".{images_token}." in key
                )
            )

            # 2. 应用过滤规则。
            if is_image and exclude_images:
                continue
            if not is_image and not should_keep(key, compiled_patterns):
                continue

            # 3. 使用干净的名称将特征添加到相应的组中。
            name = strip_prefix(key, PREFIXES_TO_STRIP)
            if is_action:
                processed_features[ACTION][name] = value
            else:
                processed_features[OBS_STR][name] = value

    # 将处理后的特征转换为最终的数据集格式。
    dataset_features = {}
    if processed_features[ACTION]:
        dataset_features.update(hw_to_dataset_features(processed_features[ACTION], ACTION, use_videos))
    if processed_features[OBS_STR]:
        dataset_features.update(hw_to_dataset_features(processed_features[OBS_STR], OBS_STR, use_videos))

    return dataset_features
