#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_pre_post_processors,
)

from .configuration_diffusion import DiffusionConfig


def make_diffusion_pre_post_processors(
    config: DiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为扩散策略构建预处理器和后处理器流水线。

    预处理流水线通过以下步骤为模型准备输入数据：
    1. 重命名特征。
    2. 根据数据集统计量对输入和输出特征进行归一化。
    3. 添加批次维度。
    4. 将数据移动到指定设备。

    后处理流水线通过以下步骤处理模型的输出：
    1. 将数据移动到 CPU。
    2. 将输出特征反归一化到其原始尺度。

    Args:
        config: 扩散策略的配置对象，
            包含特征定义、归一化映射和设备信息。
        dataset_stats: 用于归一化的统计量字典。
            默认为 None。

    Returns:
        包含配置好的预处理器和后处理器流水线的元组。
    """
    return make_default_pre_post_processors(config, dataset_stats)
