#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
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

from .configuration_gaussian_actor import GaussianActorConfig


def make_gaussian_actor_pre_post_processors(
    config: GaussianActorConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为高斯 actor 策略构建预处理器和后处理器流水线。

    预处理流水线通过以下步骤为模型准备输入数据：
    1. 重命名特征以匹配预训练配置。
    2. 基于数据集统计量对输入和输出特征进行归一化。
    3. 添加批维度。
    4. 将所有数据移动到指定设备。

    后处理流水线通过以下步骤处理模型的输出：
    1. 将数据移动到 CPU。
    2. 将输出特征反归一化回原始尺度。

    Args:
        config: tanh-高斯策略的配置对象。
        dataset_stats: 用于归一化的统计量字典。

    Returns:
        包含配置好的预处理器和后处理器流水线的元组。
    """
    return make_default_pre_post_processors(config, dataset_stats)
