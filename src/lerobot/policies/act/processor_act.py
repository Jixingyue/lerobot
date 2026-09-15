#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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

from .configuration_act import ACTConfig


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """为 ACT 策略创建预处理和后处理流水线。

    预处理流水线处理模型输入的归一化、批处理和设备放置。
    后处理流水线处理反归一化，并将模型输出移回 CPU。

    Args:
        config (ACTConfig): ACT 策略配置对象。
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): 包含用于
            归一化的数据集统计信息（例如均值和标准差）的字典。默认为 None。

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: 包含
        预处理器流水线和后处理器流水线的元组。
    """
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
