#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
    NewLineTaskProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_smolvla import SmolVLAConfig


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为 SmolVLA 策略构建预处理器和后处理器流水线。

    预处理流水线通过以下步骤为模型准备输入数据：
    1.  重命名特征，以与预训练配置保持一致。
    2.  根据数据集统计信息对输入和输出特征进行归一化。
    3.  添加批次维度。
    4.  确保语言任务描述以换行符结尾。
    5.  对语言任务描述进行分词。
    6.  将所有数据移动到指定设备。

    后处理器流水线通过以下步骤处理模型输出：
    1.  将数据移动到 CPU。
    2.  将输出动作反归一化回原始尺度。

    Args:
        config: SmolVLA 策略的配置对象。
        dataset_stats: 用于归一化的统计信息字典。

    Returns:
        一个元组，包含配置好的预处理器和后处理器流水线。
    """

    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps = [
        steps.rename_observations,  # 以模拟与预训练版本相同的处理器
        steps.add_batch_dim,
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
