#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_multi_task_dit import MultiTaskDiTConfig


def make_multi_task_dit_pre_post_processors(
    config: MultiTaskDiTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为 Multi-Task DiT 策略构建预处理器和后处理器流水线。

    预处理流水线通过以下步骤为模型准备输入数据：
    1. 重命名特征。
    2. 添加批次维度。
    3. 对语言任务描述进行分词（如果存在）。
    4. 将数据移动到指定设备。
    5. 根据数据集统计量对输入和输出特征进行归一化。

    后处理流水线通过以下步骤处理模型的输出：
    1. 将输出特征反归一化到原始尺度。
    2. 将数据移动到 CPU。

    参数：
        config：Multi-Task DiT 策略的配置对象，
            包含特征定义、归一化映射和设备信息。
        dataset_stats：用于归一化的统计量字典。
            默认为 None。

    返回：
        包含配置好的预处理器和后处理器流水线的元组。
    """

    steps = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        TokenizerProcessorStep(
            tokenizer_name=config.text_encoder_name,
            padding=config.tokenizer_padding,
            padding_side=config.tokenizer_padding_side,
            max_length=config.tokenizer_max_length,
            truncation=config.tokenizer_truncation,
        ),
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
