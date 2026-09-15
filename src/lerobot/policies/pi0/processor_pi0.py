#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    ComplementaryDataProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_pi0 import PI0Config


@ProcessorStepRegistry.register(name="pi0_new_line_processor")
class Pi0NewLineProcessor(ComplementaryDataProcessorStep):
    """
    确保任务描述字符串以换行符结尾。

    该处理步骤是为了兼容 PaliGemma 分词器而必需的，
    因为该分词器期望文本提示的末尾有一个换行符。它可以处理
    补充数据中 'task' 键对应的单个字符串和字符串列表。
    """

    def complementary_data(self, complementary_data):
        """
        如果 'task' 字段尚未以换行符结尾，则为其添加换行符。

        参数：
            complementary_data: 一个字典，可能包含值为字符串或字符串列表的
                                'task' 键。

        返回：
            包含已修改 'task' 字段的新字典。
        """
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # 同时处理字符串和字符串列表
        if isinstance(task, str):
            # 单个字符串：如果不存在则添加换行符
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # 字符串列表：为每个字符串添加换行符（如果不存在）
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # 如果 task 既不是字符串也不是字符串列表，则保持不变

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        该步骤不会改变特征定义。

        参数：
            features: 输入特征字典。

        返回：
            未改变的特征字典。
        """
        return features


def make_pi0_pre_post_processors(
    config: PI0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为 PI0 策略构建预处理器和后处理器流水线。

    预处理流水线通过以下步骤为模型准备输入数据：
    1. 重命名特征以匹配预训练配置。
    2. 根据数据集统计信息对输入和输出特征进行归一化。
    3. 添加批次维度。
    4. 在任务描述末尾追加换行符以兼容分词器。
    5. 使用 PaliGemma 分词器对文本提示进行分词。
    6. 将所有数据移动到指定设备。

    后处理流水线通过以下步骤处理模型的输出：
    1. 将数据移动到 CPU。
    2. 将输出特征反归一化到其原始尺度。

    参数：
        config: PI0 策略的配置对象。
        dataset_stats: 用于归一化的统计信息字典。
        preprocessor_kwargs: 预处理器流水线的附加参数。
        postprocessor_kwargs: 后处理器流水线的附加参数。

    返回：
        包含已配置的预处理器和后处理器流水线的元组。
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    steps = make_default_policy_processor_steps(config, dataset_stats)

    # OpenPI 顺序：原始 → 相对 → 归一化 → 模型 → 反归一化 → 绝对
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,  # 模拟与预训练模型相同的处理器
        steps.add_batch_dim,
        Pi0NewLineProcessor(),  # 在 PaliGemma 分词前添加换行符
        TokenizerProcessorStep(
            tokenizer_name=config.text_tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        steps.to_device,
        relative_step,
        steps.normalize,
    ]

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
