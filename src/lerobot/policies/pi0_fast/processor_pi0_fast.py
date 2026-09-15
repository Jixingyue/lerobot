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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    ActionTokenizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import OBS_STATE

from .configuration_pi0_fast import PI0FastConfig


@ProcessorStepRegistry.register(name="pi0_fast_prepare_state_tokenizer_processor_step")
@dataclass
class Pi0FastPrepareStateAndLanguageTokenizerProcessorStep(ProcessorStep):
    """
    用于准备状态并对语言输入进行分词的处理步骤。
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI0Fast")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: 检查这是否必要
        state = deepcopy(state)

        # 状态应已由在此步骤之前运行的 NormalizerProcessorStep 归一化到 [-1, 1]
        # 离散化为 256 个区间（参见 openpi `PaligemmaTokenizer.tokenize()`）
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\n"
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # 如有需要，将状态归一化到 [-1, 1] 范围（假设它已由归一化处理步骤归一化！！）
        # 离散化为 256 个区间（参见 openpi `PaligemmaTokenizer.tokenize()`）
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        该步骤不会改变特征定义。
        """
        return features


def make_pi0_fast_pre_post_processors(
    config: PI0FastConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为 PI0Fast 策略构建预处理器和后处理器流水线。

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
        config: PI0Fast 策略的配置对象。
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

    # Pi0Fast 顺序：相对 → 归一化 → 分词 → 模型 → 反归一化 → 绝对
    # 这与 pi0/pi0.5 一致：RelativeActionsProcessorStep 首先对原始绝对动作运行，
    # 并缓存原始状态。随后 NormalizerProcessorStep 对原始相对动作进行归一化，
    # 因此归一化器（以及动作分词器）看到的是增量值——需要相对统计信息。
    # 注意：RelativeActionsProcessorStep 仅修改转移（transition）中的动作；它从
    # 观测中读取状态但不会更改状态。NormalizerProcessorStep 仍然在
    # Pi0FastPrepareStateAndLanguageTokenizerProcessorStep 之前运行，因此状态分词器
    # 仍会按预期接收到 [-1, 1] 范围内的归一化状态。
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,  # 模拟与预训练模型相同的处理器
        steps.add_batch_dim,
        relative_step,
        steps.normalize,
        Pi0FastPrepareStateAndLanguageTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name=config.text_tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        ActionTokenizerProcessorStep(
            action_tokenizer_name=config.action_tokenizer_name,
            max_action_tokens=config.max_action_tokens,
            fast_skip_tokens=config.fast_skip_tokens,
            paligemma_tokenizer_name=config.text_tokenizer_name,
        ),
        steps.to_device,
    ]

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
