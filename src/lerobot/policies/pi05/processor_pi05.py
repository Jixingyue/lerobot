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

from .configuration_pi05 import PI05Config


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    用于准备状态并对语言输入进行分词的处理步骤。
    """

    max_state_dim: int = 32
    task_key: str = "task"
    # MEM 第 III-D 节使用线性投影将本体感知表示到主干中，而不是使用离散化的
    # 提示词 token，因此状态只被携带一次。
    # 该值由 `PI05Config.use_proprioceptive_memory` 设置；原版 PI0.5 将其保留在提示词中。
    include_state_in_prompt: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: 检查这是否必要
        state = deepcopy(state)

        discretized_states = None
        if self.include_state_in_prompt:
            # 状态在此步骤之前运行的 NormalizerProcessorStep 中应已被归一化到 [-1, 1]
            # 离散化为 256 个区间（参见 openpi `PaligemmaTokenizer.tokenize()`）
            prompt_state = state[:, -1] if state.ndim == 3 else state
            state_np = prompt_state.cpu().numpy()
            discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            if discretized_states is None:
                full_prompt = f"Task: {cleaned_text};\nAction: "
            else:
                state_str = " ".join(map(str, discretized_states[i]))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # 如有需要，将状态归一化到 [-1, 1] 范围（假设它已被 normalizer 处理步骤归一化！！）
        # 离散化为 256 个区间（参见 openpi `PaligemmaTokenizer.tokenize()`）
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        此步骤不会改变特征定义。
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为 PI0 策略构建前处理和后处理流水线。

    前处理流水线通过以下步骤为模型准备输入数据：
    1. 重命名特征以匹配预训练配置。
    2. 根据数据集统计量对输入和输出特征进行归一化。
    3. 添加批次维度。
    4. 在任务描述末尾追加换行符以兼容分词器。
    5. 使用 PaliGemma 分词器对文本提示进行分词。
    6. 将所有数据移动到指定设备。

    后处理流水线通过以下步骤处理模型的输出：
    1. 将数据移动到 CPU。
    2. 将输出特征反归一化到原始尺度。

    Args:
        config: PI0 策略的配置对象。
        dataset_stats: 用于归一化的统计量字典。
        preprocessor_kwargs: 前处理流水线的附加参数。
        postprocessor_kwargs: 后处理流水线的附加参数。

    Returns:
        包含配置好的前处理和后处理流水线的元组。
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    steps = make_default_policy_processor_steps(config, dataset_stats)

    # OpenPI 顺序：raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,  # 为了模拟与预训练模型相同的处理器
        steps.add_batch_dim,
        relative_step,
        # 注意：NormalizerProcessorStep 必须位于 Pi05PrepareStateTokenizerProcessorStep 之前，
        # 因为分词器步骤期望状态已被归一化到 [-1, 1] 范围以便离散化
        steps.normalize,
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            include_state_in_prompt=not config.use_proprioceptive_memory,
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.text_tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        steps.to_device,
    ]

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
