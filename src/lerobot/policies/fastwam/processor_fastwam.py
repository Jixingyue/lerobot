# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ActionProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_fastwam import FastWAMConfig


@dataclass
@ProcessorStepRegistry.register(name="fastwam_action_toggle_processor")
class FastWAMActionToggleProcessorStep(ActionProcessorStep):
    """将 FastWAM LIBERO 的 toggle 语义应用到已配置的动作维度上。"""

    toggle_dimensions: list[int]

    def action(self, action: PolicyAction) -> PolicyAction:
        if not self.toggle_dimensions:
            return action
        processed_action = action.clone()
        action_dim = int(processed_action.shape[-1])
        for dim in self.toggle_dimensions:
            resolved_dim = dim if dim >= 0 else action_dim + dim
            if resolved_dim < 0 or resolved_dim >= action_dim:
                raise ValueError(
                    f"FastWAM action toggle dimension {dim} is out of bounds for action dim {action_dim}."
                )
            value = processed_action[..., resolved_dim]
            value = value * 2.0 - 1.0
            processed_action[..., resolved_dim] = torch.sign(-value)
        return processed_action

    def get_config(self) -> dict[str, Any]:
        return {"toggle_dimensions": self.toggle_dimensions}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_fastwam_pre_post_processors(
    config: FastWAMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """为 FastWAM 创建 LeRobot 的预处理和后处理流水线。

    Args:
        config (FastWAMConfig): 控制设备和归一化特征元数据的策略配置。
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): 可选的
            LeRobot 数据集统计信息，供归一化处理器使用。

    Returns:
        tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]: 可被 LeRobot
        发现的输入和输出处理器流水线。
    """

    # 注意：这里不做视觉归一化。VISUAL 使用 IDENTITY（见 configuration_fastwam.normalization_mapping）
    # —— 图像以 [0, 1] 范围透传，模型在编码边界处将其映射到 Wan VAE 的 [-1, 1]。
    # 这是有意为之：`lerobot_train.py` 在微调时会用 `dataset.meta.stats` 覆盖归一化统计量，
    # 而真实数据集的逐通道图像 std 是极小的帧间亮度方差，会把图像推到 [-1,1] 之外很远并导致饱和。
    # STATE/ACTION 仍然使用下面的数据集统计量进行归一化。
    normalization_stats: dict[str, dict[str, Any]] = dict(dataset_stats or {})

    # 注意：这里没有 resize 步骤。模型是输入分辨率的唯一权威：它在
    # `_stack_video_from_images` / `_prepare_infer_image` 中把每个相机 resize 到
    # 各自的相机目标尺寸（image_size 按相机数均分），覆盖所有路径（train forward、rollout 和
    # eval select_action）。预处理器的 resize 步骤既是冗余的（模型反正会重新 resize），
    # 在微调时也不安全：其 `resize_size` 会继承基础 checkpoint 的相机几何尺寸，
    # 而不是当前数据集的，导致拼接结果宽出 N_cameras 倍。

    steps = make_default_policy_processor_steps(config, normalization_stats, normalizer_device=config.device)

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
    ]
    if config.toggle_action_dimensions:
        output_steps.append(
            FastWAMActionToggleProcessorStep(toggle_dimensions=config.toggle_action_dimensions)
        )
    output_steps.append(steps.to_cpu)
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
