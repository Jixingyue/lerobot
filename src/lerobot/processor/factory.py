#!/usr/bin/env python

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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.lerobot_types import PolicyAction, RobotAction, RobotObservation
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .batch_processor import AddBatchDimensionProcessorStep
from .converters import (
    observation_to_transition,
    policy_action_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_policy_action,
    transition_to_robot_action,
)
from .device_processor import DeviceProcessorStep
from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from .pipeline import (
    IdentityProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    RobotProcessorPipeline,
)
from .rename_processor import RenameObservationsProcessorStep


def make_default_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor


def make_default_robot_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor


def make_default_robot_observation_processor() -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_default_processors():
    teleop_action_processor = make_default_teleop_action_processor()
    robot_action_processor = make_default_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor()
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)


@dataclass
class DefaultPolicyProcessorSteps:
    """大多数策略的前/后处理流水线共享的规范处理步骤。

    各策略按自己的顺序组合这些步骤（步骤顺序是 Hub 序列化的约定，
    并且有意在每个策略中保持显式），并穿插其自定义步骤。
    """

    rename_observations: RenameObservationsProcessorStep
    add_batch_dim: AddBatchDimensionProcessorStep
    to_device: DeviceProcessorStep
    normalize: NormalizerProcessorStep
    unnormalize: UnnormalizerProcessorStep
    to_cpu: DeviceProcessorStep


def make_default_policy_processor_steps(
    config: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    normalizer_device: torch.device | str | None = None,
) -> DefaultPolicyProcessorSteps:
    """根据策略配置构建规范的策略处理步骤。

    Args:
        config: 提供 `device`、`input_features`、`output_features`
            和 `normalization_mapping` 的 `PreTrainedConfig`。
        dataset_stats: 用于（反）归一化的数据集统计量。
        normalizer_device: 传递给 `NormalizerProcessorStep` 的设备（某些策略会将其
            归一化统计量固定到策略设备上；大多数策略不设置该项）。
    """
    return DefaultPolicyProcessorSteps(
        rename_observations=RenameObservationsProcessorStep(rename_map={}),
        add_batch_dim=AddBatchDimensionProcessorStep(),
        to_device=DeviceProcessorStep(device=config.device),
        normalize=NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=normalizer_device,
        ),
        unnormalize=UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        to_cpu=DeviceProcessorStep(device="cpu"),
    )


def make_policy_processor_pipelines(
    input_steps: list[ProcessorStep],
    output_steps: list[ProcessorStep],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """将前/后处理步骤列表包装为规范的策略流水线对。

    使用标准的流水线名称（决定 Hub 上序列化后的 JSON 文件名），
    并在后处理器上使用标准的策略动作转换器。
    """
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def make_default_pre_post_processors(
    config: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    normalizer_device: torch.device | str | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """纯脚手架式的策略流水线对：Rename -> Batch -> Device -> Normalize，
    以及 Unnormalize -> Device(cpu)。具有自定义步骤或不同步骤顺序的策略
    会自行组合 `make_default_policy_processor_steps`。
    """
    s = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=normalizer_device)
    return make_policy_processor_pipelines(
        input_steps=[s.rename_observations, s.add_batch_dim, s.to_device, s.normalize],
        output_steps=[s.unnormalize, s.to_cpu],
    )
