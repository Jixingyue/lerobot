# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import logging
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.configs.types import NormalizationMode
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    EnvTransition,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION


@ProcessorStepRegistry.register(name="vla_jepa_image_prep")
class ImagePrepProcessorStep(ObservationProcessorStep):
    """为 VLA-JEPA 模型准备图像观测：转换为 float、1->3 通道扩展、缩放。

    将模型过去在内部完成的预处理暴露到序列化的 pipeline 中。模型仍保留相同操作作为幂等保护，
    因此未保存该步骤的旧 checkpoint 仍会得到预处理，而较新的 checkpoint 在那里则是空操作。

    与 `Qwen3VLInterface.to_pixel_values` 以及 `VLAJEPAPolicy._prepare_model_inputs`/
    `predict_action` 中的 `F.interpolate(mode="area")` 缩放保持一致，并且不做 clamp（模型路径
    同样不做），因此数值保持逐位一致。支持 [C,H,W]、[B,C,H,W]/[T,C,H,W] 和 [B,T,C,H,W]。
    """

    def __init__(self, resize_to: tuple[int, int] | None = None, expand_channels: bool = True):
        self.resize_to = tuple(resize_to) if resize_to is not None else None
        self.expand_channels = expand_channels

    def observation(self, observation: dict) -> dict:
        new_observation = dict(observation)
        for key in observation:
            if "image" not in key:
                continue
            image = observation[key].float()
            if self.expand_channels and image.shape[-3] == 1:
                repeats = [1] * image.ndim
                repeats[-3] = 3
                image = image.repeat(*repeats)
            if self.resize_to is not None and tuple(image.shape[-2:]) != self.resize_to:
                device = image.device
                # 注意：mps 上没有 "area" 核函数；先在 cpu 上缩放再移回原设备。
                if device.type == "mps":
                    image = image.cpu()
                lead = image.shape[:-3]
                c, h, w = image.shape[-3:]
                flat = image.reshape(-1, c, h, w)
                flat = F.interpolate(flat, size=self.resize_to, mode="area")
                image = flat.reshape(*lead, c, *self.resize_to).to(device)
            new_observation[key] = image
        return new_observation

    def get_config(self) -> dict[str, Any]:
        return {
            "resize_to": list(self.resize_to) if self.resize_to is not None else None,
            "expand_channels": self.expand_channels,
        }

    def transform_features(self, features):
        for key in features[PipelineFeatureType.OBSERVATION]:
            if "image" not in key:
                continue
            feat = features[PipelineFeatureType.OBSERVATION][key]
            # 与 `to_pixel_values` 保持一致：只有单通道才会被扩展为 3 通道。
            nb_channel = 3 if (self.expand_channels and feat.shape[0] == 1) else feat.shape[0]
            spatial = self.resize_to if self.resize_to is not None else tuple(feat.shape[1:])
            features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                type=feat.type, shape=(nb_channel, *spatial)
            )
        return features


@ProcessorStepRegistry.register(name="vla_jepa_clip_actions")
class ClipActionsProcessorStep(ProcessorStep):
    """在反归一化之前将动作张量裁剪到 [-1, 1]。"""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition = dict(transition)
            transition[TransitionKey.ACTION] = action.clamp(-1.0, 1.0)
        return transition

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register(name="vla_jepa_pre_snap_gripper")
class PreSnapGripperProcessorStep(ProcessorStep):
    """在反归一化之前将夹爪维度吸附（snap）到 {0, 1}。

    对应 starVLA 原始的 LIBERO 评估：
      normalized[:, gripper_dim] = np.where(normalized[:, gripper_dim] < threshold, 0, 1)
    这可以确保反归一化器接收到精确的二值值；当模型训练时夹爪处于 identity（mask=False）
    空间（其中 0=张开、1=闭合）时，这是必需的。
    """

    def __init__(self, gripper_dim: int = 6, threshold: float = 0.5):
        self.gripper_dim = gripper_dim
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is not None and action.shape[-1] > self.gripper_dim:
            transition = dict(transition)
            a = action.clone()
            a[..., self.gripper_dim] = (a[..., self.gripper_dim] >= self.threshold).float()
            transition[TransitionKey.ACTION] = a
        return transition

    def get_config(self) -> dict[str, Any]:
        # 没有这个方法的话，基类会序列化成 `{}`，重新加载的 pipeline 会静默地回退到类的默认值，
        # 从而丢弃训练配置中设置的内容。
        return {"gripper_dim": self.gripper_dim, "threshold": self.threshold}

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register(name="vla_jepa_binarize_gripper")
class BinarizeGripperProcessorStep(ProcessorStep):
    """在反归一化之后将夹爪维度二值化。

    将连续值映射到 {-1, 1}：> threshold → -1，<= threshold → 1（与 starVLA 的约定一致）。
    仅当 action 的维度数多于 gripper_dim 时才应用。

    警告：该步骤运行在反归一化器*下游*，因此 `threshold` 是与夹爪的**物理**值比较的，而其
    默认值 0.5 来自模型的 [0, 1]/±1 约定。对于以度、毫米或 [0, 100] 为单位的夹爪，每个值都会
    超过 0.5，输出会全部坍缩为 -1；当数据集统计信息表明存在这种情况时，
    `make_vla_jepa_pre_post_processors` 会发出警告。
    """

    def __init__(self, gripper_dim: int = 6, threshold: float = 0.5):
        self.gripper_dim = gripper_dim
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is not None and action.shape[-1] > self.gripper_dim:
            transition = dict(transition)
            a = action.clone()
            a[..., self.gripper_dim] = 1.0 - 2.0 * (a[..., self.gripper_dim] > self.threshold).float()
            transition[TransitionKey.ACTION] = a
        return transition

    def get_config(self) -> dict[str, Any]:
        # 参见 PreSnapGripperProcessorStep.get_config：`{}` 在重新加载时会变回类的默认值。
        return {"gripper_dim": self.gripper_dim, "threshold": self.threshold}

    def transform_features(self, features):
        return features


def _warn_if_gripper_steps_are_misconfigured(
    config: VLAJEPAConfig,
    gripper_dim: int,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
) -> None:
    """当夹爪后处理步骤会把夹爪固定为常量时发出警告。

    `BinarizeGripperProcessorStep` 以 `gripper_threshold` 对*未归一化*的夹爪值做阈值判断。当数据集
    的物理范围远高于该阈值时，所有值都会落在同一侧，夹爪永远不会动作——利用手头已有的统计信息
    就能检测到这种情况，因此在此明确提示。
    """
    if not (config.pre_snap_gripper_action or config.binarize_gripper_action):
        return
    action_stats = (dataset_stats or {}).get(ACTION)
    if not action_stats or "min" not in action_stats or "max" not in action_stats:
        return
    try:
        low = float(action_stats["min"][gripper_dim])
        high = float(action_stats["max"][gripper_dim])
    except (IndexError, TypeError, ValueError):
        return
    threshold = config.gripper_threshold
    # `pre_snap` 在归一化空间写入 {0, 1}，反归一化后分别对应中点和最大值。两者落在阈值的
    # 同一侧即意味着输出为常量。
    midpoint = (low + high) / 2.0
    if (midpoint > threshold) == (high > threshold):
        name = (
            config.action_feature_names[gripper_dim]
            if config.action_feature_names and gripper_dim < len(config.action_feature_names)
            else f"dim {gripper_dim}"
        )
        logging.warning(
            f"vla_jepa gripper post-processing looks misconfigured: action {name} has a physical "
            f"range of [{low:.3g}, {high:.3g}], and `gripper_threshold={threshold}` is compared "
            f"against that unnormalized value. Both {midpoint:.3g} and {high:.3g} fall on the same "
            f"side of it, so the commanded gripper will be constant. Set `gripper_threshold` in "
            f"the gripper's own units, or set `pre_snap_gripper_action=false` and "
            f"`binarize_gripper_action=false` (the defaults) unless you are running LIBERO."
        )


def make_vla_jepa_pre_post_processors(
    config: VLAJEPAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    features = {**config.input_features, **config.output_features}
    steps = make_default_policy_processor_steps(config, dataset_stats)

    # 共享的相对动作步骤（OpenPI 顺序：raw -> relative -> normalize -> model ->
    # unnormalize -> absolute）。将同一个实例传给下面的 AbsoluteActionsProcessorStep，
    # 以便其缓存的原始状态（在预处理期间设置）能够流转到后处理。
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        ImagePrepProcessorStep(
            resize_to=tuple(config.resize_images_to) if config.resize_images_to else None,
        ),
        relative_step,
        steps.normalize,
    ]
    gripper_dim = config.resolved_gripper_dim
    _warn_if_gripper_steps_are_misconfigured(config, gripper_dim, dataset_stats)

    output_steps: list[ProcessorStep] = []
    if config.clip_normalized_actions:
        # 在 MIN_MAX 下，裁剪到 [-1, 1] 只是一种范围断言；但在 MEAN_STD 下，同样的 clamp 会
        # 截断所有超过 1 sigma 的动作。这会表现为一个迟疑、低幅度的策略，且任何地方都不报错，
        # 因此这里拒绝添加该步骤，而不是机械地遵从这个开关。
        action_norm_mode = config.normalization_mapping.get("ACTION")
        if action_norm_mode == NormalizationMode.MIN_MAX:
            output_steps.append(ClipActionsProcessorStep())
        else:
            logging.warning(
                f"`clip_normalized_actions=True` is ignored: it clips normalized actions to "
                f"[-1, 1], which is only a no-op bound under MIN_MAX, but ACTION uses "
                f"{getattr(action_norm_mode, 'value', action_norm_mode)}. Under MEAN_STD this "
                f"would clamp every action to 1 sigma."
            )
    if config.pre_snap_gripper_action:
        output_steps.append(
            PreSnapGripperProcessorStep(gripper_dim=gripper_dim, threshold=config.gripper_threshold)
        )
    # 注意：与默认的策略反归一化器（仅处理输出特征）不同，VLA-JEPA 会同时对输入和输出
    # 特征进行反归一化。
    output_steps.append(
        UnnormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
    )
    # 在夹爪二值化之前，对反归一化后的动作逆转相对转换。夹爪由 relative_exclude_joints
    # 保持为绝对值，因此这两个步骤作用于互不相交的维度。
    output_steps.append(
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step)
    )
    if config.binarize_gripper_action:
        output_steps.append(
            BinarizeGripperProcessorStep(gripper_dim=gripper_dim, threshold=config.gripper_threshold)
        )
    output_steps.append(steps.to_cpu)
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
