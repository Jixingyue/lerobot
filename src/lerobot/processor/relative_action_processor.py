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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .pipeline import ProcessorStep, ProcessorStepRegistry

# 为向后兼容而重新导出
__all__ = [
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "to_relative_actions",
    "to_absolute_actions",
]


def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """将绝对动作转换为相对动作：relative = action - state（针对被掩码选中的维度）。

    Args:
        actions: (B, T, action_dim) 或 (B, action_dim)。
        state: (B, state_dim)。在时间维度上广播。
        mask: 要转换的维度。可以比 action_dim 短。
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # 将 state 对齐到与 actions 相同的设备/dtype。_last_state 是在
    # DeviceProcessorStep 移动转移之前缓存的，因此它可能在 CPU 上而 actions 在 CUDA 上。
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """将相对动作转换回绝对动作：absolute = relative + state（针对被掩码选中的维度）。

    Args:
        actions: (B, T, action_dim) 或 (B, action_dim)。
        state: (B, state_dim)。在时间维度上广播。
        mask: 要转换的维度。可以比 action_dim 短。
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # 将 state 对齐到与 actions 相同的设备/dtype。_last_state 是在
    # DeviceProcessorStep 移动转移之前缓存的，因此它可能在 CPU 上而 actions 在 CUDA 上。
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


@ProcessorStepRegistry.register("relative_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """对被掩码选中的维度，将绝对动作转换为相对动作（action -= state）。

    对应 OpenPI 的 DeltaActions 变换。在预处理阶段应用，使模型
    基于相对偏移量而非绝对位置进行训练。
    缓存最近一次的 state，以便配对的 AbsoluteActionsProcessorStep 在
    后处理阶段可以逆转该转换。

    Attributes:
        enabled: 是否应用相对转换。
        exclude_joints: 保持为绝对值的关节名称（不转换为相对值）。
        action_names: 来自数据集元数据的动作维度名称，用于根据
            exclude_joints 构建掩码。如果为 None，则转换所有维度。
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, action_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * action_dim

        mask = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))

        return mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None

        # 始终缓存 state，供配对的 AbsoluteActionsProcessorStep 使用
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        mask = self._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask)
        return new_transition

    def get_cached_state(self) -> torch.Tensor | None:
        """返回缓存的 ``observation.state``，它被用作相对/绝对动作转换的参考点。"""
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """对所有维度将相对动作转换回绝对动作（action += state）。

    对应 OpenPI 的 AbsoluteActions 变换。在后处理阶段应用，使
    预测的相对偏移量被转换回绝对位置以供执行。
    从其配对的 RelativeActionsProcessorStep 读取缓存的 state。

    Attributes:
        enabled: 是否应用绝对转换。
        relative_step: 对缓存 state 的配对 RelativeActionsProcessorStep 的引用。
    """

    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires a paired RelativeActionsProcessorStep "
                "but relative_step is None. Ensure relative_step is set when constructing the postprocessor."
            )

        cached_state = self.relative_step.get_cached_state()
        if cached_state is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires state from RelativeActionsProcessorStep "
                "but no state has been cached. Ensure the preprocessor runs before the postprocessor."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        mask = self.relative_step._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_absolute_actions(action, cached_state, mask)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
