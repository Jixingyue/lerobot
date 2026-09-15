# !/usr/bin/env python

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

from __future__ import annotations

from collections.abc import Sequence
from functools import singledispatch
from typing import Any

import numpy as np
import torch

from lerobot.lerobot_types import EnvTransition, PolicyAction, RobotAction, RobotObservation, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    DONE,
    INFO,
    MESSAGES_RENDERED,
    OBS_PREFIX,
    QUERY_KIND,
    QUERY_TEXT,
    REWARD,
    TRUNCATED,
)


@singledispatch
def to_tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    将各种数据类型转换为 PyTorch 张量，并提供可配置的选项。

    这是一个统一的张量转换函数，使用单分派（single dispatch）来
    恰当地处理不同的输入类型。

    Args:
        value: 要转换的输入值（张量、数组、标量、序列等）。
        dtype: 目标张量 dtype。如果为 None，则保留原始 dtype。
        device: 张量的目标设备。

    Returns:
        一个 PyTorch 张量。

    Raises:
        TypeError: 如果输入类型不受支持。
    """
    raise TypeError(f"Unsupported type for tensor conversion: {type(value)}")


@to_tensor.register(torch.Tensor)
def _(value: torch.Tensor, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """处理已有 PyTorch 张量的转换。"""
    if dtype is not None:
        value = value.to(dtype=dtype)
    if device is not None:
        value = value.to(device=device)
    return value


@to_tensor.register(np.ndarray)
def _(
    value: np.ndarray,
    *,
    dtype=torch.float32,
    device=None,
    **kwargs,
) -> torch.Tensor:
    """处理 numpy 数组的转换。"""
    # 检查 numpy 标量（0 维数组），并将其视为标量。
    if value.ndim == 0:
        # numpy 标量应转换为 0 维张量。
        scalar_value = value.item()
        return torch.tensor(scalar_value, dtype=dtype, device=device)

    # 从 numpy 数组创建张量。
    tensor = torch.from_numpy(value)

    # 如果有指定，则应用 dtype 和 device 转换。
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)

    return tensor


@to_tensor.register(int)
@to_tensor.register(float)
@to_tensor.register(np.integer)
@to_tensor.register(np.floating)
def _(value, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """处理标量值（包括 numpy 标量）的转换。"""
    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(list)
@to_tensor.register(tuple)
def _(value: Sequence, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """处理序列（列表、元组）的转换。"""
    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(dict)
def _(value: dict, *, device=None, **kwargs) -> dict:
    """通过递归地将字典中的值转换为张量来处理字典的转换。"""
    if not value:
        return {}

    result = {}
    for key, sub_value in value.items():
        if sub_value is None:
            continue

        if isinstance(sub_value, dict):
            # 递归处理嵌套字典。
            result[key] = to_tensor(
                sub_value,
                device=device,
                **kwargs,
            )
            continue

        # 将单个值转换为张量。
        result[key] = to_tensor(
            sub_value,
            device=device,
            **kwargs,
        )
    return result


def from_tensor_to_numpy(x: torch.Tensor | Any) -> np.ndarray | float | int | Any:
    """
    将 PyTorch 张量转换为 numpy 数组，或在适用时转换为标量。

    如果输入不是张量，则原样返回。

    Args:
        x: 输入，可以是张量或任何其他类型。

    Returns:
        numpy 数组、标量或原始输入。
    """
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().numpy()
    return x


_COMPLEMENTARY_KEYS = (
    "task",
    "index",
    "task_index",
    "episode_index",
    "frame_index",
    "timestamp",
    "language_persistent",
    "language_events",
    MESSAGES_RENDERED,
    "message_streams",
    "target_message_indices",
    # 文本生成请求键：被携带进 complementary_data，以便提示词格式化
    # 处理步骤可以读取该类型并重写 QUERY_TEXT。
    QUERY_KIND,
    QUERY_TEXT,
)


def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:
    """从批次字典中提取补充数据。

    包括填充标志（任何包含 ``_is_pad`` 的键），以及在 ``_COMPLEMENTARY_KEYS``
    中定义的固定元数据/语言键集合——每个键仅在存在于 ``batch`` 中时才包含。
    """
    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    extras = {k: batch[k] for k in _COMPLEMENTARY_KEYS if k in batch}
    return {**pad_keys, **extras}


def create_transition(
    observation: RobotObservation | None = None,
    action: PolicyAction | RobotAction | None = None,
    reward: float = 0.0,
    done: bool = False,
    truncated: bool = False,
    info: dict[str, Any] | None = None,
    complementary_data: dict[str, Any] | None = None,
) -> EnvTransition:
    """
    创建一个带有合理默认值的 `EnvTransition` 字典。

    Args:
        observation: 观测字典。
        action: 动作字典。
        reward: 标量奖励值。
        done: 回合终止标志。
        truncated: 回合截断标志。
        info: 附加信息字典。
        complementary_data: 补充数据字典。

    Returns:
        一个完整的 `EnvTransition` 字典。
    """
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info if info is not None else {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data is not None else {},
    }


def robot_action_observation_to_transition(
    action_observation: tuple[RobotAction, RobotObservation],
) -> EnvTransition:
    """
    将原始机器人动作和观测字典转换为标准化的 `EnvTransition`。

    Args:
        action: 来自遥操作设备或控制器的原始动作字典。
        observation: 来自环境的原始观测字典。

    Returns:
        包含格式化后观测的 `EnvTransition`。
    """
    if not isinstance(action_observation, tuple):
        raise ValueError("action_observation should be a tuple type with an action and observation")

    action, observation = action_observation

    if action is not None and not isinstance(action, dict):
        raise ValueError(f"Action should be a RobotAction type got {type(action)}")

    if observation is not None and not isinstance(observation, dict):
        raise ValueError(f"Observation should be a RobotObservation type got {type(observation)}")

    return create_transition(action=action, observation=observation)


def robot_action_to_transition(action: RobotAction) -> EnvTransition:
    """
    将原始机器人动作字典转换为标准化的 `EnvTransition`。

    Args:
        action: 来自遥操作设备或控制器的原始动作字典。

    Returns:
        包含格式化后动作的 `EnvTransition`。
    """
    if not isinstance(action, dict):
        raise ValueError(f"Action should be a RobotAction type got {type(action)}")
    return create_transition(action=action)


def observation_to_transition(observation: RobotObservation) -> EnvTransition:
    """
    将原始机器人观测字典转换为标准化的 `EnvTransition`。

    Args:
        observation: 来自环境的原始观测字典。

    Returns:
        包含格式化后观测的 `EnvTransition`。
    """
    if not isinstance(observation, dict):
        raise ValueError(f"Observation should be a RobotObservation type got {type(observation)}")
    return create_transition(observation=observation)


def transition_to_robot_action(transition: EnvTransition) -> RobotAction:
    """
    从 `EnvTransition` 中提取用于机器人的原始机器人动作字典。

    该函数查找格式为 "action.*.pos" 或 "action.*.vel" 的键，
    并将其转换为适合发送给机器人控制器的扁平字典。

    Args:
        transition: 包含动作的 `EnvTransition`。

    Returns:
        表示原始机器人动作的字典。
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    action = transition.get(TransitionKey.ACTION)
    if not isinstance(action, dict):
        raise ValueError(f"Action should be a RobotAction type (dict) got {type(action)}")
    return transition.get(TransitionKey.ACTION)


def transition_to_policy_action(transition: EnvTransition) -> PolicyAction:
    """
    将 `EnvTransition` 转换为 `PolicyAction`。
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    action = transition.get(TransitionKey.ACTION)
    if not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
    return action


def transition_to_observation(transition: EnvTransition) -> RobotObservation:
    """
    将 `EnvTransition` 转换为 `RobotObservation`。
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    observation = transition.get(TransitionKey.OBSERVATION)
    if not isinstance(observation, dict):
        raise ValueError(f"Observation should be a RobotObservation (dict) type got {type(observation)}")
    return observation


def policy_action_to_transition(action: PolicyAction) -> EnvTransition:
    """
    将 `PolicyAction` 转换为 `EnvTransition`。
    """
    if not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
    return create_transition(action=action)


def batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """
    将来自数据集/数据加载器的批次字典转换为 `EnvTransition`。

    该函数将批次中可识别的键映射到 `EnvTransition` 结构，
    并用合理的默认值填充缺失的键。

    Args:
        batch: 批次字典。

    Returns:
        一个 `EnvTransition` 字典。

    Raises:
        ValueError: 如果输入不是字典。
    """

    # 验证输入类型。
    if not isinstance(batch, dict):
        raise ValueError(f"EnvTransition must be a dictionary. Got {type(batch).__name__}")

    action = batch.get(ACTION)
    if action is not None and not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

    # 提取观测和补充数据键。
    observation_keys = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
    complementary_data = _extract_complementary_data(batch)

    return create_transition(
        observation=observation_keys if observation_keys else None,
        action=batch.get(ACTION),
        reward=batch.get(REWARD, 0.0),
        done=batch.get(DONE, False),
        truncated=batch.get(TRUNCATED, False),
        info=batch.get("info", {}),
        complementary_data=complementary_data if complementary_data else None,
    )


def transition_to_batch(transition: EnvTransition) -> dict[str, Any]:
    """
    将 `EnvTransition` 转换回 LeRobot 中使用的规范批次格式。

    这是 `batch_to_transition` 的逆操作。

    Args:
        transition: 要转换的 `EnvTransition`。

    Returns:
        具有规范 LeRobot 字段名的批次字典。
    """
    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    batch = {
        ACTION: transition.get(TransitionKey.ACTION),
        REWARD: transition.get(TransitionKey.REWARD, 0.0),
        DONE: transition.get(TransitionKey.DONE, False),
        TRUNCATED: transition.get(TransitionKey.TRUNCATED, False),
        INFO: transition.get(TransitionKey.INFO, {}),
    }

    # 添加补充数据。
    comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
    if comp_data:
        batch.update(comp_data)

    # 展平观测字典。
    observation = transition.get(TransitionKey.OBSERVATION)
    if isinstance(observation, dict):
        batch.update(observation)

    return batch


def identity_transition(transition: EnvTransition) -> EnvTransition:
    """
    转移的恒等函数，原样返回输入。

    可用作处理流水线中的默认值或占位符。

    Args:
        tr: 一个 `EnvTransition`。

    Returns:
        相同的 `EnvTransition`。
    """
    return transition
