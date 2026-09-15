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

########################################################################################
# 工具
########################################################################################
import time
from contextlib import nullcontext
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.utils.import_utils import _deepdiff_available, require_package

if TYPE_CHECKING or _deepdiff_available:
    from deepdiff import DeepDiff
else:
    DeepDiff = None

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset
from lerobot.lerobot_types import PolicyAction
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import Robot


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    执行单步推理，从观测中预测机器人动作。

    此函数封装了完整的推理流水线：
    1. 通过将观测转换为 PyTorch 张量并添加批量维度来准备观测。
    2. 在观测上运行预处理器流水线。
    3. 将处理后的观测馈送给策略以获取原始动作。
    4. 在原始动作上运行后处理器流水线。
    5. 通过移除批量维度并将其移动到 CPU 来格式化最终动作。

    参数:
        observation: 表示机器人当前观测的 NumPy 数组字典。
        policy: 用于动作预测的 `PreTrainedPolicy` 模型。
        device: 运行推理的 `torch.device`（例如 'cuda' 或 'cpu'）。
        preprocessor: 用于预处理观测的 `PolicyProcessorPipeline`。
        postprocessor: 用于后处理动作的 `PolicyProcessorPipeline`。
        use_amp: 启用/禁用 CUDA 推理的自动混合精度的布尔值。
        task: 任务的可选字符串标识符。
        robot_type: 机器人类型的可选字符串标识符。

    返回:
        包含预测动作的 `torch.Tensor`，可供机器人使用。
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # 转换为 pytorch 格式：通道优先，float32 在 [0,1] 范围内，带批量维度
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # 基于当前观测，使用策略计算下一个动作
        action = policy.select_action(observation)

        action = postprocessor(action)

    return action


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    对照策略配置的存在性验证数据集仓库名称。

    此函数强制执行命名约定：当且仅当提供了用于评估目的的策略配置时，
    数据集仓库 ID 应以 "eval_" 开头。

    参数:
        repo_id: 数据集的 Hugging Face Hub 仓库 ID。
        policy_cfg: 策略的配置对象，或 `None`。

    引发:
        ValueError: 如果命名约定被违反。
    """
    _, dataset_name = repo_id.split("/")
    # 要么 repo_id 不以 "eval_" 开头且没有策略
    # 要么 repo_id 以 "eval_" 开头且有策略

    # 检查 dataset_name 是否以 "eval_" 开头但缺少策略
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # 检查 dataset_name 是否不以 "eval_" 开头但提供了策略
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    检查数据集的元数据是否与当前机器人和录制设置兼容。

    此函数将数据集中的关键元数据字段（`robot_type`、`fps` 和 `features`）
    与当前配置进行比较，以确保追加的数据将保持一致。

    参数:
        dataset: 要检查的 `LeRobotDataset` 实例。
        robot: 表示当前硬件设置的 `Robot` 实例。
        fps: 当前录制频率（每秒帧数）。
        features: 当前录制会话的特征字典。

    引发:
        ValueError: 如果任何被检查的元数据字段不匹配。
    """
    require_package("deepdiff", extra="deepdiff-dep")

    from lerobot.utils.constants import DEFAULT_FEATURES

    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )


########################################################################################
# 遥操作平滑切换助手
# 注意(Maxime)：这些函数使用最少的类型提示，以保持与 utils 的兼容性
# 作为根模块。
########################################################################################


def teleop_supports_feedback(teleop) -> bool:
    """当遥操作可以接收位置反馈（已驱动）时返回 True。

    已驱动的遥操作（例如 SO-101、OpenArmMini）具有非空的 ``feedback_features``
    并暴露 ``enable_torque`` / ``disable_torque`` 电机控制方法。

    TODO(Maxime)：看看是否有可能跨遥操作统一此接口，而不是鸭子类型。
    """
    return (
        bool(teleop.feedback_features)
        and hasattr(teleop, "disable_torque")
        and hasattr(teleop, "enable_torque")
    )


def teleop_smooth_move_to(teleop, target_pos: dict, duration_s: float = 2.0, fps: int = 30) -> None:
    """通过线性插值将已驱动的遥操作平滑移动到 ``target_pos``。

    要求遥操作器支持反馈（即具有非空的
    ``feedback_features`` 并实现 ``disable_torque`` / ``enable_torque``）。

    ``target_pos`` 应位于遥操作的动作/反馈键空间中。
    对于同类设置（例如 SO-101 主臂 + SO-101 从臂），这直接匹配
    机器人动作键空间。

    TODO(Maxime)：这最多阻塞 ``duration_s`` 秒；在此期间
    从机器人不会接收新动作，这在 LeKiwi 上可能是个问题。
    """
    teleop.enable_torque()
    current = teleop.get_action()
    steps = max(int(duration_s * fps), 1)

    for step in range(steps + 1):
        t = step / steps
        interp = {
            k: current[k] * (1 - t) + target_pos[k] * t if k in target_pos else current[k] for k in current
        }
        teleop.send_feedback(interp)
        time.sleep(1 / fps)


def follower_smooth_move_to(
    robot, current: dict, target: dict, duration_s: float = 1.0, fps: int = 30
) -> None:
    """将从机器人从 ``current`` 动作平滑移动到 ``target`` 动作。

    当遥操作未驱动时使用：不是将主臂驱动到
    从臂，而是将从臂带到遥操作的当前位姿，以便
    机器人与操作员的手会合，而不是在第一帧跳向它。

    ``current`` 和 ``target`` 都必须位于机器人动作键空间中
    （即 ``robot_action_processor`` 的输出）。
    """
    steps = max(int(duration_s * fps), 1)

    for step in range(steps + 1):
        t = step / steps
        interp = {k: current[k] * (1 - t) + target[k] * t if k in target else current[k] for k in current}
        robot.send_action(interp)
        time.sleep(1 / fps)
