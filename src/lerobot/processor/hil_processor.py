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

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.teleoperators.utils import TeleopEvents

if TYPE_CHECKING:
    from lerobot.teleoperators.teleoperator import Teleoperator

from lerobot.lerobot_types import EnvTransition, PolicyAction, TransitionKey

from .pipeline import (
    ComplementaryDataProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TruncatedProcessorStep,
)

GRIPPER_KEY = "gripper"
DISCRETE_PENALTY_KEY = "discrete_penalty"
TELEOP_ACTION_KEY = "teleop_action"


@runtime_checkable
class HasTeleopEvents(Protocol):
    """
    为提供遥操作事件的对象定义的最小协议。

    本协议定义了 `get_teleop_events()` 方法，使处理器
    步骤能够与支持基于事件的控制（例如终止 episode
    或标记成功）的遥操作器交互，而无需了解遥操作器的
    具体类。
    """

    def get_teleop_events(self) -> dict[str, Any]:
        """
        从遥操作器获取额外的控制事件。

        Returns:
            包含控制事件的字典，例如：
            - `is_intervention`：bool——人类当前是否正在介入。
            - `terminate_episode`：bool——是否终止当前 episode。
            - `success`：bool——episode 是否成功。
            - `rerecord_episode`：bool——是否重新录制该 episode。
        """
        ...


# 类型变量，约束为同时实现了事件接口的 Teleoperator 子类
TeleopWithEvents = TypeVar("TeleopWithEvents", bound="Teleoperator")


def _check_teleop_with_events(teleop: "Teleoperator") -> None:
    """
    运行时检查遥操作器是否实现了 `HasTeleopEvents` 协议。

    Args:
        teleop: 要检查的遥操作器实例。

    Raises:
        TypeError: 当遥操作器没有 `get_teleop_events` 方法时。
    """
    if not isinstance(teleop, HasTeleopEvents):
        raise TypeError(
            f"Teleoperator {type(teleop).__name__} must implement get_teleop_events() method. "
            f"Compatible teleoperators: GamepadTeleop, KeyboardEndEffectorTeleop"
        )


@ProcessorStepRegistry.register("add_teleop_action_as_complementary_data")
@dataclass
class AddTeleopActionAsComplimentaryDataStep(ComplementaryDataProcessorStep):
    """
    将遥操作器的原始动作添加到 transition 的 complementary data 中。

    这适用于人在回路（human-in-the-loop）场景：人类的输入需要
    对下游处理器可用，例如在介入期间覆盖策略的
    动作。

    Attributes:
        teleop_device: 从中获取动作的遥操作器实例。
    """

    teleop_device: "Teleoperator"

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        获取遥操作器的动作并将其添加到 complementary data 中。

        Args:
            complementary_data: 传入的 complementary data 字典。

        Returns:
            一个新字典，其中遥操作器动作被添加在
            `teleop_action` 键下。
        """
        new_complementary_data = dict(complementary_data)
        new_complementary_data[TELEOP_ACTION_KEY] = self.teleop_device.get_action()
        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("add_teleop_action_as_info")
@dataclass
class AddTeleopEventsAsInfoStep(InfoProcessorStep):
    """
    将遥操作器控制事件（例如终止、成功）添加到 transition 的 info 中。

    本步骤从支持基于事件交互的遥操作器中提取控制事件，
    使这些信号可供系统的其他部分使用。

    Attributes:
        teleop_device: 实现了
                       `HasTeleopEvents` 协议的遥操作器实例。
    """

    teleop_device: TeleopWithEvents

    def __post_init__(self):
        """在初始化后校验所提供的遥操作器是否支持事件。"""
        _check_teleop_with_events(self.teleop_device)

    def info(self, info: dict) -> dict:
        """
        获取遥操作器事件并更新 info 字典。

        Args:
            info: 传入的 info 字典。

        Returns:
            包含遥操作器事件的新字典。
        """
        new_info = dict(info)

        teleop_events = self.teleop_device.get_teleop_events()
        new_info.update(teleop_events)
        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("image_crop_resize_processor")
@dataclass
class ImageCropResizeProcessorStep(ObservationProcessorStep):
    """
    对图像观测进行裁剪和/或缩放。

    本步骤遍历观测字典中的所有图像键，并应用
    指定的变换。它会处理设备放置问题：对于 MPS 等
    某些加速器不支持的操作，必要时会将张量移至
    CPU。

    Attributes:
        crop_params_dict: 将图像键映射到裁剪参数
                          （top、left、height、width）的字典。
        resize_size: 将所有图像缩放到的 (height, width) 元组。
    """

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None

    def observation(self, observation: dict) -> dict:
        """
        对观测字典中的所有图像应用裁剪和缩放。

        Args:
            observation: 观测字典，可能包含图像张量。

        Returns:
            图像经过变换后的新观测字典。
        """
        if self.resize_size is None and not self.crop_params_dict:
            return observation

        new_observation = dict(observation)

        # 处理观测中的所有图像键
        for key in observation:
            if "image" not in key:
                continue

            image = observation[key]
            device = image.device
            # 注意（maractingi）：crop 和 resize 没有 mps 内核，因此需要移到 cpu
            if device.type == "mps":
                image = image.cpu()
            # 如果为该键提供了裁剪参数，则进行裁剪
            if self.crop_params_dict is not None and key in self.crop_params_dict:
                crop_params = self.crop_params_dict[key]
                image = F.crop(image, *crop_params)
            if self.resize_size is not None:
                image = F.resize(image, self.resize_size)
                image = image.clamp(0.0, 1.0)
            new_observation[key] = image.to(device)

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        返回本步骤的配置，用于序列化。

        Returns:
            包含裁剪参数和缩放尺寸的字典。
        """
        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        如果应用了缩放，则更新策略特征字典中的图像特征形状。

        Args:
            features: 策略特征字典。

        Returns:
            更新后的策略特征字典，其中图像形状已更新。
        """
        if self.resize_size is None:
            return features
        for key in features[PipelineFeatureType.OBSERVATION]:
            if "image" in key:
                nb_channel = features[PipelineFeatureType.OBSERVATION][key].shape[0]
                features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                    type=features[PipelineFeatureType.OBSERVATION][key].type,
                    shape=(nb_channel, *self.resize_size),
                )
        return features


@dataclass
@ProcessorStepRegistry.register("time_limit_processor")
class TimeLimitProcessorStep(TruncatedProcessorStep):
    """
    跟踪 episode 步数，并通过截断 episode 来强制执行时间限制。

    Attributes:
        max_episode_steps: 每个 episode 允许的最大步数。
        current_step: 当前活动 episode 的步数计数。
    """

    max_episode_steps: int
    current_step: int = 0

    def truncated(self, truncated: bool) -> bool:
        """
        递增步数计数器，并在达到时间限制时设置截断标志。

        Args:
            truncated: 传入的截断标志。

        Returns:
            达到 episode 步数限制时返回 True，否则返回传入的值。
        """
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        # TODO (steven)：是否缺少 else truncated = False？
        return truncated

    def get_config(self) -> dict[str, Any]:
        """
        返回本步骤的配置，用于序列化。

        Returns:
            包含 `max_episode_steps` 的字典。
        """
        return {
            "max_episode_steps": self.max_episode_steps,
        }

    def reset(self) -> None:
        """重置步数计数器，通常在新 episode 开始时调用。"""
        self.current_step = 0

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("gym_hil_adapter_processor")
class GymHILAdapterProcessorStep(ProcessorStep):
    """
    将 `gym-hil` 环境的输出适配为 `lerobot` 处理器期望的格式。

    本步骤通过以下方式规范化 `transition` 对象：
    1. 将 `teleop_action` 从 `info` 复制到 `complementary_data`。
    2. 将 `is_intervention` 从 `info`（使用字符串键）复制到 `info`（使用枚举键）。
    3. 将 `discrete_penalty` 从 `info` 复制到 `complementary_data`。
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        if TELEOP_ACTION_KEY in info:
            complementary_data[TELEOP_ACTION_KEY] = info[TELEOP_ACTION_KEY]

        if DISCRETE_PENALTY_KEY in info:
            complementary_data[DISCRETE_PENALTY_KEY] = info[DISCRETE_PENALTY_KEY]

        if "is_intervention" in info:
            info[TeleopEvents.IS_INTERVENTION] = info["is_intervention"]

        transition[TransitionKey.INFO] = info
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("gripper_penalty_processor")
class GripperPenaltyProcessorStep(ProcessorStep):
    """
    对离散夹爪动作施加较小的逐 transition 代价。

    仅在所命令的动作确实会使夹爪从一个极端
    切换到另一个极端（张开时命令闭合，或闭合时命令张开）时
    触发。这可以抑制夹爪抖动，同时不对“保持”和
    继续向同方向饱和的命令进行惩罚。

    Attributes:
        penalty: 要施加的负奖励值。
        max_gripper_pos: 夹爪的最大位置值，用于归一化。
        open_threshold: 归一化状态低于该值时，夹爪被视为“张开”。
        closed_threshold: 归一化状态高于该值时，夹爪被视为“闭合”。
    """

    penalty: float = -0.02
    max_gripper_pos: float = 30.0
    open_threshold: float = 0.1
    closed_threshold: float = 0.9

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        计算夹爪惩罚并将其添加到 complementary data 中。

        Args:
            transition: 传入的环境 transition。

        Returns:
            修改后的 transition，其中 complementary data 已添加惩罚。
        """
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        raw_joint_positions = complementary_data.get("raw_joint_positions")
        if raw_joint_positions is None:
            return new_transition

        current_gripper_pos = raw_joint_positions.get(f"{GRIPPER_KEY}.pos", None)
        if current_gripper_pos is None:
            return new_transition

        # reset 期间，transition 可能尚未携带任何动作。
        if action is None:
            return new_transition

        # 夹爪动作预期位于动作的最后一个维度。
        gripper_action = action[-1].item()
        gripper_action_normalized = gripper_action / self.max_gripper_pos

        # 归一化夹爪状态和动作
        gripper_state_normalized = current_gripper_pos / self.max_gripper_pos

        # 与原始实现一致地计算惩罚布尔值：
        #   - 当前张开 且 目标闭合  -> 闭合切换
        #   - 当前闭合 且 目标张开  -> 张开切换
        is_open = gripper_state_normalized < self.open_threshold
        is_closed = gripper_state_normalized > self.closed_threshold
        cmd_close = gripper_action_normalized > self.closed_threshold
        cmd_open = gripper_action_normalized < self.open_threshold
        gripper_penalty_bool = (is_open and cmd_close) or (is_closed and cmd_open)

        gripper_penalty = self.penalty * int(gripper_penalty_bool)

        # 用惩罚信息更新 complementary data
        new_complementary_data = dict(complementary_data)
        new_complementary_data[DISCRETE_PENALTY_KEY] = gripper_penalty
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        返回本步骤的配置，用于序列化。

        Returns:
            包含惩罚值、夹爪最大位置以及
            张开/闭合阈值的字典。
        """
        return {
            "penalty": self.penalty,
            "max_gripper_pos": self.max_gripper_pos,
            "open_threshold": self.open_threshold,
            "closed_threshold": self.closed_threshold,
        }

    def reset(self) -> None:
        """重置处理器的内部状态。"""
        pass

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("intervention_action_processor")
class InterventionActionProcessorStep(ProcessorStep):
    """
    处理人类介入，覆盖策略动作并管理 episode 终止。

    当检测到介入时（通过 `info` 字典中的遥操作器事件），
    本步骤会用人类的遥操作动作替换策略的动作。
    它还会处理终止 episode 或标记成功的信号。

    Attributes:
        use_gripper: 遥操作动作中是否包含夹爪。
        terminate_on_success: 若为 True，收到 `success`
                              事件时自动设置 `done` 标志。
    """

    use_gripper: bool = False
    terminate_on_success: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        处理 transition 以应对介入。

        Args:
            transition: 传入的环境 transition。

        Returns:
            修改后的 transition，可能覆盖了动作，并更新了
            奖励和终止状态。
        """
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        # 从 complementary data 获取介入信号
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()

        # 介入处于活动状态时覆盖动作
        if is_intervention and teleop_action is not None:
            if isinstance(teleop_action, dict):
                # 将 teleop_action 字典转换为张量格式
                action_list = [
                    teleop_action.get("delta_x", 0.0),
                    teleop_action.get("delta_y", 0.0),
                    teleop_action.get("delta_z", 0.0),
                ]
                if self.use_gripper:
                    action_list.append(teleop_action.get(GRIPPER_KEY, 1.0))
            elif isinstance(teleop_action, np.ndarray):
                action_list = teleop_action.tolist()
            else:
                action_list = teleop_action

            teleop_action_tensor = torch.tensor(action_list, dtype=action.dtype, device=action.device)
            new_transition[TransitionKey.ACTION] = teleop_action_tensor

        # 处理 episode 终止
        new_transition[TransitionKey.DONE] = bool(terminate_episode) or (
            self.terminate_on_success and success
        )
        new_transition[TransitionKey.REWARD] = float(success)

        # 用介入元数据更新 info
        info = new_transition.get(TransitionKey.INFO, {})
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        # 用遥操作动作更新 complementary data
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        返回本步骤的配置，用于序列化。

        Returns:
            包含本步骤配置属性的字典。
        """
        return {
            "use_gripper": self.use_gripper,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("reward_classifier_processor")
class RewardClassifierProcessorStep(ProcessorStep):
    """
    将预训练的奖励分类器应用于图像观测以预测成功。

    本步骤使用一个模型判断当前状态是否成功，并更新
    奖励，还可能终止 episode。

    Attributes:
        pretrained_path: 预训练奖励分类器模型的路径。
        device: 运行分类器的设备。
        success_threshold: 将预测视为成功的概率阈值。
        success_reward: 成功时赋予的奖励值。
        terminate_on_success: 若为 True，分类成功时终止 episode。
        reward_classifier: 已加载的分类器模型实例。
    """

    pretrained_path: str | None = None
    device: str = "cpu"
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True

    reward_classifier: Any = None

    def __post_init__(self):
        """在 dataclass 创建后初始化奖励分类器模型。"""
        if self.pretrained_path is not None:
            from lerobot.rewards.classifier.modeling_classifier import Classifier

            self.reward_classifier = Classifier.from_pretrained(self.pretrained_path)
            self.reward_classifier.to(self.device)
            self.reward_classifier.eval()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        处理 transition，将奖励分类器应用于其图像观测。

        Args:
            transition: 传入的环境 transition。

        Returns:
            修改后的 transition，根据
            分类器的预测更新了奖励和 done 标志。
        """
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.reward_classifier is None:
            return new_transition

        # 从观测中提取图像
        images = {key: value for key, value in observation.items() if "image" in key}

        if not images:
            return new_transition

        # 运行奖励分类器
        start_time = time.perf_counter()
        with torch.inference_mode():
            success = self.reward_classifier.predict_reward(images, threshold=self.success_threshold)

        classifier_frequency = 1 / (time.perf_counter() - start_time)

        # 计算奖励和终止
        reward = new_transition.get(TransitionKey.REWARD, 0.0)
        terminated = new_transition.get(TransitionKey.DONE, False)

        if math.isclose(success, 1, abs_tol=1e-2):
            reward = self.success_reward
            if self.terminate_on_success:
                terminated = True

        # 更新 transition
        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        # 用分类器频率更新 info
        info = new_transition.get(TransitionKey.INFO, {})
        info["reward_classifier_frequency"] = classifier_frequency
        new_transition[TransitionKey.INFO] = info

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        返回本步骤的配置，用于序列化。

        Returns:
            包含本步骤配置属性的字典。
        """
        return {
            "device": self.device,
            "success_threshold": self.success_threshold,
            "success_reward": self.success_reward,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
