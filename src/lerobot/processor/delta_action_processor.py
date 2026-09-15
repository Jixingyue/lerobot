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

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import PolicyAction, RobotAction

from .pipeline import ActionProcessorStep, ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict")
@dataclass
class MapTensorToDeltaActionDictStep(ActionProcessorStep):
    """
    将策略输出的扁平动作张量映射为结构化的增量动作字典。

    该步骤通常在策略输出连续动作向量之后使用。
    它将向量分解为末端执行器增量运动（x、y、z）的具名分量，
    以及可选的夹爪分量。

    Attributes:
        use_gripper: 如果为 True，则假定张量的第 4 个元素是
                     夹爪动作。
    """

    use_gripper: bool = True

    def action(self, action: PolicyAction) -> RobotAction:
        if not isinstance(action, PolicyAction):
            raise ValueError("Only PolicyAction is supported for this processor")

        if action.dim() > 1:
            action = action.squeeze(0)

        # TODO (maractingi): 添加旋转
        delta_action = {
            "delta_x": action[0].item(),
            "delta_y": action[1].item(),
            "delta_z": action[2].item(),
        }
        if self.use_gripper:
            delta_action["gripper"] = action[3].item()
        return delta_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z"]:
            features[PipelineFeatureType.ACTION][f"delta_{axis}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        if self.use_gripper:
            features[PipelineFeatureType.ACTION]["gripper"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("map_delta_action_to_robot_action")
@dataclass
class MapDeltaActionToRobotActionStep(RobotActionProcessorStep):
    """
    将来自遥操作设备的增量动作映射为用于逆运动学的机器人目标动作。

    该步骤将增量运动字典（例如来自手柄）转换为包含 "enabled" 标志和
    目标末端执行器位置的目标动作格式。它还处理缩放和噪声过滤。

    Attributes:
        position_scale: 用于缩放增量位置输入的系数。
        noise_threshold: 低于该幅值的增量输入被视为噪声，
                         不会触发 "enabled" 状态。
    """

    # 增量运动的缩放系数
    position_scale: float = 1.0
    noise_threshold: float = 1e-3  # 1 mm 阈值，用于过滤噪声

    def action(self, action: RobotAction) -> RobotAction:
        # 注意 (maractingi)：动作可以是来自遥操作设备的字典，也可以是来自策略的张量
        # TODO (maractingi)：更改遥操作设备中 target_xyz 的命名约定
        delta_x = action.pop("delta_x")
        delta_y = action.pop("delta_y")
        delta_z = action.pop("delta_z")
        gripper = action.pop("gripper")

        # 判断遥操作设备是否在主动提供输入
        # 如果检测到任何显著的位移增量，则视为 enabled
        position_magnitude = (delta_x**2 + delta_y**2 + delta_z**2) ** 0.5  # 位置使用欧几里得范数
        enabled = position_magnitude > self.noise_threshold  # 较小的阈值以避免噪声

        # 对增量进行适当的缩放
        scaled_delta_x = delta_x * self.position_scale
        scaled_delta_y = delta_y * self.position_scale
        scaled_delta_z = delta_z * self.position_scale

        # 对于手柄/键盘，没有旋转输入，因此设为 0
        # 未来可以为更复杂的遥操作设备扩展这些值
        target_wx = 0.0
        target_wy = 0.0
        target_wz = 0.0

        # 使用机器人目标格式更新动作
        action = {
            "enabled": enabled,
            "target_x": scaled_delta_x,
            "target_y": scaled_delta_y,
            "target_z": scaled_delta_z,
            "target_wx": target_wx,
            "target_wy": target_wy,
            "target_wz": target_wz,
            "gripper_vel": float(gripper),
        }

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z"]:
            features[PipelineFeatureType.ACTION].pop(f"delta_{axis}", None)
        features[PipelineFeatureType.ACTION].pop("gripper", None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features
