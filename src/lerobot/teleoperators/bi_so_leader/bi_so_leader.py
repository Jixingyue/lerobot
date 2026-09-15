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

import logging
from functools import cached_property

from lerobot.lerobot_types import RobotAction
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected

from ..so_leader import SOLeader, SOLeaderTeleopConfig
from ..teleoperator import Teleoperator
from .config_bi_so_leader import BiSOLeaderConfig

logger = logging.getLogger(__name__)


class BiSOLeader(BimanualMixin, Teleoperator):
    """
    由 TheRobotStudio 设计的[双臂 SO Leader 机械臂](https://github.com/TheRobotStudio/SO-ARM100)
    """

    config_class = BiSOLeaderConfig
    name = "bi_so_leader"

    def __init__(self, config: BiSOLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SOLeaderTeleopConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            use_degrees=config.left_arm_config.use_degrees,
            num_read_retries=config.left_arm_config.num_read_retries,
        )

        right_arm_config = SOLeaderTeleopConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            use_degrees=config.right_arm_config.use_degrees,
            num_read_retries=config.right_arm_config.num_read_retries,
        )

        self.left_arm = SOLeader(left_arm_config)
        self.right_arm = SOLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features

        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        # 双臂遥操作具有反馈（可被驱动以实现交接）。
        # 返回与 action_features 相同的结构，以保持与左右臂的一致性。
        left_arm_features = self.left_arm.feedback_features
        right_arm_features = self.right_arm.feedback_features

        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action_dict = {}

        # 添加 "left_" 前缀
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # 添加 "right_" 前缀
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def enable_torque(self) -> None:
        """启用两个主手臂的扭矩以实现平滑交接。"""
        self.left_arm.enable_torque()
        self.right_arm.enable_torque()

    def disable_torque(self) -> None:
        """禁用两个主手臂的扭矩以允许人工控制。"""
        self.left_arm.disable_torque()
        self.right_arm.disable_torque()

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        """将双臂反馈通过正确的前缀剥离路由到左右臂。

        接收形如以下键的反馈字典：left_shoulder_pan.pos、right_shoulder_pan.pos 等。
        通过移除前缀将其拆分并路由到每个臂。

        这使得 DAgger 平滑交接成为可能：当从策略控制过渡到人工干预时，
        两个主手臂会被指令到从手的当前位姿，以避免不连续。
        """
        # 按臂前缀拆分反馈
        left_feedback = {}
        right_feedback = {}

        for key, value in feedback.items():
            if key.startswith("left_"):
                # 剥离 "left_" 前缀并传递给左臂
                stripped_key = key[5:]  # len("left_") == 5
                left_feedback[stripped_key] = value
            elif key.startswith("right_"):
                # 剥离 "right_" 前缀并传递给右臂
                stripped_key = key[6:]  # len("right_") == 6
                right_feedback[stripped_key] = value

        # 发送到每个臂
        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)
