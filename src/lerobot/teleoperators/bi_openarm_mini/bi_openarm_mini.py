#!/usr/bin/env python

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

import logging
from functools import cached_property

from lerobot.lerobot_types import RobotAction
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected

from ..openarm_mini import OpenArmMini, OpenArmMiniConfig
from ..teleoperator import Teleoperator
from .config_bi_openarm_mini import BiOpenArmMiniConfig

logger = logging.getLogger(__name__)


class BiOpenArmMini(BimanualMixin, Teleoperator):
    """双臂 OpenArm Mini 遥操作设备。

    由两个单臂 :class:`OpenArmMini` 实例组合而成。每个臂的动作和反馈键
    通过 ``left_`` / ``right_`` 前缀进行命名空间区分，从而使双臂主手能够
    遥操作双臂 OpenArm 从手机器人。
    """

    config_class = BiOpenArmMiniConfig
    name = "bi_openarm_mini"

    def __init__(self, config: BiOpenArmMiniConfig):
        super().__init__(config)
        self.config = config

        # 无论用户在单臂基础配置中传入了什么，`side` 都会被强制设置为
        # left/right —— 双臂封装层负责侧别语义。
        left_arm_config = OpenArmMiniConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            side="left",
            use_degrees=config.left_arm_config.use_degrees,
        )

        right_arm_config = OpenArmMiniConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            side="right",
            use_degrees=config.right_arm_config.use_degrees,
        )

        self.left_arm = OpenArmMini(left_arm_config)
        self.right_arm = OpenArmMini(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.action_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.feedback_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.feedback_features.items()},
        }

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action: RobotAction = {}
        for k, v in self.left_arm.get_action().items():
            action[f"left_{k}"] = v
        for k, v in self.right_arm.get_action().items():
            action[f"right_{k}"] = v
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        left_fb = {k.removeprefix("left_"): v for k, v in feedback.items() if k.startswith("left_")}
        right_fb = {k.removeprefix("right_"): v for k, v in feedback.items() if k.startswith("right_")}
        if left_fb:
            self.left_arm.send_feedback(left_fb)
        if right_fb:
            self.right_arm.send_feedback(right_fb)
