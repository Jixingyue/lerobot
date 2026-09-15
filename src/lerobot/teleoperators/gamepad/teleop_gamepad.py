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

import logging
import sys
from enum import IntEnum
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_gamepad import GamepadTeleopConfig

logger = logging.getLogger(__name__)


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class GamepadTeleop(Teleoperator):
    """
    使用手柄输入进行控制的遥操作类。
    """

    config_class = GamepadTeleopConfig
    name = "gamepad"

    def __init__(self, config: GamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.gamepad = None

        self.hidapi_fallback = config.hidapi_fallback
        if sys.platform == "darwin" and not self.hidapi_fallback:
            logger.warning(
                "On macOS, pygame may not reliably detect input from some controllers. "
                "If you experience issues, set `hidapi_fallback=true`."
            )

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        if self.hidapi_fallback:
            from .gamepad_utils import GamepadControllerHID as Gamepad
        else:
            from .gamepad_utils import GamepadController as Gamepad

        self.gamepad = Gamepad()
        self.gamepad.start()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        # 更新控制器以获取最新输入
        self.gamepad.update()

        # 从控制器获取运动增量
        delta_x, delta_y, delta_z = self.gamepad.get_deltas()

        # 根据手柄输入创建动作
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        action_dict = {
            "delta_x": gamepad_action[0],
            "delta_y": gamepad_action[1],
            "delta_z": gamepad_action[2],
        }

        # 默认夹爪动作为保持
        gripper_action = GripperAction.STAY.value
        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            gripper_action = gripper_action_map[gripper_command]
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        从手柄获取额外的控制事件，例如干预状态、回合终止、成功指示等。

        Returns:
            包含以下内容的字典：
                - is_intervention: bool - 人工当前是否正在干预
                - terminate_episode: bool - 是否终止当前回合
                - success: bool - 回合是否成功
                - rerecord_episode: bool - 是否重新录制回合
        """
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # 更新手柄状态以获取最新输入
        self.gamepad.update()

        # 检查干预是否激活
        is_intervention = self.gamepad.should_intervene()

        # 获取回合结束状态
        episode_end_status = self.gamepad.get_episode_end_status()
        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def disconnect(self) -> None:
        """断开与手柄的连接。"""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    @property
    def is_connected(self) -> bool:
        """检查手柄是否已连接。"""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """校准手柄。"""
        # 手柄无需校准
        pass

    def is_calibrated(self) -> bool:
        """检查手柄是否已校准。"""
        # 手柄不需要校准
        return True

    def configure(self) -> None:
        """配置手柄。"""
        # 无需额外配置
        pass

    def send_feedback(self, feedback: dict) -> None:
        """向手柄发送反馈。"""
        # 手柄不支持反馈
        pass
