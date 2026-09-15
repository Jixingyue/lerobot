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
import time
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_openarm_leader import OpenArmLeaderConfig

logger = logging.getLogger(__name__)


class OpenArmLeader(Teleoperator):
    """
    使用 Damiao 电机的 OpenArm 主臂/遥操作臂。

    该遥操作设备使用 CAN 总线通信，从手动移动（力矩禁用）的
    Damiao 电机读取位置。
    """

    config_class = OpenArmLeaderConfig
    name = "openarm_leader"

    def __init__(self, config: OpenArmLeaderConfig):
        super().__init__(config)
        self.config = config

        # 手臂电机
        motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(
                send_id, motor_type_str, MotorNormMode.DEGREES
            )  # Damiao 电机始终使用角度
            motor.recv_id = recv_id
            motor.motor_type_str = motor_type_str
            motors[motor_name] = motor

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

    @property
    def action_features(self) -> dict[str, type]:
        """该遥操作设备产生的特征。"""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            if self.config.use_velocity_and_torque:
                features[f"{motor}.vel"] = float
                features[f"{motor}.torque"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """反馈特征（OpenArms 未实现）。"""
        return {}

    @property
    def is_connected(self) -> bool:
        """检查遥操作设备是否已连接。"""
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        连接到遥操作设备。

        对于手动控制，我们在连接后禁用力矩，以便可以
        用手移动手臂。
        """

        # 连接到 CAN 总线
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        # 如有需要则运行校准
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()

        if self.is_calibrated:
            self.bus.set_zero_position()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """检查遥操作设备是否已校准。"""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        运行 OpenArms 主臂的校准流程。

        校准流程：
        1. 禁用力矩（如果尚未禁用）
        2. 要求用户将手臂置于零位（自然下垂且夹爪闭合）
        3. 将该位置设为零位
        4. 记录每个关节的运动范围
        5. 保存校准
        """
        if self.calibration:
            # 校准文件已存在，询问用户是使用它还是重新运行校准
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")
        self.bus.disable_torque()

        # 第 1 步：设置零位
        input(
            "\nCalibration: Set Zero Position)\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )

        # 将所有电机的当前位置设为零位
        self.bus.set_zero_position()
        logger.info("Arm zero position set.")

        logger.info("Setting range: -90° to +90° by default for all joints")
        # TODO(Steven, Pepijn): 鉴于我们只使用角度，检查这里是否真的需要 MotorCalibration
        for motor_name, motor in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,
                range_min=-90,
                range_max=90,
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """
        为手动遥操作配置电机。

        对于手动控制，我们禁用力矩，以便可以用手移动手臂。
        """

        return self.bus.disable_torque() if self.config.manual_control else self.bus.configure_motors()

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        从主臂获取当前动作。

        这是遥操作设备的主要方法——它读取主臂的当前状态，
        并将其作为可发送给从动臂的动作返回。

        在一个 CAN 刷新周期内读取所有电机状态（pos/vel/torque）。
        """
        start = time.perf_counter()

        action_dict: dict[str, Any] = {}

        # 使用 sync_read_all_states 一次性获取 pos/vel/torque
        states = self.bus.sync_read_all_states()
        for motor in self.bus.motors:
            state = states.get(motor, {})
            action_dict[f"{motor}.pos"] = state.get("position")
            if self.config.use_velocity_and_torque:
                action_dict[f"{motor}.vel"] = state.get("velocity")
                action_dict[f"{motor}.torque"] = state.get("torque")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not yet implemented for OpenArm leader.")

    @check_if_not_connected
    def disconnect(self) -> None:
        """断开与遥操作设备的连接。"""

        # 断开 CAN 总线
        # 对于手动控制，在断开连接前确保力矩已禁用
        self.bus.disconnect(disable_torque=self.config.manual_control)
        logger.info(f"{self} disconnected.")
