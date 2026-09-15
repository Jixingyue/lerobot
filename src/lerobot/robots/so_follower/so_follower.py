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
import time
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so_follower import SOFollowerRobotConfig

logger = logging.getLogger(__name__)


class SOFollower(Robot):
    """
    实现 SO-100/101/10X 通用功能的泛型 SO follower 基类。
    设计为通过针对每个硬件型号的 `config_class` 和 `name` 来派生子类。
    """

    config_class = SOFollowerRobotConfig
    name = "so_follower"

    def __init__(self, config: SOFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        # 根据配置（如果可用）选择归一化模式
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            if getattr(self.cameras[cam], "use_rgb", True):
                features[cam] = (self.cameras[cam].height, self.cameras[cam].width, 3)
            if getattr(self.cameras[cam], "use_depth", False):
                features[f"{cam}_depth"] = (self.cameras[cam].height, self.cameras[cam].width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        我们假设在连接时手臂处于静止位置，
        可以安全地禁用力矩以运行校准。
        """

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # 校准文件已存在，询问用户是使用它还是运行新的校准
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        # 在适当时尝试使用缩减的电机集合调用 record_ranges_of_motion。
        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, self.config.position_p_coefficient)
                self.bus.write("I_Coefficient", motor, self.config.position_i_coefficient)
                self.bus.write("D_Coefficient", motor, self.config.position_d_coefficient)

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)  # 最大扭矩的 50%，避免烧毁
                    self.bus.write("Protection_Current", motor, 250)  # 最大电流的 50%，避免烧毁
                    self.bus.write("Overload_Torque", motor, 25)  # 过载时扭矩为 25%

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # 读取手臂位置
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 从相机采集图像
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """命令手臂移动到目标关节构型。

        相对动作的大小可能会根据配置参数 `max_relative_target` 被裁剪。
        在这种情况下，发送的动作与原始动作不同。
        因此，该函数始终返回实际发送的动作。

        Raises:
            RobotDeviceNotConnectedError: 如果机器人未连接。

        Returns:
            RobotAction: 发送给电机的动作，可能经过裁剪。
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # 当目标位置与当前位置相距太远时对其进行限制。
        # /!\ 由于需要从 follower 读取，预计帧率会变慢。
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # 将目标位置发送给手臂
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


SO100Follower = SOFollower
SO101Follower = SOFollower
