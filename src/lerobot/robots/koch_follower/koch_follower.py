#!/usr/bin/env python

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

import logging
import time
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_koch_follower import KochFollowerConfig

logger = logging.getLogger(__name__)


class KochFollower(Robot):
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot)，带或不带腕部到肘部的
        扩展件，由 [Tau Robotics](https://tau-robotics.com) 的 Alexander Koch 开发
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1)，由 Jess Moss 开发
    """

    config_class = KochFollowerConfig
    name = "koch_follower"

    def __init__(self, config: KochFollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl430-w250", norm_mode_body),
                "shoulder_lift": Motor(2, "xl430-w250", norm_mode_body),
                "elbow_flex": Motor(3, "xl330-m288", norm_mode_body),
                "wrist_flex": Motor(4, "xl330-m288", norm_mode_body),
                "wrist_roll": Motor(5, "xl330-m288", norm_mode_body),
                "gripper": Motor(6, "xl330-m288", MotorNormMode.RANGE_0_100),
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
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
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
        self.bus.disable_torque()
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
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"Move all joints except {full_turn_motors} sequentially through their entire "
            "ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

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
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            # 除夹爪外，所有电机都使用"扩展位置模式"，因为在关节模式下舵机
            # 无法旋转超过 360 度（从 0 到 4095）。而且在组装手臂时可能会出错，
            # 导致某个舵机在关键点上处于位置 0 或 4095
            for motor in self.bus.motors:
                if motor != "gripper":
                    self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

            # 夹爪使用"基于电流的位置控制"，使其受到电流限制。对于
            # follower 夹爪，这意味着即使其目标位置是完全夹合（命令两个夹爪手指
            # 合拢并接触），它抓取物体时也不会过度用力。
            # 对于 leader 夹爪，这意味着我们可以将其用作物理触发器，因为我们可以
            # 用手指用力使其移动，并在松开力后它会回到原来的目标位置。
            self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

            # 设置更好的 PID 值，以缩小记录的状态与动作之间的差距
            # TODO(rcadene): 实现一个自动流程，为每个电机设置最优 PID 值
            self.bus.write("Position_P_Gain", "elbow_flex", 1500)
            self.bus.write("Position_I_Gain", "elbow_flex", 0)
            self.bus.write("Position_D_Gain", "elbow_flex", 600)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # 读取手臂位置
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
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

        Args:
            action (RobotAction): 电机的目标位置。

        Returns:
            RobotAction: 发送给电机的动作，可能经过裁剪。
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # 当目标位置与当前位置相距太远时对其进行限制。
        # /!\ 由于需要从 follower 读取，预计帧率会变慢。
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
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
