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
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_openarm_follower import (
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
    OpenArmFollowerConfig,
)

logger = logging.getLogger(__name__)


class OpenArmFollower(Robot):
    """
    OpenArms 从动机器人，使用 CAN 总线通信控制带夹爪的 7 自由度机械臂。
    该机械臂使用 MIT 控制模式下的 Damiao 电机。
    """

    config_class = OpenArmFollowerConfig
    name = "openarm_follower"

    def __init__(self, config: OpenArmFollowerConfig):
        super().__init__(config)
        self.config = config

        # 机械臂电机
        motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(
                send_id, motor_type_str, MotorNormMode.DEGREES
            )  # Damiao 电机始终使用度
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

        if config.side is not None:
            if config.side == "left":
                config.joint_limits = LEFT_DEFAULT_JOINTS_LIMITS
            elif config.side == "right":
                config.joint_limits = RIGHT_DEFAULT_JOINTS_LIMITS
            else:
                raise ValueError(
                    "config.side must be either 'left', 'right' (for default values) or 'None' (for CLI values)"
                )
        else:
            logger.info(
                "Set config.side to either 'left' or 'right' to use pre-configured values for joint limits."
            )
        logger.info(f"Values used for joint limits: {config.joint_limits}.")

        # 初始化相机
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """用于观测和动作空间的电机特征。"""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            if self.config.use_velocity_and_torque:
                features[f"{motor}.vel"] = float
                features[f"{motor}.torque"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """用于观测空间的相机特征。"""
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
        """电机和相机组合的观测特征。"""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征。"""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """检查机器人是否已连接。"""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        连接到机器人并可选地进行校准。

        我们假设在连接时，机械臂处于安全的静止位置，
        并且如有需要，可以安全地禁用力矩来运行校准。
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

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        self.bus.enable_torque()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """检查机器人是否已校准。"""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        运行 OpenArms 机器人的校准流程。

        校准流程：
        1. 禁用力矩
        2. 提示用户将机械臂置于悬挂位置并闭合夹爪
        3. 将此位置设为零位
        4. 记录每个关节的运动范围
        5. 保存校准
        """
        if self.calibration:
            # 校准文件已存在，询问用户是使用现有校准还是运行新的校准
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")
        self.bus.disable_torque()

        # 步骤 1：设置零位
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

        logger.info("Setting range: -90° to +90° for safety by default for all joints")
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
        """使用适当的设置配置电机。"""
        # TODO(Steven, Pepijn): 与主端发生的情况略有不同
        with self.bus.torque_disabled():
            self.bus.configure_motors()

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        从机器人获取当前观测，包括位置、速度和力矩。

        在一个 CAN 刷新周期内读取所有电机状态（位置/速度/力矩），
        而不是分 3 次单独读取。
        """
        start = time.perf_counter()

        obs_dict: dict[str, Any] = {}

        states = self.bus.sync_read_all_states()

        for motor in self.bus.motors:
            state = states.get(motor, {})
            obs_dict[f"{motor}.pos"] = state.get("position", 0.0)
            if self.config.use_velocity_and_torque:
                obs_dict[f"{motor}.vel"] = state.get("velocity", 0.0)
                obs_dict[f"{motor}.torque"] = state.get("torque", 0.0)

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

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        """
        向机器人发送动作指令。

        动作幅度可能会根据安全限制进行裁剪。

        Args:
            action: 包含电机位置的字典（例如 "joint_1.pos"、"joint_2.pos"）
            custom_kp: 每个电机可选的自定义 kp 增益（例如 {"joint_1": 120.0, "joint_2": 150.0}）
            custom_kd: 每个电机可选的自定义 kd 增益（例如 {"joint_1": 1.5, "joint_2": 2.0}）

        Returns:
            实际发送的动作（可能经过裁剪）
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # 对机械臂施加关节限位裁剪
        for motor_name, position in goal_pos.items():
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(f"Clipped {motor_name} from {position:.2f}° to {clipped_position:.2f}°")
                goal_pos[motor_name] = clipped_position

        # 当目标位置距离当前位置太远时进行限制。
        # /!\ 由于需要从从动端读取，预计帧率会降低。
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # TODO(Steven, Pepijn): 重构写入逻辑
        # 电机名称到增益索引的映射
        motor_index = {
            "joint_1": 0,
            "joint_2": 1,
            "joint_3": 2,
            "joint_4": 3,
            "joint_5": 4,
            "joint_6": 5,
            "joint_7": 6,
            "gripper": 7,
        }

        # 对机械臂使用批量 MIT 控制（先发送所有指令，再收集响应）
        commands = {}
        for motor_name, position_degrees in goal_pos.items():
            idx = motor_index.get(motor_name, 0)
            # 如果提供了自定义增益则使用，否则使用配置默认值
            if custom_kp is not None and motor_name in custom_kp:
                kp = custom_kp[motor_name]
            else:
                kp = (
                    self.config.position_kp[idx]
                    if isinstance(self.config.position_kp, list)
                    else self.config.position_kp
                )
            if custom_kd is not None and motor_name in custom_kd:
                kd = custom_kd[motor_name]
            else:
                kd = (
                    self.config.position_kd[idx]
                    if isinstance(self.config.position_kd, list)
                    else self.config.position_kd
                )
            commands[motor_name] = (kp, kd, position_degrees, 0.0, 0.0)

        self.bus._mit_control_batch(commands)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        """断开与机器人的连接。"""

        # 断开 CAN 总线
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        # 断开相机
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
