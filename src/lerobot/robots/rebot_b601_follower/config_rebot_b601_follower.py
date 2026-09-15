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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class RebotB601FollowerConfig:
    """Seeed Studio reBot B601-DM follower 手臂的基础配置类。

    B601-DM 是一款由 Damiao CAN 电机驱动的 6 自由度手臂加夹爪。电机
    通信通过 ``motorbridge`` 包进行。
    """

    # 通信端口。对于 ``can_adapter="damiao"``，这是 Damiao 串口
    # 桥接设备（例如 "/dev/ttyACM0"）；对于 ``can_adapter="socketcan"``，
    # 则是 CAN 通道名称（例如 "can0"）。
    port: str

    # CAN 适配器类型：
    #   "damiao"    - Damiao 专用串口桥接（默认）
    #   "socketcan" - 基于 SocketCAN 的适配器（PCAN、slcan、嵌入式控制器等）
    can_adapter: str = "damiao"

    # Damiao 串口桥接的波特率（仅在 can_adapter="damiao" 时使用）。
    dm_serial_baud: int = 921600

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` 出于安全目的限制相对位置目标向量的大小
    # （以度为单位）。设为正标量可对所有电机应用相同的值，
    # 或设为将电机名称映射到各电机值的字典。
    max_relative_target: float | dict[str, float] | None = None

    # 相机
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 将电机名称映射到其 (send_can_id, recv_can_id) 对。
    motor_can_ids: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "shoulder_pan": (0x01, 0x11),
            "shoulder_lift": (0x02, 0x12),
            "elbow_flex": (0x03, 0x13),
            "wrist_flex": (0x04, 0x14),
            "wrist_yaw": (0x05, 0x15),
            "wrist_roll": (0x06, 0x16),
            "gripper": (0x07, 0x17),
        }
    )

    # POS_VEL 手臂和 FORCE_POS 夹爪各关节的最大速度（度/秒）（按电机顺序）。
    pos_vel_velocity: float | list[float] = field(
        default_factory=lambda: [150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 900.0]
    )

    # 手臂控制模式："mit" 或 "pos_vel"。
    control_mode: str = "mit"

    # 各手臂关节的 MIT kp/kd（按电机顺序）。当 control_mode="pos_vel" 时不使用。
    mit_kp: float | list[float] = field(default_factory=lambda: [45.0, 45.0, 45.0, 8.0, 9.0, 8.0, 8.0])
    mit_kd: float | list[float] = field(default_factory=lambda: [12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0])

    # 夹爪控制模式："force_pos" 或 "mit"。
    gripper_control_mode: str = "force_pos"

    # 仅 FORCE_POS：最大夹持力，取值范围 [0, 1]。
    gripper_torque_ratio: float = 0.07

    # 仅 MIT 模式。
    gripper_mit_kp: float = 8.0
    gripper_mit_kd: float = 0.3

    # 软关节限位（度）。每次动作都会按此裁剪。
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-150.0, 150.0),
            "shoulder_lift": (-200.0, 1.0),
            "elbow_flex": (-200.0, 1.0),
            "wrist_flex": (-80.0, 90.0),
            "wrist_yaw": (-90.0, 90.0),
            "wrist_roll": (-90.0, 90.0),
            "gripper": (-270.0, 0.0),
        }
    )


@RobotConfig.register_subclass("rebot_b601_follower")
@dataclass
class RebotB601FollowerRobotConfig(RobotConfig, RebotB601FollowerConfig):
    """reBot B601-DM follower 机器人的注册配置。"""

    pass
