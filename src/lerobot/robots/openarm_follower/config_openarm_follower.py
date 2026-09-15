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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

LEFT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-75.0, 75.0),
    "joint_2": (-90.0, 9.0),
    "joint_3": (-85.0, 85.0),
    "joint_4": (0.0, 135.0),
    "joint_5": (-85.0, 85.0),
    "joint_6": (-40.0, 40.0),
    "joint_7": (-80.0, 80.0),
    "gripper": (-65.0, 0.0),
}

RIGHT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-75.0, 75.0),
    "joint_2": (-9.0, 90.0),
    "joint_3": (-85.0, 85.0),
    "joint_4": (0.0, 135.0),
    "joint_5": (-85.0, 85.0),
    "joint_6": (-40.0, 40.0),
    "joint_7": (-80.0, 80.0),
    "gripper": (-65.0, 0.0),
}


@dataclass
class OpenArmFollowerConfigBase:
    """使用 Damiao 电机的 OpenArms follower 机器人的基础配置。"""

    # CAN 接口 - 每条手臂一个
    # 手臂 CAN 接口（例如 "can1"）
    # Linux："can0"、"can1" 等。
    port: str

    # 手臂所在侧："left" 或 "right"。如果为 "None" 将使用默认值
    side: str | None = None

    # CAN 接口类型："socketcan"（Linux）、"slcan"（串口）或 "auto"（自动检测）
    can_interface: str = "socketcan"

    # CAN FD 设置（OpenArms 默认使用 CAN FD）
    use_can_fd: bool = True
    can_bitrate: int = 1000000  # 标称比特率（1 Mbps）
    can_data_bitrate: int = 5000000  # CAN FD 数据比特率（5 Mbps）

    # 断开连接时是否禁用力矩
    disable_torque_on_disconnect: bool = True

    # 为 True 时，在观测特征中暴露每个电机的 `.vel` 和 `.torque`。
    # 默认为 False，以兼容仅支持位置的 openarm_mini 遥操作器。
    use_velocity_and_torque: bool = False

    # 相对目标位置的安全限制
    # 设为正标量可应用于所有电机，或设为将电机名称映射到限值的字典
    max_relative_target: float | dict[str, float] | None = None

    # 相机配置
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # OpenArms 的电机配置（每条手臂 7 自由度）
    # 将电机名称映射到 (send_can_id, recv_can_id, motor_type)
    # 基于：https://docs.openarm.dev/software/setup/configure-test
    # OpenArms 使用 4 种电机：
    # - DM8009 (DM-J8009P-2EC) 用于肩部（高扭矩）
    # - DM4340P 和 DM4340 用于肩旋转和肘部
    # - DM4310 (DM-J4310-2EC V1.1) 用于腕部和夹爪
    motor_config: dict[str, tuple[int, int, str]] = field(
        default_factory=lambda: {
            "joint_1": (0x01, 0x11, "dm8009"),  # J1 - 肩部平移 (DM8009)
            "joint_2": (0x02, 0x12, "dm8009"),  # J2 - 肩部升降 (DM8009)
            "joint_3": (0x03, 0x13, "dm4340"),  # J3 - 肩部旋转 (DM4340)
            "joint_4": (0x04, 0x14, "dm4340"),  # J4 - 肘部弯曲 (DM4340)
            "joint_5": (0x05, 0x15, "dm4310"),  # J5 - 腕部横滚 (DM4310)
            "joint_6": (0x06, 0x16, "dm4310"),  # J6 - 腕部俯仰 (DM4310)
            "joint_7": (0x07, 0x17, "dm4310"),  # J7 - 腕部旋转 (DM4310)
            "gripper": (0x08, 0x18, "dm4310"),  # J8 - 夹爪 (DM4310)
        }
    )

    # 用于位置控制的 MIT 控制参数（在 send_action 中使用）
    # 包含 8 个值的列表：[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
    position_kp: list[float] = field(
        default_factory=lambda: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 25.0]
    )
    position_kd: list[float] = field(default_factory=lambda: [5.0, 5.0, 3.0, 5.0, 0.3, 0.3, 0.3, 0.3])

    # 关节限位值。可通过 CLI（自定义值）或将 config.side 设为 'left' 或 'right' 来覆盖。
    # 如果 config.side 保持为 None 且未传入 CLI 值，则默认关节限位值较小以保证安全。
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "joint_1": (-5.0, 5.0),
            "joint_2": (-5.0, 5.0),
            "joint_3": (-5.0, 5.0),
            "joint_4": (0.0, 5.0),
            "joint_5": (-5.0, 5.0),
            "joint_6": (-5.0, 5.0),
            "joint_7": (-5.0, 5.0),
            "gripper": (-5.0, 0.0),
        }
    )


@RobotConfig.register_subclass("openarm_follower")
@dataclass
class OpenArmFollowerConfig(RobotConfig, OpenArmFollowerConfigBase):
    pass
