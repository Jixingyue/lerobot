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

from ..config import TeleoperatorConfig


@dataclass
class OpenArmLeaderConfigBase:
    """使用 Damiao 电机的 OpenArms 主臂/遥操作设备的基础配置。"""

    # CAN 接口——每条手臂一个
    # 手臂的 CAN 接口（例如 "can3"）
    # Linux："can0"、"can1" 等
    port: str

    # CAN 接口类型："socketcan"（Linux）、"slcan"（串口）或 "auto"（自动检测）
    can_interface: str = "socketcan"

    # CAN FD 设置（OpenArms 默认使用 CAN FD）
    use_can_fd: bool = True
    can_bitrate: int = 1000000  # 标称比特率（1 Mbps）
    can_data_bitrate: int = 5000000  # CAN FD 的数据比特率（5 Mbps）

    # OpenArms 的电机配置（每条手臂 7 个自由度）
    # 将电机名称映射到 (send_can_id, recv_can_id, motor_type)
    # 基于：https://docs.openarm.dev/software/setup/configure-test
    # OpenArms 使用 4 种电机：
    # - DM8009 (DM-J8009P-2EC) 用于肩部（高扭矩）
    # - DM4340P 和 DM4340 用于肩部旋转和肘部
    # - DM4310 (DM-J4310-2EC V1.1) 用于腕部和夹爪
    motor_config: dict[str, tuple[int, int, str]] = field(
        default_factory=lambda: {
            "joint_1": (0x01, 0x11, "dm8009"),  # J1 - 肩部水平旋转 (DM8009)
            "joint_2": (0x02, 0x12, "dm8009"),  # J2 - 肩部俯仰 (DM8009)
            "joint_3": (0x03, 0x13, "dm4340"),  # J3 - 肩部旋转 (DM4340)
            "joint_4": (0x04, 0x14, "dm4340"),  # J4 - 肘部弯曲 (DM4340)
            "joint_5": (0x05, 0x15, "dm4310"),  # J5 - 腕部横滚 (DM4310)
            "joint_6": (0x06, 0x16, "dm4310"),  # J6 - 腕部俯仰 (DM4310)
            "joint_7": (0x07, 0x17, "dm4310"),  # J7 - 腕部旋转 (DM4310)
            "gripper": (0x08, 0x18, "dm4310"),  # J8 - 夹爪 (DM4310)
        }
    )

    # 手动控制的力矩模式设置
    # 启用时，电机将禁用力矩以便手动移动
    manual_control: bool = True

    # 为 True 时，在动作特征中暴露每个电机的 `.vel` 和 `.torque`。
    # 默认为 False，以兼容仅使用位置的 openarm_mini 遥操作设备。
    use_velocity_and_torque: bool = False

    # TODO(Steven, Pepijn): 未使用……？
    # MIT 控制参数（当 manual_control=False 时用于力矩控制）
    # 8 个值的列表：[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
    position_kp: list[float] = field(
        default_factory=lambda: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 16.0]
    )
    position_kd: list[float] = field(default_factory=lambda: [3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2])


@TeleoperatorConfig.register_subclass("openarm_leader")
@dataclass
class OpenArmLeaderConfig(TeleoperatorConfig, OpenArmLeaderConfigBase):
    pass
