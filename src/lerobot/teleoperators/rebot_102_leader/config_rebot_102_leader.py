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

from ..config import TeleoperatorConfig


@dataclass
class RebotArm102LeaderConfig:
    """Seeed Studio StarArm102 / reBot Arm 102 主臂的基础配置类。

    reBot Arm 102 是一条 7 关节（含夹爪）的主臂，由 FashionStar UART 智能舵机驱动。
    舵机通信通过 ``motorbridge-smart-servo`` 进行。
    """

    # 主臂所连接的 USB 转 UART 设备（例如 "/dev/ttyUSB0"）。
    port: str

    baudrate: int = 1_000_000

    # UART 总线上每个关节的舵机 id。
    joint_ids: dict[str, int] = field(
        default_factory=lambda: {
            "shoulder_pan": 0,
            "shoulder_lift": 1,
            "elbow_flex": 2,
            "wrist_flex": 3,
            "wrist_yaw": 4,
            "wrist_roll": 5,
            "gripper": 6,
        }
    )

    # 应用于原始舵机角度的逐关节符号，使主臂与从动臂的约定保持一致。
    # 夹爪还额外带有一个缩放系数（例如 -6），以将其范围扩展到
    # reBot B601 从动臂夹爪的行程。
    joint_directions: dict[str, int] = field(
        default_factory=lambda: {
            "shoulder_pan": -1,
            "shoulder_lift": -1,
            "elbow_flex": 1,
            "wrist_flex": 1,
            "wrist_yaw": 1,
            "wrist_roll": -1,
            "gripper": -6,
        }
    )

    # 以角度表示的逐关节 [min, max] 输出范围。与 reBot B601 从动臂的关节限位
    # 相匹配，使主臂动作可以逐键驱动从动臂。
    joint_ranges: dict[str, list[int]] = field(
        default_factory=lambda: {
            "shoulder_pan": [-150, 150],
            "shoulder_lift": [-200, 1],
            "elbow_flex": [-200, 1],
            "wrist_flex": [-80, 90],
            "wrist_yaw": [-90, 90],
            "wrist_roll": [-90, 90],
            "gripper": [-270, 0],
        }
    )


@TeleoperatorConfig.register_subclass("rebot_102_leader")
@dataclass
class RebotArm102LeaderTeleopConfig(TeleoperatorConfig, RebotArm102LeaderConfig):
    """reBot Arm 102 主臂遥操作设备的注册配置。"""

    pass
