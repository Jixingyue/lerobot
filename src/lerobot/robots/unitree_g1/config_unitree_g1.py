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

_GAINS: dict[str, dict[str, list[float]]] = {
    "left_leg": {
        "kp": [150, 150, 150, 300, 40, 40],
        "kd": [2, 2, 2, 4, 2, 2],
    },  # pitch、roll、yaw、knee、ankle_pitch、ankle_roll
    "right_leg": {"kp": [150, 150, 150, 300, 40, 40], "kd": [2, 2, 2, 4, 2, 2]},
    "waist": {"kp": [250, 250, 250], "kd": [5, 5, 5]},  # yaw、roll、pitch
    "left_arm": {"kp": [50, 50, 80, 80], "kd": [3, 3, 3, 3]},  # shoulder_pitch/roll/yaw、elbow
    "left_wrist": {"kp": [40, 40, 40], "kd": [1.5, 1.5, 1.5]},  # roll、pitch、yaw
    "right_arm": {"kp": [50, 50, 80, 80], "kd": [3, 3, 3, 3]},
    "right_wrist": {"kp": [40, 40, 40], "kd": [1.5, 1.5, 1.5]},
}


def _build_gains() -> tuple[list[float], list[float]]:
    """从身体部位分组构建 kp 和 kd 列表。"""
    kp = [v for g in _GAINS.values() for v in g["kp"]]
    kd = [v for g in _GAINS.values() for v in g["kd"]]
    return kp, kd


_DEFAULT_KP, _DEFAULT_KD = _build_gains()


@RobotConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1Config(RobotConfig):
    kp: list[float] = field(default_factory=lambda: _DEFAULT_KP.copy())
    kd: list[float] = field(default_factory=lambda: _DEFAULT_KD.copy())

    # 默认关节位置
    default_positions: list[float] = field(default_factory=lambda: [0.0] * 29)

    # 控制循环时间步长
    control_dt: float = 1.0 / 250.0  # 250Hz

    # 启动 mujoco 仿真
    is_simulation: bool = True

    # ZMQ 桥接的 Socket 配置
    robot_ip: str = "192.168.123.164"  # 默认 G1 IP

    # 相机（基于 ZMQ 的远程相机）
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 使用手臂逆运动学求解器补偿 unitree 手臂上的重力
    gravity_compensation: bool = False

    # 控制器类名，例如 GrootLocomotionController / HolosomaLocomotionController /
    # SonicWholeBodyController。设为 None 则禁用。
    controller: str | None = None
