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
"""键盘遥操作设备的配置。"""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    """KeyboardTeleopConfig"""

    # TODO(Steven): 考虑在这里设置我们想要捕获/监听的按键


@TeleoperatorConfig.register_subclass("keyboard_ee")
@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    """键盘末端执行器遥操作设备的配置。

    用于通过键盘输入控制机器人末端执行器。

    Attributes:
        use_gripper: 是否在动作中包含夹爪控制
    """

    use_gripper: bool = True


@TeleoperatorConfig.register_subclass("keyboard_rover")
@dataclass
class KeyboardRoverTeleopConfig(TeleoperatorConfig):
    """键盘漫游车遥操作设备的配置。

    用于通过 WASD 按键控制 EarthRover Mini Plus 等移动机器人。

    Attributes:
        linear_speed: 默认线速度大小（SDK 机器人的范围为 -1 到 1）
        angular_speed: 默认角速度大小（SDK 机器人的范围为 -1 到 1）
        speed_increment: 使用 +/- 键增加/减少速度的步长
        turn_assist_ratio: 使用 A/D 键转向时前进运动的倍率（0.0-1.0）
        angular_speed_ratio: 用于同步调整的角速度与线速度之比
        min_linear_speed: 减速时的最小线速度（防止降为零）
        min_angular_speed: 减速时的最小角速度（防止降为零）
    """

    linear_speed: float = 1.0
    angular_speed: float = 1.0
    speed_increment: float = 0.1
    turn_assist_ratio: float = 0.3
    angular_speed_ratio: float = 0.6
    min_linear_speed: float = 0.1
    min_angular_speed: float = 0.05
