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


@dataclass
class SOFollowerConfig:
    """SO Follower 机器人的基础配置类。"""

    # 连接手臂的端口
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` 出于安全目的限制相对位置目标向量的大小。
    # 将其设为正标量可让所有电机使用相同的值，或者设为将电机名称
    # 映射到该电机 max_relative_target 值的字典。
    max_relative_target: float | dict[str, float] | None = None

    # 相机
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 为向后兼容以前的策略/数据集，请设为 `True`
    use_degrees: bool = True

    # 连接时写入 Feetech STS3215 电机的位置模式 PID 增益。
    position_p_coefficient: int = 16
    position_i_coefficient: int = 0
    position_d_coefficient: int = 32

    # 电机 `sync_read` 失败时的额外重试次数。Feetech 总线偶尔会
    # 返回损坏的状态包（"Incorrect status packet!"），尤其是在多个关节同时
    # 移动时，这会导致控制循环中止。重试是立即执行的（无休眠）且仅在
    # 失败时发生，因此稳态读取开销不变。
    num_read_retries: int = 2


@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    pass


SO100FollowerConfig = SOFollowerRobotConfig
SO101FollowerConfig = SOFollowerRobotConfig
