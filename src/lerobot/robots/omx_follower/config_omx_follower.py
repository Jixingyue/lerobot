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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("omx_follower")
@dataclass
class OmxFollowerConfig(RobotConfig):
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
    use_degrees: bool = False
