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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("koch_leader")
@dataclass
class KochLeaderConfig(TeleoperatorConfig):
    # 用于连接机械臂的端口
    port: str

    # 将机械臂设置为力矩模式，并将夹爪电机设为该值。这样就可以挤压夹爪，
    # 并让它自行弹回张开位置。
    gripper_open_pos: float = 50.0
