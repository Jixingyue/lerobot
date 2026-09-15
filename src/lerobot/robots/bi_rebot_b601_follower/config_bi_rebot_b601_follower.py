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
from ..rebot_b601_follower import RebotB601FollowerConfig


@RobotConfig.register_subclass("bi_rebot_b601_follower")
@dataclass
class BiRebotB601FollowerConfig(RobotConfig):
    """双臂 reBot B601-DM follower 机器人的配置类。"""

    left_arm_config: RebotB601FollowerConfig
    right_arm_config: RebotB601FollowerConfig

    # 不归属于特定一侧的顶层相机。其键在观测中保持原样
    # （不加 `left_`/`right_` 前缀）。每条手臂上的相机（在
    # `{left,right}_arm_config.cameras` 中声明）会加前缀。
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
