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
"""EarthRover Mini Plus 机器人的配置。"""

from dataclasses import dataclass

from ..config import RobotConfig


@RobotConfig.register_subclass("earthrover_mini_plus")
@dataclass
class EarthRoverMiniPlusConfig(RobotConfig):
    """使用 Frodobots SDK 的 EarthRover Mini Plus 机器人配置。

    该机器人通过 Frodobots SDK HTTP API 进行云端控制。
    相机帧直接通过 SDK HTTP 端点访问。

    Attributes:
        sdk_url: Frodobots SDK 服务器的 URL（默认：http://localhost:8000）
    """

    sdk_url: str = "http://localhost:8000"
