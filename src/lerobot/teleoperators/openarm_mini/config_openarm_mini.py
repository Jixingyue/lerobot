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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@dataclass
class OpenArmMiniConfigBase:
    """OpenArm Mini 遥操作设备（Feetech STS3215，7 自由度 + 夹爪）的基础配置。"""

    # Feetech 总线的串口（例如 "/dev/ttyUSB0"）。
    port: str

    # 手臂侧别："left" 或 "right"。控制读取时应用的逐关节方向翻转。
    # 如果为 `None`，则不应用翻转。
    side: str | None = None

    use_degrees: bool = True


@TeleoperatorConfig.register_subclass("openarm_mini")
@dataclass
class OpenArmMiniConfig(TeleoperatorConfig, OpenArmMiniConfigBase):
    pass
