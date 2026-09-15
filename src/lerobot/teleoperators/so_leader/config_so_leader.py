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
class SOLeaderConfig:
    """SO Leader 遥操作设备的基础配置类。"""

    # 用于连接机械臂的端口
    port: str

    # 角度是否使用度数
    use_degrees: bool = True

    # 电机的 `sync_read` 失败时的额外重试次数。Feetech 总线偶尔会返回损坏的
    # 状态包（"Incorrect status packet!"），尤其是在多个关节同时移动时，
    # 否则会导致遥操作循环中止。重试是立即进行的（不休眠），且仅在失败时发生，
    # 因此稳态下的读取开销不变。
    num_read_retries: int = 2


@TeleoperatorConfig.register_subclass("so101_leader")
@TeleoperatorConfig.register_subclass("so100_leader")
@dataclass
class SOLeaderTeleopConfig(TeleoperatorConfig, SOLeaderConfig):
    pass


SO100LeaderConfig = SOLeaderTeleopConfig
SO101LeaderConfig = SOLeaderTeleopConfig
