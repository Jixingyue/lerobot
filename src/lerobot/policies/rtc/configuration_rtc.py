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

"""
实时分块（Real Time Chunking，RTC）与双向解码（Bidirectional Decoding，BID）配置类。

基于：
- Real Time Chunking: https://www.physicalintelligence.company/research/real_time_chunking
"""

from dataclasses import dataclass

from lerobot.configs import RTCAttentionSchedule


@dataclass
class RTCConfig:
    """实时分块（Real Time Chunking，RTC）推理配置。

    RTC 将动作块生成视为一个图像修复（inpainting）问题，通过前缀注意力（prefix attention）
    有策略地处理相邻动作块之间重叠的时间步，从而改进实时推理。
    """

    # 基础设施
    enabled: bool = True

    # ``guided`` 是最初的推理时 Jacobian 引导方式；``trained``
    # 会硬修复（hard-inpaint）一个前缀，需要使用兼容的训练时 RTC 检查点。
    mode: str = "guided"

    # RTC 核心设置
    # Todo：改为 exp
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10

    # 调试设置
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        """校验 RTC 配置参数。"""
        if self.mode not in {"guided", "trained"}:
            raise ValueError(f"mode must be 'guided' or 'trained', got {self.mode!r}")
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")
