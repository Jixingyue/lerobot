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

"""实时分块（Real-Time Chunking，RTC）的相对动作辅助函数。"""

from __future__ import annotations

import torch

from lerobot.processor import (
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
    to_relative_actions,
)


def reanchor_relative_rtc_prefix(
    prev_actions_absolute: torch.Tensor,
    current_state: torch.Tensor,
    relative_step: RelativeActionsProcessorStep,
    normalizer_step: NormalizerProcessorStep | None,
    policy_device: torch.device | str,
) -> torch.Tensor:
    """将上一块遗留的绝对动作转换为相对动作 RTC 策略所需的模型空间表示。

    使用相对动作时，RTC 前缀（上一动作块中尚未执行的尾部）是以绝对坐标存储的。
    在将其回传给策略之前，该辅助函数会以机器人当前关节状态为基准重新表达这些动作，
    并可选地对其进行归一化，以确保策略接收到尺度正确的输入。
    """
    state = current_state.detach().cpu()
    if state.dim() == 1:
        state = state.unsqueeze(0)

    action_cpu = prev_actions_absolute.detach().cpu()
    mask = relative_step._build_mask(action_cpu.shape[-1])
    relative_actions = to_relative_actions(action_cpu, state, mask)

    transition = create_transition(action=relative_actions)
    if normalizer_step is not None:
        transition = normalizer_step(transition)

    return transition[TransitionKey.ACTION].to(policy_device)
