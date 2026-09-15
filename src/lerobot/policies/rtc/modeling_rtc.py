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
LeRobot 的实时分块（Real-Time Chunking，RTC）实现。

基于 Physical Intelligence 的 Kinetix 实现：
https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py#L214
"""

import logging
import math

import torch
from torch import Tensor

from lerobot.configs import RTCAttentionSchedule

from .configuration_rtc import RTCConfig
from .debug_tracker import Tracker

logger = logging.getLogger(__name__)


class RTCProcessor:
    """面向动作分块策略的实时分块处理器。

    该类实现了 RTC 相关技术，包括速度计算、前缀注意力以及自适应动作块处理。
    """

    def __init__(self, rtc_config: RTCConfig, *, trained_mode_supported: bool = False):
        if rtc_config.enabled and rtc_config.mode == "trained" and not trained_mode_supported:
            raise ValueError(
                "RTC mode='trained' requires a PI05-compatible checkpoint trained with "
                "rtc_training_max_delay > 0."
            )
        self.rtc_config = rtc_config

        self.tracker = None

        if rtc_config.debug:
            self.tracker = Tracker(
                enabled=rtc_config.debug,
                maxlen=rtc_config.debug_maxlen,
            )

    # ====================== Tracker 代理方法 ======================
    def track(
        self,
        time: float | Tensor,
        x_t: Tensor | None = None,
        v_t: Tensor | None = None,
        x1_t: Tensor | None = None,
        correction: Tensor | None = None,
        err: Tensor | None = None,
        weights: Tensor | None = None,
        guidance_weight: float | Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        **metadata,
    ) -> None:
        """跟踪调试信息的代理方法。

        若 tracker 为 None 或被禁用，则此方法什么都不做；
        否则将调用转发给 tracker.track()。
        """
        if self.tracker is not None:
            self.tracker.track(
                time=time,
                x_t=x_t,
                v_t=v_t,
                x1_t=x1_t,
                correction=correction,
                err=err,
                weights=weights,
                guidance_weight=guidance_weight,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                **metadata,
            )

    def get_all_debug_steps(self) -> list:
        """从 tracker 获取所有调试步骤。

        若 tracker 被禁用或为 None，则返回空列表。
        """
        if self.tracker is not None:
            return self.tracker.get_all_steps()
        return []

    def is_debug_enabled(self) -> bool:
        """检查是否启用了调试跟踪。

        当 tracker 存在且处于启用状态时返回 True。
        """
        return self.tracker is not None and self.tracker.enabled

    def reset_tracker(self) -> None:
        """重置 tracker，清除所有已记录的步骤。

        若 tracker 为 None 则什么都不做。
        """
        if self.tracker is not None:
            self.tracker.reset()

    # ====================== Tracker 代理方法结束 ======================

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ) -> Tensor:
        """在已有去噪器外层包装的 RTC 引导方法。

        该方法包装一个原始的去噪可调用对象，该对象只接收 ``x_t`` 并返回基础的
        去噪速度 ``v_t``。随后利用上一动作块遗留的前缀，应用实时分块（RTC）前缀引导。

        Args:
            x_t (Tensor): 待去噪的当前潜变量/状态。形状为 ``(B, T, A)`` 或 ``(T, A)``。
            prev_chunk_left_over (Tensor | None): 上一动作块中未执行的前缀。
                形状为 ``(B, T_prev, A)`` 或 ``(T_prev, A)``。若为 ``None``，则不施加引导，
                方法直接返回原始去噪器给出的 ``v_t``。
            inference_delay (int): 前缀中用于引导的时间步数量。
            time (float | Tensor): 取值 [0, 1] 的标量，表示归一化时间。必须能够
                与 ``x_t`` 广播。
            original_denoise_step_partial (Callable[[Tensor], Tensor]): 仅给定 ``x_t``
                即可计算基础去噪速度的可调用对象。
            execution_horizon (int | None): 用于构建前缀权重的时域。若为
                ``None``，则默认取 ``self.rtc_config.execution_horizon``。

        Returns:
            Tensor: 引导后的速度，与 ``v_t`` 形状相同。

        Notes:
            - 若输入为二维，会临时添加一个批次维度，并在最后移除。
            - 若 ``prev_chunk_left_over`` 短于当前动作块长度 ``T``，会在右侧补零至与 ``T`` 等长。
            - 前缀权重通过 ``get_prefix_weights(inference_delay, execution_horizon, T)``
              构建，并广播至 ``(B, T, A)``。
            - 引导校正量通过 autograd 计算，使用 ``x1_t = x_t + time * v_t`` 以及
              ``error = (prev_chunk_left_over - x1_t) * weights``。
            - 最终的引导权重会被配置中的 ``max_guidance_weight`` 截断。

        Reference:
            https://www.physicalintelligence.company/download/real_time_chunking.pdf
        """

        # 在原始实现中，时间从 0 变化到 1；
        # 而在我们的实现中，时间从 1 变化到 0，
        # 因此需要对时间取反
        tau = 1 - time

        if prev_chunk_left_over is None:
            # 第一步，不进行引导，直接返回 v_t
            v_t = original_denoise_step_partial(x_t)
            return v_t

        x_t = x_t.clone().detach()

        squeezed = False
        if len(x_t.shape) < 3:
            # 添加批次维度
            x_t = x_t.unsqueeze(0)
            squeezed = True

        if len(prev_chunk_left_over.shape) < 3:
            # 添加批次维度
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon

        # 如果上一个动作块太短，就没有必要使用很长的执行时域，
        # 因为没有内容可供合并
        if execution_horizon > prev_chunk_left_over.shape[1]:
            execution_horizon = prev_chunk_left_over.shape[1]

        batch_size = x_t.shape[0]
        action_chunk_size = x_t.shape[1]
        action_dim = x_t.shape[2]

        if prev_chunk_left_over.shape[1] < action_chunk_size or prev_chunk_left_over.shape[2] < action_dim:
            padded = torch.zeros(batch_size, action_chunk_size, action_dim).to(x_t.device)
            padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
            prev_chunk_left_over = padded

        assert prev_chunk_left_over.shape == x_t.shape, (
            "The padded previous chunk must be the same size as the input tensor"
        )

        weights = (
            self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size)
            .to(x_t.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t)
            x_t.requires_grad_(True)

            x1_t = x_t - time * v_t  # noqa: N806
            err = (prev_chunk_left_over - x1_t) * weights
            grad_outputs = err.clone().detach()
            correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

        max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight)
        tau_tensor = torch.as_tensor(tau)
        squared_one_minus_tau = (1 - tau_tensor) ** 2
        inv_r2 = (squared_one_minus_tau + tau_tensor**2) / (squared_one_minus_tau)
        c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        result = v_t - guidance_weight * correction

        # 如果批次维度是此前添加的，则移除
        if squeezed:
            result = result.squeeze(0)
            correction = correction.squeeze(0)
            x1_t = x1_t.squeeze(0)
            err = err.squeeze(0)

        self.track(
            time=time,
            x1_t=x1_t,
            correction=correction,
            err=err,
            weights=weights,
            guidance_weight=guidance_weight,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )

        return result

    def get_prefix_weights(self, start, end, total):
        start = min(start, end)

        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR:
            lin_weights = self._linweights(start, end, total)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.EXP:
            lin_weights = self._linweights(start, end, total)
            lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)

        return weights

    def _linweights(self, start, end, total):
        skip_steps_at_end = max(total - end, 0)

        linspace_steps = total - skip_steps_at_end - start

        if end <= start or linspace_steps <= 0:
            return torch.tensor([])

        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights, total, end):
        zeros_len = total - end

        if zeros_len <= 0:
            return weights

        zeros = torch.zeros(zeros_len)
        return torch.cat([weights, zeros])

    def _add_leading_ones(self, weights, start, total):
        ones_len = min(start, total)

        if ones_len <= 0:
            return weights

        ones = torch.ones(ones_len)
        return torch.cat([ones, weights])
