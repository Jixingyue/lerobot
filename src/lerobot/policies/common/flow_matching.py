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

"""各策略共享的流匹配（flow-matching）采样原语。

这里是 beta 分布时间步采样器和前向欧拉去噪循环（及其实时分块钩子）的规范版本；
源自 openpi 的策略（pi0、pi05、smolvla、eo1）过去都各自携带一份副本。
所有函数都是无状态的；采用它们不会影响检查点。
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def sample_beta(alpha: float, beta: float, bsize: int, device) -> Tensor:  # 参见 openpi（完全一致的副本）
    # Beta 采样使用的 _sample_dirichlet 未在 MPS 上实现，因此在 CPU 上采样
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def sample_noise(shape, device) -> Tensor:
    """标准正态分布的 float32 噪声，即流匹配的 x_1 样本。"""
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )


def sample_time_beta(bsize: int, device, *, alpha: float, beta: float, scale: float, offset: float) -> Tensor:
    """Beta 分布的流匹配时间步：``Beta(alpha, beta) * scale + offset``（openpi 约定）。"""
    time_beta = sample_beta(alpha, beta, bsize, device)
    time = time_beta * scale + offset
    return time.to(dtype=torch.float32, device=device)


def euler_integrate(
    denoise_fn: Callable[[Tensor, Tensor], Tensor],
    noise: Tensor,
    num_steps: int,
    *,
    rtc_processor: "RTCProcessor | None" = None,
    rtc_enabled: bool = False,
    inference_delay: int | None = None,
    prev_chunk_left_over: Tensor | None = None,
    execution_horizon: int | None = None,
    hard_prefix: Tensor | None = None,
    hard_prefix_mask: Tensor | None = None,
) -> Tensor:
    """对速度场做前向欧拉积分，从 t=1（噪声）到 t=0（动作）。

    这就是 openpi 的采样循环：``dt = -1/num_steps``，``time = 1.0 + step*dt``，
    ``x_t <- x_t + dt * v_t``，并可选地用实时分块（RTC）引导钩子包裹速度计算，
    以及在每一步之后进行调试跟踪。

    Args:
        denoise_fn: 根据 ``(x_t, time_tensor)`` 计算速度 ``v_t``，其中
            ``time_tensor`` 是形状为 ``(batch_size,)`` 的 float32 张量。返回的
            速度必须与 ``x_t`` 具有相同的形状和 dtype。
        noise: 形状为 ``(batch_size, ...)`` 的初始样本 ``x_1``。
        num_steps: 欧拉步数。
        rtc_processor: 可选的 RTC 处理器。只要设置了该处理器且启用了调试，
            调试跟踪就会触发，即使 RTC 引导本身被禁用也是如此（这与
            过去各策略自带的循环行为一致）。
        rtc_enabled: 是否将速度计算路由到
            ``rtc_processor.denoise_step``（需要 ``rtc_processor``）。
        inference_delay: RTC 引导参数，原样转发。
        prev_chunk_left_over: RTC 引导参数，原样转发。
        execution_horizon: RTC 引导参数，原样转发。
        hard_prefix: 可选的干净动作前缀，在整个去噪过程中被钳制。
        hard_prefix_mask: 布尔掩码，选出从 ``hard_prefix`` 钳制得到的值。
    """
    bsize = noise.shape[0]
    device = noise.device

    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

        if hard_prefix is not None:
            if hard_prefix_mask is None:
                raise ValueError("hard_prefix_mask is required when hard_prefix is provided")
            x_t = torch.where(hard_prefix_mask, hard_prefix, x_t)
            time_tensor = time_tensor[:, None].expand(bsize, x_t.shape[1]).clone()
            time_tensor[hard_prefix_mask[..., 0]] = 0.0

        def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
            return denoise_fn(input_x_t, current_timestep)

        if rtc_enabled:
            v_t = rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=time,
                original_denoise_step_partial=denoise_step_partial_call,
                execution_horizon=execution_horizon,
            )
        else:
            v_t = denoise_step_partial_call(x_t)

        x_t = x_t + dt * v_t

        if hard_prefix is not None:
            x_t = torch.where(hard_prefix_mask, hard_prefix, x_t)

        if rtc_processor is not None and rtc_processor.is_debug_enabled():
            rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

    return x_t
