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

import random

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def find_stage_and_tau(
    current_frame: int,
    episode_length: int,
    subtask_names: list | None,
    subtask_start_frames: list | None,
    subtask_end_frames: list | None,
    global_subtask_names: list,
    temporal_proportions: dict,
    return_combined: bool = False,
) -> tuple[int, float] | float:
    """查找某一帧所处的 stage 及其在该 stage 内的进度（tau）。

    Args:
        current_frame: 相对于 episode 起始的帧索引
        episode_length: episode 的总帧数
        subtask_names: 该 episode 的子任务名称（single_stage 时为 None）
        subtask_start_frames: 子任务起始帧
        subtask_end_frames: 子任务结束帧
        global_subtask_names: 所有子任务名称的全局列表
        temporal_proportions: 时间比例字典
        return_combined: 若为 True，以 float 形式返回 stage+tau；否则返回 (stage_idx, tau) 元组

    Returns:
        若 return_combined 为 True 则返回浮点数 (stage.tau)，否则返回 (stage_idx, tau) 元组
    """
    stage_idx, tau = 0, 0.0
    num_stages = len(global_subtask_names)

    # 单 stage 模式：从 0 到 1 的线性进度
    if num_stages == 1:
        tau = min(1.0, max(0.0, current_frame / max(episode_length - 1, 1)))
    elif subtask_names is None:
        pass  # stage_idx=0, tau=0.0
    elif current_frame < subtask_start_frames[0]:
        pass  # 在第一个子任务之前：stage_idx=0, tau=0.0
    elif current_frame > subtask_end_frames[-1]:
        stage_idx, tau = num_stages - 1, 0.999  # 在最后一个子任务之后
    else:
        # 查找该帧属于哪个子任务
        found = False
        for name, start, end in zip(subtask_names, subtask_start_frames, subtask_end_frames, strict=True):
            if start <= current_frame <= end:
                stage_idx = global_subtask_names.index(name) if name in global_subtask_names else 0
                tau = compute_tau(current_frame, start, end)
                found = True
                break
        # 帧位于两个子任务之间——使用前一个子任务的结束状态
        if not found:
            for j in range(len(subtask_names) - 1):
                if subtask_end_frames[j] < current_frame < subtask_start_frames[j + 1]:
                    name = subtask_names[j]
                    stage_idx = global_subtask_names.index(name) if name in global_subtask_names else j
                    tau = 1.0
                    break

    if return_combined:
        # 截断以避免在末尾溢出
        if stage_idx >= num_stages - 1 and tau >= 1.0:
            return num_stages - 1 + 0.999
        return stage_idx + tau
    return stage_idx, tau


def compute_absolute_indices(
    frame_idx: int,
    ep_start: int,
    ep_end: int,
    n_obs_steps: int,
    frame_gap: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """为双向观测序列计算绝对帧索引，并对越界索引进行截断。

    以目标帧为中心的双向采样：
    - 之前：[-frame_gap * half_steps, ..., -frame_gap]（half_steps 帧）
    - 当前：[0]（1 帧）
    - 之后：[frame_gap, ..., frame_gap * half_steps]（half_steps 帧）
    - 总计：n_obs_steps + 1 帧

    越界帧会被截断（复制边界帧）。

    Args:
        frame_idx: 目标帧索引（序列的中心帧）
        ep_start: episode 起始索引
        ep_end: episode 结束索引（不含）
        n_obs_steps: 观测步数（对称采样时必须为偶数）
        frame_gap: 观测帧之间的间隔

    Returns:
        (indices, out_of_bounds_flags) 元组
    """
    half_steps = n_obs_steps // 2

    # 双向增量：过去 + 当前 + 未来
    past_deltas = [-frame_gap * i for i in range(half_steps, 0, -1)]
    future_deltas = [frame_gap * i for i in range(1, half_steps + 1)]
    delta_indices = past_deltas + [0] + future_deltas

    frames = []
    out_of_bounds = []

    for delta in delta_indices:
        target_idx = frame_idx + delta
        # 截断到 episode 边界（越界时复制边界帧）
        clamped_idx = max(ep_start, min(ep_end - 1, target_idx))
        frames.append(clamped_idx)
        # 如果发生了截断，则标记为越界
        out_of_bounds.append(1 if target_idx != clamped_idx else 0)

    return torch.tensor(frames), torch.tensor(out_of_bounds)


def apply_rewind_augmentation(
    frame_idx: int,
    ep_start: int,
    n_obs_steps: int,
    max_rewind_steps: int,
    frame_gap: int = 30,
    rewind_step: int | None = None,
) -> tuple[int, list[int]]:
    """
    生成用于时间增强的回退（rewind）帧索引。

    回退模拟在先前见过的帧中向后移动，
    从最早的观测帧之前开始（针对双向采样）。
    将倒序帧追加到观测序列之后。

    Args:
        frame_idx: 目标帧索引（双向观测窗口的中心）
        ep_start: episode 起始索引
        n_obs_steps: 观测步数
        max_rewind_steps: 最大回退步数
        frame_gap: 帧之间的间隔
        rewind_step: 若提供，则使用该确切的回退步数（用于确定性行为）。
                     若为 None，则随机采样。

    Returns:
        (rewind_step, rewind_indices) 元组
    """
    # 对于双向采样，最早的观测帧位于 frame_idx - half_steps * frame_gap
    half_steps = n_obs_steps // 2
    earliest_obs_frame = frame_idx - half_steps * frame_gap

    # 所需历史：最早观测帧之前的帧
    if earliest_obs_frame <= ep_start:
        return 0, []  # 观测窗口之前没有历史

    # 基于最早观测帧之前的可用历史计算最大有效回退步数
    available_history = earliest_obs_frame - ep_start
    max_valid_step = available_history // frame_gap
    max_rewind = min(max_rewind_steps, max(0, max_valid_step))

    if max_rewind <= 0:
        return 0, []

    # 若未提供回退步数则进行采样
    rewind_step = random.randint(1, max_rewind) if rewind_step is None else min(rewind_step, max_rewind)

    if rewind_step == 0:
        return 0, []

    # 从最早的观测帧向后生成回退索引
    # rewind_indices[0] 最接近观测窗口，rewind_indices[-1] 最靠后
    rewind_indices = []
    for i in range(1, rewind_step + 1):
        idx = earliest_obs_frame - i * frame_gap
        idx = max(ep_start, idx)  # 截断到 episode 起始
        rewind_indices.append(idx)

    return rewind_step, rewind_indices


def compute_tau(current_frame: int | float, subtask_start: int | float, subtask_end: int | float) -> float:
    """计算 τ_t = (t - s_k) / (e_k - s_k) ∈ [0, 1]。对于持续时间为零的子任务返回 1.0。"""
    duration = subtask_end - subtask_start
    if duration <= 0:
        return 1.0
    return float(np.clip((current_frame - subtask_start) / duration, 0.0, 1.0))


def pad_state_to_max_dim(state: torch.Tensor, max_state_dim: int) -> torch.Tensor:
    """用零将状态张量的最后一个维度填充到 max_state_dim。"""
    current_dim = state.shape[-1]
    if current_dim >= max_state_dim:
        return state[..., :max_state_dim]  # 如果更大则截断

    # 在右侧用零填充
    padding = (0, max_state_dim - current_dim)  # 最后一个维度的 (left, right)
    return F.pad(state, padding, mode="constant", value=0)


def temporal_proportions_to_breakpoints(
    temporal_proportions: dict[str, float] | list[float] | None,
    subtask_names: list[str] | None = None,
) -> list[float] | None:
    """将时间比例转换为用于归一化的累计断点。"""
    if temporal_proportions is None:
        return None

    if isinstance(temporal_proportions, dict):
        if subtask_names is not None:
            proportions = [temporal_proportions.get(name, 0.0) for name in subtask_names]
        else:
            proportions = list(temporal_proportions.values())
    else:
        proportions = list(temporal_proportions)

    total = sum(proportions)
    if total > 0 and abs(total - 1.0) > 1e-6:
        proportions = [p / total for p in proportions]

    breakpoints = [0.0]
    cumsum = 0.0
    for prop in proportions:
        cumsum += prop
        breakpoints.append(cumsum)
    breakpoints[-1] = 1.0

    return breakpoints


def normalize_stage_tau(
    x: float | torch.Tensor,
    num_stages: int | None = None,
    breakpoints: list[float] | None = None,
    temporal_proportions: dict[str, float] | list[float] | None = None,
    subtask_names: list[str] | None = None,
) -> float | torch.Tensor:
    """
    使用自定义断点将 stage+tau 奖励归一化到 [0, 1]。

    将 stage 索引 + stage 内的 tau 映射为归一化进度 [0, 1]。
    断点的设计旨在根据各 stage 在任务中的重要性（使用时间比例）
    为其赋予合适的权重。

    优先级：breakpoints > temporal_proportions > 线性回退

    Args:
        x: 原始奖励值（stage 索引 + tau），其中 stage ∈ [0, num_stages-1]，tau ∈ [0, 1)
        num_stages: stage 数量（未提供 breakpoints/proportions 时必需）
        breakpoints: 可选的自定义断点列表，长度为 num_stages + 1。
        temporal_proportions: 可选的时间比例字典/列表，用于计算断点。
        subtask_names: 可选的有序子任务名称列表（用于字典形式的比例）

    Returns:
        归一化后的进度值 ∈ [0, 1]
    """
    if breakpoints is not None:
        num_stages = len(breakpoints) - 1
    elif temporal_proportions is not None:
        breakpoints = temporal_proportions_to_breakpoints(temporal_proportions, subtask_names)
        num_stages = len(breakpoints) - 1
    elif num_stages is not None:
        breakpoints = [i / num_stages for i in range(num_stages + 1)]
    else:
        raise ValueError("Either num_stages, breakpoints, or temporal_proportions must be provided")

    if isinstance(x, torch.Tensor):
        result = torch.zeros_like(x)
        for i in range(num_stages):
            mask = (x >= i) & (x < i + 1)
            tau_in_stage = x - i
            result[mask] = breakpoints[i] + tau_in_stage[mask] * (breakpoints[i + 1] - breakpoints[i])
        result[x >= num_stages] = 1.0
        return result.clamp(0.0, 1.0)
    else:
        if x < 0:
            return 0.0
        if x >= num_stages:
            return 1.0
        stage = int(x)
        tau = x - stage
        return breakpoints[stage] + tau * (breakpoints[stage + 1] - breakpoints[stage])
