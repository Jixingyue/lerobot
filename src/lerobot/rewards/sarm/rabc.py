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
RA-BC（Reward-Aligned Behavior Cloning，奖励对齐的行为克隆）样本加权实现。

本模块为 RA-BC 训练实现了 SampleWeighter 协议，
根据 SARM 奖励模型所度量的任务进度对训练样本进行加权。

权重基于进度差值进行计算：
    delta = progress[t + chunk_size] - progress[t]

高质量样本（正进度）获得更高的权重，而
负进度样本（出现倒退）获得零权重。

参见：https://arxiv.org/abs/2509.25358 获取 SARM 论文。
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from lerobot.utils.import_utils import _pandas_available
from lerobot.utils.sample_weighting import SampleWeighter

if TYPE_CHECKING or _pandas_available:
    import pandas as pd
else:
    pd = None  # type: ignore[assignment]


def resolve_hf_path(path: str | Path) -> Path:
    """将可能是 HuggingFace URL（hf://datasets/...）的路径解析为本地路径。"""
    path_str = str(path)
    if path_str.startswith("hf://datasets/"):
        parts = path_str.replace("hf://datasets/", "").split("/")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))
    return Path(path)


class RABCWeights(SampleWeighter):
    """
    加载预计算的 SARM 进度值，并在训练期间计算 RA-BC 权重。

    本类实现了 SampleWeighter 抽象基类，以便与 lerobot 中通用的
    样本加权基础设施配合使用。

    进度值从 parquet 文件加载（由 compute_rabc_weights.py 生成）。
    训练期间计算：
        - progress_delta = progress[t + chunk_size] - progress[t]
        - 基于该差值的 rabc_weight（论文公式 8-9）

    Args:
        progress_path: 包含预计算进度值的 parquet 文件路径。
                      支持 HuggingFace URL（hf://datasets/...）。
        chunk_size: 用于计算进度差值的向前帧数。
        head_mode: 使用哪个 SARM 头（"sparse" 或 "dense"）。
        kappa: 高质量样本的硬阈值（默认：0.01）。
        epsilon: 用于数值稳定性的小常数（默认：1e-6）。
        fallback_weight: 对没有有效差值的帧所使用的权重（默认：1.0）。
        device: 返回张量所在的设备。
    """

    def __init__(
        self,
        progress_path: str | Path,
        chunk_size: int = 50,
        head_mode: str = "sparse",
        kappa: float = 0.01,
        epsilon: float = 1e-6,
        fallback_weight: float = 1.0,
        device: torch.device | None = None,
    ):
        self.progress_path = resolve_hf_path(progress_path)
        self.chunk_size = chunk_size
        self.head_mode = head_mode
        self.kappa = kappa
        self.epsilon = epsilon
        self.fallback_weight = fallback_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 确定进度列名称
        self.progress_column = f"progress_{head_mode}"

        # 加载进度值
        logging.info(f"Loading SARM progress values from {self.progress_path}")
        self.df = pd.read_parquet(self.progress_path)

        # 检查所请求的 head mode 列是否存在
        if self.progress_column not in self.df.columns:
            available = [c for c in self.df.columns if c.startswith("progress")]
            raise ValueError(
                f"Column '{self.progress_column}' not found. Available progress columns: {available}"
            )

        logging.info(f"Using progress column: {self.progress_column}")

        self.progress_lookup: dict[int, float] = {}
        self.episode_lookup: dict[int, int] = {}

        for _, row in self.df.iterrows():
            global_idx = int(row["index"])
            progress = row[self.progress_column]
            episode_idx = int(row["episode_index"])

            if not np.isnan(progress):
                self.progress_lookup[global_idx] = float(progress)
            self.episode_lookup[global_idx] = episode_idx

        # 构建用于差值计算的 episode 边界
        self.episode_boundaries: dict[int, dict[str, int]] = {}
        for episode_idx in self.df["episode_index"].unique():
            ep_df = self.df[self.df["episode_index"] == episode_idx]
            self.episode_boundaries[int(episode_idx)] = {
                "start": int(ep_df["index"].min()),
                "end": int(ep_df["index"].max()) + 1,
            }

        logging.info(f"Loaded {len(self.progress_lookup)} frame progress values")
        logging.info(f"Chunk size for delta computation: {chunk_size}")

        # 计算用于权重计算的全局统计量
        self._compute_global_stats()

    def _compute_global_stats(self) -> None:
        """计算进度差值的全局均值和标准差，用于权重计算。"""
        all_deltas = []

        for global_idx, progress in self.progress_lookup.items():
            episode_idx = self.episode_lookup.get(global_idx)
            if episode_idx is None:
                continue

            bounds = self.episode_boundaries.get(episode_idx)
            if bounds is None:
                continue

            future_idx = global_idx + self.chunk_size
            if future_idx >= bounds["end"]:
                # 接近 episode 末尾：使用最后一帧的进度
                future_idx = bounds["end"] - 1

            future_progress = self.progress_lookup.get(future_idx)
            if future_progress is not None:
                delta = future_progress - progress
                all_deltas.append(delta)

        if all_deltas:
            self.delta_mean = max(float(np.mean(all_deltas)), 0.0)
            self.delta_std = max(float(np.std(all_deltas)), self.epsilon)
            logging.info(f"Progress delta stats: mean={self.delta_mean:.4f}, std={self.delta_std:.4f}")
        else:
            self.delta_mean = 0.0
            self.delta_std = self.epsilon
            logging.warning("No valid progress deltas found, using default stats")

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        为一个批次计算 RA-BC 权重。

        对每个样本：
        1. 获取当前帧的进度
        2. 获取 frame + chunk_size 处的进度（在同一 episode 内）
        3. 计算 delta = future_progress - current_progress
        4. 使用论文公式 8-9 计算权重

        Args:
            batch: 包含 "index" 键（全局帧索引）的训练批次。

        Returns:
            元组，包含：
            - 权重张量 (batch_size,)，归一化后总和为 batch_size。
            - 用于日志记录的加权统计信息字典。
        """
        indices = batch.get("index")
        if indices is None:
            logging.warning("RA-BC: Batch missing 'index' key, using uniform weights")
            batch_size = self._get_batch_size(batch)
            stats = {"mean_weight": 1.0, "num_zero_weight": 0, "num_full_weight": batch_size}
            return torch.ones(batch_size, device=self.device), stats

        # 转换为整数列表
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()

        # 为每个样本计算差值和权重
        deltas = []
        for idx in indices:
            idx = int(idx)
            delta = self._compute_delta(idx)
            deltas.append(delta)

        deltas_array = np.array(deltas, dtype=np.float32)

        # 根据差值计算权重
        weights = self._compute_weights(deltas_array)

        # 在归一化之前计算统计量，用于日志记录
        raw_mean_weight = float(np.nanmean(weights))
        num_zero_weight = int(np.sum(weights == 0))
        num_full_weight = int(np.sum(weights == 1.0))
        batch_stats = {
            "mean_weight": raw_mean_weight,
            "num_zero_weight": num_zero_weight,
            "num_full_weight": num_full_weight,
        }

        weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)

        # 归一化使总和为 batch_size
        batch_size = len(weights_tensor)
        weight_sum = weights_tensor.sum() + self.epsilon
        weights_tensor = weights_tensor * batch_size / weight_sum

        return weights_tensor, batch_stats

    def _compute_delta(self, global_idx: int) -> float:
        """计算单帧的进度差值。"""
        current_progress = self.progress_lookup.get(global_idx)
        if current_progress is None:
            return np.nan

        episode_idx = self.episode_lookup.get(global_idx)
        if episode_idx is None:
            return np.nan

        bounds = self.episode_boundaries.get(episode_idx)
        if bounds is None:
            return np.nan

        future_idx = global_idx + self.chunk_size  # Δ = chunk_size
        if future_idx >= bounds["end"]:
            # 接近 episode 末尾：改用最后一帧的进度
            future_idx = bounds["end"] - 1

        future_progress = self.progress_lookup.get(future_idx)
        if future_progress is None:
            return np.nan

        return future_progress - current_progress

    def _compute_weights(self, deltas: np.ndarray) -> np.ndarray:
        """
        根据进度差值计算 RA-BC 权重。

        遵循论文公式 8-9：
        - 软权重：˜wi = clip((ri − (µ − 2σ)) / (4σ + ε), 0, 1)
        - 最终权重：wi = 1{ri > κ} + 1{0 ≤ ri ≤ κ}˜wi

        Returns:
            权重数组。
        """
        valid_mask = ~np.isnan(deltas)

        # 使用全局统计量计算软权重
        lower_bound = self.delta_mean - 2 * self.delta_std
        soft_weights = (deltas - lower_bound) / (4 * self.delta_std + self.epsilon)
        soft_weights = np.clip(soft_weights, 0.0, 1.0)

        # 应用论文公式 9
        weights = np.zeros_like(deltas, dtype=np.float32)

        # 高质量：ri > kappa → weight = 1
        high_quality_mask = deltas > self.kappa
        weights[high_quality_mask] = 1.0

        # 中等质量：0 <= ri <= kappa → weight = soft_weight
        moderate_mask = (deltas >= 0) & (deltas <= self.kappa)
        weights[moderate_mask] = soft_weights[moderate_mask]

        # 负进度：ri < 0 → weight = 0（已为 0）
        # 无效值（NaN）：使用回退权重
        weights[~valid_mask] = self.fallback_weight

        return weights

    def _get_batch_size(self, batch: dict) -> int:
        """从批次中确定批次大小。"""
        for key in ["action", "index"]:
            if key in batch:
                val = batch[key]
                if isinstance(val, (torch.Tensor, np.ndarray)):
                    return int(val.shape[0])
        return 1

    def get_stats(self) -> dict:
        """获取有关 RA-BC 加权的全局统计信息。"""
        return {
            "type": "rabc",
            "num_frames": len(self.progress_lookup),
            "chunk_size": self.chunk_size,
            "head_mode": self.head_mode,
            "delta_mean": self.delta_mean,
            "delta_std": self.delta_std,
            "kappa": self.kappa,
        }
