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
训练用的样本加权抽象。

本模块为样本加权策略（例如 RA-BC）提供一个抽象基类，
可在训练期间使用，而不会让训练脚本被
策略专属代码污染。

用法示例：
    # 训练配置中
    sample_weighting:
        type: rabc
        progress_path: hf://datasets/my-dataset/sarm_progress.parquet
        head_mode: sparse
        kappa: 0.01

    # 训练脚本中
    sample_weighter = make_sample_weighter(cfg.sample_weighting, policy, device, dataset_root=cfg.dataset.root, dataset_repo_id=cfg.dataset.repo_id)
    ...
    weights, stats = sample_weighter.compute_batch_weights(batch)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy


class SampleWeighter(ABC):
    """
    其实现类会计算逐样本权重，可用于在训练期间对
    损失加权。这可以支持如下技术：
    - RA-BC（奖励对齐的行为克隆，Reward-Aligned Behavior Cloning）
    - 重要性采样
    - 课程学习
    - 基于质量的过滤
    """

    @abstractmethod
    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        为一个训练批次计算逐样本权重。

        参数:
            batch: 训练批次字典，至少包含一个带有全局帧索引的
                   "index" 键。
        """

    @abstractmethod
    def get_stats(self) -> dict:
        """
        获取有关该加权策略的全局统计信息。
        """


@dataclass
class SampleWeightingConfig:
    """
    训练期间样本加权的配置。

    这是一个支持多种加权策略的通用配置。
    `type` 字段决定使用哪个实现，`extra_params`
    包含附加的、特定于类型的参数。

    属性:
        type: 加权策略类型（"rabc"、"uniform" 等）
        progress_path: 预计算进度值的路径（用于 RABC）
        head_mode: 进度计算使用哪个模型头（"sparse" 或 "dense"）
        kappa: 高质量样本的硬阈值（RABC 专属）
        epsilon: 用于数值稳定性的小常数
        extra_params: 传递给加权器的、特定于类型的附加参数
    """

    type: str = "rabc"
    progress_path: str | None = None
    head_mode: str = "sparse"
    kappa: float = 0.01
    epsilon: float = 1e-6
    # 附加的特定于类型的参数可以添加在这里，或通过 extra_params 传入
    extra_params: dict = field(default_factory=dict)


def make_sample_weighter(
    config: SampleWeightingConfig | None,
    policy: PreTrainedPolicy,
    device: torch.device,
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
) -> SampleWeighter | None:
    """
    根据配置创建 SampleWeighter 的工厂函数。

    这使策略专属的初始化逻辑不进入训练脚本。

    参数:
        config: 样本加权配置；为 None 则禁用加权。
        policy: 正在训练的策略（用于提取 chunk_size 等）
        device: 放置权重张量的设备。
        dataset_root: 数据集根目录的本地路径（用于自动探测 progress_path）。
        dataset_repo_id: HuggingFace 仓库 ID（用于自动探测 progress_path）。
    """
    if config is None:
        return None

    if config.type == "rabc":
        return _make_rabc_weighter(config, policy, device, dataset_root, dataset_repo_id)

    if config.type == "uniform":
        # 返回均匀权重的空操作加权器
        return UniformWeighter(device=device)

    raise ValueError(f"Unknown sample weighting type: '{config.type}'. Supported types: 'rabc', 'uniform'")


def _make_rabc_weighter(
    config: SampleWeightingConfig,
    policy: PreTrainedPolicy,
    device: torch.device,
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
) -> SampleWeighter:
    """创建带策略专属初始化的 RABC 加权器。

    参数:
        config: 样本加权配置。
        policy: 正在训练的策略（用于提取 chunk_size）。
        device: 放置权重张量的设备。
        dataset_root: 数据集根目录的本地路径（用于自动探测 progress_path）。
        dataset_repo_id: HuggingFace 仓库 ID（用于自动探测 progress_path）。
    """
    # 在此处导入，以避免循环导入，并将 RABC 代码保留在 SARM 模块中
    from lerobot.rewards.sarm.rabc import RABCWeights

    # 从策略配置中提取 chunk_size
    chunk_size = getattr(policy.config, "chunk_size", None)
    if chunk_size is None:
        raise ValueError(
            "RABC sample weighting requires a policy with 'chunk_size' in its config. "
            "This is typically set for action-chunking policies like ACT, Diffusion, PI0, etc."
        )

    # 确定 progress_path：使用显式配置，或从数据集自动探测
    progress_path = config.progress_path
    if progress_path is None:
        if dataset_root:
            progress_path = str(Path(dataset_root) / "sarm_progress.parquet")
        elif dataset_repo_id:
            progress_path = f"hf://datasets/{dataset_repo_id}/sarm_progress.parquet"
        else:
            raise ValueError(
                "RABC sample weighting requires 'progress_path' to be set, "
                "or dataset_root/dataset_repo_id for auto-detection. "
                "Generate progress values using: "
                "python -m lerobot.rewards.sarm.compute_rabc_weights --help"
            )

    return RABCWeights(
        progress_path=progress_path,
        chunk_size=chunk_size,
        head_mode=config.head_mode,
        kappa=config.kappa,
        epsilon=config.epsilon,
        device=device,
        **config.extra_params,
    )


class UniformWeighter(SampleWeighter):
    """
    返回均匀权重的空操作样本加权器。

    适合作为基线，或在你希望禁用加权而又不
    改变训练代码结构时使用。

    注意：
        批次大小通过在批次字典中查找张量值来确定。该方法先检查
        "action"、"index" 和 "observation.state" 等常见键，
        然后回退到扫描所有值。
    """

    def __init__(self, device: torch.device):
        self.device = device

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """返回均匀权重（全部为 1）。"""
        batch_size = self._determine_batch_size(batch)

        weights = torch.ones(batch_size, device=self.device)
        stats = {"mean_weight": 1.0, "type": "uniform"}
        return weights, stats

    def _determine_batch_size(self, batch: dict) -> int:
        """
        从批次字典中确定批次大小。

        先检查常见键，然后扫描所有值以查找张量。

        参数:
            batch: 训练批次字典。
        """
        if not batch:
            raise ValueError("Cannot determine batch size from empty batch")

        # 先检查常见键
        for key in ["action", "index", "observation.state"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key].shape[0]

        # 扫描所有值以查找任意张量
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return value.shape[0]

        # 最后手段：返回 1（用于处理非张量批次）
        return 1

    def get_stats(self) -> dict:
        """返回均匀加权的空统计信息。"""
        return {"type": "uniform"}
