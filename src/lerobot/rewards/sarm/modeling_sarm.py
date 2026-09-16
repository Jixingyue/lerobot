# Copyright 2025 Qianzhong Chen, Justin Yu, Mac Schwager, Pieter Abbeel, Yide Shentu, Philipp Wu
# and The HuggingFace Inc. team. All rights reserved.
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
SARM: 面向长时域机器人操作的阶段感知奖励建模（Stage-Aware Reward Modeling）。

论文：https://arxiv.org/abs/2509.25358

- StageTransformer：预测阶段分类（sparse/dense）
- SubtaskTransformer：在给定阶段的条件下，预测阶段内的 progress（tau）
"""

import json
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.utils.constants import OBS_STR

from ..pretrained import PreTrainedRewardModel
from .configuration_sarm import SARMConfig
from .sarm_utils import (
    normalize_stage_tau,
    pad_state_to_max_dim,
)


class StageTransformer(nn.Module):
    """
    用于 SARM 的阶段分类 Transformer。

    预测当前帧属于哪个阶段/子任务。
    同时支持稀疏（高层级）和稠密（细粒度）两种标注方案。

    输入流：[vis_proj, lang_proj, state_proj] 拼接 -> (B, N+2, T, D)
    输出：阶段 logits (B, T, num_classes)
    """

    def __init__(
        self,
        d_model: int = 512,
        vis_emb_dim: int = 512,
        text_emb_dim: int = 512,
        state_dim: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        num_cameras: int = 1,
        num_classes_sparse: int = 4,
        num_classes_dense: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras

        # 投影层
        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        # 编码器
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

        # 第一帧视觉 token 上的位置偏置
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        # 共享融合 MLP
        # 融合 (num_cameras + 2) 个流：摄像头 + 语言 + 状态
        fused_in = d_model * (num_cameras + 2)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, d_model),
            nn.ReLU(),
        )

        # 各方案对应的 head
        self.heads = nn.ModuleDict(
            {
                "sparse": nn.Linear(d_model, num_classes_sparse),
                "dense": nn.Linear(d_model, num_classes_dense),
            }
        )

    def _prep_lang(self, lang_emb: torch.Tensor, B: int, T: int, D: int) -> torch.Tensor:  # noqa: N803
        """
        准备语言嵌入以供融合。

        接受以下形状的 lang_emb：
          - (B, text_emb_dim) -> 在时间维度上广播
          - (B, T, text_emb_dim) -> 每个时间步一个（稠密标注模式）

        返回：(B, 1, T, D)
        """
        if lang_emb.dim() == 3:
            # (B, T, E) -> (B, T, D) -> (B, 1, T, D)
            lang_proj = self.lang_proj(lang_emb).unsqueeze(1)
        else:
            # (B, E) -> (B, 1, 1, D) -> 扩展为 (B, 1, T, D)
            lang_proj = self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(B, 1, T, D)
        return lang_proj

    def forward(
        self,
        img_seq: torch.Tensor,  # (B, N, T, vis_emb_dim)
        lang_emb: torch.Tensor,  # (B, E) 或 (B, T, E)
        state: torch.Tensor,  # (B, T, state_dim)
        lengths: torch.Tensor,  # (B,) - 有效序列长度
        scheme: str = "sparse",  # "sparse" 或 "dense"
    ) -> torch.Tensor:
        """
        阶段分类的前向传播。

        Args:
            img_seq: 图像嵌入 (B, N, T, vis_emb_dim)，其中 N=num_cameras
            lang_emb: 语言嵌入 (B, E) 或 (B, T, E)（稠密模式下）
            state: 状态特征 (B, T, state_dim)
            lengths: 有效序列长度 (B,)，用于掩码
            scheme: "sparse" 或 "dense"，用于选择 head

        Returns:
            阶段 logits (B, T, num_classes)
        """
        assert scheme in self.heads, f"Unknown scheme '{scheme}'. Use one of {list(self.heads.keys())}."

        B, N, T, _ = img_seq.shape  # noqa: N806
        D = self.d_model  # noqa: N806
        device = img_seq.device

        # 对输入进行投影
        vis_proj = self.visual_proj(img_seq)  # (B, N, T, D)
        state_proj = self.state_proj(state).unsqueeze(1)  # (B, 1, T, D)
        lang_proj = self._prep_lang(lang_emb, B, T, D)  # (B, 1, T, D)

        # 拼接各流
        # 摄像头 + 语言 + 状态 -> (B, N+2, T, D)
        x = torch.cat([vis_proj, lang_proj, state_proj], dim=1)

        # 向第一帧视觉 token 添加位置偏置
        x[:, :N, 0, :] = x[:, :N, 0, :] + self.first_pos

        # 展平为 token 以供 Transformer 处理
        x_tokens = x.view(B, (N + 2) * T, D)
        L = x_tokens.size(1)  # noqa: N806

        # 创建填充掩码
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)  # (B, T)
        mask = base_mask.unsqueeze(1).expand(B, N + 2, T).reshape(B, (N + 2) * T)

        # 创建因果掩码
        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

        # 编码
        h = self.transformer(x_tokens, mask=causal_mask, src_key_padding_mask=mask, is_causal=True)

        # 重塑并融合
        h = h.view(B, N + 2, T, D).permute(0, 2, 1, 3).reshape(B, T, (N + 2) * D)
        fused = self.fusion_backbone(h)  # (B, T, D)

        # 各方案对应的 logits
        logits = self.heads[scheme](fused)  # (B, T, num_classes)
        return logits


class SubtaskTransformer(nn.Module):
    """
    用于 SARM 的子任务 progress 回归 Transformer。

    在给定阶段先验的条件下，预测阶段内的归一化 progress（tau）。
    阶段先验是由 StageTransformer 预测生成的独热编码。

    输入流：[vis_proj, lang_proj, state_proj, stage_emb] -> (B, N+3, T, D)
    输出：tau 预测 (B, T)，取值范围 [0, 1]
    """

    def __init__(
        self,
        d_model: int = 512,
        vis_emb_dim: int = 512,
        text_emb_dim: int = 512,
        state_dim: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        num_cameras: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras

        # 投影层
        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        # 编码器
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, n_layers)

        # 第一帧视觉 token 上的可学习偏置
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        # 共享融合骨干网络
        # 融合 (num_cameras + 3) 个流：摄像头 + 语言 + 状态 + stage_emb
        fused_in = d_model * (num_cameras + 3)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, d_model),
            nn.ReLU(),
        )

        # 各方案对应的回归 head
        self.heads = nn.ModuleDict(
            {
                "sparse": nn.Linear(d_model, 1),
                "dense": nn.Linear(d_model, 1),
            }
        )

    def _prep_lang(self, lang_emb: torch.Tensor, B: int, T: int, D: int) -> torch.Tensor:  # noqa: N803
        """
        准备语言嵌入以供融合。
        """
        if lang_emb.dim() == 3:
            # (B, T, E) -> (B, T, D) -> (B, 1, T, D)
            return self.lang_proj(lang_emb).unsqueeze(1)
        else:
            # (B, E) -> (B, 1, 1, D) -> 扩展为 (B, 1, T, D)
            return self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(B, 1, T, D)

    def _stage_to_dmodel(self, stage_prior: torch.Tensor) -> torch.Tensor:
        """
        通过对 one-hot 阶段进行填充/截断，确定性地投影到 d_model。

        Args:
            stage_prior: one-hot 阶段嵌入 (B, 1, T, C)

        Returns:
            投影后的阶段嵌入 (B, 1, T, d_model)
        """
        B, one, T, C = stage_prior.shape  # noqa: N806
        D = self.d_model  # noqa: N806
        if D == C:
            return stage_prior
        elif D > C:
            pad = torch.zeros(B, one, T, D - C, device=stage_prior.device, dtype=stage_prior.dtype)
            return torch.cat([stage_prior, pad], dim=-1)
        else:
            return stage_prior[..., :D]

    def forward(
        self,
        img_seq: torch.Tensor,  # (B, N, T, vis_emb_dim)
        lang_emb: torch.Tensor,  # (B, E) 或 (B, T, E)
        state: torch.Tensor,  # (B, T, state_dim)
        lengths: torch.Tensor,  # (B,) - 有效序列长度
        stage_prior: torch.Tensor,  # (B, 1, T, C) 来自 gen_stage_emb 的 one-hot
        scheme: str = "sparse",  # "sparse" 或 "dense"
    ) -> torch.Tensor:
        """
        子任务进度回归的前向传播。

        Args:
            img_seq: 图像嵌入 (B, N, T, vis_emb_dim)
            lang_emb: 语言嵌入 (B, E) 或 (B, T, E)
            state: 状态特征 (B, T, state_dim)
            lengths: 用于掩码的有效序列长度 (B,)
            stage_prior: one-hot 阶段先验 (B, 1, T, num_classes)
            scheme: 用于选择回归头的 "sparse" 或 "dense"

        Returns:
            通过 sigmoid 得到的 tau 预测 (B, T)，取值于 [0, 1]
        """
        assert scheme in self.heads, f"Unknown scheme '{scheme}'. Use one of {list(self.heads.keys())}."

        B, N, T, _ = img_seq.shape  # noqa: N806
        D = self.d_model  # noqa: N806
        device = img_seq.device

        # 投影各输入
        vis_proj = self.visual_proj(img_seq)  # (B, N, T, D)
        state_proj = self.state_proj(state).unsqueeze(1)  # (B, 1, T, D)
        lang_proj = self._prep_lang(lang_emb, B, T, D)  # (B, 1, T, D)
        stage_emb = self._stage_to_dmodel(stage_prior)  # (B, 1, T, D)

        # 拼接所有流
        # 摄像头 + 语言 + 状态 + stage_emb -> (B, N+3, T, D)
        x = torch.cat([vis_proj, lang_proj, state_proj, stage_emb], dim=1)

        # 为第一个视觉帧加上位置偏置
        x[:, :N, 0, :] = x[:, :N, 0, :] + self.first_pos

        # 展平为 token
        x_tokens = x.view(B, (N + 3) * T, D)
        L = x_tokens.size(1)  # noqa: N806

        # 创建 padding 掩码
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)
        mask = base_mask.unsqueeze(1).expand(B, N + 3, T).reshape(B, (N + 3) * T)

        # 创建因果掩码
        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

        # 编码
        h = self.transformer(x_tokens, mask=causal_mask, src_key_padding_mask=mask, is_causal=True)

        # 重塑形状并融合
        h = h.view(B, N + 3, T, D)
        h_flat = h.permute(0, 2, 1, 3).reshape(B, T, (N + 3) * D)
        fused = self.fusion_backbone(h_flat)  # (B, T, D)

        # 对应 scheme 的回归头 -> sigmoid
        r = torch.sigmoid(self.heads[scheme](fused)).squeeze(-1)  # (B, T)
        return r


def gen_stage_emb(num_classes: int, targets: torch.Tensor) -> torch.Tensor:
    """
    根据 targets 生成 one-hot 阶段嵌入。

    Args:
        num_classes: 阶段类别数
        targets: 目标值 (B, T)，其整数部分为阶段索引

    Returns:
        one-hot 阶段嵌入 (B, 1, T, num_classes)
    """
    # 浮点 targets 的整数部分 -> [0, C-1]
    idx = targets.long().clamp(min=0, max=num_classes - 1)  # (B, T)
    C = num_classes  # noqa: N806
    # 通过单位矩阵查表得到 one-hot
    stage_onehot = torch.eye(C, device=targets.device)[idx]  # (B, T, C)
    stage_onehot = stage_onehot.unsqueeze(1)  # (B, 1, T, C)
    return stage_onehot


class SARMRewardModel(PreTrainedRewardModel):
    """
    用于阶段感知任务完成奖励的 SARM 奖励模型。

    使用两个独立的 transformer 模型：
    - StageTransformer：分类当前处于哪个阶段/子任务
    - SubtaskTransformer：预测阶段内进度（tau）

    训练采用 75%/25% 的 GT/预测阶段条件（teacher forcing）。
    """

    name = "sarm"
    config_class = SARMConfig

    def __init__(self, config: SARMConfig, dataset_stats: dict | None = None, dataset_meta=None):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats
        self.device = torch.device(
            config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 根据 annotation_mode 加载时间占比
        if config.annotation_mode == "single_stage":
            logging.info(f"Using single_stage mode: sparse_subtask_names={config.sparse_subtask_names}")
        elif dataset_meta is not None:
            self._load_temporal_proportions(dataset_meta)

        # 创建两个独立的模型
        self.stage_model = StageTransformer(
            d_model=config.hidden_dim,
            vis_emb_dim=config.image_dim,
            text_emb_dim=config.text_dim,
            state_dim=config.max_state_dim,
            n_layers=config.num_layers,
            n_heads=config.num_heads,
            dropout=config.dropout,
            num_cameras=1,  # 暂时使用单相机
            num_classes_sparse=config.num_sparse_stages,
            num_classes_dense=config.num_dense_stages or config.num_sparse_stages,
        )

        self.subtask_model = SubtaskTransformer(
            d_model=config.hidden_dim,
            vis_emb_dim=config.image_dim,
            text_emb_dim=config.text_dim,
            state_dim=config.max_state_dim,
            n_layers=config.num_layers,
            n_heads=config.num_heads,
            dropout=config.dropout,
            num_cameras=1,
        )

        self.stage_model.to(self.device)
        self.subtask_model.to(self.device)

        # 用于 teacher forcing 的 GT/预测阶段比例
        self.gt_stage_ratio = 0.75

        if config.uses_dual_heads:
            logging.info(
                f"SARM initialized with dual heads: {config.num_sparse_stages} sparse stages, "
                f"{config.num_dense_stages} dense stages"
            )
        else:
            logging.info(f"SARM initialized with sparse head only: {config.num_sparse_stages} stages")

        logging.info(f"SARM initialized on {self.device}")

    def _load_proportions_from_json(self, path, annotation_type: str) -> tuple[list[str], list[float]]:
        """从 JSON 文件加载时间占比（保留顺序）。"""
        if not path.exists():
            raise ValueError(
                f"{annotation_type.capitalize()} temporal proportions not found at {path}. "
                f"Run the subtask annotation tool with --{annotation_type}-subtasks to generate annotations."
            )
        with open(path) as f:
            proportions_dict = json.load(f)
        names = list(proportions_dict.keys())
        logging.info(f"Loaded {len(names)} {annotation_type} subtasks: {names}")
        logging.info(f"{annotation_type.capitalize()} temporal proportions: {proportions_dict}")
        return names, [proportions_dict[name] for name in names]

    def _load_temporal_proportions(self, dataset_meta) -> None:
        """根据 annotation_mode 加载时间占比。"""
        meta_path = dataset_meta.root / "meta"

        if self.config.annotation_mode == "dual":
            names, props = self._load_proportions_from_json(
                meta_path / "temporal_proportions_sparse.json", "sparse"
            )
            (
                self.config.num_sparse_stages,
                self.config.sparse_subtask_names,
                self.config.sparse_temporal_proportions,
            ) = len(names), names, props

        if self.config.annotation_mode in ["dense_only", "dual"]:
            names, props = self._load_proportions_from_json(
                meta_path / "temporal_proportions_dense.json", "dense"
            )
            (
                self.config.num_dense_stages,
                self.config.dense_subtask_names,
                self.config.dense_temporal_proportions,
            ) = len(names), names, props
            if self.config.annotation_mode == "dense_only":
                logging.info(f"Using auto-generated sparse 'task' stage: {self.config.sparse_subtask_names}")

    def to(self, device):
        """重写 to 方法，确保所有组件一起迁移。"""
        super().to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.stage_model.to(device)
        self.subtask_model.to(device)
        return self

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        """从 batch 计算取值于 [0, 1] 的稠密进度奖励。

        要求 batch 包含：
        - "observation_features" 或 video 嵌入：(B, T, 512)
        - "language_embedding" 或 text 嵌入：(B, 512)
        - 可选的 "observation.state"：(B, T, state_dim)
        """
        text_emb = batch.get("language_embedding", batch.get("text_features"))
        video_emb = batch.get("observation_features", batch.get("video_features"))
        state = batch.get("observation.state", batch.get("state_features"))

        rewards = self.calculate_rewards(text_emb, video_emb, state)
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).float()
        return rewards

    @torch.no_grad()
    def calculate_rewards(
        self,
        text_embeddings: np.ndarray | torch.Tensor,
        video_embeddings: np.ndarray | torch.Tensor,
        state_features: np.ndarray | torch.Tensor | None = None,
        lengths: np.ndarray | torch.Tensor | None = None,
        return_all_frames: bool = False,
        return_stages: bool = False,
        return_confidence: bool = False,
        head_mode: str | None = "sparse",
        frame_index: int | None = None,
    ) -> np.ndarray | tuple:
        """
        为给定的 text、video 和 state 表示计算奖励。

        这是 SARM 奖励计算的规范方法，用于：
        - 推理/可视化
        - RA-BC 权重计算

        Args:
            text_embeddings: 编码后的 text 表示 (batch_size, 512)
            video_embeddings: 编码后的 video 表示 (batch_size, num_frames, 512)
            state_features: 关节状态特征 (batch_size, num_frames, state_dim)
            lengths: 有效序列长度 (batch_size,)
            return_all_frames: 若为 True，返回所有帧的奖励
            return_stages: 若为 True，同时返回阶段预测
            return_confidence: 若为 True，同时返回阶段置信度
            head_mode: 使用哪个回归头（"sparse" 或 "dense"）
            frame_index: 要提取的目标帧索引（默认：n_obs_steps）。

        Returns:
            奖励，以及可选的阶段 probs/置信度。
        """
        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        if isinstance(video_embeddings, np.ndarray):
            video_embeddings = torch.tensor(video_embeddings, dtype=torch.float32)
        if state_features is not None and isinstance(state_features, np.ndarray):
            state_features = torch.tensor(state_features, dtype=torch.float32)

        # 处理单样本情况
        if text_embeddings.dim() == 1:
            text_embeddings = text_embeddings.unsqueeze(0)
            video_embeddings = video_embeddings.unsqueeze(0)
            if state_features is not None:
                state_features = state_features.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        batch_size = video_embeddings.shape[0]
        seq_len = video_embeddings.shape[1]

        scheme = head_mode

        # 若未提供则使用默认 lengths
        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.int32)
        elif isinstance(lengths, np.ndarray):
            lengths = torch.tensor(lengths, dtype=torch.int32)

        # 将 video 重塑为 (B, N, T, D) 以适配多相机格式
        # 当前为单相机：(B, T, D) -> (B, 1, T, D)
        img_seq = video_embeddings.unsqueeze(1).to(self.device)
        lang_emb = text_embeddings.to(self.device)
        state = (
            state_features.to(self.device)
            if state_features is not None
            else torch.zeros(batch_size, seq_len, self.config.max_state_dim, device=self.device)
        )
        lens = lengths.to(self.device)

        # 将 state 填充到 max_state_dim
        state = pad_state_to_max_dim(state, self.config.max_state_dim)

        # 获取此 scheme 的 num_classes
        num_classes = self.config.num_sparse_stages if scheme == "sparse" else self.config.num_dense_stages

        # 运行阶段模型
        stage_logits = self.stage_model(img_seq, lang_emb, state, lens, scheme=scheme)
        stage_probs = F.softmax(stage_logits, dim=-1)  # (B, T, num_classes)
        stage_idx = stage_probs.argmax(dim=-1)  # (B, T)
        stage_conf = stage_probs.gather(-1, stage_idx.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # 创建 one-hot 阶段先验
        stage_onehot = F.one_hot(stage_idx, num_classes=num_classes).float()  # (B, T, C)
        stage_emb = stage_onehot.unsqueeze(1)  # (B, 1, T, C)

        # 运行子任务模型
        tau_pred = self.subtask_model(img_seq, lang_emb, state, lens, stage_emb, scheme=scheme)

        # 计算最终奖励：stage + tau
        raw_reward = stage_idx.float() + tau_pred  # (B, T)

        # 使用时间占比归一化到 [0, 1] 以进行正确加权
        if scheme == "sparse":
            normalized_reward = normalize_stage_tau(
                raw_reward,
                num_stages=num_classes,
                temporal_proportions=self.config.sparse_temporal_proportions,
                subtask_names=self.config.sparse_subtask_names,
            )
        else:
            normalized_reward = normalize_stage_tau(
                raw_reward,
                num_stages=num_classes,
                temporal_proportions=self.config.dense_temporal_proportions,
                subtask_names=self.config.dense_subtask_names,
            )

        # 默认帧索引为 n_obs_steps（最后一个观测帧）
        if frame_index is None:
            frame_index = self.config.n_obs_steps

        # 准备输出（批处理模式或不平滑）
        if return_all_frames:
            rewards = normalized_reward.cpu().numpy()
        else:
            rewards = normalized_reward[:, frame_index].cpu().numpy()

        if single_sample:
            rewards = rewards[0] if not return_all_frames else rewards[0]

        outputs = [rewards]
        if return_stages:
            probs = stage_probs.cpu().numpy()
            if single_sample:
                probs = probs[0]
            outputs.append(probs)
        if return_confidence:
            conf = stage_conf.cpu().numpy()
            if single_sample:
                conf = conf[0]
            outputs.append(conf)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def train(self, mode: bool = True):
        """为两个模型设置训练模式。"""
        super().train(mode)
        self.stage_model.train(mode)
        self.subtask_model.train(mode)
        return self

    def eval(self):
        """为两个模型设置评估模式。"""
        return self.train(False)

    def parameters(self):
        """重写以返回两个模型的可训练参数。"""
        from itertools import chain

        return chain(self.stage_model.parameters(), self.subtask_model.parameters())

    def get_optim_params(self):
        """重写以返回两个模型的优化器参数。"""
        return self.parameters()

    def reset(self):
        """SARM 没有需要重置的 episode 级状态。"""
        pass

    def _train_step(
        self,
        img_emb: torch.Tensor,  # (B, N, T, D)
        lang_emb: torch.Tensor,  # (B, E) 或 (B, T, E)
        state: torch.Tensor,  # (B, T, state_dim)
        lengths: torch.Tensor,  # (B,)
        targets: torch.Tensor,  # (B, T) - 格式：stage.tau
        scheme: str,
    ) -> dict[str, torch.Tensor]:
        """
        针对一种标注方案的单步训练。

        实现 75%/25% 的 GT/预测阶段条件。

        Args:
            img_emb: 图像嵌入 (B, N, T, D)
            lang_emb: 语言嵌入
            state: 状态特征
            lengths: 有效序列长度
            targets: 目标值，其中 floor=stage，余数=tau
            scheme: "sparse" 或 "dense"

        Returns:
            包含 stage_loss、subtask_loss、total_loss 的字典
        """
        num_classes = self.config.num_sparse_stages if scheme == "sparse" else self.config.num_dense_stages

        # 真值：stage（整数）与 tau（小数）
        # 将阶段索引鈐制到有效范围 [0, num_classes-1]，以处理 targets
        # 可能超出预期范围的边界情况（例如子任务之间的帧）
        gt_stage = torch.floor(targets).long().clamp(0, num_classes - 1)  # (B, T)
        gt_tau = torch.remainder(targets, 1.0)  # (B, T)

        # 运行阶段模型
        stage_pred = self.stage_model(img_emb, lang_emb, state, lengths, scheme=scheme)

        # 75%/25% 的 GT/预测阶段条件
        if random.random() < self.gt_stage_ratio:
            # 模式 1：使用真值阶段 -> one-hot
            stage_emb = gen_stage_emb(num_classes, targets)  # (B, 1, T, C)
        else:
            # 模式 2：使用预测阶段的 argmax -> one-hot
            stage_idx = stage_pred.argmax(dim=-1)  # (B, T)
            stage_onehot = F.one_hot(stage_idx, num_classes=num_classes).float()  # (B, T, C)
            stage_emb = stage_onehot.unsqueeze(1)  # (B, 1, T, C)

        # 使用阶段先验运行子任务模型
        tau_pred = self.subtask_model(img_emb, lang_emb, state, lengths, stage_emb, scheme=scheme)

        # 计算损失
        stage_loss = F.cross_entropy(stage_pred.view(-1, num_classes), gt_stage.view(-1), reduction="mean")
        subtask_loss = F.mse_loss(tau_pred, gt_tau, reduction="mean")

        return {
            "stage_loss": stage_loss,
            "subtask_loss": subtask_loss,
            "total_loss": stage_loss + subtask_loss,
        }

    def forward(self, batch):
        """
        SARM 奖励模型训练的前向传播。

        使用 stage+tau 目标格式，其中：
        - 整数部分 = 阶段索引
        - 小数部分 = 阶段内进度（tau）

        训练采用 75%/25% 的 GT/预测阶段条件。

        Args:
            batch: 带有 'observation' 的字典，其中包含：
                - 'video_features': (B, T, 512) 预编码的 video 特征
                - 'text_features': (B, 512) 或 (B, T, 512) 的 text 特征
                - 'state_features': (B, T, state_dim) 的关节状态特征
                - 'lengths': (B,) 的有效序列长度
                - 'sparse_targets': (B, T) 的稀疏目标（stage.tau 格式）
                - 'dense_targets': (B, T) 的稠密目标（可选，用于 dual 模式）

        Returns:
            (total_loss, 包含各损失分量的 output_dict) 组成的元组
        """
        observation = batch.get(OBS_STR, batch)

        # 提取特征
        video_features = observation["video_features"].to(self.device)
        text_features = observation["text_features"].to(self.device)
        state_features = observation.get("state_features")
        if state_features is not None:
            state_features = state_features.to(self.device)

        batch_size = video_features.shape[0]
        seq_len = video_features.shape[1]

        # 获取 lengths（默认为完整序列）
        lengths = observation.get("lengths")
        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=self.device)
        else:
            lengths = lengths.to(self.device)

        # 将 video 重塑为 (B, N, T, D) - 单相机
        img_emb = video_features.unsqueeze(1)

        # 将 state 填充到 max_state_dim
        if state_features is None:
            state_features = torch.zeros(batch_size, seq_len, self.config.max_state_dim, device=self.device)
        else:
            state_features = pad_state_to_max_dim(state_features, self.config.max_state_dim)

        output_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # 稀疏训练（始终执行）
        sparse_targets = observation.get("sparse_targets")
        if sparse_targets is None:
            # 尝试旧格式
            sparse_targets = observation.get("targets")
        if sparse_targets is None:
            raise ValueError("sparse_targets (or targets) is required for SARM training")
        sparse_targets = sparse_targets.to(self.device)

        sparse_result = self._train_step(
            img_emb, text_features, state_features, lengths, sparse_targets, scheme="sparse"
        )
        output_dict["sparse_stage_loss"] = sparse_result["stage_loss"].item()
        output_dict["sparse_subtask_loss"] = sparse_result["subtask_loss"].item()
        total_loss = total_loss + sparse_result["total_loss"]

        # 稠密训练（若为 dual 模式）
        if self.config.uses_dual_heads:
            dense_targets = observation.get("dense_targets")
            if dense_targets is not None:
                dense_targets = dense_targets.to(self.device)
                dense_result = self._train_step(
                    img_emb, text_features, state_features, lengths, dense_targets, scheme="dense"
                )
                output_dict["dense_stage_loss"] = dense_result["stage_loss"].item()
                output_dict["dense_subtask_loss"] = dense_result["subtask_loss"].item()
                total_loss = total_loss + dense_result["total_loss"]

        output_dict["total_loss"] = total_loss.item()
        return total_loss, output_dict


def compute_stage_loss(stage_logits: torch.Tensor, target_stages: torch.Tensor) -> torch.Tensor:
    """计算阶段分类的交叉熵损失。"""
    _, _, num_stages = stage_logits.shape
    stage_logits_flat = stage_logits.reshape(-1, num_stages)
    # 将目标阶段索引鈐制到有效范围 [0, num_stages-1]
    target_stages_flat = target_stages.reshape(-1).clamp(0, num_stages - 1)
    return F.cross_entropy(stage_logits_flat, target_stages_flat)
