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
SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation.
Paper: https://arxiv.org/abs/2509.25358
"""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@RewardModelConfig.register_subclass("sarm")
@dataclass
class SARMConfig(RewardModelConfig):
    """SARM（Stage-Aware Reward Modeling，阶段感知奖励建模）的配置类。

    支持三种标注模式：

    1. single_stage（默认）：无需标注。使用 episode 的任务描述
       作为覆盖整个 episode 的单个阶段。

    2. dense_only：使用来自 VLM 的稠密（细粒度）标注，并自动生成
       覆盖整个 episode 的单个稀疏 "task" 阶段。dense 头学习详细的
       子任务进展，而 sparse 头提供整体任务完成情况。

    3. dual：完整的双头模式，同时使用来自 VLM 的稀疏（高层级）和
       稠密（细粒度）标注。两个头分别在各自的标注上训练。

    annotation_mode 决定了模型初始化时如何加载/生成
    sparse_temporal_proportions 和 dense_temporal_proportions。
    """

    annotation_mode: str = "single_stage"  # "single_stage"、"dense_only" 或 "dual"
    n_obs_steps: int = 8  # 观测历史步数
    frame_gap: int = 30  # 帧间隔（30 fps 下 = 1 秒）
    max_rewind_steps: int = 4  # 时间增强的最大回退步数

    # 总帧数 = 1 + n_obs_steps + max_rewind_steps（在 property 中计算）
    # 带回退的训练期间：[obs_frames] + [rewind_frames]
    # 推理期间：仅 [obs_frames]

    # 架构参数
    image_dim: int = 512
    text_dim: int = 512
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 8
    max_state_dim: int = 32
    drop_n_last_frames: int = 1
    batch_size: int = 64
    clip_batch_size: int = 64
    dropout: float = 0.1
    stage_loss_weight: float = 1.0  # 使用子任务标注时阶段分类损失的权重

    rewind_probability: float = 0.8
    language_perturbation_probability: float = 0.2

    # 稀疏标注（高层级阶段）
    num_sparse_stages: int = 1
    sparse_subtask_names: list | None = None
    sparse_temporal_proportions: list | None = None

    # 稠密标注（细粒度阶段）
    num_dense_stages: int | None = None
    dense_subtask_names: list | None = None
    dense_temporal_proportions: list | None = None

    pretrained_model_path: str | None = None
    device: str | None = None
    image_key: str = OBS_IMAGES + ".top"  # 从数据集中使用的图像键
    state_key: str = OBS_STATE

    # 由处理器填充（video_features、state_features、text_features）
    input_features: dict = field(default_factory=lambda: {})

    # 输出特征（在 __post_init__ 中更新）
    output_features: dict = field(
        default_factory=lambda: {
            "stage": PolicyFeature(shape=(9, 5), type=FeatureType.REWARD),
            "progress": PolicyFeature(shape=(9, 1), type=FeatureType.REWARD),
        }
    )

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "LANGUAGE": NormalizationMode.IDENTITY,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.annotation_mode not in ["single_stage", "dense_only", "dual"]:
            raise ValueError(
                f"annotation_mode must be 'single_stage', 'dense_only', or 'dual', got {self.annotation_mode}"
            )

        if self.annotation_mode == "single_stage":
            # 使用任务描述作为阶段名，整个 episode 作为单个阶段
            self.num_sparse_stages = 1
            self.sparse_subtask_names = ["task"]
            self.sparse_temporal_proportions = [1.0]
            self.num_dense_stages = None
            self.dense_subtask_names = None
            self.dense_temporal_proportions = None

        elif self.annotation_mode == "dense_only":
            self.num_sparse_stages = 1
            self.sparse_subtask_names = ["task"]
            self.sparse_temporal_proportions = [1.0]

        self.input_features = {}
        self.output_features = {}

        if self.image_key:
            self.input_features[self.image_key] = PolicyFeature(shape=(480, 640, 3), type=FeatureType.VISUAL)

        self.input_features[self.state_key] = PolicyFeature(
            shape=(self.max_state_dim,),
            type=FeatureType.STATE,
        )

        # 根据 annotation_mode 更新输出特征
        if self.annotation_mode in ["dense_only", "dual"]:
            self.output_features["sparse_stage"] = PolicyFeature(
                shape=(self.num_frames, self.num_sparse_stages), type=FeatureType.REWARD
            )
            self.output_features["sparse_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), type=FeatureType.REWARD
            )
            dense_stages = self.num_dense_stages or self.num_sparse_stages
            self.output_features["dense_stage"] = PolicyFeature(
                shape=(self.num_frames, dense_stages), type=FeatureType.REWARD
            )
            self.output_features["dense_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), type=FeatureType.REWARD
            )
        else:
            self.output_features["sparse_stage"] = PolicyFeature(
                shape=(self.num_frames, self.num_sparse_stages), type=FeatureType.REWARD
            )
            self.output_features["sparse_progress"] = PolicyFeature(
                shape=(self.num_frames, 1), type=FeatureType.REWARD
            )

        if self.max_rewind_steps >= self.n_obs_steps:
            raise ValueError(
                f"max_rewind_steps ({self.max_rewind_steps}) must be less than n_obs_steps ({self.n_obs_steps})"
            )
        if self.num_sparse_stages < 1:
            raise ValueError(f"num_sparse_stages must be at least 1, got {self.num_sparse_stages}")
        if (
            self.annotation_mode in ["dense_only", "dual"]
            and self.num_dense_stages is not None
            and self.num_dense_stages < 2
        ):
            raise ValueError(f"num_dense_stages must be at least 2, got {self.num_dense_stages}")

    def get_optimizer_preset(self) -> AdamWConfig:
        """获取 SARM 训练的默认优化器配置。"""
        return AdamWConfig(
            lr=5e-5,
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """获取默认的学习率调度器配置。"""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=5e-5,
            decay_lr=5e-6,
            num_warmup_steps=500,
            num_decay_steps=50000,
        )

    def validate_features(self) -> None:
        pass

    @property
    def uses_dual_heads(self) -> bool:
        """模型是否使用双头（dense_only 或 dual 标注模式）。"""
        return self.annotation_mode in ["dense_only", "dual"]

    @property
    def num_frames(self) -> int:
        """序列中的总帧数。

        训练时：1 + n_obs_steps + max_rewind_steps
        序列为：[obs_frames (n_obs_steps + 1)] + [rewind_frames (max_rewind_steps)]
        """
        return 1 + self.n_obs_steps + self.max_rewind_steps

    @property
    def max_length(self) -> int:
        return self.num_frames

    @property
    def observation_delta_indices(self) -> list[int]:
        """以目标帧为中心的双向帧采样。

        n_obs_steps=8、gap=30 的示例：
        之前：[-120, -90, -60, -30]  （4 帧）
        当前：[0]                   （1 帧）
        之后：[30, 60, 90, 120]      （4 帧）
        总计：9 帧
        """
        half_steps = self.n_obs_steps // 2

        past_deltas = [-self.frame_gap * i for i in range(half_steps, 0, -1)]
        future_deltas = [self.frame_gap * i for i in range(1, half_steps + 1)]
        obs_deltas = past_deltas + [0] + future_deltas

        # 回退占位符
        rewind_deltas = [-self.frame_gap * (i + 1) for i in range(self.max_rewind_steps)]

        return obs_deltas + rewind_deltas

    @property
    def action_delta_indices(self) -> None:
        """SARM 是奖励模型，不是动作策略。"""
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
