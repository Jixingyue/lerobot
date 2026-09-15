#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # 可选项："bfloat16"、"float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # 要预测的动作步数，在 openpi 中称为 "action_horizon"
    n_action_steps: int = 50  # 要执行的动作步数

    # MEM 短时域观测记忆（https://arxiv.org/abs/2603.03596）。
    # 历史图像 token 在 SigLIP 内部融合，并在进入语言主干之前被丢弃。
    # 历史本体感知状态变为每帧一个连续的骨干 token。
    # 这两条路径都是可选启用的，且相互独立。
    #
    # MEM 在六个间隔一秒的观测上进行预训练。`memory_stride` 以数据集帧数
    # 计数，因此默认值只有在 30 fps（LeRobot 的常用录制帧率）下才与该间隔匹配。
    # 请根据数据集进行缩放：像 `lerobot/robomme` 这样的 10 fps 数据集
    # 需要 `memory_stride=10` 才能同样表示一秒。
    use_visual_memory: bool = False
    use_proprioceptive_memory: bool = False
    memory_frames: int = 6
    memory_stride: int = 30
    memory_temporal_attention_every: int = 4

    # 较短的状态向量和动作向量将被填充到这些维度
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching 参数：参见 openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # 相对动作：将绝对动作转换为相对（相对于状态）动作。
    use_relative_actions: bool = False
    # 要从相对动作中排除（保持绝对）的关节名称。空列表 = 所有维度均为相对。
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # 在运行时由 make_policy 根据数据集元数据填充。
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) 配置
    rtc_config: RTCConfig | None = None
    # 训练期间采样的最大干净动作前缀长度。为零则禁用训练时 RTC。
    rtc_training_max_delay: int = 0

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # 参见 openpi `preprocessing_pytorch.py`

    # 添加空图像。用于在不存在图像特征时添加空相机。
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # 参见 openpi `__post_init__`
    text_tokenizer_name: str = "google/paligemma-3b-pt-224"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 对状态使用分位数归一化
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 对动作使用分位数归一化
        }
    )

    # 训练设置
    gradient_checkpointing: bool = False  # 启用梯度检查点以优化显存
    compile_model: bool = False  # 是否使用 torch.compile 优化模型
    compile_mode: str = "max-autotune"  # Torch compile 模式
    device: str | None = None  # 模型使用的设备（None = 自动检测）

    # 微调设置
    freeze_vision_encoder: bool = False  # 仅冻结视觉编码器
    train_expert_only: bool = False  # 冻结整个 VLM，仅训练动作专家和投影层

    # 优化器设置：参见 openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # 参见 openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # 调度器设置：参见 openpi `CosineDecaySchedule`
    # 注意：如果 --steps < scheduler_decay_steps，这些值会自动缩放
    # 例如，--steps=3000 会将 warmup 缩放到 100，将 decay 缩放到 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        # 校验配置
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if not 0 <= self.rtc_training_max_delay < self.chunk_size:
            raise ValueError(
                "rtc_training_max_delay must satisfy "
                f"0 <= delay < chunk_size ({self.chunk_size}), got {self.rtc_training_max_delay}"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.memory_frames < 1:
            raise ValueError("memory_frames must be at least 1")
        if self.memory_stride < 1:
            raise ValueError("memory_stride must be at least 1")
        if self.memory_temporal_attention_every < 1:
            raise ValueError("memory_temporal_attention_every must be at least 1")

    def validate_features(self) -> None:
        """校验并设置输入/输出特征。"""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # 使用配置的图像分辨率
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # 填充到 max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # 填充到 max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def image_observation_delta_indices(self) -> list[int] | None:
        if not self.use_visual_memory:
            return None
        horizon = (self.memory_frames - 1) * self.memory_stride
        return list(range(-horizon, 1, self.memory_stride))

    @property
    def state_observation_delta_indices(self) -> list[int] | None:
        if not self.use_proprioceptive_memory:
            return None
        horizon = (self.memory_frames - 1) * self.memory_stride
        return list(range(-horizon, 1, self.memory_stride))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
