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


@PreTrainedConfig.register_subclass("pi0")
@dataclass
class PI0Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # 可选项："bfloat16"、"float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # 要预测的动作步数，在 openpi 中称为 "action_horizon"
    n_action_steps: int = 50  # 要执行的动作步数

    # 较短的状态和动作向量将被填充到这些维度
    max_state_dim: int = 32
    max_action_dim: int = 32

    # 流匹配参数：参见 openpi `PI0Pytorch`
    num_inference_steps: int = 10  # 推理时的去噪步数
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # 相对动作：将绝对动作转换为相对动作（相对于状态）。
    use_relative_actions: bool = False
    # 要从相对动作中排除（保持绝对）的关节名称。空列表 = 所有维度均为相对。
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # 由 make_policy 在运行时根据数据集元数据填充。
    action_feature_names: list[str] | None = None

    # 实时分块（Real-Time Chunking, RTC）配置
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # 参见 openpi `preprocessing_pytorch.py`

    # 添加空图像。用于在不存在图像特征时添加空相机。
    empty_cameras: int = 0

    # 归一化
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # 训练设置
    gradient_checkpointing: bool = False  # 启用梯度检查点以优化显存
    compile_model: bool = False  # 是否使用 torch.compile 优化模型
    compile_mode: str = "max-autotune"  # Torch 编译模式
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
    # 例如，--steps=3000 会将 warmup 缩放为 100，将衰减缩放为 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48  # 参见 openpi `__post_init__`
    text_tokenizer_name: str = "google/paligemma-3b-pt-224"

    def __post_init__(self):
        super().__post_init__()

        # 校验配置
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """校验并设置输入/输出特征。"""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
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
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
