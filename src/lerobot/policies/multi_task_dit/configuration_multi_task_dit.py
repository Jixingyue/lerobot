#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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

import logging
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("multi_task_dit")
@dataclass
class MultiTaskDiTConfig(PreTrainedConfig):
    """多任务扩散 Transformer（DiT）策略的配置。

    一个基于 Transformer 的策略，同时支持扩散（diffusion）和流匹配（flow matching）目标，
    用于带文本和视觉条件的多任务机器人学习。
    """

    n_obs_steps: int = 2  # 用于时间上下文的观测步数
    horizon: int = 32  # 要预测的动作步数
    n_action_steps: int = 24  # 每次策略调用执行的动作数（30Hz 下约 0.8 秒）

    # 目标选择
    objective: str = "diffusion"  # "diffusion" 或 "flow_matching"

    # --- 扩散专用（当 objective="diffusion" 时使用）---
    noise_scheduler_type: str = "DDPM"  # "DDPM" 或 "DDIM"
    num_train_timesteps: int = 100  # 扩散时间步数
    beta_schedule: str = "squaredcos_cap_v2"  # 噪声调度类型
    beta_start: float = 0.0001  # 起始噪声水平
    beta_end: float = 0.02  # 终止噪声水平
    prediction_type: str = "epsilon"  # "epsilon"（预测噪声）或 "sample"（预测干净样本）
    clip_sample: bool = True  # 在去噪过程中钳制样本
    clip_sample_range: float = 1.0  # 钳制范围 [-x, x]
    num_inference_steps: int | None = None  # 推理时的去噪步数（默认为 num_train_timesteps）

    # --- 流匹配专用（当 objective="flow_matching" 时使用）---
    sigma_min: float = 0.0  # 流插值路径中的最小噪声
    num_integration_steps: int = 100  # 推理时的 ODE 积分步数
    integration_method: str = "euler"  # ODE 求解器："euler" 或 "rk4"
    timestep_sampling_strategy: str = "beta"  # "uniform" 或 "beta"

    timestep_sampling_s: float = 0.999  # （仅 beta）最大时间步阈值
    timestep_sampling_alpha: float = 1.5  # （仅 beta）Beta 分布的 alpha
    timestep_sampling_beta: float = 1.0  # （仅 beta）Beta 分布的 beta

    # Transformer 架构
    hidden_dim: int = 512  # Transformer 隐藏层维度
    num_layers: int = 6  # Transformer 层数
    num_heads: int = 8  # 注意力头数
    dropout: float = 0.1  # Dropout 比率
    use_positional_encoding: bool = False  # 使用绝对位置编码
    timestep_embed_dim: int = 256  # 时间步嵌入维度
    use_rope: bool = True  # 使用旋转位置嵌入（RoPE）
    rope_base: float = 10000.0  # RoPE 基频

    # 视觉编码器（CLIP）
    vision_encoder_name: str = "openai/clip-vit-base-patch16"  # HuggingFace CLIP 模型
    use_separate_rgb_encoder_per_camera: bool = False  # 每个相机视角使用独立编码器
    vision_encoder_lr_multiplier: float = 0.1  # 视觉编码器的学习率乘数
    image_resize_shape: tuple[int, int] | None = None  # 裁剪前先缩放图像
    image_crop_shape: tuple[int, int] | None = (224, 224)  # 裁剪形状（CLIP 默认值）
    image_crop_is_random: bool = True  # 训练时随机裁剪，推理时中心裁剪

    # 文本编码器（CLIP）
    text_encoder_name: str = "openai/clip-vit-base-patch16"  # HuggingFace CLIP 模型
    tokenizer_max_length: int = 77  # 分词后文本的最大长度（CLIP 默认为 77）
    tokenizer_padding: str = "max_length"  # 填充策略："max_length" 或 "longest"
    tokenizer_padding_side: str = "right"  # 填充侧："left" 或 "right"
    tokenizer_truncation: bool = True  # 是否截断超过 max_length 的序列

    # 归一化
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # 训练/优化器
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 0
    do_mask_loss_for_padding: bool = False

    # 自动计算
    drop_n_last_frames: int | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = self.horizon - self.n_action_steps - self.n_obs_steps + 1

        self._validate()

    def _validate(self):
        """校验配置参数。"""
        # 目标校验
        if self.objective not in ["diffusion", "flow_matching"]:
            raise ValueError(f"objective must be 'diffusion' or 'flow_matching', got '{self.objective}'")

        # Transformer 校验
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be between 0.0 and 1.0")

        # 视觉编码器校验
        if "clip" not in self.vision_encoder_name.lower():
            raise ValueError(
                f"vision_encoder_name must be a CLIP model (contain 'clip'), got '{self.vision_encoder_name}'"
            )
        if (
            self.image_resize_shape
            and self.image_crop_shape
            and (
                self.image_crop_shape[0] > self.image_resize_shape[0]
                or self.image_crop_shape[1] > self.image_resize_shape[1]
            )
        ):
            logging.warning(
                "image_crop_shape %s must be <= image_resize_shape %s; disabling cropping.",
                self.image_crop_shape,
                self.image_resize_shape,
            )
            self.image_crop_shape = None

        # 文本编码器校验
        if "clip" not in self.text_encoder_name.lower():
            raise ValueError(
                f"text_encoder_name must be a CLIP model (contain 'clip'), got '{self.text_encoder_name}'"
            )

        # 目标特定校验
        if self.objective == "diffusion":
            if self.noise_scheduler_type not in ["DDPM", "DDIM"]:
                raise ValueError(
                    f"noise_scheduler_type must be 'DDPM' or 'DDIM', got {self.noise_scheduler_type}"
                )
            if self.prediction_type not in ["epsilon", "sample"]:
                raise ValueError(f"prediction_type must be 'epsilon' or 'sample', got {self.prediction_type}")
            if self.num_train_timesteps <= 0:
                raise ValueError(f"num_train_timesteps must be positive, got {self.num_train_timesteps}")
            if not (0.0 <= self.beta_start <= self.beta_end <= 1.0):
                raise ValueError(f"Invalid beta values: {self.beta_start}, {self.beta_end}")

        elif self.objective == "flow_matching":
            if not (0.0 <= self.sigma_min <= 1.0):
                raise ValueError(f"sigma_min must be in [0, 1], got {self.sigma_min}")
            if self.num_integration_steps <= 0:
                raise ValueError(f"num_integration_steps must be positive, got {self.num_integration_steps}")
            if self.integration_method not in ["euler", "rk4"]:
                raise ValueError(
                    f"integration_method must be 'euler' or 'rk4', got {self.integration_method}"
                )
            if self.timestep_sampling_strategy not in ["uniform", "beta"]:
                raise ValueError("timestep_sampling_strategy must be 'uniform' or 'beta'")
            if self.timestep_sampling_strategy == "beta":
                if not (0.0 < self.timestep_sampling_s <= 1.0):
                    raise ValueError(f"timestep_sampling_s must be in (0, 1], got {self.timestep_sampling_s}")
                if self.timestep_sampling_alpha <= 0:
                    raise ValueError("timestep_sampling_alpha must be positive")
                if self.timestep_sampling_beta <= 0:
                    raise ValueError("timestep_sampling_beta must be positive")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        """校验所需的输入特征是否存在且配置正确。"""
        # 如果配置的裁剪不合适，则禁用裁剪而不是报错。
        # 注意：如果设置了 image_resize_shape，裁剪将在缩放*之后*应用。
        if self.image_crop_shape is not None:
            for key, image_ft in self.image_features.items():
                # image_ft.shape 为 (C, H, W)
                effective_h, effective_w = (
                    self.image_resize_shape
                    if self.image_resize_shape is not None
                    else (image_ft.shape[1], image_ft.shape[2])
                )
                if self.image_crop_shape[0] > effective_h or self.image_crop_shape[1] > effective_w:
                    logging.warning(
                        "image_crop_shape %s doesn't fit within effective image shape (%s, %s) for '%s'; disabling cropping.",
                        self.image_crop_shape,
                        effective_h,
                        effective_w,
                        key,
                    )
                    self.image_crop_shape = None
                    break

        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_ft.shape:
                    raise ValueError(
                        f"Image '{key}' shape {image_ft.shape} != '{first_key}' shape {first_ft.shape}"
                    )

    @property
    def is_diffusion(self) -> bool:
        return self.objective == "diffusion"

    @property
    def is_flow_matching(self) -> bool:
        return self.objective == "flow_matching"

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
