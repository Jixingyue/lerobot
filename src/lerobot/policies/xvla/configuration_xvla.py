#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import CosineDecayWithWarmupSchedulerConfig, XVLAAdamWConfig
from lerobot.utils.constants import OBS_IMAGES

# 用于类型检查和懒加载的条件导入
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import Florence2Config
else:
    Florence2Config = None


def _translate_vision_config(vision_config: dict[str, Any]) -> dict[str, Any]:
    """将视觉配置从 Microsoft 原始远程代码版 Florence-2 格式（现有 XVLA checkpoint
    所使用的格式）转换为原生 ``transformers`` 格式。

    已经是原生格式的配置会原样透传，不做修改。
    """
    vision = dict(vision_config)
    model_type = vision.pop("model_type", None)
    if model_type not in (None, "davit", "florence_vision"):
        raise ValueError(f"Unsupported Florence-2 vision backbone: {model_type!r}")
    vision.pop("enable_checkpoint", None)

    image_pos_embed = vision.pop("image_pos_embed", None)
    if image_pos_embed is not None:
        if image_pos_embed.get("type") != "learned_abs_2d":
            raise ValueError(f"Unsupported image_pos_embed type: {image_pos_embed.get('type')!r}")
        vision["max_position_embeddings"] = image_pos_embed["max_pos_embeddings"]

    visual_temporal_embedding = vision.pop("visual_temporal_embedding", None)
    if visual_temporal_embedding is not None:
        if visual_temporal_embedding.get("type") != "COSINE":
            raise ValueError(
                f"Unsupported visual_temporal_embedding type: {visual_temporal_embedding.get('type')!r}"
            )
        vision["max_temporal_embeddings"] = visual_temporal_embedding["max_temporal_embeddings"]

    image_feature_source = vision.pop("image_feature_source", None)
    if image_feature_source is not None and list(image_feature_source) != [
        "spatial_avg_pool",
        "temporal_avg_pool",
    ]:
        # 原生的 Florence2MultiModalProjector 硬编码了这一特征组合
        raise ValueError(f"Unsupported image_feature_source: {image_feature_source!r}")

    if "dim_embed" in vision:
        vision["embed_dim"] = vision.pop("dim_embed")
    return vision


@PreTrainedConfig.register_subclass("xvla")
@dataclass
class XVLAConfig(PreTrainedConfig):
    """
    XVLA（Extended Vision-Language-Action）策略的配置类，使其能够
    接入 LeRobot 训练栈。

    该配置镜像了原始 XVLA 仓库中暴露的各项开关，同时也声明了
    LeRobot 所要求的输入/输出特征契约。
    """

    # 输入 / 输出结构
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 32
    dtype: str = "float32"  # 可选："bfloat16"、"float32"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Florence2 主干网络和分词器配置
    florence_config: dict[str, Any] = field(default_factory=dict)
    tokenizer_name: str = "facebook/bart-large"
    tokenizer_max_length: int = 64
    tokenizer_padding_side: str = "right"
    pad_language_to: str = "max_length"

    # Transformer 动作头
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_domains: int = 30
    len_soft_prompts: int = 32
    dim_time: int = 32
    max_len_seq: int = 512
    use_hetero_proj: bool = False

    # 动作与本体感知
    action_mode: str = "ee6d"
    num_denoising_steps: int = 10
    use_proprio: bool = True
    max_state_dim: int = 32
    max_action_dim: int = 20  # 用于填充的最大动作维度（"auto" 动作模式使用）
    domain_feature_key: str | None = None

    # 视觉预处理
    resize_imgs_with_padding: tuple[int, int] | None = None
    num_image_views: int | None = None
    empty_cameras: int = 0

    # VLM 组件的冻结选项
    # 默认情况下冻结 VLM 编码器，只训练策略 transformer 和 soft prompt
    freeze_vision_encoder: bool = False  # 冻结 VLM 视觉编码器权重
    freeze_language_encoder: bool = False  # 冻结 VLM 语言编码器权重
    train_policy_transformer: bool = True  # 允许训练策略 transformer
    train_soft_prompts: bool = True  # 允许训练 soft prompt

    # 训练预设
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.99)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 10.0
    # Soft-prompt 学习率设置（用于可选的 warm-up）
    optimizer_soft_prompt_lr_scale: float = 1.0  # soft-prompt 学习率的缩放因子
    optimizer_soft_prompt_warmup_lr_scale: float | None = None  # warmup 的起始缩放（例如 0.01）

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.chunk_size <= 0:
            raise ValueError("`chunk_size` must be strictly positive.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must be <= `chunk_size` ({self.chunk_size})."
            )
        if self.num_image_views is not None and self.num_image_views <= 0:
            raise ValueError("`num_image_views` must be > 0 when specified.")
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        self._florence_config_obj: Florence2Config | None = None

    def get_florence_config(self) -> Florence2Config:
        """
        构建（并缓存）支撑 VLM 的原生 ``transformers`` Florence-2 配置。

        ``florence_config`` 既可以用原生 ``transformers`` 格式给出，也可以用现有 XVLA
        checkpoint 所存储的 Microsoft 原始远程代码格式给出（例如视觉配置中带有
        ``dim_embed`` / ``image_pos_embed``）；后者会被逐字段转换为原生格式。
        """
        if self._florence_config_obj is None:
            config_dict = dict(self.florence_config)
            if config_dict.get("vision_config") is None:
                raise ValueError("vision_config is required")
            if config_dict.get("text_config") is None:
                raise ValueError("text_config is required")

            vision_config = _translate_vision_config(config_dict["vision_config"])
            text_config = dict(config_dict["text_config"])
            if text_config.get("model_type", "florence2_language") == "florence2_language":
                # 微软远程代码版的语言配置逐字段对应 BART。
                text_config["model_type"] = "bart"

            kwargs = {
                key: config_dict[key]
                for key in (
                    "pad_token_id",
                    "bos_token_id",
                    "eos_token_id",
                    "image_token_id",
                    "is_encoder_decoder",
                    "tie_word_embeddings",
                )
                if key in config_dict
            }
            self._florence_config_obj = Florence2Config(
                vision_config=vision_config, text_config=text_config, **kwargs
            )
        return self._florence_config_obj

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("XVLA requires at least one visual feature in the inputs.")
        if self.use_proprio and self.robot_state_feature is None:
            raise ValueError("`use_proprio=True` requires a proprioceptive state feature.")
        if self.num_image_views is None:
            self.num_image_views = len(self.image_features) + self.empty_cameras
        else:
            self.num_image_views = max(self.num_image_views, len(self.image_features) + self.empty_cameras)

        if self.empty_cameras > 0:
            height, width = (480, 640)
            if self.resize_imgs_with_padding is not None:
                height, width = self.resize_imgs_with_padding
            for idx in range(self.empty_cameras):
                key = f"{OBS_IMAGES}.empty_camera_{idx}"
                if key not in self.input_features:
                    self.input_features[key] = PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, height, width),
                    )

    def get_optimizer_preset(self) -> XVLAAdamWConfig:
        """返回 XVLA 专用的、带差异化学习率的优化器。

        该优化器应用：
        - VLM 参数使用 1/10 的学习率（稳定优化）
        - transformer/动作头使用完整学习率
        - soft-prompt 使用可配置的学习率（支持可选的 warm-up）
        """
        return XVLAAdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            soft_prompt_lr_scale=self.optimizer_soft_prompt_lr_scale,
            soft_prompt_warmup_lr_scale=self.optimizer_soft_prompt_warmup_lr_scale,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list[int] | None:
        return None
