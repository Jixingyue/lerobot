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

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )
else:
    Qwen2_5_VLConfig = None
    Qwen2_5_VLTextConfig = None
    Qwen2_5_VLVisionConfig = None


EO1_DEFAULT_SYSTEM_MESSAGE = "You are a helpful physical assistant."


def _eo1_default_recipe() -> dict:
    """序列化后的 recipe；使策略配置发现与数据集 extras 保持独立。"""
    return {
        "messages": [
            {
                "role": "system",
                "content": EO1_DEFAULT_SYSTEM_MESSAGE,
                "stream": "low_level",
            },
            {
                "role": "user",
                "content": "${task}\nPredict the next action in language.",
                "stream": "low_level",
            },
            {
                "role": "assistant",
                "content": "${subtask}",
                "stream": "low_level",
                "target": True,
                "if_present": "subtask",
            },
        ]
    }


@PreTrainedConfig.register_subclass("eo1")
@dataclass
class EO1Config(PreTrainedConfig):
    """LeRobot 中原生 EO1 策略集成的配置。"""

    vlm_base: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_config: dict | None = None

    # 视觉处理器设置。
    image_min_pixels: int | None = 64 * 28 * 28
    image_max_pixels: int | None = 128 * 28 * 28
    use_fast_processor: bool = False

    # 执行和动作范围。
    n_obs_steps: int = 1
    # 与已发布的 IPEC-COMMUNITY/EO-1-3B 检查点保持一致。
    chunk_size: int = 16
    n_action_steps: int = 16

    # 状态/动作填充，以匹配 EO1 流匹配头的维度。
    max_state_dim: int = 32
    max_action_dim: int = 32

    # 流匹配采样。
    num_denoise_steps: int = 10
    num_action_layers: int = 2
    action_act: str = "linear"
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    supervise_padding_action_dims: bool = True
    supervise_padding_actions: bool = True

    # 针对 Qwen 主干的策略级 dtype 请求。
    # - "auto"：遵循主干配置/检查点的默认 dtype。对于 Qwen2.5-VL，这会解析为 bf16。
    #           EO1 流匹配头仍将其自身参数保持在 fp32。
    # - "bfloat16"：无论保存的配置默认值如何，都强制主干以 bf16 初始化/加载。
    # - "float32"：强制主干以 fp32 初始化/加载，以获得最大的数值保守性。
    dtype: str = "auto"  # 选项："auto"、"bfloat16"、"float32"
    force_fp32_autocast: bool = True

    # 传递给 Qwen 主干的可选注意力后端请求。
    # 常见取值：None、"eager"、"sdpa"、"flash_attention_2"。
    attn_implementation: str | None = None

    # 训练设置。
    gradient_checkpointing: bool = False  # 启用梯度检查点以优化显存
    # 内置 recipe 处理带标注的训练和运行时提示词。
    # recipe_path 可选地覆盖它；recipe=None 则禁用 recipe 训练。
    recipe_path: str | None = None
    # EO-1 的语言约定。默认使用已发布检查点所回答的 subtask 措辞；
    # 使用 `recipe_path` 的微调会替换它，此后检查点会用其训练时
    # 所用的 recipe 来提示自己。
    recipe: dict | None = field(default_factory=_eo1_default_recipe)
    tokenizer_max_length: int = 1000
    text_temperature: float = 0.0
    text_top_p: float = 1.0
    flow_loss_weight: float = 1.0
    text_loss_weight: float = 0.01

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # 与 EO1/experiments/2_libero/train.sh 及 EO1 TrainPipelineConfig 默认值对齐的优化器设置。
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.1
    optimizer_grad_clip_norm: float = 1.0

    # 与 EO1 train.sh 对齐的调度器设置：带预热的余弦调度，warmup_ratio=0.03。
    # 注意：如果 --steps < scheduler_decay_steps，这些值会自动缩放
    # 例如，--steps=3000 会将预热缩放为 100，衰减缩放为 3000
    scheduler_warmup_steps: int = 900  # 0.03 * 30_000 长期运行步数
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        if self.recipe_path is not None:
            from lerobot.datasets.recipe import resolve_recipe_override

            self.recipe = asdict(resolve_recipe_override(self.recipe, self.recipe_path))

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.tokenizer_max_length < self.chunk_size + 1:
            raise ValueError("tokenizer_max_length must leave room for the EO-1 action chunk.")
        if self.flow_loss_weight < 0 or self.text_loss_weight < 0:
            raise ValueError("EO-1 loss weights must be non-negative.")
        if self.flow_loss_weight == 0 and self.text_loss_weight == 0:
            raise ValueError("At least one EO-1 training loss must be enabled.")
        if self.text_temperature < 0:
            raise ValueError("text_temperature must be non-negative.")
        if not 0 < self.text_top_p <= 1:
            raise ValueError("text_top_p must be in (0, 1].")

        # 仅在调用方未提供序列化的主干配置时才填充它。
        if self.vlm_config is None:
            require_package("transformers", extra="eo1")
            self.vlm_config = Qwen2_5_VLConfig.from_pretrained(self.vlm_base).to_dict()

    @property
    def vlm_backbone_config(self) -> Qwen2_5_VLConfig:
        require_package("transformers", extra="eo1")
        config_dict = deepcopy(self.vlm_config)
        if self.attn_implementation is not None:
            config_dict["attn_implementation"] = self.attn_implementation
        return Qwen2_5_VLConfig(**config_dict)

    @property
    def text_config(self) -> Qwen2_5_VLTextConfig:
        return self.vlm_backbone_config.text_config

    @property
    def vision_config(self) -> Qwen2_5_VLVisionConfig:
        return self.vlm_backbone_config.vision_config

    def validate_features(self) -> None:
        """校验并设置 EO1 的输入和输出特征。"""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "EO1 policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
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
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
