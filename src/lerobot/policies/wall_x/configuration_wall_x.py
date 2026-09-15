# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from dataclasses import asdict, dataclass, field

from lerobot.configs import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
    PreTrainedConfig,
)
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


def _wall_x_default_recipe() -> dict:
    """序列化的 recipe；使策略配置的发现不依赖于数据集的 extras。"""
    return {
        "messages": [
            {
                "role": "user",
                "content": "${task}\nPredict the next action in language.\n",
                "stream": "high_level",
            },
            {
                "role": "assistant",
                "content": "${subtask}",
                "stream": "high_level",
                "target": True,
                "if_present": "subtask",
            },
        ]
    }


@PreTrainedConfig.register_subclass("wall_x")
@dataclass
class WallXConfig(PreTrainedConfig):
    """
    Wall-X 策略的配置类。

    Wall-X 基于 Qwen2.5-VL，具备使用 flow matching 进行动作预测的能力。
    它通过统一的动作表示支持跨本体（cross-embodiment）机器人控制。

    该配置支持结合视觉、语言和动作数据的多模态学习。
    """

    # ==================== 输入 / 输出结构 ====================
    n_obs_steps: int = 1
    chunk_size: int = 32  # wall-x 中的 action_horizon
    n_action_steps: int = 32

    # 动作维度 - wall-x 使用 20
    max_action_dim: int = 20
    max_state_dim: int = 20  # 用于本体感知（proprioception）

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ==================== 动作预测 ====================
    # 预训练模型路径
    pretrained_name_or_path: str = "x-square-robot/wall-oss-flow"

    # 分词器设置
    action_tokenizer_path: str | None = "lerobot/fast-action-tokenizer"

    # 动作预测模式："diffusion" 或 "fast"
    prediction_mode: str = "diffusion"

    # Wall-X 的双向动作 token 岛（island）目前需要使用 eager attention。
    attn_implementation: str = "eager"

    # 视觉注意力独立于文本动作 token 掩码。``auto`` 会在运行时支持时使用 PyTorch 的
    # packed 变长注意力，否则回退到原生的逐块 SDPA 实现。
    vision_attn_implementation: str = "auto"

    # 可选的、显式的外部语言 recipe 覆盖。
    recipe_path: str | None = None
    # WALL-X 的语言契约：默认为 WALL-OSS 训练时使用的子任务措辞；使用 `recipe_path`
    # 微调会替换它，之后 checkpoint 会用自己训练时所用的 recipe 来提示自身。
    recipe: dict | None = field(default_factory=_wall_x_default_recipe)
    tokenizer_max_length: int = 768
    text_temperature: float = 0.0
    text_top_p: float = 1.0
    flow_loss_weight: float = 1.0
    text_loss_weight: float = 0.01

    # ==================== 优化器预设 ====================
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self):
        super().__post_init__()

        if self.recipe_path is not None:
            from lerobot.datasets.recipe import resolve_recipe_override

            self.recipe = asdict(resolve_recipe_override(self.recipe, self.recipe_path))

        # 输入校验
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        if self.prediction_mode not in ["diffusion", "fast"]:
            raise ValueError(f"prediction_mode must be 'diffusion' or 'fast', got {self.prediction_mode}")

        if self.attn_implementation != "eager":
            raise ValueError(
                "Wall-X currently supports only attn_implementation='eager' because its "
                "bidirectional action-token islands require an explicit attention mask."
            )

        if self.vision_attn_implementation not in {"auto", "sdpa", "varlen"}:
            raise ValueError(
                "vision_attn_implementation must be one of 'auto', 'sdpa', or 'varlen', got "
                f"{self.vision_attn_implementation!r}"
            )
        if self.tokenizer_max_length < self.chunk_size + 1:
            raise ValueError("tokenizer_max_length must leave room for the WALL-OSS action chunk.")
        if self.flow_loss_weight < 0 or self.text_loss_weight < 0:
            raise ValueError("WALL-OSS loss weights must be non-negative.")
        if self.flow_loss_weight == 0 and self.text_loss_weight == 0:
            raise ValueError("At least one WALL-OSS training loss must be enabled.")

        # 根据 prediction_mode 设置 use_fast_tokenizer
        if self.prediction_mode == "fast":
            self.use_fast_tokenizer = True
        elif self.prediction_mode == "diffusion":
            self.use_fast_tokenizer = False
            self.action_tokenizer_path = None  # diffusion 模式下禁用动作分词器
        else:
            raise ValueError(f"prediction_mode must be 'diffusion' or 'fast', got {self.prediction_mode}")

    def validate_features(self) -> None:
        """校验并设置输入/输出特征。"""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "Wall-X policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # 填充到 max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature
        else:
            state_shape = self.input_features[OBS_STATE].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # 填充到 max_action_dim
            )
            self.output_features[ACTION] = action_feature
        else:
            action_shape = self.output_features[ACTION].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    f"Either reduce action dimension or increase max_action_dim in config."
                )

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
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
