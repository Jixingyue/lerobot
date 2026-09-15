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

"""LingBot-VA 策略的配置。

LingBot-VA 是构建在 Wan2.2 视频扩散技术栈之上的自回归视频-动作世界模型策略。
它在单个双流 transformer 中交替预测未来视频潜变量与机器人动作。参见
``docs/source/lingbot_va.mdx`` 及上游仓库（https://github.com/Robbyant/lingbot-va）。

以下默认值与上游 LIBERO 配置（``wan_va/configs/va_libero_cfg.py``）及已发布检查点的
``transformer/config.json`` 保持一致。
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig, LRSchedulerConfig
from lerobot.utils.constants import ACTION


@PreTrainedConfig.register_subclass("lingbot_va")
@dataclass
class LingBotVAConfig(PreTrainedConfig):
    """LeRobot 中原生 LingBot-VA 策略集成的配置。"""

    # Wan transformer 架构
    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    in_channels: int = 48
    out_channels: int = 48
    action_dim: int = 30
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 14336
    num_layers: int = 30
    cross_attn_norm: bool = True
    eps: float = 1e-6
    rope_max_seq_len: int = 1024
    # "flex" = 仅用于训练（需要较新版本的 torch）；推理使用 "torch" SDPA 或 "flashattn"。
    attn_mode: str = "torch"

    # 冻结的子模型（VAE + UMT5 文本编码器 + tokenizer）
    # 约 20 GB 冻结权重，不随检查点打包；从该 HF 仓库/本地目录惰性拉取
    # （目录中须包含 diffusers 风格的 ``vae/``、``text_encoder/``、``tokenizer/`` 子目录）。
    wan_pretrained_path: str = "robbyant/lingbot-va-base"
    dtype: str = "bfloat16"  # transformer / VAE / 文本编码器的 dtype："bfloat16"、"float16"、"float32"
    # 冻结的 UMT5-XXL 编码器所在设备；"cpu" 可释放约 11 GB 显存（每个 episode 只运行一次）。
    text_encoder_device: str = "cpu"

    # 观测相机（顺序很重要：潜变量沿宽度方向拼接；LIBERO 默认值）
    obs_cam_keys: list[str] = field(
        default_factory=lambda: ["observation.images.image", "observation.images.image2"]
    )
    # 撤销 LIBERO 环境处理器额外施加的水平翻转，以匹配模型训练时的方向。
    image_hflip: bool = False
    # 相机潜变量布局："width_concat"（各相机沿宽度方向拼接；LIBERO）或
    # "robotwin_tshape"（全分辨率头部相机 + 半分辨率腕部相机排成"T"形；RoboTwin）。
    camera_layout: str = "width_concat"

    # 推理超参数（LIBERO 默认值）
    n_obs_steps: int = 1
    height: int = 128
    width: int = 128
    action_per_frame: int = 4
    frame_chunk_size: int = 4
    attn_window: int = 30
    num_inference_steps: int = 20
    video_exec_step: int = -1
    action_num_inference_steps: int = 50
    guidance_scale: float = 5.0
    action_guidance_scale: float = 1.0
    snr_shift: float = 5.0
    action_snr_shift: float = 0.05
    max_sequence_length: int = 512  # UMT5 prompt 长度

    # 基准测试实际使用的 30 维动作空间子集（LIBERO = 7-DoF）。动作（反）归一化的
    # 分位数保存在检查点的 ``policy_postprocessor.json`` 中，而不是这里。
    used_action_channel_ids: list[int] = field(default_factory=lambda: list(range(7)))

    # 可选项：将预测的视频潜变量经 VAE 解码到 ``self.last_predicted_frames``，用于保存 MP4。
    save_predicted_video: bool = False

    # 归一化：此处使用 IDENTITY；图像会在策略/专用处理器步骤内部完成缩放 + VAE 编码，
    # 动作则在其中完成分位数（反）归一化。
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # 优化器 / 学习率调度器（训练用；按照上游 train.py 使用 AdamW + warmup-constant）
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000

    def __post_init__(self):
        super().__post_init__()
        if self.attn_mode not in ("torch", "flashattn", "flex"):
            raise ValueError(f"attn_mode must be one of 'torch', 'flashattn', 'flex'; got {self.attn_mode!r}")

    @property
    def chunk_size(self) -> int:
        """每个自回归分块产生的单步动作数量。"""
        return self.frame_chunk_size * self.action_per_frame

    @property
    def n_action_steps(self) -> int:
        """重新填充前执行的动作数量（即整个分块）。"""
        return self.chunk_size

    def validate_features(self) -> None:
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "LingBot-VA requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(len(self.used_action_channel_ids),)
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        # 上游采用线性 warmup 后接恒定学习率（warmup_constant_lambda）。
        return ConstantWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    @property
    def observation_delta_indices(self) -> list[int]:
        temporal_downsample = 4
        stride = max(1, self.action_per_frame // temporal_downsample)
        return list(range(0, self.frame_chunk_size * temporal_downsample * stride, stride))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
