#!/usr/bin/env python

# Copyright 2024 Seungjae Lee and Yibin Wang and Haritheja Etukuru
# and H. Jin Kim and Nur Muhammad Mahi Shafiullah and Lerrel Pinto
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

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, VQBeTSchedulerConfig


@PreTrainedConfig.register_subclass("vqbet")
@dataclass
class VQBeTConfig(PreTrainedConfig):
    """VQ-BeT 的配置类。

    默认值是针对使用 PushT（提供本体感知和单相机观测）训练而配置的。

    你最可能需要修改的参数是那些依赖于环境 / 传感器的参数，
    即：`input_features` 和 `output_features`。

    关于输入和输出的说明：
        - 必须以 "observation.state" 作为输入键。
        - 至少需要一个以 "observation.image" 开头的键作为输入。
        - 如果有多个以 "observation.image" 开头的键，它们会被视为多个相机
          视角。目前我们只支持所有图像具有相同形状。
        - 必须以 "action" 作为输出键。

    Args:
        n_obs_steps: 传给策略的观测所覆盖的环境步数（包含当前步以及向前回溯的若干步）。
        n_action_pred_token: VQ-BeT 预测的当前 token 和未来 token 的总数。
        action_chunk_size: 每个动作预测 token 对应的动作块大小。
        input_features: 定义策略输入数据 PolicyFeature 的字典。键表示输入数据名称，值为
            PolicyFeature，由 FeatureType 和 shape 属性组成。
        output_features: 定义策略输出数据 PolicyFeature 的字典。键表示输出数据名称，值为
            PolicyFeature，由 FeatureType 和 shape 属性组成。
        normalization_mapping: 将 FeatureType 的 str 值（例如 "STATE"、"VISUAL"）映射到
            相应 NormalizationMode（例如 NormalizationMode.MIN_MAX）的字典。
        vision_backbone: 用于编码图像的 torchvision resnet 主干网络名称。
        crop_shape: 作为视觉主干预处理步骤、将图像裁剪成的 (H, W) 形状。必须能容纳在
            图像尺寸之内。如果为 None，则不进行裁剪。
        crop_is_random: 训练时裁剪是否随机（评估模式下始终为中心裁剪）。
        pretrained_backbone_weights: 用于初始化主干网络的 torchvision 预训练权重。
            `None` 表示不使用预训练权重。
        use_group_norm: 是否在主干网络中用 group normalization 替换 batch normalization。
            group 大小设置为约 16（准确地说是 feature_dim // 16）。
        spatial_softmax_num_keypoints: SpatialSoftmax 的关键点数量。
        n_vqvae_training_steps: 训练 Residual VQ 的优化步数。
        vqvae_n_embed: RVQ 码本中嵌入向量的数量（每一层）。
        vqvae_embedding_dim: RVQ 码本中每个嵌入向量的维度。
        vqvae_enc_hidden_dim: Residual VQ-VAE 编码器 / 解码器部分隐藏层维度的大小。
        gpt_block_size: minGPT 的最大块大小（应大于输入 token 的数量）。
        gpt_input_dim: GPT 输入维度的大小。这也用作观测特征的维度。
        gpt_output_dim: GPT 输出维度的大小。这也用作 offset / bin 预测头的输入维度。
        gpt_n_layer: GPT 的层数。
        gpt_n_head: GPT 的注意力头数。
        gpt_hidden_dim: GPT 隐藏层维度的大小。
        dropout: GPT 的 dropout 率。
        offset_loss_weight: 与 offset 损失相乘的常数。
        primary_code_loss_weight: 与主码（primary code）预测损失相乘的常数。
        secondary_code_loss_weight: 与次码（secondary code）预测损失相乘的常数。
        bet_softmax_temperature: 使用 VQ-BeT 进行 rollout 时代码采样的温度。
        sequentially_select: 主码 / 次码是顺序选择（先选主码，再选次码），
            还是同时选择。
    """

    # 输入 / 输出结构。
    n_obs_steps: int = 5
    n_action_pred_token: int = 3
    action_chunk_size: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # 架构 / 建模。
    # 视觉主干网络。
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    # VQ-VAE
    n_vqvae_training_steps: int = 20000
    vqvae_n_embed: int = 16
    vqvae_embedding_dim: int = 256
    vqvae_enc_hidden_dim: int = 128
    # VQ-BeT
    gpt_block_size: int = 500
    gpt_input_dim: int = 512
    gpt_output_dim: int = 512
    gpt_n_layer: int = 8
    gpt_n_head: int = 8
    gpt_hidden_dim: int = 512
    dropout: float = 0.1
    offset_loss_weight: float = 10000.0
    primary_code_loss_weight: float = 5.0
    secondary_code_loss_weight: float = 0.5
    bet_softmax_temperature: float = 0.1
    sequentially_select: bool = False

    # 训练预设
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    optimizer_vqvae_lr: float = 1e-3
    optimizer_vqvae_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> VQBeTSchedulerConfig:
        return VQBeTSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_vqvae_training_steps=self.n_vqvae_training_steps,
        )

    def validate_features(self) -> None:
        # 注意：该检查以前在 VQBeTRgbEncoder 内部以下面的断言形式执行：
        # assert len(image_keys) == 1
        if not len(self.image_features) == 1:
            raise ValueError("You must provide only one image among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # 检查所有输入图像是否具有相同的形状。
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, self.n_action_pred_token + self.action_chunk_size - 1))

    @property
    def reward_delta_indices(self) -> None:
        return None
