#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
"""Action Chunking Transformer 策略

依据 Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware（https://huggingface.co/papers/2304.13705）。
这里的大部分改动涉及移除未使用的代码、统一命名以及添加有帮助的注释。
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from ..pretrained import PreTrainedPolicy
from .configuration_act import ACTConfig


class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer 策略，依据 Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware（论文：https://huggingface.co/papers/2304.13705，代码：https://github.com/tonyzhaozh/act）
    """

    config_class = ACTConfig
    name = "act"
    # FSDP2 包装单元：两个栈的每个 transformer 层各为一个单元。
    _fsdp_wrap_modules = ["ACTEncoderLayer", "ACTDecoderLayer"]

    def __init__(
        self,
        config: ACTConfig,
        **kwargs,
    ):
        """
        Args:
            config: 策略配置类实例，或为 None（此时使用配置类的默认实例化）。
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): 目前 lr_backbone == lr
        # 我们是否应该移除它，直接 `return self.parameters()`？
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """每当环境被重置时都应调用此方法。"""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """根据环境观测选择单个动作。

        此方法包装了 `select_actions`，以便每次返回一个动作供环境执行。
        它通过管理队列中的动作来工作，仅在队列为空时才调用 `select_actions`。
        """
        self.eval()  # 保持策略处于 eval 模式，因为在消费队列期间它可能被设为 train 模式

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # n_action_steps > 1 时的动作队列逻辑。当 action_queue 耗尽时，
        # 通过查询策略来填充它。
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` 返回 (batch_size, n_action_steps, action_dim) 张量，但队列
            # 的实际形状为 (n_action_steps, batch_size, *)，因此需要转置。
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """根据环境观测预测一个动作块。"""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝，以免添加键时修改原始字典
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """将批次通过模型运行，并计算训练或验证的损失。"""
        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝，以免添加键时修改原始字典
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae and log_sigma_x2_hat is not None:
            # 计算 Dₖₗ(latent_pdf || standard_normal)。注意：在独立计算每个维度的
            # KL 散度之后，我们沿潜在维度求和得到每个批次元素的总
            # KL 散度，然后对批次取均值。
            # （更多细节参见 https://huggingface.co/papers/1312.6114 的附录 B）。
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """如 https://huggingface.co/papers/2304.13705 的算法 2 所述的时间集成。

        权重按 wᵢ = exp(-temporal_ensemble_coeff * i) 计算，其中 w₀ 是最旧的动作。
        然后通过除以 Σwᵢ 将其归一化使总和为 1。以下是关于该系数
        如何工作的一些直觉：
            - 设为 0 时对所有动作均匀加权。
            - 设为正值时给较旧的动作更高的权重。
            - 设为负值时给较新的动作更高的权重。
        注意：原始 ACT 工作使用的 `temporal_ensemble_coeff` 默认值为 0.01。这
        使得较旧的动作比新动作权重更高（https://github.com/huggingface/lerobot/pull/319
        中记录的实验暗示了为什么给新动作过高权重可能有害：
        激进地这样做可能会削弱动作分块的好处）。

        这里我们使用在线方法计算平均值，而不是缓存动作历史
        以便离线计算平均值。对于一个简单的一维序列，它看起来像这样：

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # 离线计算
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # 在线计算
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """重置在线计算变量。"""
        self.ensembled_actions = None
        # (chunk_size,) 记录序列中每个时间步的集成中包含多少个动作的计数。
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        接收 (batch, chunk_size, action_dim) 的动作序列，更新所有时间步的
        时间集成，并弹出/返回序列中的下一批动作。
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # 将 `self._ensembled_action` 初始化为回合第一个时间步
            # 预测的动作序列。
            self.ensembled_actions = actions.clone()
            # 注意：对最后一个维度进行 unsqueeze，以确保后续的张量
            # 运算可以正确广播。
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions 的形状为 (batch_size, chunk_size - 1, action_dim)。
            # 对这些条目计算在线更新。
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # 最后一个动作没有先前的在线平均值，需要拼接到末尾。
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "消费"第一个动作。
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer：ACTPolicy 的底层神经网络。

    注意：在这段代码中我们使用 `vae_encoder`、'encoder'、`decoder` 这些术语。含义如下。
        - `vae_encoder` 依据变分自编码器（VAE）相关文献，是模型中
          编码目标数据（动作序列）和条件（机器人关节空间）的部分。
        - 一个带有 `encoder`（不是 VAE 编码器）和 `decoder`（不是 VAE 解码器）的
          带交叉注意力的 transformer 被用作 VAE 解码器。对于这些术语，我们去掉了
          `vae_` 前缀，因为我们可以选择不使用变分目标来训练此模型（在这种情况下
          我们完全去掉 `vae_encoder`，此模型与 VAE 毫无关系）。

                                 Transformer
                                 推理时单独使用
                                 （训练期间充当
                                  VAE 解码器）
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTConfig):
        # BERT 风格的 VAE 编码器，输入 token 为 [cls, robot_state, *action_sequence]。
        # cls token 构成潜在分布的参数（形如 [*means, *log_variances]）。
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # 关节空间配置到隐藏维度的投影层。
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # 动作（关节空间目标）到隐藏维度的投影层。
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # 从 VAE 编码器输出到潜在分布参数空间的投影层。
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # VAE 编码器输入的固定正弦位置嵌入。为批次维度进行 unsqueeze。
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # 用于图像特征提取的骨干网络。
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # 注意：这里的假设是我们使用 ResNet 模型（因此 layer4 是最终的
            # 特征图）。
            # 注意：其 forward 方法返回一个字典：{"feature_map": output}。
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer（使用变分目标训练时充当 VAE 解码器）。
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer 编码器输入投影。token 的结构为
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)]。
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer 编码器位置嵌入。
        n_1d_tokens = 1  # 用于 latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer 解码器。
        # transformer 解码器的可学习位置嵌入（采用 DETR 对象查询的风格）。
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # transformer 解码器输出上的最终动作回归头。
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """与原始代码一致的 transformer 参数 Xavier 均匀初始化。"""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """通过 Action Chunking Transformer（带可选 VAE 编码器）的前向传播。

        `batch` 应具有以下结构：
        {
            [robot_state_feature]（可选）：(B, state_dim) 的机器人状态批次。

            [image_features]：(B, n_cameras, C, H, W) 的图像批次。
                和/或
            [env_state_feature]：(B, env_dim) 的环境状态批次。

            [action_feature]（可选，仅在使用 VAE 训练时）：(B, chunk_size, action dim) 的动作批次。
        }

        Returns:
            (B, chunk_size, action_dim) 的动作序列批次
            包含潜在 PDF 参数（mean, log(σ²)）的元组，两者均为 (B, L) 张量，
            其中 L 是潜在维度。
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # 准备 latent 作为 transformer 编码器的输入。
        if self.config.use_vae and ACTION in batch and self.training:
            # 准备 VAE 编码器的输入：[cls, *joint_space_configuration, *action_sequence]。
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # 准备固定的位置嵌入。
            # 注意：detach() 应该不是必需的，但为了与原始代码保持一致以防万一。
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # 准备 transformer 编码器的键填充掩码。根据是否使用输入状态，
            # 序列开头有 1 个或 2 个额外的 token（cls 和机器人状态）
            # False 表示不是填充 token。
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 或 2)

            # 通过 VAE 编码器进行前向传播以获取潜在 PDF 参数。
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # 选择 class token，形状为 (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # 这是 2log(sigma)。这样做是为了匹配原始实现。
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # 使用重参数化技巧对 latent 进行采样。
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # 不使用 VAE 编码器时，我们将 latent 设为全零。
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): 移除对 `.to` 的调用以加速前向传播；预先计算并使用 buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # 准备 transformer 编码器输入。
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # 机器人状态 token。
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # 环境状态 token。
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            # 对于图像列表，H 和 W 可能不同，但 H*W 是恒定的。
            # 注意：如果修改此部分，请在 MPS 设备上验证
            # 梯度保持稳定（无爆炸或 NaN）。
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # 将特征重排为 (sequence, batch, dim)。
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # 直接扩展，而不是累积后再拼接
                # 转换为列表以正确扩展
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # 沿序列维度堆叠所有 token。
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # 通过 transformer 模块进行前向传播。
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): 移除对 `device` 的调用；预先计算并使用 buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # 移回 (B, S, C)。
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """运行多个编码器层的便捷模块，后面可能跟随归一化。"""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # 前馈层。
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # 注意：[0] 用于只选择输出，而不是注意力权重
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """运行多个解码器层后接归一化的便捷模块。"""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # 前馈层。
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) 的输入 token 张量。
            encoder_out: (Encoder Sequence, B, C)，来自我们所交叉注意力的编码器
                最后一层的输出特征。
            encoder_pos_embed: (ES, 1, C)，键（来自编码器）的位置嵌入。
            decoder_pos_embed: (DS, 1, C)，查询（来自解码器）的位置嵌入。
        Returns:
            (DS, B, C) 的解码器输出特征张量。
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # 只选择输出，而不是注意力权重
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # 只选择输出，而不是注意力权重
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """如 Attention is All You Need 中的一维正弦位置嵌入。

    Args:
        num_positions: 所需的 token 位置数量。
    Returns: (num_positions, dimension) 的位置嵌入（第一个维度是批次维度）。

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 维度 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 维度 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """类似于 Attention Is All You Need 中提出的二维正弦位置嵌入。

    不同之处在于位置索引被归一化到 [0, 2π]（不完全是：垂直方向的下界是 1/H，
    水平方向的下界是 1/W）。
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: 嵌入的期望维度。
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # 正弦频率几何级数的逆"公比"。
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) 的二维特征图批次，用于为其生成嵌入。
        Returns:
            (1, C, H, W) 的对应正弦位置嵌入批次。
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # 注意：这些分别类似于 range(1, H+1) 和 range(1, W+1)，但在大多数实现中
        # 它们是 range(0, H) 和 range(0, W)。保持原样以匹配原始代码。
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "归一化"位置索引，使其范围在 [0, 2π] 内。
        # 注意：在分母上加 epsilon 应该是不必要的，因为根据构造，
        # y_embed 和 x_range 的所有值都非零。这是原始代码的遗留产物。
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # 注意：这个先堆叠再展平的操作产生了交错的 sin 和 cos 项。
        # pos_embed_x 和 pos_embed_y 为 (1, H, W, C // 2)。
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """根据字符串返回激活函数。"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
