#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
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

from collections.abc import Callable
from dataclasses import asdict

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import MultivariateNormal, TanhTransform, Transform, TransformedDistribution

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from ..pretrained import PreTrainedPolicy
from ..utils import get_device_from_parameters
from .configuration_gaussian_actor import GaussianActorConfig, is_image_feature

DISCRETE_DIMENSION_INDEX = -1  # 夹爪始终是最后一个维度


class GaussianActorPolicy(
    PreTrainedPolicy,
):
    config_class = GaussianActorConfig
    name = "gaussian_actor"

    def __init__(
        self,
        config: GaussianActorConfig | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # 确定动作维度并初始化所有组件
        continuous_action_dim = config.output_features[ACTION].shape[0]
        self._init_encoders()
        self._init_actor(continuous_action_dim)
        self._init_discrete_critic()

    def get_optim_params(self) -> dict:
        optim_params = {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
        }
        return optim_params

    def reset(self):
        """重置策略"""
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """根据环境观测预测一段动作序列。"""
        raise NotImplementedError(
            "GaussianActorPolicy does not support action chunking. It returns single actions!"
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """为推理/评估选择动作"""

        observations_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            observations_features = self.actor.encoder.get_cached_image_features(batch)

        actions, _, _ = self.actor(batch, observations_features)

        if self.config.num_discrete_actions is not None:
            if self.discrete_critic is not None:
                discrete_action_value = self.discrete_critic(batch, observations_features)
                discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
            else:
                discrete_action = torch.ones(
                    (*actions.shape[:-1], 1), device=actions.device, dtype=actions.dtype
                )
            actions = torch.cat([actions, discrete_action], dim=-1)

        return actions

    def forward(self, batch: dict[str, Tensor | dict[str, Tensor]]) -> dict[str, Tensor]:
        """Actor 前向传播：采样动作并返回对数概率。

        Args:
            batch: 扁平的观测字典，或包含 ``"state"``（观测）以及可选的
                ``"observation_feature"``（预先计算的编码器特征）的训练字典。

        Returns:
            包含 ``"action"``、``"log_prob"`` 和 ``"action_mean"`` 张量的字典。
        """
        observations = batch.get("state", batch)
        observation_features = batch.get("observation_feature") if isinstance(batch, dict) else None
        actions, log_probs, means = self.actor(observations, observation_features)
        return {"action": actions, "log_prob": log_probs, "action_mean": means}

    def _init_encoders(self):
        """为 actor 和 critic 初始化共享或独立的编码器。"""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = GaussianActorObservationEncoder(self.config)
        self.encoder_actor = (
            self.encoder_critic if self.shared_encoder else GaussianActorObservationEncoder(self.config)
        )

    def _init_actor(self, continuous_action_dim):
        """初始化策略 actor 网络。"""
        # 注意：actor 只选择连续动作部分
        self.actor = Policy(
            encoder=self.encoder_actor,
            network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )

    def _init_discrete_critic(self) -> None:
        """初始化离散 critic 网络。"""
        if self.config.num_discrete_actions is None:
            self.discrete_critic = None
            return

        # TODO(Khalil)：编译离散 critic
        self.discrete_critic = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )


class GaussianActorObservationEncoder(nn.Module):
    """对图像和/或状态向量观测进行编码。"""

    def __init__(self, config: GaussianActorConfig) -> None:
        super().__init__()
        self.config = config
        self._init_image_layers()
        self._init_state_layers()
        self._compute_output_dim()

    def _init_image_layers(self) -> None:
        self.image_keys = [k for k in self.config.input_features if is_image_feature(k)]
        self.has_images = bool(self.image_keys)
        if not self.has_images:
            return

        if self.config.vision_encoder_name is not None:
            self.image_encoder = PretrainedImageEncoder(self.config)
        else:
            self.image_encoder = DefaultImageEncoder(self.config)

        if self.config.freeze_vision_encoder:
            freeze_image_encoder(self.image_encoder)

        dummy = torch.zeros(1, *self.config.input_features[self.image_keys[0]].shape)
        with torch.no_grad():
            _, channels, height, width = self.image_encoder(dummy).shape

        self.spatial_embeddings = nn.ModuleDict()
        self.post_encoders = nn.ModuleDict()

        for key in self.image_keys:
            name = key.replace(".", "_")
            self.spatial_embeddings[name] = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channels,
                num_features=self.config.image_embedding_pooling_dim,
            )
            self.post_encoders[name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(
                    in_features=channels * self.config.image_embedding_pooling_dim,
                    out_features=self.config.latent_dim,
                ),
                nn.LayerNorm(normalized_shape=self.config.latent_dim),
                nn.Tanh(),
            )

    def _init_state_layers(self) -> None:
        self.has_env = OBS_ENV_STATE in self.config.input_features
        self.has_state = OBS_STATE in self.config.input_features
        if self.has_env:
            dim = self.config.input_features[OBS_ENV_STATE].shape[0]
            self.env_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )
        if self.has_state:
            dim = self.config.input_features[OBS_STATE].shape[0]
            self.state_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )

    def _compute_output_dim(self) -> None:
        out = 0
        if self.has_images:
            out += len(self.image_keys) * self.config.latent_dim
        if self.has_env:
            out += self.config.latent_dim
        if self.has_state:
            out += self.config.latent_dim
        self._out_dim = out

    def forward(
        self, obs: dict[str, Tensor], cache: dict[str, Tensor] | None = None, detach: bool = False
    ) -> Tensor:
        parts = []
        if self.has_images:
            if cache is None:
                cache = self.get_cached_image_features(obs)
            parts.append(self._encode_images(cache, detach))
        if self.has_env:
            parts.append(self.env_encoder(obs[OBS_ENV_STATE]))
        if self.has_state:
            parts.append(self.state_encoder(obs[OBS_STATE]))
        if parts:
            return torch.cat(parts, dim=-1)

        raise ValueError(
            "No parts to concatenate, you should have at least one image or environment state or state"
        )

    def get_cached_image_features(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """从观测中提取图像特征，并可选择性地缓存。

        该函数将图像观测通过视觉编码器处理一次，并返回得到的特征。
        当图像编码器在 actor 与 critic 之间共享且被冻结时，这些特征可以安全地
        缓存并在各策略组件（actor、critic、discrete_critic）之间复用，
        避免冗余的前向传播。

        性能影响：
        - 视觉编码器的前向传播通常是训练和推理中的主要计算瓶颈
        - 缓存这些特征可以在训练和推理中带来 2-4 倍的加速

        使用场景：
        - 在 select_action() 中调用
        - 在 learner.py 的 get_observation_features() 中调用，为所有策略组件预计算特征
        - 在 forward() 内部调用

        Args:
            obs: 包含图像键的观测张量字典

        Returns:
            将图像键映射到其对应编码特征的字典
        """
        batched = torch.cat([obs[k] for k in self.image_keys], dim=0)
        out = self.image_encoder(batched)
        chunks = torch.chunk(out, len(self.image_keys), dim=0)
        return dict(zip(self.image_keys, chunks, strict=False))

    def _encode_images(self, cache: dict[str, Tensor], detach: bool) -> Tensor:
        """对缓存的观测进行图像特征编码。

        该函数从缓存中取出预先编码的图像特征，并应用空间嵌入与后编码器。
        如有需要，也支持对编码后的特征进行 detach。

        Args:
            cache (dict[str, Tensor]): 缓存的图像特征。
            detach (bool): 通常当编码器在 actor 与 critic 之间共享时，
            我们希望在策略侧对编码特征做 detach，以避免梯度回传穿过编码器。
            更多细节见 `https://cdn.aaai.org/ojs/17276/17276-13-20770-1-2-20210518.pdf`

        Returns:
            Tensor: 编码后的图像特征。
        """
        feats = []
        for k, feat in cache.items():
            safe_key = k.replace(".", "_")
            x = self.spatial_embeddings[safe_key](feat)
            x = self.post_encoders[safe_key](x)
            if detach:
                x = x.detach()
            feats.append(x)
        return torch.cat(feats, dim=-1)

    @property
    def output_dim(self) -> int:
        return self._out_dim


class MLP(nn.Module):
    """多层感知机构建器。

    根据 `hidden_dims` 动态构建层序列：
      1) Linear (in_dim -> out_dim)
      2) 当 `dropout_rate` > 0 且（不是最后一层或 `activate_final`）时，可选地添加 Dropout
      3) 对输出特征应用 LayerNorm
      4) 激活（中间层使用标准激活，若 `activate_final` 则最后一层使用 `final_activation`）

    Arguments:
        input_dim (int): 输入特征的维度大小。
        hidden_dims (list[int]): 各隐藏层的大小。
        activations (Callable or str): 层与层之间应用的激活函数。
        activate_final (bool): 是否在最后一层应用激活函数。
        dropout_rate (Optional[float]): 在归一化和激活之前应用的 Dropout 概率。
        final_activation (Optional[Callable or str]): 当 `activate_final` 为 True 时最后一层使用的激活函数。

    对于每一层，`in_dim` 会更新为上一层的 `out_dim`。所有构建的模块都作为
    `nn.Sequential` 容器存放在 `self.net` 中。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        total = len(hidden_dims)

        for idx, out_dim in enumerate(hidden_dims):
            # 1) 线性变换
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = idx == total - 1
            # 2-4) 可选地添加 dropout、归一化和激活
            if not is_last or activate_final:
                if dropout_rate and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.LayerNorm(out_dim))
                act_cls = final_activation if is_last and final_activation else activations
                act = act_cls if isinstance(act_cls, nn.Module) else getattr(nn, act_cls)()
                layers.append(act)

            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscreteCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 3,
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim

        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )

        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=self.output_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self, observations: torch.Tensor, observation_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}
        obs_enc = self.encoder(observations, cache=observation_features)
        return self.output_layer(self.net(obs_enc))


class Policy(nn.Module):
    def __init__(
        self,
        encoder: GaussianActorObservationEncoder,
        network: nn.Module,
        action_dim: int,
        std_min: float = -5,
        std_max: float = 2,
        fixed_std: torch.Tensor | None = None,
        init_final: float | None = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: GaussianActorObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
        self.fixed_std = fixed_std
        self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared

        # 找到最后一个 Linear 层的输出维度
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break
        # 均值层
        self.mean_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)

        # 标准差层或参数
        if fixed_std is None:
            self.std_layer = nn.Linear(out_features, action_dim)
            if init_final is not None:
                nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
            else:
                orthogonal_init()(self.std_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 如果编码器是共享的，则对其做 detach，以避免梯度回传穿过它
        # 这一点对防止编码器经由策略被更新非常重要
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # 获取网络输出
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)

        # 计算标准差
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)  # 与 JAX 的 "exp" 保持一致
            std = torch.clamp(std, self.std_min, self.std_max)  # 与 JAX 的默认 clip 保持一致
        else:
            std = self.fixed_std.expand_as(means)

        # 构建变换分布
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        # 采样动作（重参数化）
        actions = dist.rsample()

        # 计算 log_probs
        log_probs = dist.log_prob(actions)

        return actions, log_probs, means

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """从观测中获取编码后的特征"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations


class DefaultImageEncoder(nn.Module):
    def __init__(self, config: GaussianActorConfig):
        super().__init__()
        image_key = next(key for key in config.input_features if is_image_feature(key))
        self.image_enc_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=config.input_features[image_key].shape[0],
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=7,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=5,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.image_enc_layers(x)
        return x


def freeze_image_encoder(image_encoder: nn.Module):
    """冻结编码器中的所有参数"""
    for param in image_encoder.parameters():
        param.requires_grad = False


class PretrainedImageEncoder(nn.Module):
    def __init__(self, config: GaussianActorConfig):
        super().__init__()

        self.image_enc_layers, self.image_enc_out_shape = self._load_pretrained_vision_encoder(config)

    def _load_pretrained_vision_encoder(self, config: GaussianActorConfig):
        """构建 CNN 编码器"""
        from transformers import AutoModel

        self.image_enc_layers = AutoModel.from_pretrained(config.vision_encoder_name, trust_remote_code=True)

        if hasattr(self.image_enc_layers.config, "hidden_sizes"):
            self.image_enc_out_shape = self.image_enc_layers.config.hidden_sizes[-1]  # 最后一个通道维度
        elif hasattr(self.image_enc_layers, "fc"):
            self.image_enc_out_shape = self.image_enc_layers.fc.in_features
        else:
            raise ValueError("Unsupported vision encoder architecture, make sure you are using a CNN")
        return self.image_enc_layers, self.image_enc_out_shape

    def forward(self, x):
        enc_feat = self.image_enc_layers(x).last_hidden_state
        return enc_feat


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        可学习空间嵌入的 PyTorch 实现

        Args:
            height: 输入特征的空间高度
            width: 输入特征的空间宽度
            channel: 输入通道数
            num_features: 输出嵌入的维度数量
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        空间嵌入的前向传播

        Args:
            features: 形状为 [B, C, H, W] 的输入张量，其中 B 是批大小，
                     C 是通道数，H 是高度，W 是宽度
        Returns:
            形状为 [B, C*F] 的输出张量，其中 F 是特征数量
        """

        features_expanded = features.unsqueeze(-1)  # [B, C, H, W, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, C, H, W, F]

        # 逐元素相乘并在空间维度上归约
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # 对 H、W 维度求和

        # 重塑形状以合并通道维度与特征维度
        output = output.view(output.size(0), -1)  # [B, C*F]

        return output


class RescaleFromTanh(Transform):
    def __init__(self, low: float = -1, high: float = 1):
        super().__init__()

        self.low = low

        self.high = high

    def _call(self, x):
        # 从 (-1, 1) 重缩放到 (low, high)

        return 0.5 * (x + 1.0) * (self.high - self.low) + self.low

    def _inverse(self, y):
        # 从 (low, high) 重缩放回 (-1, 1)

        return 2.0 * (y - self.low) / (self.high - self.low) - 1.0

    def log_abs_det_jacobian(self, x, y):
        # log|d(rescale)/dx| = sum(log(0.5 * (high - low)))

        scale = 0.5 * (self.high - self.low)

        return torch.sum(torch.log(scale), dim=-1)


class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(self, loc, scale_diag, low=None, high=None):
        base_dist = MultivariateNormal(loc, torch.diag_embed(scale_diag))

        transforms = [TanhTransform(cache_size=1)]

        if low is not None and high is not None:
            low = torch.as_tensor(low)

            high = torch.as_tensor(high)

            transforms.insert(0, RescaleFromTanh(low, high))

        super().__init__(base_dist, transforms)

    def mode(self):
        # 众数即基础分布的均值，再经过各变换得到

        x = self.base_dist.mean

        for transform in self.transforms:
            x = transform(x)

        return x

    def stddev(self):
        std = self.base_dist.stddev

        x = std

        for transform in self.transforms:
            x = transform(x)

        return x
