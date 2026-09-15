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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import (
    CriticNetworkConfig,
    GaussianActorConfig,
)

from ..configs import RLAlgorithmConfig


@RLAlgorithmConfig.register_subclass("sac")
@dataclass
class SACAlgorithmConfig(RLAlgorithmConfig):
    """Soft Actor-Critic（SAC）算法配置。

    SAC 是一种基于最大熵强化学习框架的离策略（off-policy）
    actor-critic 深度强化学习算法。它使用从环境中收集的经验，
    同时学习策略和 Q 函数。

    该配置类包含算法侧的超参数：critic 集成、目标网络、
    温度/熵调节以及 Bellman 更新循环。策略侧（actor + 观测编码器）
    位于 :class:`~lerobot.policies.gaussian_actor.GaussianActorConfig` 中，
    并通过 :attr:`policy_config` 引用。
    """

    # 优化器学习率
    # actor 网络的学习率
    actor_lr: float = 3e-4
    # critic 网络的学习率
    critic_lr: float = 3e-4
    # 温度参数的学习率
    temperature_lr: float = 3e-4

    # Bellman 更新
    # SAC 算法的折扣因子
    discount: float = 0.99
    # SAC 算法是否使用 backup 熵
    use_backup_entropy: bool = True
    # critic 目标网络更新的权重
    critic_target_update_weight: float = 0.005

    # Critic 集成
    # 集成中 critic 的数量
    num_critics: int = 2
    # 训练时子采样的 critic 数量
    num_subsample_critics: int | None = None
    # critic 网络架构的配置
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # 离散 critic 网络的配置
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    # 温度/熵
    # 初始温度值
    temperature_init: float = 1.0
    # 用于自动温度调节的目标熵。若为 ``None``，默认为
    # ``-|A|/2``，其中 ``|A|`` 是总动作维度（连续动作维度，
    # 若存在离散动作头则再 +1）。
    target_entropy: float | None = None

    # 更新循环
    # 更新-数据比（update-to-data ratio）。设为 >1 可在每个环境步进行额外的 critic 更新。
    utd_ratio: int = 1
    # 策略更新的频率
    policy_update_freq: int = 1
    # SAC 算法的梯度裁剪范数
    grad_clip_norm: float = 40.0

    # 优化选项
    # 目前默认禁用 torch.compile
    use_torch_compile: bool = False

    # 策略配置
    policy_config: PreTrainedConfig | None = None

    @classmethod
    def from_policy_config(cls, policy_cfg: GaussianActorConfig) -> SACAlgorithmConfig:
        """为给定策略构建使用默认超参数的算法配置。"""
        return cls(
            policy_config=policy_cfg,
            discrete_critic_network_kwargs=policy_cfg.discrete_critic_network_kwargs,
        )
