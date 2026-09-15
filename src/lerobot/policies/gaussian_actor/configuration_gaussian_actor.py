#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
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

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import MultiAdamConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def is_image_feature(key: str) -> bool:
    """检查某个特征键是否表示图像特征。

    Args:
        key: 待检查的特征键

    Returns:
        如果该键表示图像特征则返回 True，否则返回 False
    """
    return key.startswith(OBS_IMAGE)


@dataclass
class ConcurrencyConfig:
    """actor 与 learner 的并发方式配置。

    可选值为：
    - "threads"：为 actor 和 learner 使用线程。
    - "processes"：为 actor 和 learner 使用进程。

    当使用进程时，``multiprocessing_context`` 用于选择进程级的启动方式。
    将其设为 ``None`` 可保留 Python 的默认方式，或保留嵌入应用已选定的方式。
    """

    actor: str = "threads"
    learner: str = "threads"
    multiprocessing_context: str | None = "spawn"


@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2


@dataclass
class CriticNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    final_activation: str | None = None


@dataclass
class ActorNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True


@dataclass
class PolicyConfig:
    use_tanh_squash: bool = True
    std_min: float = 1e-5
    std_max: float = 10.0
    init_final: float = 0.05


@PreTrainedConfig.register_subclass("gaussian_actor")
@dataclass
class GaussianActorConfig(PreTrainedConfig):
    """高斯 actor 配置。

    用于配置高斯策略的策略侧（actor + 观测编码器），供 SAC 及相关的
    最大熵连续控制算法使用。默认情况下，actor 的输出是经过 tanh 压缩的
    对角高斯分布（``TanhMultivariateNormalDiag``）；可通过
    ``policy_kwargs.use_tanh_squash`` 禁用 tanh 压缩。critic、温度以及
    Bellman 更新逻辑位于算法侧（见 ``lerobot.rl.algorithms.sac``）。

    CLI：``--policy.type=gaussian_actor``。
    """

    # 特征类型到归一化模式的映射
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # 用于归一化各类输入的统计量
    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            OBS_IMAGE: {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            OBS_STATE: {
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            },
            ACTION: {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
            },
        }
    )

    # 架构相关细节
    # 运行模型的设备（例如 "cuda"、"cpu"）
    device: str = "cpu"
    # 存储模型的设备
    storage_device: str = "cpu"
    # 视觉编码器模型的名称（hil serl resnet10 请设为 "lerobot/resnet10"）
    vision_encoder_name: str | None = None
    # 训练期间是否冻结视觉编码器
    freeze_vision_encoder: bool = True
    # 图像编码器的隐藏层维度
    image_encoder_hidden_dim: int = 32
    # actor 与 critic 是否使用共享编码器
    shared_encoder: bool = True
    # 离散动作的数量，例如夹爪动作
    num_discrete_actions: int | None = None
    # 图像嵌入池化的维度
    image_embedding_pooling_dim: int = 8

    # 编码器架构
    # 状态编码器的隐藏层维度
    state_encoder_hidden_dim: int = 256
    # 潜在空间的维度
    latent_dim: int = 256

    # 在线训练（TODO(Khalil)：迁移到 TrainRLServerPipelineConfig）
    # 在线训练的步数
    online_steps: int = 1000000
    # 在线经验回放缓冲区的容量
    online_buffer_capacity: int = 100000
    # 离线经验回放缓冲区的容量
    offline_buffer_capacity: int = 100000
    # 是否为缓冲区使用异步预取
    async_prefetch: bool = False
    # 开始学习之前的步数
    online_step_before_learning: int = 100

    # Actor-learner 传输（TODO(Khalil)：迁移到 TrainRLServerPipelineConfig）。
    # actor-learner 架构的配置
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    # 并发设置（actor 和 learner 可以使用线程或进程）
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # 网络架构
    # actor 网络架构的配置
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    # 策略参数（高斯头）的配置
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    # 离散 critic 网络的配置
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    def __post_init__(self):
        super().__post_init__()
        # GaussianActor 配置相关的校验可在此处添加

    def get_optimizer_preset(self) -> MultiAdamConfig:
        # 这里的默认学习率只是为了满足 ``PreTrainedConfig`` 中抽象方法
        # ``get_optimizer_preset()`` 的约定。RL 训练实际使用的优化器由
        # ``SACAlgorithm.make_optimizers_and_scheduler()`` 基于
        # ``SACAlgorithmConfig.{actor_lr,critic_lr,temperature_lr}`` 构建，
        # 完全绕过此预设。
        default_lr = 3e-4
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": default_lr},
                "critic": {"lr": default_lr},
                "temperature": {"lr": default_lr},
            },
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        has_image = any(is_image_feature(key) for key in self.input_features)
        has_state = OBS_STATE in self.input_features

        if not (has_state or has_image):
            raise ValueError(
                "You must provide either 'observation.state' or an image observation (key starting with 'observation.image') in the input features"
            )

        if ACTION not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if is_image_feature(key)]

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return None  # SAC 通常一次只预测一个动作

    @property
    def reward_delta_indices(self) -> None:
        return None
