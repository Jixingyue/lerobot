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

import math
from collections.abc import Callable, Iterator
from dataclasses import asdict
from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.optim import Optimizer

from lerobot.lerobot_types import BatchType
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import (
    DISCRETE_DIMENSION_INDEX,
    MLP,
    DiscreteCritic,
    GaussianActorObservationEncoder,
    GaussianActorPolicy,
    orthogonal_init,
)
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.constants import ACTION
from lerobot.utils.transition import move_state_dict_to_device

from ..base import RLAlgorithm
from ..configs import TrainingStats
from .configuration_sac import SACAlgorithmConfig


class SACAlgorithm(RLAlgorithm):
    """Soft Actor-Critic。持有 critic、目标网络、温度参数，并负责损失计算。"""

    config_class = SACAlgorithmConfig
    name = "sac"

    def __init__(
        self,
        policy: GaussianActorPolicy,
        config: SACAlgorithmConfig,
    ):
        self.config = config
        self.policy_config = config.policy_config
        self.policy = policy
        self.optimizers: dict[str, Optimizer] = {}
        self._optimization_step: int = 0

        action_dim = self.policy.config.output_features[ACTION].shape[0]
        self._init_critics(action_dim)
        self._init_temperature(action_dim)

        self._device = torch.device(self.policy.config.device)
        self._move_to_device()

    def _init_critics(self, action_dim) -> None:
        """构建 critic 集成和目标网络。"""
        encoder = self.policy.encoder_critic

        heads = [
            CriticHead(
                input_dim=encoder.output_dim + action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(encoder=encoder, ensemble=heads)
        target_heads = [
            CriticHead(
                input_dim=encoder.output_dim + action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=encoder, ensemble=target_heads)
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        # TODO(Khalil): 调查并修复 torch.compile
        # 注意：torch.compile 已被禁用，启用时策略无法收敛。
        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        self.discrete_critic_target = None
        if self.policy_config.num_discrete_actions is not None:
            self.discrete_critic_target = self._init_discrete_critic_target(encoder)

    def _init_discrete_critic_target(self, encoder: GaussianActorObservationEncoder) -> DiscreteCritic:
        """构建离散 critic 的目标网络（主网络由策略持有）。"""
        discrete_critic_target = DiscreteCritic(
            encoder=encoder,
            input_dim=encoder.output_dim,
            output_dim=self.policy_config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )
        # TODO(Khalil): 对离散 critic 进行编译（torch.compile）
        discrete_critic_target.load_state_dict(self.policy.discrete_critic.state_dict())
        return discrete_critic_target

    def _init_temperature(self, continuous_action_dim: int) -> None:
        """设置温度参数（log_alpha）和目标熵。"""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            total_action_dim = continuous_action_dim + (
                1 if self.policy_config.num_discrete_actions is not None else 0
            )
            self.target_entropy = -total_action_dim / 2

    def _move_to_device(self) -> None:
        self.policy.to(self._device)
        self.critic_ensemble.to(self._device)
        self.critic_target.to(self._device)
        self.log_alpha = nn.Parameter(self.log_alpha.data.to(self._device))
        if self.discrete_critic_target is not None:
            self.discrete_critic_target.to(self._device)

    @property
    def temperature(self) -> float:
        """返回当前温度值，始终与 log_alpha 保持同步。"""
        return self.log_alpha.exp().item()

    def _critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """对 critic 网络集成进行前向传播

        Args:
            observations: 观测字典
            actions: 动作张量
            use_target: 若为 True，使用目标 critic；否则使用集成 critic

        Returns:
            所有 critic 输出的 Q 值张量
        """

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features)
        return q_values

    def _discrete_critic_forward(
        self, observations, use_target=False, observation_features=None
    ) -> torch.Tensor:
        """对离散 critic 网络进行前向传播

        Args:
            observations: 观测字典
            use_target: 若为 True，使用目标 critic；否则使用集成 critic
            observation_features: 可选的预计算观测特征，以避免重复计算编码器输出

        Returns:
            离散 critic 网络输出的 Q 值张量
        """
        discrete_critic = self.discrete_critic_target if use_target else self.policy.discrete_critic
        q_values = discrete_critic(observations, observation_features)
        return q_values

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """执行一次 SAC 训练步骤（critic / 离散 critic / actor / 温度）。

        从 ``batch_iterator`` 中取出 ``utd_ratio`` 个批次，计算相应的
        损失，对每个损失进行反向传播，并更新目标网络。

        Args:
            batch_iterator: 生成的每个批次包含
                - ``action``：动作张量
                - ``reward``：奖励张量
                - ``state``：观测张量字典
                - ``next_state``：下一观测张量字典
                - ``done``：完成掩码张量
                - ``observation_feature``：可选的预计算观测特征
                - ``next_observation_feature``：可选的预计算下一观测特征
                - ``complementary_info``（可选）：每步附加信息，如离散惩罚项

        Returns:
            包含各组件损失和梯度范数的 TrainingStats。
        """
        clip = self.config.grad_clip_norm

        for _ in range(self.config.utd_ratio - 1):
            batch = next(batch_iterator)
            fb = self._prepare_forward_batch(batch, include_complementary_info=True)

            loss_critic = self._compute_loss_critic(fb)
            self.optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_ensemble.parameters(), max_norm=clip)
            self.optimizers["critic"].step()

            if self.policy_config.num_discrete_actions is not None:
                loss_dc = self._compute_loss_discrete_critic(fb)
                self.optimizers["discrete_critic"].zero_grad()
                loss_dc.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.discrete_critic.parameters(), max_norm=clip)
                self.optimizers["discrete_critic"].step()

            self._update_target_networks()

        batch = next(batch_iterator)
        fb = self._prepare_forward_batch(batch, include_complementary_info=False)

        loss_critic = self._compute_loss_critic(fb)
        self.optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(self.critic_ensemble.parameters(), max_norm=clip).item()
        self.optimizers["critic"].step()

        stats = TrainingStats(
            losses={"loss_critic": loss_critic.item()},
            grad_norms={"critic": critic_grad},
        )

        if self.policy_config.num_discrete_actions is not None:
            loss_dc = self._compute_loss_discrete_critic(fb)
            self.optimizers["discrete_critic"].zero_grad()
            loss_dc.backward()
            dc_grad = torch.nn.utils.clip_grad_norm_(
                self.policy.discrete_critic.parameters(), max_norm=clip
            ).item()
            self.optimizers["discrete_critic"].step()
            stats.losses["loss_discrete_critic"] = loss_dc.item()
            stats.grad_norms["discrete_critic"] = dc_grad

        if self._optimization_step % self.config.policy_update_freq == 0:
            for _ in range(self.config.policy_update_freq):
                loss_actor = self._compute_loss_actor(fb)
                self.optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), max_norm=clip
                ).item()
                self.optimizers["actor"].step()

                loss_temp = self._compute_loss_temperature(fb)
                self.optimizers["temperature"].zero_grad()
                loss_temp.backward()
                temp_grad = torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=clip).item()
                self.optimizers["temperature"].step()

            stats.losses["loss_actor"] = loss_actor.item()
            stats.losses["loss_temperature"] = loss_temp.item()
            stats.grad_norms["actor"] = actor_grad
            stats.grad_norms["temperature"] = temp_grad
            stats.extra["temperature"] = self.temperature

        self._update_target_networks()
        self._optimization_step += 1
        return stats

    def _compute_loss_critic(self, batch: dict[str, Any]) -> Tensor:
        # 从批次中提取公共组件
        observations = batch["state"]
        actions = batch[ACTION]
        observation_features = batch.get("observation_feature")
        # 提取 critic 专用的组件
        rewards = batch["reward"]
        next_observations = batch["next_state"]
        done = batch["done"]
        next_observation_features = batch.get("next_observation_feature")

        with torch.no_grad():
            next_action_preds, next_log_probs, _ = self.policy.actor(
                next_observations, next_observation_features
            )

            # 2- 计算 q 目标
            q_targets = self._critic_forward(
                observations=next_observations,
                actions=next_action_preds,
                use_target=True,
                observation_features=next_observation_features,
            )

            # 在使用高 UTD（更新-数据比）时对 critic 进行子采样，以防止过拟合
            # TODO: 在前向传播之前获取索引，以避免不必要的计算
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            # critic 子采样大小
            min_q, _ = q_targets.min(dim=0)  # 从 min 操作中获取值
            if self.config.use_backup_entropy:
                min_q = min_q - (self.temperature * next_log_probs)

            td_target = rewards + (1 - done) * self.config.discount * min_q

        # 3- 计算预测的 q 值
        if self.policy_config.num_discrete_actions is not None:
            # 注意：我们只想保留连续动作部分
            # 在缓冲区中我们拥有完整的动作空间（连续 + 离散）
            # 在将它们拼接进 critic 前向传播之前，需要先进行拆分
            actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
        q_preds = self._critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        # 4- 计算损失
        # 为集成中的所有 Q 函数计算状态-动作值损失（TD 损失）。
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # 先计算每个 critic 的批次平均损失，然后将它们求和得到最终损失
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()
        return critics_loss

    def _compute_loss_discrete_critic(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        actions = batch[ACTION]
        rewards = batch["reward"]
        next_observations = batch["next_state"]
        done = batch["done"]
        observation_features = batch.get("observation_feature")
        next_observation_features = batch.get("next_observation_feature")
        complementary_info = batch.get("complementary_info")

        # 注意：我们只想保留离散动作部分
        # 在缓冲区中我们拥有完整的动作空间（连续 + 离散）
        # 在将它们拼接进 critic 前向传播之前，需要先进行拆分
        actions_discrete: Tensor = actions[:, DISCRETE_DIMENSION_INDEX:].clone()
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()

        discrete_penalties: Tensor | None = None
        if complementary_info is not None:
            discrete_penalties = complementary_info.get("discrete_penalty")

        with torch.no_grad():
            # 对于 DQN，使用在线网络选择动作，使用目标网络进行评估
            next_discrete_qs = self._discrete_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )
            best_next_discrete_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)

            # 从目标网络获取目标 Q 值
            target_next_discrete_qs = self._discrete_critic_forward(
                observations=next_observations,
                use_target=True,
                observation_features=next_observation_features,
            )

            # 使用 gather 选出最优动作对应的 Q 值
            target_next_discrete_q = torch.gather(
                target_next_discrete_qs, dim=1, index=best_next_discrete_action
            ).squeeze(-1)

            # 使用 Bellman 方程计算目标 Q 值
            rewards_discrete = rewards
            if discrete_penalties is not None:
                rewards_discrete = rewards + discrete_penalties
            target_discrete_q = rewards_discrete + (1 - done) * self.config.discount * target_next_discrete_q

        # 获取当前观测的预测 Q 值
        predicted_discrete_qs = self._discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )

        # 使用 gather 选出所执行动作对应的 Q 值
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=1, index=actions_discrete).squeeze(-1)

        # 计算预测 Q 值与目标 Q 值之间的 MSE 损失
        discrete_critic_loss = F.mse_loss(input=predicted_discrete_q, target=target_discrete_q)
        return discrete_critic_loss

    def _compute_loss_actor(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        observation_features = batch.get("observation_feature")

        actions_pi, log_probs, _ = self.policy.actor(observations, observation_features)

        q_preds = self._critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]

        actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()
        return actor_loss

    def _compute_loss_temperature(self, batch: dict[str, Any]) -> Tensor:
        """计算温度损失"""
        observations = batch["state"]
        observation_features = batch.get("observation_feature")

        # 计算温度损失
        with torch.no_grad():
            _, log_probs, _ = self.policy.actor(observations, observation_features)

        temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
        return temperature_loss

    def _update_target_networks(self) -> None:
        """使用指数移动平均更新目标网络"""
        for target_p, p in zip(
            self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=True
        ):
            target_p.data.copy_(
                p.data * self.config.critic_target_update_weight
                + target_p.data * (1.0 - self.config.critic_target_update_weight)
            )
        if self.policy_config.num_discrete_actions is not None:
            for target_p, p in zip(
                self.discrete_critic_target.parameters(),
                self.policy.discrete_critic.parameters(),
                strict=True,
            ):
                target_p.data.copy_(
                    p.data * self.config.critic_target_update_weight
                    + target_p.data * (1.0 - self.config.critic_target_update_weight)
                )

    def _prepare_forward_batch(
        self, batch: BatchType, *, include_complementary_info: bool = True
    ) -> dict[str, Any]:
        observations = batch["state"]
        next_observations = batch["next_state"]
        observation_features, next_observation_features = self.get_observation_features(
            observations, next_observations
        )
        forward_batch: dict[str, Any] = {
            ACTION: batch[ACTION],
            "reward": batch["reward"],
            "state": observations,
            "next_state": next_observations,
            "done": batch["done"],
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
        }
        if include_complementary_info and "complementary_info" in batch:
            forward_batch["complementary_info"] = batch["complementary_info"]
        return forward_batch

    def make_optimizers_and_scheduler(self) -> dict[str, Optimizer]:
        """
        为强化学习策略的 actor、critic 和温度组件创建并返回优化器。

        该函数为以下组件设置 Adam 优化器：
        - **actor 网络**，确保只优化相关的参数。
        - **critic 集成**，用于评估价值函数。
        - **温度参数**，用于控制 soft actor-critic（SAC）类方法中的熵。

        它还会初始化一个学习率调度器，不过目前设为 `None`。

        注意：
        - 如果编码器是共享的，其参数会被排除在 actor 的优化过程之外。
        - 策略的对数温度（`log_alpha`）被包装在列表中，以确保作为独立张量被正确优化。

        Args:
            cfg: 包含超参数的配置对象。
            policy (nn.Module): 包含 actor、critic 和温度组件的策略模型。

        Returns:
            将组件名称（"actor"、"critic"、"temperature"）映射到
            各自 Adam 优化器的字典。
        """
        actor_params = self.policy.get_optim_params()["actor"]
        self.optimizers = {
            "actor": torch.optim.Adam(actor_params, lr=self.config.actor_lr),
            "critic": torch.optim.Adam(self.critic_ensemble.parameters(), lr=self.config.critic_lr),
            "temperature": torch.optim.Adam([self.log_alpha], lr=self.config.temperature_lr),
        }
        if self.policy_config.num_discrete_actions is not None:
            self.optimizers["discrete_critic"] = torch.optim.Adam(
                self.policy.discrete_critic.parameters(), lr=self.config.critic_lr
            )
        return self.optimizers

    def get_optimizers(self) -> dict[str, Optimizer]:
        return self.optimizers

    def get_weights(self) -> dict[str, Any]:
        """发送 actor + 离散 critic 的 state dict。"""
        state_dicts: dict[str, Any] = {
            "policy": move_state_dict_to_device(self.policy.actor.state_dict(), device="cpu"),
        }
        if self.policy_config.num_discrete_actions is not None:
            state_dicts["discrete_critic"] = move_state_dict_to_device(
                self.policy.discrete_critic.state_dict(), device="cpu"
            )
        return state_dicts

    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        """将 actor + 离散 critic 的权重加载到策略中。"""
        actor_sd = move_state_dict_to_device(weights["policy"], device=device)
        self.policy.actor.load_state_dict(actor_sd)
        if "discrete_critic" in weights and self.policy.discrete_critic is not None:
            discrete_sd = move_state_dict_to_device(weights["discrete_critic"], device=device)
            self.policy.discrete_critic.load_state_dict(discrete_sd)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """算法持有的可训练张量。

        编码器权重会被剔除，因为它们由策略持有
        （``policy.encoder_critic``），并且已通过 ``policy.save_pretrained`` 保存。
        """
        bundle: dict[str, torch.Tensor] = {}
        for k, v in _strip_encoder_keys(self.critic_ensemble.state_dict()).items():
            bundle[f"critic_ensemble.{k}"] = v
        for k, v in _strip_encoder_keys(self.critic_target.state_dict()).items():
            bundle[f"critic_target.{k}"] = v
        if self.discrete_critic_target is not None:
            for k, v in _strip_encoder_keys(self.discrete_critic_target.state_dict()).items():
                bundle[f"discrete_critic_target.{k}"] = v
        bundle["log_alpha"] = self.log_alpha.detach()
        return bundle

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
    ) -> None:
        """原地加载算法持有的张量。

        ``log_alpha`` 通过 ``Parameter.data.copy_`` 恢复，以确保
        ``temperature`` 优化器对该参数对象的引用在恢复之后
        仍然有效。
        """
        critic_ensemble_state = _split_prefix(state_dict, "critic_ensemble.")
        critic_target_state = _split_prefix(state_dict, "critic_target.")
        self.critic_ensemble.load_state_dict(critic_ensemble_state, strict=False)
        self.critic_target.load_state_dict(critic_target_state, strict=False)

        if self.discrete_critic_target is not None:
            discrete_target_state = _split_prefix(state_dict, "discrete_critic_target.")
            self.discrete_critic_target.load_state_dict(discrete_target_state, strict=False)

        if "log_alpha" in state_dict:
            self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.log_alpha.device))

    def get_observation_features(
        self, observations: Tensor, next_observations: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """
        从策略编码器获取观测特征。它充当观测特征的缓存。
        当编码器被冻结时，观测特征不会更新。
        通过缓存观测特征可以节省计算量。

        Args:
            policy: 策略模型
            observations: 当前观测
            next_observations: 下一观测

        Returns:
            tuple: observation_features, next_observation_features
        """

        if self.policy.config.vision_encoder_name is None or not self.policy.config.freeze_vision_encoder:
            return None, None

        with torch.no_grad():
            observation_features = self.policy.actor.encoder.get_cached_image_features(observations)
            next_observation_features = self.policy.actor.encoder.get_cached_image_features(next_observations)

        return observation_features, next_observation_features


def _strip_encoder_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """从 critic 模块的 state dict 中剔除 ``encoder.*`` 键。"""
    return {k: v for k, v in state.items() if not k.startswith("encoder.")}


def _split_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """返回 ``state`` 中键以 ``prefix`` 开头的子集，并去除前缀。"""
    return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}


class CriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x))


class CriticEnsemble(nn.Module):
    """
    CriticEnsemble 将多个 CriticHead 模块包装成一个集成。

    Args:
        encoder (GaussianActorObservationEncoder): 观测的编码器。
        ensemble (List[CriticHead]): critic 头列表。
        init_final (float | None): 可选的最后一层初始化器缩放系数。

    前向传播返回形状为 (num_critics, batch_size) 的张量，其中包含 Q 值。
    """

    def __init__(
        self,
        encoder: GaussianActorObservationEncoder,
        ensemble: list[CriticHead],
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        # 将 observations 中的每个张量移动到设备上
        observations = {k: v.to(device) for k, v in observations.items()}

        obs_enc = self.encoder(observations, cache=observation_features)

        inputs = torch.cat([obs_enc, actions], dim=-1)

        # 遍历各个 critic 并收集输出
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # 堆叠输出以匹配期望的形状 [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values
