#!/usr/bin/env python

# Copyright 2024 Nicklas Hansen, Xiaolong Wang, Hao Su,
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
"""《在真实世界中微调离线世界模型》（Finetuning Offline World Models in the Real World）的实现。

本代码中的注释有时会引用以下参考文献：
    TD-MPC 论文：Temporal Difference Learning for Model Predictive Control (https://huggingface.co/papers/2203.04955)
    FOWM 论文：Finetuning Offline World Models in the Real World (https://huggingface.co/papers/2310.16029)
"""

# ruff: noqa: N806

from collections import deque
from collections.abc import Callable
from copy import deepcopy
from functools import partial

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_PREFIX, OBS_STATE, OBS_STR, REWARD

from ..pretrained import PreTrainedPolicy
from ..utils import get_device_from_parameters, get_output_shape, populate_queues
from .configuration_tdmpc import TDMPCConfig


class TDMPCPolicy(PreTrainedPolicy):
    """TD-MPC 训练 + 推理的实现。

    请注意关于该策略的若干提醒：
        - 使用原始 FOWM 代码（https://github.com/fyhMer/fowm）生成的预训练权重，
            其评估表现符合预期。准确地说：我们用 FOWM 代码针对 xarm_lift_medium_replay
            数据集训练并评估了一个模型，并将权重移植到了 LeRobot，能够以相同的成功率指标
            进行评估。但是，我们不得不借助进程间通信来使用 FOWM 中的 xarm 环境。这是因为
            我们的 xarm 环境使用了更新的依赖，与 FOWM 中的环境不匹配。实现细节参见
            https://github.com/huggingface/lerobot/pull/103。
        - 我们尚未验证在 LeRobot 上训练能否复现 FOWM 的结果。
        - 尽管如此，我们已经验证可以针对 PushT 训练 TD-MPC。参见
          `lerobot/configs/policy/tdmpc_pusht_keypoints.yaml`。
        - 我们当前的 xarm 数据集是使用 FOWM 中的环境生成的，因此与我们的 xarm 环境不匹配。
    """

    config_class = TDMPCConfig
    name = "tdmpc"

    def __init__(
        self,
        config: TDMPCConfig,
        **kwargs,
    ):
        """
        Args:
            config: 策略配置类实例；若为 None，则使用该配置类的默认实例化结果。
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = TDMPCTOLD(config)
        self.model_target = deepcopy(self.model)
        for param in self.model_target.parameters():
            param.requires_grad = False

        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """
        清空观测队列和动作队列，并清除用于 MPPI/CEM 热启动的上一次均值。
        应在 `env.reset()` 时调用。
        """
        self._queues = {
            OBS_STATE: deque(maxlen=1),
            ACTION: deque(maxlen=max(self.config.n_action_steps, self.config.n_action_repeats)),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGE] = deque(maxlen=1)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=1)
        # MPC 过程中使用的交叉熵方法（CEM）得到的上一次均值，用于为下一步的 CEM 热启动。
        self._prev_mean: torch.Tensor | None = None

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """根据环境观测预测一个动作块。"""
        batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch if key in self._queues}

        # 移除时间维度，因为目前尚不处理该维度。
        for key in batch:
            assert batch[key].shape[1] == 1
            batch[key] = batch[key][:, 0]

        # 注意：观测的顺序在这里很重要。
        encode_keys = []
        if self.config.image_features:
            encode_keys.append(OBS_IMAGE)
        if self.config.env_state_feature:
            encode_keys.append(OBS_ENV_STATE)
        encode_keys.append(OBS_STATE)
        z = self.model.encode({k: batch[k] for k in encode_keys})
        if self.config.use_mpc:  # noqa: SIM108
            actions = self.plan(z)  # (horizon, batch, action_dim)
        else:
            # 仅使用策略（π）进行规划。这里总是返回一个动作，因此用 unsqueeze 增加一个
            # 与 MPC 分支相同的序列维度。
            actions = self.model.pi(z).unsqueeze(0)

        actions = torch.clamp(actions, -1, +1)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """根据环境观测选择单个动作。"""
        # 注意：离线评估时 batch 中包含 action，需要将其弹出
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝，以免新增键时修改原始字典
            batch[OBS_IMAGE] = batch[next(iter(self.config.image_features))]
        # 注意：离线评估时 batch 中包含 action，需要将其弹出
        if ACTION in batch:
            batch.pop(ACTION)

        self._queues = populate_queues(self._queues, batch)

        # 当动作队列耗尽时，再次查询策略来填充它。
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)

            if self.config.n_action_repeats > 1:
                for _ in range(self.config.n_action_repeats):
                    self._queues[ACTION].append(actions[0])
            else:
                # 动作队列为 (n_action_steps, batch_size, action_dim)，因此这里对动作进行转置。
                self._queues[ACTION].extend(actions[: self.config.n_action_steps])

        action = self._queues[ACTION].popleft()
        return action

    @torch.no_grad()
    def plan(self, z: Tensor) -> Tensor:
        """使用 TD-MPC 推理规划动作序列。

        Args:
            z: 初始状态张量，形状 (batch, latent_dim,)。
        Returns:
            规划出的动作轨迹张量，形状 (horizon, batch, action_dim,)。
        """
        device = get_device_from_parameters(self)

        batch_size = z.shape[0]

        # 从策略采样 Nπ 条轨迹。
        pi_actions = torch.empty(
            self.config.horizon,
            self.config.n_pi_samples,
            batch_size,
            self.config.action_feature.shape[0],
            device=device,
        )
        if self.config.n_pi_samples > 0:
            _z = einops.repeat(z, "b d -> n b d", n=self.config.n_pi_samples)
            for t in range(self.config.horizon):
                # 注意：在推理时加入少量噪声并无害处，甚至可能对 CEM 有所帮助。
                pi_actions[t] = self.model.pi(_z, self.config.min_std)
                _z = self.model.latent_dynamics(_z, pi_actions[t])

        # 在 CEM 循环中，需要用它来对高斯采样得到的轨迹调用 estimate_value。
        z = einops.repeat(z, "b d -> n b d", n=self.config.n_gaussian_samples + self.config.n_pi_samples)

        # 模型预测路径积分（Model Predictive Path Integral，MPPI），以交叉熵方法（CEM）
        # 作为优化算法。
        # 交叉熵方法（CEM）的初始均值和标准差。
        mean = torch.zeros(
            self.config.horizon, batch_size, self.config.action_feature.shape[0], device=device
        )
        # 可能用上一步的均值为 CEM 热启动。
        if self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]
        std = self.config.max_std * torch.ones_like(mean)

        for _ in range(self.config.cem_iterations):
            # 从高斯分布中随机采样动作轨迹。
            std_normal_noise = torch.randn(
                self.config.horizon,
                self.config.n_gaussian_samples,
                batch_size,
                self.config.action_feature.shape[0],
                device=std.device,
            )
            gaussian_actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * std_normal_noise, -1, 1)

            # 计算精英动作。
            actions = torch.cat([gaussian_actions, pi_actions], dim=1)
            value = self.estimate_value(z, actions).nan_to_num_(0)
            elite_idxs = torch.topk(value, self.config.n_elites, dim=0).indices  # (n_elites, batch)
            elite_value = value.take_along_dim(elite_idxs, dim=0)  # (n_elites, batch)
            # (horizon, n_elites, batch, action_dim)
            elite_actions = actions.take_along_dim(einops.rearrange(elite_idxs, "n b -> 1 n b 1"), dim=1)

            # 将高斯 PDF 参数更新为精英样本的（加权）均值和标准差。
            max_value = elite_value.max(0, keepdim=True)[0]  # (1, batch)
            # 加权方式是对轨迹价值做 softmax。注意，这与 TD-MPC 论文公式 4 中 Ω 的用法不同；
            # 这里使用的是其归一化版本：s = Ω/ΣΩ。由此公式变为：
            # μ = Σ(s⋅Γ)，σ = Σ(s⋅(Γ-μ)²)。
            score = torch.exp(self.config.elite_weighting_temperature * (elite_value - max_value))
            score /= score.sum(axis=0, keepdim=True)
            # (horizon, batch, action_dim)
            _mean = torch.sum(einops.rearrange(score, "n b -> n b 1") * elite_actions, dim=1)
            _std = torch.sqrt(
                torch.sum(
                    einops.rearrange(score, "n b -> n b 1")
                    * (elite_actions - einops.rearrange(_mean, "h b d -> h 1 b d")) ** 2,
                    dim=1,
                )
            )
            # 均值使用指数移动平均更新，标准差则直接替换。
            mean = (
                self.config.gaussian_mean_momentum * mean + (1 - self.config.gaussian_mean_momentum) * _mean
            )
            std = _std.clamp_(self.config.min_std, self.config.max_std)

        # 记录均值，用于后续步骤的热启动。
        self._prev_mean = mean

        # 使用 MPPI/CEM 最后一次迭代得到的 softmax 分数，
        # 从最后一次迭代的精英动作中随机选取一个。
        actions = elite_actions[:, torch.multinomial(score.T, 1).squeeze(), torch.arange(batch_size)]

        return actions

    @torch.no_grad()
    def estimate_value(self, z: Tensor, actions: Tensor):
        """按照 FOWM 论文公式 4 估计一条轨迹的价值。

        Args:
            z: 初始潜状态张量，形状 (batch, latent_dim)。
            actions: 动作轨迹张量，形状 (horizon, batch, action_dim)。
        Returns:
            价值张量，形状 (batch,)。
        """
        # 初始化回报和累积折扣因子。
        G, running_discount = 0, 1
        # 遍历轨迹中的动作，利用潜空间动态模型模拟轨迹，并记录回报。
        for t in range(actions.shape[0]):
            # 稍后再计算奖励。首先计算 FOWM 论文公式 4 中的不确定性正则化项。
            if self.config.uncertainty_regularizer_coeff > 0:
                regularization = -(
                    self.config.uncertainty_regularizer_coeff * self.model.Qs(z, actions[t]).std(0)
                )
            else:
                regularization = 0
            # 估计下一个状态（潜变量）和奖励。
            z, reward = self.model.latent_dynamics_and_reward(z, actions[t])
            # 更新回报和累积折扣。
            G += running_discount * (reward + regularization)
            running_discount *= self.config.discount
        # 加上最终状态的估计价值（使用最小值以得到保守估计）。
        # 做法是先预测下一个动作，再在状态-动作价值估计器集成上取最小值。
        # 注意：在 xarm_lift_medium_replay 上 50 个回合的成功率指标表明，
        # 推理时加入这少量噪声似乎略有帮助。
        next_action = self.model.pi(z, self.config.min_std)  # (batch, action_dim)
        terminal_values = self.model.Qs(z, next_action)  # (ensemble, batch)
        # 随机选取 2 个 Q 用于终止价值估计（如 FOWM 论文附录 C 所述）。
        if self.config.q_ensemble_size > 2:
            G += (
                running_discount
                * torch.min(terminal_values[torch.randint(0, self.config.q_ensemble_size, size=(2,))], dim=0)[
                    0
                ]
            )
        else:
            G += running_discount * torch.min(terminal_values, dim=0)[0]
        # 最后，也对终止价值进行正则化。
        if self.config.uncertainty_regularizer_coeff > 0:
            G -= running_discount * self.config.uncertainty_regularizer_coeff * terminal_values.std(0)
        return G

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """将批次输入模型并计算损失。

        返回一个字典，其中损失为张量，其他信息为 Python 原生浮点数。
        """
        device = get_device_from_parameters(self)

        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝，以免新增键时修改原始字典
            batch[OBS_IMAGE] = batch[next(iter(self.config.image_features))]

        info = {}

        # (b, t) -> (t, b)
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        action = batch[ACTION]  # (t, b, action_dim)
        reward = batch[REWARD]  # (t, b)
        observations = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}

        # 施加随机图像增强。
        if self.config.image_features and self.config.max_random_shift_ratio > 0:
            observations[OBS_IMAGE] = flatten_forward_unflatten(
                partial(random_shifts_aug, max_random_shift_ratio=self.config.max_random_shift_ratio),
                observations[OBS_IMAGE],
            )

        # 获取用于预测轨迹的当前观测，以及用于潜变量一致性损失和 TD 损失的所有未来观测。
        current_observation, next_observations = {}, {}
        for k in observations:
            current_observation[k] = observations[k][0]
            next_observations[k] = observations[k][1:]
        horizon, batch_size = next_observations[
            OBS_IMAGE if self.config.image_features else OBS_ENV_STATE
        ].shape[:2]

        # 使用潜空间动态模型和策略模型进行潜变量展开（latent rollout）。
        # 注意其形状为 `horizon+1`，因为有 `horizon` 个动作和一个当前 `z`；
        # 每个动作都会给出下一个 `z`。
        batch_size = batch["index"].shape[0]
        z_preds = torch.empty(horizon + 1, batch_size, self.config.latent_dim, device=device)
        z_preds[0] = self.model.encode(current_observation)
        reward_preds = torch.empty_like(reward, device=device)
        for t in range(horizon):
            z_preds[t + 1], reward_preds[t] = self.model.latent_dynamics_and_reward(z_preds[t], action[t])

        # 基于潜变量展开计算 Q 和 V 价值预测。
        q_preds_ensemble = self.model.Qs(z_preds[:-1], action)  # (ensemble, horizon, batch)
        v_preds = self.model.V(z_preds[:-1])
        info.update({"Q": q_preds_ensemble.mean().item(), "V": v_preds.mean().item()})

        # 使用 stopgrad 计算各类目标值。
        with torch.no_grad():
            # 潜状态一致性目标。
            z_targets = self.model_target.encode(next_observations)
            # 状态-动作价值目标（或称 TD 目标），如 FOWM 公式 3 所示。TD-MPC 将学习到的
            # 状态-动作价值函数与学习到的策略结合使用：Q(z, π(z))，而 FOWM 使用学习到的
            # 状态价值函数：V(z)。这意味着 TD 目标只依赖于样本内动作（而非 π 估计的动作）。
            # 注意：这里没有使用 self.model_target，而是使用 self.model，以遵循原始代码和 FOWM 论文。
            q_targets = reward + self.config.discount * self.model.V(self.model.encode(next_observations))
            # 来自 FOWM 公式 3。它们在论文中表现为 Q(z, a)。这里称之为 v_targets，
            # 以强调我们用它们来计算 V 的损失。
            v_targets = self.model_target.Qs(z_preds[:-1].detach(), action, return_min=True)

        # 计算各项损失。
        # 损失权重随时间步指数衰减；越遥远的未来步骤对损失的影响越小。
        # 注意：unsqueeze 使我们可以广播到 (seq, batch)。
        temporal_loss_coeffs = torch.pow(
            self.config.temporal_decay_coeff, torch.arange(horizon, device=device)
        ).unsqueeze(-1)
        # 一致性损失：展开预测的潜变量与（目标模型的）观测编码器预测的潜变量之间的 MSE 损失。
        consistency_loss = (
            (
                temporal_loss_coeffs
                * F.mse_loss(z_preds[1:], z_targets, reduction="none").mean(dim=-1)
                # `z_preds` 依赖于当前观测和动作。
                * ~batch[f"{OBS_STR}.state_is_pad"][0]
                * ~batch["action_is_pad"]
                # `z_targets` 依赖于下一个观测。
                * ~batch[f"{OBS_STR}.state_is_pad"][1:]
            )
            .sum(0)
            .mean()
        )
        # 奖励损失：展开预测的奖励与数据集奖励之间的 MSE 损失。
        reward_loss = (
            (
                temporal_loss_coeffs
                * F.mse_loss(reward_preds, reward, reduction="none")
                * ~batch["next.reward_is_pad"]
                # `reward_preds` 依赖于当前观测和动作。
                * ~batch[f"{OBS_STR}.state_is_pad"][0]
                * ~batch["action_is_pad"]
            )
            .sum(0)
            .mean()
        )
        # 为集成中所有 Q 函数计算状态-动作价值损失（TD 损失）。
        q_value_loss = (
            (
                temporal_loss_coeffs
                * F.mse_loss(
                    q_preds_ensemble,
                    einops.repeat(q_targets, "t b -> e t b", e=q_preds_ensemble.shape[0]),
                    reduction="none",
                ).sum(0)  # 在集成维度上求和
                # `q_preds_ensemble` 依赖于第一个观测和动作。
                * ~batch[f"{OBS_STR}.state_is_pad"][0]
                * ~batch["action_is_pad"]
                # q_targets 依赖于奖励和下一个观测。
                * ~batch["next.reward_is_pad"]
                * ~batch[f"{OBS_STR}.state_is_pad"][1:]
            )
            .sum(0)
            .mean()
        )
        # 如 FOWM 公式 3 那样计算状态价值损失。
        diff = v_targets - v_preds
        # 期望分位数损失（expectile loss）的惩罚方式：
        #   - `v_preds <  v_targets` 时权重为 `expectile_weight`
        #   - `v_preds >= v_targets` 时权重为 `1 - expectile_weight`
        raw_v_value_loss = torch.where(
            diff > 0, self.config.expectile_weight, (1 - self.config.expectile_weight)
        ) * (diff**2)
        v_value_loss = (
            (
                temporal_loss_coeffs
                * raw_v_value_loss
                # `v_targets` 与 `v_preds` 一样，依赖于第一个观测和动作。
                * ~batch[f"{OBS_STR}.state_is_pad"][0]
                * ~batch["action_is_pad"]
            )
            .sum(0)
            .mean()
        )

        # 如 FOWM 3.1 节所述，计算 π 的优势加权回归损失。
        # 后续不再需要这些梯度，因此进行 detach。
        z_preds = z_preds.detach()
        # 优势计算使用 stopgrad。
        with torch.no_grad():
            advantage = self.model_target.Qs(z_preds[:-1], action, return_min=True) - self.model.V(
                z_preds[:-1]
            )
            info["advantage"] = advantage[0]
            # (t, b)
            exp_advantage = torch.clamp(torch.exp(advantage * self.config.advantage_scaling), max=100.0)
        action_preds = self.model.pi(z_preds[:-1])  # (t, b, a)
        # 计算动作与动作预测之间的 MSE。
        # 注意：FOWM 原始代码计算的是（相对于单位标准差高斯的）对数概率，并在动作维度上求和。
        # 计算（负）对数概率等价于将 MSE 乘以 0.5 再加上一个常数偏移（即 log(2*pi)/2 项
        # 乘以动作维度）。这里我们舍弃常数偏移，因为它不会改变优化步骤；同时舍弃 0.5，
        # 因为我们改用一个配置参数来处理它（参见下方计算总损失的位置）。
        mse = F.mse_loss(action_preds, action, reduction="none").sum(-1)  # (t, b)
        # 注意：与其他损失不同，原始实现没有在时间维度上求和。
        # TODO(alexander-soare)：改为在时间维度上求和，并检查训练是否仍能达到预期效果。
        pi_loss = (
            exp_advantage
            * mse
            * temporal_loss_coeffs
            # `action_preds` 依赖于第一个观测和动作。
            * ~batch[f"{OBS_STR}.state_is_pad"][0]
            * ~batch["action_is_pad"]
        ).mean()

        loss = (
            self.config.consistency_coeff * consistency_loss
            + self.config.reward_coeff * reward_loss
            + self.config.value_coeff * q_value_loss
            + self.config.value_coeff * v_value_loss
            + self.config.pi_coeff * pi_loss
        )

        info.update(
            {
                "consistency_loss": consistency_loss.item(),
                "reward_loss": reward_loss.item(),
                "Q_value_loss": q_value_loss.item(),
                "V_value_loss": v_value_loss.item(),
                "pi_loss": pi_loss.item(),
                "sum_loss": loss.item() * self.config.horizon,
            }
        )

        # 撤销 (b, t) -> (t, b) 的转置。
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        return loss, info

    def update(self):
        """通过一步 EMA 更新目标模型的参数。"""
        # 注意：这里与原始 FOWM 代码有一处细微差别。原始代码基于一个 EMA 更新频率参数
        # 来执行更新，该参数设为 2（每 2 步更新一次）。为简化代码，我们每一步都更新，
        # 并相应调整衰减参数 `alpha`（0.99 -> 0.995）。
        update_ema_parameters(self.model_target, self.model, self.config.target_model_momentum)


class TDMPCTOLD(nn.Module):
    """TD-MPC 中使用的面向任务的潜空间动态（Task-Oriented Latent Dynamics，TOLD）模型。"""

    def __init__(self, config: TDMPCConfig):
        super().__init__()
        self.config = config
        self._encoder = TDMPCObservationEncoder(config)
        self._dynamics = nn.Sequential(
            nn.Linear(config.latent_dim + config.action_feature.shape[0], config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.Sigmoid(),
        )
        self._reward = nn.Sequential(
            nn.Linear(config.latent_dim + config.action_feature.shape[0], config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, 1),
        )
        self._pi = nn.Sequential(
            nn.Linear(config.latent_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.action_feature.shape[0]),
        )
        self._Qs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.latent_dim + config.action_feature.shape[0], config.mlp_dim),
                    nn.LayerNorm(config.mlp_dim),
                    nn.Tanh(),
                    nn.Linear(config.mlp_dim, config.mlp_dim),
                    nn.ELU(),
                    nn.Linear(config.mlp_dim, 1),
                )
                for _ in range(config.q_ensemble_size)
            ]
        )
        self._V = nn.Sequential(
            nn.Linear(config.latent_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Tanh(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.ELU(),
            nn.Linear(config.mlp_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重。

        所有线性层和卷积层的权重采用正交初始化（奖励网络和 Q 网络的最终层除外，
        它们采用零初始化）。
        所有线性层和卷积层的偏置均采用零初始化。
        """

        def _apply_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                gain = nn.init.calculate_gain("relu")
                nn.init.orthogonal_(m.weight.data, gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_apply_fn)
        for m in [self._reward, *self._Qs]:
            assert isinstance(m[-1], nn.Linear), (
                "Sanity check. The last linear layer needs 0 initialization on weights."
            )
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)  # 前面已经做过零初始化，但保留这一行以防万一

    def encode(self, obs: dict[str, Tensor]) -> Tensor:
        """将观测编码为其潜变量表示。"""
        return self._encoder(obs)

    def latent_dynamics_and_reward(self, z: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        """在给定当前潜变量和动作的情况下，预测下一状态的潜变量表示和奖励。

        Args:
            z: 当前状态潜变量表示的张量，形状 (*, latent_dim)。
            a: 待执行动作的张量，形状 (*, action_dim)。
        Returns:
            一个元组，包含：
                - 下一状态潜变量表示的张量，形状 (*, latent_dim)。
                - 估计奖励的张量，形状 (*,)。
        """
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x).squeeze(-1)

    def latent_dynamics(self, z: Tensor, a: Tensor) -> Tensor:
        """在给定当前潜变量和动作的情况下，预测下一状态的潜变量表示。

        Args:
            z: 当前状态潜变量表示的张量，形状 (*, latent_dim)。
            a: 待执行动作的张量，形状 (*, action_dim)。
        Returns:
            下一状态潜变量表示的张量，形状 (*, latent_dim)。
        """
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x)

    def pi(self, z: Tensor, std: float = 0.0) -> Tensor:
        """从学习到的策略中采样一个动作。

        在为在线训练生成展开（rollout）时，策略还可以注入（截断的）高斯噪声以鼓励探索。

        Args:
            z: 当前状态潜变量表示的张量，形状 (*, latent_dim)。
            std: 注入噪声的标准差。
        Returns:
            采样动作的张量，形状 (*, action_dim)。
        """
        action = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(action) * std
            action += torch.randn_like(action) * std
        return action

    def V(self, z: Tensor) -> Tensor:  # noqa: N802
        """预测状态价值（V）。

        Args:
            z: 当前状态潜变量表示的张量，形状 (*, latent_dim)。
        Returns:
            估计状态价值的张量，形状 (*,)。
        """
        return self._V(z).squeeze(-1)

    def Qs(self, z: Tensor, a: Tensor, return_min: bool = False) -> Tensor:  # noqa: N802
        """为所有学习到的 Q 函数预测状态-动作价值。

        Args:
            z: 当前状态潜变量表示的张量，形状 (*, latent_dim)。
            a: 待执行动作的张量，形状 (*, action_dim)。
            return_min: 设为 true 以实现 FOWM 论文附录 C 中的细节：随机选取
                2 个 Q 并返回其中的最小值。
        Returns:
            集成中每个学习到的 Q 函数价值预测的张量，形状 (q_ensemble, *)；
            若 return_min=True，则返回形状 (*,) 的张量。
        """
        x = torch.cat([z, a], dim=-1)
        if not return_min:
            return torch.stack([q(x).squeeze(-1) for q in self._Qs], dim=0)
        else:
            if len(self._Qs) > 2:  # noqa: SIM108
                Qs = [self._Qs[i] for i in np.random.choice(len(self._Qs), size=2)]
            else:
                Qs = self._Qs
            return torch.stack([q(x).squeeze(-1) for q in Qs], dim=0).min(dim=0)[0]


class TDMPCObservationEncoder(nn.Module):
    """对图像和/或状态向量观测进行编码。"""

    def __init__(self, config: TDMPCConfig):
        """
        为像素和/或状态模态创建编码器。
        TODO(alexander-soare)：原始工作通过沿通道维度拼接来支持多张图像，
            请重新实现这一能力。
        """
        super().__init__()
        self.config = config

        if config.image_features:
            self.image_enc_layers = nn.Sequential(
                nn.Conv2d(
                    next(iter(config.image_features.values())).shape[0],
                    config.image_encoder_hidden_dim,
                    7,
                    stride=2,
                ),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 5, stride=2),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                nn.ReLU(),
            )
            dummy_shape = (1, *next(iter(config.image_features.values())).shape)
            out_shape = get_output_shape(self.image_enc_layers, dummy_shape)[1:]
            self.image_enc_layers.extend(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(out_shape), config.latent_dim),
                    nn.LayerNorm(config.latent_dim),
                    nn.Sigmoid(),
                )
            )

        if config.robot_state_feature:
            self.state_enc_layers = nn.Sequential(
                nn.Linear(config.robot_state_feature.shape[0], config.state_encoder_hidden_dim),
                nn.ELU(),
                nn.Linear(config.state_encoder_hidden_dim, config.latent_dim),
                nn.LayerNorm(config.latent_dim),
                nn.Sigmoid(),
            )

        if config.env_state_feature:
            self.env_state_enc_layers = nn.Sequential(
                nn.Linear(config.env_state_feature.shape[0], config.state_encoder_hidden_dim),
                nn.ELU(),
                nn.Linear(config.state_encoder_hidden_dim, config.latent_dim),
                nn.LayerNorm(config.latent_dim),
                nn.Sigmoid(),
            )

    def forward(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """对图像和/或状态向量进行编码。

        每种模态都被编码为大小为 (latent_dim,) 的特征向量，
        然后对所有特征取均匀平均。
        """
        feat = []
        # 注意：观测的顺序在这里很重要。
        if self.config.image_features:
            feat.append(
                flatten_forward_unflatten(
                    self.image_enc_layers, obs_dict[next(iter(self.config.image_features))]
                )
            )
        if self.config.env_state_feature:
            feat.append(self.env_state_enc_layers(obs_dict[OBS_ENV_STATE]))
        if self.config.robot_state_feature:
            feat.append(self.state_enc_layers(obs_dict[OBS_STATE]))
        return torch.stack(feat, dim=0).mean(0)


def random_shifts_aug(x: Tensor, max_random_shift_ratio: float) -> Tensor:
    """对图像进行水平和垂直方向的随机平移。

    改编自 https://github.com/facebookresearch/drqv2
    """
    b, _, h, w = x.size()
    assert h == w, "non-square images not handled yet"
    pad = int(round(max_random_shift_ratio * h))
    x = F.pad(x, tuple([pad] * 4), "replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(
        -1.0 + eps,
        1.0 - eps,
        h + 2 * pad,
        device=x.device,
        dtype=torch.float32,
    )[:h]
    arange = einops.repeat(arange, "w -> h w 1", h=h)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = einops.repeat(base_grid, "h w c -> b h w c", b=b)
    # 以像素为单位、且位于填充边界内的随机平移量。
    shift = torch.randint(
        0,
        2 * pad + 1,
        size=(b, 1, 1, 2),
        device=x.device,
        dtype=torch.float32,
    )
    shift *= 2.0 / (h + 2 * pad)
    grid = base_grid + shift
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def update_ema_parameters(ema_net: nn.Module, net: nn.Module, alpha: float):
    """原地更新 EMA 参数，更新方式为 ema_param <- alpha * ema_param + (1 - alpha) * param。"""
    for ema_module, module in zip(ema_net.modules(), net.modules(), strict=True):
        for (n_p_ema, p_ema), (n_p, p) in zip(
            ema_module.named_parameters(recurse=False), module.named_parameters(recurse=False), strict=True
        ):
            assert n_p_ema == n_p, "Parameter names don't match for EMA model update"
            if isinstance(p, dict):
                raise RuntimeError("Dict parameter not supported")
            if isinstance(module, nn.modules.batchnorm._BatchNorm) or not p.requires_grad:
                # 直接拷贝 BatchNorm 参数以及不可训练的参数。
                p_ema.copy_(p.to(dtype=p_ema.dtype).data)
            with torch.no_grad():
                p_ema.mul_(alpha)
                p_ema.add_(p.to(dtype=p_ema.dtype).data, alpha=1 - alpha)


def flatten_forward_unflatten(fn: Callable[[Tensor], Tensor], image_tensor: Tensor) -> Tensor:
    """临时将图像张量开头的额外维度展平的辅助函数。

    Args:
        fn: 图像张量将被传入的可调用对象。它应接受 (B, C, H, W) 并返回
            (B, *)，其中 * 表示任意数量的维度。
        image_tensor: 形状为 (**, C, H, W) 的图像张量，其中 ** 表示任意数量的维度，
            通常与 * 不同。
    Returns:
        可调用对象的返回值，被重塑为 (**, *) 的形状。
    """
    if image_tensor.ndim == 4:
        return fn(image_tensor)
    start_dims = image_tensor.shape[:-3]
    inp = torch.flatten(image_tensor, end_dim=-4)
    flat_out = fn(inp)
    return torch.reshape(flat_out, (*start_dims, *flat_out.shape[1:]))
