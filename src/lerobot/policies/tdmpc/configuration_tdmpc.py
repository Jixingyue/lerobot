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
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig


@PreTrainedConfig.register_subclass("tdmpc")
@dataclass
class TDMPCConfig(PreTrainedConfig):
    """TDMPCPolicy 的配置类。

    默认值针对使用 xarm_lift_medium_replay（提供本体感知和单相机观测）进行训练而配置。

    你最可能需要修改的是那些依赖于环境/传感器的参数，
    即：`input_features`、`output_features`，以及可能需要修改的 `max_random_shift_ratio`。

    Args:
        n_action_repeats: 规划返回的动作需要重复执行的次数。（提示：可以 Google 一下
            Q-learning 中的动作重复，或者问问你最喜欢的聊天机器人）
        horizon: 模型预测控制（model predictive control）的预测时域。
        n_action_steps: 从模型预测控制给出的规划中取用的动作步数。这是使用动作重复的
            一种替代方案。如果该值大于 1，则要求 `n_action_repeats == 1`、
            `use_mpc == True` 且 `n_action_steps <= horizon`。注意，这种从规划中
            取用多个步骤的做法并不在原始实现中。
        input_features: 定义策略输入数据 PolicyFeature 的字典。键表示输入数据名称，
            值为 PolicyFeature，由 FeatureType 和 shape 属性组成。
        output_features: 定义策略输出数据 PolicyFeature 的字典。键表示输出数据名称，
            值为 PolicyFeature，由 FeatureType 和 shape 属性组成。
        normalization_mapping: 将 FeatureType 的字符串值（如 "STATE"、"VISUAL"）映射到
            对应 NormalizationMode（如 NormalizationMode.MIN_MAX）的字典。
        image_encoder_hidden_dim: 用于图像编码的卷积层通道数。
        state_encoder_hidden_dim: 用于状态向量编码的 MLP 的隐藏维度。
        latent_dim: 观测的潜变量嵌入维度。
        q_ensemble_size: 用于不确定性估计的 Q 函数估计器集成数量。
        mlp_dim: 用于建模动态编码器、奖励函数、策略（π）、Q 集成和 V 的各 MLP 的隐藏维度。
        discount: 强化学习形式化中使用的折扣因子（γ）。
        use_mpc: 是否使用模型预测控制。另一种选择是在每一步直接对策略模型（π）采样。
        cem_iterations: MPC 中 MPPI/CEM 循环的迭代次数。
        max_std: CEM 中从高斯 PDF 采样动作时使用的最大标准差。
        min_std: 对从策略模型（π）采样的动作所施加噪声的最小标准差。同时也充当
            CEM 中从高斯 PDF 采样动作时的最小标准差。
        n_gaussian_samples: 每次 CEM 迭代从高斯分布中抽取的样本数。必须非零。
        n_pi_samples: 每次 CEM 迭代从策略/世界模型展开（rollout）中抽取的样本数。可以为零。
        uncertainty_regularizer_coeff: 估计轨迹价值时所用不确定性正则化的系数
            （即 FOWM 论文公式 4 中的 λ 系数）。
        n_elites: 每次 CEM 迭代用于更新高斯参数的精英样本数量。
        elite_weighting_temperature: 在 CEM 更新高斯参数时，对精英样本按轨迹价值进行
            softmax 加权所使用的温度。
        gaussian_mean_momentum: 对 CEM 中优化的高斯参数均值 μ 进行 EMA 更新时使用的
            动量（α）。更新计算为 μ⁻ ← αμ⁻ + (1-α)μ。
        max_random_shift_ratio: 训练时数据增强对图像施加的最大随机平移量（占图像尺寸的
            比例，以像素为单位）。若设为 0，则不施加此增强。注意，该增强假设输入图像为正方形。
        reward_coeff: 奖励回归损失的损失加权系数。
        expectile_weight: 状态价值函数（V）的期望分位数回归（expectile regression）中
            使用的权重（τ）。v_pred < v_target 时权重为 τ，v_pred >= v_target 时权重为 (1-τ)。
            τ 应位于 [0, 1]。将 τ 设得越接近 1，得到的 V 越“乐观”。这样做是合理的，
            因为 v_target 是用学习到的状态-动作价值函数（Q）对样本内动作评估得到的，
            这些动作未必总是最优的。
        value_coeff: 状态-动作价值（Q）TD 损失与状态价值（V）期望分位数回归损失
            共同使用的损失加权系数。
        consistency_coeff: 一致性损失的损失加权系数。
        advantage_scaling: 在对策略（π）估计器参数进行优势加权回归时，优势量在取指数前
            缩放所使用的因子。注意，取指数后的优势量会被截断在 100.0。
        pi_coeff: 动作回归损失的损失加权系数。
        temporal_decay_coeff: 对未来时间步损失系数进行指数衰减的系数。提示：每次损失
            计算都包含从当前时间步开始、共 `horizon` 步的动作。
        target_model_momentum: 对目标模型进行 EMA 更新时使用的动量（α）。更新计算为
            ϕ ← αϕ + (1-α)θ，其中 ϕ 是目标模型的参数，θ 是正在训练的模型的参数。
    """

    # 输入/输出结构。
    n_obs_steps: int = 1
    n_action_repeats: int = 2
    horizon: int = 5
    n_action_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # 架构/建模。
    # 神经网络。
    image_encoder_hidden_dim: int = 32
    state_encoder_hidden_dim: int = 256
    latent_dim: int = 50
    q_ensemble_size: int = 5
    mlp_dim: int = 512
    # 强化学习。
    discount: float = 0.9

    # 推理。
    use_mpc: bool = True
    cem_iterations: int = 6
    max_std: float = 2.0
    min_std: float = 0.05
    n_gaussian_samples: int = 512
    n_pi_samples: int = 51
    uncertainty_regularizer_coeff: float = 1.0
    n_elites: int = 50
    elite_weighting_temperature: float = 0.5
    gaussian_mean_momentum: float = 0.1

    # 训练与损失计算。
    max_random_shift_ratio: float = 0.0476
    # 损失系数。
    reward_coeff: float = 0.5
    expectile_weight: float = 0.9
    value_coeff: float = 0.1
    consistency_coeff: float = 20.0
    advantage_scaling: float = 3.0
    pi_coeff: float = 0.5
    temporal_decay_coeff: float = 0.5
    # 目标模型。
    target_model_momentum: float = 0.995

    # 训练预设
    optimizer_lr: float = 3e-4

    def __post_init__(self):
        super().__post_init__()

        """输入校验（并非穷举所有情况）。"""
        if self.n_gaussian_samples <= 0:
            raise ValueError(
                f"The number of gaussian samples for CEM should be non-zero. Got `{self.n_gaussian_samples=}`"
            )
        if self.normalization_mapping["ACTION"] is not NormalizationMode.MIN_MAX:
            raise ValueError(
                "TD-MPC assumes the action space dimensions to all be in [-1, 1]. Therefore it is strongly "
                f"advised that you stick with the default. See {self.__class__.__name__} docstring for more "
                "information."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.n_action_steps > 1:
            if self.n_action_repeats != 1:
                raise ValueError(
                    "If `n_action_steps > 1`, `n_action_repeats` must be left to its default value of 1."
                )
            if not self.use_mpc:
                raise ValueError("If `n_action_steps > 1`, `use_mpc` must be set to `True`.")
            if self.n_action_steps > self.horizon:
                raise ValueError("`n_action_steps` must be less than or equal to `horizon`.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(lr=self.optimizer_lr)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        # 应当只有一个图像键。
        if len(self.image_features) > 1:
            raise ValueError(
                f"{self.__class__.__name__} handles at most one image for now. Got image keys {self.image_features}."
            )

        if len(self.image_features) > 0:
            image_ft = next(iter(self.image_features.values()))
            if image_ft.shape[-2] != image_ft.shape[-1]:
                # TODO(alexander-soare)：这一限制完全是由随机平移增强中的代码导致的，
                # 应当可以移除。
                raise ValueError(f"Only square images are handled now. Got image shape {image_ft.shape}.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(self.horizon + 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return list(range(self.horizon))
