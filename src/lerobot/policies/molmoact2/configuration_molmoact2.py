# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import (
    LRSchedulerConfig,
    OptimizerConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig


class MolmoAct2AdamW(torch.optim.AdamW):
    """带有按组件裁剪和低显存 BF16 更新补偿的 AdamW。

    LeRobot 的共享训练器在调用 ``Optimizer.step`` 之前会把整个策略作为一个
    向量进行裁剪。而官方 MolmoAct2 则是独立裁剪每个优化器组（LLM、ViT、
    connector 和 action expert）。在这里保留裁剪逻辑可以让该策略与官方行为
    保持一致，而无需修改共享训练器或其他任何策略。

    参数和优化器状态的 dtype 遵循 Pi0.5 风格的存储策略：大型 VLM 张量保持
    BF16，而 action expert 和明确标记为敏感的张量保持 FP32。一个惰性创建的
    BF16 Kahan 风格残差可以保留小于一个 BF16 参数 ULP 的 VLM 更新。这为每个
    可训练的 BF16 参数额外消耗一个 BF16 张量，而不是官方 AMP/FSDP 方案所需的
    FP32 参数副本和优化器副本。原生 AdamW 仍然是 FP32 参数（包括完整的
    action expert 和 LoRA 适配器）的快速路径。
    """

    def __init__(self, params, *, group_grad_clip_norm: float, **kwargs) -> None:
        if group_grad_clip_norm <= 0:
            raise ValueError(f"MolmoAct2 group_grad_clip_norm must be positive, got {group_grad_clip_norm}.")
        super().__init__(params, **kwargs)
        self.group_grad_clip_norm = float(group_grad_clip_norm)

    def _clip_grad_groups(self) -> tuple[torch.Tensor, ...]:
        norms: list[torch.Tensor] = []
        for group in self.param_groups:
            params_with_grad = [param for param in group["params"] if param.grad is not None]
            if not params_with_grad:
                continue
            norms.append(
                torch.nn.utils.clip_grad_norm_(
                    params_with_grad,
                    max_norm=self.group_grad_clip_norm,
                    error_if_nonfinite=False,
                )
            )
        return tuple(norms)

    def _step_native_non_bfloat16(self) -> None:
        """仅对非 BF16 参数运行 PyTorch 原生 AdamW。"""
        original_group_params: list[list[torch.Tensor]] = []
        try:
            for group in self.param_groups:
                original_params = group["params"]
                original_group_params.append(original_params)
                group["params"] = [param for param in original_params if param.dtype != torch.bfloat16]
            super().step()
        finally:
            for group, original_params in zip(self.param_groups, original_group_params, strict=True):
                group["params"] = original_params

    def _step_compensated_bfloat16(self) -> None:
        """对 BF16 存储的参数应用 AdamW，同时保留亚 ULP 级别的更新。

        Adam 动量刻意保持为 BF16，以维持 Pi0.5 风格的显存开销。
        ``effective_parameter`` 是每个参数的一个 FP32 临时量，绝不是持久化的
        模型级主副本。残差为 BF16，且仅对收到梯度的参数惰性初始化，因此冻结的
        VLM 权重和仅 LoRA 的训练不会产生额外开销。
        """
        for group in self.param_groups:
            bfloat16_group = dict(group)
            bfloat16_group["params"] = [param for param in group["params"] if param.dtype == torch.bfloat16]

            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            max_exp_avg_sqs: list[torch.Tensor] = []
            state_steps: list[torch.Tensor] = []
            self._init_group(
                bfloat16_group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            if torch.is_tensor(lr):
                lr = lr.item()
            lr = float(lr)

            for index, parameter in enumerate(params_with_grad):
                grad = grads[index]
                if group["maximize"]:
                    grad = -grad

                state = self.state[parameter]
                compensation = state.get("compensation")
                if compensation is None:
                    compensation = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["compensation"] = compensation

                state_step = state_steps[index]
                state_step.add_(1)
                step_value = state_step.item()

                exp_avg = exp_avgs[index]
                exp_avg_sq = exp_avg_sqs[index]
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step_value
                bias_correction2_sqrt = (1 - beta2**step_value) ** 0.5
                step_size = lr / bias_correction1
                if group["amsgrad"]:
                    max_exp_avg_sq = max_exp_avg_sqs[index]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])

                effective_parameter = parameter.float().add_(compensation)
                if group["weight_decay"] != 0:
                    effective_parameter.mul_(1 - lr * group["weight_decay"])
                effective_parameter.addcdiv_(exp_avg, denom, value=-step_size)
                parameter.copy_(effective_parameter)
                compensation.copy_(effective_parameter.sub_(parameter))

    @staticmethod
    def _any_nonfinite_across_ranks(norms: tuple[torch.Tensor, ...]) -> bool:
        """与官方 MolmoAct2 的全 rank 非有限值步进保护保持一致。"""
        local_nonfinite = any(not bool(torch.isfinite(norm).all().item()) for norm in norms)
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return local_nonfinite

        backend = str(torch.distributed.get_backend()).lower()
        if "nccl" in backend:
            flag_device = torch.device("cuda", torch.cuda.current_device())
        else:
            flag_device = torch.device("cpu")
        nonfinite_flag = torch.tensor(int(local_nonfinite), device=flag_device, dtype=torch.int32)
        torch.distributed.all_reduce(nonfinite_flag, op=torch.distributed.ReduceOp.MAX)
        return bool(nonfinite_flag.item())

    @torch.no_grad()
    def step(self, closure=None):
        # LeRobot 从不传入 closure，但为确实传入的调用者保留标准
        # Optimizer 语义。
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        grad_norms = self._clip_grad_groups()
        if self._any_nonfinite_across_ranks(grad_norms):
            # 官方 MolmoAct2 会在所有 rank 上跳过本次更新并清除无效梯度。
            # 特别地，当某个组件的范数为非有限值时，Adam 动量和步数
            # 绝不能前进。
            self.zero_grad(set_to_none=True)
            return loss
        self._step_native_non_bfloat16()
        self._step_compensated_bfloat16()
        return loss


@OptimizerConfig.register_subclass("molmoact2_adamw")
@dataclass
class MolmoAct2AdamWConfig(OptimizerConfig):
    """策略内置的 AdamW 预设，支持按组件独立裁剪。"""

    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-6
    weight_decay: float = 0.0
    # 设为零可禁用共享训练器的全局裁剪。策略现有的
    # optimizer_grad_clip_norm 会作为按组的阈值透传。
    grad_clip_norm: float = 0.0
    group_grad_clip_norm: float = 1.0

    def build(self, params) -> torch.optim.Optimizer:
        return MolmoAct2AdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            group_grad_clip_norm=self.group_grad_clip_norm,
        )


@LRSchedulerConfig.register_subclass("molmoact2_cosine_with_warmup")
@dataclass
class MolmoAct2CosineWithWarmupSchedulerConfig(LRSchedulerConfig):
    """官方 MolmoAct2 的预热后接余弦衰减。

    LeRobot 共享的 Pi0 调度器是基于绝对全局步数来计算余弦值的。这会在预热
    结束时造成学习率不连续（在较短的调度中尤为明显）。原生 MolmoAct2 则是在
    预热结束后将余弦时间从零开始，并对每个组件专属的基础学习率应用相同的
    乘数。
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> LambdaLR:
        if self.num_warmup_steps < 0:
            raise ValueError(f"num_warmup_steps must be >= 0, got {self.num_warmup_steps}.")
        if self.num_decay_steps < 1:
            raise ValueError(f"num_decay_steps must be >= 1, got {self.num_decay_steps}.")
        if self.peak_lr <= 0:
            raise ValueError(f"peak_lr must be > 0, got {self.peak_lr}.")
        if not 0 <= self.decay_lr < self.peak_lr:
            raise ValueError(
                f"decay_lr must be in [0, peak_lr), got decay_lr={self.decay_lr}, peak_lr={self.peak_lr}."
            )

        # 官方 Trainer 使用其配置的 max_duration 作为余弦终点。当 LeRobot
        # 刻意运行较短的诊断性检查时（例如在配置的 24K 调度上只跑 3K 次更新），
        # 保留该时钟。在这里做截断会悄悄地把检查压缩到衰减下限。
        decay_steps = int(self.num_decay_steps)
        warmup_steps = min(int(self.num_warmup_steps), decay_steps)
        alpha = float(self.decay_lr / self.peak_lr)

        def lr_lambda(current_step: int) -> float:
            # LambdaLR 会在第一次优化器更新之前安装 lambda(0)，而 LeRobot
            # 在每次更新之后才推进它。官方 MolmoAct2 是先递增 global_step，
            # 然后在该次更新之前计算学习率，因此其第 k 次优化器更新使用的是
            # f(k)，而不是 f(k - 1)。
            step = max(int(current_step) + 1, 0)
            if warmup_steps > 0 and step < warmup_steps:
                return float(step / warmup_steps)
            if step >= decay_steps:
                return alpha
            cosine_span = decay_steps - warmup_steps
            if cosine_span <= 0:
                return alpha
            cosine_step = step - warmup_steps
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_span))
            return alpha + (1.0 - alpha) * cosine_decay

        return LambdaLR(optimizer, lr_lambda, -1)


@PreTrainedConfig.register_subclass("molmoact2")
@dataclass
class MolmoAct2Config(PreTrainedConfig):
    """基于转换后的 HF 检查点实现的 MolmoAct2 策略。"""

    checkpoint_path: str = "allenai/MolmoAct2"
    checkpoint_revision: str | None = None
    checkpoint_force_download: bool = False

    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30

    # 官方 MolmoAct2 机器人微调只优化连续 flow-matching 目标。已发布的
    # 检查点保留了离散动作权重，``both`` 仍可作为显式消融选项使用。
    action_mode: str = "continuous"
    inference_action_mode: str | None = "continuous"
    discrete_action_tokenizer: str = "allenai/MolmoAct2-FAST-Tokenizer"
    discrete_generation_max_steps: int | None = None
    norm_tag: str | None = None
    # 可选的独立 norm_stats.json。这可以让通用基础模型使用随下游检查点
    # 发布的精确本体（embodiment）统计量，而不改变所加载的模型权重。
    norm_stats_path: str | None = None

    setup_type: str = ""
    control_mode: str = ""
    image_keys: list[str] = field(default_factory=list)
    normalize_language: bool = True
    add_setup_tokens: bool = True
    add_control_tokens: bool = True
    normalize_gripper: bool = False
    num_state_tokens: int = 256
    # 保持未设置即可使用默认的 MolmoAct2 序列预算，该预算由固定的
    # 图像/提示/状态/动作 token 布局推断得出。仅在提示异常冗长时才覆盖。
    max_sequence_length: int | None = None

    # 由已发布的 MolmoAct2 检查点固定。我们会在模型加载时进行校验。
    expected_max_action_dim: int = 32

    # 从原始 MolmoAct2 训练路径复制过来的 flow-matching 训练参数。
    num_flow_timesteps: int = 8
    flow_matching_cutoff: float = 1.0
    flow_matching_time_offset: float = 0.001
    flow_matching_time_scale: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5
    num_inference_steps: int | None = None
    mask_action_dim_padding: bool = True
    enable_inference_cuda_graph: bool = True
    # MolmoAct2 内置的评估选项。启用后，随机性的连续动作生成会使用
    # 由 eval_seed 派生的 rollout 级生成器。
    per_episode_seed: bool = False
    eval_seed: int | None = None
    rtc_config: RTCConfig | None = None

    # 用于跨标定兼容性的关节坐标系变换。部分 MolmoAct2 检查点训练所用的
    # 关节约定与当前 LeRobot 标定不同。同时设置这两项即可在运行时应用
    # 符号/偏移校正（状态在进模型前校正，动作在出模型后校正）。
    # 参见：https://huggingface.co/docs/lerobot/backwardcomp
    # 默认为 None（不变换）。两者必须同时设置。
    joint_signs: list[float] | None = None
    joint_offsets: list[float] | None = None

    # 仅控制 VLM 部分。action expert 始终全量微调。
    train_mode_vlm: str = "lora"
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    enable_knowledge_insulation: bool = False
    freeze_embedding: bool = True
    gradient_checkpointing: bool = False
    # Pi0.5 风格的公开开关，底层采用 MolmoAct2 官方的按块编译策略。
    # 策略刻意将编译器后端/作用域保持在内部，因此只有一种受支持的
    # 执行方案。
    compile_model: bool = False

    # Pi0.5 风格的精度开关，控制参数存储和 autocast。
    # ``bfloat16`` 将大型文本和视觉矩阵以 bf16 存储，而完整的 action expert、
    # 选定的 norm/head/LoRA 参数以及 RoPE 状态保持 fp32；算子计算遵循
    # bf16 autocast，外加显式的敏感 fp32 运算。
    # ``float32`` 则让整个模型和计算都保持 fp32。
    dtype: str = "bfloat16"
    # 基于已发布的 ``allenai/MolmoAct2`` HF 基础模型的官方微调明确应用了
    # 0.1 的无掩码残差 dropout，并禁用了仅响应（response-only）变体。
    # 因此转换后的 HF decoder 使用该常规残差 dropout 值，与官方
    # HF 检查点路径保持一致。
    llm_residual_dropout: float = 0.1
    softmax_auxiliary_loss: bool = True
    softmax_auxiliary_loss_scale: float = 1e-4
    discrete_loss_token_weighting: str = "root_subsegments_root_tokens"

    optimizer_lr: float = 1e-5
    optimizer_vit_lr: float = 5e-6
    optimizer_connector_lr: float = 5e-6
    optimizer_action_expert_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-6
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 200
    scheduler_decay_steps: int = 24_000
    scheduler_decay_lr: float = 1e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    dataset_feature_names: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.joint_signs is None) != (self.joint_offsets is None):
            raise ValueError("joint_signs and joint_offsets must both be set or both be None.")
        if self.joint_signs is not None and len(self.joint_signs) != len(self.joint_offsets):
            raise ValueError("joint_signs and joint_offsets must have the same length.")
        if self.action_mode not in {"continuous", "discrete", "both"}:
            raise ValueError(
                f"Unsupported action_mode={self.action_mode!r}. "
                "Expected one of {'continuous', 'discrete', 'both'}."
            )
        if self.inference_action_mode not in {None, "continuous", "discrete"}:
            raise ValueError(
                f"Unsupported inference_action_mode={self.inference_action_mode!r}. "
                "Expected one of {None, 'continuous', 'discrete'}."
            )
        if self.inference_action_mode == "continuous" and self.action_mode == "discrete":
            raise ValueError("MolmoAct2 action_mode='discrete' cannot run continuous inference.")
        if self.inference_action_mode == "discrete" and self.action_mode == "continuous":
            raise ValueError("MolmoAct2 action_mode='continuous' cannot run discrete inference.")
        if self.norm_stats_path is not None and not str(self.norm_tag or "").strip():
            raise ValueError("MolmoAct2 norm_stats_path requires norm_tag to select an embodiment.")
        if self.train_mode_vlm not in {"fft", "lora", "freeze"}:
            raise ValueError(
                f"Unsupported train_mode_vlm={self.train_mode_vlm!r}. "
                "Expected one of {'fft', 'lora', 'freeze'}."
            )
        if self.train_mode_vlm == "freeze" and self.action_mode != "continuous":
            raise ValueError("MolmoAct2 train_mode_vlm='freeze' requires action_mode='continuous'.")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}.")
        if self.n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {self.n_action_steps}.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})."
            )
        if self.expected_max_action_dim != 32:
            raise ValueError("MolmoAct2 released checkpoints use expected_max_action_dim=32.")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"Unsupported dtype={self.dtype!r}. Expected 'float32' or 'bfloat16'.")
        if not 0 <= self.llm_residual_dropout <= 1:
            raise ValueError(f"llm_residual_dropout must be in [0, 1], got {self.llm_residual_dropout}.")
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}.")
        if self.lora_alpha < 1:
            raise ValueError(f"lora_alpha must be >= 1, got {self.lora_alpha}.")
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError(f"lora_dropout must be in [0, 1], got {self.lora_dropout}.")
        if self.lora_bias not in {"none", "all", "lora_only"}:
            raise ValueError(
                f"Unsupported lora_bias={self.lora_bias!r}. Expected one of 'none', 'all', or 'lora_only'."
            )
        if self.discrete_loss_token_weighting not in {
            "none",
            "token",
            "root_tokens",
            "root_subsegments",
            "root_subsegments_root_tokens",
        }:
            raise ValueError(
                f"Unsupported discrete_loss_token_weighting={self.discrete_loss_token_weighting!r}."
            )
        if self.discrete_generation_max_steps is not None and self.discrete_generation_max_steps < 1:
            raise ValueError(
                f"discrete_generation_max_steps must be >= 1 or None, got {self.discrete_generation_max_steps}."
            )
        if self.max_sequence_length is not None and self.max_sequence_length < 1:
            raise ValueError(f"max_sequence_length must be >= 1 or None, got {self.max_sequence_length}.")

    def _save_pretrained(self, save_directory: Path) -> None:
        """保存可移植的配置，不含仅用于初始化的归一化元数据。"""
        config_for_save = replace(self, norm_tag=None, norm_stats_path=None)
        PreTrainedConfig._save_pretrained(config_for_save, save_directory)

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return MolmoAct2AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=0.0,
            group_grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return MolmoAct2CosineWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def set_dataset_feature_metadata(self, features: dict[str, Any]) -> None:
        self.dataset_feature_names = {}
        for key in (ACTION, OBS_STATE):
            feature = features.get(key) if isinstance(features, dict) else None
            if isinstance(feature, dict) and feature.get("names") is not None:
                self.dataset_feature_names[key] = feature["names"]

    def validate_features(self) -> None:
        """校验并设置 MolmoAct2 的输入和输出特征。"""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "MolmoAct2 policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(0,),
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.expected_max_action_dim,),
            )
            self.output_features[ACTION] = action_feature
