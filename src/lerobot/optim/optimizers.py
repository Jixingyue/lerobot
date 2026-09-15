#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import abc
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import draccus
import torch
from safetensors.torch import load_file, save_file

from lerobot.utils.constants import (
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
)
from lerobot.utils.io_utils import deserialize_json_into_object, write_json
from lerobot.utils.utils import flatten_dict, unflatten_dict

# 优化器 build() 方法接受的参数的类型别名。
# 这与 PyTorch 优化器的签名一致，同时还支持：
# - dict[str, Parameter]: 用于按名称差异化学习率的命名参数（例如 XVLA）
# - dict[str, Iterable]: 用于多优化器配置的多个参数组（例如 SAC）
OptimizerParams = (
    Iterable[torch.nn.Parameter]  # 来自 model.parameters()
    | Iterable[dict[str, Any]]  # 带有 lr/weight_decay 覆盖的参数组列表
    | dict[str, torch.nn.Parameter]  # 来自 dict(model.named_parameters())，用于基于名称的学习率
    | dict[str, Any]  # 用于带多个参数组的多优化器配置（SAC）
)


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    lr: float
    weight_decay: float
    grad_clip_norm: float

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    def builds_multiple_optimizers(self) -> bool:
        """当 build() 返回优化器字典时为 True（分片训练下不支持）。"""
        return False

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "adam"

    @abc.abstractmethod
    def build(self, params: OptimizerParams) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
        """
        构建优化器。可以是单个优化器，也可以是优化器字典。

        注意：当需要优化不同的模型时，多优化器很有用。
        例如，在强化学习场景中，可以有一个优化器用于策略，另一个用于价值函数。

        Args:
            params: 待优化的参数。根据优化器不同，接受多种格式：
                - Iterable[Parameter]: 来自 model.parameters() —— 标准 PyTorch 用法
                - Iterable[dict]: 带有 'params' 键以及可选的
                  'lr'、'weight_decay' 覆盖的参数组列表（例如 ACT、VQBeT 策略）
                - dict[str, Parameter]: 来自 dict(model.named_parameters())，用于
                  按参数名称应用差异化学习率的优化器（例如 XVLA）
                - dict[str, Iterable]: 用于多优化器配置，其中每个键映射到
                  一个独立优化器的参数（例如带 actor/critic/temperature 的 SAC）

        Returns:
            优化器，或优化器字典。
        """
        raise NotImplementedError


@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.Adam(params, **kwargs)


@OptimizerConfig.register_subclass("adamw")
@dataclass
class AdamWConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_clip_norm: float = 10.0

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.AdamW(params, **kwargs)


@OptimizerConfig.register_subclass("sgd")
@dataclass
class SGDConfig(OptimizerConfig):
    lr: float = 1e-3
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.SGD(params, **kwargs)


@OptimizerConfig.register_subclass("xvla-adamw")
@dataclass
class XVLAAdamWConfig(OptimizerConfig):
    """XVLA 专用的带差异化学习率的自定义 AdamW 优化器。

    视觉语言模型（VLM）使用基础学习率的 1/10 进行训练，
    以实现稳定的优化，而所有其他组件使用完整的学习率。

    这个学习率比例对于实现强大且稳定的微调性能至关重要。

    Soft-prompts 可以选择性地使用独立的学习率并支持预热。
    将 `soft_prompt_lr_scale` 设置为小于 1.0 的值（例如 0.1），
    可以让 soft-prompts 以较低的学习率开始。结合预热调度器可获得最佳效果。

    Note:
        要完全匹配官方报告的性能，可能需要为 soft-prompts 额外使用
        预热学习率调度，这可以带来微小的提升。
        当设置了 `soft_prompt_warmup_lr_scale` 时，soft-prompts 从
        `lr * soft_prompt_warmup_lr_scale` 开始，并应通过调度器进行预热。

    Parameter Groups:
        - Group 0 (vlm): VLM 参数，学习率为 lr * 0.1，weight_decay * 0.1
        - Group 1 (soft_prompts): Soft-prompt 参数，学习率为 lr * soft_prompt_lr_scale
        - Group 2 (other): 所有其他参数，使用完整学习率
    """

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    # Soft-prompt 专用设置
    soft_prompt_lr_scale: float = 1.0  # soft-prompt 学习率的缩放因子（1.0 = 与基础学习率相同）
    soft_prompt_warmup_lr_scale: float | None = None  # 如果设置，soft-prompts 以此比例开始（例如 0.01）

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        """
        构建带差异化学习率的 AdamW 优化器。

        Args:
            params: 必须是来自 dict(model.named_parameters())
                或等价形式的 dict[str, Parameter]。

        Returns:
            带有针对 VLM、soft-prompts 和其他组件的参数组的 AdamW 优化器

        Raises:
            AssertionError: 如果 params 不是字典（例如来自 model.parameters()）
        """
        assert isinstance(params, dict), "Custom LR optimizer requires `named_parameters()` as inputs."

        vlm_group, soft_prompt_group, other_group = [], [], []
        for name, p in params.items():
            if not p.requires_grad:
                continue
            if "vlm" in name.lower():
                vlm_group.append(p)
            elif "soft_prompt" in name.lower():
                soft_prompt_group.append(p)
            else:
                other_group.append(p)

        # 确定 soft-prompt 的学习率
        soft_prompt_lr = self.lr * self.soft_prompt_lr_scale
        if self.soft_prompt_warmup_lr_scale is not None:
            # 从预热比例开始，调度器会将其预热到 soft_prompt_lr
            soft_prompt_lr = self.lr * self.soft_prompt_warmup_lr_scale

        param_groups: list[dict[str, Any]] = [
            {
                "params": vlm_group,
                "lr": self.lr * 0.1,
                "weight_decay": self.weight_decay * 0.1,
                "name": "vlm",
            },
            {
                "params": soft_prompt_group,
                "lr": soft_prompt_lr,
                "weight_decay": self.weight_decay,
                "name": "soft_prompts",
            },
            {
                "params": other_group,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "name": "other",
            },
        ]

        # 过滤掉空的参数组
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        return torch.optim.AdamW(
            param_groups,
            betas=self.betas,
            eps=self.eps,
        )


@OptimizerConfig.register_subclass("multi_adam")
@dataclass
class MultiAdamConfig(OptimizerConfig):
    """带有不同参数组的多个 Adam 优化器的配置。

    这会创建一个 Adam 优化器字典，每个优化器有自己的超参数。

    Args:
        lr: 默认学习率（当某个组未指定时使用）
        weight_decay: 默认权重衰减（当某个组未指定时使用）
        optimizer_groups: 将参数组名称映射到其超参数的字典
        grad_clip_norm: 梯度裁剪范数
    """

    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    optimizer_groups: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def builds_multiple_optimizers(self) -> bool:
        return True

    def build(self, params: OptimizerParams) -> dict[str, torch.optim.Optimizer]:
        """构建多个 Adam 优化器。

        Args:
            params: 必须是 dict[str, Iterable[Parameter]]，将参数组名称
                映射到参数的可迭代对象。键应与 optimizer_groups 中的键匹配。
                通常来自需要独立优化器的策略（例如带
                actor/critic/temperature 的 SAC）。

        Returns:
            将参数组名称映射到其优化器的字典

        Raises:
            AssertionError: 如果 params 不是字典
        """
        assert isinstance(params, dict), "MultiAdamConfig requires a dict of parameter groups as inputs."
        optimizers = {}

        for name, group_params in params.items():
            # 获取组特定的超参数，或使用默认值
            group_config = self.optimizer_groups.get(name, {})

            # 使用合并后的参数创建优化器（默认值 + 组特定值）
            optimizer_kwargs = {
                "lr": group_config.get("lr", self.lr),
                "betas": group_config.get("betas", (0.9, 0.999)),
                "eps": group_config.get("eps", 1e-5),
                "weight_decay": group_config.get("weight_decay", self.weight_decay),
            }

            optimizers[name] = torch.optim.Adam(group_params, **optimizer_kwargs)

        return optimizers


def save_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
    save_dir: Path,
) -> None:
    """将优化器状态保存到磁盘（非分片运行；分片运行使用 DCP 通道）。

    Args:
        optimizer: 单个优化器或优化器字典。
        save_dir: 保存优化器状态的目录。
    """
    if isinstance(optimizer, dict):
        # 处理优化器字典
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            optimizer_dir.mkdir(exist_ok=True, parents=True)
            _save_single_optimizer_state(opt, optimizer_dir)
    else:
        # 处理单个优化器
        _save_single_optimizer_state(optimizer, save_dir)


def _save_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> None:
    """将单个优化器的状态保存到磁盘。"""
    state = optimizer.state_dict()
    param_groups = state.pop("param_groups")
    flat_state = flatten_dict(state)
    save_file(flat_state, save_dir / OPTIMIZER_STATE)
    write_json(param_groups, save_dir / OPTIMIZER_PARAM_GROUPS)


def load_optimizer_state(
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer], save_dir: Path
) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
    """从磁盘加载优化器状态。

    Args:
        optimizer: 单个优化器或优化器字典。
        save_dir: 用于加载优化器状态的目录。

    Returns:
        加载了状态的更新后的优化器。
    """
    if isinstance(optimizer, dict):
        # 处理优化器字典
        loaded_optimizers = {}
        for name, opt in optimizer.items():
            optimizer_dir = save_dir / name
            if optimizer_dir.exists():
                loaded_optimizers[name] = _load_single_optimizer_state(opt, optimizer_dir)
            else:
                loaded_optimizers[name] = opt
        return loaded_optimizers
    else:
        # 处理单个优化器
        return _load_single_optimizer_state(optimizer, save_dir)


def _load_single_optimizer_state(optimizer: torch.optim.Optimizer, save_dir: Path) -> torch.optim.Optimizer:
    """从磁盘加载单个优化器的状态。"""
    current_state_dict = optimizer.state_dict()
    flat_state = load_file(save_dir / OPTIMIZER_STATE)
    state = unflatten_dict(flat_state)

    # 处理 'state' 键可能不存在的情况（针对新创建的优化器）
    if "state" in state:
        loaded_state_dict = {"state": {int(k): v for k, v in state["state"].items()}}
    else:
        loaded_state_dict = {"state": {}}

    if "param_groups" in current_state_dict:
        param_groups = deserialize_json_into_object(
            save_dir / OPTIMIZER_PARAM_GROUPS, current_state_dict["param_groups"]
        )
        loaded_state_dict["param_groups"] = param_groups

    optimizer.load_state_dict(loaded_state_dict)
    return optimizer
