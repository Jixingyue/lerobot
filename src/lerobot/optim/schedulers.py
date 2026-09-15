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
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import draccus
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from lerobot.utils.constants import SCHEDULER_STATE
from lerobot.utils.import_utils import _diffusers_available, require_package
from lerobot.utils.io_utils import deserialize_json_into_object, write_json

if TYPE_CHECKING or _diffusers_available:
    from diffusers.optimization import get_scheduler
else:
    get_scheduler = None


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
    num_warmup_steps: int | None

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("diffuser")
@dataclass
class DiffuserSchedulerConfig(LRSchedulerConfig):
    name: str = "cosine"
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        require_package("diffusers", extra="diffusion")

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


@LRSchedulerConfig.register_subclass("vqbet")
@dataclass
class VQBeTSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_vqvae_training_steps: int
    num_cycles: float = 0.5

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step):
            if current_step < self.num_vqvae_training_steps:
                return float(1)
            else:
                adjusted_step = current_step - self.num_vqvae_training_steps
                if adjusted_step < self.num_warmup_steps:
                    return float(adjusted_step) / float(max(1, self.num_warmup_steps))
                progress = float(adjusted_step - self.num_warmup_steps) / float(
                    max(1, num_training_steps - self.num_warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("constant_with_warmup")
@dataclass
class ConstantWithWarmupSchedulerConfig(LRSchedulerConfig):
    """线性预热后接恒定学习率。

    与 LingBot-VA 使用的 ``warmup_constant_lambda`` 一致（上游 ``wan_va/train.py``）：
    学习率在 ``num_warmup_steps`` 步内从 0 线性上升到峰值，然后保持不变。
    """

    num_warmup_steps: int = 1000

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        warmup_steps = self.num_warmup_steps or 0

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_annealing_with_warmup")
@dataclass
class CosineAnnealingWithWarmupSchedulerConfig(LRSchedulerConfig):
    """线性预热后接从峰值学习率到零的余弦退火。

    由 EVO1 使用；退火阶段始终覆盖剩余的训练步数。
    """

    num_warmup_steps: int

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step: int) -> float:
            if current_step < self.num_warmup_steps:
                return current_step / max(1, self.num_warmup_steps)
            progress = (current_step - self.num_warmup_steps) / max(
                1, num_training_steps - self.num_warmup_steps
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_decay_with_warmup")
@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Physical Intelligence 用于训练 Pi0 的调度器。

    当 num_training_steps < num_decay_steps 时，自动缩放预热和衰减步数。
    这确保即使在较短的训练运行中，学习率调度也能正确完成。
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        # 如果训练步数少于配置的衰减步数，则自动缩放调度器参数
        actual_warmup_steps = self.num_warmup_steps
        actual_decay_steps = self.num_decay_steps

        if num_training_steps < self.num_decay_steps:
            # 计算缩放因子，使调度适配可用的训练步数
            scale_factor = num_training_steps / self.num_decay_steps
            actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
            actual_decay_steps = num_training_steps

            logging.info(
                f"Auto-scaling LR scheduler: "
                f"num_training_steps ({num_training_steps}) < num_decay_steps ({self.num_decay_steps}). "
                f"Scaling warmup: {self.num_warmup_steps} → {actual_warmup_steps}, "
                f"decay: {self.num_decay_steps} → {actual_decay_steps} "
                f"(scale factor: {scale_factor:.3f})"
            )

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                step = min(current_step, actual_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < actual_warmup_steps:
                return linear_warmup_schedule(current_step)

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


def save_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler
