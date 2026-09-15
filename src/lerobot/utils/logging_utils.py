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
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist

from .utils import format_big_number

_VALID_REDUCTIONS = ("none", "max", "mean", "sum")


class AverageMeter:
    """
    计算并存储平均值和当前值
    改编自 https://github.com/pytorch/examples/blob/main/imagenet/main.py

    参数:
        name: 指标的显示名称。
        fmt: 渲染该指标时使用的格式字符串。
        reduction: 在记录日志之前由
            :meth:`MetricsTracker.reduce_across_ranks` 应用的跨进程归约。
            取值为 ``"none"``（各 rank 自身的值，默认）、``"max"``、``"mean"``
            或 ``"sum"`` 之一。对于瓶颈类指标（例如数据加载或
            更新的挂钟时间），请使用 ``"max"``，这样多 GPU 运行报告的是最慢的 rank 而不是 rank 0。
    """

    def __init__(self, name: str, fmt: str = ":f", reduction: str = "none"):
        if reduction not in _VALID_REDUCTIONS:
            raise ValueError(
                f"Invalid reduction {reduction!r} for AverageMeter; expected one of {_VALID_REDUCTIONS}."
            )
        self.name = name
        self.fmt = fmt
        self.reduction = reduction
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class MetricsTracker:
    """
    一个用于随时间跟踪和记录指标的辅助类。

    参数:
        batch_size (int): 每进程批次大小（每个数据并行工作进程上
            每个微批次的样本数）。
        num_frames (int): 训练数据集中的总帧数。
        num_episodes (int): 训练数据集中的总 episode 数。
        metrics (dict[str, AverageMeter]): 要跟踪的计量器，以指标名为键。
        initial_step (int): 起始步计数器（恢复运行时非零）。
            默认为 0。
        dp_world_size (int): 相互独立的数据并行工作进程数
            （`dp_replicate * dp_shard`），用于缩放样本计数；上下文并行的
            各 peer 消费同一批次，不能被重复计数。默认为 1。

    使用模式：

    ```python
    # 初始化，起始步可能非零（例如恢复运行时）
    metrics = {"loss": AverageMeter("loss", ":.3f")}
    train_metrics = MetricsTracker(
        batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        metrics,
        initial_step=step,
        dp_world_size=dp_world,
    )

    # 在每个训练步更新由 step 派生的指标（样本数、episode 数、epoch 数）
    train_metrics.step()

    # 更新各类指标
    loss = policy.forward(batch)
    train_metrics.loss = loss

    # 显示当前指标
    logging.info(train_metrics)

    # 导出到 wandb
    wandb.log(train_metrics.to_dict())

    # 记录日志后重置平均值
    train_metrics.reset_averages()
    ```
    """

    __keys__ = [
        "_batch_size",
        "_num_frames",
        "_avg_samples_per_ep",
        "_dp_world_size",
        "metrics",
        "steps",
        "samples",
        "episodes",
        "epochs",
        "_caller_metrics",
    ]

    def __init__(
        self,
        batch_size: int,
        num_frames: int,
        num_episodes: int,
        metrics: dict[str, AverageMeter],
        initial_step: int = 0,
        dp_world_size: int = 1,
    ):
        self.__dict__.update(dict.fromkeys(self.__keys__))
        self._batch_size = batch_size
        self._num_frames = num_frames
        self._avg_samples_per_ep = num_frames / num_episodes
        # 样本计数按相互独立的数据并行工作进程数缩放，即
        # dp_replicate * dp_shard——而不是 world 大小：上下文并行的各 peer
        # 消费同一批次，不能被重复计数。`step` 计数的是微批次，因此
        # 这里同样不应包含梯度累积因子。
        self._dp_world_size = dp_world_size
        self.metrics = metrics

        self.steps = initial_step
        # 一个样本是一对 (observation, action)，其中观测和动作
        # 可以跨越多个时间戳。在一个批次中，我们有 `batch_size` 个样本。
        self.samples = self.steps * self._batch_size * self._dp_world_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames
        # 调用方预先注册的计量器名称。update_metrics() 不会触碰这些计量器，
        # 因此在输出字典中回传例如 "loss" 的策略不会覆盖已聚合的计量器。
        self._caller_metrics: set[str] = set(self.metrics)

    def __getattr__(self, name: str) -> int | dict[str, AverageMeter] | AverageMeter | Any:
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.metrics:
            return self.metrics[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            super().__setattr__(name, value)
        elif name in self.metrics:
            self.metrics[name].update(value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def step(self) -> None:
        """
        将依赖于 'step' 的指标向前更新一步。
        """
        self.steps += 1
        self.samples += self._batch_size * self._dp_world_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames

    def update_metrics(self, values: dict[str, Any]) -> None:
        """累积一个标量指标字典，并为每个新键自动注册一个计量器。

        非数值和布尔值会被忽略。
        调用方注册的指标（即传给构造函数的那些）永远不会被覆盖。
        """
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if name in self._caller_metrics:
                continue
            if name not in self.metrics:
                self.metrics[name] = AverageMeter(name, ":.3f", reduction="mean")
            self.metrics[name].update(float(value))

    def reduce_across_ranks(self) -> None:
        """
        在所有分布式进程之间（原地）同步每个 ``reduction`` 不为 ``"none"``
        的指标的运行平均值。

        这是一个集合操作，必须在每个 rank 上调用——通常就在
        记录日志之前。在非分布式运行中它是空操作。没有它，主进程
        报告的指标只反映 rank 0；对于瓶颈类计时（``dataloading_s``、
        ``update_s``……），这意味着最慢工作进程的卡顿将不可见。

        有意使用 Torch 原生实现：指标代码不携带任何 Accelerator 依赖。
        请注意归约跨越的是 WORLD 组——对于与计数无关的平均值这是正确的
        （在一个上下文并行组内损失值相同，因此包含 CP peer 是一次加权后
        的空操作）。
        """
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return

        buckets: dict[str, list[str]] = defaultdict(list)
        for name, meter in self.metrics.items():
            if meter.reduction != "none":
                buckets[meter.reduction].append(name)
        if not buckets:
            return

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        reduce_ops = {
            "mean": dist.ReduceOp.AVG,
            "sum": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
        }
        for reduction, names in buckets.items():
            tensor = torch.tensor([self.metrics[n].avg for n in names], dtype=torch.float32, device=device)
            dist.all_reduce(tensor, op=reduce_ops[reduction])
            for name, value in zip(names, tensor.tolist(), strict=True):
                meter = self.metrics[name]
                # 保持 avg == sum / count，这样稍后对此计量器调用 .update()
                # 时会基于集群视角累积，而不是基于过时的逐 rank 历史。
                meter.avg = value
                meter.sum = value * meter.count

    def __str__(self) -> str:
        display_list = [
            f"step:{format_big_number(self.steps)}",
            # 训练期间见到的样本数
            f"smpl:{format_big_number(self.samples)}",
            # 训练期间见到的 episode 数
            f"ep:{format_big_number(self.episodes)}",
            # 所有唯一样本被见到的次数
            f"epch:{self.epochs:.2f}",
            *[str(m) for m in self.metrics.values()],
        ]
        return " ".join(display_list)

    def to_dict(self, use_avg: bool = True) -> dict[str, int | float]:
        """
        以字典形式返回当前指标值（当 `use_avg=True` 时返回平均值）。
        """
        return {
            "steps": self.steps,
            "samples": self.samples,
            "episodes": self.episodes,
            "epochs": self.epochs,
            **{k: m.avg if use_avg else m.val for k, m in self.metrics.items()},
        }

    def reset_averages(self) -> None:
        """重置各平均计量器。"""
        for m in self.metrics.values():
            m.reset()
