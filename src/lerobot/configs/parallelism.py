#!/usr/bin/env python

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
"""用于分布式训练和推理的声明式进程拓扑。

mesh 约定（规范的行优先 rank 布局，最外层在前）::

    (dp_replicate, dp_shard, ring, ulysses)

- ``dp_replicate x dp_shard`` 是数据并行世界：HSDP 在 ``dp_replicate`` 上复制，
  在 ``dp_shard`` 上分片参数。FSDP2 的实际分片组会并入上下文并行
  （``dp_shard x ring x ulysses``），与 accelerate 的 ``dp_shard_cp`` 展平
  以及 torchtitan 的 ``fsdp`` 轴一致。
- ``ring`` 是外层、``ulysses`` 是内层的上下文并行维度
  （diffusers 约定：ulysses 的 all-to-all 交换在相邻的、通常通过
  NVLink 连接的 rank 之间进行）。
- ``cfg_parallel``（classifier-free-guidance 并行）是一个分支并行的、
  仅用于推理的维度，位于 dp 和序列维度之间。它永远不会
  影响权重分片或检查点。

本模块是纯配置：draccus 可以通过 CLI 和 ``train_config.json`` 往返的
简单类型 dataclass。运行时对象（设备 mesh、进程组）位于
:mod:`lerobot.distributed`。
"""

import os
from dataclasses import dataclass, field


@dataclass
class ContextParallelConfig:
    """Ring x Ulysses 上下文并行（注意力的序列并行）。

    在本版本中，两个度数都是配置的占位符：CP 引擎尚未实现，
    启用任一度数 > 1 都会在配置验证时快速失败。这些字段现在存在，
    是为了在引擎落地时，CLI 接口、检查点元数据和 mesh 计算保持稳定。
    """

    ring_degree: int = 1
    ulysses_degree: int = 1

    def __post_init__(self) -> None:
        """验证声明的上下文并行度数。

        Raises:
            ValueError: 如果 ``ring_degree`` 或 ``ulysses_degree`` < 1。
        """
        if self.ring_degree < 1 or self.ulysses_degree < 1:
            raise ValueError(
                f"Context-parallel degrees must be >= 1, got ring_degree={self.ring_degree}, "
                f"ulysses_degree={self.ulysses_degree}."
            )

    @property
    def size(self) -> int:
        """完整序列被分片到的总 rank 数。"""
        return self.ring_degree * self.ulysses_degree


@dataclass
class ParallelismConfig:
    """所有并行维度的度数。不变量：它们的乘积等于世界大小。

    降级完全通过度数来表达（没有模式标志）：

    - 单进程：所有度数为 1；
    - DDP：``dp_replicate == world_size``（当所有分片字段保持默认值时自动填充——
      普通的 ``torchrun`` 保持现有的开箱即用行为）；
    - FSDP：``dp_shard > 1``（或 ``-1`` 表示将剩余的世界大小填入分片维度）；
    - HSDP：``dp_replicate > 1`` 且 ``dp_shard > 1``。

    ``resolve()`` 在世界大小已知后将声明的度数转换为具体值，
    并且是唯一执行世界大小等式的地方。它由
    :func:`lerobot.distributed.factory.make_accelerator` 调用；在此之前配置处于惰性状态。
    """

    dp_replicate: int = 1
    # -1 是一个显式的选择加入哨兵值：在 world_size // (dp_replicate * cp) 上分片。
    dp_shard: int = 1
    context_parallel: ContextParallelConfig = field(default_factory=ContextParallelConfig)
    # Classifier-free-guidance 并行 —— 仅用于推理（cosmos/vllm-omni 先例：
    # cond/uncond 分支位于不同的 rank）。保留给 serving 轮次；训练时
    # 验证其为 1。有意义的值是 1 或 2（Cosmos3 有两个 CFG 分支）。
    cfg_parallel: int = 1

    def __post_init__(self) -> None:
        """验证声明的度数（仅进行与世界大小无关的检查）。

        Raises:
            ValueError: 如果 ``dp_replicate`` < 1，``dp_shard`` 既不 >= 1 也不是
                ``-1`` 推断哨兵值，或 ``cfg_parallel`` 不是 1 或 2。
        """
        if self.dp_replicate < 1:
            raise ValueError(f"dp_replicate must be >= 1, got {self.dp_replicate}.")
        if self.dp_shard < 1 and self.dp_shard != -1:
            raise ValueError(f"dp_shard must be >= 1, or -1 to infer, got {self.dp_shard}.")
        if self.cfg_parallel not in (1, 2):
            raise ValueError(f"cfg_parallel must be 1 or 2, got {self.cfg_parallel}.")

    @property
    def cp_size(self) -> int:
        """上下文并行总大小（``ring_degree * ulysses_degree``）。"""
        return self.context_parallel.size

    @property
    def is_sharded(self) -> bool:
        """当运行使用 FSDP2（参数已分片）时为 True；用于选择分片引擎路径。"""
        return self.dp_shard != 1 or self.cp_size > 1

    @property
    def is_replicated_only(self) -> bool:
        """对于普通 DDP（权重复制，无分片）为 True。"""
        return not self.is_sharded and self.dp_replicate > 1

    @property
    def dp_world_size(self) -> int:
        """不同数据并行 worker 的数量（批次按此数量分片）。

        Returns:
            int: ``dp_replicate * dp_shard``。

        Raises:
            RuntimeError: 如果在 ``dp_shard`` 仍为 ``-1`` 哨兵值时访问，
                即在 :meth:`resolve` 将度数绑定到世界大小之前。
        """
        if self.dp_shard == -1:
            raise RuntimeError("dp_world_size is undefined before resolve() fills dp_shard=-1.")
        return self.dp_replicate * self.dp_shard

    def resolve(self, world_size: int) -> None:
        """将声明的度数绑定到具体的世界大小（幂等）。

        填充 ``dp_shard=-1`` 哨兵值，为 DDP 降级自动填充 ``dp_replicate``，
        并强制 ``dp_replicate * dp_shard * cp == world_size``，失败时
        回显所有度数。

        Args:
            world_size (int): 启动的进程总数（torchrun 的 ``WORLD_SIZE``）。

        Raises:
            ValueError: 如果上下文并行度数 > 1（CP 引擎尚未实现），
                如果由于 ``world_size`` 不能被 ``dp_replicate * cp`` 整除而
                无法推断 ``dp_shard=-1``，或者如果解析后的度数乘积
                不等于 ``world_size``。
        """
        if self.cp_size > 1:
            raise ValueError(
                "Context parallelism is not implemented yet: ring_degree and ulysses_degree "
                "must be 1. The fields are reserved for the CP engine round."
            )
        if self.is_sharded:
            if self.dp_shard == -1:
                self.dp_shard, remainder = divmod(world_size, self.dp_replicate * self.cp_size)
                if remainder or self.dp_shard < 1:
                    raise ValueError(
                        f"Cannot infer dp_shard: world_size={world_size} is not divisible by "
                        f"dp_replicate={self.dp_replicate} * cp={self.cp_size}."
                    )
        elif self.dp_replicate == 1:
            # 多进程启动时配置未被改动：填充 DDP 降级。
            self.dp_replicate = world_size
        total = self.dp_replicate * self.dp_shard * self.cp_size
        if total != world_size:
            raise ValueError(
                f"Parallelism degrees do not multiply to the world size: dp_replicate="
                f"{self.dp_replicate} * dp_shard={self.dp_shard} * ring="
                f"{self.context_parallel.ring_degree} * ulysses="
                f"{self.context_parallel.ulysses_degree} = {total} != WORLD_SIZE={world_size}."
            )


def world_size_from_env() -> int:
    """torchrun 设置的世界大小（非分布式启动时为 1）。

    Returns:
        int: ``WORLD_SIZE`` 环境变量，未设置时为 1。
    """
    return int(os.environ.get("WORLD_SIZE", "1"))
