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
"""从声明式 :class:`ParallelismConfig` 派生的运行时网格计算。

`ParallelDims` 是训练脚本获取拓扑派生数值（数据并行 world size 和 rank、
样本核算的输入）的唯一事实来源，并且 —— 一旦 CP 引擎落地 —— 它还将拥有
LeRobot 的私有 ``(dp_replicate, dp_shard, ring, ulysses)`` 网格。它是一个运行时
对象，永远不会被序列化（它派生自的配置才是写入 ``train_config.json`` 的内容）。
"""

from dataclasses import dataclass

import torch.distributed as dist

from lerobot.configs.parallelism import ParallelismConfig


@dataclass(frozen=True)
class ParallelDims:
    """绑定到 world size 的具体并行度（标准行优先 rank 布局）。"""

    dp_replicate: int
    dp_shard: int
    ring: int
    ulysses: int
    world_size: int
    device_type: str

    @classmethod
    def from_config(cls, cfg: ParallelismConfig, world_size: int, device_type: str) -> "ParallelDims":
        """将*已解析*的配置绑定到实际运行时 world size（此处进行交叉校验）。

        Args:
            cfg (ParallelismConfig): 声明式拓扑，已通过
                `ParallelismConfig.resolve(world_size)` 解析。
            world_size (int): 已启动的 world size，声明的并行度乘积必须等于它。
            device_type (str): 支撑网格的加速器设备类型（例如 "cuda"）。

        Returns:
            ParallelDims: 绑定到此 world 的具体并行度。

        Raises:
            ValueError: 如果配置未解析（`dp_shard == -1`），或其并行度乘积
                不等于 `world_size`。
        """
        total = cfg.dp_replicate * cfg.dp_shard * cfg.cp_size
        if cfg.dp_shard == -1 or total != world_size:
            raise ValueError(
                f"ParallelismConfig is not resolved against this world: dp_replicate="
                f"{cfg.dp_replicate} * dp_shard={cfg.dp_shard} * cp={cfg.cp_size} != "
                f"world_size={world_size}. Call ParallelismConfig.resolve(world_size) first "
                "(make_accelerator does this)."
            )
        return cls(
            dp_replicate=cfg.dp_replicate,
            dp_shard=cfg.dp_shard,
            ring=cfg.context_parallel.ring_degree,
            ulysses=cfg.context_parallel.ulysses_degree,
            world_size=world_size,
            device_type=device_type,
        )

    @property
    def cp_size(self) -> int:
        """总的上下文并行度（`ring * ulysses`）。"""
        return self.ring * self.ulysses

    @property
    def is_sharded(self) -> bool:
        """参数是否被分片（`dp_shard > 1` 或存在任何上下文并行）。"""
        return self.dp_shard > 1 or self.cp_size > 1

    @property
    def dp_world_size(self) -> int:
        """不同数据并行 worker 的数量 —— 所有样本核算的除数。"""
        return self.dp_replicate * self.dp_shard

    @property
    def dp_rank(self) -> int:
        """本进程的数据并行坐标（CP 同伴共享同一个 dp_rank）。

        在标准行优先布局下且 (ring, ulysses) 位于最内层时，CP 同伴是
        连续的全局 rank，因此 dp 坐标是除以 cp_size 的整数商 ——
        与 accelerate 的网格感知 dataloader 所应用的算术相同。
        """
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        return global_rank // self.cp_size

    def cp_mesh(self) -> None:
        """供 CP 引擎使用的私有 (ring, ulysses) 网格 —— 为 CP 轮次保留。

        Raises:
            NotImplementedError: 始终抛出 —— 上下文并行尚未实现。
        """
        raise NotImplementedError(
            "Context parallelism is not implemented yet; ParallelDims.cp_mesh is reserved for "
            "the CP engine round (a private mesh aligned with accelerate's cp block)."
        )
