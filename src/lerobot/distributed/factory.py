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
"""`Accelerator` 工厂 —— 唯一配置 accelerate 的地方。

`torchrun` 是启动器；所有 accelerate 参数都来自 `TrainPipelineConfig`
（`cfg.parallelism` + `cfg.accelerator`），因此仅凭 `train_config.json`
即可复现一次运行。不带 `--config_file` 的 `accelerate launch` 保持等价
（该模式下它只设置 rendezvous 环境变量）；yaml 流程已被取代。
"""

import os
from typing import TYPE_CHECKING

from lerobot.configs.parallelism import world_size_from_env
from lerobot.configs.train import TrainPipelineConfig

if TYPE_CHECKING:
    from accelerate import Accelerator

    from lerobot.policies.pretrained import PreTrainedPolicy

# 以下环境变量会让 `accelerate launch --config_file`（或残留的 shell）
# 在配置系统背后配置 accelerate，使 train_config.json 无法如实反映实际运行。
_ACCELERATE_ENV_VARS = (
    "ACCELERATE_USE_FSDP",
    "ACCELERATE_USE_PARALLELISM_CONFIG",
    "ACCELERATE_GRADIENT_ACCUMULATION_STEPS",
)
_ENV_OVERRIDE = "LEROBOT_ALLOW_ACCELERATE_ENV"


def guard_against_env_interference() -> None:
    """当设置了用于配置 accelerate 的环境变量时，硬性报错。

    被环境变量静默覆盖的"可复现"配置比直接停止更糟糕：从旧的
    `accelerate launch --config_file fsdp.yaml` 流程迁移的用户会得到精确的错误提示，
    而不是一个说谎的配置。设置 LEROBOT_ALLOW_ACCELERATE_ENV=1 以确认并继续。

    Raises:
        RuntimeError: 如果设置了任何用于配置 accelerate 的环境变量，
            且未设置 LEROBOT_ALLOW_ACCELERATE_ENV 覆盖。
    """
    if os.environ.get(_ENV_OVERRIDE):
        return
    offending = sorted(name for name in _ACCELERATE_ENV_VARS if name in os.environ)
    if offending:
        raise RuntimeError(
            f"Accelerate-configuring environment variables are set: {', '.join(offending)}. "
            "LeRobot manages accelerate exclusively through TrainPipelineConfig "
            "(--parallelism.* / --accelerator.*); launch with plain torchrun and remove these "
            "variables (the `accelerate launch --config_file` flow is superseded), or set "
            f"{_ENV_OVERRIDE}=1 to acknowledge that they may override your config."
        )


def make_accelerator(cfg: TrainPipelineConfig) -> "Accelerator":
    """根据已启动的 world 解析拓扑，并构建 `Accelerator`。

    必须在每个进程中运行一次，且在任何其他组件需要设备或进程组之前
    （`Accelerator.__init__` 会初始化两者并构建设备网格）。

    Args:
        cfg (TrainPipelineConfig): 完整的训练配置；`cfg.parallelism` 会根据
            已启动的 world size 原地解析，`cfg.accelerator` 用于构建结果。

    Returns:
        Accelerator: 配置好的 accelerator，设备和进程组已初始化。

    Raises:
        ValueError: 如果 `cfg.checkpoint_format` 需要 DCP，但拓扑解析为
            非分片运行。
    """
    guard_against_env_interference()
    cfg.parallelism.resolve(world_size_from_env())
    # 解析时的格式检查是针对声明的并行度运行的，其中 dp_shard=-1
    # 哨兵值被视为分片；但它可能解析为非分片运行（例如 world size 为 1 时的 -1）。
    # 针对具体的并行度重新检查，使记录的格式永远不会与检查点
    # 实际包含的产物不符。
    if cfg.checkpoint_format.wants_dcp and not cfg.parallelism.is_sharded:
        raise ValueError(
            f"checkpoint_format={cfg.checkpoint_format.value} requires a sharded run, but the "
            f"topology resolved to a non-sharded one (dp_replicate={cfg.parallelism.dp_replicate}, "
            f"dp_shard={cfg.parallelism.dp_shard}); non-sharded checkpoints are always safetensors."
        )
    return cfg.accelerator.build(
        cfg.parallelism,
        cpu=cfg.trainable_config.device == "cpu",
    )


def set_fsdp_wrap_modules(accelerator: "Accelerator", policy: "PreTrainedPolicy") -> None:
    """在 `accelerator.prepare()` 之前，将 FSDP wrap 单元类名解析到插件上。

    解析顺序：用户覆盖（`--accelerator.fsdp.wrap_modules`，已在插件上）
    -> 策略的 `_fsdp_wrap_modules` 声明 -> 硬性报错。仅包装根节点 —— 即不存在
    wrap 来源时的静默默认行为 —— 绝不被接受：它会悄然放弃所有分片带来的内存节省。

    对于基于大小的策略（`--accelerator.fsdp.min_num_params`，不需要类名）
    以及非分片运行（无 fsdp 插件）为空操作。

    Args:
        accelerator (Accelerator): 其 FSDP 插件将接收 wrap 单元类名的 accelerator。
        policy (PreTrainedPolicy): 其类可能声明了 `_fsdp_wrap_modules` 的可训练对象。

    Raises:
        ValueError: 如果配置了分片的基于类的包装，但用户覆盖和策略声明
            都未提供 wrap 单元类名。
    """
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if plugin is None or plugin.min_num_params:
        return
    if plugin.transformer_cls_names_to_wrap:  # 用户覆盖，在构建时设置
        return
    # 使用 getattr 而非属性访问：非策略类可训练对象（没有 `_fsdp_wrap_modules` 属性）
    # 必须到达下方可操作的错误，而不是 AttributeError。
    declared = getattr(type(policy), "_fsdp_wrap_modules", None)
    if not declared:
        raise ValueError(
            f"Policy '{type(policy).__name__}' declares no FSDP wrap units. Sharded training "
            "requires wrap-unit class names: set --accelerator.fsdp.wrap_modules='[\"MyBlock\"]' "
            "(or --accelerator.fsdp.min_num_params for a size-based policy), or declare "
            "`_fsdp_wrap_modules` on the policy class."
        )
    plugin.transformer_cls_names_to_wrap = list(declared)
