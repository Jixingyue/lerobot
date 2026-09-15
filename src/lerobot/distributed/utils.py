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
"""Rank 工具以及 `prepare()` 之后的分片收尾。"""

import logging
from typing import TYPE_CHECKING

import torch.distributed as dist
from torch import nn

if TYPE_CHECKING:
    from lerobot.distributed.parallel_dims import ParallelDims


def is_main_process() -> bool:
    """在拥有仅限 rank 0 的副作用（文件写入、上传、日志记录）的进程上为 True。

    有意使用 Torch 原生实现：持久化代码不能依赖 `Accelerator` 句柄 ——
    `_save_pretrained` 和 hub 发布器运行的上下文中没有它。在非分布式
    运行中，每个进程都是主进程。

    Returns:
        bool: 当本进程是 rank 0 或未初始化进程组时为 True。
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def strip_accelerate_cp_hooks(model: nn.Module) -> int:
    """从每个模块上移除 accelerate 的上下文并行 forward-pre-hook。

    当声明了 `cp_size > 1` 时，`accelerator.prepare()` 会无条件地附加 hook，
    将 `*self_attn` 模块的任何 `attention_mask` kwarg 静默替换为 `is_causal=True`
    （`accelerate.big_modeling._attach_context_parallel_hooks`）—— 对于使用非因果
    注意力的策略，这会破坏 mask。LeRobot 自己实现 CP，从不进入 accelerate 的 CP
    上下文，因此这些 hook 纯属隐患。可通过其定义模块确定性地识别；
    版本金丝雀固定了该标识。

    Args:
        model (nn.Module): 要移除 hook 的已 prepare 模型（会访问所有子模块）。

    Returns:
        int: 移除的 hook 数量。
    """
    removed = 0
    for module in model.modules():
        for hook_id, hook in list(module._forward_pre_hooks.items()):
            if getattr(hook, "__module__", None) == "accelerate.big_modeling":
                del module._forward_pre_hooks[hook_id]
                module._forward_pre_hooks_with_kwargs.pop(hook_id, None)
                removed += 1
    return removed


def finalize_sharded_policy(policy: nn.Module, parallel_dims: "ParallelDims") -> None:
    """分片正确性协议，在 `accelerator.prepare()` 之后立即应用一次。

    1. 移除 accelerate 的 CP mask hook（仅在声明了 cp > 1 时才附加）。
    2. 注册策略的非 `forward` 入口点（`_fsdp_forward_methods`），使 FSDP2
       在 `select_action` 等方法周围取消分片参数 —— 没有它，对分片策略的
       任何推理式调用都会因混合 Tensor/DTensor 而崩溃。

    对 DDP/单进程运行为空操作。

    Args:
        policy (nn.Module): `accelerator.prepare()` 返回的策略。
        parallel_dims (ParallelDims): 本次运行解析后的拓扑；决定是否应用该协议。
    """
    if not parallel_dims.is_sharded:
        return
    if parallel_dims.cp_size > 1:
        removed = strip_accelerate_cp_hooks(policy)
        logging.info("Stripped %d accelerate context-parallel attention-mask hooks.", removed)

    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method

    if isinstance(policy, FSDPModule):
        for method_name in getattr(type(policy), "_fsdp_forward_methods", ()):
            if callable(getattr(policy, method_name, None)):
                register_fsdp_forward_method(policy, method_name)
