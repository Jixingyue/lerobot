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
"""LeRobot 的分布式训练运行时。

本包负责将 :class:`lerobot.configs.parallelism.ParallelismConfig` 中的声明式拓扑
转化为可运行引擎所需的一切：网格计算
（:class:`~lerobot.distributed.parallel_dims.ParallelDims`）、
`Accelerator` 工厂（:func:`~lerobot.distributed.factory.make_accelerator`）、
分片感知的检查点辅助函数，以及小型 rank 工具。

初始化顺序约定（规范性）：
CP dispatch 安装 -> 激活检查点 -> torch.compile ->
``fully_shard``/DDP（通过 ``accelerator.prepare``）-> 优化器重新绑定。
目前只有最后两步处于激活状态；CP/AC/compile 是已配置的占位符，将在后续轮次中接入。
"""

from .factory import guard_against_env_interference, make_accelerator, set_fsdp_wrap_modules
from .parallel_dims import ParallelDims
from .utils import finalize_sharded_policy, is_main_process, strip_accelerate_cp_hooks

__all__ = [
    "ParallelDims",
    "finalize_sharded_policy",
    "guard_against_env_interference",
    "is_main_process",
    "make_accelerator",
    "set_fsdp_wrap_modules",
    "strip_accelerate_cp_hooks",
]
