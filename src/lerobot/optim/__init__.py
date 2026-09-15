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

from .optimizers import (
    AdamConfig as AdamConfig,
    AdamWConfig as AdamWConfig,
    MultiAdamConfig as MultiAdamConfig,
    OptimizerConfig as OptimizerConfig,
    SGDConfig as SGDConfig,
    XVLAAdamWConfig as XVLAAdamWConfig,
    load_optimizer_state,
    save_optimizer_state,
)
from .schedulers import (
    CosineDecayWithWarmupSchedulerConfig as CosineDecayWithWarmupSchedulerConfig,
    DiffuserSchedulerConfig as DiffuserSchedulerConfig,
    LRSchedulerConfig as LRSchedulerConfig,
    VQBeTSchedulerConfig as VQBeTSchedulerConfig,
    load_scheduler_state,
    save_scheduler_state,
)

# 注意：make_optimizer_and_scheduler 有意不在此处重新导出，
# 以避免循环依赖（它会导入 lerobot.configs.train 和 lerobot.policies）。
# 请直接导入：``from lerobot.optim.factory import make_optimizer_and_scheduler``

__all__ = [
    # 优化器配置
    "AdamConfig",
    "AdamWConfig",
    "MultiAdamConfig",
    "OptimizerConfig",
    "SGDConfig",
    "XVLAAdamWConfig",
    # 调度器配置
    "CosineDecayWithWarmupSchedulerConfig",
    "DiffuserSchedulerConfig",
    "LRSchedulerConfig",
    "VQBeTSchedulerConfig",
    # 状态管理
    "load_optimizer_state",
    "load_scheduler_state",
    "save_optimizer_state",
    "save_scheduler_state",
]
