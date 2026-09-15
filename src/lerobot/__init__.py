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

"""
LeRobot -- 用于真实世界机器人的 PyTorch 库。

提供数据集、预训练策略，以及用于训练、评估、数据收集和机器人控制的工具。
与 Hugging Face Hub 集成，用于模型和数据集共享。

基础安装刻意保持轻量。特定功能的依赖通过可选扩展来控制：

    pip install 'lerobot[dataset]'       # 数据集加载与创建
    pip install 'lerobot[training]'      # 训练循环 + wandb
    pip install 'lerobot[hardware]'      # 真实机器人控制
    pip install 'lerobot[core_scripts]'  # 数据集 + 硬件 + 可视化（录制、回放、标定等）
    pip install 'lerobot[all]'           # 全部功能
"""

from lerobot.__version__ import __version__

# 将可选扩展映射到它们解锁的 CLI 入口点。
available_extras: dict[str, list[str]] = {
    "dataset": ["lerobot-dataset-viz", "lerobot-imgtransform-viz", "lerobot-edit-dataset"],
    "training": ["lerobot-train"],
    "hardware": [
        "lerobot-calibrate",
        "lerobot-find-port",
        "lerobot-find-cameras",
        "lerobot-find-joint-limits",
        "lerobot-setup-motors",
    ],
    "core_scripts": ["lerobot-record", "lerobot-replay", "lerobot-teleoperate"],
    "evaluation": ["lerobot-eval"],
}

__all__ = ["__version__", "available_extras"]
