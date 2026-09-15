#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""实时分块（Real-Time Chunking, RTC）的动作队列管理。

本模块提供 ActionQueue，一个用于在实时控制场景中管理动作块的
线程安全队列。它同时支持启用 RTC 和未启用 RTC 的模式，
处理动作合并和剩余动作跟踪。
"""

import logging
from threading import Lock

import torch
from torch import Tensor

from .configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """用于在实时控制中管理动作块的线程安全队列。

    该队列处理两种类型的动作序列：
    - 原始动作：供 RTC 用于计算前一块的剩余动作
    - 处理后动作：可供机器人执行的后处理动作

    该队列以两种模式运行：
    1. 启用 RTC：用新动作替换整个队列，并考虑推理延迟
    2. 未启用 RTC：将新动作追加到队列中，保持连续性

    每个入队动作都在同一把锁下、以锁步（lockstep）方式，与其所属块
    推理生成时所用的任务一起标注；``get_with_task`` 返回该标注。

    参数：
        cfg (RTCConfig): 实时分块行为的配置。

    属性：
        queue (Tensor | None): 用于机器人执行的已处理动作 (time_steps, action_dim)。
        original_queue (Tensor | None): 用于 RTC 计算的原始动作 (time_steps, action_dim)。
        last_index (int): 队列中当前的消费索引。
    """

    def __init__(self, cfg: RTCConfig):
        """初始化动作队列。

        参数：
            cfg: 控制队列行为的 RTC 配置。
        """
        self.queue = None  # 用于机器人执行的已处理动作
        self.original_queue = None  # 用于 RTC 的原始动作
        self._task_queue: list[str | None] | None = None
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> Tensor | None:
        """从队列中获取下一个动作。

        返回：
            Tensor | None: 下一个动作 (action_dim,)，如果队列为空则返回 None。
                          返回一个克隆，以防止外部修改。
        """
        queued = self.get_with_task()
        return None if queued is None else queued[0]

    def get_with_task(self) -> tuple[Tensor, str | None] | None:
        """获取下一个动作，以及生成其所属块的任务。"""
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            if self._task_queue is not None and len(self._task_queue) != len(self.queue):
                # 不匹配意味着某次修改破坏了动作/任务的锁步关系。
                raise RuntimeError(
                    f"ActionQueue task labels out of sync with actions "
                    f"({len(self._task_queue)} labels for {len(self.queue)} actions) — "
                    "a queue mutation broke the action/task lockstep invariant"
                )
            action = self.queue[self.last_index]
            task = None if self._task_queue is None else self._task_queue[self.last_index]
            self.last_index += 1
            return action.clone(), task

    def clear(self) -> None:
        """清空已入队的动作并重置消费索引。"""
        with self.lock:
            self.queue = None
            self.original_queue = None
            self._task_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        """获取队列中剩余动作的数量。

        返回：
            int: 未消费动作的数量。
        """
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        """检查队列是否为空。

        返回：
            bool: 如果没有剩余动作则为 True，否则为 False。
        """
        with self.lock:
            if self.queue is None:
                return True
            return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        """获取当前的动作消费索引。

        返回：
            int: 下一个待消费动作的索引。
        """
        with self.lock:
            return self.last_index

    def get_left_over(self) -> Tensor | None:
        """获取供 RTC prev_chunk_left_over 使用的剩余原始动作。

        这些是当前块中尚未消费的动作，将被 RTC 用于
        计算下一块的修正。

        返回：
            Tensor | None: 剩余的原始动作 (remaining_steps, action_dim)，
                          如果不存在原始队列则返回 None。
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].clone()

    def get_processed_left_over(self) -> Tensor | None:
        """获取剩余的已处理动作（即机器人当前正在执行的动作）。

        返回：
            Tensor | None: 剩余的已处理动作 (remaining_steps, action_dim)，
                如果不存在已处理队列则返回 None。
        """
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index :].clone()

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = None,
        *,
        task: str | None = None,
    ):
        """将新动作合并到队列中。

        该方法的行为因 RTC 模式而异：
        - 启用 RTC：替换队列，并考虑推理延迟
        - 未启用 RTC：追加到队列中，保持连续性

        参数：
            original_actions: 来自策略的未处理动作 (time_steps, action_dim)。
            processed_actions: 供机器人使用的后处理动作 (time_steps, action_dim)。
            real_delay: 推理延迟的时间步数。
            action_index_before_inference: 推理开始前的索引，用于校验。
            task: 用于生成传入动作块的指令。
        """
        with self.lock:
            delay = self._check_and_resolve_delays(real_delay, action_index_before_inference)

            if self.cfg.enabled:
                self._replace_actions_queue(original_actions, processed_actions, delay, task)
                return

            self._append_actions_queue(original_actions, processed_actions, task)

    def _replace_actions_queue(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        task: str | None,
    ):
        """用新动作替换队列（RTC 模式）。

        丢弃前 `real_delay` 个动作，因为它们对应推理期间
        机器人正在执行先前动作的那段时间。

        参数：
            original_actions: 来自策略的未处理动作。
            processed_actions: 供机器人使用的后处理动作。
            real_delay: 因推理延迟而需要跳过的时间步数。
            task: 生成该块的指令；用于标注每个入队动作。
        """
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = original_actions[clamped_delay:].clone()
        self.queue = processed_actions[clamped_delay:].clone()
        self._task_queue = [task] * len(self.queue)

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}, clamped_delay: {clamped_delay}")

        self.last_index = 0

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor, task: str | None):
        """将新动作追加到队列中（非 RTC 模式）。

        移除已消费的动作并追加新动作，在不替换的情况下
        保持队列的连续性。

        参数：
            original_actions: 来自策略的未处理动作。
            processed_actions: 供机器人使用的后处理动作。
            task: 生成所追加块的指令；已入队的动作保留其原有标注。
        """
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            self._task_queue = [task] * len(self.queue)
            return

        existing_tasks = self._task_queue or [None] * len(self.queue)
        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]
        self._task_queue = existing_tasks[self.last_index :] + [task] * len(processed_actions)

        self.last_index = 0

    def _check_and_resolve_delays(
        self, real_delay: int, action_index_before_inference: int | None = None
    ) -> int:
        """校验计算得到的延迟是否符合预期。

        将依据推理延迟计算出的延迟与推理期间实际
        消费的动作数量进行比较。

        参数：
            real_delay: 依据推理延迟计算出的延迟。
            action_index_before_inference: 推理开始时的动作索引。

        返回：
            int: 要使用的延迟。
        """
        effective_delay = max(0, real_delay)

        if action_index_before_inference is not None:
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            if indexes_diff != real_delay:
                logger.info(
                    "Indexes diff is not equal to real delay. indexes_diff=%d, real_delay=%d",
                    indexes_diff,
                    real_delay,
                )
                return real_delay

        return effective_delay
