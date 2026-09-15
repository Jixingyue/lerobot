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

"""同步推理引擎：每个控制节拍内联调用一次策略。"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from .base import InferenceEngine, PolicyQuery

logger = logging.getLogger(__name__)


# TODO(Steven)：支持相对动作策略。当前逐节拍的流程每次调用都会刷新
# ``RelativeActionsProcessorStep._last_state``，于是后续节拍弹出的缓存动作块
# 会被重新锚定到*当前*机器人状态，绝对目标会在整个动作块内漂移。
# 目前相对动作策略会在构建上下文时被拒绝；RTC 对整个动作块进行后处理，
# 因而不受影响。
#
# 候选修复方案：通过 ``predict_action_chunk`` 驱动策略，并维护一个本地的
# 后处理动作 FIFO 队列来提供动作。这从根本上消除了漂移，也省去了逐节拍的
# 前/后处理工作，但会绕过 ``select_action``——需要为以下情况提供兜底：
# SAC（会抛异常）、ACT 时序集成（集成器位于 ``select_action`` 中），
# 以及 Diffusion 系列（观测历史队列是作为 ``select_action`` 的副作用
# 被填充的）。


class SyncInferenceEngine(InferenceEngine):
    """内联同步推理：每次调用计算一个动作。

    ``get_action`` 会在给定的观测帧上运行完整的策略流水线
    （前/后处理器 + ``select_action``），并返回一个 CPU 动作张量，
    其顺序已按数据集的动作键重新排列。
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
    ) -> None:
        super().__init__(task=task)
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d)",
            self._device,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        """没有需要启动的后台资源。"""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """没有需要停止的后台资源。"""
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """重置策略以及前/后处理器。"""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        # 策略刚刚被重置，因此待处理的任务变更没有任何陈旧内容需要清空。
        self._discard_task_change()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """在 ``obs_frame`` 上运行完整推理流水线，并返回动作张量。"""
        if obs_frame is None:
            return None
        # 这里有意使用浅拷贝：调用方（`send_next_action`）每个节拍都会通过
        # ``build_dataset_frame`` 重新构建 ``obs_frame``，因此其中的
        # 张量/数组值不会与任何其他读取方共享。
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        task, task_changed = self._take_task()
        with torch.inference_mode(), autocast_ctx:
            if task_changed:
                # 分块（chunking）策略会把在旧指令下计算出的动作排入队列，因此将
                # 其丢弃，让新指令在本拍生效。这比 ``policy.reset`` 更窄：
                # 观测历史和其他回合状态都会保留。
                logger.info("Task changed to '%s' — dropping precomputed actions", task)
                self._policy.drop_queued_actions()
            observation = prepare_observation_for_inference(observation, self._device, task, self._robot_type)
            observation = self._preprocessor(observation)
            action = self._policy.select_action(observation)
            action = self._postprocessor(action)
        action_tensor = action.squeeze(0).cpu()

        # 按数据集的动作顺序重新排列，以便调用方在不同后端之间可以统一地
        # 处理返回的张量。
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        # ``task`` 是推理前的快照：推理过程中到达的 /subtask 不应
        # 重新标记本动作。
        self._set_dispatched_task(task)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])

    # ------------------------------------------------------------------
    # 文本查询
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """当策略具有文本头（text head）时为 True。"""
        return self._policy.supports_text_generation()

    @property
    def control_thread_owns_policy(self) -> bool:
        """推理在控制线程上内联运行，因此查询也在该线程上处理。"""
        return True

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """在当前观测上运行策略的文本头。"""
        obs_frame = build_dataset_frame(self._dataset_features, obs_processed, prefix=OBS_STR)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        # 当前任务，读取时不消费"任务已变更"这一边沿信号（动作路径需要它）。
        task = self.task
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(obs_frame, self._device, task, self._robot_type)
            observation = self._mark_query(observation, query)
            # 复用动作路径的预处理器只有在其各个步骤每次调用均无状态时才安全；
            # 那个有状态的步骤（启用的 RelativeActionsProcessorStep）会在构建
            # 上下文时就为此后端被拒绝。
            observation = self._preprocessor(observation)
            # 不做 str() 强制转换：_service_query 会校验返回值。
            return self._policy.generate_text(observation)
