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

"""Real-Time Chunking 推理引擎。

后台线程通过 :meth:`policy.predict_action_chunk` 异步生成动作块（action
chunks）。主控制循环通过 ``get_action`` 轮询获取下一个就绪的动作；
观测则沿相反方向通过 ``notify_observation`` 传入。
"""

from __future__ import annotations

import inspect
import logging
import math
import time
import traceback
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionQueue, LatencyTracker, reanchor_relative_rtc_prefix
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine, PolicyQuery

logger = logging.getLogger(__name__)

# RTC 循环在暂停、空闲或因队列已满而受到反压时的休眠时长。
_RTC_IDLE_SLEEP_S: float = 0.01
# 瞬时推理错误之间的退避时间（按每次连续失败计）。
_RTC_ERROR_RETRY_DELAY_S: float = 0.5
# 在放弃并传播关闭信号之前，所能容忍的连续瞬时错误次数。
_RTC_MAX_CONSECUTIVE_ERRORS: int = 10
# 在判定该延迟无法支持之前，所能容忍的连续不可用 trained-RTC 动作块数量。
_RTC_MAX_CONSECUTIVE_DISCARDS: int = 5
# stop() 时 join RTC 线程的硬性超时时间。
_RTC_JOIN_TIMEOUT_S: float = 3.0


class _FatalRTCInferenceError(RuntimeError):
    """即使重试也无法恢复正常的 RTC 错误的基类。"""


class _TrainedRTCDelayExceededError(_FatalRTCInferenceError):
    """当测得的延迟持续超出 trained RTC 检查点所支持的范围时抛出。"""


# ---------------------------------------------------------------------------
# RTC 辅助函数
# ---------------------------------------------------------------------------


def supports_rtc_inference(policy: PreTrainedPolicy) -> bool:
    """策略是否声明支持 RTC 且接受 RTC 的调用形式。"""
    supports_rtc = getattr(policy, "supports_rtc", None)
    if not callable(supports_rtc) or not supports_rtc():
        return False

    try:
        inspect.signature(policy.predict_action_chunk).bind(
            object(),
            inference_delay=0,
            prev_chunk_left_over=None,
        )
    except (TypeError, ValueError):
        return False
    return True


def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """将 RTC 前缀动作填充或截断到固定长度，以保证编译后推理的稳定性。"""
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] tensor, got shape={tuple(prev_actions.shape)}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


def _trained_rtc_chunk_can_merge(
    *,
    conditioned_delay: int,
    measured_delay: int,
    training_max_delay: int,
    has_previous_actions: bool,
) -> bool:
    """一个 trained RTC 动作块是否仍然覆盖实际经过的重叠区间。

    动作块不可用有两种情形：要么推理耗时超过了它作为条件的前缀长度，要么实际
    经过的延迟超出了检查点训练时所覆盖的范围。两者本质上都是瞬时的
    （一次延迟尖峰），因此本函数以相同方式上报，让调用方重试；
    只有持续出现一连串不可用的动作块时才是致命的。
    """
    if not has_previous_actions:
        return True
    if measured_delay > training_max_delay:
        return False
    return measured_delay <= conditioned_delay


def _estimate_rtc_delay(
    *,
    latency: float,
    time_per_step: float,
    mode: str,
    training_max_delay: int,
    has_previous_actions: bool,
) -> int:
    """估计重叠区间，并利用训练时的容量来引导（bootstrap）首次转换。"""
    if latency:
        return math.ceil(latency / time_per_step)
    if mode == "trained" and has_previous_actions:
        return training_max_delay
    return 0


def _clamp_trained_rtc_delay(*, conditioned_delay: int, available_steps: int, training_max_delay: int) -> int:
    """将硬性前缀截断（clamp）到检查点和队列都能支撑的范围内。

    超过 ``training_max_delay`` 后，模型从未见过那么长的前缀；而超过
    ``available_steps`` 后，``_normalize_prev_actions_length`` 会对尾部补零，
    于是多出来的那些步会被当作"已提交的动作是零"一样补全。截断可以保证
    动作块仍然可用；如果实际经过的延迟超过了此前缀，
    ``_trained_rtc_chunk_can_merge`` 仍会将其丢弃。
    """
    clamped = min(conditioned_delay, training_max_delay, available_steps)
    if clamped < conditioned_delay:
        logger.warning(
            "Trained RTC wanted a %d-step prefix but the checkpoint supports %d and the queue "
            "holds %d committed actions; conditioning on %d. Raise --inference.queue_threshold "
            "and --inference.rtc.execution_horizon, or retrain with a larger "
            "--policy.rtc_training_max_delay, to keep the full overlap.",
            conditioned_delay,
            training_max_delay,
            available_steps,
            clamped,
        )
    return clamped


# ---------------------------------------------------------------------------
# RTCInferenceEngine
# ---------------------------------------------------------------------------


class RTCInferenceEngine(InferenceEngine):
    """异步 RTC 推理：由后台线程生成动作块。

    ``get_action`` 从共享队列中弹出下一个动作（如果队列为空则
    返回 ``None``）。主循环应当在每个节拍调用 ``notify_observation``，
    并在人工干预阶段前后调用 ``pause``/``resume``。
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        rtc_config: RTCConfig,
        hw_features: dict,
        task: str,
        fps: float,
        device: str | None,
        use_torch_compile: bool = False,
        compile_warmup_inferences: int = 2,
        rtc_queue_threshold: int = 30,
        shutdown_event: Event | None = None,
    ) -> None:
        super().__init__(task=task)
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._rtc_config = rtc_config
        self._hw_features = hw_features
        self._fps = fps
        self._device = device or "cpu"
        self._use_torch_compile = use_torch_compile
        self._compile_warmup_inferences = compile_warmup_inferences
        self._rtc_queue_threshold = rtc_queue_threshold

        self._action_queue: ActionQueue | None = None
        self._obs_holder: dict[str, Any] = {}
        self._obs_lock = Lock()
        # 由 reset() 在 _obs_lock 保护下递增，这样在 reset 之前就已开始推理的
        # 动作块会被丢弃，而不会被合并到全新的队列中。
        self._reset_epoch = 0
        self._policy_active = Event()
        self._compile_warmup_done = Event()
        self._shutdown_event = Event()
        self._rtc_error = Event()
        self._failure_traceback: str | None = None
        self._global_shutdown_event = shutdown_event
        self._rtc_thread: Thread | None = None

        if not self._use_torch_compile:
            self._compile_warmup_done.set()
            logger.info("RTCInferenceEngine initialized (torch.compile disabled, no warmup needed)")
        else:
            logger.info(
                "RTCInferenceEngine initialized (torch.compile enabled, %d warmup inferences)",
                compile_warmup_inferences,
            )

        # 对处理器进行内省，用于相对动作的重新锚定（re-anchoring）。
        self._relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        self._normalizer_step = next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        if self._relative_step is not None:
            if self._relative_step.action_names is None:
                cfg_names = getattr(policy.config, "action_feature_names", None)
                if cfg_names:
                    self._relative_step.action_names = list(cfg_names)
                else:
                    self._relative_step.action_names = [
                        k for k in robot_wrapper.action_features if k.endswith(".pos")
                    ]
            logger.info("Relative actions enabled: RTC prefix will be re-anchored")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """torch.compile 预热完成后为 True（若禁用了编译则立即为 True）。"""
        return self._compile_warmup_done.is_set()

    @property
    def failed(self) -> bool:
        """当 RTC 后台线程因不可恢复的错误而退出时为 True。"""
        return self._rtc_error.is_set()

    @property
    def failure_traceback(self) -> str | None:
        """RTC 线程死亡时捕获的回溯信息（参见 ``failed``）。

        将其作为数据保留而不仅仅是写入日志，以便使用者在有人查看时能够再次呈现它。
        """
        return self._failure_traceback

    @property
    def action_queue(self) -> ActionQueue | None:
        """RTC 线程与主循环之间共享的动作队列。"""
        return self._action_queue

    def start(self) -> None:
        """启动 RTC 后台线程。"""
        self._action_queue = ActionQueue(self._rtc_config)
        self._obs_holder = {
            "obs": None,
            "robot_type": self._robot.robot_type,
        }
        self._shutdown_event.clear()
        self._rtc_thread = Thread(
            target=self._rtc_loop,
            daemon=True,
            name="RTCInference",
        )
        self._rtc_thread.start()
        logger.info("RTC inference thread started")

    def stop(self) -> None:
        """通知 RTC 线程停止并等待其结束。"""
        logger.info("Stopping RTC inference thread...")
        self._shutdown_event.set()
        self._policy_active.clear()
        if self._rtc_thread is not None and self._rtc_thread.is_alive():
            self._rtc_thread.join(timeout=_RTC_JOIN_TIMEOUT_S)
            if self._rtc_thread.is_alive():
                logger.warning("RTC thread did not join within %.1fs", _RTC_JOIN_TIMEOUT_S)
            else:
                logger.info("RTC inference thread stopped")
            self._rtc_thread = None

    def pause(self) -> None:
        """暂停 RTC 后台线程。"""
        logger.info("Pausing RTC inference thread")
        self._policy_active.clear()

    def resume(self) -> None:
        """恢复 RTC 后台线程。"""
        logger.info("Resuming RTC inference thread")
        self._policy_active.set()

    def reset(self) -> None:
        """重置策略、处理器和动作队列。

        在 RTC 线程暂停或运行时调用都是安全的。此外还会丢弃最近一次发布的
        观测——基于陈旧观测计算出的动作块会让机器人猛地移向旧位姿——
        并递增 reset 纪元（epoch），使正在处理中的动作块被丢弃，而不是
        被合并到已清空的队列中。
        """
        logger.info("Resetting RTC inference state (policy + processors + queue)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        with self._obs_lock:
            # 在同一个临界区中完成清空与递增，与 _rtc_loop 的"纪元检查 + 合并"
            # 相对应，从而保证 reset 不可能把 reset 之前的动作块泄漏到全新的
            # 队列中。两侧的加锁顺序均为 _obs_lock -> queue.lock。
            if self._action_queue is not None:
                self._action_queue.clear()
            self._obs_holder["obs"] = None
            self._reset_epoch += 1
        # 队列已为空，因此待处理的任务变更不会与任何陈旧内容混合。
        self._discard_task_change()

    # ------------------------------------------------------------------
    # 动作生产（从主线程调用）
    # ------------------------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """从 RTC 队列中弹出下一个动作（忽略 ``obs_frame``）。"""
        if self._action_queue is None:
            return None
        queued = self._action_queue.get_with_task()
        if queued is None:
            return None
        # 队列在队列锁的保护下将每个动作与其所属动作块的任务配对，因此并发的
        # 合并不会在不同动作块之间错配标签。
        action, task = queued
        if task is None:
            # 这里的每次合并都会为其动作块打上标签，因此缺失标签意味着存在外来
            # 写入方：直接显式报错，而不是破坏 dispatched_task 和帧标签。
            raise RuntimeError("RTC action queue returned an action without task provenance")
        self._set_dispatched_task(task)
        return action

    def notify_observation(self, obs: dict) -> None:
        """发布最新观测，供 RTC 线程消费。"""
        with self._obs_lock:
            self._obs_holder["obs"] = obs

    # ------------------------------------------------------------------
    # 文本查询
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """当策略具有文本头（text head）时为 True。"""
        return self._policy.supports_text_generation()

    @property
    def control_thread_owns_policy(self) -> bool:
        """策略由 RTC 后台线程拥有；查询在 ``_rtc_loop`` 中被处理。"""
        return False

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """运行策略的文本头。在 RTC 线程上调用（参见 ``_rtc_loop``）。"""
        obs_batch = build_dataset_frame(self._hw_features, obs_processed, prefix="observation")
        # 当前任务，读取时不消费"任务已变更"这一边沿信号：它属于动作块路径。
        task = self.task
        obs_batch = prepare_observation_for_inference(
            obs_batch, torch.device(self._device), task, self._robot.robot_type
        )
        obs_batch = self._mark_query(obs_batch, query)
        preprocessed = self._preprocessor(obs_batch)
        with torch.inference_mode():
            # 不做 str() 强制转换：_service_query 会校验返回值。
            return self._policy.generate_text(preprocessed)

    # ------------------------------------------------------------------
    # RTC：后台推理线程
    # ------------------------------------------------------------------

    def _rtc_loop(self) -> None:
        """通过 RTC 生成动作块的后台线程。"""
        try:
            latency_tracker = LatencyTracker()
            time_per_chunk = 1.0 / self._fps
            policy_device = torch.device(self._device)

            warmup_required = max(1, self._compile_warmup_inferences) if self._use_torch_compile else 0
            inference_count = 0
            consecutive_errors = 0
            consecutive_discards = 0

            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                queue = self._action_queue
                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                    epoch_before = self._reset_epoch
                if queue is None or obs is None:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                # 在此处理排队中的文本查询——本线程才是拥有策略的线程。特意放在
                # 队列补充分支之前：否则在队列已满时发出的查询将不得不等待队列
                # 被排空。next-subtask 的回答通过 ``set_task`` 应用，因此下面的
                # 动作块路径会使用它，而该路径会基于自己的观测重新运行预处理器，
                # 于是有状态的步骤（相对动作锚定）不会残留本次查询的状态。
                if self._service_query(obs):
                    # 生成过程耗时数秒，因此上面的快照已经陈旧：重新读取观测以及
                    # 纪元，使下面的丢弃保护也能覆盖查询期间发生的 reset。
                    with self._obs_lock:
                        obs = self._obs_holder.get("obs")
                        epoch_before = self._reset_epoch
                    if obs is None:  # 查询中途发生的 reset 丢弃了观测
                        continue

                if queue.qsize() <= self._rtc_queue_threshold:
                    try:
                        current_time = time.perf_counter()
                        idx_before = queue.get_action_index()
                        prev_actions = queue.get_left_over()
                        has_previous_actions = prev_actions is not None and prev_actions.numel() > 0

                        policy_config = getattr(self._policy, "config", None)
                        training_max_delay = int(getattr(policy_config, "rtc_training_max_delay", 0))
                        latency = latency_tracker.max()
                        delay = _estimate_rtc_delay(
                            latency=latency,
                            time_per_step=time_per_chunk,
                            mode=self._rtc_config.mode,
                            training_max_delay=training_max_delay,
                            has_previous_actions=has_previous_actions,
                        )
                        if self._rtc_config.mode == "trained" and delay > 0:
                            delay = _clamp_trained_rtc_delay(
                                conditioned_delay=delay,
                                available_steps=0 if prev_actions is None else prev_actions.shape[0],
                                training_max_delay=training_max_delay,
                            )

                        task, task_changed = self._take_task()
                        if task_changed:
                            # 故意不清空队列：丢弃已排队的动作会让机器人在整整一个
                            # 推理延迟内收不到指令。启用 RTC 混合时，该动作块会基于
                            # 上一个动作块的剩余前缀进行合并，因此切换会在一次推理
                            # 之内生效；关闭混合时则先排空队列。
                            logger.info("Task changed to '%s' — applied from the next merged chunk", task)

                        obs_batch = build_dataset_frame(self._hw_features, obs, prefix="observation")
                        obs_batch = prepare_observation_for_inference(
                            obs_batch, policy_device, task, self._robot.robot_type
                        )
                        obs_batch["task"] = [task]

                        preprocessed = self._preprocessor(obs_batch)

                        if prev_actions is not None and self._relative_step is not None:
                            # 基于缓存的原始状态重新定基（rebase），使剩余的尾部保持在
                            # 训练时的坐标系中。
                            raw_state = self._relative_step.get_cached_state()
                            if raw_state is not None:
                                prev_abs = queue.get_processed_left_over()
                                if prev_abs is not None and prev_abs.numel() > 0:
                                    prev_actions = reanchor_relative_rtc_prefix(
                                        prev_actions_absolute=prev_abs,
                                        current_state=raw_state,
                                        relative_step=self._relative_step,
                                        normalizer_step=self._normalizer_step,
                                        policy_device=policy_device,
                                    )

                        if prev_actions is not None:
                            prev_actions = _normalize_prev_actions_length(
                                prev_actions, target_steps=self._rtc_config.execution_horizon
                            )

                        actions = self._policy.predict_action_chunk(
                            preprocessed, inference_delay=delay, prev_chunk_left_over=prev_actions
                        )

                        original = actions.squeeze(0).clone()
                        processed = self._postprocessor(actions).squeeze(0)
                        new_latency = time.perf_counter() - current_time
                        new_delay = math.ceil(new_latency / time_per_chunk)

                        inference_count += 1
                        consecutive_errors = 0
                        is_warmup = self._use_torch_compile and inference_count <= warmup_required
                        is_initial_trained_chunk = (
                            self._rtc_config.mode == "trained" and not has_previous_actions
                        )
                        if is_warmup or is_initial_trained_chunk:
                            latency_tracker.reset()
                        else:
                            latency_tracker.add(new_latency)

                        if (
                            not is_warmup
                            and self._rtc_config.mode == "trained"
                            and not _trained_rtc_chunk_can_merge(
                                conditioned_delay=delay,
                                measured_delay=new_delay,
                                training_max_delay=training_max_delay,
                                has_previous_actions=has_previous_actions,
                            )
                        ):
                            consecutive_discards += 1
                            logger.warning(
                                "Discarding trained RTC chunk (%d/%d): measured delay %d exceeded "
                                "conditioned delay %d (checkpoint supports %d); retrying with "
                                "updated latency",
                                consecutive_discards,
                                _RTC_MAX_CONSECUTIVE_DISCARDS,
                                new_delay,
                                delay,
                                training_max_delay,
                            )
                            if consecutive_discards >= _RTC_MAX_CONSECUTIVE_DISCARDS:
                                raise _TrainedRTCDelayExceededError(
                                    f"Measured RTC inference delay ({new_delay}) stayed above the "
                                    f"usable overlap for {consecutive_discards} consecutive chunks; "
                                    f"the checkpoint supports rtc_training_max_delay="
                                    f"{training_max_delay}. Retrain with a larger delay, lower "
                                    "--fps, or switch to --inference.rtc.mode=guided."
                                )
                            continue

                        consecutive_discards = 0
                        with self._obs_lock:
                            # 在同一个临界区中完成检查与合并，与 reset() 的"清空 + 递增"
                            # 相对应，从而保证 reset 不可能落在两者之间并泄漏 reset
                            # 之前的动作块。加锁顺序：_obs_lock -> queue.lock。
                            epoch_unchanged = epoch_before == self._reset_epoch
                            if epoch_unchanged:
                                queue.merge(original, processed, new_delay, idx_before, task=task)
                        if not epoch_unchanged:
                            logger.info("Discarding action chunk computed before an engine reset")

                        if (
                            is_warmup
                            and inference_count >= warmup_required
                            and not self._compile_warmup_done.is_set()
                        ):
                            self._compile_warmup_done.set()
                            logger.info("Compile warmup complete (%d inferences)", inference_count)

                        logger.debug("RTC inference latency=%.2fs, queue=%d", new_latency, queue.qsize())

                    except _FatalRTCInferenceError:
                        raise
                    except Exception as e:
                        consecutive_errors += 1
                        logger.error(
                            "RTC inference error (%d/%d): %s",
                            consecutive_errors,
                            _RTC_MAX_CONSECUTIVE_ERRORS,
                            e,
                        )
                        logger.debug(traceback.format_exc())
                        if consecutive_errors >= _RTC_MAX_CONSECUTIVE_ERRORS:
                            # 持续性故障：停止重试并传播关闭信号。
                            raise
                        time.sleep(_RTC_ERROR_RETRY_DELAY_S)
                else:
                    time.sleep(_RTC_IDLE_SLEEP_S)

        except Exception as e:
            self._failure_traceback = traceback.format_exc()
            logger.error("Fatal error in RTC thread: %s", e)
            logger.error(self._failure_traceback)
            self._rtc_error.set()
            # 解除所有预热等待者的阻塞，以免主循环无限空转
            self._compile_warmup_done.set()
            # 通知顶层关闭，使各个策略（strategies）退出其控制循环
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
