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

"""DAgger rollout 策略：人在回路（Human-in-the-Loop）数据采集。

为交互式模仿学习实现 RaC 范式（Recovery and Correction，恢复与纠正）。
在自主策略执行与通过遥操作器进行的人工干预之间交替。

输入通过键盘或脚踏板控制，由 ``input_device`` 配置字段选择。
每种设备提供三个动作：

    1. **pause_resume** — 切换策略执行（AUTONOMOUS <-> PAUSED）。
    2. **correction**   — 切换纠正录制（PAUSED <-> CORRECTING）。
    3. **upload**        — 按需将数据集推送到 hub（仅纠正模式）。
    ESC（仅键盘） — 停止会话。

录制模式：
    ``record_autonomous=True``：  类似哨兵的连续录制，按时间轮转 episode。
        自主帧和纠正帧都会被录制；纠正帧标记 ``intervention=True``。
    ``record_autonomous=False``： 只录制纠正窗口。
        每次纠正（从开始到停止）成为一个 episode。

遥操作器交接：
    在 AUTONOMOUS → PAUSED 时，对于可驱动的遥操作器（那些 ``feedback_features``
    非空的，例如 SO-101、OpenArmMini），会通过 ``send_feedback`` 平滑地驱动到
    从端的最后一个位置，使操作者接管时不会有突兀的抖动。不可驱动的遥操作器
    无法被驱动，因此在 PAUSED → CORRECTING 时，改为在纠正开始前将从端滑动到
    遥操作器当前的姿态。
"""

from __future__ import annotations

import contextlib
import enum
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock
from typing import Any

import numpy as np

from lerobot.common.control_utils import (
    follower_smooth_move_to,
    teleop_smooth_move_to,
    teleop_supports_feedback,
)
from lerobot.datasets import VideoEncodingManager
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.pedal import start_pedal_listener
from lerobot.utils.utils import log_say

from ..configs import DAggerKeyboardConfig, DAggerPedalConfig, DAggerStrategyConfig
from ..context import RolloutContext
from .core import (
    RolloutStrategy,
    estimate_max_episode_seconds,
    safe_push_to_hub,
    send_next_action,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DAgger 状态机
# ---------------------------------------------------------------------------


class DAggerPhase(enum.Enum):
    """DAgger episode 的可观测阶段。"""

    AUTONOMOUS = "autonomous"  # 策略驱动
    PAUSED = "paused"  # 引擎已暂停，遥操作已对齐，等待输入
    CORRECTING = "correcting"  # 人工通过遥操作驱动，记录干预


# 合法的 (current_phase, event) -> next_phase
_DAGGER_TRANSITIONS: dict[tuple[DAggerPhase, str], DAggerPhase] = {
    (DAggerPhase.AUTONOMOUS, "pause_resume"): DAggerPhase.PAUSED,
    (DAggerPhase.PAUSED, "pause_resume"): DAggerPhase.AUTONOMOUS,
    (DAggerPhase.PAUSED, "correction"): DAggerPhase.CORRECTING,
    (DAggerPhase.CORRECTING, "correction"): DAggerPhase.PAUSED,
}


class DAggerEvents:
    """DAgger 输入设备事件的线程安全容器。

    键盘/脚踏板线程写入转换请求；主循环消费它们。
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._phase = DAggerPhase.AUTONOMOUS
        self._pending_transition: str | None = None

        # 会话级标志
        self.stop_recording = Event()
        self.upload_requested = Event()

    #  -- 线程安全阶段访问 ------------------------------------------
    @property
    def phase(self) -> DAggerPhase:
        """DAgger 状态机的当前阶段。"""
        with self._lock:
            return self._phase

    @phase.setter
    def phase(self, value: DAggerPhase) -> None:
        with self._lock:
            self._phase = value

    def request_transition(self, event: str) -> None:
        """请求一次阶段转换（从键盘/脚踏板线程调用）。

        仅当该请求对应从当前阶段出发的一个合法转换时才入队，
        从而防止不可能的状态变更。
        """
        with self._lock:
            if (self._phase, event) in _DAGGER_TRANSITIONS:
                self._pending_transition = event

    def consume_transition(self) -> tuple[DAggerPhase, DAggerPhase] | None:
        """消费一个待处理的转换（从主循环调用）。"""
        with self._lock:
            if self._pending_transition is None:
                return None
            key = (self._phase, self._pending_transition)
            self._pending_transition = None
            new_phase = _DAGGER_TRANSITIONS.get(key)
            if new_phase is None:
                return None
            old_phase = self._phase
            self._phase = new_phase
            return old_phase, new_phase

    def reset(self) -> None:
        """为新的会话重置所有瞬态状态。"""
        with self._lock:
            self._phase = DAggerPhase.AUTONOMOUS
            self._pending_transition = None
        self.upload_requested.clear()


# ---------------------------------------------------------------------------
# 输入设备处理器
# ---------------------------------------------------------------------------


def _init_dagger_keyboard(events: DAggerEvents, cfg: DAggerKeyboardConfig):
    """为 DAgger 的 3 个控制初始化一个键盘监听器。

    后端选择（在 X11 / 受信 macOS / Windows 上用 pynput，在 Wayland /
    无头 TTY 上用终端读取器）委托给 :func:`create_key_listener`。返回该
    监听器（暴露 ``stop()``），当没有可用的键盘后端时返回 ``None``。
    """
    # 将配置中的按键名映射到 DAgger 事件名。
    key_to_event = {
        cfg.pause_resume: "pause_resume",
        cfg.correction: "correction",
    }

    def dispatch(name: str) -> None:
        """将解析出的按键名应用到 DAgger 事件上。"""
        if name == "esc":
            logger.info("Stop recording...")
            events.stop_recording.set()
            return
        if name in key_to_event:
            events.request_transition(key_to_event[name])
        if name == cfg.upload:
            events.upload_requested.set()

    return create_key_listener(
        dispatch,
        controls_help=(
            f"pause_resume='{cfg.pause_resume}', correction='{cfg.correction}', "
            f"upload='{cfg.upload}', ESC=stop"
        ),
    )


def _init_dagger_pedal(events: DAggerEvents, cfg: DAggerPedalConfig):
    """使用 DAgger 的 3 踏板控制初始化脚踏板监听器。

    返回踏板监听线程（若 evdev 不可用则返回 ``None``）。
    """
    code_to_event = {
        cfg.pause_resume: "pause_resume",
        cfg.correction: "correction",
    }

    def on_press(code: str) -> None:
        if code in code_to_event:
            events.request_transition(code_to_event[code])
        if code == cfg.upload:
            events.upload_requested.set()

    logger.info("Initializing DAgger foot pedal listener (device=%s)", cfg.device_path)
    return start_pedal_listener(on_press, device_path=cfg.device_path)


# ---------------------------------------------------------------------------
# DAgger 策略
# ---------------------------------------------------------------------------


class DAggerStrategy(RolloutStrategy):
    """带有干预标记的人在回路（Human-in-the-Loop）数据采集。

    状态机::

        AUTONOMOUS --(key1)--> PAUSED --(key2)--> CORRECTING --(key2)--> PAUSED
                               --(key1)--> AUTONOMOUS

    录制模式：
        ``record_autonomous=True``：类似哨兵的连续录制，按时间轮转 episode。
            干预帧标记为 True。
        ``record_autonomous=False``：只录制纠正窗口。
            每次纠正 = 一个 episode。通过 key3 按需上传。
    """

    config: DAggerStrategyConfig

    def __init__(self, config: DAggerStrategyConfig):
        super().__init__(config)
        self._listener = None
        self._pedal_thread = None
        self._events = DAggerEvents()
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._needs_push = Event()
        self._episode_lock = Lock()

    def setup(self, ctx: RolloutContext) -> None:
        """初始化推理引擎和输入设备监听器。"""
        self._init_engine(ctx)
        dataset_cfg = ctx.runtime.cfg.dataset  # 永不为 None：dataset_mode="required"
        if self.config.num_episodes is None:
            self.config.num_episodes = dataset_cfg.num_episodes
            logger.info(
                "DAgger num_episodes not set — using --dataset.num_episodes=%d", self.config.num_episodes
            )
        if not self.config.record_autonomous and not dataset_cfg.streaming_encoding:
            logger.info(
                "Streaming encoding is disabled for DAgger corrections-only mode. "
                "Consider enabling it for faster episode saving: "
                "--dataset.streaming_encoding=true --dataset.encoder_threads=2"
            )
        self._push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dagger-push")
        target_mb = self.config.target_video_file_size_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB
        self._episode_duration_s = estimate_max_episode_seconds(
            ctx.data.dataset_features, ctx.runtime.cfg.fps, target_size_mb=target_mb
        )

        if self.config.input_device == "keyboard":
            self._listener = _init_dagger_keyboard(self._events, self.config.keyboard)
        else:
            self._pedal_thread = _init_dagger_pedal(self._events, self.config.pedal)

        record_mode = "all frames (sentry-like)" if self.config.record_autonomous else "corrections only"
        logger.info(
            "DAgger strategy ready (input=%s, episodes=%d, record=%s, episode_duration=%.0fs)",
            self.config.input_device,
            self.config.num_episodes,
            record_mode,
            self._episode_duration_s,
        )

    def run(self, ctx: RolloutContext) -> None:
        """运行带人在回路干预的 DAgger episode。"""
        if self.config.record_autonomous:
            self._run_continuous(ctx)
        else:
            self._run_corrections_only(ctx)

    def teardown(self, ctx: RolloutContext) -> None:
        """停止监听器、最终化数据集并断开硬件。"""
        play_sounds = ctx.runtime.cfg.play_sounds
        logger.info("Stopping DAgger recording")
        log_say("Stopping DAgger recording", play_sounds)

        if self._listener is not None:
            logger.info("Stopping keyboard listener")
            self._listener.stop()

        # 干净地刷完任何已排队/正在运行的推送
        if self._push_executor is not None:
            logger.info("Shutting down push executor (waiting for pending pushes)...")
            self._push_executor.shutdown(wait=True)
            self._push_executor = None

        if ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()
            if self._needs_push.is_set() and ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.push_to_hub:
                logger.info("Pushing final dataset to hub...")
                if safe_push_to_hub(
                    ctx.data.dataset,
                    tags=ctx.runtime.cfg.dataset.tags,
                    private=ctx.runtime.cfg.dataset.private,
                ):
                    logger.info("Dataset uploaded to hub")
                    log_say("Dataset uploaded to hub", play_sounds)

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("DAgger strategy teardown complete")

    # ------------------------------------------------------------------
    # 连续录制模式（record_autonomous=True）
    # ------------------------------------------------------------------

    def _run_continuous(self, ctx: RolloutContext) -> None:
        """带有干预标记的类哨兵连续录制。

        每 ``episode_time_s`` 秒自动轮转一次 episode，并每
        ``upload_every_n_episodes`` 个 episode 在后台上传一次。
        自主帧和纠正帧都会被录制；纠正帧标记为 ``intervention=True``。
        """
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        timer = CycleTimer(cfg.fps, interpolator.multiplier)
        correction_stride = interpolator.multiplier
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        play_sounds = cfg.play_sounds

        engine.reset()
        interpolator.reset()
        events.reset()
        engine.resume()

        last_action: dict[str, Any] | None = None
        correction_tick = 0
        start_time = time.perf_counter()
        episode_start = time.perf_counter()
        episodes_since_push = 0
        episode_duration_s = self._episode_duration_s
        logger.info("DAgger continuous recording started (episode_duration=%.0fs)", episode_duration_s)

        with VideoEncodingManager(dataset):
            try:
                while not events.stop_recording.is_set() and not ctx.runtime.shutdown_event.is_set():
                    timer.tick(new_cycle=interpolator.needs_new_action())

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    # 处理转换
                    transition = events.consume_transition()
                    if transition is not None:
                        old_phase, new_phase = transition
                        self._apply_transition(
                            old_phase,
                            new_phase,
                            engine,
                            interpolator,
                            ctx,
                            last_action,
                            timer,
                        )
                        if new_phase == DAggerPhase.AUTONOMOUS:
                            last_action = None
                        elif new_phase == DAggerPhase.CORRECTING:
                            # 纠正带有自己的录制阶段：每次干预都以一个
                            # 被录制的帧开始，然后每第 ``multiplier`` 个 tick
                            # 录制一次。自主帧则改由插值器门控，因此
                            # 两种节奏永不共享一个其奇偶性可能被对方改变的计数器。
                            correction_tick = 0

                    phase = events.phase
                    with timer.section("observe"):
                        obs = robot.get_observation()

                    # --- CORRECTING：人工遥操作控制 ---
                    # TODO(Steven)：teleop 以与策略相同的 FPS 运行。为了
                    # 解耦两者，请按其原生速率采样 teleop，并
                    # 插值到控制循环的 tick 速率。
                    if phase == DAggerPhase.CORRECTING:
                        with timer.section("process_obs"):
                            obs_processed = ctx.processors.robot_observation_processor(obs)
                        with timer.section("teleop"):
                            teleop_action = teleop.get_action()
                            processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                            robot_action_to_send = ctx.processors.robot_action_processor(
                                (processed_teleop, obs)
                            )
                        with timer.section("send"):
                            robot.send_action(robot_action_to_send)
                        last_action = robot_action_to_send
                        with timer.section("telemetry"):
                            self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)
                        if correction_tick % correction_stride == 0:
                            with timer.section("record"):
                                obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                                action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                                frame = {
                                    **obs_frame,
                                    **action_frame,
                                    "task": task_str,
                                    "intervention": np.array([True], dtype=bool),
                                }
                                dataset.add_frame(frame)
                        correction_tick += 1

                    # --- PAUSED：保持位置 ---
                    elif phase == DAggerPhase.PAUSED:
                        if last_action:
                            with timer.section("send"):
                                robot.send_action(last_action)

                    # --- AUTONOMOUS：策略控制 ---
                    else:
                        with timer.section("process_obs"):
                            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                        if self._handle_warmup(cfg.use_torch_compile, timer):
                            continue

                        action_dict = send_next_action(obs_processed, obs, ctx, interpolator, timer)
                        if action_dict is not None:
                            with timer.section("telemetry"):
                                self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                            last_action = ctx.processors.robot_action_processor((action_dict, obs))
                            if interpolator.emitted_policy_action:
                                with timer.section("record"):
                                    obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                                    action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                                    frame = {
                                        **obs_frame,
                                        **action_frame,
                                        "task": task_str,
                                        "intervention": np.array([False], dtype=bool),
                                    }
                                    dataset.add_frame(frame)

                    # episode 轮转由视频文件大小目标推导而来。
                    # 当纠正正在进行时会推迟保存，以便
                    # episode 边界落在一个干净的自主帧上。
                    elapsed = time.perf_counter() - episode_start
                    if elapsed >= episode_duration_s and phase != DAggerPhase.CORRECTING:
                        with self._episode_lock:
                            dataset.save_episode()
                        episodes_since_push += 1
                        self._needs_push.set()
                        logger.info(
                            "Episode saved (total: %d, elapsed: %.1fs)",
                            dataset.num_episodes,
                            elapsed,
                        )
                        log_say(f"Episode {dataset.num_episodes} saved", play_sounds)
                        # ``save_episode`` 在计时的循环体内阻塞：先报告
                        # 该 episode，然后丢弃那个不完整分组及其造成的
                        # 间隙，它们属于收尾而非节奏。
                        timer.log_episode_summary(f"episode {dataset.num_episodes}")
                        timer.restart()

                        if episodes_since_push >= self.config.upload_every_n_episodes:
                            self._background_push(dataset, cfg)
                            episodes_since_push = 0

                        episode_start = time.perf_counter()

                    timer.wait()

            finally:
                logger.info("DAgger continuous control loop ended — pausing engine")
                timer.log_run_summary()
                engine.pause()
                with contextlib.suppress(Exception):
                    with self._episode_lock:
                        dataset.save_episode()
                    self._needs_push.set()
                    logger.info("Final in-progress episode saved")

    # ------------------------------------------------------------------
    # 仅纠正模式（record_autonomous=False）
    # ------------------------------------------------------------------

    def _run_corrections_only(self, ctx: RolloutContext) -> None:
        """只录制人工纠正窗口。每次纠正 = 一个 episode。

        策略以自主方式运行而不录制。当用户暂停并开始纠正时，
        帧会以 ``intervention=True`` 被录制。停止纠正即保存该 episode。
        数据集可通过上传按键/踏板按需上传。
        """
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        timer = CycleTimer(cfg.fps, interpolator.multiplier)
        correction_stride = interpolator.multiplier
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        play_sounds = cfg.play_sounds

        engine.reset()
        interpolator.reset()
        events.reset()
        engine.resume()

        last_action: dict[str, Any] | None = None
        start_time = time.perf_counter()
        correction_tick = 0
        recorded = 0
        logger.info(
            "DAgger corrections-only recording started (target: %d episodes)", self.config.num_episodes
        )

        with VideoEncodingManager(dataset):
            try:
                while (
                    recorded < self.config.num_episodes
                    and not events.stop_recording.is_set()
                    and not ctx.runtime.shutdown_event.is_set()
                ):
                    timer.tick(new_cycle=interpolator.needs_new_action())

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    # 处理转换
                    transition = events.consume_transition()
                    if transition is not None:
                        old_phase, new_phase = transition
                        self._apply_transition(
                            old_phase,
                            new_phase,
                            engine,
                            interpolator,
                            ctx,
                            last_action,
                            timer,
                        )
                        if new_phase == DAggerPhase.AUTONOMOUS:
                            last_action = None
                        elif new_phase == DAggerPhase.CORRECTING:
                            # 每次干预都以一个被录制的帧开始，然后每第
                            # ``multiplier`` 个 tick 录制一次，因此无论自主运行
                            # 留下的是哪种阶段，每个纠正 episode 都保持
                            # 每秒 ``fps`` 帧。
                            correction_tick = 0

                        # 纠正结束 -> 保存 episode（若非流式则阻塞）
                        if old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
                            with self._episode_lock:
                                dataset.save_episode()
                            recorded += 1
                            self._needs_push.set()
                            logger.info(
                                "Correction %d/%d saved",
                                recorded,
                                self.config.num_episodes,
                            )
                            log_say(f"Correction {recorded} saved", play_sounds)
                            # ``save_episode`` 在计时的循环体内阻塞：先报告
                            # 该纠正，然后丢弃那个不完整分组及其造成的
                            # 间隙，它们属于收尾而非节奏。
                            timer.log_episode_summary(f"correction {recorded}")
                            timer.restart()

                    # 按需上传
                    if events.upload_requested.is_set():
                        events.upload_requested.clear()
                        logger.info("Upload requested by user")
                        self._background_push(dataset, cfg)

                    phase = events.phase
                    with timer.section("observe"):
                        obs = robot.get_observation()

                    # --- CORRECTING：人工遥操作控制 + 录制 ---
                    # TODO(Steven)：teleop 以与策略相同的 FPS 运行。为了
                    # 解耦两者，请按其原生速率采样 teleop，并
                    # 插值到控制循环的 tick 速率。
                    if phase == DAggerPhase.CORRECTING:
                        with timer.section("process_obs"):
                            obs_processed = ctx.processors.robot_observation_processor(obs)
                        with timer.section("teleop"):
                            teleop_action = teleop.get_action()
                            processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                            robot_action_to_send = ctx.processors.robot_action_processor(
                                (processed_teleop, obs)
                            )
                        with timer.section("send"):
                            robot.send_action(robot_action_to_send)
                        last_action = robot_action_to_send
                        with timer.section("telemetry"):
                            self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)

                        if correction_tick % correction_stride == 0:
                            with timer.section("record"):
                                obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                                action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                                dataset.add_frame(
                                    {
                                        **obs_frame,
                                        **action_frame,
                                        "task": task_str,
                                        "intervention": np.array([True], dtype=bool),
                                    }
                                )
                        correction_tick += 1

                    # --- PAUSED：保持位置 ---
                    elif phase == DAggerPhase.PAUSED:
                        if last_action:
                            with timer.section("send"):
                                robot.send_action(last_action)

                    # --- AUTONOMOUS：策略控制（不录制） ---
                    else:
                        with timer.section("process_obs"):
                            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                        if self._handle_warmup(cfg.use_torch_compile, timer):
                            continue

                        action_dict = send_next_action(obs_processed, obs, ctx, interpolator, timer)
                        if action_dict is not None:
                            with timer.section("telemetry"):
                                self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                            last_action = ctx.processors.robot_action_processor((action_dict, obs))

                    timer.wait()

            finally:
                logger.info("DAgger corrections-only loop ended — pausing engine")
                timer.log_run_summary()
                engine.pause()
                with contextlib.suppress(Exception):
                    with self._episode_lock:
                        dataset.save_episode()
                    self._needs_push.set()
                    logger.info("Final in-progress episode saved")

    # ------------------------------------------------------------------
    # 状态机转换的副作用
    # ------------------------------------------------------------------

    def _apply_transition(
        self,
        old_phase: DAggerPhase,
        new_phase: DAggerPhase,
        engine,
        interpolator,
        ctx: RolloutContext,
        prev_action: dict | None,
        timer: CycleTimer | None = None,
    ) -> None:
        """为一个已校验的阶段转换执行副作用，包括平滑交接。

        下面的平滑交接可以通过 ``--strategy.smooth_handover=false`` 禁用
        （适用于在接合时以机器人当前姿态重新对齐的离合器式遥操作器）。

        AUTONOMOUS -> PAUSED（可驱动的遥操作器）：
            暂停引擎，然后驱动主臂到达从端最后指令的位置，
            使操作者接管时不会有突兀的抖动。

        PAUSED -> CORRECTING（不可驱动的遥操作器）：
            将从端滑动到遥操作器当前的姿态，使机器人迎向操作者的手，
            而不是在第一个帧上跳向它。

        CORRECTING -> PAUSED（可驱动的遥操作器）：
            在纠正之后重新启用力矩以保持位置。
            如果取消纠正录制，这可能会很有用

        PAUSED -> AUTONOMOUS：
            重置并恢复推理引擎。
        """
        teleop = ctx.hardware.teleop
        robot = ctx.hardware.robot_wrapper

        logger.info("Phase transition: %s -> %s", old_phase.value, new_phase.value)
        if old_phase == DAggerPhase.AUTONOMOUS and new_phase == DAggerPhase.PAUSED:
            logger.info("Pausing engine - robot holds position")
            engine.pause()

            if self.config.smooth_handover and teleop_supports_feedback(teleop) and prev_action is not None:
                # TODO(Maxime)：prev_action 处于机器人动作键空间（robot_action_processor 的输出）。
                # send_feedback 期望遥操作反馈键空间。对于同构配置（例如 SO-101
                # 主端 + SO-101 从端），键是相同的，所以这样可行。如果处理器流水线
                # 进行了非平凡的键重命名（例如对动作键使用 rename_map），那么
                # teleop_smooth_move_to 中的插值会静默地空操作，手臂不会移动。
                logger.info("Smooth handover: moving leader arm to follower position")
                teleop_smooth_move_to(teleop, prev_action)

        elif old_phase == DAggerPhase.PAUSED and new_phase == DAggerPhase.CORRECTING:
            logger.info("Entering correction mode - human teleop control")
            if (
                self.config.smooth_handover
                and not teleop_supports_feedback(teleop)
                and prev_action is not None
            ):
                logger.info("Smooth handover: sliding follower to teleop position")
                obs = robot.get_observation()
                teleop_action = teleop.get_action()
                processed = ctx.processors.teleop_action_processor((teleop_action, obs))
                target = ctx.processors.robot_action_processor((processed, obs))
                follower_smooth_move_to(robot, prev_action, target)

            # 为人工控制解锁遥操作器
            if teleop_supports_feedback(teleop):
                teleop.disable_torque()

        elif old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
            if teleop_supports_feedback(teleop):
                teleop.enable_torque()

        elif new_phase == DAggerPhase.AUTONOMOUS:
            logger.info("Resuming autonomous mode - resetting engine and interpolator")
            interpolator.reset()
            engine.reset()
            engine.resume()

            # 在恢复策略之前释放遥操作器
            if teleop_supports_feedback(teleop):
                teleop.disable_torque()

        # 转换是一次性的操作者事件，运行在控制循环的计时体内，
        # 而上面的平滑交接斜坡会阻塞约零点几秒。若把它们留在
        # 计时器的累加器中，会超出该分组的预算并把一个健康的循环报告为缓慢，
        # 因此丢弃包含该转换的那个不完整分组。这也会重新布防
        # 启动豁免，而回到 AUTONOMOUS 本来就需要它：重置后的插值器
        # 会在两个 tick 上重新预热，正如循环启动那样，
        # 因此跨越这两个 tick 的分组合理地会有超时。
        if timer is not None:
            timer.restart()

    # ------------------------------------------------------------------
    # 后台推送（两种模式共用）
    # ------------------------------------------------------------------

    def _background_push(self, dataset, cfg) -> None:
        """在单工作线程的执行器上排入一次 Hub 推送。

        执行器的 max_workers=1 保证一次最多运行一个推送；
        提交的任务会被排队而非丢弃。当操作者正处于纠正过程中时
        推送会被阻塞，以避免上传一个只录制了一部分的 episode。
        """
        if self._push_executor is None:
            return

        if self._events.phase == DAggerPhase.CORRECTING:
            logger.info("Skipping push — correction in progress")
            return

        if self._pending_push is not None and not self._pending_push.done():
            logger.info("Previous push still in progress; queueing next")

        def _push():
            try:
                with self._episode_lock:
                    if safe_push_to_hub(
                        dataset,
                        tags=cfg.dataset.tags if cfg.dataset else None,
                        private=cfg.dataset.private if cfg.dataset else False,
                    ):
                        self._needs_push.clear()
                        logger.info("Background push to hub complete")
            except Exception as e:
                logger.error("Background push failed: %s", e)

        self._pending_push = self._push_executor.submit(_push)
        logger.info("Background push task submitted")
