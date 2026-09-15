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

"""精彩片段（Highlight Reel）策略：通过环形缓冲区按需录制。"""

from __future__ import annotations

import contextlib
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event as ThreadingEvent, Lock

from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.utils import log_say

from ..configs import HighlightStrategyConfig
from ..context import RolloutContext
from ..ring_buffer import RolloutRingBuffer
from .core import RolloutStrategy, safe_push_to_hub, send_next_action

logger = logging.getLogger(__name__)


class HighlightStrategy(RolloutStrategy):
    """自主 rollout，并通过环形缓冲区按需录制。

    机器人自主运行，同时一个有内存上限的环形缓冲区持续捕获遥测
    数据。当用户按下保存键时：

    1. 环形缓冲区的内容被刷新到数据集（最近 *Z* 秒）。
    2. 继续实时录制，直到再次按下保存键。
    3. 保存该回合，环形缓冲区恢复捕获。

    要求 ``streaming_encoding=True``（在配置校验中强制执行），这样
    ``dataset.add_frame`` 就是一次非阻塞的队列放入操作——在一个节拍内
    刷新整个环形缓冲区绝不能拖慢控制循环。
    """

    config: HighlightStrategyConfig

    def __init__(self, config: HighlightStrategyConfig):
        super().__init__(config)
        self._ring: RolloutRingBuffer | None = None
        self._listener = None
        self._save_requested = ThreadingEvent()
        self._recording_live = ThreadingEvent()
        self._push_requested = ThreadingEvent()
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._episode_lock = Lock()

    def setup(self, ctx: RolloutContext) -> None:
        """初始化推理引擎、环形缓冲区和键盘监听器。"""
        self._init_engine(ctx)

        self._ring = RolloutRingBuffer(
            max_seconds=self.config.ring_buffer_seconds,
            max_memory_mb=self.config.ring_buffer_max_memory_mb,
            fps=ctx.runtime.cfg.fps,
        )

        self._push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="highlight-push")
        logger.info(
            "Ring buffer initialized (max_seconds=%.0f, max_memory=%.0fMB)",
            self.config.ring_buffer_seconds,
            self.config.ring_buffer_max_memory_mb,
        )
        self._setup_keyboard(ctx.runtime.shutdown_event)
        logger.info(
            "Highlight strategy ready (buffer=%.0fs, save='%s', push='%s')",
            self.config.ring_buffer_seconds,
            self.config.save_key,
            self.config.push_key,
        )

    def run(self, ctx: RolloutContext) -> None:
        """运行自主循环，缓冲帧并按需录制。"""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        ring = self._ring
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        timer = CycleTimer(cfg.fps, interpolator.multiplier)

        engine.resume()
        play_sounds = cfg.play_sounds

        start_time = time.perf_counter()
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        logger.info("Highlight strategy recording started (press '%s' to save)", self.config.save_key)

        with VideoEncodingManager(dataset):
            try:
                while not ctx.runtime.shutdown_event.is_set():
                    timer.tick(new_cycle=interpolator.needs_new_action())

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    with timer.section("observe"):
                        obs = robot.get_observation()
                    with timer.section("process_obs"):
                        obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                    if self._handle_warmup(cfg.use_torch_compile, timer):
                        continue

                    action_dict = send_next_action(obs_processed, obs, ctx, interpolator, timer)

                    if action_dict is not None:
                        with timer.section("telemetry"):
                            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

                        if self._push_requested.is_set():
                            self._push_requested.clear()
                            logger.info("Push requested by user")
                            self._background_push(dataset, cfg)

                        # 每个插值周期录制一次（缓冲或实时），使帧节拍与数据集
                        # 声明的 fps 以及环形缓冲区基于 fps 的容量相匹配；被
                        # 插值的节拍只向机器人发送指令。保存开关的切换也在
                        # 这里处理，从而保证回合边界总是落在一帧已录制的数据上。
                        if interpolator.emitted_policy_action:
                            with timer.section("record"):
                                obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                                action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                                frame = {**obs_frame, **action_frame, "task": task_str}

                                toggled = False
                                frame_consumed = False
                                # 注意：先 ``is_set()`` 再 ``clear()`` 对于键盘线程
                                # 在两者之间再次设置标志的情况并不是原子的——但这
                                # 并无大碍：我们至多丢失一次切换，它会在下一次迭代
                                # 中被处理。
                                if self._save_requested.is_set():
                                    self._save_requested.clear()
                                    toggled = True
                                    if not self._recording_live.is_set():
                                        logger.info(
                                            "Flushing ring buffer (%d frames) + starting live recording",
                                            len(ring),
                                        )
                                        for buffered_frame in ring.drain():
                                            dataset.add_frame(buffered_frame)
                                        self._recording_live.set()
                                    else:
                                        dataset.add_frame(frame)
                                        with self._episode_lock:
                                            dataset.save_episode()
                                        logger.info("Episode saved (total: %d)", dataset.num_episodes)
                                        log_say(
                                            f"Episode {dataset.num_episodes} saved",
                                            play_sounds,
                                        )
                                        self._recording_live.clear()
                                        frame_consumed = True

                                if not frame_consumed:
                                    if self._recording_live.is_set():
                                        dataset.add_frame(frame)
                                    else:
                                        ring.append(frame)

                            # 排空环形缓冲区和终结回合都会在被计时的循环体中
                            # 阻塞相当一部分秒。它们属于操作员触发的事件，而不是
                            # 稳态开销，因此丢弃不完整的分组以及它们造成的空隙。
                            if toggled:
                                if frame_consumed:
                                    timer.log_episode_summary(f"episode {dataset.num_episodes}")
                                timer.restart()

                    timer.wait()

            finally:
                logger.info("Highlight control loop ended")
                timer.log_run_summary()
                if self._recording_live.is_set():
                    logger.info("Saving in-progress live episode")
                    with contextlib.suppress(Exception), self._episode_lock:
                        dataset.save_episode()

    def teardown(self, ctx: RolloutContext) -> None:
        """停止监听器、终结数据集，并断开硬件连接。"""
        play_sounds = ctx.runtime.cfg.play_sounds
        logger.info("Stopping highlight recording")
        log_say("Stopping highlight recording", play_sounds)

        if self._listener is not None:
            logger.info("Stopping keyboard listener")
            self._listener.stop()

        if self._push_executor is not None:
            logger.info("Shutting down push executor (waiting for pending pushes)...")
            self._push_executor.shutdown(wait=True)
            self._push_executor = None

        if ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()
            if ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.push_to_hub:
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
        logger.info("Highlight strategy teardown complete")

    def _setup_keyboard(self, shutdown_event: ThreadingEvent) -> None:
        """为保存键和推送键设置键盘监听器。

        后端的选择（在 X11 / 受信任的 macOS / Windows 上使用 pynput，在
        Wayland / 无头 TTY 上使用终端读取器）委托给
        :func:`create_key_listener`。
        """
        save_key = self.config.save_key
        push_key = self.config.push_key

        def dispatch(name: str) -> None:
            """将解析出的按键名应用到 highlight 的各个事件上。"""
            if name == save_key:
                self._save_requested.set()
            elif name == push_key:
                self._push_requested.set()
            elif name == "esc":
                self._save_requested.clear()
                shutdown_event.set()

        self._listener = create_key_listener(
            dispatch, controls_help=f"save='{save_key}', push='{push_key}', ESC=stop"
        )

    def _background_push(self, dataset, cfg) -> None:
        """在单工作线程执行器上排队一次 Hub 推送。"""
        if self._push_executor is None:
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
                        logger.info("Background push to hub complete")
            except Exception as e:
                logger.error("Background push failed: %s", e)

        self._pending_push = self._push_executor.submit(_push)
        logger.info("Background push task submitted")
