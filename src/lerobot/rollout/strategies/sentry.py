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

"""Sentry rollout 策略：持续自主录制并自动上传。"""

from __future__ import annotations

import contextlib
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock

from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.utils import log_say

from ..configs import SentryStrategyConfig
from ..context import RolloutContext
from .core import (
    RolloutStrategy,
    estimate_max_episode_seconds,
    safe_push_to_hub,
    send_next_action,
)

logger = logging.getLogger(__name__)


class SentryStrategy(RolloutStrategy):
    """持续自主 rollout，录制始终开启。

    回合时长由相机分辨率、FPS 和
    ``DEFAULT_VIDEO_FILE_SIZE_IN_MB`` 推导得出，使每个保存的回合所
    产生的视频文件都已越过块大小边界。这保证了 ``push_to_hub`` 的
    高效——它上传的是完整的视频文件，而不是重复上传一个仍在增长
    的文件。

    数据集通过一个有界的单工作线程执行器推送到 Hub，因此任何推送都
    不会被静默丢弃，且同一时刻恰好只有一个推送在运行。

    策略状态（隐藏状态、RTC 队列）有意跨回合边界保留——Sentry 是对
    一段连续的 rollout 进行切片，机器人在各切片之间不会复位。

    要求 ``streaming_encoding=True``（在配置校验中强制执行），
    以防止磁盘 I/O 阻塞控制循环。

    ``run()`` 可重复启动，这正是 ``--interactive=true`` 所要求的：
    每次调用会录制若干完整回合外加最后一个不完整的回合，并且只有
    ``teardown()`` 才会终结数据集。
    """

    config: SentryStrategyConfig

    def __init__(self, config: SentryStrategyConfig):
        super().__init__(config)
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._needs_push = Event()
        self._episode_lock = Lock()
        # 实例状态，而不是 run() 的局部变量，从而使上传节奏能够跨片段保留。
        self._episodes_since_push = 0
        # 当 save_episode 在写入中途失败时锁存：此时磁盘上的数据集可能含有
        # 已提交但无法从元数据访问的行，因此继续往里录制或推送都会扩大/
        # 上传损坏的数据。
        self._dataset_poisoned = False

    def setup(self, ctx: RolloutContext) -> None:
        """初始化推理引擎和后台推送执行器。"""
        self._init_engine(ctx)
        self._push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentry-push")
        target_mb = self.config.target_video_file_size_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB
        self._episode_duration_s = estimate_max_episode_seconds(
            ctx.data.dataset_features, ctx.runtime.cfg.fps, target_size_mb=target_mb
        )
        logger.info(
            "Sentry strategy ready (episode_duration=%.0fs, upload_every=%d eps)",
            self._episode_duration_s,
            self.config.upload_every_n_episodes,
        )

    def run(self, ctx: RolloutContext) -> None:
        """运行持续录制循环，并自动轮转回合。"""
        if self._dataset_poisoned:
            raise RuntimeError(
                "Refusing to start a new segment: a previous save_episode failed mid-write, so "
                "the dataset on disk may be partially committed. Inspect it before recording more."
            )
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        # 每个片段独立的计时器，绝不提升到实例上（见 ``RolloutStrategy.run``）。
        timer = CycleTimer(cfg.fps, interpolator.multiplier, report=ctx.runtime.cadence_report)

        engine.resume()
        episode_duration_s = self._episode_duration_s

        start_time = time.perf_counter()
        episode_start = time.perf_counter()
        logger.info("Sentry recording started (episode_duration=%.0fs)", episode_duration_s)

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
                    # 每个插值周期只录制一次，使数据集的节拍与其声明的 fps
                    # 一致；被插值的节拍只向机器人发送指令。
                    if interpolator.emitted_policy_action:
                        with timer.section("record"):
                            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                            action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                            # 使用 ``dispatched_task`` 作为标签，即生成刚刚发送的那个动作
                            # 所用的指令（实时的 ``engine.task`` 会错误标记仍在队列中、
                            # 由上一条指令生成的动作）；在任意倍乘系数下都可靠，因为
                            # 从产生此动作的补数据节拍以来还没有任何 ``get_action`` 执行过。
                            frame = {**obs_frame, **action_frame, "task": engine.dispatched_task}
                            # ``add_frame`` 写入的是进行中回合的缓冲区；后台推送方
                            # 只会接触磁盘上*已终结*的回合产物。两者操作的状态互不
                            # 相交，因此 ``add_frame`` 不需要 ``_episode_lock``。
                            dataset.add_frame(frame)

                # 由视频文件大小目标推导出的回合轮转。
                # 此时长是保守估计，因此实际视频到目前为止必已超过
                # DEFAULT_VIDEO_FILE_SIZE_IN_MB，从而保持 push_to_hub 的
                # 高效（上传完整文件）。
                elapsed = time.perf_counter() - episode_start
                if elapsed >= episode_duration_s:
                    self._checked_save_episode(dataset)
                    logger.info(
                        "Episode saved (total: %d, elapsed: %.1fs)",
                        dataset.num_episodes,
                        elapsed,
                    )
                    # ``save_episode`` 会在被计时的循环体中阻塞相当一部分秒。
                    # 这属于回合终结，而不是稳态节拍，因此先上报该回合，然后
                    # 丢弃不完整的分组以及保存操作造成的空隙。
                    timer.log_episode_summary(f"episode {dataset.num_episodes}")
                    timer.restart()

                    self._register_saved_episode(dataset, cfg)

                    episode_start = time.perf_counter()

                # 在帧录制完成后再处理文本查询通道，这样耗时数秒的生成就不会
                # 落在本拍的观测与其 ``add_frame`` 之间；放在上面的守卫之外，
                # 以便饥饿节拍也能执行 pump。
                with timer.section("query"):
                    engine.pump_query(obs_processed)

                timer.wait()

        finally:
            logger.info("Sentry control loop ended")
            # 先上报，再进行尾部保存，因为后者在保存失败时会重新抛出异常。
            timer.log_run_summary()
            self._save_tail_episode(dataset, cfg)

    def _checked_save_episode(self, dataset) -> None:
        """在推送锁的保护下执行 ``save_episode``；一旦失败就将数据集标记为中毒并重新抛出异常。

        失败的 ``save_episode`` *无法*通过丢弃缓冲区来恢复：数据行和计数器
        在那些容易失败的步骤（视频编码、元数据提交）之前就已经提交了，因此
        下一个片段会复用相同的回合索引。这就是中毒锁存（poison latch）存在
        的原因，它会拒绝后续的片段和推送。
        """
        self._warn_if_push_in_flight()
        try:
            with self._episode_lock:
                dataset.save_episode()
        except Exception:
            self._dataset_poisoned = True
            with contextlib.suppress(Exception):
                dataset.clear_episode_buffer(delete_images=False)
            raise

    def _save_tail_episode(self, dataset, cfg) -> None:
        """提交该片段不完整的尾部回合；遇到真正的错误时显式报错。

        在 ``run()`` 的 ``finally`` 中运行。当数据集已处于中毒状态时提前
        返回，以便原始错误得以传播；当该片段没有录制任何内容时也提前返回，
        从而使 :meth:`_checked_save_episode` 只可能因保存本身损坏而失败。
        """
        if self._dataset_poisoned:
            return
        if not dataset.has_pending_frames():
            logger.info("No frames pending at segment end — nothing to save")
            return
        logger.info("Saving the segment's final (partial) episode")
        self._checked_save_episode(dataset)
        self._register_saved_episode(dataset, cfg)

    def _register_saved_episode(self, dataset, cfg) -> None:
        """保存后的记账工作，由轮转处和尾部保存处共用。

        尾部回合也必须计入 ``upload_every_n_episodes``，否则一个由短小
        片段组成的会话将永远不会触发后台推送。
        """
        self._episodes_since_push += 1
        self._needs_push.set()
        log_say(f"Episode {dataset.num_episodes} saved", cfg.play_sounds)
        if self._episodes_since_push >= self.config.upload_every_n_episodes:
            self._background_push(dataset, cfg)
            self._episodes_since_push = 0

    def _warn_if_push_in_flight(self) -> None:
        """在一次必须等待后台上传完成的保存之前发出警告。

        ``save_episode`` 会与后台 Hub 推送争夺 ``_episode_lock``，因此在
        上行链路较慢时，一次 ``/reset`` 可能会让机器人僵住数分钟。
        """
        if self._pending_push is not None and not self._pending_push.done():
            logger.warning(
                "Waiting for an in-flight Hub upload to finish before saving the episode — "
                "the robot will hold position until it completes..."
            )

    def teardown(self, ctx: RolloutContext) -> None:
        """冲刷待处理的推送、终结数据集，并断开硬件连接。"""
        play_sounds = ctx.runtime.cfg.play_sounds
        logger.info("Stopping sentry recording")
        log_say("Stopping sentry recording", play_sounds)

        # 干净地冲刷所有已排队/正在运行的推送。
        if self._push_executor is not None:
            logger.info("Shutting down push executor (waiting for pending pushes)...")
            self._push_executor.shutdown(wait=True)
            self._push_executor = None

        if ctx.data.dataset is not None:
            if self._dataset_poisoned:
                logger.error(
                    "The dataset may be partially committed (a save_episode failed mid-write): "
                    "closing it without pushing. Inspect it before using or uploading it."
                )
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()
            if (
                not self._dataset_poisoned
                and self._needs_push.is_set()
                and ctx.runtime.cfg.dataset
                and ctx.runtime.cfg.dataset.push_to_hub
            ):
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
        logger.info("Sentry strategy teardown complete")

    def _background_push(self, dataset, cfg) -> None:
        """在单工作线程执行器上排队一次 Hub 推送。

        执行器的 max_workers=1 保证同一时刻至多有一个推送在运行；
        提交的任务会被排队而不是被丢弃。
        """
        if self._push_executor is None:
            return
        if self._dataset_poisoned:
            logger.error("Skipping Hub push: a failed save_episode left the dataset possibly corrupt")
            return

        if self._pending_push is not None and not self._pending_push.done():
            logger.info("Previous push still in progress; queueing next")

        def _push():
            try:
                with self._episode_lock:
                    if self._dataset_poisoned:
                        # 在提交时的检查通过之后、任务排队期间变成了中毒状态。
                        logger.error(
                            "Skipping queued Hub push: a failed save_episode left the dataset "
                            "possibly corrupt"
                        )
                        return
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
