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

"""回合制（episodic）rollout 策略：行为与 ``lerobot-record`` 相同。

- 在每个录制回合中由策略驱动机器人。
- 在复位（reset）阶段，可以由可选的遥操作器驱动机器人，使操作员能够
  将环境恢复到起始配置。如果未连接遥操作器，机器人则保持在当前位置。
- 键盘控制：

      右方向键  —— 提前结束当前回合或复位阶段
      左方向键  —— 丢弃当前回合并重新录制
      Escape    —— 停止录制会话

数据集命名遵循 rollout 约定：仓库名必须以 ``rollout_`` 开头。
"""

from __future__ import annotations

import contextlib
import logging
import time

from lerobot.common.control_utils import (
    follower_smooth_move_to,
    teleop_smooth_move_to,
    teleop_supports_feedback,
)
from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import log_visualization_data

from ..configs import EpisodicStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, safe_push_to_hub, send_next_action

logger = logging.getLogger(__name__)


class EpisodicStrategy(RolloutStrategy):
    """由策略驱动的多回合录制，行为与 ``lerobot-record`` 相同。

    每个录制回合最多运行策略 ``dataset.episode_time_s`` 秒，每个策略
    动作录制一帧（``1/fps`` 的节拍——当
    ``interpolation_multiplier > 1`` 时，被插值的节拍只向机器人发送
    指令）。每个回合（最后一个除外）之后都有一个时长为
    ``dataset.reset_time_s`` 的复位阶段，以便操作员手动复位环境。
    在复位阶段，可由可选的遥操作器驱动机器人；如果不存在遥操作器，
    机器人会回到启动时捕获的初始关节位置。

    策略状态（隐藏状态、RTC 队列、插值器）会在每个录制回合开始时
    重置。

    键盘事件：
        右方向键  → 提前结束当前回合或复位阶段
        左方向键  → 丢弃并重新录制当前回合
        ESC       → 停止会话
    """

    config: EpisodicStrategyConfig

    def __init__(self, config: EpisodicStrategyConfig) -> None:
        super().__init__(config)
        self._listener = None
        self._events: dict | None = None

    def setup(self, ctx: RolloutContext) -> None:
        """启动推理引擎并挂载键盘监听器。"""
        self._init_engine(ctx)
        self._listener, self._events = init_keyboard_listener()
        logger.info("Episodic strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """多回合录制的主循环。"""
        cfg = ctx.runtime.cfg
        dataset_cfg = cfg.dataset
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        features = ctx.data.dataset_features

        fps = cfg.fps
        episode_time_s = dataset_cfg.episode_time_s
        reset_time_s = dataset_cfg.reset_time_s
        num_episodes = dataset_cfg.num_episodes
        single_task = dataset_cfg.single_task or cfg.task
        play_sounds = cfg.play_sounds

        display_compressed = (
            True
            if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
            else cfg.display_compressed_images
        )

        # 整个会话共用一个计时器：每个回合各有自己的节拍统计行，而运行摘要
        # 会跨回合取平均，且不包含未计时的复位阶段。
        timer = CycleTimer(fps, self._interpolator.multiplier)

        with VideoEncodingManager(dataset):
            try:
                recorded_episodes = 0
                while recorded_episodes < num_episodes and not events["stop_recording"]:
                    if ctx.runtime.shutdown_event.is_set():
                        break

                    # 在回合开始时重置策略状态（丢弃残留的隐藏状态 / 队列）
                    self._engine.reset()
                    self._interpolator.reset()
                    # 重置后的插值器会像循环启动时一样，在连续两个推理节拍内
                    # 重新预置，因此对跨越这两个节拍的分组予以豁免，而不是把
                    # 一个健康的回合误报为缓慢。
                    timer.restart()
                    self._engine.resume()

                    log_say(f"Recording episode {dataset.num_episodes}", play_sounds)
                    self._policy_loop(
                        ctx=ctx,
                        robot=robot,
                        events=events,
                        features=features,
                        timer=timer,
                        control_time_s=episode_time_s,
                        dataset=dataset,
                        single_task=single_task,
                    )

                    # 复位阶段，最后一个回合之后跳过（但重新录制时仍会执行）
                    if not events["stop_recording"] and (
                        recorded_episodes < num_episodes - 1 or events["rerecord_episode"]
                    ):
                        log_say("Reset the environment", play_sounds)

                        if teleop:
                            # 平滑交接，使切换到遥操作控制时不产生抖动。
                            # 对于有驱动的遥操作器：将主臂（leader）移动到从臂（follower）
                            # 当前的位置，使操作员接管时不必与机械臂较劲。
                            # 对于无驱动的遥操作器：由于主臂无法被驱动，改为将从臂
                            # 平滑移动到遥操作器当前的位姿。
                            # 可通过 --strategy.smooth_handover=false 完全禁用（适用于
                            # 离合器式遥操作器，这类设备在接合时会以机器人当前位姿
                            # 重新建立参考）。
                            if self.config.smooth_handover:
                                obs = robot.get_observation()
                                current_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
                                if (
                                    teleop_supports_feedback(teleop)
                                    and self.config.smooth_leader_to_follower_handover
                                ):
                                    logger.info("Smooth handover: moving leader arm to follower position")
                                    teleop_smooth_move_to(teleop, current_pos, duration_s=2)
                                    teleop.disable_torque()
                                else:
                                    logger.info("Smooth handover: sliding follower to teleop position")
                                    teleop_action = teleop.get_action()
                                    processed = ctx.processors.teleop_action_processor((teleop_action, obs))
                                    target = ctx.processors.robot_action_processor((processed, obs))
                                    follower_smooth_move_to(robot, current_pos, target, duration_s=1)

                        elif self.config.reset_to_initial_position:
                            # 没有遥操作器：让机器人回到启动时的位置。
                            self.return_to_initial_position(hw=ctx.hardware, duration_s=1)

                        self._reset_loop(
                            ctx=ctx,
                            robot=robot,
                            teleop=teleop,
                            events=events,
                            fps=fps,
                            control_time_s=reset_time_s,
                            display_data=cfg.display_data,
                            display_mode=cfg.display_mode,
                            display_compressed=display_compressed,
                        )

                    if events["rerecord_episode"]:
                        log_say("Re-record episode", play_sounds)
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                        dataset.clear_episode_buffer()
                        timer.log_episode_summary("discarded episode")

                        # 回到启动时捕获的初始关节位置
                        if not teleop and self.config.reset_to_initial_position:
                            self.return_to_initial_position(hw=ctx.hardware, duration_s=1)

                        continue

                    dataset.save_episode()
                    recorded_episodes += 1
                    timer.log_episode_summary(f"episode {dataset.num_episodes}")
            finally:
                # 保存当前回合中已缓冲的所有帧，以免意外异常或
                # KeyboardInterrupt 静默丢失已录制的数据。
                # suppress：缓冲区为空时 save_episode 会抛异常（此时没什么可丢失的）。
                logger.info("Episodic control loop ended — saving any in-progress episode")
                timer.log_run_summary()
                with contextlib.suppress(Exception):
                    dataset.save_episode()

    def _policy_loop(
        self,
        ctx: RolloutContext,
        robot,
        events: dict,
        features: dict,
        timer: CycleTimer,
        control_time_s: float,
        dataset,
        single_task: str,
    ) -> None:
        """单个回合的、由策略驱动的录制循环。

        *timer* 由 :meth:`run` 拥有并在各回合之间共享，因此其运行摘要
        覆盖整个会话；调用方会在回合之间重新激活（re-arm）它。
        """
        interpolator = self._interpolator

        timestamp = 0.0
        start_t = time.perf_counter()

        while timestamp < control_time_s:
            timer.tick(new_cycle=interpolator.needs_new_action())

            if events["exit_early"]:
                events["exit_early"] = False
                break

            if ctx.runtime.shutdown_event.is_set():
                break

            with timer.section("observe"):
                obs = robot.get_observation()
            with timer.section("process_obs"):
                obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(ctx.runtime.cfg.use_torch_compile, timer):
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
                        dataset.add_frame({**obs_frame, **action_frame, "task": single_task})

            timer.wait()
            timestamp = time.perf_counter() - start_t

    def _reset_loop(
        self,
        ctx: RolloutContext,
        robot,
        teleop,
        events: dict,
        fps: float,
        control_time_s: float,
        display_data: bool,
        display_mode: str,
        display_compressed: bool,
    ) -> None:
        """复位阶段循环：如果遥操作器可用则由其驱动机器人，不进行录制。"""
        processors = ctx.processors
        control_interval = 1.0 / fps

        timestamp = 0.0
        start_t = time.perf_counter()

        while timestamp < control_time_s:
            loop_start = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            if ctx.runtime.shutdown_event.is_set():
                break

            obs = robot.get_observation()

            if teleop is not None:
                act = teleop.get_action()
                act_teleop = processors.teleop_action_processor((act, obs))
                robot_action = processors.robot_action_processor((act_teleop, obs))
                robot.send_action(robot_action)

                if display_data:
                    obs_processed = processors.robot_observation_processor(obs)
                    log_visualization_data(
                        display_mode,
                        observation=obs_processed,
                        action=act_teleop,
                        compress_images=display_compressed,
                    )

            dt = time.perf_counter() - loop_start
            sleep_t = control_interval - dt
            precise_sleep(max(sleep_t, 0.0))
            timestamp = time.perf_counter() - start_t

    def teardown(self, ctx: RolloutContext) -> None:
        """终结数据集、停止监听器、推送到 hub，并断开硬件连接。"""
        cfg = ctx.runtime.cfg
        play_sounds = cfg.play_sounds

        log_say("Stop recording", play_sounds, blocking=True)

        if self._listener is not None:
            self._listener.stop()

        if ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()

        if (
            cfg.dataset is not None
            and cfg.dataset.push_to_hub
            and ctx.data.dataset is not None
            and safe_push_to_hub(
                ctx.data.dataset,
                tags=cfg.dataset.tags,
                private=cfg.dataset.private,
            )
        ):
            logger.info("Dataset uploaded to hub")
            log_say("Dataset uploaded to hub", play_sounds)

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=cfg.return_to_initial_position,
        )
        log_say("Exiting", play_sounds)
        logger.info("Episodic strategy teardown complete")
