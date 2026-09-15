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

"""基础 rollout 策略：自主执行策略，不记录数据。"""

from __future__ import annotations

import logging
import time

from lerobot.utils.cycle_timer import CycleTimer

from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


class BaseStrategy(RolloutStrategy):
    """自主策略 rollout，不记录数据。

    所有动作在送达机器人之前，都会经过 ``robot_action_processor``
    流水线处理。
    """

    def setup(self, ctx: RolloutContext) -> None:
        """初始化推理引擎。"""
        self._init_engine(ctx)
        logger.info("Base strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """运行自主控制循环，直到收到关闭信号或达到持续时长限制。"""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        timer = CycleTimer(
            cfg.fps,
            interpolator.multiplier,
            records_data=False,
            report=ctx.runtime.cadence_report,
        )

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Base strategy control loop started")

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
                with timer.section("telemetry"):
                    self._log_telemetry(obs_processed, action_dict, ctx.runtime)

                # 在节拍末尾处理文本查询通道（/vqa 回答、/autosteer 转向）；
                # 队列为空时为空操作。
                with timer.section("query"):
                    engine.pump_query(obs_processed)

                timer.wait()
        finally:
            logger.info("Base strategy control loop ended")
            timer.log_run_summary()

    def teardown(self, ctx: RolloutContext) -> None:
        """断开硬件连接并停止推理。"""
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Base strategy teardown complete")
