# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""以编程方式控制 rollout：在硬件和策略保持连接和热状态的同时，
启动、暂停、重新下达指令并停止策略。

:class:`RolloutController` 是对嵌入友好的核心：它自身没有任何 I/O，因此可以从
CLI（:class:`lerobot.rollout.interactive.InteractiveSession`）、网络服务器或
notebook 来驱动。完整的嵌入示例参见 ``docs/source/inference.mdx``。
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable
from enum import Enum
from threading import Event, Lock
from typing import TYPE_CHECKING

from .inference import QueryAnswer, QueryKind

if TYPE_CHECKING:
    from .context import RolloutContext
    from .strategies import RolloutStrategy

logger = logging.getLogger(__name__)


class LinkedEvent(Event):
    """一种 ``threading.Event``，其 ``is_set`` 同时反映父事件的状态。

    ``set``/``clear`` 只作用于本地标志，因此控制器可以置位和清除自己
    的片段停止请求，而不会掩盖（或重新激活）由 ``parent`` 携带的关闭事件。
    """

    _WAIT_SLICE_S = 0.05

    def __init__(self, parent: Event) -> None:
        super().__init__()
        self.parent = parent

    def is_set(self) -> bool:
        return super().is_set() or self.parent.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """等待本地或父标志，以短时间片轮询。"""
        deadline = None if timeout is None else time.perf_counter() + timeout
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.perf_counter()
            if remaining is not None and remaining <= 0:
                return False
            wait_slice = self._WAIT_SLICE_S if remaining is None else min(self._WAIT_SLICE_S, remaining)
            super().wait(wait_slice)
        return True


class AskResult(Enum):
    """:meth:`RolloutController.ask` 的结果。"""

    QUEUED = "queued"
    """已接受；答案将以 ``QUERY_ANSWERED`` 事件的形式到达。"""

    NOT_RUNNING = "not_running"
    """已拒绝：没有片段在运行，因此没有新的观测数据流。"""

    BUSY = "busy"
    """已拒绝：另一个问题或 autosteer 回合占用了单槽通道。"""

    UNSUPPORTED = "unsupported"
    """已拒绝：策略没有文本头；与其他情况不同，这在整个会话期间都是永久性的。"""


class RolloutEvent(Enum):
    """由 :class:`RolloutController` 发出的生命周期通知。

    所有事件都在运行 :meth:`RolloutController.serve` 的线程上触发；回调必须
    快速完成，且不能回调控制器的阻塞方法。
    """

    SEGMENT_STARTED = "segment_started"
    """一个控制循环片段即将运行（控制状态刚刚重置）。"""

    SEGMENT_ENDED = "segment_ended"
    """片段自行返回（例如 ``--duration`` 已耗尽）；机器人保持当前位置。"""

    RESET_STARTED = "reset_started"
    """正在执行重置：推理已暂停，机器人即将返回初始位置。"""

    RESET_DONE = "reset_done"
    """机器人已回到初始位置并保持。"""

    RESET_SKIPPED = "reset_skipped"
    """未捕获初始位置；机器人保持当前姿态。"""

    RESET_FAILED = "reset_failed"
    """返回动作中途出错：机器人可能停在任意姿态，*而不是*
    初始位置。"""

    QUERY_ANSWERED = "query_answered"
    """一个文本查询已得到解答（:meth:`RolloutController.ask` 提出的问题或 autosteer 回合）；
    载荷是 :class:`~lerobot.rollout.inference.QueryAnswer`，读取 ``answer`` 前请先检查 ``ok``。"""

    ENGINE_FAILED = "engine_failed"
    """引擎遇到不可恢复的错误；``serve()`` 即将返回。请读取
    :attr:`RolloutController.failure_traceback`（``STRATEGY_FAILED`` 同理）。"""

    STRATEGY_FAILED = "strategy_failed"
    """``strategy.run()`` 在片段中途抛出异常（机器人 I/O、记录等）；``serve()`` 即将返回。"""

    STOPPED = "stopped"
    """``serve()`` 即将返回（停止、前端 EOF、失败或父级关闭信号）。"""


class RolloutController:
    """通过线程安全的 start/reset/stop/set_task 调用驱动 rollout 策略。

    在 :meth:`start` 之前机器人处于空闲状态；每个运行*片段*在调用 :meth:`serve` 的
    线程上执行 ``strategy.run(ctx)``，而 ``strategy.setup``/``teardown`` 仍由调用者负责。

    - ``ctx.runtime.shutdown_event`` 必须是 :class:`LinkedEvent`，这样结束一个片段不会
      触发进程关闭：``build_rollout_context(cfg, LinkedEvent(shutdown_event))``。
    - 控制方法可从任何线程调用，并由内部锁串行化，因此从同一线程
      按顺序发出的调用保持该顺序。事件在 :meth:`serve` 线程上触发。
    - 一次性使用：一旦 :meth:`serve` 返回，控制器即处于终态 :attr:`stopped`，
      控制方法会以 ``False`` 拒绝，再次调用 :meth:`serve` 会抛出异常。
    """

    _POLL_INTERVAL_S = 0.2

    def __init__(
        self,
        strategy: RolloutStrategy,
        ctx: RolloutContext,
        on_event: Callable[[RolloutEvent, QueryAnswer | None], None] | None = None,
    ) -> None:
        stop_event = ctx.runtime.shutdown_event
        if not isinstance(stop_event, LinkedEvent):
            raise TypeError(
                "RolloutController requires ctx.runtime.shutdown_event to be a LinkedEvent so "
                "reset() can end a run segment without triggering process shutdown. Build the "
                "rollout context with build_rollout_context(cfg, LinkedEvent(shutdown_event))."
            )
        if not strategy.config.supports_interactive:
            # 一次性策略在 run() 退出时会终结其数据集，因此第二次 start()
            # 会向已终结的数据集写入记录（与 RolloutConfig.__post_init__ 中的守卫相同）。
            raise ValueError(
                f"RolloutController drives strategy.run() in restartable segments, but "
                f"'{strategy.config.type}' is a one-shot strategy "
                f"(supports_interactive is False). Use a strategy that honours the "
                f"restartable-run() contract (see RolloutStrategy in strategies/core.py)."
            )
        self._strategy = strategy
        self._ctx = ctx
        self._segment_stop = stop_event
        self._global_shutdown = stop_event.parent
        self._on_event = on_event
        self._initial_task = ctx.policy.inference.task
        self._autosteer_interval_s = ctx.runtime.cfg.autosteer_interval_s

        # 将控制方法串行化，使多写入者的任务更新保持调用顺序。
        self._control_lock = Lock()

        # 由控制方法（任意线程）写入，由 serve 循环消费。
        self._start_requested = Event()
        self._reset_requested = Event()
        self._stop_requested = Event()
        self._wake = Event()
        self._running = Event()
        # 在 serve() 退出时锁存（永不清除）；此后控制方法会拒绝执行。
        self._stopped = Event()
        self._strategy_failure_traceback: str | None = None

        # 答案只通过 serve 线程上调用的 pump_query() 离开引擎，因此这个
        # 观察者保持了"事件在 serve 线程上触发"的保证。
        ctx.policy.inference.set_answer_observer(self._on_query_answer)

    # ------------------------------------------------------------------
    # 内省
    # ------------------------------------------------------------------

    @property
    def task(self) -> str:
        """当前用于条件化推理的语言指令。"""
        return self._ctx.policy.inference.task

    @property
    def initial_task(self) -> str:
        """rollout 启动时使用的指令（由 :meth:`reset` 恢复）。"""
        return self._initial_task

    @property
    def running(self) -> bool:
        """控制循环片段正在执行时为 True。"""
        return self._running.is_set()

    @property
    def stopped(self) -> bool:
        """:meth:`serve` 返回后为 True；控制器处于终态（一次性）。"""
        return self._stopped.is_set()

    @property
    def failed(self) -> bool:
        """引擎或策略遇到不可恢复的错误时为 True。"""
        return self._ctx.policy.inference.failed or self._strategy_failure_traceback is not None

    @property
    def failure_traceback(self) -> str | None:
        """当 :attr:`failed` 为 True 时，失败的格式化回溯信息。"""
        return self._strategy_failure_traceback or self._ctx.policy.inference.failure_traceback

    # ------------------------------------------------------------------
    # 控制方法（可从任何线程调用）
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """请求一个控制循环片段，该片段在 :meth:`serve` 线程上执行。

        片段已排程时返回 ``True``；当已有片段在运行、发生故障之后，
        或控制器正在停止/已停止时返回 ``False``（被接受的 start 永远不会执行）。
        """
        with self._control_lock:
            if self._stopped.is_set() or self._stop_requested.is_set() or self.failed:
                return False
            if self._running.is_set():
                return False
            self._start_requested.set()
            self._wake.set()
            return True

    def reset(self) -> bool:
        """停止运动，将机器人返回初始位置，并恢复启动任务。

        硬件和策略保持热状态；调用 :meth:`start` 可再次运行。
        当任务被恢复（即任务曾被修改）时返回 ``True``；
        当任务本来就是启动任务，或控制器正在停止/已停止时返回 ``False``。
        """
        with self._control_lock:
            if self._stopped.is_set() or self._stop_requested.is_set():
                return False
            # 最后一条命令优先：取消待处理的 start()。先置标志，后停片段
            # （参见 _run_segment 中的顺序说明）。
            self._start_requested.clear()
            # 回到起点，因此排序器也要停止：否则会覆盖已恢复的任务。
            self._ctx.policy.inference.stop_autosteer()
            # 在这里恢复，而不是稍后在 serve 线程上恢复，这样后续的 set_task() 才能生效。
            restored = self._ctx.policy.inference.set_task(self._initial_task)
            self._reset_requested.set()
            self._segment_stop.set()
            self._wake.set()
            return restored

    def stop(self) -> None:
        """结束 :meth:`serve`，使调用者可以运行 ``strategy.teardown(ctx)``。幂等。"""
        with self._control_lock:
            if self._stopped.is_set():
                return
            self._start_requested.clear()  # 最后一条命令优先，参见 reset()
            self._stop_requested.set()
            self._segment_stop.set()
            self._wake.set()

    def set_task(self, task: str) -> bool:
        """更改策略遵循的指令，从下一次推理开始生效。

        当值确实发生变化时返回 ``True``。片段运行期间调用也是安全的：
        引擎在自己的推理线程上应用切换（同步后端还会丢弃在
        先前指令下预计算的动作）。控制器正在停止/已停止后
        会被拒绝（``False``，引擎不受影响）。会停止 :meth:`autosteer`，
        因为它会覆盖这条指令。
        """
        with self._control_lock:
            if self._stopped.is_set() or self._stop_requested.is_set():
                return False
            self._ctx.policy.inference.stop_autosteer()
            return self._ctx.policy.inference.set_task(task)

    def ask(self, question: str) -> AskResult:
        """将一个关于机器人当前所见的问题加入队列。

        立即返回；答案以 :attr:`RolloutEvent.QUERY_ANSWERED` 事件的形式到达，
        且在调用者线程上绝不会触碰策略。拒绝情况包括：
        :attr:`AskResult.UNSUPPORTED`（没有文本头）、
        :attr:`AskResult.NOT_RUNNING`（没有片段在运行，因此没有可作答的观测）、
        或 :attr:`AskResult.BUSY`（通道被占用）。
        """
        # 静态能力：最先检查，且在控制锁之外。
        if not self._ctx.policy.inference.supports_text_queries:
            return AskResult.UNSUPPORTED
        with self._control_lock:
            # 与 _run_segment 清除 _running 使用同一把锁，因此问题永远不会成为孤儿。
            if not self._running.is_set():
                return AskResult.NOT_RUNNING
            if not self._ctx.policy.inference.ask(question):
                return AskResult.BUSY
            return AskResult.QUEUED

    @property
    def autosteer_goal(self) -> str | None:
        """当前驱动任务的高层目标（如果有）。"""
        return self._ctx.policy.inference.autosteer_goal

    def autosteer(self, goal: str) -> AskResult:
        """让策略分解 ``goal`` 并自主驱动其子任务。

        每隔 ``autosteer_interval_s`` 秒，引擎向策略询问下一个子任务，
        并通过*引擎的* ``set_task`` 应用它。规划进度保存在策略中，
        因此排序器不会存活过片段；它也会被 :meth:`reset` 和
        :meth:`set_task` 停止。守卫和拒绝值与 :meth:`ask` 相同。
        """
        if not self._ctx.policy.inference.supports_text_queries:
            return AskResult.UNSUPPORTED
        with self._control_lock:
            if not self._running.is_set():
                return AskResult.NOT_RUNNING
            self._ctx.policy.inference.start_autosteer(goal, self._autosteer_interval_s)
            return AskResult.QUEUED

    def stop_autosteer(self) -> str | None:
        """停止排序器，并返回其正在驱动的目标（若无则返回 ``None``）。"""
        with self._control_lock:
            return self._ctx.policy.inference.stop_autosteer()

    # ------------------------------------------------------------------
    # Serve 循环（阻塞调用线程）
    # ------------------------------------------------------------------

    def serve(self) -> None:
        """处理控制请求，直到 :meth:`stop`、发生故障或父级关闭。

        阻塞调用线程；运行片段在此执行，:class:`RolloutEvent`
        通知通过 ``on_event`` 发出。一次性使用：一旦返回，控制器
        即处于终态 :attr:`stopped`，再次调用 ``serve()`` 会抛出异常。
        """
        if self._stopped.is_set():
            raise RuntimeError(
                "RolloutController.serve() is one-shot: this controller has already stopped. "
                "Build a new controller to run again."
            )
        try:
            while not self._global_shutdown.is_set():
                if self._ctx.policy.inference.failed:
                    self._emit(RolloutEvent.ENGINE_FAILED)
                    break
                if self._strategy_failure_traceback is not None:
                    self._emit(RolloutEvent.STRATEGY_FAILED)
                    break
                if self._stop_requested.is_set():
                    break
                if self._reset_requested.is_set():
                    self._reset_requested.clear()
                    self._reset_robot()
                    continue
                if self._start_requested.is_set():
                    # 在一个原子步骤中消费请求并标记片段运行中，
                    # 这样并发的 start() 无法在运行中片段背后重新激活该标志。
                    with self._control_lock:
                        starting = self._start_requested.is_set()
                        if starting:
                            self._start_requested.clear()
                            self._running.set()
                    if starting:
                        self._run_segment()
                    continue
                # 控制循环中每 tick 泵送的空闲对应物：送达一个
                # 恰好在片段结束时落地的答案。
                self._ctx.policy.inference.pump_query()
                self._wake.wait(timeout=self._POLL_INTERVAL_S)
                self._wake.clear()
        finally:
            # 先锁存再通告，这样对 STOPPED 做出反应的观察者看到的是已停止的控制器。
            self._stopped.set()
            self._emit(RolloutEvent.STOPPED)

    def _run_segment(self) -> None:
        """执行一个 ``strategy.run`` 片段，直到被中断或完成。

        serve 循环已经设置了 ``_running``（在控制锁下），因此本方法
        必须在每个退出路径上清除它。
        """
        engine = self._ctx.policy.inference
        try:
            # 在检查请求标志*之前*清除本地标志：控制方法先置标志
            # 后置事件，因此竞争中的 reset()/stop() 要么在这里被看到，
            # 要么立即结束新循环。清除操作还会吸收垂死引擎的信号：
            # 因此要检查 engine.failed。
            self._segment_stop.clear()
            if (
                self._stop_requested.is_set()
                or self._reset_requested.is_set()
                or self._global_shutdown.is_set()
                or engine.failed
            ):
                return
            self._strategy.reset_control_state()
            self._emit(RolloutEvent.SEGMENT_STARTED)
            try:
                self._strategy.run(self._ctx)
            except Exception:
                # 路由到与引擎失败相同的公共失败面，
                # 而不是以一个看似干净的 STOPPED 从 serve() 中展开。
                self._strategy_failure_traceback = traceback.format_exc()
                logger.exception("Rollout strategy failed mid-segment")
            finally:
                engine.pause()
        finally:
            # 在控制锁下一起清除并丢弃：ask() 在同一把锁下以 _running 为门控，
            # 因此问题要么在此之前落地并被丢弃，要么被直接拒绝。
            with self._control_lock:
                self._running.clear()
                # 排序器不能存活过片段：其规划进度保存在策略中。
                engine.stop_autosteer()
                dropped = engine.drop_pending_query()
                # 否则空闲泵送会在排序器结束后通告子任务；VQA 保留。
                engine.drop_ready_subtask_answers()
            # 只有操作者的问题才值得报告。
            if dropped is not None and dropped.kind is QueryKind.VQA:
                self._emit(
                    RolloutEvent.QUERY_ANSWERED,
                    QueryAnswer(question=dropped.text, error="the run ended before it could be answered"),
                )
        if engine.failed or self._strategy_failure_traceback is not None:
            return  # serve 循环发出失败事件并关闭
        if not (
            self._stop_requested.is_set() or self._reset_requested.is_set() or self._global_shutdown.is_set()
        ):
            self._emit(RolloutEvent.SEGMENT_ENDED)

    def _reset_robot(self) -> None:
        """暂停推理并将机器人返回初始位置（任务已由 :meth:`reset` 恢复）。"""
        self._emit(RolloutEvent.RESET_STARTED)
        self._ctx.policy.inference.pause()
        if not self._ctx.hardware.initial_position:
            logger.warning("No initial position captured — skipping the return move")
            self._emit(RolloutEvent.RESET_SKIPPED)
        elif self._strategy.return_to_initial_position(self._ctx.hardware):
            self._emit(RolloutEvent.RESET_DONE)
        else:
            # RESET_DONE 保证"已回到初始位置"；失败的移动不能声称这一点。
            self._emit(RolloutEvent.RESET_FAILED)

    def _on_query_answer(self, answer: QueryAnswer) -> None:
        """引擎答案观察者——在 serve 线程上运行（参见 ``__init__``）。"""
        self._emit(RolloutEvent.QUERY_ANSWERED, answer)

    def _emit(self, event: RolloutEvent, payload: QueryAnswer | None = None) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event, payload)
        except Exception:  # 损坏的观察者不能杀死 serve 循环
            logger.exception("Error in RolloutController event callback for %s", event)
