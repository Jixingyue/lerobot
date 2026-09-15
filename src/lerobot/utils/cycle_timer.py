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

"""实时控制循环的节拍控制与报告。

LeRobot 中每一个以固定频率驱动硬件的循环——遥操作、
录制、回放、策略 rollout——除了自身的循环体之外，都有同样两项职责：
睡眠适当的时间，使下一次迭代准时开始；以及在无法跟上节奏时
告知用户。:class:`CycleTimer` 同时负责这两件事，因此这些循环共用
同一套节拍规则、同一条慢循环警告和同一份运行结束报告。
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import time
from collections.abc import Callable, Iterator

from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _SectionStat:
    """一个具名循环体步骤的调用次数与计时信息。"""

    calls: int = 0
    total: float = 0.0
    worst: float = 0.0


@dataclasses.dataclass
class _CadenceStats:
    """在一个报告窗口内累积的节拍计数器。

    一个窗口是两个 episode 边界之间的循环区段；运行级别的
    实例则是所有已关闭窗口折叠在一起的结果。每个字段都是可加的，因此
    两者可以共用同一个格式化器。

    耗时以**逐 tick 间隔之和**的形式累积，而不是取首/末
    时间戳之差，这使得有效节拍不会混入循环并未运行的区段：
    即各 episode 之间不计时的重置阶段，以及
    :meth:`CycleTimer.restart` 所丢弃的一次性阻塞工作。
    """

    ticks: int = 0
    work: float = 0.0
    span: float = 0.0
    span_ticks: int = 0
    slot_overruns: int = 0
    starved_ticks: int = 0
    groups_judged: int = 0
    groups_over: int = 0
    group_work: float = 0.0
    group_work_worst: float = 0.0
    span_misses: int = 0
    sleep: float = 0.0
    sleep_worst: float = 0.0
    sections: dict[str, _SectionStat] = dataclasses.field(default_factory=dict)

    def record_section(self, name: str, elapsed: float) -> None:
        """累积一次 *name* 步骤的计时运行结果。"""
        stat = self.sections.setdefault(name, _SectionStat())
        stat.calls += 1
        stat.total += elapsed
        stat.worst = max(stat.worst, elapsed)

    def fold(self, other: _CadenceStats) -> None:
        """将一个已结束的窗口合并到此累积窗口中。"""
        self.ticks += other.ticks
        self.work += other.work
        self.span += other.span
        self.span_ticks += other.span_ticks
        self.slot_overruns += other.slot_overruns
        self.starved_ticks += other.starved_ticks
        self.groups_judged += other.groups_judged
        self.groups_over += other.groups_over
        self.group_work += other.group_work
        self.group_work_worst = max(self.group_work_worst, other.group_work_worst)
        self.span_misses += other.span_misses
        self.sleep += other.sleep
        self.sleep_worst = max(self.sleep_worst, other.sleep_worst)
        for name, stat in other.sections.items():
            mine = self.sections.setdefault(name, _SectionStat())
            mine.calls += stat.calls
            mine.total += stat.total
            mine.worst = max(mine.worst, stat.worst)


class CycleTimer:
    """控制控制循环 tick 的节拍，并对照循环的目标节奏报告计时情况。

    在默认的 ``multiplier == 1`` 下，每个 tick 就是一个周期，约定仅此而已：
    计时器会睡眠，使每次迭代耗时 ``1/fps``；当循环体本身无法
    放进该预算时发出警告；并累积 :meth:`log_episode_summary` 和
    :meth:`log_run_summary` 所报告的统计信息。遥操作、
    录制和回放都正是以这种方式使用它——``tick()``、各个 section、``wait()``。

    本 docstring 的其余部分讨论的是策略 rollout，因为它引入了
    动作插值。当 ``interpolation_multiplier == N`` 时，控制循环每个
    策略周期运行
    ``N`` 个 tick：机器人指令以 ``fps × N`` Hz 的频率每个 tick 发出，
    而策略推理和数据集录制以 ``fps`` Hz 的频率每个周期推进一步。
    推理运行在为插值器补充数据的那个 tick 上；各策略在发出
    策略自身端点动作（即周期的最后一个 tick）的那个 tick 上录制
    帧，并将其与产生该动作的观测配对。该 tick 由
    :attr:`~lerobot.utils.action_interpolator.ActionInterpolator.emitted_policy_action`
    标识，因此录制过程自身不携带任何相位状态。

    计时器维护两套相互独立的时间概念，因为节拍控制和
    报告需要不同的锚点：

    **节拍控制**使用 ``_cycle_start``，每当调用方报告一个
    新周期时就重新锚定，因此每个 tick 的截止时间都是相对于产生当前
    策略动作的那个 tick 的绝对偏移。这样，一个缓慢的 tick 会
    向其后的 tick 借用预算，而不是把整个周期向后推。在
    multiplier 为 2 的 30 FPS 下，一个 25 ms 的策略 tick 后接一个 5 ms
    的插值 tick，仍然能够放进 33.3 ms 的周期内。

    **报告**将连续 ``N`` 个 tick 的*工作量*相加——即循环体
    实际花费的时间，不含节拍睡眠——并在该总和超过
    ``1/fps`` 预算时发出警告，因为正是这种情况导致循环无法维持
    帧率。上面 25 ms + 5 ms 的周期总和为 30 ms，因此保持
    静默。

    有两个特性使这一度量成为正确的选择，而且两者都至关重要：

    - 它**与相位无关**。分组按 tick 计数且从不
      重新锚定，因此它们会与周期逐渐错位：插值器的
      第一个缓冲区只含一个动作，这使两者永久相差
      一个 tick。无论一个分组从哪个 tick 开始，工作量总和都相同，
      而对一个错位分组求挂钟时间跨度则会把一整段
      节拍睡眠吞进去，把一个健康的循环误报为缓慢。
    - 当插值器**缺数据或冻结**时（异步后端没有产生动作，
      或 DAgger 的暂停与纠偏阶段），它仍然有意义，此时
      每个 tick 都报告 ``new_cycle=True``。若把
      度量绑定到周期完成上，警告就会恰好在循环缓慢时
      永远无法触发。

    节拍睡眠*期间*损失给操作系统的时间被有意排除在
    警告之外，这与这些循环在引入插值之前的行为一致，当时
    也只测量循环体开头到其睡眠之间的工作量：这不是
    调用方能够处理的事情，而且在负载较高的机器上它会
    不停地触发。不过它并非完全不可见——每当实际达到的
    组起始到组起始节奏（含睡眠）未能满足工作量总和已满足的
    预算时，都会以 DEBUG 级别记录。单个 tick 超时只会
    损失插值平滑度，同样只是一条 DEBUG 记录。

    除了逐 tick 的遥测信息，计时器还会累积节拍统计——
    有效帧率、超出预算的频率、循环体工作量的去向——并分两次
    报告：每个 episode 一行摘要
    （:meth:`log_episode_summary`），以及整个运行的完整报告块
    （:meth:`log_run_summary`，在循环的 ``finally`` 中调用，因此
    即使遇到 ``KeyboardInterrupt`` 也仍会产生报告）。两者都发送到
    ``logger.info``，或者在提供了 ``report`` 接收函数时发送到该接收处
    ——供静音 logger 的调用方使用。只有
    摘要会被路由；慢循环警告和 DEBUG 遥测仍保留在
    logger 上。

    用法::

        timer = CycleTimer(cfg.fps, interpolator.multiplier)
        try:
            while ...:
                timer.tick(new_cycle=interpolator.needs_new_action())
                with timer.section("observe"):
                    ...  # one section per big loop-body step
                timer.wait()
                if episode_boundary:
                    timer.log_episode_summary()
        finally:
            timer.log_run_summary()

    没有插值器的循环只需调用 ``timer.tick()``：在 multiplier 为 1 时
    锚点无论如何都会在每个 tick 重新取定，因此该标志没有影响。

    ``new_cycle=True`` 标记插值器请求全新
    策略动作的那些 tick，使节拍锚点与实际推理
    节奏保持对齐（否则插值器第一个只含单动作的缓冲区会使
    之后每个周期都产生一个 tick 的相位偏移）。
    """

    #: 一个分组的挂钟时间跨度在 :meth:`_report_achieved_cadence` 发出提示之前
    #: 允许超出周期预算的比例。``precise_sleep`` 会自旋到其
    #: 截止时间，但调度器抖动每个 tick 仍会耗费数十微秒，因此
    #: 若没有任何容差，该提示几乎会在每个分组上触发——一次健康的 30 Hz
    #: 运行在 576 个分组中有 556 个都记录了它。
    SPAN_TOLERANCE = 0.01

    def __init__(
        self,
        fps: float,
        multiplier: int = 1,
        records_data: bool = True,
        report: Callable[[str], None] | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        self.fps = fps
        self.multiplier = multiplier
        self.tick_interval = 1.0 / (fps * multiplier)
        self.cycle_interval = 1.0 / fps
        self.records_data = records_data
        self._report = report if report is not None else logger.info
        # 节拍锚点——由 ``new_cycle`` 重新锚定。
        self._cycle_start: float | None = None
        self._tick_start: float | None = None
        self._ticks_done = 0
        # 报告累加器——严格按 tick 推进，从不重新锚定。
        self._group_ticks = 0
        self._group_work = 0.0
        self._groups_closed = 0
        # 挂钟锚点 + 上一个已关闭分组的工作量，用于覆盖工作量总和
        # 所看不到内容的实际节奏遥测。
        self._group_start: float | None = None
        self._last_group_work = 0.0
        # 统计信息：``_window`` 是当前进行中的 episode，``_run`` 是迄今
        # 已关闭的每个窗口。每个 tick 只触碰 ``_window``，因此逐 tick
        # 的记账始终集中在一个对象上；边界处会将其折叠进 ``_run``。
        self._window = _CadenceStats()
        self._run = _CadenceStats()
        self._windows_closed = 0
        self._prev_tick_start: float | None = None
        # 边界效果在下一次 ``tick()`` 时生效——参见 :meth:`tick`。
        self._pending_close: str | None = None
        self._drop_next_gap = False

    def restart(self) -> None:
        """在控制状态于运行中途被重置后，重新启用启动豁免。

        当插值器在循环继续运行期间被重置时调用（预热刷新、
        DAgger 切回自主模式）：此时它会用单动作缓冲区重新预置，
        于是推理又一次发生在两个连续的 tick 上，而横跨这两个
        tick 的分组超出预算是合理的。在循环体内任何其他一次性
        阻塞工作之后（例如 DAgger 的平滑交接爬坡）也应调用它，
        因为这类耗时不属于稳态节奏。
        只有报告累加器会被清空——节拍状态保持不动，因此
        ``tick()`` 与 ``wait()`` 之间的 restart 不会跳过任何一次节拍睡眠。
        统计信息描述整个运行并有意保留；唯一被丢弃的
        是*在下一个 tick 处结束*的那一段耗时，也就是包含
        阻塞工作的那一段，因此它不会被计入有效节奏。
        """
        self._group_ticks = 0
        self._group_work = 0.0
        self._groups_closed = 0
        self._group_start = None
        self._last_group_work = 0.0
        self._drop_next_gap = True

    @contextlib.contextmanager
    def section(self, name: str) -> Iterator[None]:
        """为节拍摘要计时循环体的一个大步骤。

        用它包裹 :meth:`tick` 与 :meth:`wait` 之间的粗粒度步骤——观测、
        处理、推理、执行、录制——以便摘要能够说明循环体工作量花在了
        哪里。请保持各个 section **扁平且互不重叠**：占比是相对于
        所测总工作量报告的，因此相互嵌套会导致
        重复计数。只在部分 tick 上运行的步骤（录制、引擎
        拉取）只会报告更少的调用次数。
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self._window.record_section(name, time.perf_counter() - start)

    def note_starved_tick(self) -> None:
        """记录一个因为引擎未产生动作而没有动作可发送的 tick。

        由 :func:`send_next_action` 调用。这样的 tick 既不发出指令也不
        录制任何内容，因此包含大量此类 tick 的运行写出的数据集会
        *短于*采集所经历的挂钟时间——这在数据集自身中不可见，因为其
        时间戳是由帧索引合成的。把计数呈现出来是
        用户能得到的唯一警告。
        """
        self._window.starved_ticks += 1

    def _report_achieved_cadence(self, group_start: float) -> None:
        """记录一个已关闭分组实际达到的节奏（含睡眠）。

        按组起始到组起始来测量，因此与工作量总和不同，这一
        跨度包含节拍睡眠。当它未达到预算而工作量
        并未超限时，说明时间丢失在循环体*之外*——计时器
        睡过头，或操作系统在睡眠中途将进程调度走——这正是
        :meth:`wait` 的警告所看不到的那一类
        不足。这不是循环自身的问题，也无法以同样方式处理，因此保持在 DEBUG 级别。
        """
        if self._group_start is None:
            return
        span = group_start - self._group_start
        if (
            span <= self.cycle_interval * (1.0 + self.SPAN_TOLERANCE)
            or self._last_group_work > self.cycle_interval
        ):
            return
        self._window.span_misses += 1
        logger.debug(
            "Control loop held only %.1f Hz against a %g Hz target, though its loop-body work "
            "(%.1f ms) fit the %.1f ms budget: %.1f ms went missing outside the loop body "
            "(sleep overshoot or CPU starvation while pacing).",
            1.0 / span,
            self.fps,
            self._last_group_work * 1000,
            self.cycle_interval * 1000,
            (span - self.cycle_interval) * 1000,
        )

    def tick(self, new_cycle: bool = False) -> None:
        """标记一个控制 tick 的开始。在循环体开头调用。

        episode 边界和 :meth:`restart` 也都在这里生效，
        耗时同样在这里累积——两者出于同一个原因。调用方是从
        循环体*内部*到达这两处的：轮换发生在阻塞的
        ``save_episode`` 之后、:meth:`wait` 之前。因此进行中的
        tick 是正在结束的 episode 的最后一个 tick，其耗时应归属该
        episode，而必须丢弃的间隔是*在下一个 tick 处结束*的那一段
        （即包含阻塞工作的那一段），而不是我们身后已经过去的健康
        间隔。推迟到此处处理，无论调用方当时位于循环体中间还是
        两个 tick 之间，都能把两者处理正确。
        """
        self._tick_start = time.perf_counter()
        if self._pending_close is not None:
            self._close_window(self._pending_close)
            self._pending_close = None
            self._drop_next_gap = True
        # 耗时按 tick 起始到 tick 起始测量，逐段相加以求和，而不是
        # 取首/末一对时间戳，这样才有可能丢弃其中某一段。
        if self._prev_tick_start is not None and not self._drop_next_gap:
            self._window.span += self._tick_start - self._prev_tick_start
            self._window.span_ticks += 1
        self._drop_next_gap = False
        self._prev_tick_start = self._tick_start
        if new_cycle or self._cycle_start is None:
            self._cycle_start = self._tick_start
            self._ticks_done = 0

    def wait(self) -> None:
        """睡眠直到本 tick 的截止时间。在循环体末尾调用。

        若一组 ``multiplier`` 个 tick 的工作量超过 ``1/fps`` 预算，
        就意味着无法维持策略/录制节奏——这是唯一会
        发出警告的情况。
        """
        now = time.perf_counter()
        if self._cycle_start is None or self._tick_start is None:
            return
        tick_start = self._tick_start
        tick_dt = now - tick_start
        if self._group_ticks == 0:
            self._report_achieved_cadence(tick_start)
            self._group_start = tick_start
        self._tick_start = None
        self._ticks_done += 1
        self._group_ticks += 1
        self._group_work += tick_dt

        stats = self._window
        stats.ticks += 1
        stats.work += tick_dt
        if tick_dt > self.tick_interval:
            stats.slot_overruns += 1

        deadline = self._cycle_start + self._ticks_done * self.tick_interval
        if self._ticks_done >= self.multiplier:
            self._cycle_start = None

        warned = False
        if self._group_ticks >= self.multiplier:
            group_work = self._group_work
            self._group_ticks = 0
            self._group_work = 0.0
            self._last_group_work = group_work
            self._groups_closed += 1
            # 第一个分组属于启动阶段，而非稳态：插值器用单个动作
            # 预置其缓冲区，因此推理发生在两个连续的 tick 上，而且一次性
            # 开销（惰性设备初始化、相机
            # 预热）也落在这里。报告它会导致每次健康的
            # 启动都发出警告，而把它计入平均则会使运行摘要产生偏差。
            if self._groups_closed > 1:
                stats.groups_judged += 1
                stats.group_work += group_work
                stats.group_work_worst = max(stats.group_work_worst, group_work)
                if group_work > self.cycle_interval:
                    stats.groups_over += 1
                    warned = True
                    consequence = (
                        "Dataset frames might be dropped and robot control might be unstable."
                        if self.records_data
                        else "Robot control might be unstable."
                    )
                    logger.warning(
                        f"Control loop is running slower ({1 / group_work:.1f} Hz) than the target FPS "
                        f"({self.fps:g} Hz). {consequence} Common causes are: 1) Camera FPS not keeping up "
                        "2) Policy inference (action or text) taking too long 3) CPU starvation"
                    )
        # 一个未突破周期预算的迟到 tick 只会损失插值
        # 平滑度，因此只是一条 DEBUG 记录——而在 multiplier 为 1 时本就
        # 没有平滑度可损失，上面的警告已说明全部情况。关闭分组的
        # tick 也包含在内，除非它已经发出过警告；这里曾使用 ``elif``，
        # 结果吞掉了每个第 multiplier 个 tick 的超时。
        if self.multiplier > 1 and not warned and now > deadline and tick_dt > self.tick_interval:
            logger.debug(
                "Control tick overran its %.1f ms slot (took %.1f ms). Interpolated commands are sent "
                "less smoothly; the %g Hz %s cadence is judged per group of %d ticks.",
                self.tick_interval * 1000,
                tick_dt * 1000,
                self.fps,
                "policy/recording" if self.records_data else "policy",
                self.multiplier,
            )
        if (sleep_t := deadline - now) > 0:
            sleep_start = time.perf_counter()
            precise_sleep(sleep_t)
            slept = time.perf_counter() - sleep_start
            stats.sleep += slept
            stats.sleep_worst = max(stats.sleep_worst, slept)

    # ------------------------------------------------------------------
    # 节拍摘要
    # ------------------------------------------------------------------

    def _effective_hz(self, stats: _CadenceStats) -> float | None:
        """实际达到的*指令*速率（Hz）；测量数据过少时为 ``None``。"""
        if stats.span_ticks == 0 or stats.span <= 0:
            return None
        return stats.span_ticks / stats.span

    @property
    def _judged(self) -> str:
        """在报告中如何称呼一个由 ``multiplier`` 个 tick 组成的受评判分组。

        多于一个 tick 的分组是*周期*。它们与插值器实际的策略周期
        相差一个 tick 的相位（参见类 docstring），但无论如何每个策略动作
        恰好对应一个分组，因此读者关心的计数是相同的。在 multiplier 为 1 时，
        一个分组*就是*一个 tick，称之为周期会凭空造出一个该循环并不具备的概念。
        """
        return "cycle" if self.multiplier > 1 else "tick"

    def _summary_line(self, stats: _CadenceStats) -> str:
        """一个窗口的一行摘要：维持的节奏、超出预算的次数、耗时。"""
        ms = 1e3
        parts: list[str] = []
        hz = self._effective_hz(stats)
        ticks = f"{stats.ticks} tick{'s' if stats.ticks != 1 else ''}"
        if hz is None:
            parts.append(f"{ticks}, too short to measure a rate")
        else:
            # 只有带插值的循环才有两个需要区分的速率。
            rate = f"{hz / self.multiplier:.2f} Hz policy" if self.multiplier > 1 else f"{hz:.2f} Hz"
            parts.append(f"{rate} vs {self.fps:g} Hz target")
            parts.append(f"{ticks}, {stats.span:.1f} s measured")
        if stats.groups_judged:
            parts.append(
                f"{stats.groups_over}/{stats.groups_judged} {self._judged}s over the "
                f"{self.cycle_interval * ms:.1f} ms budget (work mean "
                f"{stats.group_work / stats.groups_judged * ms:.1f} ms, worst "
                f"{stats.group_work_worst * ms:.1f} ms)"
            )
        if stats.starved_ticks:
            parts.append(f"starved ticks: {stats.starved_ticks}")
        return " · ".join(parts)

    def _summary_lines(self, stats: _CadenceStats, heading: str) -> list[str]:
        """一个窗口的完整多行报告（关于 *cycles* 与 *ticks* 参见 :meth:`_judged`）。"""
        ms = 1e3
        # 带插值的循环需要说明两个预算，普通循环只有单个
        # 逐 tick 预算，整个报告块中不存在第二个速率。
        if self.multiplier > 1:
            target = (
                f"target {self.fps:g} Hz × {self.multiplier} ({self.tick_interval * ms:.1f} ms tick "
                f"slot, {self.cycle_interval * ms:.1f} ms cycle budget)"
            )
            judged = f"{stats.groups_judged} cycles judged"
        else:
            target = f"target {self.fps:g} Hz ({self.cycle_interval * ms:.1f} ms budget per tick)"
            judged = f"{stats.groups_judged} judged"
        # 样本量无条件放入标题：这里的其他每个数字都是
        # 基于它的速率或平均值，因此读者需要它来评判其中任何一项——
        # 而当窗口短到根本无法测量速率时，下面的有效节奏行
        # 会被跳过。
        lines = [
            f"Cadence summary — {heading} · {target}: "
            f"{stats.ticks} tick{'s' if stats.ticks != 1 else ''}, {judged}"
        ]
        hz = self._effective_hz(stats)
        if hz is not None:
            rate = (
                f"{hz / self.multiplier:.2f} Hz policy / {hz:.2f} Hz commands"
                if self.multiplier > 1
                else f"{hz:.2f} Hz"
            )
            # 该跨度并非 ticks/rate：它是逐 tick 间隔之和，因此每个窗口边界
            # 和每次 ``restart()`` 都会使它少算一个间隔。
            lines.append(f"  effective cadence: {rate} over {stats.span:.1f} s measured")
        if stats.groups_judged:
            lines.append(
                f"  {self._judged}s over the {self.cycle_interval * ms:.1f} ms work budget: "
                f"{stats.groups_over}/{stats.groups_judged} "
                f"({100 * stats.groups_over / stats.groups_judged:.1f}%) — work mean "
                f"{stats.group_work / stats.groups_judged * ms:.1f} ms, worst "
                f"{stats.group_work_worst * ms:.1f} ms"
            )
        if self.multiplier > 1:
            lines.append(
                f"  ticks over their {self.tick_interval * ms:.1f} ms slot: "
                f"{stats.slot_overruns}/{stats.ticks} (costs interpolation smoothness only)"
            )
        if stats.starved_ticks:
            lines.append(
                f"  ticks with no action to send (inference engine starved): {stats.starved_ticks} — "
                "each commanded nothing and recorded no frame"
            )
        if stats.span_misses:
            lines.append(
                f"  {self._judged}s whose cadence slipped outside the loop body (sleep overshoot / "
                f"CPU starvation while pacing): {stats.span_misses}"
            )
        if stats.sections:
            lines.append("  loop-body steps (share of measured work):")
            width = max(len(name) for name in stats.sections)
            for name, stat in stats.sections.items():
                if stat.calls == 0:
                    continue
                share = 100 * stat.total / stats.work if stats.work > 0 else 0.0
                lines.append(
                    f"    {name:<{width}}  mean {stat.total / stat.calls * ms:6.2f} ms · worst "
                    f"{stat.worst * ms:6.2f} ms · {share:5.1f}% of work · {stat.calls} calls"
                )
        if stats.ticks:
            lines.append(
                f"  pacing headroom: {stats.sleep / stats.ticks * ms:.1f} ms slept per tick on average "
                f"(max {stats.sleep_worst * ms:.1f} ms) — near zero means the loop is saturated"
            )
        return lines

    def _close_window(self, label: str) -> None:
        """报告已结束窗口的摘要，并将其折叠进运行总量。"""
        stats = self._window
        if stats.ticks == 0:
            return
        self._windows_closed += 1
        self._report(f"Cadence ({label}): {self._summary_line(stats)}")
        self._run.fold(stats)
        self._window = _CadenceStats()

    def log_episode_summary(self, label: str | None = None) -> None:
        """标记一个 episode 边界；其一行节拍摘要会发送到报告接收处。

        在每个 episode 边界、紧跟 ``save_episode`` 之后调用。摘要会在
        *下一个* tick 开始时发出——或者在循环先行结束时由
        :meth:`log_run_summary` 发出——因为调用方此时正处于 tick 中间：
        进行中的 tick 是正在结束的 episode 的最后一个 tick，其
        ``save_episode`` 耗时应归属该 episode，而不是即将开始的下一个。
        参见 :meth:`tick`。

        边界一旦落地，该窗口就被折叠进运行总量，并开启一个全新的
        窗口，因此每个 episode 都被单独测量，而
        :meth:`log_run_summary` 仍能覆盖全部。在空窗口上标记边界
        不会报告任何内容，因此在没有录制任何内容的轮换上调用也是安全的。

        参数:
            label: 在日志中如何命名此窗口。默认为迄今
                已关闭窗口的计数；跟踪 episode 的策略应传入
                数据集自身的编号。
        """
        self._pending_close = label or f"episode {self._windows_closed + 1}"

    def log_run_summary(self) -> None:
        """报告整个运行的节拍摘要。在循环的 ``finally`` 中调用一次。

        放在 ``finally`` 中正是关键所在：时长限制、``KeyboardInterrupt``
        和崩溃都仍会产生该摘要。任何尚未处理的待决边界和任何
        仍打开的窗口都会先关闭——当运行存在 episode 边界时单独成行
        报告，当不存在时静默折叠，因为无边界
        循环的单个窗口*就是*整个运行。
        """
        if self._pending_close is not None:
            self._close_window(self._pending_close)
            self._pending_close = None
        if self._window.ticks:
            if self._windows_closed:
                self._close_window("final episode")
            else:
                self._run.fold(self._window)
                self._window = _CadenceStats()
        if self._run.ticks == 0:
            return
        closed = self._windows_closed
        heading = f"whole run, {closed} episode{'s' if closed != 1 else ''}" if closed else "whole run"
        self._report("\n".join(self._summary_lines(self._run, heading)))
