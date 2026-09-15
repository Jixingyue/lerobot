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

"""推理引擎抽象基类。

rollout 策略通过这个小型接口消费动作，因此它们
不需要知道推理是在控制线程上内联执行，
还是在后台线程中异步执行（RTC）。
"""

from __future__ import annotations

import abc
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Lock

import torch

from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT

logger = logging.getLogger(__name__)


class QueryKind(Enum):
    """向策略文本头请求的内容类型。"""

    VQA = "vqa"
    """关于当前场景的自由形式问题；回复会交给操作者。"""

    NEXT_SUBTASK = "next_subtask"
    """一个高层目标；回复是下一个子任务，会被送入 ``set_task``。"""


@dataclass(frozen=True)
class PolicyQuery:
    """已入队的、针对策略文本头的请求。"""

    kind: QueryKind
    text: str


@dataclass(frozen=True)
class QueryAnswer:
    """策略文本查询的结果。

    ``answer`` / ``error`` 恰好有一个被设置。``NEXT_SUBTASK`` 成功时携带的
    子任务是引擎*已经应用*的，因此接收方只需通告即可。
    """

    question: str
    answer: str | None = None
    error: str | None = None
    kind: QueryKind = QueryKind.VQA

    @property
    def ok(self) -> bool:
        """策略生成了答案时为 True。"""
        return self.error is None


class InferenceEngine(abc.ABC):
    """rollout 期间生成动作的抽象后端。

    子类决定推理是在控制线程上内联执行，
    还是在后台线程中异步执行。契约保持最小化，
    以便在不触碰 rollout 策略的情况下插入额外的后端。

    生命周期
    ---------
    ``start`` — 准备后端（例如启动后台线程）。
    ``stop`` — 干净地关闭后端。
    ``reset`` — 清除回合级状态（策略隐藏状态、队列等）。

    动作生成
    -----------------
    ``get_action(obs_frame)`` — 返回下一个动作张量；
    若无可用动作则返回 ``None``（例如异步队列为空）。同步
    后端总是从 ``obs_frame`` 计算；异步后端忽略它
    （它们通过 ``notify_observation`` 接收观测）。

    任务
    ----
    ``set_task`` 可从任何线程调用；子类在自己的推理线程上
    通过 :meth:`_take_task` 取走该值，因此策略状态绝不会跨线程被修改。

    文本查询
    ------------
    ``ask`` 可从任何线程调用，且绝不触碰策略：查询由拥有策略的
    线程上的 :meth:`_service_query` 处理（参见
    :attr:`control_thread_owns_policy`），答案只在控制线程上
    通过 :meth:`pump_query` 送达观察者。

    可选钩子
    --------------
    ``notify_observation`` / ``pause`` / ``resume`` 默认是空操作，
    因此 rollout 策略可以无条件地调用它们。

    子类必须调用 ``super().__init__(task=...)``。
    """

    def __init__(self, task: str = "") -> None:
        self._task = task
        self._task_changed = False
        self._dispatched_task = task
        self._task_lock = Lock()

        # 文本查询通道。使用独立的锁，且绝不在文本生成期间持有。
        self._query_lock = Lock()
        self._pending_query: PolicyQuery | None = None
        # 从认领（``_take_query``）开始，直到答案发布或该回合被丢弃为止都保持置位，
        # 这样 autosteer 轮询在此期间不会将重复的回合入队。
        self._query_in_flight = False
        # 等待送达的答案；使用队列以保证未送达的答案永远不会被覆盖。
        self._ready_answers: deque[QueryAnswer] = deque()
        self._answer_observer: Callable[[QueryAnswer], None] | None = None

        # Autosteer 排序器状态（同一把锁：它会写 _pending_query）。
        self._autosteer_goal: str | None = None
        self._autosteer_interval_s: float = 0.0
        self._autosteer_due_at: float = 0.0

    # ------------------------------------------------------------------
    # 任务（语言指令）
    # ------------------------------------------------------------------

    @property
    def task(self) -> str:
        """当前用于条件化推理的语言指令。"""
        with self._task_lock:
            return self._task

    def set_task(self, task: str) -> bool:
        """设置从下一次推理开始使用的指令。

        可从任何线程调用。当值确实发生变化时返回 ``True``。
        """
        with self._task_lock:
            if task == self._task:
                return False
            previous, self._task = self._task, task
            self._task_changed = True
        logger.info("Task changed: '%s' -> '%s'", previous, task)
        return True

    def _take_task(self) -> tuple[str, bool]:
        """读取任务并消费"已变更"边沿。从推理线程调用。"""
        with self._task_lock:
            changed, self._task_changed = self._task_changed, False
            return self._task, changed

    @property
    def dispatched_task(self) -> str:
        """生成最近一次返回动作所用的指令。

        当先前指令产生的动作仍在被消费时，它会滞后于 :attr:`task`
        （*请求的*指令）；记录型策略用它给帧打标签。
        只在 ``get_action`` 之后的控制线程上有意义；重置之后
        它保存的是请求的任务。
        """
        with self._task_lock:
            return self._dispatched_task

    def _set_dispatched_task(self, task: str) -> None:
        """记录 ``get_action`` 调用所返回动作对应的任务。"""
        with self._task_lock:
            self._dispatched_task = task

    def _discard_task_change(self) -> None:
        """丢弃待处理的任务变更边沿，例如来自 ``reset``（状态已被清除）。

        同时重新初始化 ``dispatched_task``：队列中的动作清空后，
        下一个分发的动作只能来自当前指令。
        """
        with self._task_lock:
            self._task_changed = False
            self._dispatched_task = self._task

    # ------------------------------------------------------------------
    # 文本查询（VQA）
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """此后端的策略能够提供文本查询服务时为 True。

        默认为 False，这样没有文本通道的后端永远不会接受
        它无法处理的查询；调用者在入队前检查此属性。
        """
        return False

    def set_answer_observer(self, observer: Callable[[QueryAnswer], None] | None) -> None:
        """注册 :meth:`pump_query` 用于交付就绪答案的回调。"""
        with self._query_lock:
            self._answer_observer = observer

    @property
    def has_pending_query(self) -> bool:
        """查询已入队但尚未处理时为 True。"""
        with self._query_lock:
            return self._pending_query is not None

    @property
    def autosteer_goal(self) -> str | None:
        """当前驱动任务的高层目标（如果有）。"""
        with self._query_lock:
            return self._autosteer_goal

    def ask(self, question: str) -> bool:
        """将一个自由形式的 ``question`` 入队，交给策略文本头。

        可从任何线程调用。当已有查询待处理时返回 ``False``：
        该通道一次只保存一个查询。
        """
        return self._queue_query(PolicyQuery(kind=QueryKind.VQA, text=question))

    def start_autosteer(self, goal: str, interval_s: float) -> None:
        """从 ``goal`` 驱动任务，每 ``interval_s`` 秒重新规划一次。

        可从任何线程调用。每个回合向策略询问下一个子任务，
        并通过 :meth:`set_task` 应用它；每次查询都会重新发送同一个目标，
        因此规划进度保存在策略中。间隔从子任务被*应用*的时刻
        开始计算，因此缓慢的生成不会让机器人没有动作可执行。
        """
        with self._query_lock:
            self._autosteer_goal = goal
            self._autosteer_interval_s = max(0.0, interval_s)
            # 立即到期：在下一个控制 tick 上请求第一个子任务。
            self._autosteer_due_at = time.perf_counter()
        logger.info("Autosteer started for goal '%s' (every %.1fs)", goal, interval_s)

    def stop_autosteer(self) -> str | None:
        """停止排序器，并返回其正在驱动的目标（若无则返回 ``None``）。"""
        with self._query_lock:
            goal, self._autosteer_goal = self._autosteer_goal, None
        if goal is not None:
            logger.info("Autosteer stopped (goal was '%s')", goal)
        return goal

    def drop_pending_query(self) -> PolicyQuery | None:
        """丢弃未处理的查询，并将其返回（若无则返回 ``None``）。

        在运行片段结束时调用，避免该查询在机器人下次启动时
        针对一个完全不同的场景被处理。
        """
        with self._query_lock:
            dropped, self._pending_query = self._pending_query, None
        return dropped

    @property
    @abc.abstractmethod
    def control_thread_owns_policy(self) -> bool:
        """控制线程是否是唯一允许触碰策略的线程。

        True（内联后端）：:meth:`pump_query` 自行处理待处理的查询。False
        （异步后端）：其推理线程必须调用 :meth:`_service_query`，
        而 :meth:`pump_query` 只推进排序器并交付已完成的答案。
        """

    def pump_query(self, obs_processed: dict | None = None) -> bool:
        """将文本查询通道推进一个 tick。仅限控制线程。

        轮询 autosteer 排序器，当 :attr:`control_thread_owns_policy` 时
        处理待处理的查询（异步后端在自己的线程上作答），
        然后送达就绪的答案，使观察者总是在此线程上触发。
        在 tick 结束时调用，而不是从 :meth:`get_action` 调用：
        一次文本生成远长于一个控制 tick。查询被内联处理时返回 ``True``。
        当 ``obs_processed=None``（控制器的空闲轮询）时，
        待处理的查询保持入队状态，排序器也不推进。
        """
        self._poll_autosteer(obs_processed)
        served = False
        if self.control_thread_owns_policy:
            served = self._service_query(obs_processed)
        self._deliver_answer()
        return served

    def _queue_query(self, query: PolicyQuery) -> bool:
        with self._query_lock:
            if self._pending_query is not None:
                return False
            self._pending_query = query
        return True

    def _poll_autosteer(self, obs_processed: dict | None) -> None:
        """若排序器已到期，则将下一个子任务查询入队（仅控制循环）。"""
        if obs_processed is None:
            return
        with self._query_lock:
            if self._autosteer_goal is None:
                return
            if time.perf_counter() < self._autosteer_due_at:
                return
            if self._pending_query is not None or self._query_in_flight:
                # 一个 /vqa（或我们自己的先前查询）仍在队列中或正在生成。
                # 截止时间保持在过去，因此下一个 tick 会重试这个回合。
                return
            self._pending_query = PolicyQuery(kind=QueryKind.NEXT_SUBTASK, text=self._autosteer_goal)

    def _take_query(self) -> PolicyQuery | None:
        """认领待处理的查询。从拥有策略的线程调用。"""
        with self._query_lock:
            query, self._pending_query = self._pending_query, None
            if query is not None:
                self._query_in_flight = True
            return query

    def _service_query(self, obs_processed: dict | None) -> bool:
        """处理待处理的查询。只能从拥有策略的线程调用。

        失败会转化为错误答案而不是异常，因此一个坏查询永远不会
        拖垮调用线程。当查询被认领并处理完成时返回 ``True``。
        """
        if obs_processed is None:
            return False
        query = self._take_query()
        if query is None:
            return False
        try:
            text = self._generate_text(obs_processed, query)
            if not isinstance(text, str) or not text.strip():
                # 在这里失败，使垃圾输出变成错误答案，
                # 而不是去操纵机器人并给记录的帧打标签。
                raise TypeError(
                    f"generate_text() must return a non-empty str, got {text!r} ({type(text).__name__})"
                )
        except Exception as e:
            logger.exception("Policy text query failed (%s) for %r", query.kind.value, query.text)
            if query.kind is QueryKind.NEXT_SUBTASK and not self._fail_subtask(query):
                return True  # 该回合所属的排序器已不存在；丢弃
            self._publish_answer(
                QueryAnswer(question=query.text, error=f"{type(e).__name__}: {e}", kind=query.kind)
            )
            return True
        if query.kind is QueryKind.NEXT_SUBTASK and not self._apply_subtask(query, text):
            return True  # 排序器在此期间已停止；该回合被丢弃
        # 在应用之后才发布，因此负责通告的观察者永远不会
        # 跑到它所描述的任务前面。
        self._publish_answer(QueryAnswer(question=query.text, answer=text, kind=query.kind))
        return True

    def _fail_subtask(self, query: PolicyQuery) -> bool:
        """在回合失败后停止排序器——除非它在此期间已停止或重新设定目标。

        无法获得下一个子任务的排序器必须停止，而不是每个间隔
        都失败一次——但前提是它仍是请求这个回合的那个排序器。
        当应该发布失败答案时返回 ``True``。
        """
        with self._query_lock:
            live = self._autosteer_goal == query.text
            if live:
                self._autosteer_goal = None
            else:
                self._query_in_flight = False  # 不会发布任何答案
        if live:
            logger.info("Autosteer stopped (goal was '%s') — planning failed", query.text)
        else:
            logger.info(
                "Discarding failed autosteer turn for %r — the sequencer stopped or was "
                "retargeted while it was being generated",
                query.text,
            )
        return live

    def _apply_subtask(self, query: PolicyQuery, subtask: str) -> bool:
        """应用生成的子任务，除非排序器在此期间已停止。

        生成过程无锁运行了数秒，因此检查和应用必须在
        ``_query_lock`` 下原子地进行（``_task_lock`` 嵌套在其内部，
        绝不能反过来），否则过期的规划可能覆盖更新的指令。
        应用成功时返回 ``True``。
        """
        with self._query_lock:
            live = self._autosteer_goal == query.text
            if live:
                self.set_task(subtask)
                # 只在此时才启动计时，使间隔度量的是子任务之间的运动时间。
                self._autosteer_due_at = time.perf_counter() + self._autosteer_interval_s
            else:
                self._query_in_flight = False  # 不会发布任何答案
        if not live:
            logger.info(
                "Discarding autosteer subtask %r — the sequencer stopped while it was being generated",
                subtask,
            )
        return live

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """在 ``obs_processed`` 上运行策略的文本头。与后端相关。

        实现方构建批次，用 :meth:`_mark_query` 打上标记，进行预处理，
        然后调用 ``policy.generate_text``。
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support text queries — no /vqa or /autosteer on this backend."
        )

    @staticmethod
    def _mark_query(batch: dict, query: PolicyQuery) -> dict:
        """用查询的类型和文本给 ``batch`` 打标记，供预处理器使用。

        在 ``prepare_observation_for_inference`` 和预处理器流水线之间调用。
        ``QUERY_KIND`` / ``QUERY_TEXT`` 是白名单中的补充数据，因此它们
        会落在 ``task`` 旁边：策略专属的 ``ComplementaryDataProcessorStep``
        可以在那里读取类型，并将 ``QUERY_TEXT`` 改写为其提示词格式。
        """
        batch[QUERY_KIND] = query.kind.value
        batch[QUERY_TEXT] = query.text
        return batch

    def drop_ready_subtask_answers(self) -> None:
        """丢弃未送达的 ``NEXT_SUBTASK`` 答案。

        在片段结束时、停止排序器之后立即调用，这样后续的
        通告就不会描述一个已不再驱动任何内容的排序器。
        VQA 答案仍可送达。
        """
        with self._query_lock:
            kept = [a for a in self._ready_answers if a.kind is not QueryKind.NEXT_SUBTASK]
            dropped = len(self._ready_answers) - len(kept)
            self._ready_answers = deque(kept)
        if dropped:
            logger.debug("Dropped %d undelivered autosteer answer(s) at segment end", dropped)

    def _publish_answer(self, answer: QueryAnswer) -> None:
        with self._query_lock:
            self._query_in_flight = False
            self._ready_answers.append(answer)

    def _deliver_answer(self) -> None:
        with self._query_lock:
            answers = list(self._ready_answers)
            self._ready_answers.clear()
            observer = self._answer_observer
        if observer is None:
            return
        for answer in answers:
            try:
                observer(answer)
            except Exception:  # 损坏的观察者不能杀死控制循环
                logger.exception("Error in inference-engine answer observer")

    @abc.abstractmethod
    def start(self) -> None:
        """初始化后端。"""

    @abc.abstractmethod
    def stop(self) -> None:
        """关闭后端。"""

    @abc.abstractmethod
    def reset(self) -> None:
        """清除回合级状态。"""

    @abc.abstractmethod
    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """返回下一个动作张量；若不可用则返回 ``None``。"""

    def notify_observation(self, obs: dict) -> None:  # noqa: B027
        """发布最新的已处理观测。默认：空操作。"""

    def pause(self) -> None:  # noqa: B027
        """暂停后台推理。默认：空操作。"""

    def resume(self) -> None:  # noqa: B027
        """恢复后台推理。默认：空操作。"""

    @property
    def ready(self) -> bool:
        """后端可以生成动作（例如预热完成）时为 True。"""
        return True

    @property
    def failed(self) -> bool:
        """后端发生不可恢复的错误时为 True。"""
        return False

    @property
    def failure_traceback(self) -> str | None:
        """当 ``failed`` 为 True 时，不可恢复错误的格式化回溯信息。"""
        return None
