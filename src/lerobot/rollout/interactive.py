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

"""交互式 rollout 会话：``lerobot-rollout`` 的聊天式 stdin 命令。

通过 ``--interactive=true`` 启用后，本模块允许操作者从终端驱动 rollout
（``/help`` 列出所有命令），同时硬件和策略保持连接和热状态。它只添加
CLI 前端——stdin 读取、命令解析、终端输出和日志静音。真正的关闭
信号（SIGINT/SIGTERM）通过会话的 :class:`LinkedEvent` 父事件传播，
因此 Ctrl-C 的行为与非交互运行完全一致。
"""

from __future__ import annotations

import contextlib
import logging
import sys
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING

from lerobot.utils.stdin_input import StdinCommandListener
from lerobot.utils.utils import log_say

from .controller import AskResult, RolloutController, RolloutEvent
from .inference import QueryAnswer, QueryKind

if TYPE_CHECKING:
    from .context import RolloutContext
    from .strategies import RolloutStrategy

logger = logging.getLogger(__name__)

_BANNER_RULE = "─" * 60


@contextlib.contextmanager
def _mute_system_output() -> Iterator[None]:
    """在整个进程范围内抑制低于 ERROR 级别的日志记录和 Python 警告。

    常规系统日志会与聊天提示符争抢输出。``logging.disable`` 在
    处理器分发之前就拦截记录，因此不传播的库日志器和会话中途
    创建的日志器也会被覆盖（文件处理器同样如此）；ERROR 及以上
    级别仍能通过，因此故障依然可见。
    """
    previous_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        # 与过滤器快照不同，catch_warnings 还会恢复变更计数器和 showwarning。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.disable(previous_disable)


@dataclass(frozen=True)
class InteractiveCommand:
    """从交互式提示符解析出的 ``/name args`` 行。"""

    name: str
    args: str = ""


def _format_task(task: str) -> str:
    """为操作者渲染任务字符串，显式标注空任务的情况。"""
    return repr(task) if task else "(none — set one with /subtask <text>)"


def _strip_quotes(text: str) -> str:
    """从命令参数中去掉一层成对的外围引号。"""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def parse_command(line: str) -> InteractiveCommand | None:
    """将输入行解析为 :class:`InteractiveCommand`。

    命令是 ``/name``，后面可选跟自由文本参数。对于不是命令的行
    （没有前导 ``/`` 或只有一个裸 ``/``）返回 ``None``。
    """
    line = line.strip()
    if not line.startswith("/"):
        return None
    head, *rest = line.split(maxsplit=1)
    name = head[1:].lower()
    if not name:
        return None
    return InteractiveCommand(name=name, args=rest[0].strip() if rest else "")


class InteractiveSession:
    """通过聊天式 stdin 命令驱动 rollout。

    这是 :class:`RolloutController` 之上的一个轻量终端前端，
    以 :attr:`controller` 的形式暴露给测试和嵌入使用者：
    stdin 监听器将行解析为命令并调用控制器的线程安全方法，
    控制器事件则被渲染回终端输出。

    命令采用最后写入优先：``/reset`` 和 ``/stop`` 会取消待处理的 ``/start``。
    命令流上的 EOF 会停止会话（再没有东西可以指挥机器人了），
    因此管道脚本必须在预期时长内保持 stdin 打开，例如
    ``(printf '/start\\n'; sleep 60; printf '/stop\\n') | lerobot-rollout ... --interactive=true``。
    """

    def __init__(
        self,
        strategy: RolloutStrategy,
        ctx: RolloutContext,
        input_stream: IO[str] | None = None,
    ) -> None:
        self.controller = RolloutController(strategy, ctx, on_event=self._on_event)
        self._runtime = ctx.runtime
        self._play_sounds = ctx.runtime.cfg.play_sounds
        self._listener = StdinCommandListener(self._handle_line, on_eof=self._handle_eof, stream=input_stream)

        # name -> (处理器, 参数提示, 帮助行)；/help 和横幅都从此表渲染。
        self._commands: dict[str, tuple[Callable[[InteractiveCommand], None], str, str]] = {
            "start": (self._cmd_start, "", "start (or restart) the policy control loop"),
            "subtask": (self._cmd_subtask, " <text>", "set the instruction the policy follows"),
            "vqa": (self._cmd_vqa, " <text>", "ask the policy a question about what it sees"),
            "autosteer": (
                self._cmd_autosteer,
                " <goal>|off",
                "let the policy pick its own subtasks toward a high-level goal",
            ),
            "reset": (self._cmd_reset, "", "stop movement, return to initial position, restore the task"),
            "stop": (self._cmd_stop, "", "end the session and shut down"),
            "help": (self._cmd_help, "", "show this help"),
        }

    @contextlib.contextmanager
    def _route_cadence_reports(self) -> Iterator[None]:
        """将控制循环的节奏摘要发送到聊天流，而不是被静音的日志。

        仅在边界处输出，在 serve 线程上打印；作用域与 :func:`_mute_system_output` 相同。
        """
        previous = self._runtime.cadence_report
        self._runtime.cadence_report = self._print
        try:
            yield
        finally:
            self._runtime.cadence_report = previous

    def run(self) -> None:
        """运行会话，直到 ``/stop``、EOF、引擎失败或关闭信号。"""
        try:
            with _mute_system_output(), self._route_cadence_reports():
                self._print(self._render_banner())
                self._listener.start()
                try:
                    self.controller.serve()
                finally:
                    self._listener.stop()
        finally:
            # 在静音上下文之外，因此结束通告和清理日志重新可见。
            log_say("Interactive session ended", self._play_sounds)

    # ------------------------------------------------------------------
    # 控制器事件（在 serve 线程上触发）-> 终端输出
    # ------------------------------------------------------------------

    def _on_event(self, event: RolloutEvent, payload: QueryAnswer | None = None) -> None:
        if event is RolloutEvent.QUERY_ANSWERED and payload is not None:
            self._report_answer(payload)
        elif event is RolloutEvent.SEGMENT_STARTED:
            log_say("Starting rollout", self._play_sounds)
            self._print(
                f"Rollout running — task {_format_task(self.controller.task)}. "
                "/subtask <text> to change it, /reset to return to initial position, /stop to shut down."
            )
        elif event is RolloutEvent.SEGMENT_ENDED:
            self._print(
                "Rollout run ended on its own (duration reached). Robot is holding position — "
                "/start to run again, /reset to return to initial position, /stop to shut down."
            )
        elif event is RolloutEvent.RESET_STARTED:
            log_say("Resetting robot to initial position", self._play_sounds)
            self._print("Resetting — returning the robot to its initial position...")
        elif event is RolloutEvent.RESET_DONE:
            self._print("Robot reset — holding at initial position. /start to run.")
        elif event is RolloutEvent.RESET_SKIPPED:
            self._print("Robot paused — no initial position captured, holding current pose. /start to run.")
        elif event is RolloutEvent.RESET_FAILED:
            self._print(
                "Reset FAILED — the return move errored, so the robot may NOT be at its "
                "initial position. Check the robot before /start."
            )
        elif event is RolloutEvent.ENGINE_FAILED:
            self._report_failure("Inference engine failed — shutting down.")
        elif event is RolloutEvent.STRATEGY_FAILED:
            self._report_failure("Rollout strategy failed (robot or recording error) — shutting down.")

    def _report_answer(self, answer: QueryAnswer) -> None:
        """渲染已解答的文本查询（操作者的问题或 autosteer 回合）。"""
        if answer.kind is QueryKind.NEXT_SUBTASK:
            if answer.ok:
                # 引擎已经通过 set_task 应用了它；只需通告即可。
                self._print(f"Autosteer subtask: {answer.answer!r}")
            else:
                self._print(
                    f"Autosteer stopped — could not plan the next subtask for {answer.question!r}: "
                    f"{answer.error}"
                )
        elif answer.ok:
            self._print(f"Q: {answer.question}\nA: {answer.answer}")
        else:
            self._print(f"Could not answer {answer.question!r} — {answer.error}")

    def _report_failure(self, headline: str) -> None:
        """尽管控制台日志已被静音，仍要将致命的引擎/策略错误呈现出来。"""
        self._print(headline)
        failure_traceback = self.controller.failure_traceback
        if failure_traceback:
            self._print(failure_traceback)
        else:
            self._print("Re-run without --interactive=true to see the error output.")

    # ------------------------------------------------------------------
    # 命令处理器（从监听器线程调用）
    # ------------------------------------------------------------------

    def _handle_line(self, line: str) -> None:
        cmd = parse_command(line)
        if cmd is None:
            self._print("Input not recognized — commands start with '/'. Type /help for the list.")
            return
        entry = self._commands.get(cmd.name)
        if entry is None:
            self._print(f"Unknown command '/{cmd.name}'. Type /help for the list.")
            return
        handler = entry[0]
        handler(cmd)

    def _handle_eof(self) -> None:
        self._print("Input stream closed — stopping the session.")
        self.controller.stop()

    def _cmd_start(self, cmd: InteractiveCommand) -> None:
        if self.controller.start():
            return
        # start() 在停止过程中或失败之后也会拒绝——不要把空闲的机器人误标为运行中。
        if self.controller.running:
            self._print("Already running — /reset to pause first, or /stop to shut down.")
        else:
            self._print("Can't start — the session is stopping or has failed.")

    def _cmd_subtask(self, cmd: InteractiveCommand) -> None:
        # 在判空之前去掉引号，这样 /subtask "" 会报告当前任务，
        # 而不是悄悄应用空指令。
        task = _strip_quotes(cmd.args)
        if not task:
            self._print(f"Current task: {_format_task(self.controller.task)}")
            return
        previous = self.controller.task
        steering = self.controller.autosteer_goal
        if steering is not None:
            self._print(f"Autosteer off (was {steering!r}) — setting the instruction by hand takes over.")
        if self.controller.set_task(task):
            self._print(
                f"Task: {_format_task(previous)} → {_format_task(task)} "
                "(applies from the next policy inference)"
            )
        elif task == self.controller.task:
            self._print(f"Task unchanged: {_format_task(task)}")
        else:
            # set_task 在停止过程中也会拒绝；说"未改变"会暗示它已被应用。
            self._print("Can't change the task — the session is stopping.")

    def _cmd_vqa(self, cmd: InteractiveCommand) -> None:
        # 先去掉引号，这样 /vqa "" 会打印用法提示，而不是将一个空问题入队。
        question = _strip_quotes(cmd.args)
        if not question:
            self._print("Usage: /vqa <question> — e.g. /vqa is the cube inside the box?")
            return
        result = self.controller.ask(question)
        if result is AskResult.QUEUED:
            self._print(f"Asked: {question!r} — answering from the next observation...")
        elif result is AskResult.UNSUPPORTED:
            self._print("This policy has no text head — it cannot answer questions.")
        elif result is AskResult.NOT_RUNNING:
            self._print("Not running — /start first so the policy has a live view to answer from.")
        elif result is AskResult.BUSY:
            # 可能是之前的 /vqa 或 autosteer 查询——通道不会说明是哪一个。
            self._print("The policy is busy with another query — try again in a moment.")
        else:  # 未来的 AskResult 变体不能被误标为 busy
            logger.error("Unhandled AskResult %r for /vqa", result)
            self._print(f"Could not queue the question ({result.value}).")

    def _cmd_autosteer(self, cmd: InteractiveCommand) -> None:
        goal = _strip_quotes(cmd.args)
        if not goal:
            current = self.controller.autosteer_goal
            self._print(
                f"Autosteer on — goal {current!r}." if current else "Autosteer off. Usage: /autosteer <goal>"
            )
            return
        if goal.lower() == "off":
            stopped = self.controller.stop_autosteer()
            self._print(
                f"Autosteer off (was {stopped!r}). The last subtask stays in effect."
                if stopped
                else "Autosteer was not running."
            )
            return
        result = self.controller.autosteer(goal)
        if result is AskResult.UNSUPPORTED:
            self._print("This policy has no text head — it cannot plan subtasks.")
        elif result is AskResult.NOT_RUNNING:
            self._print("Not running — /start first so the policy has a live view to plan from.")
        elif result is AskResult.QUEUED:
            self._print(
                f"Autosteer on — goal {goal!r}. The policy picks its own subtasks; "
                "each one is announced here. Take over with /subtask <text> or /autosteer off."
            )
        else:  # 未来的 AskResult 变体不能被通告为成功
            logger.error("Unhandled AskResult %r for /autosteer", result)
            self._print(f"Could not start autosteer ({result.value}).")

    def _cmd_reset(self, cmd: InteractiveCommand) -> None:
        if self.controller.reset():
            self._print(f"Task restored to {_format_task(self.controller.initial_task)}")
        elif self.controller.stopped:
            self._print("Can't reset — the session has stopped.")

    def _cmd_stop(self, cmd: InteractiveCommand) -> None:
        self.controller.stop()

    def _cmd_help(self, cmd: InteractiveCommand) -> None:
        self._print(self._render_help())

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def _render_help(self) -> str:
        usages = {name: f"/{name}{entry[1]}" for name, entry in self._commands.items()}
        width = max(len(usage) for usage in usages.values())
        lines = [f"  {usages[name]:<{width}}   {entry[2]}" for name, entry in self._commands.items()]
        return "Available commands:\n" + "\n".join(lines)

    def _render_banner(self) -> str:
        return (
            f"{_BANNER_RULE}\n"
            "Interactive rollout session — the robot will NOT move until you type /start.\n"
            f"Task: {_format_task(self.controller.initial_task)}\n"
            f"{self._render_help()}\n"
            "Routine system logs and warnings are muted during the session (errors and the "
            "cadence summary of each run still show).\n"
            f"{_BANNER_RULE}"
        )

    @staticmethod
    def _print(message: str) -> None:
        """面向用户的聊天输出；日志保持在 stderr，回复输出到 stdout。

        每条消息一次 ``write`` 调用（包含换行符）：``print()`` 将消息和换行
        分开写入，可能在监听器线程和 serve 线程之间于行中交错。
        """
        sys.stdout.write(message + "\n")
        sys.stdout.flush()
