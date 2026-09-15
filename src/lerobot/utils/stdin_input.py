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

"""非阻塞、按行读取 stdin 的工具。

与 :mod:`lerobot.utils.keyboard_input` 不同——后者在 cbreak 模式下逐字节读取
原始按键用于快捷键——:class:`StdinCommandListener` 会拼装出完整的输入行，
并让终端保持在规范模式，因此两者不能共享 stdin。监听器必须是该流的*唯一*
消费者：它直接用 ``os.read`` 读取文件描述符，所以已经被缓冲的字节
（例如早先 ``input()`` 调用留下的）对它不可见。

读取在 SSH、无头环境（没有显示服务器，不像 ``pynput`` 键盘后端）
以及管道输入的 stdin 下都能工作。
"""

from __future__ import annotations

import logging
import os
import select
import sys
from collections.abc import Callable
from threading import Thread
from typing import IO

logger = logging.getLogger(__name__)


class StdinCommandListener:
    """读取输入行并转发给回调的守护线程。

    在 POSIX 上，读取器用 ``select`` 轮询流，使 ``stop()`` 能尽快结束线程；
    在其他平台（或没有文件描述符的类文件对象）上，回退到阻塞式
    ``readline`` 守护线程，随进程一起退出。空行会被跳过；文件结束和
    意外的读取错误会触发 ``on_eof``，因此失效的命令通道不会让消费者一直等待。
    """

    def __init__(
        self,
        on_line: Callable[[str], None],
        on_eof: Callable[[], None] | None = None,
        stream: IO[str] | IO[bytes] | None = None,
        poll_interval_s: float = 0.2,
    ) -> None:
        self._on_line = on_line
        self._on_eof = on_eof
        # sys.stdin 本身可能是 None（pythonw、守护进程化的进程）。
        self._stream = stream if stream is not None else sys.stdin
        self._poll_interval_s = poll_interval_s
        self._running = False
        self._thread: Thread | None = None
        self._use_select = False
        if os.name == "posix":
            try:
                self._stream.fileno()
                self._use_select = True
            except (OSError, ValueError, AttributeError):
                pass

    def start(self) -> None:
        """启动读取线程（幂等）。

        回调在读取线程中触发——除非流缺失（``sys.stdin`` 为 ``None``），
        此时 ``on_eof`` 会在此处同步触发。
        """
        if self._thread is not None:
            return
        if self._stream is None:
            logger.warning("No stdin available for command input — treating as EOF")
            self._emit_eof()
            return
        self._running = True
        self._thread = Thread(target=self._run, daemon=True, name="StdinCommandListener")
        self._thread.start()
        if not self._use_select:
            logger.info("stdin listener running in blocking mode (select unavailable for this stream)")

    def stop(self) -> None:
        """停止读取线程。

        阻塞模式的线程可能卡在 ``readline`` 里而无法 join；
        它们是守护线程，会随进程退出。迟到的输入行无论如何都会被忽略。
        """
        self._running = False
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive() and self._use_select:
            thread.join(timeout=1.0)

    def _run(self) -> None:
        if self._use_select:
            self._run_select()
        else:
            self._run_blocking()

    def _run_select(self) -> None:
        """轮询文件描述符，并从原始字节中切分出完整的行。

        用原始字节而不是 ``stream.readline()``：带缓冲的文件对象可能一次
        吞掉好几行，之后 ``select`` 会报告已读空的 fd 未就绪，
        那些行就永远不会被投递出去了。
        """
        try:
            fd = self._stream.fileno()
        except (OSError, ValueError):  # 在构造和线程启动之间被关闭
            self._emit_read_error()
            return
        buffer = b""
        while self._running:
            try:
                ready, _, _ = select.select([fd], [], [], self._poll_interval_s)
            except (OSError, ValueError):  # 流在我们脚下被关闭
                self._emit_read_error()
                return
            if not ready:
                continue
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                self._emit_read_error()
                return
            if not self._running:
                return
            if chunk == b"":  # EOF：Ctrl-D 或管道输入结束
                # 最后一条没有结尾换行的命令也算数。
                self._emit_line(buffer.decode(errors="replace"))
                self._emit_eof()
                return
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                self._emit_line(raw.decode(errors="replace"))

    def _run_blocking(self) -> None:
        while self._running:
            try:
                line = self._stream.readline()
            except (OSError, ValueError, AttributeError):
                self._emit_read_error()
                return
            if not self._running:
                return
            if not line:  # EOF：文本流上是 ""，字节流上是 b""
                self._emit_eof()
                return
            self._emit_line(line if isinstance(line, str) else line.decode(errors="replace"))

    def _emit_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            self._on_line(line)
        except Exception:  # 绝不让处理器错误杀死读取线程
            logger.exception("Error while handling input line %r", line)

    def _emit_eof(self) -> None:
        logger.info("Input stream closed (EOF)")
        if self._on_eof is not None:
            try:
                self._on_eof()
            except Exception:
                logger.exception("Error while handling input EOF")

    def _emit_read_error(self) -> None:
        """将意外的读取失败按 EOF 处理，让消费者关闭。

        主动调用 ``stop()`` 会先清除 ``_running``，不会走到这里。
        """
        if self._running:
            logger.warning("Input stream failed — treating as EOF")
            self._emit_eof()
