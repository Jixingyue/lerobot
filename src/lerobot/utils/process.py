#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

import logging
import multiprocessing
import os
import signal
import sys


def ensure_multiprocessing_start_method(start_method: str | None) -> None:
    """设置一次 multiprocessing 启动方法，或验证现有方法与之一致。

    传入 ``None`` 则不改动 Python 进程级的默认设置。当 LeRobot
    被嵌入到由应用自身负责 multiprocessing 配置的场景时，这很有用。
    """
    if start_method is None:
        return

    available_methods = multiprocessing.get_all_start_methods()
    if start_method not in available_methods:
        raise ValueError(
            f"Multiprocessing start method must be one of {available_methods} on this platform, "
            f"got {start_method!r}."
        )

    current_method = multiprocessing.get_start_method(allow_none=True)
    if current_method is None:
        multiprocessing.set_start_method(start_method)
    elif current_method != start_method:
        raise RuntimeError(
            f"Multiprocessing start method is already {current_method!r}; cannot change it to "
            f"{start_method!r}. Set the configured multiprocessing context to null to keep the "
            "application's existing method, or launch LeRobot in a fresh process."
        )


class ProcessSignalHandler:
    """用于挂载优雅关闭信号处理器的工具类。

    该类暴露一个 shutdown_event 属性，当收到关闭信号时会被置位。
    一个计数器记录已捕获的关闭信号数量。收到第二个信号时，
    进程以状态码 1 退出。
    """

    _SUPPORTED_SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT")

    def __init__(self, use_threads: bool, display_pid: bool = False):
        # TODO: 检查是否可以使用 threading 的 Event，因为
        # multiprocessing 的 Event 就是 threading.Event 的克隆。
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Event
        if use_threads:
            from threading import Event
        else:
            from multiprocessing import Event

        self.shutdown_event = Event()
        self._counter: int = 0
        self._display_pid = display_pid

        self._register_handlers()

    @property
    def counter(self) -> int:  # pragma: no cover – 简单的访问器
        """已拦截的关闭信号数量。"""
        return self._counter

    def _register_handlers(self):
        """将内部的 _signal_handler 挂载到一部分 POSIX 信号上。"""

        def _signal_handler(signum, frame):
            pid_str = ""
            if self._display_pid:
                pid_str = f"[PID: {os.getpid()}]"
            logging.info(f"{pid_str} Shutdown signal {signum} received. Cleaning up…")
            self.shutdown_event.set()
            self._counter += 1

            # 收到第二个 Ctrl-C（或任何受支持的信号）时强制退出，
            # 以复现此前的行为，同时给调用方一次优雅关闭的机会。
            # TODO: 之后调查是否还需要此逻辑
            if self._counter > 1:
                logging.info("Force shutdown")
                sys.exit(1)

        for sig_name in self._SUPPORTED_SIGNALS:
            sig = getattr(signal, sig_name, None)
            if sig is None:
                # 该信号在此平台上不可用（例如 Windows 不提供
                # SIGHUP、SIGQUIT……）。跳过。
                continue
            try:
                signal.signal(sig, _signal_handler)
            except (ValueError, OSError):  # pragma: no cover – 不太可能发生，但更安全
                # 信号不受支持，或者我们不在主线程中。
                continue
