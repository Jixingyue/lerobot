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

"""与显示环境无关的键盘输入，用于交互式控制。

本模块集中处理所有与*离散*键盘控制相关的内容
（提前结束 episode、重新录制、停止，以及 rollout 策略的自定义按键）：

* 环境检测——:func:`is_headless`、:func:`is_wayland`、
  :func:`pynput_can_capture`（每个调用点都应使用这一个谓词来
  判断 ``pynput`` 在此处是否真的能捕获按键）；
* 一套共享的按键映射——:func:`apply_recording_control`；以及
* 在同一个 ``(listener, events)`` 约定之后的两个可互换后端：
  ``pynput`` 全局监听器（X11 / 受信任的 macOS / Windows）和一个
  标准库 :class:`TerminalKeyListener`，它读取控制终端 TTY
  （Wayland / 带 TTY 的无头 SSH / 没有辅助功能权限的 macOS）。

注意：*连续*按键状态遥操作（“按住一个键持续移动”）被有意
放在本模块之外处理。cbreak 模式下的终端只会传递按键按下
字节——没有按键释放事件——因此无法重现
按住按键的模型。这类遥操作器仍使用 ``pynput``，并通过
:func:`pynput_can_capture` 发出警告，而不是静默地什么都不做。
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import platform
import select
import sys
import threading
import time
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING

from .import_utils import _pynput_available

logger = logging.getLogger(__name__)

# 仅 POSIX 提供的终端模块（Windows 上不存在，那里使用 pynput 后端）。
if TYPE_CHECKING:
    import termios
    import tty

    _TERMIOS_AVAILABLE = True
else:
    try:
        import termios
        import tty

        _TERMIOS_AVAILABLE = True
    except ImportError:  # 仅 POSIX 提供的模块；Windows 上不可用
        termios = tty = None
        _TERMIOS_AVAILABLE = False

keyboard = None
if _pynput_available:
    try:
        from pynput import keyboard
    except Exception as e:  # 例如无头 Linux 机器上没有可连接的 X 显示
        logger.info("Could not import pynput keyboard backend: %s", e)


@cache
def is_headless() -> bool:
    """当没有可用的显示服务器时返回 ``True``。

    * Linux：当 ``DISPLAY``（X11）和 ``WAYLAND_DISPLAY`` 都未设置时为无头环境。
    * macOS / Windows：始终假定存在显示。一个真正没有 GUI 的
      Mac/Windows CI 主机可能会被误分类，但这无关紧要，因为
      sys.stdin.isatty() 门控在那里无论如何都会返回 None。
    """
    if platform.system() == "Linux":
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False


@cache
def is_wayland() -> bool:
    """当运行在 Wayland 会话下时返回 ``True``。

    ``pynput`` 依赖 X11 后端。在 Wayland 下它仍能导入（XWayland
    通常存在且 ``$DISPLAY`` 已设置），但无法捕获*全局*
    热键，因此文档中所述的方向键/Esc 快捷键会静默失效。:func:`is_headless`
    无法识别这种情况，因此需要专门的检查。
    """
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" or bool(
        os.environ.get("WAYLAND_DISPLAY")
    )


@cache
def pynput_can_capture() -> bool:
    """当 ``pynput`` 全局监听器确实能捕获按键时返回 ``True``。

    这是每个键盘调用点在 ``pynput`` 后端与回退方案之间做选择时
    都应使用的唯一谓词。它被有意设计得
    保守：

    * Linux：只有真正的 X11 会话（存在显示*且*不是 Wayland）才为真。
    * macOS：此处返回 ``True``——辅助功能 / 输入监控权限
      （``IS_TRUSTED``）只能在启动监听器*之后*于运行时确认，
      因此 :func:`init_keyboard_listener` 会用
      :func:`pynput_listener_is_trusted` 进一步细化。
    * Windows：``True``（底层全局钩子不需要特殊权限）。

    当未安装 ``pynput`` 时始终为 ``False``。
    """
    if not _pynput_available:
        return False
    if platform.system() == "Linux":
        return not is_headless() and not is_wayland()
    return True


def pynput_listener_is_trusted(listener, timeout_s: float = 1.0) -> bool:
    """尽力检查一个刚启动的 ``pynput`` 监听器是否能够捕获按键。

    在 macOS 上，``pynput`` 会在 Quartz 事件点击创建后，在其*监听器线程*上
    设置 ``listener.IS_TRUSTED``；类默认值为 ``False``。因此
    我们等待该线程把它翻转为 ``True``（受信任），或等待一个
    短暂的超时过去（不受信任——它会永远保持 ``False``）。这偏向于
    常见的受信任情形（标志一翻转就立即返回），只有在已经不可用的
    不受信任机器上才会付出完整的 ``timeout_s`` 等待。

    在非 macOS 后端上不存在该属性，并假定捕获可以正常工作。
    """
    if platform.system() != "Darwin":
        return True
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if getattr(listener, "IS_TRUSTED", False):
            return True
        time.sleep(0.02)
    return bool(getattr(listener, "IS_TRUSTED", False))


def apply_recording_control(control: str, events: dict) -> None:
    """将一次录制控制流按键应用到共享的 ``events`` 字典上。

    集中管理该映射，使 ``pynput`` 和终端后端表现
    一致。``control`` 是 ``"right"``（提前结束循环）、``"left"``
    （重新录制上一个 episode）或 ``"esc"``（停止录制）之一。
    """
    if control == "right":
        print("Right arrow key pressed. Exiting loop...")
        events["exit_early"] = True
    elif control == "left":
        print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
        events["rerecord_episode"] = True
        events["exit_early"] = True
    elif control == "esc":
        print("Escape key pressed. Stopping data recording...")
        events["stop_recording"] = True
        events["exit_early"] = True


# 终端方向键以 3 字节转义序列的形式到达，其*最后一个*字节标识
# 方向。根据终端的光标键模式存在两种编码——CSI
# （"ESC [ X"）和 SS3（"ESC O X"，在 SSH/tmux 上常见）——但两者的
# 最后一个字节相同，因此这一张表即可解码两者。由
# TerminalKeyListener._parse 查表；遇到未知的最后一个字节时返回
# None（忽略该序列）。
_ARROW_FINAL_BYTES = {"A": "up", "B": "down", "C": "right", "D": "left"}


class TerminalKeyListener:
    """与显示环境无关的键盘监听器，从控制终端 TTY 读取按键。

    在 *离散*控制方面用作 ``pynput``
    监听器在 Wayland / 无头 / macOS 不受信任情形下的等价替代。它将终端置于
    关闭回显的 cbreak 模式，并在一个守护线程上读取字节，将其解码为
    传递给 ``on_key`` 的逻辑按键名：

    * 方向键（``ESC [ C`` / ``ESC O C``……）-> ``"right"`` / ``"left"`` / ``"up"`` / ``"down"``
    * 单独的 ``ESC`` -> ``"esc"``
    * 回车 / Tab / 空格 / 退格 -> ``"enter"`` / ``"tab"`` / ``"space"`` / ``"backspace"``
    * 任何其他可打印字节 -> 对应字符（例如 ``"n"``、``"s"``）

    它只产生按键按下事件（终端没有按键释放），因此
    适用于离散命令，但不适用于连续的“按住移动”遥操作。

    终端会在 :meth:`stop` 时恢复，也会通过 ``atexit`` 钩子恢复，因此
    崩溃或 Ctrl-C 绝不会让 shell 停留在无回显的 cbreak 状态。仅支持 POSIX
    （``termios`` / ``tty`` / ``select``）；这些模块采用懒导入，因此本
    文件在 Windows 上仍可导入（那里改用 ``pynput``）。
    """

    def __init__(self, on_key: Callable[[str], None]):
        self._on_key = on_key
        self._running = False
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._old_attrs = None

    def _read_char(self, timeout: float) -> str | None:
        """在 ``timeout`` 秒内从 stdin 返回一个字符，超时则返回 ``None``。"""
        if self._fd is None:
            return None
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return None
        try:
            data = os.read(self._fd, 1)
        except OSError:
            return None
        if not data:
            return None
        return data.decode(errors="ignore")

    def _parse(self, ch: str) -> str | None:
        """将从 ``ch`` 开始的一个（可能是多字节的）按键解码为按键名。"""
        if ch == "\x1b":
            # 可能是 CSI / SS3 转义序列（方向键），也可能是单独的 ESC。
            # 使用短暂的后续读取，以免把孤立的 ESC 误认为序列。
            ch2 = self._read_char(timeout=0.02)
            if ch2 is None:
                return "esc"
            if ch2 in ("[", "O"):
                ch3 = self._read_char(timeout=0.02)
                return _ARROW_FINAL_BYTES.get(ch3 or "")
            # 其他某种转义序列（例如 Alt+键）；忽略它。
            return None
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\t":
            return "tab"
        if ch == " ":
            return "space"
        if ch in ("\x7f", "\x08"):
            return "backspace"
        if ch.isprintable():
            return ch
        return None

    def _run(self) -> None:
        while self._running:
            ch = self._read_char(timeout=0.05)
            if ch is None:
                continue
            name = self._parse(ch)
            if name is None:
                continue
            try:
                self._on_key(name)
            except Exception as e:  # 绝不让处理器的错误杀死读取线程
                logger.debug("Terminal key handler error: %s", e)

    def start(self) -> None:
        """将终端切换到 cbreak 模式（关闭回显），并在守护线程上读取按键。

        当 stdin 不是 TTY（管道/重定向输入）或所在平台没有
        ``termios``（例如 Windows）时为空操作，因此非交互式运行不受影响。
        """
        if not sys.stdin.isatty():
            return
        if not _TERMIOS_AVAILABLE:  # 仅 POSIX 提供的模块（例如 Windows 上不可用）
            logger.warning("Terminal keyboard input is not supported on this platform.")
            return

        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        # 显式禁用 ECHO，以免方向键转义序列（例如 ^[[C）作为乱码
        # 回显到录制终端中。（这与 tty.setcbreak
        # 随版本而异的行为无关。）
        new_attrs = termios.tcgetattr(self._fd)
        new_attrs[3] &= ~termios.ECHO  # 索引 3 == lflags
        termios.tcsetattr(self._fd, termios.TCSADRAIN, new_attrs)
        # 安全网：即使永远走不到 stop()（崩溃），也要恢复终端。
        atexit.register(self.stop)

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止读取线程并恢复终端的原始属性。

        幂等：可安全地多次调用（例如显式调用与通过 atexit 调用）。
        """
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
            self._thread = None
        if self._fd is not None and self._old_attrs is not None and _TERMIOS_AVAILABLE:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            finally:
                self._old_attrs = None
        with contextlib.suppress(Exception):
            atexit.unregister(self.stop)


# 将 pynput 按键对象映射为与 TerminalKeyListener 发出的相同的规范名称，
# 以便同一个 dispatch 在两个后端上都能工作。pynput 不可用时为空。
if keyboard is not None:
    _PYNPUT_KEY_NAMES = {
        keyboard.Key.right: "right",
        keyboard.Key.left: "left",
        keyboard.Key.up: "up",
        keyboard.Key.down: "down",
        keyboard.Key.esc: "esc",
        keyboard.Key.enter: "enter",
        keyboard.Key.tab: "tab",
        keyboard.Key.space: "space",
        keyboard.Key.backspace: "backspace",
    }
else:
    _PYNPUT_KEY_NAMES = {}


def _resolve_pynput_key(key) -> str | None:
    """将一个 pynput 按键事件解析为 TerminalKeyListener 同样会发出的规范名称。

    特殊按键通过 :data:`_PYNPUT_KEY_NAMES` 映射；字符按键回退到其
    ``.char``（例如 ``"n"``）。对于既无映射又无字符的按键返回 ``None``。
    """
    name = _PYNPUT_KEY_NAMES.get(key)
    if name is not None:
        return name
    # ``or None`` 保留了历史上“字符为真”的语义：空字符/None 字符表示“没有按键”。
    return getattr(key, "char", None) or None


def create_key_listener(dispatch: Callable[[str], None], *, controls_help: str = ""):
    """启动一个键盘监听器，将解析出的按键名路由给 ``dispatch``。

    这是录制和 rollout 策略共用的后端选择逻辑：

    * 在 X11 / 受信任的 macOS / Windows 上使用 ``pynput`` 全局监听器
      （在 macOS 上，启动后会检查监听器的 ``IS_TRUSTED`` 标志，不受信任的监听器会被
      停止，从而改用终端后端）；
    * 在带 TTY 的 Wayland / 无头会话上使用标准库
      :class:`TerminalKeyListener`；
    * 当没有可用后端时返回 ``None``（非交互式 / 管道运行）。

    两个后端都会向 ``dispatch`` 传递相同的规范按键名
    （"right" / "left" / "up" /
    "down" / "esc" / "enter" / "tab" / "space" / "backspace"，或单个字符），因此无论
    使用哪个后端，同一个 ``dispatch`` 都能工作。``controls_help`` 是一个可选提示，
    会追加到日志消息之后。

    返回监听器（暴露 ``.stop()``）或 ``None``。
    """
    suffix = f" ({controls_help})" if controls_help else ""

    if pynput_can_capture() and keyboard is not None:

        def on_press(key):
            with contextlib.suppress(Exception):
                name = _resolve_pynput_key(key)
                if name is not None:
                    dispatch(name)

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        if pynput_listener_is_trusted(listener):
            logger.info("Keyboard listener started%s.", suffix)
            return listener
        # macOS 缺少辅助功能 / 输入监控权限时：监听器永远不会
        # 触发。停止它并继续回退到终端后端。
        logger.warning(
            "pynput keyboard listener is not trusted (missing macOS Accessibility / "
            "Input Monitoring permission); falling back to terminal keyboard input."
        )
        listener.stop()

    if sys.stdin.isatty():
        listener = TerminalKeyListener(dispatch)
        listener.start()
        logger.info("Using terminal keyboard input — keep this terminal focused%s.", suffix)
        return listener

    logger.warning(
        "Keyboard controls unavailable: no usable display (Wayland/headless) and stdin is "
        "not an interactive terminal%s.",
        suffix,
    )
    return None


def init_keyboard_listener():
    """为交互式录制控制初始化一个非阻塞键盘监听器。

    后端选择：

    * 当 :func:`pynput_can_capture` 为真时使用 ``pynput`` 全局监听器
      （真正的 X11、macOS、Windows）。在 macOS 上，启动后会检查
      监听器的 ``IS_TRUSTED`` 标志；如果进程缺少辅助功能 /
      输入监控权限，监听器会被停止并改用终端后端。
    * 当 ``pynput`` 无法捕获（Wayland / 无头 SSH / macOS 不受信任）
      *且* stdin 是 TTY 时，使用读取控制终端 TTY 的
      :class:`TerminalKeyListener`。
    * 否则不使用监听器（非交互式 / 管道运行）——录制依赖
      episode/重置计时器（或 Ctrl+C）。

    两个后端接受相同的控制：右/左/Esc，以及对应的单字节字母
    等价键 ``n``（下一个）、``r``（重新录制）和 ``q``（退出）。在高延迟的
    SSH/VNC 链路上，字母是最可靠的选择，因为方向键转义序列可能
    被终端拆分、延迟或拦截。

    返回:
        一个元组 ``(listener, events)``，其中 ``listener`` 暴露 ``.stop()`` 或为
        ``None``，``events`` 是由按键设置的标志字典
        （``exit_early``、
        ``rerecord_episode``、``stop_recording``）。
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    # 在方向键/Esc 之外接受单字节字母等价键 n/r/q：在卡顿的
    # SSH/VNC 链路上，字母不受影响方向键的转义序列拆分/延迟/拦截的影响。
    # 不区分大小写，因此 Shift+字母仍然有效。
    def on_key(name: str) -> None:
        key = name.lower()
        if key in ("right", "n"):
            apply_recording_control("right", events)
        elif key in ("left", "r"):
            apply_recording_control("left", events)
        elif key in ("esc", "q"):
            apply_recording_control("esc", events)
        # 其他按键（包括上/下）被有意忽略

    listener = create_key_listener(on_key, controls_help="Right/Left/Esc, or n=next, r=re-record, q=quit")
    return listener, events
