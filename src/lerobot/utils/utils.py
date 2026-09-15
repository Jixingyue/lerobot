#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import logging
import os
import platform
import select
import subprocess
import sys
import time
from collections.abc import Iterator
from copy import copy, deepcopy
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from accelerate import Accelerator


def inside_slurm():
    """检查该 python 进程是否是通过 slurm 启动的"""
    # TODO(rcadene)：对于交互模式 `--pty bash` 应返回 False
    return "SLURM_JOB_ID" in os.environ


def init_logging(
    log_file: Path | None = None,
    display_pid: bool = False,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    accelerator: Accelerator | None = None,
):
    """初始化 LeRobot 的日志配置。

    在多 GPU 训练中，只有主进程向控制台记录日志，以避免重复输出。
    非主进程的控制台日志会被抑制，但仍可记录到文件。

    参数:
        log_file: 可选的日志文件写入路径
        display_pid: 在日志消息中包含进程 ID（有助于调试多进程）
        console_level: 控制台输出的日志级别
        file_level: 文件输出的日志级别
        accelerator: 可选的 Accelerator 实例（用于多 GPU 检测）
    """

    class LeRobotFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            record.lerobot_location = f"{record.pathname}:{record.lineno}"[-15:]
            record.lerobot_pid = f"[PID: {os.getpid()}] " if display_pid else ""
            return super().format(record)

    formatter = LeRobotFormatter(
        "%(levelname)s %(lerobot_pid)s%(asctime)s %(lerobot_location)15s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # 清除所有已有的处理器
    logger.handlers.clear()

    # 判断这是否为分布式训练中的非主进程
    is_main_process = accelerator.is_main_process if accelerator is not None else True

    # 控制台日志（仅主进程）
    if is_main_process:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level.upper())
        logger.addHandler(console_handler)
    else:
        # 抑制非主进程的控制台输出
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.ERROR)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level.upper())
        logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)


def format_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


def say(text: str, blocking: bool = False):
    system = platform.system()

    if system == "Darwin":
        cmd = ["say", text]

    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")

    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]

    else:
        raise RuntimeError("Unsupported operating system for text-to-speech.")

    try:
        if blocking:
            subprocess.run(cmd, check=True, timeout=5)
        else:
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if system == "Windows" else 0)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logging.warning("Text-to-speech command failed: %s | Error: %s", cmd, e)


def log_say(text: str, play_sounds: bool = True, blocking: bool = False):
    logging.info(text)

    if play_sounds:
        say(text, blocking)


def get_channel_first_image_shape(image_shape: tuple) -> tuple:
    shape = copy(image_shape)
    if shape[2] < shape[0] and shape[2] < shape[1]:  # (h, w, c) -> (c, h, w)，形状记号保持英文
        shape = (shape[2], shape[0], shape[1])
    elif not (shape[0] < shape[1] and shape[0] < shape[2]):
        raise ValueError(image_shape)

    return shape


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def unwrap_scalar(value: Any) -> Any:
    """将张量 / numpy 标量 / 单元素列表解包为 Python 标量。

    张量和 numpy 标量暴露了 ``.item()``；单元素列表会被
    递归解包。其他任何内容原样返回。将其集中
    在此处，以便语言渲染器和处理器步骤共用同一定义。

    异常:
        ValueError: 如果 ``value`` 是包含零个或多个元素的列表。
    """
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected a scalar, got list of length {len(value)}: {value!r}")
        return unwrap_scalar(value[0])
    return value


def is_valid_numpy_dtype_string(dtype_str: str) -> bool:
    """
    当给定字符串可以转换为 numpy dtype 时返回 True。
    """
    try:
        # 尝试将字符串转换为 numpy dtype
        np.dtype(dtype_str)
        return True
    except TypeError:
        # 如果抛出 TypeError，则该字符串不是合法的 dtype
        return False


def enter_pressed() -> bool:
    if platform.system() == "Windows":
        import msvcrt

        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key in (b"\r", b"\n")  # 回车键
        return False
    else:
        return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline().strip() == ""


def move_cursor_up(lines):
    """将光标向上移动指定的行数。"""
    print(f"\033[{lines}A", end="")


def get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time_s: float):
    days = int(elapsed_time_s // (24 * 3600))
    elapsed_time_s %= 24 * 3600
    hours = int(elapsed_time_s // 3600)
    elapsed_time_s %= 3600
    minutes = int(elapsed_time_s // 60)
    seconds = elapsed_time_s % 60
    return days, hours, minutes, seconds


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """通过用分隔符连接键来展平一个嵌套字典。

    示例：
        >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        >>> print(flatten_dict(dct))
        {'a/b': 1, 'a/c/d': 2, 'e': 3}

    参数:
        d (dict): 要展平的字典。
        parent_key (str): 要预置到该层各键之前的基础键。
        sep (str): 键之间使用的分隔符。

    返回:
        dict: 展平后的字典。
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """将带分隔键的字典还原为嵌套字典。

    示例：
        >>> flat_dct = {"a/b": 1, "a/c/d": 2, "e": 3}
        >>> print(unflatten_dict(flat_dct))
        {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}

    参数:
        d (dict): 带有展平键的字典。
        sep (str): 键中使用的分隔符。

    返回:
        dict: 嵌套字典。
    """
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_inner = outdict
        for part in parts[:-1]:
            if part not in d_inner:
                d_inner[part] = {}
            d_inner = d_inner[part]
        d_inner[parts[-1]] = value
    return outdict


def cycle(iterable: Any) -> Iterator[Any]:
    """创建一个对 dataloader 安全的循环迭代器。

    这等价于 `itertools.cycle`，但可以安全地用于
    具有多个工作进程的 PyTorch DataLoader。
    详情参见 https://github.com/pytorch/pytorch/issues/23900。

    参数:
        iterable: 要循环遍历的可迭代对象。

    生成:
        来自该可迭代对象的条目；耗尽时从头重新开始。
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class SuppressProgressBars:
    """
    用于抑制进度条的上下文管理器。

    示例
    --------
    ```python
    with SuppressProgressBars():
        # 通常会显示进度条的代码
    ```
    """

    def __enter__(self):
        try:
            from datasets.utils.logging import disable_progress_bar

            disable_progress_bar()
        except ImportError:
            logging.getLogger(__name__).debug(
                "SuppressProgressBars is a no-op because 'datasets' is not installed."
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            from datasets.utils.logging import enable_progress_bar

            enable_progress_bar()
        except ImportError:
            pass


class TimerManager:
    """
    用于测量耗时的轻量工具。

    示例
    --------
    ```python
    # 示例 1：使用上下文管理器
    timer = TimerManager("Policy", log=False)
    for _ in range(3):
        with timer:
            time.sleep(0.01)
    print(timer.last, timer.fps_avg, timer.percentile(90))  # 输出：0.01 100.0 0.01
    ```

    ```python
    # 示例 2：使用 start/stop 方法
    timer = TimerManager("Policy", log=False)
    timer.start()
    time.sleep(0.01)
    timer.stop()
    print(timer.last, timer.fps_avg, timer.percentile(90))  # 输出：0.01 100.0 0.01
    ```
    """

    def __init__(
        self,
        label: str = "Elapsed-time",
        log: bool = True,
        logger: logging.Logger | None = None,
    ):
        self.label = label
        self.log = log
        self.logger = logger
        self._start: float | None = None
        self._history: list[float] = []

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer was never started.")
        elapsed = time.perf_counter() - self._start
        self._history.append(elapsed)
        self._start = None
        if self.log:
            if self.logger is not None:
                self.logger.info(f"{self.label}: {elapsed:.6f} s")
            else:
                logging.info(f"{self.label}: {elapsed:.6f} s")
        return elapsed

    def reset(self):
        self._history.clear()

    @property
    def last(self) -> float:
        return self._history[-1] if self._history else 0.0

    @property
    def avg(self) -> float:
        return mean(self._history) if self._history else 0.0

    @property
    def total(self) -> float:
        return sum(self._history)

    @property
    def count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> list[float]:
        return deepcopy(self._history)

    @property
    def fps_last(self) -> float:
        return 0.0 if self.last == 0 else 1.0 / self.last

    @property
    def fps_avg(self) -> float:
        return 0.0 if self.avg == 0 else 1.0 / self.avg

    def percentile(self, p: float) -> float:
        """
        返回所记录时间的第 p 个百分位数。
        """
        if not self._history:
            return 0.0
        return float(np.percentile(self._history, p))

    def fps_percentile(self, p: float) -> float:
        """
        与第 p 个百分位时间相对应的 FPS。
        """
        val = self.percentile(p)
        return 0.0 if val == 0 else 1.0 / val
