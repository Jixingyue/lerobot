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

import abc
import warnings
from typing import Any

from numpy.typing import NDArray  # type: ignore  # TODO: 为 numpy.typing 添加类型存根

from .configs import CameraConfig


class Camera(abc.ABC):
    """相机实现的基类。

    为不同后端的相机操作定义标准接口。
    子类必须实现所有抽象方法。

    管理基本的相机属性（FPS、分辨率）和核心操作：
    - 连接/断开
    - 帧捕获（同步/异步/最新）

    属性：
        fps (int | None): 配置的每秒帧数
        width (int | None): 帧宽度（像素）
        height (int | None): 帧高度（像素）
    """

    def __init__(self, config: CameraConfig):
        """使用给定配置初始化相机。

        参数：
            config: 包含 FPS 和分辨率的相机配置。
        """
        self.fps: int | None = config.fps
        self.width: int | None = config.width
        self.height: int | None = config.height

    def __enter__(self):
        """
        上下文管理器入口。
        自动连接到相机。
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        上下文管理器出口。
        自动断开连接，确保即使出错也能释放资源。
        """
        self.disconnect()

    def __del__(self) -> None:
        """
        析构安全网。
        如果对象在未清理的情况下被垃圾回收，尝试断开连接。
        """
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """检查相机当前是否已连接。

        返回：
            bool: 如果相机已连接并准备好捕获帧则为 True，
                  否则为 False。
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def find_cameras() -> list[dict[str, Any]]:
        """检测连接到系统的可用相机。
        返回：
            List[Dict[str, Any]]: 一个字典列表，
            每个字典包含一个检测到的相机的信息。
        """
        pass

    @abc.abstractmethod
    def connect(self, warmup: bool = True) -> None:
        """建立到相机的连接。

        参数：
            warmup: 如果为 True（默认），在返回前捕获一个预热帧。
                   适用于需要时间调整捕获设置的相机。
                   如果为 False，跳过预热帧。
        """
        pass

    @abc.abstractmethod
    def read(self) -> NDArray[Any]:
        """以同步方式捕获并返回相机的单帧。

        这是一个阻塞调用，会等待硬件及其 SDK。

        返回：
            np.ndarray: 捕获的帧（numpy 数组）。
        """
        pass

    @abc.abstractmethod
    def async_read(self, timeout_ms: float = ...) -> NDArray[Any]:
        """返回最新的帧。

        此方法获取后台线程捕获的最新帧。
        如果缓冲区中已有新帧（自上次调用以来捕获的），
        则立即返回。

        仅当缓冲区为空或最新帧已被先前的 `async_read` 调用消费时，
        才会阻塞至多 `timeout_ms`。

        本质上，此方法返回最新的未消费帧，如有必要会等待新帧
        在指定超时内到达。

        用法：
            - 非常适合控制循环，当你想确保每个处理的帧
            都是新鲜的，从而有效地将循环与相机的 FPS 同步。
            - 超时的常见原因包括：相机 FPS 过低、处理负载过重，
            或相机已断开连接。

        参数：
            timeout_ms: 等待新帧的最长时间（毫秒）。
                        默认为 200ms（0.2 秒）。

        返回：
            np.ndarray: 捕获的帧（numpy 数组）。

        异常：
            TimeoutError: 如果在 `timeout_ms` 内没有新帧到达。
        """
        pass

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """立即返回最近捕获的帧（窥视模式）。

        此方法是非阻塞的，直接返回当前内存缓冲区中的内容。
        该帧可能已过期，
        即它可能是很久之前捕获的（例如相机挂起的场景）。

        用法：
            适合需要零延迟或频率解耦、且需要保证有帧返回的场景，
            例如 UI 可视化、日志记录或非关键监控。

        返回：
            NDArray[Any]: 帧图像（numpy 数组）。

        异常：
            TimeoutError: 如果最新帧的年龄超过 `max_age_ms`。
            NotConnectedError: 如果相机未连接。
            RuntimeError: 如果相机已连接但尚未捕获任何帧。
        """
        warnings.warn(
            f"{self.__class__.__name__}.read_latest() is not implemented. "
            "Please override read_latest(); it will be required in future releases.",
            FutureWarning,
            stacklevel=2,
        )
        return self.async_read()

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开与相机的连接并释放资源。"""
        pass
