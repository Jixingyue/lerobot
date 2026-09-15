#!/usr/bin/env python

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

"""
ZMQCamera - 通过 ZeroMQ 使用 JSON 协议从远程相机捕获帧，
协议格式如下：
    {
        "timestamps": {"camera_name": float},
        "images": {"camera_name": "<base64-jpeg>"}
    }
"""

import base64
import json
import logging
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.import_utils import _zmq_available, require_package

if TYPE_CHECKING or _zmq_available:
    import zmq
else:
    zmq = None

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from .configuration_zmq import ZMQCameraConfig

logger = logging.getLogger(__name__)


class ZMQCamera(Camera):
    """
    通过 ZeroMQ 管理相机交互，用于从远程服务器接收帧。

    该类连接到 ZMQ Publisher，订阅帧主题，并解码
    传入的包含 Base64 编码图像的 JSON 消息。它支持
    同步和异步两种帧读取方式。

    用法示例：
        ```python
        from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

        config = ZMQCameraConfig(server_address="192.168.123.164", port=5555, camera_name="head_camera")
        camera = ZMQCamera(config)
        camera.connect()

        # 同步读取 1 帧（阻塞）
        color_image = camera.read()

        # 异步读取 1 帧（带超时等待新帧）
        async_image = camera.async_read()

        # 立即获取最新帧（不等待，返回时间戳）
        latest_image, timestamp = camera.read_latest()

        camera.disconnect()
        ```
    """

    def __init__(self, config: ZMQCameraConfig):
        require_package("pyzmq", extra="pyzmq-dep", import_name="zmq")
        super().__init__(config)

        self.config = config
        self.server_address = config.server_address
        self.port = config.port
        self.camera_name = config.camera_name
        self.color_mode = config.color_mode
        self.timeout_ms = config.timeout_ms

        # ZMQ 上下文和套接字
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self._connected = False

        # 线程资源
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        return f"ZMQCamera({self.camera_name}@{self.server_address}:{self.port})"

    @property
    def is_connected(self) -> bool:
        """检查 ZMQ 套接字是否已初始化并连接。"""
        return self._connected and self.context is not None and self.socket is not None

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """连接到 ZMQ 相机服务器。

        参数：
            warmup (bool): 如果为 True，在返回前等待相机提供
                           至少一个有效帧。默认为 True。
        """

        logger.info(f"Connecting to {self}...")

        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.CONFLATE, True)
            self.socket.connect(f"tcp://{self.server_address}:{self.port}")
            self._connected = True

            # 如果未提供分辨率，则自动检测
            if self.width is None or self.height is None:
                # 由于线程尚未运行，直接从硬件读取
                temp_frame = self._read_from_hardware()
                h, w = temp_frame.shape[:2]
                self.height = h
                self.width = w
                logger.info(f"{self} resolution detected: {w}x{h}")

            self._start_read_thread()
            logger.info(f"{self} connected.")

            if warmup:
                # 确保已通过线程至少捕获了一帧
                start_time = time.time()
                while time.time() - start_time < (self.config.warmup_s):  # 等待比超时稍长的时间
                    self.async_read(timeout_ms=self.config.warmup_s * 1000)
                    time.sleep(0.1)

                with self.frame_lock:
                    if self.latest_frame is None:
                        raise ConnectionError(f"{self} failed to capture frames during warmup.")

        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to connect to {self}: {e}") from e

    def _cleanup(self):
        """清理 ZMQ 资源。"""
        self._connected = False
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        ZMQ 相机未实现检测功能。这些相机需要手动配置（服务器地址/端口）。
        """
        raise NotImplementedError("Camera detection is not implemented for ZMQ cameras.")

    def _read_from_hardware(self) -> NDArray[Any]:
        """
        直接从 ZMQ 套接字读取单帧。
        """
        if not self.is_connected or self.socket is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            message = self.socket.recv_string()
        except zmq.Again as e:
            raise TimeoutError(f"{self} timeout after {self.timeout_ms}ms") from e

        # 解码 JSON 消息
        data = json.loads(message)

        if "images" not in data:
            raise RuntimeError(f"{self} invalid message: missing 'images' key")

        images = data["images"]

        # 按相机名称获取图像，或获取第一个可用的图像
        if self.camera_name in images:
            img_b64 = images[self.camera_name]
        elif images:
            img_b64 = next(iter(images.values()))
        else:
            raise RuntimeError(f"{self} no images in message")

        # 解码 base64 JPEG
        img_bytes = base64.b64decode(img_b64)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            raise RuntimeError(f"{self} failed to decode image")

        return frame

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        以同步方式从相机读取单帧。

        这是一个阻塞调用。它等待来自相机后台线程的
        下一个可用帧。

        返回：
            np.ndarray: 解码后的帧 (height, width, 3)
        """
        start_time = time.perf_counter()

        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        frame = self.async_read(timeout_ms=10000)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def _read_loop(self) -> None:
        """
        后台线程运行的内部循环，用于异步读取。
        """
        stop_event = self.stop_event
        if stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized.")

        failure_count = 0
        while not stop_event.is_set():
            try:
                frame = self._read_from_hardware()
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except (TimeoutError, Exception) as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Read error: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, daemon=True, name=f"{self}_read_loop")
        self.thread.start()
        time.sleep(0.1)

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning(f"{self} read thread did not terminate within timeout.")

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        以异步方式读取最新可用的帧。

        参数：
            timeout_ms (float): 等待帧可用的最长时间（毫秒）。
                默认为 200ms。

        返回：
            np.ndarray: 最新捕获的帧。

        异常：
            DeviceNotConnectedError: 如果相机未连接。
            TimeoutError: 如果在指定超时内没有帧数据可用。
            RuntimeError: 如果后台线程未运行。
        """

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"{self} async_read timeout after {timeout_ms}ms")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"{self} no frame available")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 1000) -> NDArray[Any]:
        """立即返回最近捕获的帧（窥视模式）。

        此方法是非阻塞的，直接返回当前内存缓冲区中的内容。
        该帧可能已过期，
        即它可能是很久之前捕获的（例如相机挂起的场景）。

        返回：
            NDArray[Any]: 帧图像（numpy 数组）。

        异常：
            TimeoutError: 如果最新帧的年龄超过 `max_age_ms`。
            DeviceNotConnectedError: 如果相机未连接。
            RuntimeError: 如果相机已连接但尚未捕获任何帧。
        """

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def disconnect(self) -> None:
        """断开与 ZMQ 相机的连接。"""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        self._cleanup()

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
