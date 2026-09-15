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

"""
提供 Reachy2Camera 类，用于使用 Reachy 2 的 CameraManager 从 Reachy 2 相机捕获帧。
"""

from __future__ import annotations

import logging
import os
import platform
import time
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray  # type: ignore  # TODO: 为 numpy.typing 添加类型存根

# 在导入 cv2 之前，修复 Windows 上 MSMF 硬件变换的兼容性问题
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2  # type: ignore  # TODO: 为 OpenCV 添加类型存根
import numpy as np  # type: ignore  # TODO: 为 numpy 添加类型存根

from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import _reachy2_sdk_available, require_package

if TYPE_CHECKING or _reachy2_sdk_available:
    from reachy2_sdk.media.camera import CameraView
    from reachy2_sdk.media.camera_manager import CameraManager
else:
    CameraManager = None

    class CameraView:
        LEFT = 0
        RIGHT = 1


from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from .configuration_reachy2_camera import ColorMode, Reachy2CameraConfig

logger = logging.getLogger(__name__)


class Reachy2Camera(Camera):
    """
    使用 Reachy 2 CameraManager 管理 Reachy 2 相机。

    该类提供了一个高级接口，用于连接、配置和读取
    Reachy 2 相机的帧。它支持同步和异步两种
    帧读取方式。

    Reachy2Camera 实例需要在配置中指定相机名称
    （如 "teleop"）和图像类型（如 "left"）。

    除非在配置中覆盖，否则使用相机的默认设置（FPS、分辨率、颜色模式）。
    """

    def __init__(self, config: Reachy2CameraConfig):
        """
        初始化 Reachy2Camera 实例。

        参数：
            config: 相机的配置项。
        """
        require_package("reachy2_sdk", extra="reachy2")
        super().__init__(config)

        self.config = config

        self.color_mode = config.color_mode
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None

        self.cam_manager: CameraManager | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.name}, {self.config.image_type})"

    @property
    def is_connected(self) -> bool:
        """检查相机当前是否已连接并打开。"""
        if self.config.name == "teleop":
            return bool(
                self.cam_manager._grpc_connected and self.cam_manager.teleop if self.cam_manager else False
            )
        elif self.config.name == "depth":
            return bool(
                self.cam_manager._grpc_connected and self.cam_manager.depth if self.cam_manager else False
            )
        else:
            raise ValueError(f"Invalid camera name '{self.config.name}'. Expected 'teleop' or 'depth'.")

    def connect(self, warmup: bool = True) -> None:
        """
        按配置连接到 Reachy2 CameraManager。

        异常：
            DeviceNotConnectedError: 如果相机未连接。
        """
        self.cam_manager = CameraManager(host=self.config.ip_address, port=self.config.port)
        if self.cam_manager is None:
            raise DeviceNotConnectedError(f"Could not connect to {self}.")
        self.cam_manager.initialize_cameras()

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Reachy2 相机未实现检测功能。
        """
        raise NotImplementedError("Camera detection is not implemented for Reachy2 cameras.")

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        以同步方式从相机读取单帧。

        此方法获取 Reachy 2 底层软件中最新可用的帧。

        返回：
            np.ndarray: 捕获的帧（NumPy 数组），格式为
                       (height, width, channels)，使用指定或默认的
                       颜色模式，并应用已配置的旋转。
        """
        start_time = time.perf_counter()

        if self.cam_manager is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )

        frame: NDArray[Any] = np.empty((0, 0, 3), dtype=np.uint8)

        if self.config.name == "teleop" and hasattr(self.cam_manager, "teleop"):
            if self.config.image_type == "left":
                frame = self.cam_manager.teleop.get_frame(
                    CameraView.LEFT, size=(self.config.width, self.config.height)
                )[0]
            elif self.config.image_type == "right":
                frame = self.cam_manager.teleop.get_frame(
                    CameraView.RIGHT, size=(self.config.width, self.config.height)
                )[0]
        elif self.config.name == "depth" and hasattr(self.cam_manager, "depth"):
            if self.config.image_type == "depth":
                frame = self.cam_manager.depth.get_depth_frame()[0]
            elif self.config.image_type == "rgb":
                frame = self.cam_manager.depth.get_frame(size=(self.config.width, self.config.height))[0]
        else:
            raise ValueError(f"Invalid camera name '{self.config.name}'. Expected 'teleop' or 'depth'.")

        if frame is None:
            raise RuntimeError(f"Internal error: No frame available for {self}.")

        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )
        is_depth_frame = self.config.name == "depth" and self.config.image_type == "depth"
        if not is_depth_frame and self.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.latest_frame = frame
        self.latest_timestamp = time.perf_counter()

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        与 read() 相同

        返回：
            np.ndarray: 最新捕获的帧（NumPy 数组），格式为
                       (height, width, channels)，已按配置处理。

        异常：
            DeviceNotConnectedError: 如果相机未连接。
            TimeoutError: 如果在指定超时内没有帧可用。
            RuntimeError: 如果发生意外错误。
        """

        return self.read()

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """立即返回最近捕获的帧（窥视模式）。

        此方法是非阻塞的，直接返回当前内存缓冲区中的内容。
        该帧可能已过期，
        即它可能是很久之前捕获的（例如相机挂起的场景）。

        返回：
            tuple[NDArray, float]:
                - 帧图像（numpy 数组）。
                - 捕获该帧时的时间戳 (time.perf_counter)。

        异常：
            TimeoutError: 如果最新帧的年龄超过 `max_age_ms`。
            DeviceNotConnectedError: 如果相机未连接。
            RuntimeError: 如果相机已连接但尚未捕获任何帧。
        """

        if self.latest_frame is None or self.latest_timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - self.latest_timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return self.latest_frame

    @check_if_not_connected
    def disconnect(self) -> None:
        """
        停止后台读取线程（如果正在运行）。

        异常：
            DeviceNotConnectedError: 如果相机已断开连接。
        """

        if self.cam_manager is not None:
            self.cam_manager.disconnect()

        logger.info(f"{self} disconnected.")
