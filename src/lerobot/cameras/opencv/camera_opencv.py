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
提供 OpenCVCamera 类，用于使用 OpenCV 从相机捕获帧。
"""

import logging
import math
import os
import platform
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from numpy.typing import NDArray  # type: ignore  # TODO: 为 numpy.typing 添加类型存根

# 在导入 cv2 之前，修复 Windows 上 MSMF 硬件变换的兼容性问题
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2  # type: ignore  # TODO: 为 OpenCV 添加类型存根

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_rotation
from .configuration_opencv import ColorMode, OpenCVCameraConfig

# 注意(Steven)：opencv 的最大设备索引取决于你的操作系统。例如，
# 如果你有 3 个相机，它们应对应索引 0、1 和 2。
# 在 MacOS 上是这样的。但在 Ubuntu 上，索引可能不同，如 6、16、23。
# 当你更换 USB 端口或重启计算机时，操作系统可能会
# 将相同的相机视为新设备。因此我们选择一个较大的上界来搜索索引。
MAX_OPENCV_INDEX = 60

logger = logging.getLogger(__name__)


class OpenCVCamera(Camera):
    """
    使用 OpenCV 管理相机交互，实现高效的帧记录。

    该类提供了一个高级接口，用于连接、配置和读取
    与 OpenCV VideoCapture 兼容的相机的帧。它支持
    同步和异步两种帧读取方式。

    OpenCVCamera 实例需要一个相机索引（如 0）或设备路径
    （如 Linux 上的 '/dev/video0'）。相机索引在重启
    或更换端口后可能不稳定，尤其是在 Linux 上。请使用提供的工具脚本
    查找可用的相机索引或路径：
    ```bash
    lerobot-find-cameras opencv
    ```

    除非在配置中覆盖，否则使用相机的默认设置（FPS、分辨率、颜色模式）。

    示例：
        ```python
        from lerobot.cameras.opencv import OpenCVCamera
        from lerobot.cameras.configuration_opencv import OpenCVCameraConfig

        # 使用相机索引 0 的基本用法
        config = OpenCVCameraConfig(index_or_path=0)
        camera = OpenCVCamera(config)
        camera.connect()

        # 同步读取 1 帧（阻塞）
        color_image = camera.read()

        # 异步读取 1 帧（带超时等待新帧）
        async_image = camera.async_read()

        # 立即获取最新帧（不等待，返回时间戳）
        latest_image, timestamp = camera.read_latest()

        # 完成后，使用以下方式正确断开相机连接
        camera.disconnect()
        ```
    """

    def __init__(self, config: OpenCVCameraConfig):
        """
        初始化 OpenCVCamera 实例。

        参数：
            config: 相机的配置项。
        """
        super().__init__(config)

        self.config = config
        self.index_or_path = config.index_or_path

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s

        self.videocapture: cv2.VideoCapture | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.backend: int = config.backend

        self.capture_width: int | None = None
        self.capture_height: int | None = None
        self._reset_connection_settings()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    def _reset_connection_settings(self) -> None:
        """恢复可能在连接失败期间被自动检测到的设置。"""
        self.fps = self.config.fps
        self.width = self.config.width
        self.height = self.config.height
        self.capture_width, self.capture_height = self.width, self.height
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.capture_width, self.capture_height = self.height, self.width

    @property
    def is_connected(self) -> bool:
        """检查相机当前是否已连接并打开。"""
        return isinstance(self.videocapture, cv2.VideoCapture) and self.videocapture.isOpened()

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """
        连接到配置中指定的 OpenCV 相机。

        初始化 OpenCV VideoCapture 对象，设置所需的相机属性
        （FPS、宽度、高度），启动后台读取线程并执行初始检查。

        参数：
            warmup (bool): 如果为 True，在 connect() 时等待，直到后台线程
                           至少捕获到一个有效帧。默认为 True。

        异常：
            DeviceAlreadyConnectedError: 如果相机已连接。
            ConnectionError: 如果找不到指定的相机索引/路径或打开失败。
            RuntimeError: 如果相机打开但未能应用请求的设置。
        """

        # OpenCV 操作使用 1 个线程，以避免在多线程应用中
        # （尤其是在数据采集期间）出现潜在的冲突或阻塞。
        cv2.setNumThreads(1)

        self.videocapture = cv2.VideoCapture(self.index_or_path, self.backend)

        if not self.videocapture.isOpened():
            self.videocapture.release()
            self.videocapture = None
            raise ConnectionError(
                f"Failed to open {self}.Run `lerobot-find-cameras opencv` to find available cameras."
            )

        try:
            self._configure_capture_settings()
            self._start_read_thread()

            if warmup and self.warmup_s > 0:
                start_time = time.time()
                while time.time() - start_time < self.warmup_s:
                    self.async_read(timeout_ms=self.warmup_s * 1000)
                    time.sleep(0.1)
                with self.frame_lock:
                    if self.latest_frame is None:
                        raise ConnectionError(f"{self} failed to capture frames during warmup.")
        except BaseException:
            try:
                self._cleanup_resources()
            except Exception:
                logger.exception(f"Failed to fully clean up {self} after connect() failed.")
            self._reset_connection_settings()
            raise

        logger.info(f"{self} connected.")

    @check_if_not_connected
    def _configure_capture_settings(self) -> None:
        """
        将指定的 FOURCC、FPS、宽度和高度设置应用到已连接的相机。

        此方法尝试通过 OpenCV 设置相机属性。它会检查
        相机是否成功应用了设置，如果没有则抛出错误。
        FOURCC 最先设置（如果指定了的话），因为它可能影响可用的 FPS 和分辨率选项。

        参数：
            fourcc: 期望的 FOURCC 代码（如 "MJPG"、"YUYV"）。如果为 None，则自动检测。
            fps: 期望的每秒帧数。如果为 None，则跳过该设置。
            width: 期望的捕获宽度。如果为 None，则跳过该设置。
            height: 期望的捕获高度。如果为 None，则跳过该设置。

        异常：
            RuntimeError: 如果相机未能将任何指定属性
                          设置为请求的值。
            DeviceNotConnectedError: 如果相机未连接。
        """

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        set_fourcc_after_size_and_fps = platform.system() == "Windows"
        if self.config.fourcc is not None and not set_fourcc_after_size_and_fps:
            self._validate_fourcc()

        default_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if self.width is None or self.height is None:
            self.width, self.height = default_width, default_height
            self.capture_width, self.capture_height = default_width, default_height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = default_height, default_width
                self.capture_width, self.capture_height = default_width, default_height
        else:
            self._validate_width_and_height()

        if self.fps is None:
            self.fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        else:
            self._validate_fps()

        if self.config.fourcc is not None and set_fourcc_after_size_and_fps:
            # 在 Windows 上使用 DSHOW 时，更改分辨率可能会悄悄覆盖 FOURCC 设置。
            # 最后设置 FOURCC，以确保请求的像素格式确实生效。
            self._validate_fourcc()

    def _validate_fps(self) -> None:
        """验证并设置相机的每秒帧数（FPS）。"""

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        if self.fps is None:
            raise ValueError(f"{self} FPS is not set")

        success = self.videocapture.set(cv2.CAP_PROP_FPS, float(self.fps))
        actual_fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        # 使用 math.isclose 进行稳健的浮点数比较
        if not success or not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            raise RuntimeError(f"{self} failed to set fps={self.fps} ({actual_fps=}).")

    def _validate_fourcc(self) -> None:
        """验证并设置相机的 FOURCC 代码。"""

        fourcc_code = cv2.VideoWriter_fourcc(*self.config.fourcc)

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        success = self.videocapture.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        actual_fourcc_code = self.videocapture.get(cv2.CAP_PROP_FOURCC)

        # 将实际的 FOURCC 代码转换回字符串以便比较
        actual_fourcc_code_int = int(actual_fourcc_code)
        actual_fourcc = "".join([chr((actual_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)])

        if not success or actual_fourcc != self.config.fourcc:
            logger.warning(
                f"{self} failed to set fourcc={self.config.fourcc} (actual={actual_fourcc}, success={success}). "
                f"Continuing with default format."
            )

    def _validate_width_and_height(self) -> None:
        """验证并设置相机的帧捕获宽度和高度。"""

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        if self.capture_width is None or self.capture_height is None:
            raise ValueError(f"{self} capture_width or capture_height is not set")

        width_success = self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
        height_success = self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))

        actual_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if not width_success or self.capture_width != actual_width:
            raise RuntimeError(
                f"{self} failed to set capture_width={self.capture_width} ({actual_width=}, {width_success=})."
            )

        actual_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not height_success or self.capture_height != actual_height:
            raise RuntimeError(
                f"{self} failed to set capture_height={self.capture_height} ({actual_height=}, {height_success=})."
            )

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        检测连接到系统的可用 OpenCV 相机。

        在 Linux 上，扫描 '/dev/video*' 路径。在其他系统上（如 macOS、Windows），
        检查从 0 到 `MAX_OPENCV_INDEX` 的索引。

        返回：
            List[Dict[str, Any]]: 一个字典列表，
            每个字典包含 'type'、'id'（端口索引或路径），
            以及默认配置属性（width、height、fps、format）。
        """
        found_cameras_info = []

        targets_to_scan: list[str | int]
        if platform.system() == "Linux":
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            targets_to_scan = [str(p) for p in possible_paths]
        else:
            targets_to_scan = [int(i) for i in range(MAX_OPENCV_INDEX)]

        for target in targets_to_scan:
            camera = cv2.VideoCapture(target)
            try:
                if camera.isOpened():
                    default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    default_fps = camera.get(cv2.CAP_PROP_FPS)
                    default_format = camera.get(cv2.CAP_PROP_FORMAT)

                    # 获取 FOURCC 代码并转换为字符串
                    default_fourcc_code = camera.get(cv2.CAP_PROP_FOURCC)
                    default_fourcc_code_int = int(default_fourcc_code)
                    default_fourcc = "".join(
                        [chr((default_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)]
                    )

                    camera_info = {
                        "name": f"OpenCV Camera @ {target}",
                        "type": "OpenCV",
                        "id": target,
                        "backend_api": camera.getBackendName(),
                        "default_stream_profile": {
                            "format": default_format,
                            "fourcc": default_fourcc,
                            "width": default_width,
                            "height": default_height,
                            "fps": default_fps,
                        },
                    }

                    found_cameras_info.append(camera_info)
            finally:
                camera.release()

        return found_cameras_info

    def _read_from_hardware(self) -> NDArray[Any]:
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        ret, frame = self.videocapture.read()

        if not ret:
            raise RuntimeError(f"{self} read failed (status={ret}).")

        return frame

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        以同步方式从相机读取单帧。

        这是一个阻塞调用。它通过 OpenCV 等待来自
        相机硬件的下一个可用帧。

        返回：
            np.ndarray: 捕获的帧（NumPy 数组），格式为
                       (height, width, channels)，使用指定或默认的
                       颜色模式，并应用已配置的旋转。

        异常：
            DeviceNotConnectedError: 如果相机未连接。
            RuntimeError: 如果从相机读取帧失败，或者
                          收到的帧尺寸在旋转前与预期不符。
            ValueError: 如果请求了无效的 `color_mode`。
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

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        """
        对原始帧应用颜色转换、尺寸验证和旋转。

        参数：
            image (np.ndarray): 原始图像帧（预期为来自 OpenCV 的 BGR 格式）。

        返回：
            np.ndarray: 处理后的图像帧。

        异常：
            ValueError: 如果请求的 `color_mode` 无效。
            RuntimeError: 如果原始帧的尺寸与配置的
                          `width` 和 `height` 不匹配。
        """

        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        processed_image = image
        if self.color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self) -> None:
        """
        后台线程运行的内部循环，用于异步读取。

        每次迭代：
        1. 读取一个彩色帧（阻塞调用）
        2. 将结果存入 latest_frame 并更新时间戳（线程安全）
        3. 设置 new_frame_event 以通知监听者

        遇到 DeviceNotConnectedError 时停止，记录其他错误并继续。
        """
        stop_event = self.stop_event
        if stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not stop_event.is_set():
            try:
                raw_frame = self._read_from_hardware()
                processed_frame = self._postprocess_image(raw_frame)
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = processed_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        """如果后台读取线程未运行，则启动或重启它。"""
        self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.1)

    def _stop_read_thread(self) -> None:
        """通知后台读取线程停止，并等待其结束。"""
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

    def _cleanup_resources(self) -> None:
        """停止后台读取并释放捕获资源，包括在部分设置完成后的清理。"""
        read_thread = self.thread
        videocapture = self.videocapture

        try:
            self._stop_read_thread()
        finally:
            self.videocapture = None
            try:
                if videocapture is not None:
                    videocapture.release()
            finally:
                # 释放设备可能会解除一个硬件读取的阻塞，该读取可能
                # 在 _stop_read_thread() 中第一次有限 join 之后仍然存活。
                if read_thread is not None and read_thread.is_alive():
                    read_thread.join(timeout=2.0)
                    if read_thread.is_alive():  # pragma: no cover
                        logger.warning(f"{self} read thread remained alive after releasing the capture.")

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        以异步方式读取最新可用的帧。

        此方法获取后台读取线程捕获的最新帧。
        它不会直接阻塞等待相机硬件，
        但可能会等待至多 timeout_ms 让后台线程提供一帧。
        在高 FPS 下是"尽力而为"的。

        参数：
            timeout_ms (float): 等待帧可用的最长时间（毫秒）。
                默认为 200ms（0.2 秒）。

        返回：
            np.ndarray: 最新捕获的帧（NumPy 数组），格式为
                       (height, width, channels)，已按配置处理。

        异常：
            DeviceNotConnectedError: 如果相机未连接。
            TimeoutError: 如果在指定超时内没有帧可用。
            RuntimeError: 如果发生意外错误。
        """

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
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
        """
        断开与相机的连接并清理资源。

        停止后台读取线程（如果正在运行）并释放 OpenCV
        VideoCapture 对象。

        异常：
            DeviceNotConnectedError: 如果相机已断开连接。
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        self._cleanup_resources()

        logger.info(f"{self} disconnected.")
