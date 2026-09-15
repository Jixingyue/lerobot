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
提供 RealSenseCamera 类，用于从 Intel RealSense 相机捕获帧。
"""

import logging
import sys
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import cv2  # type: ignore  # TODO: 为 OpenCV 添加类型存根
import numpy as np  # type: ignore  # TODO: 为 numpy 添加类型存根
from numpy.typing import NDArray  # type: ignore  # TODO: 为 numpy.typing 添加类型存根

from lerobot.utils.import_utils import _pyrealsense2_available, require_package

if TYPE_CHECKING or _pyrealsense2_available:
    import pyrealsense2 as rs
else:
    rs = None

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_realsense import RealSenseCameraConfig

logger = logging.getLogger(__name__)
pkg_name = "pyrealsense2-macosx" if sys.platform == "darwin" else "pyrealsense2"


class RealSenseCamera(Camera):
    """
    管理与 Intel RealSense 相机的交互，用于记录帧和深度。

    该类提供了与 `OpenCVCamera` 类似的接口，但专为
    RealSense 设备定制，利用 `pyrealsense2` 库。它使用相机
    唯一的序列号进行标识，比设备索引更稳定，
    尤其是在 Linux 上。它还支持在捕获彩色帧的同时
    捕获深度图。

    使用提供的工具脚本查找可用的相机索引和默认配置：
    ```bash
    lerobot-find-cameras realsense
    ```

    `RealSenseCamera` 实例需要一个配置对象，用于指定
    相机的序列号或唯一的设备名称。如果使用名称，请确保
    只连接了一个具有该名称的相机。

    除非在配置中覆盖，否则使用流配置中相机的
    默认设置（FPS、分辨率、颜色模式）。

    示例：
        ```python
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
        from lerobot.cameras import ColorMode, Cv2Rotation

        # 使用序列号的基本用法
        config = RealSenseCameraConfig(serial_number_or_name="0123456789") # 替换为实际序列号
        camera = RealSenseCamera(config)
        camera.connect()

        # 同步读取 1 帧（阻塞）
        color_image = camera.read()

        # 异步读取 1 帧（带超时等待新帧）
        async_image = camera.async_read()

        # 立即获取最新帧（不等待，返回时间戳）
        latest_image, timestamp = camera.read_latest()

        # 带深度捕获和自定义设置的示例
        custom_config = RealSenseCameraConfig(
            serial_number_or_name="0123456789", # 替换为实际序列号
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR, # 请求 BGR 输出
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True
        )
        depth_camera = RealSenseCamera(custom_config)
        depth_camera.connect()

        # 读取 1 帧深度帧
        depth_map = depth_camera.read_depth()

        # 使用唯一相机名称的示例
        name_config = RealSenseCameraConfig(serial_number_or_name="Intel RealSense D435") # 如果名称唯一
        name_camera = RealSenseCamera(name_config)
        # ... 连接、读取、断开 ...
        ```
    """

    # connect() 执行的最大预热尝试次数。失败的尝试首先
    # 通过简单的 pipeline 停止/启动来重试，这通常足以恢复
    # 流；作为最后手段，在最终尝试前执行 USB 硬件复位。
    _MAX_CONNECT_ATTEMPTS = 3

    def __init__(self, config: RealSenseCameraConfig):
        """
        初始化 RealSenseCamera 实例。

        参数：
            config: 相机的配置项。
        """
        require_package(pkg_name, extra="intelrealsense", import_name="pyrealsense2")
        super().__init__(config)

        self.config = config

        self.width: int | None = config.width
        self.height: int | None = config.height

        if config.serial_number_or_name.isdigit():
            self.serial_number = config.serial_number_or_name
        else:
            self.serial_number = self._find_serial_number_from_name(config.serial_number_or_name)

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_rgb = config.use_rgb
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s
        self.exposure: int | None = config.exposure
        self.gain: int | None = config.gain
        self.white_balance: int | None = config.white_balance

        self.rs_pipeline: rs.pipeline | None = None
        self.rs_profile: rs.pipeline_profile | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.capture_width: int | None = None
        self.capture_height: int | None = None
        self._reset_connection_settings()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    def _reset_connection_settings(self) -> None:
        """恢复可能在连接失败期间被自动检测到的设置。"""
        self.fps = self.config.fps
        self.width = self.config.width
        self.height = self.config.height
        self.warmup_s = self.config.warmup_s
        self.capture_width, self.capture_height = self.width, self.height
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.capture_width, self.capture_height = self.height, self.width

    @property
    def is_connected(self) -> bool:
        """检查相机 pipeline 是否已启动且流是否处于活动状态。"""
        return self.rs_pipeline is not None and self.rs_profile is not None

    def _hardware_reset(self, wait_s: float = 5.0) -> None:
        """发出 USB 硬件复位以恢复无响应的设备（在 D405 上常见）。"""
        context = rs.context()
        for device in context.query_devices():
            if device.get_info(rs.camera_info.serial_number) == self.serial_number:
                logger.info(f"{self} performing hardware reset.")
                device.hardware_reset()
                time.sleep(wait_s)
                return
        logger.warning(f"{self} device not found for hardware reset, skipping.")

    def _open_pipeline(self) -> None:
        """初始化 RealSense pipeline，启动它，并启动后台读取线程。

        异常：
            ValueError: 如果配置无效、请求的传感器选项不受支持，
                或请求的传感器值无效。
            ConnectionError: 如果找到了相机但无法启动 pipeline，或完全未检测到 RealSense 设备。
            RuntimeError: 如果 pipeline 启动但未能应用请求的设置。
        """
        rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        self._configure_rs_pipeline_config(rs_config)

        try:
            rs_profile = rs_pipeline.start(rs_config)
        except RuntimeError as e:
            raise ConnectionError(
                f"Failed to open {self}.Run `lerobot-find-cameras realsense` to find available cameras."
            ) from e

        self.rs_pipeline = rs_pipeline
        self.rs_profile = rs_profile

        try:
            self._configure_capture_settings()
            self._configure_sensor_options()
            self._start_read_thread()
        except BaseException:
            self._release_after_failed_setup()
            raise

    def _run_warmup(self) -> None:
        """阻塞直到后台线程至少捕获到一个有效帧。

        异常：
            ConnectionError: 如果在 ``warmup_s`` 结束前没有帧到达。
        """
        # 注意(Steven/Caroline)：强制至少一秒的预热，因为 RS 相机在第一次读取前需要一点时间。如果不等待，预热的第一次读取会抛出异常。
        self.warmup_s = max(self.warmup_s, 1)

        warmup_read = self.async_read if self.use_rgb else self.async_read_depth
        start_time = time.time()
        while time.time() - start_time < self.warmup_s:
            warmup_read(timeout_ms=self.warmup_s * 1000)
            time.sleep(0.1)
        with self.frame_lock:
            if (self.use_rgb and self.latest_color_frame is None) or (
                self.use_depth and self.latest_depth_frame is None
            ):
                raise ConnectionError(f"{self} failed to capture frames during warmup.")

    def _release_after_failed_setup(self) -> None:
        """在失败的尝试后释放设备句柄并恢复自动检测到的设置。"""
        try:
            self._cleanup_resources()
        except Exception:
            logger.exception(f"Failed to fully clean up {self} after connect() failed.")
        self._reset_connection_settings()

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """
        连接到配置中指定的 RealSense 相机。

        初始化 RealSense pipeline，配置所需的流（彩色流
        以及可选的深度流），启动 pipeline，并验证实际的流设置。

        如果 pipeline 启动但预热期间没有帧到达，最多重试
        ``_MAX_CONNECT_ATTEMPTS`` 次，并在最终尝试前
        执行 USB 硬件复位。

        参数：
            warmup (bool): 如果为 True，在 connect() 时等待，直到后台线程
                           至少捕获到一个有效帧。默认为 True。

        异常：
            DeviceAlreadyConnectedError: 如果相机已连接。
            ValueError: 如果配置无效（如缺少序列号/名称、名称不唯一）。
            ConnectionError: 如果找到了相机但无法启动 pipeline，或完全未检测到 RealSense 设备。
            RuntimeError: 如果 pipeline 启动但未能应用请求的设置。
        """

        if not warmup:
            self._open_pipeline()
            logger.info(f"{self} connected.")
            return

        last_error: Exception | None = None

        for attempt in range(1, self._MAX_CONNECT_ATTEMPTS + 1):
            if attempt == self._MAX_CONNECT_ATTEMPTS:
                self._hardware_reset()

            self._open_pipeline()

            connected = False
            try:
                self._run_warmup()
                connected = True
            except (TimeoutError, ConnectionError) as e:
                last_error = e
            finally:
                if not connected:
                    self._release_after_failed_setup()

            if connected:
                logger.info(f"{self} connected.")
                return

            logger.warning(f"{self} warmup failed (attempt {attempt}/{self._MAX_CONNECT_ATTEMPTS}).")

        raise ConnectionError(
            f"{self} failed to capture frames after {self._MAX_CONNECT_ATTEMPTS} attempts."
        ) from last_error

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        检测连接到系统的可用 Intel RealSense 相机。

        返回：
            List[Dict[str, Any]]: 一个字典列表，
            每个字典包含 'type'、'id'（序列号）、'name'、
            固件版本、USB 类型及其他可用规格，以及默认配置属性（width、height、fps、format）。

        异常：
            OSError: 如果未安装 pyrealsense2。
            ImportError: 如果未安装 pyrealsense2。
        """
        found_cameras_info = []
        context = rs.context()
        devices = context.query_devices()

        for device in devices:
            camera_info = {
                "name": device.get_info(rs.camera_info.name),
                "type": "RealSense",
                "id": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor),
                "physical_port": device.get_info(rs.camera_info.physical_port),
                "product_id": device.get_info(rs.camera_info.product_id),
                "product_line": device.get_info(rs.camera_info.product_line),
            }

            # 获取每个传感器的流配置
            sensors = device.query_sensors()
            for sensor in sensors:
                profiles = sensor.get_stream_profiles()

                for profile in profiles:
                    if profile.is_video_stream_profile() and profile.is_default():
                        vprofile = profile.as_video_stream_profile()
                        stream_info = {
                            "stream_type": vprofile.stream_name(),
                            "format": vprofile.format().name,
                            "width": vprofile.width(),
                            "height": vprofile.height(),
                            "fps": vprofile.fps(),
                        }
                        camera_info["default_stream_profile"] = stream_info

            found_cameras_info.append(camera_info)

        return found_cameras_info

    def _find_serial_number_from_name(self, name: str) -> str:
        """根据给定的唯一相机名称查找序列号。"""
        camera_infos = self.find_cameras()
        found_devices = [cam for cam in camera_infos if str(cam["name"]) == name]

        if not found_devices:
            available_names = [cam["name"] for cam in camera_infos]
            raise ValueError(
                f"No RealSense camera found with name '{name}'. Available camera names: {available_names}"
            )

        if len(found_devices) > 1:
            serial_numbers = [dev["id"] for dev in found_devices]
            raise ValueError(
                f"Multiple RealSense cameras found with name '{name}'. "
                f"Please use a unique serial number instead. Found SNs: {serial_numbers}"
            )

        serial_number = str(found_devices[0]["id"])
        return serial_number

    def _configure_rs_pipeline_config(self, rs_config: Any) -> None:
        """创建并配置 RealSense pipeline 配置对象。"""
        rs.config.enable_device(rs_config, self.serial_number)

        if self.width and self.height and self.fps:
            if self.use_rgb:
                rs_config.enable_stream(
                    rs.stream.color, self.capture_width, self.capture_height, rs.format.rgb8, self.fps
                )
            if self.use_depth:
                rs_config.enable_stream(
                    rs.stream.depth, self.capture_width, self.capture_height, rs.format.z16, self.fps
                )
        else:
            if self.use_rgb:
                rs_config.enable_stream(rs.stream.color)
            if self.use_depth:
                rs_config.enable_stream(rs.stream.depth)

    @check_if_not_connected
    def _configure_capture_settings(self) -> None:
        """如果尚未配置，则从设备流中设置 fps、width 和 height。

        使用彩色流配置（或在彩色流被禁用时使用深度流配置）
        来更新未设置的属性。通过在需要时交换
        宽度/高度来处理旋转。始终存储原始捕获尺寸。

        异常：
            DeviceNotConnectedError: 如果设备未连接。
        """

        if self.rs_profile is None:
            raise RuntimeError(f"{self}: rs_profile must be initialized before use.")

        rs_stream = rs.stream.color if self.use_rgb else rs.stream.depth
        stream = self.rs_profile.get_stream(rs_stream).as_video_stream_profile()

        if self.fps is None:
            self.fps = stream.fps()

        if self.width is None or self.height is None:
            actual_width = int(round(stream.width()))
            actual_height = int(round(stream.height()))
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_height, actual_width
                self.capture_width, self.capture_height = actual_width, actual_height
            else:
                self.width, self.height = actual_width, actual_height
                self.capture_width, self.capture_height = actual_width, actual_height

    def _read(self, read_depth: bool = False) -> NDArray[Any]:
        """:meth:`read`/:meth:`read_depth` 的共享辅助方法：等待一个新鲜的彩色帧或深度帧。"""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        return self._async_read(timeout_ms=10000, read_depth=read_depth)

    def _get_color_sensor(self) -> "rs.sensor":
        """返回控制彩色流的专用 "RGB Camera" 传感器。

        手动彩色控制仅应用于专用的 RGB 模块。没有该模块的相机
        （如 D405，其彩色流来自共享的
        "Stereo Module"）不受支持，因此我们绝不回退到其他传感器，
        以避免影响深度流。
        """
        if self.rs_profile is None:
            raise RuntimeError(f"{self}: rs_profile must be initialized before use.")

        device = self.rs_profile.get_device()
        sensors = {s.get_info(rs.camera_info.name): s for s in device.query_sensors()}

        if "RGB Camera" in sensors:
            return sensors["RGB Camera"]

        available = list(sensors.keys())
        raise RuntimeError(
            f"{self}: manual color controls require a dedicated 'RGB Camera' module, which this camera does not have. ",
            f"Available sensors: {available}.",
        )

    def _set_sensor_option(self, sensor: "rs.sensor", option: "rs.option", value: float, label: str) -> None:
        """设置传感器选项，在范围错误时重新抛出并附带可操作的诊断信息。"""
        try:
            sensor.set_option(option, value)
        except Exception as e:
            range_info = ""
            try:
                option_range = sensor.get_option_range(option)
                range_info = (
                    f" (supported range: min={option_range.min}, max={option_range.max}, "
                    f"step={option_range.step}, default={option_range.default})"
                )
            except Exception:
                range_info = " (option range unavailable)"
            raise ValueError(
                f"{self}: failed to set {label} to {value}{range_info}. Original error: {e}"
            ) from e

    def _configure_sensor_options(self) -> None:
        """将手动传感器选项（曝光、增益、白平衡）应用到彩色传感器。

        当设置了 exposure 或 gain 时，会先禁用自动曝光。当设置了
        white_balance 时，会先禁用自动白平衡。省略的选项保持不变，
        如果所有选项都被省略，则完全跳过配置。

        异常：
            ValueError: 如果传感器不支持请求的选项，或请求的
                值无效。无效值错误会包含选项名称、请求的
                值以及可用时的支持范围。
        """
        if self.exposure is None and self.gain is None and self.white_balance is None:
            return

        color_sensor = self._get_color_sensor()

        requested_options = (
            (rs.option.exposure, self.exposure, "exposure"),
            (rs.option.gain, self.gain, "gain"),
            (rs.option.white_balance, self.white_balance, "white balance"),
        )
        unsupported_options = [
            label
            for option, value, label in requested_options
            if value is not None and not color_sensor.supports(option)
        ]
        if unsupported_options:
            raise ValueError(
                f"{self}: color sensor does not support requested manual options: {unsupported_options}."
            )

        manual_exposure_requested = self.exposure is not None or self.gain is not None
        if manual_exposure_requested:
            if color_sensor.supports(rs.option.enable_auto_exposure):
                self._set_sensor_option(color_sensor, rs.option.enable_auto_exposure, 0, "auto-exposure")
                logger.info(f"{self} auto-exposure disabled.")
            else:
                logger.warning(
                    f"{self} sensor does not support disabling auto-exposure; "
                    "applying manual exposure/gain directly."
                )

        if self.exposure is not None:
            self._set_sensor_option(color_sensor, rs.option.exposure, self.exposure, "exposure")
            logger.info(f"{self} exposure set to {self.exposure}.")

        if self.gain is not None:
            self._set_sensor_option(color_sensor, rs.option.gain, self.gain, "gain")
            logger.info(f"{self} gain set to {self.gain}.")

        if self.white_balance is not None:
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                self._set_sensor_option(
                    color_sensor, rs.option.enable_auto_white_balance, 0, "auto white balance"
                )
                logger.info(f"{self} auto white balance disabled.")
            else:
                logger.warning(
                    f"{self} sensor does not support disabling auto white balance; "
                    "applying manual white balance directly."
                )
            self._set_sensor_option(
                color_sensor, rs.option.white_balance, self.white_balance, "white balance"
            )
            logger.info(f"{self} white balance set to {self.white_balance}.")

    @check_if_not_connected
    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        """
        以同步方式从相机读取单帧（深度）。

        这是一个阻塞调用。它通过 RealSense pipeline 等待来自
        相机硬件的一组一致的帧（深度）。

        返回：
            np.ndarray: 深度图（NumPy 数组），形状为 (height, width, 1)，
                  类型为 `np.uint16`（原始深度值，单位为毫米）。

        异常：
            DeviceNotConnectedError: 如果相机未连接。
            RuntimeError: 如果从 pipeline 读取帧失败或帧无效。
        """
        if timeout_ms:
            logger.warning(
                f"{self} read() timeout_ms parameter is deprecated and will be removed in future versions."
            )

        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame '.read_depth()'. Depth stream is not enabled for {self}."
            )

        return self._read(read_depth=True)

    def _read_from_hardware(self):
        if self.rs_pipeline is None:
            raise RuntimeError(f"{self}: rs_pipeline must be initialized before use.")

        ret, frame = self.rs_pipeline.try_wait_for_frames(timeout_ms=10000)

        if not ret or frame is None:
            raise RuntimeError(f"{self} read failed (status={ret}).")

        return frame

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 0) -> NDArray[Any]:
        """
        以同步方式从相机读取单帧（彩色）。

        这是一个阻塞调用。它通过 RealSense pipeline 等待来自
        相机硬件的一组一致的帧（彩色）。

        返回：
            np.ndarray: 捕获的彩色帧（NumPy 数组），
              形状为 (height, width, channels)，已按 `color_mode` 和旋转处理。

        异常：
            DeviceNotConnectedError: 如果相机未连接。
            RuntimeError: 如果从 pipeline 读取帧失败或帧无效。
            ValueError: 如果请求了无效的 `color_mode`。
        """

        start_time = time.perf_counter()

        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )

        if timeout_ms:
            logger.warning(
                f"{self} read() timeout_ms parameter is deprecated and will be removed in future versions."
            )

        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — camera was configured with use_rgb=False.")

        frame = self._read()

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def _postprocess_image(self, image: NDArray[Any], depth_frame: bool = False) -> NDArray[Any]:
        """
        对原始彩色帧应用颜色转换、尺寸验证和旋转。

        参数：
            image (np.ndarray): 原始图像帧（预期为来自 RealSense 的 RGB 格式）。

        返回：
            np.ndarray: 按 `self.color_mode` 和 `self.rotation` 处理后的图像帧。

        异常：
            ValueError: 如果请求的 `color_mode` 无效。
            RuntimeError: 如果原始帧的尺寸与配置的
                          `width` 和 `height` 不匹配。
        """

        if self.color_mode and self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if depth_frame:
            h, w = image.shape
        else:
            h, w, c = image.shape

            if c != 3:
                raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        processed_image = image
        if not depth_frame and self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self) -> None:
        """
        后台线程运行的内部循环，用于异步读取。

        每次迭代：
        1. 读取一个彩色/深度帧（阻塞调用，10 秒超时）
        2. 将结果存入 latest_color_frame/latest_depth_frame 并更新时间戳（线程安全）
        3. 设置 new_frame_event 以通知监听者

        遇到 DeviceNotConnectedError 时停止，记录其他错误并继续。
        """
        stop_event = self.stop_event
        if stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not stop_event.is_set():
            try:
                frame = self._read_from_hardware()

                if self.use_rgb:
                    color_frame_raw = frame.get_color_frame()
                    color_frame = np.asanyarray(color_frame_raw.get_data())
                    processed_color_frame = self._postprocess_image(color_frame)

                if self.use_depth:
                    depth_frame_raw = frame.get_depth_frame()
                    depth_frame = np.asanyarray(depth_frame_raw.get_data())
                    processed_depth_frame = self._postprocess_image(depth_frame, depth_frame=True)
                    if processed_depth_frame.ndim == 2:  # (H, W) -> (H, W, 1)
                        processed_depth_frame = processed_depth_frame[..., np.newaxis]

                capture_time = time.perf_counter()

                with self.frame_lock:
                    # 在锁内执行，这样迟到的帧不会复活已被 _stop_read_thread() 清空的缓冲区。
                    if stop_event.is_set():
                        break
                    if self.use_rgb:
                        self.latest_color_frame = processed_color_frame
                    if self.use_depth:
                        self.latest_depth_frame = processed_depth_frame
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

    def _stop_read_thread(self) -> None:
        """通知后台读取线程停止，并等待其结束。"""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():  # pragma: no cover
                logger.warning(f"{self} read thread did not terminate within timeout.")

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    def _cleanup_resources(self) -> None:
        """停止后台读取并停止 pipeline，包括在部分设置完成后的清理。"""
        read_thread = self.thread
        rs_pipeline = self.rs_pipeline

        try:
            self._stop_read_thread()
        finally:
            self.rs_pipeline = None
            self.rs_profile = None
            try:
                if rs_pipeline is not None:
                    rs_pipeline.stop()
            finally:
                # 停止 pipeline 可能会解除一个硬件读取的阻塞，该读取可能
                # 在 _stop_read_thread() 中第一次有限 join 之后仍然存活。
                if read_thread is not None and read_thread.is_alive():
                    read_thread.join(timeout=2.0)
                    if read_thread.is_alive():  # pragma: no cover
                        logger.warning(f"{self} read thread remained alive after stopping the pipeline.")

    def _async_read(self, timeout_ms: float, read_depth: bool = False) -> NDArray[Any]:
        """:meth:`async_read`/:meth:`async_read_depth` 的共享辅助方法：返回最新缓冲的帧。"""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        以异步方式读取最新可用的帧数据（彩色）。

        此方法获取后台读取线程捕获的最新彩色帧。
        它不会直接阻塞等待相机硬件，
        但可能会等待至多 timeout_ms 让后台线程提供一帧。
        在高 FPS 下是"尽力而为"的。

        参数：
            timeout_ms (float): 等待帧可用的最长时间（毫秒）。
                默认为 200ms（0.2 秒）。

        返回：
            np.ndarray:
            最新捕获的帧数据（彩色图像），已按配置处理。

        异常：
            DeviceNotConnectedError: 如果相机未连接。
            TimeoutError: 如果在指定超时内没有帧数据可用。
            RuntimeError: 如果后台线程意外终止或发生其他错误。
        """

        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — camera was configured with use_rgb=False.")

        return self._async_read(timeout_ms=timeout_ms)

    def _read_latest(self, max_age_ms: int, read_depth: bool = False) -> NDArray[Any]:
        """:meth:`read_latest`/:meth:`read_latest_depth` 的共享辅助方法：窥视最新缓冲的帧。"""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """立即返回最近捕获的（彩色）帧（窥视模式）。

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
        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — camera was configured with use_rgb=False.")

        return self._read_latest(max_age_ms=max_age_ms)

    @check_if_not_connected
    def async_read_depth(self, timeout_ms: float = 200) -> NDArray[np.uint16]:
        """以异步方式读取最新深度帧，单位为毫米。

        与 :meth:`async_read` 类似，但返回深度流而非
        彩色流。输出为形状 ``(H, W, 1)`` 的 ``np.uint16``，其中每个
        像素是到传感器的距离（毫米）。

        异常：
            DeviceNotConnectedError: 如果相机未连接。
            RuntimeError: 如果该相机的 ``use_depth`` 为 ``False``，或
                后台读取线程未运行。
            TimeoutError: 如果在 ``timeout_ms`` 内没有帧可用。
        """
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera was configured with use_depth=False.")

        return self._async_read(timeout_ms=timeout_ms, read_depth=True)

    @check_if_not_connected
    def read_latest_depth(self, max_age_ms: int = 500) -> NDArray[Any]:
        """立即返回最新的深度帧（窥视模式），单位为毫米。

        :meth:`read_latest` 针对深度流的非阻塞对应方法。
        输出为形状 ``(H, W, 1)`` 的 ``np.uint16``，其中每个像素是
        到传感器的距离（毫米）。

        异常：
            DeviceNotConnectedError: 如果相机未连接。
            RuntimeError: 如果该相机的 ``use_depth`` 为 ``False``，或
                尚未捕获任何深度帧。
            TimeoutError: 如果最新深度帧的年龄超过 ``max_age_ms``。
        """
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera was configured with use_depth=False.")

        return self._read_latest(max_age_ms=max_age_ms, read_depth=True)

    def disconnect(self) -> None:
        """
        断开与相机的连接，停止 pipeline，并清理资源。

        停止后台读取线程（如果正在运行）并停止 RealSense pipeline。

        异常：
            DeviceNotConnectedError: 如果相机已断开连接（pipeline 未运行）。
        """

        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        self._cleanup_resources()

        logger.info(f"{self} disconnected.")
