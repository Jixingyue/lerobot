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
"""使用 Frodobots SDK 的 EarthRover Mini Plus 机器人。"""

import base64
import logging
from functools import cached_property

import cv2
import numpy as np
import requests

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_earthrover_mini_plus import EarthRoverMiniPlusConfig

logger = logging.getLogger(__name__)

# 动作特征键
ACTION_LINEAR_VEL = "linear_velocity"
ACTION_ANGULAR_VEL = "angular_velocity"

# 观测特征键 — 相机
OBS_FRONT = "front"
OBS_REAR = "rear"

# 观测特征键 — 遥测
OBS_SPEED = "speed"
OBS_BATTERY_LEVEL = "battery_level"
OBS_ORIENTATION = "orientation"
OBS_GPS_LATITUDE = "gps_latitude"
OBS_GPS_LONGITUDE = "gps_longitude"
OBS_GPS_SIGNAL = "gps_signal"
OBS_SIGNAL_LEVEL = "signal_level"
OBS_VIBRATION = "vibration"
OBS_LAMP = "lamp"

# 观测特征键 — IMU 传感器
OBS_ACCELEROMETER_X = "accelerometer_x"
OBS_ACCELEROMETER_Y = "accelerometer_y"
OBS_ACCELEROMETER_Z = "accelerometer_z"
OBS_GYROSCOPE_X = "gyroscope_x"
OBS_GYROSCOPE_Y = "gyroscope_y"
OBS_GYROSCOPE_Z = "gyroscope_z"
OBS_MAGNETOMETER_X = "magnetometer_filtered_x"
OBS_MAGNETOMETER_Y = "magnetometer_filtered_y"
OBS_MAGNETOMETER_Z = "magnetometer_filtered_z"

# 观测特征键 — 车轮转速（RPM）
OBS_WHEEL_RPM_0 = "wheel_rpm_0"
OBS_WHEEL_RPM_1 = "wheel_rpm_1"
OBS_WHEEL_RPM_2 = "wheel_rpm_2"
OBS_WHEEL_RPM_3 = "wheel_rpm_3"


class EarthRoverMiniPlus(Robot):
    """
    通过 Frodobots SDK HTTP API 控制的 EarthRover Mini Plus 机器人。

    该机器人通过 Frodobots SDK 使用基于云的控制，而不是直接
    连接硬件。相机通过 Agora 云以 WebRTC 方式传输视频流，控制
    命令通过 HTTP POST 请求发送。

    该机器人支持：
    - 通过 SDK HTTP 端点访问的双相机（前和后）
    - 线速度和角速度控制
    - 电池和朝向遥测

    Attributes:
        config: 机器人配置
        sdk_base_url: Frodobots SDK 服务器的基础 URL，取自
            ``config.sdk_url``（默认：http://localhost:8000）
    """

    config_class = EarthRoverMiniPlusConfig
    name = "earthrover_mini_plus"

    def __init__(self, config: EarthRoverMiniPlusConfig):
        """初始化 EarthRover Mini Plus 机器人。

        Args:
            config: 机器人配置，包括 SDK URL
        """
        super().__init__(config)
        self.config = config
        self.sdk_base_url = config.sdk_url

        # 空的相机字典，用于兼容录制脚本
        # 相机通过 SDK 直接访问，而不是通过 Camera 对象
        self.cameras = {}
        self._is_connected = False

        # 相机帧缓存（请求失败时的回退）
        self._last_front_frame = None
        self._last_rear_frame = None

        # 机器人遥测数据缓存（请求失败时的回退）
        self._last_robot_data = None

        logger.info(f"Initialized {self.name} with SDK at {self.sdk_base_url}")

    @property
    def is_connected(self) -> bool:
        """检查机器人是否已连接到 SDK。"""
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """通过 Frodobots SDK 连接到机器人。

        Args:
            calibrate: 对基于 SDK 的机器人不使用（为 API 兼容性而保留）

        Raises:
            DeviceAlreadyConnectedError: 如果机器人已连接
            DeviceNotConnectedError: 如果无法连接到 SDK 服务器
        """

        # 验证 SDK 正在运行且可访问
        try:
            response = requests.get(f"{self.sdk_base_url}/data", timeout=10.0)
            if response.status_code != 200:
                raise DeviceNotConnectedError(
                    f"Cannot connect to SDK at {self.sdk_base_url}. "
                    "Make sure it's running: hypercorn main:app --reload"
                )
        except requests.RequestException as e:
            raise DeviceNotConnectedError(f"Cannot connect to SDK at {self.sdk_base_url}: {e}") from e

        self._is_connected = True
        logger.info(f"{self.name} connected to SDK")

        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        """基于 SDK 的机器人不需要校准。"""
        logger.info("Calibration not required for SDK-based robot")

    @property
    def is_calibrated(self) -> bool:
        """SDK 机器人不需要校准。

        Returns:
            bool: 对基于 SDK 的机器人始终为 True
        """
        return True

    def configure(self) -> None:
        """配置机器人（对基于 SDK 的机器人为空操作）。"""
        pass

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """定义用于数据集录制的观测空间。

        Returns:
            dict: 带类型/形状的观测特征：
                - front: (480, 640, 3) - 前相机 RGB 图像
                - rear: (480, 640, 3) - 后相机 RGB 图像
                - speed: float - 当前速度（SDK 原始值）
                - battery_level: float - 电池电量（0-100）
                - orientation: float - 机器人朝向（度）
                - gps_latitude: float - GPS 纬度坐标
                - gps_longitude: float - GPS 经度坐标
                - gps_signal: float - GPS 信号强度（百分比）
                - signal_level: float - 网络信号等级（0-5）
                - vibration: float - 振动传感器读数
                - lamp: float - 灯状态（0=关，1=开）
                - accelerometer_x: float - 加速度计 X 轴（SDK 原始值）
                - accelerometer_y: float - 加速度计 Y 轴（SDK 原始值）
                - accelerometer_z: float - 加速度计 Z 轴（SDK 原始值）
                - gyroscope_x: float - 陀螺仪 X 轴（SDK 原始值）
                - gyroscope_y: float - 陀螺仪 Y 轴（SDK 原始值）
                - gyroscope_z: float - 陀螺仪 Z 轴（SDK 原始值）
                - magnetometer_filtered_x: float - 磁力计 X 轴（SDK 原始值）
                - magnetometer_filtered_y: float - 磁力计 Y 轴（SDK 原始值）
                - magnetometer_filtered_z: float - 磁力计 Z 轴（SDK 原始值）
                - wheel_rpm_0: float - 车轮 0 转速（RPM）
                - wheel_rpm_1: float - 车轮 1 转速（RPM）
                - wheel_rpm_2: float - 车轮 2 转速（RPM）
                - wheel_rpm_3: float - 车轮 3 转速（RPM）
        """
        return {
            # 相机（高度、宽度、通道数）
            OBS_FRONT: (480, 640, 3),
            OBS_REAR: (480, 640, 3),
            # 遥测
            OBS_SPEED: float,
            OBS_BATTERY_LEVEL: float,
            OBS_ORIENTATION: float,
            OBS_GPS_LATITUDE: float,
            OBS_GPS_LONGITUDE: float,
            OBS_GPS_SIGNAL: float,
            OBS_SIGNAL_LEVEL: float,
            OBS_VIBRATION: float,
            OBS_LAMP: float,
            # IMU — 加速度计
            OBS_ACCELEROMETER_X: float,
            OBS_ACCELEROMETER_Y: float,
            OBS_ACCELEROMETER_Z: float,
            # IMU — 陀螺仪
            OBS_GYROSCOPE_X: float,
            OBS_GYROSCOPE_Y: float,
            OBS_GYROSCOPE_Z: float,
            # IMU — 磁力计
            OBS_MAGNETOMETER_X: float,
            OBS_MAGNETOMETER_Y: float,
            OBS_MAGNETOMETER_Z: float,
            # 车轮转速（RPM）
            OBS_WHEEL_RPM_0: float,
            OBS_WHEEL_RPM_1: float,
            OBS_WHEEL_RPM_2: float,
            OBS_WHEEL_RPM_3: float,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        """定义动作空间。

        Returns:
            dict: 带类型的动作特征：
                - linear_velocity: float - 目标线速度（-1 到 1）
                - angular_velocity: float - 目标角速度（-1 到 1）
        """
        return {
            ACTION_LINEAR_VEL: float,
            ACTION_ANGULAR_VEL: float,
        }

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """从 SDK 获取当前机器人观测。

        相机帧从 SDK 端点 /v2/front 和 /v2/rear 获取。
        帧从 base64 解码，并从 BGR 格式转换为 RGB 格式。
        机器人遥测从 /data 端点获取。
        传感器数组（accels、gyros、mags、rpms）的每个条目都是
        [values..., timestamp] 的形式；使用每个数组中的最新读数。

        Returns:
            RobotObservation: 包含以下内容的观测：
                - front: 前相机图像 (480, 640, 3)，RGB 格式
                - rear: 后相机图像 (480, 640, 3)，RGB 格式
                - speed: float - 当前速度（SDK 原始值）
                - battery_level: float - 电池电量（0-100）
                - orientation: float - 机器人朝向（度）
                - gps_latitude: float - GPS 纬度坐标
                - gps_longitude: float - GPS 经度坐标
                - gps_signal: float - GPS 信号强度（百分比）
                - signal_level: float - 网络信号等级（0-5）
                - vibration: float - 振动传感器读数
                - lamp: float - 灯状态（0=关，1=开）
                - accelerometer_x/y/z: float - 加速度计各轴（SDK 原始值）
                - gyroscope_x/y/z: float - 陀螺仪各轴（SDK 原始值）
                - magnetometer_filtered_x/y/z: float - 磁力计各轴（SDK 原始值）
                - wheel_rpm_0/1/2/3: float - 车轮转速（RPM）

        Raises:
            DeviceNotConnectedError: 如果机器人未连接

        Note:
            相机帧从 SDK 端点 /v2/front 和 /v2/rear 获取。
            帧从 base64 解码，并从 BGR 格式转换为 RGB 格式。
            机器人遥测从 /data 端点获取。
            所有 SDK 值都归一化到适合数据集录制的范围。
        """

        observation = {}

        # 从 SDK 获取相机图像
        frames = self._get_camera_frames()
        observation[OBS_FRONT] = frames["front"]
        observation[OBS_REAR] = frames["rear"]

        # 从 SDK 获取机器人状态
        robot_data = self._get_robot_data()

        # 遥测
        observation[OBS_SPEED] = float(robot_data["speed"])
        observation[OBS_BATTERY_LEVEL] = float(robot_data["battery"])
        observation[OBS_ORIENTATION] = float(robot_data["orientation"])
        observation[OBS_GPS_LATITUDE] = float(robot_data["latitude"])
        observation[OBS_GPS_LONGITUDE] = float(robot_data["longitude"])
        observation[OBS_GPS_SIGNAL] = float(robot_data["gps_signal"])
        observation[OBS_SIGNAL_LEVEL] = float(robot_data["signal_level"])
        observation[OBS_VIBRATION] = float(robot_data["vibration"])
        observation[OBS_LAMP] = float(robot_data["lamp"])

        # 加速度计 — accels 数组的最新读数 [x, y, z, ts]
        accel = self._latest_sensor_reading(robot_data, "accels", n_values=3)
        observation[OBS_ACCELEROMETER_X] = accel[0]
        observation[OBS_ACCELEROMETER_Y] = accel[1]
        observation[OBS_ACCELEROMETER_Z] = accel[2]

        # 陀螺仪 — gyros 数组的最新读数 [x, y, z, ts]
        gyro = self._latest_sensor_reading(robot_data, "gyros", n_values=3)
        observation[OBS_GYROSCOPE_X] = gyro[0]
        observation[OBS_GYROSCOPE_Y] = gyro[1]
        observation[OBS_GYROSCOPE_Z] = gyro[2]

        # 磁力计 — mags 数组的最新读数 [x, y, z, ts]
        mag = self._latest_sensor_reading(robot_data, "mags", n_values=3)
        observation[OBS_MAGNETOMETER_X] = mag[0]
        observation[OBS_MAGNETOMETER_Y] = mag[1]
        observation[OBS_MAGNETOMETER_Z] = mag[2]

        # 车轮转速 — rpms 数组的最新读数 [w0, w1, w2, w3, ts]
        rpm = self._latest_sensor_reading(robot_data, "rpms", n_values=4)
        observation[OBS_WHEEL_RPM_0] = rpm[0]
        observation[OBS_WHEEL_RPM_1] = rpm[1]
        observation[OBS_WHEEL_RPM_2] = rpm[2]
        observation[OBS_WHEEL_RPM_3] = rpm[3]

        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """通过 SDK 向机器人发送动作。

        Args:
            action: 包含以下键的动作字典：
                - linear_velocity: 目标线速度（-1 到 1）
                - angular_velocity: 目标角速度（-1 到 1）

        Returns:
            RobotAction: 已发送的动作（与 action_features 的键匹配）

        Raises:
            DeviceNotConnectedError: 如果机器人未连接

        Note:
            动作通过 POST /control 端点发送到 SDK。
            SDK 期望命令在 [-1, 1] 范围内。
        """
        linear = float(action.get(ACTION_LINEAR_VEL, 0.0))
        angular = float(action.get(ACTION_ANGULAR_VEL, 0.0))

        try:
            self._send_command_to_sdk(linear, angular)
        except Exception as e:
            logger.error(f"Error sending action: {e}")

        return {
            ACTION_LINEAR_VEL: linear,
            ACTION_ANGULAR_VEL: angular,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        """断开与机器人的连接。

        停止机器人并关闭与 SDK 的连接。

        Raises:
            DeviceNotConnectedError: 如果机器人未连接
        """

        # 断开连接前先停止机器人
        try:
            self._send_command_to_sdk(0.0, 0.0)
        except Exception as e:
            logger.warning(f"Failed to stop robot during disconnect: {e}")

        self._is_connected = False
        logger.info(f"{self.name} disconnected")

    # 用于 SDK 通信的私有辅助方法

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """使用 v2 端点从 SDK 获取相机帧，带缓存回退。

        Returns:
            dict: 包含 'front' 和 'rear' 键的字典，内容为：
                - 当前帧（如果请求成功）
                - 缓存帧（如果请求失败但缓存存在）
                - 零数组（如果请求失败且尚无缓存）

        Note:
            使用 /v2/front 和 /v2/rear 端点，比 /screenshot 快 15 倍。
            图像为 base64 编码，调整为 640x480 大小，并从 BGR 转换为 RGB。
            如果请求失败，返回最后一次成功获取的帧（缓存）。
        """
        frames = {}

        # 获取前相机
        try:
            response = requests.get(f"{self.sdk_base_url}/v2/front", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                if "front_frame" in data and data["front_frame"]:
                    front_img = self._decode_base64_image(data["front_frame"])
                    if front_img is not None:
                        # 调整大小并将 BGR 转换为 RGB
                        front_img = cv2.resize(front_img, (640, 480))
                        front_rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
                        frames["front"] = front_rgb
                        # 缓存成功的帧
                        self._last_front_frame = front_rgb
        except Exception as e:
            logger.warning(f"Error fetching front camera: {e}")

        # 回退：使用缓存或零数组
        if "front" not in frames:
            if self._last_front_frame is not None:
                frames["front"] = self._last_front_frame
            else:
                frames["front"] = np.zeros((480, 640, 3), dtype=np.uint8)

        # 获取后相机
        try:
            response = requests.get(f"{self.sdk_base_url}/v2/rear", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                if "rear_frame" in data and data["rear_frame"]:
                    rear_img = self._decode_base64_image(data["rear_frame"])
                    if rear_img is not None:
                        # 调整大小并将 BGR 转换为 RGB
                        rear_img = cv2.resize(rear_img, (640, 480))
                        rear_rgb = cv2.cvtColor(rear_img, cv2.COLOR_BGR2RGB)
                        frames["rear"] = rear_rgb
                        # 缓存成功的帧
                        self._last_rear_frame = rear_rgb
        except Exception as e:
            logger.warning(f"Error fetching rear camera: {e}")

        # 回退：使用缓存或零数组
        if "rear" not in frames:
            if self._last_rear_frame is not None:
                frames["rear"] = self._last_rear_frame
            else:
                frames["rear"] = np.zeros((480, 640, 3), dtype=np.uint8)

        return frames

    def _decode_base64_image(self, base64_string: str) -> np.ndarray | None:
        """将 base64 字符串解码为图像。

        Args:
            base64_string: Base64 编码的图像字符串

        Returns:
            np.ndarray: BGR 格式的解码图像（OpenCV 默认），解码失败则返回 None
        """
        try:
            img_bytes = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img  # 以 BGR 格式返回（OpenCV 默认）
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None

    @staticmethod
    def _latest_sensor_reading(robot_data: dict, key: str, n_values: int) -> list[float]:
        """从 SDK 传感器数组中提取最新的传感器读数。

        SDK 返回如 ``accels``、``gyros``、``mags``、``rpms`` 这样的
        传感器数组，其中每个条目都是 ``[value_0, ..., value_n, timestamp]``。
        此辅助方法返回最后一个条目的前 *n_values* 个浮点数，
        当键缺失或数组为空时回退为零。
        """
        readings = robot_data.get(key)
        if readings and len(readings) > 0:
            latest = readings[-1]
            return [float(v) for v in latest[:n_values]]
        return [0.0] * n_values

    def _get_robot_data(self) -> dict:
        """从 SDK 获取机器人遥测数据。

        Returns:
            dict: 机器人遥测数据，包括电池、速度、朝向、GPS
                和传感器数组（accels、gyros、mags、rpms）：
                - 当前数据（如果请求成功）
                - 缓存数据（如果请求失败但缓存存在）
                - 默认值（如果请求失败且尚无缓存）

        Note:
            使用提供完整机器人状态的 /data 端点。
            如果请求失败，返回最后一次成功获取的数据（缓存）。
        """
        try:
            response = requests.get(f"{self.sdk_base_url}/data", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                # 缓存成功的数据
                self._last_robot_data = data
                return data
        except Exception as e:
            logger.warning(f"Error fetching robot data: {e}")

        # 回退：使用缓存或默认值
        if self._last_robot_data is not None:
            return self._last_robot_data

        # 返回带默认值的字典（仅在任何缓存存在之前的首次失败时使用）
        return {
            "speed": 0,
            "battery": 0,
            "orientation": 0,
            "latitude": 0.0,
            "longitude": 0.0,
            "gps_signal": 0,
            "signal_level": 0,
            "vibration": 0.0,
            "lamp": 0,
            "accels": [],
            "gyros": [],
            "mags": [],
            "rpms": [],
        }

    def _send_command_to_sdk(self, linear: float, angular: float, lamp: int = 0) -> bool:
        """向 SDK 发送控制命令。

        Args:
            linear: 线速度命令（-1 到 1）
            angular: 角速度命令（-1 到 1）
            lamp: 灯控制（0=关，1=开）

        Returns:
            bool: 命令发送成功返回 True，否则返回 False

        Note:
            使用 POST /control 端点。命令以 JSON 负载形式发送。
        """
        try:
            payload = {
                "command": {
                    "linear": linear,
                    "angular": angular,
                    "lamp": lamp,
                }
            }

            response = requests.post(
                f"{self.sdk_base_url}/control",
                json=payload,
                timeout=1.0,
            )

            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False
