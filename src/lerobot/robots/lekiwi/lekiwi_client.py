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

# TODO(aliberts, Steven, Pepijn): 使用 gRPC 调用代替 zmq？

import json
import logging
from functools import cached_property

import cv2
import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_lekiwi import LeKiwiClientConfig


class LeKiwiClient(Robot):
    config_class = LeKiwiClientConfig
    name = "lekiwi_client"

    def __init__(self, config: LeKiwiClientConfig):
        import zmq

        self._zmq = zmq
        super().__init__(config)
        self.config = config
        self.id = config.id
        self.robot_type = config.type

        depth_cameras = [name for name, cfg in config.cameras.items() if getattr(cfg, "use_depth", False)]
        if depth_cameras:
            raise NotImplementedError(
                f"Depth cameras are not supported on LeKiwi (got depth-enabled cameras: {depth_cameras}). "
                "The host/client transport only carries color frames."
            )

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations

        self.teleop_keys = config.teleop_keys

        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None

        self.last_frames = {}

        self.last_remote_state = {}

        # 定义三个速度档位和当前索引
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # 慢
            {"xy": 0.2, "theta": 60},  # 中
            {"xy": 0.3, "theta": 90},  # 快
        ]
        self.speed_index = 0  # 从慢速开始

        self._is_connected = False
        self.logs = {}

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "arm_shoulder_pan.pos",
                "arm_shoulder_lift.pos",
                "arm_elbow_flex.pos",
                "arm_wrist_flex.pos",
                "arm_wrist_roll.pos",
                "arm_gripper.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @cached_property
    def _state_order(self) -> tuple[str, ...]:
        return tuple(self._state_ft.keys())

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        """与远程移动机器人建立 ZMQ 套接字"""

        zmq = self._zmq
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        zmq_cmd_locator = f"tcp://{self.remote_ip}:{self.port_zmq_cmd}"
        self.zmq_cmd_socket.connect(zmq_cmd_locator)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        zmq_observations_locator = f"tcp://{self.remote_ip}:{self.port_zmq_observations}"
        self.zmq_observation_socket.connect(zmq_observations_locator)
        # CONFLATE 不支持多部分消息；小的接收队列加上
        # 现有的排空到最新消息的循环，保持了只保留最新值的语义。
        self.zmq_observation_socket.setsockopt(zmq.RCVHWM, 2)

        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.connect_timeout_s * 1000))
        if self.zmq_observation_socket not in socks or socks[self.zmq_observation_socket] != zmq.POLLIN:
            raise DeviceNotConnectedError("Timeout waiting for LeKiwi Host to connect expired.")

        self._is_connected = True

    def calibrate(self) -> None:
        pass

    def _poll_and_get_latest_message(self) -> list[bytes] | None:
        """在限定时间内轮询 ZMQ 套接字，并返回最新消息的帧。"""
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)

        try:
            socks = dict(poller.poll(self.polling_timeout_ms))
        except zmq.ZMQError as e:
            logging.error(f"ZMQ polling error: {e}")
            return None

        if self.zmq_observation_socket not in socks:
            logging.info("No new data available within timeout.")
            return None

        last_msg = None
        while True:
            try:
                msg = self.zmq_observation_socket.recv_multipart(zmq.NOBLOCK)
                last_msg = msg
            except zmq.Again:
                break

        if last_msg is None:
            logging.warning("Poller indicated data, but failed to retrieve message.")

        return last_msg

    def _parse_observation(self, frames: list[bytes]) -> RobotObservation | None:
        """解析多部分观测：JSON 头部 + 每个相机一个原始 JPEG 帧。"""
        try:
            header = json.loads(frames[0])
            cam_names = header.pop("_cams")
            observation: RobotObservation = header
            for cam_name, jpeg in zip(cam_names, frames[1:], strict=True):
                observation[cam_name] = jpeg
            return observation
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.error(f"Error decoding observation: {e}")
            return None

    def _decode_image(self, jpeg: bytes) -> np.ndarray | None:
        """将原始 JPEG 缓冲区解码为 OpenCV 图像。"""
        if not jpeg:
            return None
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            logging.warning("cv2.imdecode returned None for an image.")
        return frame

    def _remote_state_from_obs(
        self, observation: RobotObservation
    ) -> tuple[dict[str, np.ndarray], RobotObservation]:
        """从解析后的观测中提取帧和状态。"""

        flat_state = {key: observation.get(key, 0.0) for key in self._state_order}

        state_vec = np.array([flat_state[key] for key in self._state_order], dtype=np.float32)

        obs_dict: RobotObservation = {**flat_state, OBS_STATE: state_vec}

        # 解码图像
        current_frames: dict[str, np.ndarray] = {}
        for cam_name, jpeg in observation.items():
            if cam_name not in self._cameras_ft:
                continue
            frame = self._decode_image(jpeg)
            if frame is not None:
                current_frames[cam_name] = frame

        return current_frames, obs_dict

    def _get_data(self) -> tuple[dict[str, np.ndarray], RobotObservation]:
        """
        轮询视频套接字以获取最新的观测数据。

        尝试在短超时内获取并解码最新消息。
        如果成功，更新并返回新的帧、速度和机械臂状态。
        如果没有新数据到达或解码失败，返回最后已知的值。
        """

        # 1. 从套接字获取最新消息的帧
        latest_frames = self._poll_and_get_latest_message()

        # 2. 如果没有消息，返回缓存数据
        if latest_frames is None:
            return self.last_frames, self.last_remote_state

        # 3. 解析多部分消息
        observation = self._parse_observation(latest_frames)

        # 4. 如果 JSON 解析失败，返回缓存数据
        if observation is None:
            return self.last_frames, self.last_remote_state

        # 5. 处理有效的观测数据
        try:
            new_frames, new_state = self._remote_state_from_obs(observation)
        except Exception as e:
            logging.error(f"Error processing observation data, serving last observation: {e}")
            return self.last_frames, self.last_remote_state

        self.last_frames = new_frames
        self.last_remote_state = new_state

        return new_frames, new_state

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        从远程机器人捕获观测：当前从动机械臂位置、
        当前车轮速度（转换为机体系速度：x、y、theta），
        以及相机帧。通过 ZMQ 接收，并转换为机体系速度
        """

        frames, obs_dict = self._get_data()

        # 遍历每个已配置的相机
        for cam_name, frame in frames.items():
            if frame is None:
                logging.warning("Frame is None")
                frame = np.zeros((640, 480, 3), dtype=np.uint8)
            obs_dict[cam_name] = frame

        return obs_dict

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        # 速度控制
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # 例如 0.1、0.25 或 0.4
        theta_speed = speed_setting["theta"]  # 例如 30、60 或 90

        x_cmd = 0.0  # m/s 前进/后退
        y_cmd = 0.0  # m/s 横向
        theta_cmd = 0.0  # deg/s 旋转

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed
        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }

    def configure(self):
        pass

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """命令 lekiwi 移动到目标关节配置。转换到电机空间 + 通过 ZMQ 发送

        Args:
            action (RobotAction): 包含电机目标位置的数组。
        Raises:
            RobotDeviceNotConnectedError: 如果机器人未连接。

        Returns:
            np.ndarray: 发送给电机的动作，可能已被裁剪。
        """

        # 动作值可能是 torch 张量（例如从数据集回放）或 numpy
        # 标量；json.dumps 只能序列化 Python 原生类型，因此在发送前
        # 将每个值强制转换为普通浮点数。
        action = {key: float(value) for key, value in action.items()}
        self.zmq_cmd_socket.send_string(json.dumps(action))  # 动作位于电机空间

        # TODO(Steven): 当可以录制非 numpy 数组值时，移除 np 转换
        actions = np.array([action.get(k, 0.0) for k in self._state_order], dtype=np.float32)

        action_sent = {key: actions[i] for i, key in enumerate(self._state_order)}
        action_sent[ACTION] = actions
        return action_sent

    @check_if_not_connected
    def disconnect(self):
        """清理 ZMQ 通信"""

        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        self._is_connected = False
