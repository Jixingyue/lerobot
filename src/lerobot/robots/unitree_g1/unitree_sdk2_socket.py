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

import base64
import json
from typing import Any

import zmq

from .config_unitree_g1 import UnitreeG1Config

# 模块级 ZMQ 状态，与 Unitree SDK 的全局 ChannelFactory 单例相对应。
# 每个进程仅支持一个机器人连接。
_ctx: zmq.Context | None = None
_lowcmd_sock: zmq.Socket | None = None
_lowstate_sock: zmq.Socket | None = None

LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001

# DDS 主题名遵循 Unitree SDK 的命名约定
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"


class LowStateMsg:
    """
    模仿 Unitree SDK LowState_ 消息结构的包装类。

    从反序列化后的 JSON 数据重建消息，以兼容那些期望接收 SDK 消息对象的
    现有代码。
    """

    class MotorState:
        """单个关节的电机状态数据。"""

        def __init__(self, data: dict[str, Any]) -> None:
            self.q: float = data.get("q", 0.0)
            self.dq: float = data.get("dq", 0.0)
            self.tau_est: float = data.get("tau_est", 0.0)
            self.temperature: float = data.get("temperature", 0.0)

    class IMUState:
        """IMU 传感器数据。"""

        def __init__(self, data: dict[str, Any]) -> None:
            self.quaternion: list[float] = data.get("quaternion", [1.0, 0.0, 0.0, 0.0])
            self.gyroscope: list[float] = data.get("gyroscope", [0.0, 0.0, 0.0])
            self.accelerometer: list[float] = data.get("accelerometer", [0.0, 0.0, 0.0])
            self.rpy: list[float] = data.get("rpy", [0.0, 0.0, 0.0])
            self.temperature: float = data.get("temperature", 0.0)

    def __init__(self, data: dict[str, Any]) -> None:
        """根据反序列化后的 JSON 数据初始化。"""
        self.motor_state = [self.MotorState(m) for m in data.get("motor_state", [])]
        self.imu_state = self.IMUState(data.get("imu_state", {}))
        # 解码 base64 编码的 wireless_remote 字节
        wireless_b64 = data.get("wireless_remote", "")
        self.wireless_remote: bytes = base64.b64decode(wireless_b64) if wireless_b64 else b""
        self.mode_machine: int = data.get("mode_machine", 0)


def lowcmd_to_dict(topic: str, msg: Any) -> dict[str, Any]:
    """将 LowCmd 消息转换为可 JSON 序列化的字典。"""
    motor_cmds = []
    # 遍历消息中的所有电机命令
    for i in range(len(msg.motor_cmd)):
        motor_cmds.append(
            {
                "mode": int(msg.motor_cmd[i].mode),
                "q": float(msg.motor_cmd[i].q),
                "dq": float(msg.motor_cmd[i].dq),
                "kp": float(msg.motor_cmd[i].kp),
                "kd": float(msg.motor_cmd[i].kd),
                "tau": float(msg.motor_cmd[i].tau),
            }
        )

    return {
        "topic": topic,
        "data": {
            "mode_pr": int(msg.mode_pr),
            "mode_machine": int(msg.mode_machine),
            "motor_cmd": motor_cmds,
        },
    }


def ChannelFactoryInitialize(domain_id: int = 0, config: Any = None) -> None:  # noqa: N802
    """
    初始化用于机器人通信的 ZMQ 套接字。

    该函数模仿 Unitree SDK 的 ChannelFactoryInitialize，但使用 ZMQ 套接字
    连接到机器人服务器桥接，而不是 DDS。

    Args:
        domain_id: 忽略（为与 Unitree SDK 的 API 保持兼容而保留）
        config: 包含 robot_ip 的 UnitreeG1Config 实例
    """
    global _ctx, _lowcmd_sock, _lowstate_sock

    # 读取套接字配置
    if config is None:
        config = UnitreeG1Config()
    robot_ip = config.robot_ip

    ctx = zmq.Context.instance()
    _ctx = ctx

    # lowcmd：发送机器人命令
    lowcmd_sock = ctx.socket(zmq.PUSH)
    lowcmd_sock.setsockopt(zmq.CONFLATE, 1)  # 只保留最后一条消息
    lowcmd_sock.connect(f"tcp://{robot_ip}:{LOWCMD_PORT}")
    _lowcmd_sock = lowcmd_sock

    # lowstate：接收机器人观测
    lowstate_sock = ctx.socket(zmq.SUB)
    lowstate_sock.setsockopt(zmq.CONFLATE, 1)  # 只保留最后一条消息
    lowstate_sock.connect(f"tcp://{robot_ip}:{LOWSTATE_PORT}")
    lowstate_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    _lowstate_sock = lowstate_sock


class ChannelPublisher:
    """基于 ZMQ 的发布者，用于向机器人服务器发送命令。"""

    def __init__(self, topic: str, msg_type: type) -> None:
        self.topic = topic
        self.msg_type = msg_type

    def Init(self) -> None:  # noqa: N802
        """初始化发布者（对 ZMQ 而言为空操作）。"""
        pass

    def Write(self, msg: Any) -> None:  # noqa: N802
        """序列化命令消息并发送给机器人。"""
        if _lowcmd_sock is None:
            raise RuntimeError("ChannelFactoryInitialize must be called first")

        payload = json.dumps(lowcmd_to_dict(self.topic, msg)).encode("utf-8")
        _lowcmd_sock.send(payload)


class ChannelSubscriber:
    """基于 ZMQ 的订阅者，用于从机器人服务器接收状态。"""

    def __init__(self, topic: str, msg_type: type) -> None:
        self.topic = topic
        self.msg_type = msg_type

    def Init(self) -> None:  # noqa: N802
        """初始化订阅者（对 ZMQ 而言为空操作）。"""
        pass

    def Read(self) -> LowStateMsg:  # noqa: N802
        """接收并反序列化来自机器人的状态消息。"""
        if _lowstate_sock is None:
            raise RuntimeError("ChannelFactoryInitialize must be called first")

        payload = _lowstate_sock.recv()
        msg_dict = json.loads(payload.decode("utf-8"))
        return LowStateMsg(msg_dict.get("data", {}))
