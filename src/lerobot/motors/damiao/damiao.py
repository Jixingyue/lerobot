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

# Portions of this file are derived from DM_Control_Python by cmjang.
# Licensed under the MIT License; see `LICENSE` for the full text:
# https://github.com/cmjang/DM_Control_Python

import logging
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, TypedDict

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _can_available, require_package

if TYPE_CHECKING or _can_available:
    import can
else:

    class can:  # noqa: N801
        Message = object
        interface = None


import numpy as np

from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..motors_bus import Motor, MotorCalibration, MotorsBusBase, NameOrID, Value
from .tables import (
    AVAILABLE_BAUDRATES,
    CAN_CMD_DISABLE,
    CAN_CMD_ENABLE,
    CAN_CMD_REFRESH,
    CAN_CMD_SET_ZERO,
    CAN_PARAM_ID,
    DEFAULT_BAUDRATE,
    DEFAULT_TIMEOUT_MS,
    MIT_KD_RANGE,
    MIT_KP_RANGE,
    MOTOR_LIMIT_PARAMS,
    MotorType,
)

logger = logging.getLogger(__name__)


LONG_TIMEOUT_SEC = 0.1
MEDIUM_TIMEOUT_SEC = 0.01
SHORT_TIMEOUT_SEC = 0.001
PRECISE_TIMEOUT_SEC = 0.0001


class MotorState(TypedDict):
    position: float
    velocity: float
    torque: float
    temp_mos: float
    temp_rotor: float


class DamiaoMotorsBus(MotorsBusBase):
    """
    使用 CAN 总线通信的 MotorsBus 的 Damiao 实现。

    本类使用 python-can 与 Damiao 电机进行 CAN 总线通信。
    更多信息请参阅：
    - python-can 文档：https://python-can.readthedocs.io/en/stable/
    - Seedstudio 文档：https://wiki.seeedstudio.com/damiao_series/
    - DM_Control_Python 仓库：https://github.com/cmjang/DM_Control_Python
    """

    # CAN 专用设置
    available_baudrates = deepcopy(AVAILABLE_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        can_interface: str = "auto",
        use_can_fd: bool = True,
        bitrate: int = 1000000,
        data_bitrate: int | None = 5000000,
    ):
        """
        初始化 Damiao 电机总线。

        Args:
            port: CAN 接口名称（例如 Linux 下为 "can0"，macOS 下为 "/dev/cu.usbmodem*"）
            motors: 电机名称到 Motor 对象的映射字典
            calibration: 可选的校准数据
            can_interface: CAN 接口类型 - "auto"（默认）、"socketcan"（Linux）或 "slcan"（macOS/serial）
            use_can_fd: 是否使用 CAN FD 模式（OpenArms 默认为 True）
            bitrate: 标称比特率，单位 bps（默认：1000000 = 1 Mbps）
            data_bitrate: CAN FD 的数据比特率，单位 bps（默认：5000000 = 5 Mbps），当 use_can_fd 为 False 时被忽略
        """
        require_package("python-can", extra="damiao", import_name="can")
        super().__init__(port, motors, calibration)
        self.port = port
        self.can_interface = can_interface
        self.use_can_fd = use_can_fd
        self.bitrate = bitrate
        self.data_bitrate = data_bitrate
        self.canbus: can.interface.Bus | None = None
        self._is_connected = False

        # 将电机名称映射到 CAN ID
        self._motor_can_ids: dict[str, int] = {}
        self._recv_id_to_motor: dict[int, str] = {}
        self._motor_types: dict[str, MotorType] = {}

        for name, motor in self.motors.items():
            if motor.motor_type_str is None:
                raise ValueError(f"Motor '{name}' is missing required 'motor_type'")
            self._motor_types[name] = getattr(MotorType, motor.motor_type_str.upper().replace("-", "_"))

            # 将 recv_id 映射到电机名称，用于过滤响应
            if motor.recv_id is not None:
                self._recv_id_to_motor[motor.recv_id] = name

        # 状态缓存，用于安全地处理丢包
        self._last_known_states: dict[str, MotorState] = {
            name: {
                "position": 0.0,
                "velocity": 0.0,
                "torque": 0.0,
                "temp_mos": 0.0,
                "temp_rotor": 0.0,
            }
            for name in self.motors
        }

        # 动态增益存储
        # 默认值：Kp=10.0（刚度），Kd=0.5（阻尼）
        self._gains: dict[str, dict[str, float]] = {name: {"kp": 10.0, "kd": 0.5} for name in self.motors}

    @property
    def is_connected(self) -> bool:
        """检查 CAN 总线是否已连接。"""
        return self._is_connected and self.canbus is not None

    @check_if_already_connected
    def connect(self, handshake: bool = True) -> None:
        """
        打开 CAN 总线并初始化通信。

        Args:
            handshake: 如果为 True，则 ping 所有电机以确认它们存在
        """

        try:
            # 根据端口名称自动检测接口类型
            if self.can_interface == "auto":
                if self.port.startswith("/dev/"):
                    self.can_interface = "slcan"
                    logger.info(f"Auto-detected slcan interface for port {self.port}")
                else:
                    self.can_interface = "socketcan"
                    logger.info(f"Auto-detected socketcan interface for port {self.port}")

            # 连接到 CAN 总线
            kwargs = {
                "channel": self.port,
                "bitrate": self.bitrate,
                "interface": self.can_interface,
            }

            if self.can_interface == "socketcan" and self.use_can_fd and self.data_bitrate is not None:
                kwargs.update({"data_bitrate": self.data_bitrate, "fd": True})
                logger.info(
                    f"Connected to {self.port} with CAN FD (bitrate={self.bitrate}, data_bitrate={self.data_bitrate})"
                )
            else:
                logger.info(f"Connected to {self.port} with {self.can_interface} (bitrate={self.bitrate})")

            self.canbus = can.interface.Bus(**kwargs)
            self._is_connected = True

            if handshake:
                self._handshake()

            logger.debug(f"{self.__class__.__name__} connected via {self.can_interface}.")
        except Exception as e:
            self._is_connected = False
            raise ConnectionError(f"Failed to connect to CAN bus: {e}") from e

    def _handshake(self) -> None:
        """
        验证所有电机是否存在，并填充初始状态缓存。
        如果有任何电机未响应，则抛出 ConnectionError。
        """
        logger.info("Starting handshake with motors...")

        # 排空所有待处理的消息
        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        while self.canbus.recv(timeout=0.01):
            pass

        missing_motors = []
        for motor_name in self.motors:
            motor_id = self._get_motor_id(motor_name)
            recv_id = self._get_motor_recv_id(motor_name)

            # 发送使能命令
            data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, CAN_CMD_ENABLE]
            msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd)
            self.canbus.send(msg)

            # 使用较长的超时时间等待响应
            response = None
            start_time = time.time()
            while time.time() - start_time < 0.1:
                response = self.canbus.recv(timeout=0.1)
                if response and response.arbitration_id == recv_id:
                    break
                response = None

            if response is None:
                missing_motors.append(motor_name)
            else:
                self._process_response(motor_name, msg)
            time.sleep(MEDIUM_TIMEOUT_SEC)

        if missing_motors:
            raise ConnectionError(
                f"Handshake failed. The following motors did not respond: {missing_motors}. "
                "Check power (24V) and CAN wiring."
            )
        logger.info("Handshake successful. All motors ready.")

    @check_if_not_connected
    def disconnect(self, disable_torque: bool = True) -> None:
        """
        关闭 CAN 总线连接。

        Args:
            disable_torque: 如果为 True，则在断开连接前禁用所有电机的力矩
        """

        if disable_torque:
            try:
                self.disable_torque()
            except Exception as e:
                logger.warning(f"Failed to disable torque during disconnect: {e}")

        if self.canbus:
            self.canbus.shutdown()
            self.canbus = None
        self._is_connected = False
        logger.debug(f"{self.__class__.__name__} disconnected.")

    def configure_motors(self) -> None:
        """使用默认设置配置所有电机。"""
        # Damiao 电机在 MIT 模式下不需要太多配置
        # 只需确保它们已使能即可
        for motor in self.motors:
            self._send_simple_command(motor, CAN_CMD_ENABLE)
            time.sleep(MEDIUM_TIMEOUT_SEC)

    def _send_simple_command(self, motor: NameOrID, command_byte: int) -> None:
        """辅助方法，用于发送简单的 8 字节命令（Enable、Disable、Zero）。"""
        motor_id = self._get_motor_id(motor)
        motor_name = self._get_motor_name(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [0xFF] * 7 + [command_byte]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd)

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        self.canbus.send(msg)
        if msg := self._recv_motor_response(expected_recv_id=recv_id):
            self._process_response(motor_name, msg)
        else:
            logger.debug(f"No response from {motor_name} after command 0x{command_byte:02X}")

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """启用所选电机的力矩。"""
        target_motors = self._get_motors_list(motors)
        for motor in target_motors:
            for _ in range(num_retry + 1):
                try:
                    self._send_simple_command(motor, CAN_CMD_ENABLE)
                    break
                except Exception as e:
                    if _ == num_retry:
                        raise e
                    time.sleep(MEDIUM_TIMEOUT_SEC)

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """禁用所选电机的力矩。"""
        target_motors = self._get_motors_list(motors)
        for motor in target_motors:
            for _ in range(num_retry + 1):
                try:
                    self._send_simple_command(motor, CAN_CMD_DISABLE)
                    break
                except Exception as e:
                    if _ == num_retry:
                        raise e
                    time.sleep(MEDIUM_TIMEOUT_SEC)

    @contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):
        """
        保证力矩会被重新启用的上下文管理器。

        此辅助方法在临时禁用力矩以配置电机时很有用。
        """
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def set_zero_position(self, motors: str | list[str] | None = None) -> None:
        """将所选电机的当前位置设为零点。"""
        target_motors = self._get_motors_list(motors)
        for motor in target_motors:
            self._send_simple_command(motor, CAN_CMD_SET_ZERO)
            time.sleep(MEDIUM_TIMEOUT_SEC)

    def _refresh_motor(self, motor: NameOrID) -> can.Message | None:
        """刷新电机状态并返回响应。"""
        motor_id = self._get_motor_id(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
        msg = can.Message(arbitration_id=CAN_PARAM_ID, data=data, is_extended_id=False, is_fd=self.use_can_fd)

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        self.canbus.send(msg)
        return self._recv_motor_response(expected_recv_id=recv_id)

    def _recv_motor_response(
        self, expected_recv_id: int | None = None, timeout: float = 0.001
    ) -> can.Message | None:
        """
        接收来自电机的响应。

        Args:
            expected_recv_id: 如果提供，则只返回来自该 CAN ID 的消息
            timeout: 超时时间，单位秒（默认：1ms，用于高速操作）
        Returns:
            如果收到则返回 CAN 消息，否则返回 None
        """

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        try:
            start_time = time.time()
            messages_seen = []
            while time.time() - start_time < timeout:
                msg = self.canbus.recv(timeout=PRECISE_TIMEOUT_SEC)
                if msg:
                    messages_seen.append(f"0x{msg.arbitration_id:02X}")
                    if expected_recv_id is None or msg.arbitration_id == expected_recv_id:
                        return msg
                    logger.debug(
                        f"Ignoring message from 0x{msg.arbitration_id:02X}, expected 0x{expected_recv_id:02X}"
                    )

            if logger.isEnabledFor(logging.DEBUG):
                if messages_seen:
                    logger.debug(
                        f"Received {len(messages_seen)} msgs from {set(messages_seen)}, expected 0x{expected_recv_id:02X}"
                    )
                else:
                    logger.debug(f"No CAN messages received (expected 0x{expected_recv_id:02X})")
        except Exception as e:
            logger.debug(f"Failed to receive CAN message: {e}")
        return None

    def _recv_all_responses(
        self, expected_recv_ids: list[int], timeout: float = 0.002
    ) -> dict[int, can.Message]:
        """
        高效地一次性接收多个电机的响应。
        使用 OpenArms 模式：在超时时间内收集所有可用的消息。

        Args:
            expected_recv_ids: 期望收到响应的 CAN ID 列表
            timeout: 总超时时间，单位秒（默认：2ms）

        Returns:
            recv_id 到 CAN 消息的映射字典
        """
        responses: dict[int, can.Message] = {}
        expected_set = set(expected_recv_ids)
        start_time = time.time()

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        try:
            while len(responses) < len(expected_recv_ids) and (time.time() - start_time) < timeout:
                # 100us 轮询超时
                msg = self.canbus.recv(timeout=PRECISE_TIMEOUT_SEC)
                if msg and msg.arbitration_id in expected_set:
                    responses[msg.arbitration_id] = msg
                    if len(responses) == len(expected_recv_ids):
                        break
        except Exception as e:
            logger.debug(f"Error receiving responses: {e}")

        return responses

    def _encode_mit_packet(
        self,
        motor_type: MotorType,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float,
        torque: float,
    ) -> list[int]:
        """辅助方法，用于将控制参数编码为 MIT 模式的 8 字节数据。"""
        # 将角度转换为弧度
        position_rad = np.radians(position_degrees)
        velocity_rad_per_sec = np.radians(velocity_deg_per_sec)

        # 获取电机限位参数
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]

        # 编码参数
        kp_uint = self._float_to_uint(kp, *MIT_KP_RANGE, 12)
        kd_uint = self._float_to_uint(kd, *MIT_KD_RANGE, 12)
        q_uint = self._float_to_uint(position_rad, -pmax, pmax, 16)
        dq_uint = self._float_to_uint(velocity_rad_per_sec, -vmax, vmax, 12)
        tau_uint = self._float_to_uint(torque, -tmax, tmax, 12)

        # 打包数据
        data = [0] * 8
        data[0] = (q_uint >> 8) & 0xFF
        data[1] = q_uint & 0xFF
        data[2] = dq_uint >> 4
        data[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)
        data[4] = kp_uint & 0xFF
        data[5] = kd_uint >> 4
        data[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF)
        data[7] = tau_uint & 0xFF
        return data

    def _mit_control(
        self,
        motor: NameOrID,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float,
        torque: float,
    ) -> None:
        """向电机发送 MIT 控制命令。"""
        motor_id = self._get_motor_id(motor)
        motor_name = self._get_motor_name(motor)
        motor_type = self._motor_types[motor_name]

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        data = self._encode_mit_packet(motor_type, kp, kd, position_degrees, velocity_deg_per_sec, torque)
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd)
        self.canbus.send(msg)

        recv_id = self._get_motor_recv_id(motor)
        if msg := self._recv_motor_response(expected_recv_id=recv_id):
            self._process_response(motor_name, msg)
        else:
            logger.debug(f"No response from {motor_name} after MIT control command")

    def _mit_control_batch(
        self,
        commands: dict[NameOrID, tuple[float, float, float, float, float]],
    ) -> None:
        """
        批量向多个电机发送 MIT 控制命令。
        先发送所有命令，然后收集响应。

        Args:
            commands: 电机名称/ID 到 (kp, kd, position_deg, velocity_deg/s, torque) 的映射字典
                     示例：{'joint_1': (10.0, 0.5, 45.0, 0.0, 0.0), ...}
        """
        if not commands:
            return

        recv_id_to_motor: dict[int, str] = {}

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        # 第 1 步：发送所有 MIT 控制命令
        for motor, (kp, kd, position_degrees, velocity_deg_per_sec, torque) in commands.items():
            motor_id = self._get_motor_id(motor)
            motor_name = self._get_motor_name(motor)
            motor_type = self._motor_types[motor_name]

            data = self._encode_mit_packet(motor_type, kp, kd, position_degrees, velocity_deg_per_sec, torque)
            msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd)
            self.canbus.send(msg)

            recv_id_to_motor[self._get_motor_recv_id(motor)] = motor_name

        # 第 2 步：收集响应并更新状态缓存
        responses = self._recv_all_responses(list(recv_id_to_motor.keys()), timeout=SHORT_TIMEOUT_SEC)
        for recv_id, motor_name in recv_id_to_motor.items():
            if msg := responses.get(recv_id):
                self._process_response(motor_name, msg)

    def _float_to_uint(self, x: float, x_min: float, x_max: float, bits: int) -> int:
        """将浮点数转换为无符号整数，用于 CAN 传输。"""
        x = max(x_min, min(x_max, x))  # 限幅到有效范围
        span = x_max - x_min
        data_norm = (x - x_min) / span
        return int(data_norm * ((1 << bits) - 1))

    def _uint_to_float(self, x: int, x_min: float, x_max: float, bits: int) -> float:
        """将来自 CAN 的无符号整数转换为浮点数。"""
        span = x_max - x_min
        data_norm = float(x) / ((1 << bits) - 1)
        return data_norm * span + x_min

    def _decode_motor_state(
        self, data: bytearray | bytes, motor_type: MotorType
    ) -> tuple[float, float, float, int, int]:
        """
        从 CAN 数据解码电机状态。
        返回：(position_deg, velocity_deg_s, torque, temp_mos, temp_rotor)
        """
        if len(data) < 8:
            raise ValueError("Invalid motor state data")

        # 提取编码值
        q_uint = (data[1] << 8) | data[2]
        dq_uint = (data[3] << 4) | (data[4] >> 4)
        tau_uint = ((data[4] & 0x0F) << 8) | data[5]
        t_mos = data[6]
        t_rotor = data[7]

        # 获取电机限位参数
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]

        # 解码为物理量
        position_rad = self._uint_to_float(q_uint, -pmax, pmax, 16)
        velocity_rad_per_sec = self._uint_to_float(dq_uint, -vmax, vmax, 12)
        torque = self._uint_to_float(tau_uint, -tmax, tmax, 12)

        return np.degrees(position_rad), np.degrees(velocity_rad_per_sec), torque, t_mos, t_rotor

    def _process_response(self, motor: str, msg: can.Message) -> None:
        """解码消息并更新电机状态缓存。"""
        try:
            motor_type = self._motor_types[motor]
            pos, vel, torque, t_mos, t_rotor = self._decode_motor_state(msg.data, motor_type)

            self._last_known_states[motor] = {
                "position": pos,
                "velocity": vel,
                "torque": torque,
                "temp_mos": float(t_mos),
                "temp_rotor": float(t_rotor),
            }
        except Exception as e:
            logger.warning(f"Failed to decode response from {motor}: {e}")

    @check_if_not_connected
    def read(self, data_name: str, motor: str) -> Value:
        """从单个电机读取一个值。位置始终以度为单位。"""

        # 刷新电机以获取最新状态
        msg = self._refresh_motor(motor)
        if msg is None:
            motor_id = self._get_motor_id(motor)
            recv_id = self._get_motor_recv_id(motor)
            raise ConnectionError(
                f"No response from motor '{motor}' (send ID: 0x{motor_id:02X}, recv ID: 0x{recv_id:02X}). "
                f"Check that: 1) Motor is powered (24V), 2) CAN wiring is correct, "
                f"3) Motor IDs are configured correctly using Damiao Debugging Tools"
            )

        self._process_response(motor, msg)
        return self._get_cached_value(motor, data_name)

    def _get_cached_value(self, motor: str, data_name: str) -> Value:
        """从缓存中获取指定的值。"""
        state = self._last_known_states[motor]
        mapping: dict[str, Any] = {
            "Present_Position": state["position"],
            "Present_Velocity": state["velocity"],
            "Present_Torque": state["torque"],
            "Temperature_MOS": state["temp_mos"],
            "Temperature_Rotor": state["temp_rotor"],
        }
        if data_name not in mapping:
            raise ValueError(f"Unknown data_name: {data_name}")
        return mapping[data_name]

    @check_if_not_connected
    def write(
        self,
        data_name: str,
        motor: str,
        value: Value,
    ) -> None:
        """
        向单个电机写入一个值。位置始终以度为单位。
        可写入 'Goal_Position'、'Kp' 或 'Kd'。
        """

        if data_name in ("Kp", "Kd"):
            self._gains[motor][data_name.lower()] = float(value)
        elif data_name == "Goal_Position":
            kp = self._gains[motor]["kp"]
            kd = self._gains[motor]["kd"]
            self._mit_control(motor, kp, kd, float(value), 0.0, 0.0)
        else:
            raise ValueError(f"Writing {data_name} not supported in MIT mode")

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
    ) -> dict[str, Value]:
        """
        同时从多个电机读取相同的值。
        """
        target_motors = self._get_motors_list(motors)
        self._batch_refresh(target_motors)

        result = {}
        for motor in target_motors:
            result[motor] = self._get_cached_value(motor, data_name)
        return result

    def sync_read_all_states(
        self,
        motors: str | list[str] | None = None,
        *,
        num_retry: int = 0,
    ) -> dict[str, MotorState]:
        """
        在一次刷新周期内从多个电机读取全部电机状态（position、velocity、torque）。

        Returns:
            电机名称到状态字典的映射字典，状态字典的键为：'position'、'velocity'、'torque'
            示例：{'joint_1': {'position': 45.2, 'velocity': 1.3, 'torque': 0.5}, ...}
        """
        target_motors = self._get_motors_list(motors)
        self._batch_refresh(target_motors)

        result = {}
        for motor in target_motors:
            result[motor] = self._last_known_states[motor].copy()
        return result

    def _batch_refresh(self, motors: list[str]) -> None:
        """内部辅助方法，用于刷新一组电机并更新缓存。"""

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        # 发送刷新命令
        for motor in motors:
            motor_id = self._get_motor_id(motor)
            data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
            msg = can.Message(
                arbitration_id=CAN_PARAM_ID, data=data, is_extended_id=False, is_fd=self.use_can_fd
            )
            self.canbus.send(msg)

        # 收集响应
        expected_recv_ids = [self._get_motor_recv_id(m) for m in motors]
        responses = self._recv_all_responses(expected_recv_ids, timeout=MEDIUM_TIMEOUT_SEC)

        # 更新缓存
        for motor in motors:
            recv_id = self._get_motor_recv_id(motor)
            msg = responses.get(recv_id)
            if msg:
                self._process_response(motor, msg)
            else:
                logger.warning(f"Packet drop: {motor} (ID: 0x{recv_id:02X}). Using last known state.")

    @check_if_not_connected
    def sync_write(self, data_name: str, values: dict[str, Value]) -> None:
        """
        同时向多个电机写入值。位置始终以度为单位。
        """

        if data_name in ("Kp", "Kd"):
            key = data_name.lower()
            for motor, val in values.items():
                self._gains[motor][key] = float(val)

        elif data_name == "Goal_Position":
            # 第 1 步：发送所有 MIT 控制命令
            recv_id_to_motor: dict[int, str] = {}
            if self.canbus is None:
                raise RuntimeError("CAN bus is not initialized.")
            for motor, value_degrees in values.items():
                motor_id = self._get_motor_id(motor)
                motor_name = self._get_motor_name(motor)
                motor_type = self._motor_types[motor_name]

                kp = self._gains[motor]["kp"]
                kd = self._gains[motor]["kd"]

                data = self._encode_mit_packet(motor_type, kp, kd, float(value_degrees), 0.0, 0.0)
                msg = can.Message(
                    arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd
                )
                self.canbus.send(msg)
                precise_sleep(PRECISE_TIMEOUT_SEC)

                recv_id_to_motor[self._get_motor_recv_id(motor)] = motor_name

            # 第 2 步：收集响应并更新状态缓存
            responses = self._recv_all_responses(list(recv_id_to_motor.keys()), timeout=MEDIUM_TIMEOUT_SEC)
            for recv_id, motor_name in recv_id_to_motor.items():
                if msg := responses.get(recv_id):
                    self._process_response(motor_name, msg)
        else:
            # 回退到逐个写入
            for motor, value in values.items():
                self.write(data_name, motor, value)

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """从电机读取校准数据。"""
        # Damiao 电机不在内部存储校准数据
        # 返回已有的校准数据或空字典
        return self.calibration if self.calibration else {}

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """向电机写入校准数据。"""
        # Damiao 电机不在内部存储校准数据
        # 只在内存中缓存
        if cache:
            self.calibration = calibration_dict

    def record_ranges_of_motion(
        self,
        motors: str | list[str] | None = None,
        display_values: bool = True,
    ) -> tuple[dict[str, Value], dict[str, Value]]:
        """
        以交互方式记录每个电机的最小/最大值（单位：度）。

        在力矩禁用的状态下手动移动关节，该方法会实时显示当前位置。
        按 Enter 键结束。
        """
        target_motors = self._get_motors_list(motors)

        self.disable_torque(target_motors)
        time.sleep(LONG_TIMEOUT_SEC)

        start_positions = self.sync_read("Present_Position", target_motors)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        print("\nMove joints through their full range of motion. Press ENTER when done.")
        user_pressed_enter = False

        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", target_motors)

            for motor in target_motors:
                if motor in positions:
                    mins[motor] = min(positions[motor], mins.get(motor, positions[motor]))
                    maxes[motor] = max(positions[motor], maxes.get(motor, positions[motor]))

            if display_values:
                print("\n" + "=" * 50)
                print(f"{'MOTOR':<20} | {'MIN (deg)':>12} | {'POS (deg)':>12} | {'MAX (deg)':>12}")
                print("-" * 50)
                for motor in target_motors:
                    if motor in positions:
                        print(
                            f"{motor:<20} | {mins[motor]:>12.1f} | {positions[motor]:>12.1f} | {maxes[motor]:>12.1f}"
                        )

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                move_cursor_up(len(target_motors) + 4)

            time.sleep(LONG_TIMEOUT_SEC)

        self.enable_torque(target_motors)

        for motor in target_motors:
            if (motor in mins) and (motor in maxes) and (int(abs(maxes[motor] - mins[motor])) < 5):
                raise ValueError(f"Motor {motor} has insufficient range of motion (< 5 degrees)")

        return mins, maxes

    def _get_motors_list(self, motors: str | list[str] | None) -> list[str]:
        """将电机指定参数转换为电机名称列表。"""
        if motors is None:
            return list(self.motors.keys())
        elif isinstance(motors, str):
            return [motors]
        elif isinstance(motors, list):
            return motors
        else:
            raise TypeError(f"Invalid motors type: {type(motors)}")

    def _get_motor_id(self, motor: NameOrID) -> int:
        """获取电机的 CAN ID。"""
        if isinstance(motor, str):
            if motor in self.motors:
                return self.motors[motor].id
            else:
                raise ValueError(f"Unknown motor: {motor}")
        else:
            return motor

    def _get_motor_name(self, motor: NameOrID) -> str:
        """根据名称或 ID 获取电机名称。"""
        if isinstance(motor, str):
            return motor
        else:
            for name, m in self.motors.items():
                if m.id == motor:
                    return name
            raise ValueError(f"Unknown motor ID: {motor}")

    def _get_motor_recv_id(self, motor: NameOrID) -> int:
        """根据名称或 ID 获取电机的 recv_id。"""
        motor_name = self._get_motor_name(motor)
        motor_obj = self.motors.get(motor_name)
        if motor_obj and motor_obj.recv_id is not None:
            return motor_obj.recv_id
        else:
            raise ValueError(f"Motor {motor_obj} doesn't have a valid recv_id (None).")

    @property
    def is_calibrated(self) -> bool:
        """检查电机是否已校准。"""
        return bool(self.calibration)
