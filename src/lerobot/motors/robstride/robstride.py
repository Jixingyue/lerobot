# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

# TODO(Virgile)：增强模式控制的健壮性，目前只实现了 MIT 协议

import logging
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import cached_property
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypedDict

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _can_available, require_package

if TYPE_CHECKING or _can_available:
    import can
else:
    can = SimpleNamespace(Message=object, interface=None, BusABC=object)
import numpy as np

from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..motors_bus import Motor, MotorCalibration, MotorsBusBase, NameOrID, Value
from .tables import (
    AVAILABLE_BAUDRATES,
    CAN_CMD_CLEAR_FAULT,
    CAN_CMD_DISABLE,
    CAN_CMD_ENABLE,
    CAN_CMD_SET_ZERO,
    DEFAULT_BAUDRATE,
    DEFAULT_TIMEOUT_MS,
    HANDSHAKE_TIMEOUT_S,
    MODEL_RESOLUTION,
    MOTOR_LIMIT_PARAMS,
    NORMALIZED_DATA,
    PARAM_TIMEOUT,
    RUNNING_TIMEOUT,
    STATE_CACHE_TTL_S,
    ControlMode,
    MotorType,
)

logger = logging.getLogger(__name__)


class MotorState(TypedDict):
    position: float
    velocity: float
    torque: float
    temp_mos: float
    temp_rotor: float


class RobstrideMotorsBus(MotorsBusBase):
    """
    使用 CAN 总线通信的 MotorsBus 的 Robstride 实现。

    本类使用 python-can 与 Robstride 电机进行 CAN 总线通信。
    电机需要切换到 MIT 控制模式才能与此实现兼容。
    协议的更多细节可参阅以下文档链接：
    - python-can 文档：https://python-can.readthedocs.io/en/stable/
    - Robstride CAN 协议：https://github.com/RobStride/MotorStudio
    """

    # CAN 专用设置
    available_baudrates = deepcopy(AVAILABLE_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS

    # 电机配置
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    normalized_data = deepcopy(NORMALIZED_DATA)

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
        初始化 Robstride 电机总线。

        Args:
            port: CAN 接口名称（例如 Linux 下为 "can0"，macOS 下为 "/dev/cu.usbmodem*"）
            motors: 电机名称到 Motor 对象的映射字典
            calibration: 可选的校准数据
            can_interface: CAN 接口类型 - "auto"（默认）、"socketcan"（Linux）或 "slcan"（macOS/serial）
            use_can_fd: 是否使用 CAN FD 模式（OpenArms 默认为 True）
            bitrate: 标称比特率，单位 bps（默认：1000000 = 1 Mbps）
            data_bitrate: CAN FD 的数据比特率，单位 bps（默认：5000000 = 5 Mbps），当 use_can_fd 为 False 时被忽略
        """
        require_package("python-can", extra="robstride", import_name="can")
        super().__init__(port, motors, calibration)
        self.port = port
        self.can_interface = can_interface
        self.use_can_fd = use_can_fd
        self.bitrate = bitrate
        self.data_bitrate = data_bitrate
        self.canbus: can.BusABC | None = None
        self._is_connected = False

        # 将电机名称映射到 CAN ID
        self._motor_can_ids: dict[str, int] = {}
        self._recv_id_to_motor: dict[int, str] = {}

        # 存储电机类型和接收 ID
        self._motor_types: dict[str, MotorType] = {}
        # 动态增益存储（通过 write/sync_write 的 Damiao 风格更新路径）
        self._gains: dict[str, dict[str, float]] = {}
        for name, motor in self.motors.items():
            if motor.motor_type_str is not None:
                self._motor_types[name] = getattr(MotorType, motor.motor_type_str.upper())
            else:
                # 未指定时默认为 O0
                self._motor_types[name] = MotorType.O0

            # Damiao 风格的默认值：启动时为每个电机设置固定增益。
            self._gains[name] = {"kp": 10.0, "kd": 0.5}

            # 将 recv_id 映射到电机名称，用于过滤响应
            if motor.recv_id is not None:
                self._recv_id_to_motor[motor.recv_id] = name
        # 电机模式
        self.enabled: dict[str, bool] = {}
        self.operation_mode: dict[str, ControlMode] = {}
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
        self.last_feedback_time: dict[str, float | None] = {}
        self._id_to_name: dict[int, str] = {}
        for name in self.motors:
            self.enabled[name] = False
            self.operation_mode[name] = ControlMode.MIT  # 默认模式
            self.last_feedback_time[name] = None

        for name, motor in self.motors.items():
            key = motor.recv_id if motor.recv_id is not None else motor.id
            self._id_to_name[key] = name

    @property
    def is_connected(self) -> bool:
        """检查 CAN 总线是否已连接。"""
        return self._is_connected and self.canbus is not None

    def _bus(self) -> can.BusABC:
        if self.canbus is None:
            raise DeviceNotConnectedError(f"{self.__class__.__name__}('{self.port}') is not connected.")
        return self.canbus

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

    def _query_status_via_clear_fault(
        self, motor: NameOrID, timeout: float = RUNNING_TIMEOUT
    ) -> tuple[bool, can.Message | None]:
        motor_name = self._get_motor_name(motor)
        motor_id = self._get_motor_id(motor_name)
        recv_id = self._get_motor_recv_id(motor_name)
        data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self._bus().send(msg)
        return self._recv_status_via_clear_fault(expected_recv_id=recv_id, timeout=timeout)

    def _recv_status_via_clear_fault(
        self, expected_recv_id: int | None = None, timeout: float = RUNNING_TIMEOUT
    ) -> tuple[bool, can.Message | None]:
        """
        轮询总线以获取故障清除请求的响应。

        Args:
            expected_recv_id: 如果提供，则只接受来自该 CAN ID 的帧。
            timeout: 轮询总线的最长时间（秒）。

        Returns:
            元组，第一个元素在收到故障帧时为 True，
            第二个元素是 CAN 消息（超时则为 None）。
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            msg = self._bus().recv(timeout=RUNNING_TIMEOUT / 10)
            if not msg:
                continue

            if expected_recv_id is not None and msg.data[0] != expected_recv_id:
                continue

            # 故障状态帧的启发式判断（基于文档）
            fault_bits = int.from_bytes(msg.data[1:5], "little")
            if fault_bits != 0 and msg.data[5] == msg.data[6] == msg.data[7] == 0:
                logger.error(
                    f"Motor fault received from CAN ID 0x{msg.arbitration_id:02X}: "
                    f"fault_bits=0x{fault_bits:08X}"
                )
                return True, msg

            # 否则：有效的正常响应
            return False, msg

        return False, None

    def update_motor_state(self, motor: NameOrID) -> bool:
        has_fault, msg = self._query_status_via_clear_fault(motor)
        if msg is None:
            logger.warning(f"No response received from motor '{motor}' during state update.")
            raise ConnectionError(f"No response received from motor '{motor}' during state update.")
        if has_fault:
            logger.error(f"Fault reported by motor '{motor}' during state update. msg={msg.data.hex()}")
            raise RuntimeError(f"Fault reported by motor '{motor}' during state update.")

        self._decode_motor_state(msg.data)  # 更新缓存
        return True

    def _handshake(self) -> None:
        logger.info("Starting handshake with motors...")
        missing_motors = []
        faulted_motors = []

        for motor_name in self.motors:
            has_fault, msg = self._query_status_via_clear_fault(motor_name, timeout=HANDSHAKE_TIMEOUT_S)
            if msg is None:
                missing_motors.append(motor_name)
            elif has_fault:
                faulted_motors.append(motor_name)
            else:
                # CLEAR_FAULT 响应并不保证在所有固件版本上都与 MIT 反馈布局一致。
                # 不应仅因为缓存预热失败就让握手失败。
                try:
                    self._decode_motor_state(msg.data)
                except Exception as e:
                    logger.debug(
                        "Handshake cache warm-up decode failed for motor '%s': %s",
                        motor_name,
                        e,
                    )
            time.sleep(0.01)

        if missing_motors or faulted_motors:
            details = []
            if missing_motors:
                details.append(f"did not respond: {missing_motors}")
            if faulted_motors:
                details.append(f"reported fault: {faulted_motors}")
            raise ConnectionError("Handshake failed. " + "; ".join(details))

        logger.info("Handshake successful. All motors ready.")

    def _switch_operation_mode(self, motor: NameOrID, mode: ControlMode) -> None:
        """切换电机的运行模式。"""
        motor_name = self._get_motor_name(motor)
        motor_id = self._get_motor_id(motor_name)
        recv_id = self._get_motor_recv_id(motor_name)
        data = [0xFF] * 8
        data[6] = mode.value
        data[7] = 0xFC
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self._bus().send(msg)
        msg = self._recv_motor_response(expected_recv_id=recv_id, timeout=PARAM_TIMEOUT)
        if msg is not None:
            self.operation_mode[motor_name] = mode

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
        # Robstride 电机在 MIT 模式下不需要太多配置
        # 只需确保它们已使能即可
        for motor in self.motors:
            self._enable_motor(self._get_motor_name(motor))
            self._switch_operation_mode(motor, ControlMode.MIT)
            time.sleep(0.01)

    def switch_to_mode(self, mode: ControlMode) -> None:
        """切换所选电机的运行模式。"""
        for motor in self.motors:
            self._switch_operation_mode(motor, mode)
            time.sleep(0.01)

    def _enable_motor(self, motor: NameOrID) -> None:
        """使能单个电机。"""
        motor_id = self._get_motor_id(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [0xFF] * 7 + [CAN_CMD_ENABLE]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self._bus().send(msg)
        self._recv_motor_response(expected_recv_id=recv_id, timeout=PARAM_TIMEOUT)

    def _disable_motor(self, motor: NameOrID) -> None:
        """禁用单个电机。"""
        motor_id = self._get_motor_id(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [0xFF] * 7 + [CAN_CMD_DISABLE]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self._bus().send(msg)
        self._recv_motor_response(expected_recv_id=recv_id)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """启用所选电机的力矩。"""
        motors = self._get_motors_list(motors)
        for motor in motors:
            for _ in range(num_retry + 1):
                try:
                    self._get_motor_name(motor)
                    self._enable_motor(self._get_motor_name(motor))
                    break
                except Exception as e:
                    if _ == num_retry:
                        raise e
                    time.sleep(0.01)

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """禁用所选电机的力矩。"""
        motors = self._get_motors_list(motors)
        for motor in motors:
            for _ in range(num_retry + 1):
                try:
                    self._disable_motor(self._get_motor_name(motor))
                    break
                except Exception as e:
                    if _ == num_retry:
                        raise e
                    time.sleep(0.01)

    @contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):
        """
        保证力矩会被重新启用的上下文管理器。

        此辅助方法在临时禁用力矩以配置电机时很有用。

        Examples:
            >>> with bus.torque_disabled():
            ...     # Safe operations here with torque disabled
            ...     pass
        """
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def set_zero_position(self, motors: str | list[str] | None = None) -> None:
        """将所选电机的当前位置设为零点。"""
        motors = self._get_motors_list(motors)
        for motor in motors:
            motor_id = self._get_motor_id(motor)
            recv_id = self._get_motor_recv_id(motor)
            data = [0xFF] * 7 + [CAN_CMD_SET_ZERO]
            msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
            self._bus().send(msg)
            self._recv_motor_response(expected_recv_id=recv_id)
            time.sleep(0.01)

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
        try:
            start_time = time.time()
            messages_seen = []
            while time.time() - start_time < timeout:
                msg = self._bus().recv(timeout=RUNNING_TIMEOUT / 10)  # 100us 超时，用于快速轮询
                if msg:
                    messages_seen.append(f"0x{msg.arbitration_id:02X}")
                    # 如果未指定过滤器，则返回任意消息
                    if expected_recv_id is None:
                        return msg
                    # 否则，只在匹配预期的 recv_id 时返回
                    if msg.data[0] == expected_recv_id:
                        return msg
                    else:
                        logger.debug(
                            f"Ignoring message from CAN ID 0x{msg.arbitration_id:02X}, expected 0x{expected_recv_id:02X}"
                        )

            # 仅在调试模式下记录警告，以减少开销
            if logger.isEnabledFor(logging.DEBUG):
                if messages_seen:
                    logger.debug(
                        f"Received {len(messages_seen)} message(s) from IDs {set(messages_seen)}, but expected 0x{expected_recv_id:02X}"
                    )
                else:
                    logger.debug(f"No CAN messages received (expected from 0x{expected_recv_id:02X})")
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

        try:
            while len(responses) < len(expected_recv_ids) and (time.time() - start_time) < timeout:
                msg = self._bus().recv(timeout=RUNNING_TIMEOUT / 10)  # 100us 轮询超时
                if msg and msg.data[0] in expected_set:
                    responses[msg.data[0]] = msg
                    if len(responses) == len(expected_recv_ids):
                        break  # 已收到所有响应，提前退出
        except Exception as e:
            logger.debug(f"Error receiving responses: {e}")

        return responses

    def _recv_all_messages_until_quiet(
        self,
        *,
        timeout: float = RUNNING_TIMEOUT,
        max_messages: int = 4096,
    ) -> list[can.Message]:
        """
        接收帧直到总线安静下来。

        Args:
            timeout: 每次 recv() 调用使用的轮询超时。当某次 recv()
                超时时（安静间隙）停止收集。
            max_messages: 防止无限循环的安全上限。
        """
        out: list[can.Message] = []
        max_messages = max(1, max_messages)
        timeout = max(0.0, timeout)

        try:
            while len(out) < max_messages:
                msg = self._bus().recv(timeout=timeout)
                if msg is None:
                    break
                out.append(msg)
        except (can.CanError, OSError) as e:
            logger.debug(f"Error draining CAN RX queue on {self.port}: {e}")

        return out

    def _process_feedback_messages(self, messages: list[can.Message]) -> set[int]:
        """
        解码所有收到的反馈帧并更新缓存的电机状态。

        Returns:
            成功映射到电机的负载 recv_id 集合。
        """
        processed_recv_ids: set[int] = set()
        for msg in messages:
            if len(msg.data) < 1:
                logger.debug(
                    f"Dropping short CAN frame on {self.port} "
                    f"(arb=0x{int(msg.arbitration_id):02X}, data={bytes(msg.data).hex()})"
                )
                continue

            recv_id = int(msg.data[0])
            motor_name = self._recv_id_to_motor.get(recv_id)
            if motor_name is None:
                logger.debug(
                    f"Unmapped CAN frame on {self.port} "
                    f"(arb=0x{int(msg.arbitration_id):02X}, recv_id=0x{recv_id:02X}, data={bytes(msg.data).hex()})"
                )
                continue

            self._process_response(motor_name, msg)
            processed_recv_ids.add(recv_id)

        return processed_recv_ids

    def flush_rx_queue(self, poll_timeout_s: float = 0.0005, max_messages: int = 4096) -> int:
        """
        排空 CAN 接口上待处理的 RX 帧。

        上层控制器用它在新读取周期开始前丢弃过时的反馈，
        使后续的状态读取基于最新的响应。
        在控制器实例创建/连接时也应调用一次，
        以清除之前会话残留在接口上的帧。
        """
        drained = 0
        poll_timeout_s = max(0.0, poll_timeout_s)
        max_messages = max(1, max_messages)
        try:
            while drained < max_messages:
                msg = self._bus().recv(timeout=poll_timeout_s)
                if msg is None:
                    break
                drained += 1
        except (can.CanError, OSError) as e:
            logger.debug(f"Failed to flush CAN RX queue on {self.port}: {e}")
        return drained

    def _speed_control(
        self,
        motor: NameOrID,
        velocity_deg_per_sec: float,
        current_limit_a: float,
    ) -> None:
        """
        向单个电机发送速度模式控制命令（Command 11）。

        Args:
            motor: 电机名称或 CAN ID。
            velocity_rad_per_sec: 目标速度，单位 rad/s（32 位浮点）。
            current_limit_a: 电流限制，单位 A（32 位浮点）。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        motor_id = self._get_motor_id(motor)
        motor_name = self._get_motor_name(motor)
        # 可选：确保电机处于速度控制模式

        if self.operation_mode[motor_name] != ControlMode.VEL:
            raise RuntimeError(f"Motor '{motor_name}' is not in velocity control mode.")
        # 转换为 rad/s 以匹配协议规范

        velocity_rad_per_sec = np.radians(velocity_deg_per_sec)

        # 不使用 struct，将 float32 编码为小端字节（字节列表）
        def _float32_to_le_bytes(x: float) -> list[int]:
            b = np.float32(x).tobytes()  # 4 字节，小端序
            return [b[0], b[1], b[2], b[3]]

        speed_bytes = _float32_to_le_bytes(velocity_rad_per_sec)
        limit_bytes = _float32_to_le_bytes(current_limit_a)

        data = speed_bytes + limit_bytes  # 8 个字节：[0–3]=速度，[4–7]=电流限制

        msg = can.Message(
            arbitration_id=motor_id,
            data=data,
            is_extended_id=False,
        )
        self._bus().send(msg)

        # 如果协议返回状态类型的响应，可以像 MIT 一样解码
        recv_id = self._get_motor_recv_id(motor)
        if recv_id is not None:
            resp = self._recv_motor_response(expected_recv_id=recv_id)
            if resp:
                self._decode_motor_state(resp.data)

    def _mit_control(
        self,
        motor: NameOrID,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float,
        torque: float,
        *,
        wait_for_response: bool = True,
    ) -> None:
        """
        向电机发送 MIT 控制命令。

        Args:
            motor: 电机名称或 ID
            kp: 位置增益
            kd: 速度增益
            position_degrees: 目标位置（度）
            velocity_deg_per_sec: 目标速度（度/秒）
            torque: 目标力矩（N·m）
        """
        motor_name = self._get_motor_name(motor)
        motor_type = self._motor_types[motor_name]
        if self.operation_mode[motor_name] != ControlMode.MIT:
            raise RuntimeError(f"Motor '{motor_name}' is not in MIT control mode.")
        motor_id = self._get_motor_id(motor)
        data = self._encode_mit_packet(motor_type, kp, kd, position_degrees, velocity_deg_per_sec, torque)
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self._bus().send(msg)

        if wait_for_response:
            recv_id = self._get_motor_recv_id(motor)
            msg = self._recv_motor_response(expected_recv_id=recv_id)
            if msg:
                self._process_response(motor_name, msg)

    def _encode_mit_packet(
        self,
        motor_type: MotorType,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float,
        torque: float,
    ) -> list[int]:
        """从物理量编码 MIT 控制命令负载。"""
        position_rad = np.radians(position_degrees)
        velocity_rad_per_sec = np.radians(velocity_deg_per_sec)
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]

        kp_uint = self._float_to_uint(kp, 0, 500, 12)
        kd_uint = self._float_to_uint(kd, 0, 5, 12)
        q_uint = self._float_to_uint(position_rad, -pmax, pmax, 16)
        dq_uint = self._float_to_uint(velocity_rad_per_sec, -vmax, vmax, 12)
        tau_uint = self._float_to_uint(torque, -tmax, tmax, 12)

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

    def _mit_control_batch(
        self,
        commands: dict[NameOrID, tuple[float, float, float, float, float]],
    ) -> None:
        """批量发送 MIT 命令，并根据收集的响应更新缓存。"""
        if not commands:
            return

        recv_id_to_motor: dict[int, str] = {}
        for motor, (kp, kd, position_degrees, velocity_deg_per_sec, torque) in commands.items():
            motor_name = self._get_motor_name(motor)
            if self.operation_mode[motor_name] != ControlMode.MIT:
                raise RuntimeError(f"Motor '{motor_name}' is not in MIT control mode.")

            motor_id = self._get_motor_id(motor)
            motor_type = self._motor_types[motor_name]
            data = self._encode_mit_packet(motor_type, kp, kd, position_degrees, velocity_deg_per_sec, torque)
            msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
            self._bus().send(msg)
            recv_id_to_motor[self._get_motor_recv_id(motor)] = motor_name
        # 读取所有反馈帧直到 RX 安静下来，然后全部解码。
        # 这可以避免在不同电机的响应交错时丢弃有用的帧。
        messages = self._recv_all_messages_until_quiet()
        processed_recv_ids = self._process_feedback_messages(messages)

        for recv_id, motor_name in recv_id_to_motor.items():
            if recv_id not in processed_recv_ids:
                logger.warning(f"Packet drop: {motor_name} (ID: 0x{recv_id:02X}). Using last known state.")

    def _float_to_uint(self, x: float, x_min: float, x_max: float, bits: int) -> int:
        """将浮点数转换为无符号整数以进行 CAN 传输。"""
        x = max(x_min, min(x_max, x))  # 限幅到范围内
        span = x_max - x_min
        data_norm = (x - x_min) / span
        return int(data_norm * ((1 << bits) - 1))

    def _uint_to_float(self, x: int, x_min: float, x_max: float, bits: int) -> float:
        """将来自 CAN 的无符号整数转换为浮点数。"""
        span = x_max - x_min
        data_norm = float(x) / ((1 << bits) - 1)
        return data_norm * span + x_min

    def _decode_motor_state(self, data: bytearray | bytes) -> tuple[float, float, float, float]:
        """
        从 CAN 数据解码电机状态。

        Returns:
            (position_degrees, velocity_deg_per_sec, torque, temp_mos) 元组
        """
        if len(data) < 8:
            raise ValueError("Invalid motor state data")

        # 提取编码值
        motor_id = data[0]
        motor_name = self._id_to_name[motor_id]
        q_uint = (data[1] << 8) | data[2]
        dq_uint = (data[3] << 4) | (data[4] >> 4)
        tau_uint = ((data[4] & 0x0F) << 8) | data[5]
        t_mos = (data[6] << 8) | data[7]

        motor_type = self._motor_types[motor_name]
        # 获取电机限位参数
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]

        # 解码为物理量（弧度）
        position_rad = self._uint_to_float(q_uint, -pmax, pmax, 16)
        velocity_rad_per_sec = self._uint_to_float(dq_uint, -vmax, vmax, 12)
        torque = self._uint_to_float(tau_uint, -tmax, tmax, 12)

        # 转换为度
        position_degrees = np.degrees(position_rad)
        velocity_deg_per_sec = np.degrees(velocity_rad_per_sec)

        # 更新缓存状态
        self.last_feedback_time[motor_name] = time.time()
        self._last_known_states[motor_name] = {
            "position": position_degrees,
            "velocity": velocity_deg_per_sec,
            "torque": torque,
            "temp_mos": t_mos / 10,
            # Robstride MIT 反馈中不可用。
            "temp_rotor": 0.0,
        }
        return position_degrees, velocity_deg_per_sec, torque, t_mos / 10

    def _process_response(self, motor: str, msg: can.Message) -> None:
        """解码反馈帧并更新单个电机的缓存。"""
        try:
            self._decode_motor_state(msg.data)
        except Exception as e:
            logger.warning(
                f"Failed to decode response from {motor} "
                f"(arb=0x{int(msg.arbitration_id):02X}, data={bytes(msg.data).hex()}): {e}"
            )

    def _get_cached_value(self, motor: str, data_name: str) -> Value:
        """从状态缓存中获取特定值。"""
        state = self._last_known_states[motor]
        mapping: dict[str, Any] = {
            "Present_Position": state["position"],
            "Present_Velocity": state["velocity"],
            "Present_Torque": state["torque"],
            "Temperature_MOS": state["temp_mos"],
        }
        if data_name == "Temperature_Rotor":
            raise NotImplementedError("Rotor temperature reading not accessible.")
        if data_name not in mapping:
            raise ValueError(f"Unknown data_name: {data_name}")
        return mapping[data_name]

    @check_if_not_connected
    def read(
        self,
        data_name: str,
        motor: str,
    ) -> Value:
        """从单个电机读取值。位置始终以度为单位。"""

        # 刷新电机以获取最新状态
        t_init = time.time()
        if (
            self.last_feedback_time[motor] is None
            or t_init - (self.last_feedback_time[motor] or 0) > STATE_CACHE_TTL_S
        ):
            self.update_motor_state(motor)

        return self._get_cached_value(motor, data_name)

    @check_if_not_connected
    def write(
        self,
        data_name: str,
        motor: str,
        value: Value,
    ) -> None:
        """向单个电机写入值。位置始终以度为单位。"""
        motor_name = self._get_motor_name(motor)

        if data_name in ("Kp", "Kd"):
            self._gains[motor_name][data_name.lower()] = float(value)
        elif data_name == "Goal_Position":
            # 使用 MIT 控制，位置以度为单位
            kp = self._gains[motor_name]["kp"]
            kd = self._gains[motor_name]["kd"]
            self._mit_control(motor, kp, kd, value, 0, 0)
        elif data_name == "Goal_Velocity":
            # 使用速度控制模式
            if self.operation_mode[motor_name] != ControlMode.VEL:
                raise RuntimeError(f"Motor '{motor_name}' is not in velocity control mode.")
            current_limit_a = 5.0  # 示例电流限制 / 文档中未指定。此模式很少使用，主要用于诊断
            self._speed_control(motor, value, current_limit_a)
        else:
            raise ValueError(f"Writing {data_name} not supported in MIT mode")

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
    ) -> dict[str, Value]:
        """
        同时从多个电机读取相同的值。
        使用批量操作：先发送所有刷新命令，然后收集所有响应。
        这比顺序读取快得多（OpenArms 模式）。
        """
        target_motors = self._get_motors_list(motors)
        self._batch_refresh(target_motors)
        return {motor: self._get_cached_value(motor, data_name) for motor in target_motors}

    @check_if_not_connected
    def sync_write(
        self,
        data_name: str,
        values: dict[str, Value],
    ) -> None:
        """
        同时向多个电机写入不同的值。位置始终以度为单位。
        使用批量操作：先发送所有命令，当使用 MIT 模式时再收集响应；
        否则对每个电机分别发送命令并等待响应。
        """
        if data_name in ("Kp", "Kd"):
            key = data_name.lower()
            for motor, val in values.items():
                motor_name = self._get_motor_name(motor)
                self._gains[motor_name][key] = float(val)
        elif data_name == "Goal_Position":
            commands: dict[NameOrID, tuple[float, float, float, float, float]] = {}
            for motor, value_degrees in values.items():
                motor_name = self._get_motor_name(motor)
                commands[motor] = (
                    self._gains[motor_name]["kp"],
                    self._gains[motor_name]["kd"],
                    float(value_degrees),
                    0.0,
                    0.0,
                )
            self._mit_control_batch(commands)
        else:
            # 其他数据类型回退到逐个写入
            for motor, value in values.items():
                self.write(data_name, motor, value)

    def sync_read_all_states(
        self,
        motors: str | list[str] | None = None,
        *,
        num_retry: int = 0,
    ) -> dict[str, MotorState]:
        """
        使用 Robstride TTL 刷新策略读取所有电机状态（位置、速度、力矩）。
        """
        target_motors = self._get_motors_list(motors)
        self._batch_refresh(target_motors)
        return {motor: self._last_known_states[motor].copy() for motor in target_motors}

    def _batch_refresh(self, motors: list[str]) -> None:
        """刷新一组电机并更新反馈缓存。"""
        init_time = time.time()
        updated_motors: list[str] = []

        for motor in motors:
            if (
                self.last_feedback_time[motor] is not None
                and (init_time - (self.last_feedback_time[motor] or 0)) < STATE_CACHE_TTL_S
            ):
                continue
            motor_id = self._get_motor_id(motor)
            data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
            msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
            self._bus().send(msg)
            updated_motors.append(motor)

        messages = self._recv_all_messages_until_quiet()
        processed_recv_ids = self._process_feedback_messages(messages)

        for motor in updated_motors:
            recv_id = self._get_motor_recv_id(motor)
            if recv_id not in processed_recv_ids:
                logger.warning(f"Packet drop: {motor} (ID: 0x{recv_id:02X}). Using last known state.")

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """从电机读取校准数据。"""
        # Robstride 电机不在内部存储校准数据
        # 返回现有校准数据或空字典
        return self.calibration if self.calibration else {}

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """向电机写入校准数据。"""
        # Robstride 电机不在内部存储校准数据
        # 仅在内存中缓存
        if cache:
            self.calibration = calibration_dict

    def record_ranges_of_motion(
        self, motors: str | list[str] | None = None, display_values: bool = True
    ) -> tuple[dict[str, Value], dict[str, Value]]:
        """
        以交互方式记录每个电机的最小/最大值（单位：度）。

        在方法实时输出当前位置时，用手移动关节（力矩已禁用）。
        按 Enter 键结束。
        """
        target_motors = self._get_motors_list(motors)

        # 禁用力矩以便手动移动
        self.disable_torque(target_motors)
        time.sleep(0.1)

        # 获取初始位置（已经是度）
        start_positions = self.sync_read("Present_Position", target_motors)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        print("\nMove joints through their full range of motion. Press ENTER when done.")
        user_pressed_enter = False

        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", target_motors)

            for motor in target_motors:
                if motor in positions:
                    mins[motor] = int(
                        min(
                            positions[motor],
                            mins.get(motor, positions[motor]),
                        )
                    )
                    maxes[motor] = int(
                        max(
                            positions[motor],
                            maxes.get(motor, positions[motor]),
                        )
                    )

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
                # 上移光标以覆盖之前的输出
                move_cursor_up(len(target_motors) + 4)

            time.sleep(0.05)

        # 重新启用力矩
        self.enable_torque(target_motors)

        # 验证范围
        for motor in target_motors:
            if (motor in mins) and (motor in maxes) and (abs(maxes[motor] - mins[motor]) < 5.0):
                raise ValueError(f"Motor {motor} has insufficient range of motion (< 5 degrees)")

        return mins, maxes

    def _get_motors_list(self, motors: str | list[str] | None) -> list[str]:
        """将电机规格转换为电机名称列表。"""
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
        """返回该电机反馈负载 byte0 中预期的 ID。

        Robstride MIT 反馈帧在 data[0] 中编码了一个 ID。某些配置将其暴露为
        `motor.recv_id`；否则回退到配置的 `motor.id`。
        """
        motor_name = self._get_motor_name(motor)
        motor_obj = self.motors[motor_name]

        recv_id = getattr(motor_obj, "recv_id", None)
        if recv_id is None:
            logger.debug(
                "Motor '%s' has no recv_id; falling back to motor.id=%s for feedback demux.",
                motor_name,
                motor_obj.id,
            )
            return motor_obj.id

        return recv_id

    @cached_property
    def is_calibrated(self) -> bool:
        """检查电机是否已校准。"""
        return bool(self.calibration)
