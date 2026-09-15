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

# 文档：
# hebi: https://docs.hebi.us/tools.html#mobile-io
# teleop: https://github.com/SpesRobotics/teleop

import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _hebi_available, _teleop_available, require_package
from lerobot.utils.rotation import Rotation

if TYPE_CHECKING or _hebi_available:
    import hebi
else:
    hebi = None

if TYPE_CHECKING or _teleop_available:
    from teleop import Teleop
else:
    Teleop = None

from ..teleoperator import Teleoperator
from .config_phone import PhoneConfig, PhoneOS

logger = logging.getLogger(__name__)


class BasePhone:
    _enabled: bool = False
    _calib_pos: np.ndarray | None = None
    _calib_rot_inv: Rotation | None = None

    def _reapply_position_calibration(self, pos: np.ndarray) -> None:
        self._calib_pos = pos.copy()

    @property
    def is_calibrated(self) -> bool:
        return (self._calib_pos is not None) and (self._calib_rot_inv is not None)

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "phone.pos": np.ndarray,  # 形状 (3,)
            "phone.rot": Rotation,  # scipy.spatial.transform.Rotation
            "phone.raw_inputs": dict,  # 模拟量/按键或 webXR 元数据
            "phone.enabled": bool,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        # 尚未实现触觉或其他反馈
        pass

    def configure(self) -> None:
        # 手机遥操作无需额外配置
        pass

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # 我们可以在这里添加触觉反馈（振动），但尚未实现
        raise NotImplementedError


class IOSPhone(BasePhone, Teleoperator):
    name = "ios_phone"

    def __init__(self, config: PhoneConfig):
        require_package("hebi-py", extra="phone", import_name="hebi")
        require_package("teleop", extra="phone")
        super().__init__(config)
        self.config = config
        self._group = None

    @property
    def is_connected(self) -> bool:
        return self._group is not None

    @check_if_already_connected
    def connect(self) -> None:
        logger.info("Connecting to IPhone, make sure to open the HEBI Mobile I/O app.")
        lookup = hebi.Lookup()
        time.sleep(2.0)
        group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
        if group is None:
            raise RuntimeError("Mobile I/O not found — check name/family settings in the app.")
        self._group = group
        logger.info(f"{self} connected to HEBI group with {group.size} module(s).")

        self.calibrate()

    def calibrate(self) -> None:
        print(
            "Hold the phone so that: top edge points forward in same direction as the robot (robot +x) and screen points up (robot +z)"
        )
        print("Press and hold B1 in the HEBI Mobile I/O app to capture this pose...\n")
        position, rotation = self._wait_for_capture_trigger()
        self._calib_pos = position.copy()
        self._calib_rot_inv = rotation.inv()
        self._enabled = False
        print("Calibration done\n")

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """
        阻塞执行，直到从 iOS 设备检测到校准触发。

        该方法进入循环，持续读取手机状态。它等待用户在 HEBI Mobile I/O 应用中
        按住 'B1' 按钮。一旦按下 B1，循环终止并返回手机在那一刻的位姿。

        Returns:
            一个元组，包含触发激活时刻手机的位置 (np.ndarray) 和旋转 (Rotation)。
        """
        while True:
            has_pose, position, rotation, fb_pose = self._read_current_pose()
            if not has_pose:
                time.sleep(0.01)
                continue

            io = getattr(fb_pose, "io", None)
            button_b = getattr(io, "b", None) if io is not None else None
            button_b1_pressed = False
            if button_b is not None:
                button_b1_pressed = bool(button_b.get_int(1))
            if button_b1_pressed:
                return position, rotation

            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None, object | None]:
        """
        通过 HEBI SDK 从已连接的 iOS 设备读取瞬时 6 自由度位姿。

        该方法从 HEBI 组获取最新的反馈数据包，提取 ARKit 的位置和方向，
        并将其转换为标准格式。它还应用配置的摄像头偏移，将位姿从摄像头坐标系
        调整到手机的物理坐标系。

        Returns:
            一个元组，包含：
            - 一个布尔值，指示是否成功读取到有效位姿。
            - 以 NumPy 数组表示的 3D 位置，如果不可用则为 None。
            - 以 `Rotation` 对象表示的方向，如果不可用则为 None。
            - 原始 HEBI 反馈对象，用于访问按键等其他数据。
        """
        fbk = self._group.get_next_feedback()
        pose = fbk[0]
        ar_pos = getattr(pose, "ar_position", None)
        ar_quat = getattr(pose, "ar_orientation", None)
        if ar_pos is None or ar_quat is None:
            return False, None, None, None
        # HEBI 以 w, x, y, z 格式提供方向。
        # Scipy 的 Rotation 期望 x, y, z, w。
        quat_xyzw = np.concatenate((ar_quat[1:], [ar_quat[0]]))  # wxyz 转 xyzw
        # 在跟踪就绪之前或数据包丢失时，ARKit 可能会发出零/NaN 四元数。
        # Rotation.from_quat 现在会拒绝这些值；采用与位姿缺失相同的降级方式，
        # 使遥操作在会话过程中保持运行。
        try:
            rot = Rotation.from_quat(quat_xyzw)
        except ValueError:
            return False, None, None, None
        pos = ar_pos - rot.apply(self.config.camera_offset)
        return True, pos, rot, pose

    @check_if_not_connected
    def get_action(self) -> dict:
        has_pose, raw_position, raw_rotation, fb_pose = self._read_current_pose()
        if not has_pose or not self.is_calibrated:
            return {}

        # 收集原始输入（iOS 上的 B1 / 模拟量，Android 上的 move/scale）
        raw_inputs: dict[str, float | int | bool] = {}
        io = getattr(fb_pose, "io", None)
        if io is not None:
            bank_a, bank_b = io.a, io.b
            if bank_a:
                for ch in range(1, 9):
                    if bank_a.has_float(ch):
                        raw_inputs[f"a{ch}"] = float(bank_a.get_float(ch))
            if bank_b:
                for ch in range(1, 9):
                    if bank_b.has_int(ch):
                        raw_inputs[f"b{ch}"] = int(bank_b.get_int(ch))
                    elif hasattr(bank_b, "has_bool") and bank_b.has_bool(ch):
                        raw_inputs[f"b{ch}"] = int(bank_b.get_bool(ch))

        enable = bool(raw_inputs.get("b1", 0))

        # 上升沿时立即从当前原始位姿重新捕获校准
        if enable and not self._enabled:
            self._reapply_position_calibration(raw_position)

        # 应用校准
        pos_cal = self._calib_rot_inv.apply(raw_position - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rotation

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        self._group = None


class AndroidPhone(BasePhone, Teleoperator):
    name = "android_phone"

    def __init__(self, config: PhoneConfig):
        require_package("hebi-py", extra="phone", import_name="hebi")
        require_package("teleop", extra="phone")
        super().__init__(config)
        self.config = config
        self._teleop = None
        self._teleop_thread = None
        self._latest_pose = None
        self._latest_message = None
        self._android_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._teleop is not None

    @check_if_already_connected
    def connect(self) -> None:
        logger.info("Starting teleop stream for Android...")
        self._teleop = Teleop()
        self._teleop.subscribe(self._android_callback)
        self._teleop_thread = threading.Thread(target=self._teleop.run, daemon=True)
        self._teleop_thread.start()
        logger.info(f"{self} connected, teleop stream started.")

        self.calibrate()

    def calibrate(self) -> None:
        print(
            "Hold the phone so that: top edge points forward in same direction as the robot (robot +x) and screen points up (robot +z)"
        )
        print("Touch and move on the WebXR page to capture this pose...\n")

        pos, rot = self._wait_for_capture_trigger()
        self._calib_pos = pos.copy()
        self._calib_rot_inv = rot.inv()
        self._enabled = False
        print("Calibration done\n")

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """
        阻塞执行，直到从 Android 设备检测到校准触发。

        该方法进入循环，持续检查从 WebXR 会话接收到的最新消息。它等待用户在屏幕上
        触摸并移动手指，这会生成一个 `move` 事件。一旦检测到该事件，循环终止并返回
        手机的当前位姿。

        Returns:
            一个元组，包含触发激活时刻手机的位置 (np.ndarray) 和旋转 (Rotation)。
        """
        while True:
            with self._android_lock:
                msg = self._latest_message or {}

            if bool(msg.get("move", False)):
                ok, pos, rot, _pose = self._read_current_pose()
                if ok:
                    return pos, rot

            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None, object | None]:
        """
        读取从 Android 设备的 WebXR 会话接收到的最新 6 自由度位姿。

        该方法访问由 `_android_callback` 存储的最新位姿数据。它使用线程锁安全地读取
        共享的 `_latest_pose` 变量。该位姿是一个 4x4 矩阵，随后被分解为位置和旋转，
        并应用配置的摄像头偏移。

        Returns:
            一个元组，包含：
            - 一个布尔值，指示是否有有效位姿可用。
            - 以 NumPy 数组表示的 3D 位置，如果尚未收到位姿则为 None。
            - 以 `Rotation` 对象表示的方向，如果尚未收到位姿则为 None。
            - 从遥操作流接收到的原始 4x4 位姿矩阵。
        """
        with self._android_lock:
            if self._latest_pose is None:
                return False, None, None, None
            p = self._latest_pose.copy()
            pose = self._latest_pose
        rot = Rotation.from_matrix(p[:3, :3])
        pos = p[:3, 3] - rot.apply(self.config.camera_offset)
        return True, pos, rot, pose

    def _android_callback(self, pose: np.ndarray, message: dict) -> None:
        """
        处理来自 Android 遥操作流的传入数据的回调函数。

        每当从 Android 手机上的 WebXR 会话接收到新的位姿和消息时，该方法就由
        `teleop` 包的订阅者线程执行。它用新数据更新内部状态
        （`_latest_pose` 和 `_latest_message`）。使用线程锁确保这些共享变量被
        原子地更新，防止与读取它们的主线程产生竞态条件。

        Args:
            pose: 表示手机变换矩阵的 4x4 NumPy 数组。
            message: 包含额外数据的字典，例如按键或触摸事件。
        """
        with self._android_lock:
            self._latest_pose = pose
            self._latest_message = message

    @check_if_not_connected
    def get_action(self) -> dict:
        ok, raw_pos, raw_rot, pose = self._read_current_pose()
        if not ok or not self.is_calibrated:
            return {}

        # 收集原始输入（iOS 上的 B1 / 模拟量，Android 上的 move/scale）
        raw_inputs: dict[str, float | int | bool] = {}
        msg = self._latest_message or {}
        raw_inputs["move"] = bool(msg.get("move", False))
        raw_inputs["scale"] = float(msg.get("scale", 1.0))
        raw_inputs["reservedButtonA"] = bool(msg.get("reservedButtonA", False))
        raw_inputs["reservedButtonB"] = bool(msg.get("reservedButtonB", False))

        enable = bool(raw_inputs.get("move", False))

        # 上升沿时立即从当前原始位姿重新捕获校准
        if enable and not self._enabled:
            self._reapply_position_calibration(raw_pos)

        # 应用校准
        pos_cal = self._calib_rot_inv.apply(raw_pos - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rot

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        self._teleop = None
        if self._teleop_thread and self._teleop_thread.is_alive():
            self._teleop_thread.join(timeout=1.0)
            self._teleop_thread = None
            self._latest_pose = None


class Phone(Teleoperator):
    """
    基于手机的遥操作设备，使用 ARKit（iOS 通过 HEBI Mobile I/O 应用）或 teleop Python 包（Android 通过 WebXR API）。
    对于 HEBI Mobile I/O，我们还暴露 8 个模拟量输入 (a1-a8) 和 8 个数字输入 (b1-b8)。

    按住 **B1** 以启用遥操作。启用状态下，第一次按下 B1 会捕获参考位姿和旋转；
    禁用后再次按下时，位置会被重新应用。
    """

    config_class = PhoneConfig
    name = "phone"

    def __init__(self, config: PhoneConfig):
        super().__init__(config)
        self.config = config

        self._phone_impl: Teleoperator

        if self.config.phone_os == PhoneOS.IOS:
            self._phone_impl = IOSPhone(config)
        elif self.config.phone_os == PhoneOS.ANDROID:
            self._phone_impl = AndroidPhone(config)
        else:
            raise ValueError(f"Invalid config phone_os: {self.config.phone_os}")

    @property
    def is_connected(self) -> bool:
        return self._phone_impl.is_connected

    def connect(self) -> None:
        return self._phone_impl.connect()

    def calibrate(self) -> None:
        return self._phone_impl.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._phone_impl.is_calibrated

    @property
    def action_features(self) -> dict[str, type]:
        return self._phone_impl.action_features

    @property
    def feedback_features(self) -> dict[str, type]:
        return self._phone_impl.feedback_features

    def configure(self) -> None:
        return self._phone_impl.configure()

    def get_action(self) -> dict:
        return self._phone_impl.get_action()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        return self._phone_impl.send_feedback(feedback)

    def disconnect(self) -> None:
        return self._phone_impl.disconnect()
