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

import logging
import time
from queue import Queue
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, require_package
from lerobot.utils.keyboard_input import pynput_can_capture

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardRoverTeleopConfig,
    KeyboardTeleopConfig,
)

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info("Could not import pynput keyboard backend: %s", e)


class KeyboardTeleop(Teleoperator):
    """
    使用键盘输入进行控制的遥操作类。
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        require_package("pynput", extra="pynput-dep")
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        if PYNPUT_AVAILABLE and pynput_can_capture():
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.warning(
                "Keyboard teleoperation is unavailable in this environment. pynput can only "
                "capture key events on an X11 session (Linux), a Windows desktop, or macOS with "
                "Accessibility / Input Monitoring granted - not on Wayland or headless machines. "
                "This keyboard teleoperator will produce no actions; use an X11 session, a "
                "gamepad, or a leader-arm teleoperator instead."
            )
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        # 根据当前按键状态生成动作
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    使用键盘输入进行末端执行器控制的遥操作类。
    设计用于配合 `So100FollowerEndEffector` 机器人使用。
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # 根据当前按键状态生成动作
        for key, val in self.current_pressed.items():
            if key == keyboard.Key.up:
                delta_y = -int(val)
            elif key == keyboard.Key.down:
                delta_y = int(val)
            elif key == keyboard.Key.left:
                delta_x = int(val)
            elif key == keyboard.Key.right:
                delta_x = -int(val)
            elif key == keyboard.Key.shift:
                delta_z = -int(val)
            elif key == keyboard.Key.shift_r:
                delta_z = int(val)
            elif key == keyboard.Key.ctrl_r:
                # 夹爪动作的取值应为 0（闭合）、1（保持）、2（张开）之间
                gripper_action = int(val) + 1
            elif key == keyboard.Key.ctrl_l:
                gripper_action = int(val) - 1
            elif val:
                # 如果按键被按下，将其加入 misc_keys_queue
                # 这会记录不属于 delta_x、delta_y、delta_z 的按键
                # 这对于获取其他事件（如强化学习中的干预、回合成功等）很有用
                self.misc_keys_queue.put(key)

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        从键盘获取额外的控制事件，例如干预状态、回合终止、成功指示等。

        按键映射：
        - 按下任意移动键 = 干预激活
        - 's' 键 = 成功（成功终止回合）
        - 'r' 键 = 重新录制回合（终止并重新录制）
        - 'q' 键 = 退出回合（以失败终止）

        Returns:
            包含以下内容的字典：
                - is_intervention: bool - 人类当前是否正在干预
                - terminate_episode: bool - 是否终止当前回合
                - success: bool - 回合是否成功
                - rerecord_episode: bool - 是否重新录制回合
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # 检查当前是否有移动键被按下（表示干预）
        movement_keys = [
            keyboard.Key.up,
            keyboard.Key.down,
            keyboard.Key.left,
            keyboard.Key.right,
            keyboard.Key.shift,
            keyboard.Key.shift_r,
            keyboard.Key.ctrl_r,
            keyboard.Key.ctrl_l,
        ]
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

        self.current_pressed.clear()

        # 从 misc_keys_queue 中检查回合控制命令
        terminate_episode = False
        success = False
        rerecord_episode = False

        # 处理所有待处理的杂项按键
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "q":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


class KeyboardRoverTeleop(KeyboardTeleop):
    """
    用于 EarthRover Mini Plus 等移动机器人的键盘遥操作设备。

    提供直观的 WASD 风格控制来驾驶移动机器人：
    - 线性移动（前进/后退）
    - 角向移动（转向/旋转）
    - 速度调节
    - 紧急停止

    键盘控制：
        移动：
            - W：前进
            - S：后退
            - A：左转（带前进运动）
            - D：右转（带前进运动）
            - Q：原地左转
            - E：原地右转
            - X：紧急停止

        速度控制：
            - +/=：加速
            - -：减速

        系统：
            - ESC：断开遥操作设备

    Attributes:
        config: 遥操作设备配置
        current_linear_speed: 当前线速度大小
        current_angular_speed: 当前角速度大小

    Example:
        ```python
        from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig

        teleop = KeyboardRoverTeleop(
            KeyboardRoverTeleopConfig(linear_speed=1.0, angular_speed=1.0, speed_increment=0.1)
        )
        teleop.connect()

        while teleop.is_connected:
            action = teleop.get_action()
            robot.send_action(action)
        ```
    """

    config_class = KeyboardRoverTeleopConfig
    name = "keyboard_rover"

    def __init__(self, config: KeyboardRoverTeleopConfig):
        super().__init__(config)
        # 添加漫游车专用的速度设置
        self.current_linear_speed = config.linear_speed
        self.current_angular_speed = config.angular_speed

    @property
    def action_features(self) -> dict:
        """返回漫游车的动作格式（线速度和角速度）。"""
        return {
            "linear_velocity": float,
            "angular_velocity": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """漫游车遥操作不需要校准。"""
        return True

    def _drain_pressed_keys(self):
        """从事件队列更新 current_pressed 状态，且不清除仍按住的按键"""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                # 仅在按键被释放时才移除
                self.current_pressed.pop(key_char, None)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        根据按下的按键获取当前动作。

        Returns:
            包含 'linear_velocity' 和 'angular_velocity' 键的 RobotAction。
        """
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        linear_velocity = 0.0
        angular_velocity = 0.0

        # 检查当前哪些按键被按下（未释放）
        active_keys = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        # 线性移动（W/S）——这些具有优先级
        if "w" in active_keys:
            linear_velocity = self.current_linear_speed
        elif "s" in active_keys:
            linear_velocity = -self.current_linear_speed

        # 转向（A/D/Q/E）
        if "d" in active_keys:
            angular_velocity = -self.current_angular_speed
            if linear_velocity == 0:  # 如果未在前进/后退，则添加少量前进运动
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "a" in active_keys:
            angular_velocity = self.current_angular_speed
            if linear_velocity == 0:  # 如果未在前进/后退，则添加少量前进运动
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "q" in active_keys:
            angular_velocity = self.current_angular_speed
            linear_velocity = 0  # 原地旋转
        elif "e" in active_keys:
            angular_velocity = -self.current_angular_speed
            linear_velocity = 0  # 原地旋转

        # 停止（X）——覆盖一切
        if "x" in active_keys:
            linear_velocity = 0
            angular_velocity = 0

        # 速度调节
        if "+" in active_keys or "=" in active_keys:
            self.current_linear_speed += self.config.speed_increment
            self.current_angular_speed += self.config.speed_increment * self.config.angular_speed_ratio
            logging.info(
                f"Speed increased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )
        if "-" in active_keys:
            self.current_linear_speed = max(
                self.config.min_linear_speed, self.current_linear_speed - self.config.speed_increment
            )
            self.current_angular_speed = max(
                self.config.min_angular_speed,
                self.current_angular_speed - self.config.speed_increment * self.config.angular_speed_ratio,
            )
            logging.info(
                f"Speed decreased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }
