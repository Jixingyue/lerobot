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

"""用于并发观测/动作访问的线程安全机器人包装器。"""

from __future__ import annotations

from threading import Lock
from typing import Any

from lerobot.robots import Robot


class ThreadSafeRobot:
    """围绕 :class:`Robot` 的锁保护包装器，供后台线程使用。

    当 RTC 推理在后台线程运行而主循环在执行动作时，
    两个线程可能同时访问机器人。
    此包装器将 ``get_observation`` 和 ``send_action`` 调用串行化。

    只读属性无需加锁即可代理，因为它们不会
    改变硬件状态。
    """

    def __init__(self, robot: Robot) -> None:
        self._robot = robot
        self._lock = Lock()

    #  -- 锁保护的 I/O --------------------------------------------------
    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            return self._robot.get_observation()

    def send_action(self, action: dict[str, Any] | Any) -> Any:
        with self._lock:
            return self._robot.send_action(action)

    #  -- 只读代理（无需锁）-----------------------------------
    @property
    def observation_features(self) -> dict:
        return self._robot.observation_features

    @property
    def action_features(self) -> dict:
        return self._robot.action_features

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def robot_type(self) -> str:
        return self._robot.robot_type

    @property
    def cameras(self):
        return getattr(self._robot, "cameras", {})

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected

    @property
    def inner(self) -> Robot:
        """访问底层机器人（例如用于连接/断开连接）。"""
        return self._robot
