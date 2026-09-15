#!/usr/bin/env python

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

from typing import Any

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected


class BimanualMixin:
    """双臂机器人和遥操作设备的生命周期委托。

    具体子类必须在自己的 ``__init__`` 中填充 ``self.left_arm`` 和 ``self.right_arm``。
    它们保留对特征字典和数据路由方法（``get_action`` / ``send_action`` /
    ``get_observation`` / ``send_feedback``）的所有权，
    因为这些方法因本体形态而异。

    继承时应放在 ``Robot`` / ``Teleoperator`` 基类之前，
    使混入类的方法在 MRO 中优先::

        class BiFooFollower(BimanualMixin, Robot): ...
    """

    left_arm: Any
    right_arm: Any

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
