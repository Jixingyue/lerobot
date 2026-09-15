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

import abc
import builtins
from pathlib import Path
from typing import Any

import draccus

from lerobot.lerobot_types import RobotAction
from lerobot.motors.motors_bus import MotorCalibration
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

from .config import TeleoperatorConfig


class Teleoperator(abc.ABC):
    """
    所有 LeRobot 兼容遥操作设备的基抽象类。

    该类提供了与物理遥操作设备交互的标准化接口。
    子类必须实现所有抽象方法和属性才能使用。

    Attributes:
        config_class (RobotConfig): 该遥操作设备预期的配置类。
        name (str): 用于标识该遥操作设备类型的唯一名称。
    """

    # 在所有子类中设置这些
    config_class: builtins.type[TeleoperatorConfig]
    name: str

    def __init__(self, config: TeleoperatorConfig):
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir
            if config.calibration_dir
            else HF_LEROBOT_CALIBRATION / TELEOPERATORS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    def __enter__(self):
        """
        上下文管理器入口。
        自动连接到相机。
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        上下文管理器出口。
        自动断开连接，确保即使发生错误也能释放资源。
        """
        self.disconnect()

    def __del__(self) -> None:
        """
        析构函数安全网。
        如果对象在未清理的情况下被垃圾回收，尝试断开连接。
        """
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        """
        一个描述遥操作设备产生的动作的结构和类型的字典。其结构（键）应与
        :pymeth:`get_action` 返回内容的结构一致。字典的值应为对应值的类型
        （如果是简单值），例如单个本体感知值（关节的目标位置/速度）对应 `float`

        注意：无论机器人是否已连接，都应能调用此属性。
        """
        pass

    @property
    @abc.abstractmethod
    def feedback_features(self) -> dict:
        """
        一个描述机器人预期的反馈动作的结构和类型的字典。其结构（键）应与
        传递给 :pymeth:`send_feedback` 的内容的结构一致。字典的值应为对应值的
        类型（如果是简单值），例如单个本体感知值（关节的目标位置/速度）对应 `float`

        注意：无论机器人是否已连接，都应能调用此属性。
        """
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        遥操作设备当前是否已连接。如果为 `False`，调用 :pymeth:`get_action`
        或 :pymeth:`send_feedback` 应抛出错误。
        """
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """
        建立与遥操作设备的通信。

        Args:
            calibrate (bool): 如果为 True，连接后自动校准遥操作设备（如果尚未校准
                或需要校准，具体取决于硬件）。
        """
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """遥操作设备当前是否已校准。如果不适用，应始终为 `True`"""
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """
        如果适用，校准遥操作设备。如果不适用，应为空操作。

        该方法应收集任何必要的数据（例如电机偏移量），并相应地更新
        :pyattr:`calibration` 字典。
        """
        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """
        从指定文件加载校准数据的辅助方法。

        Args:
            fpath (Path | None): 校准文件的可选路径。默认为 `self.calibration_fpath`。
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            self.calibration = draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """
        将校准数据保存到指定文件的辅助方法。

        Args:
            fpath (Path | None): 保存校准文件的可选路径。默认为 `self.calibration_fpath`。
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @abc.abstractmethod
    def configure(self) -> None:
        """
        对遥操作设备应用任何一次性或运行时配置。
        这可能包括设置电机参数、控制模式或初始状态。
        """
        pass

    @abc.abstractmethod
    def get_action(self) -> RobotAction:
        """
        从遥操作设备获取当前动作。

        Returns:
            RobotAction: 表示遥操作设备当前动作的扁平字典。其结构应与
                :pymeth:`observation_features` 一致。
        """
        pass

    @abc.abstractmethod
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """
        向遥操作设备发送反馈动作命令。

        Args:
            feedback (dict[str, Any]): 表示期望反馈的字典。其结构应与
                :pymeth:`feedback_features` 一致。

        Returns:
            dict[str, Any]: 实际发送给电机的动作，可能经过裁剪或修改，
                例如被速度安全限制所修改。
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开与遥操作设备的连接并执行任何必要的清理。"""
        pass
