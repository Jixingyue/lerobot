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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig, ColorMode
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("reachy2")
@dataclass
class Reachy2RobotConfig(RobotConfig):
    # `max_relative_target` 出于安全目的限制相对位置目标向量的大小。
    # 将其设为正标量可让所有电机使用相同的值。
    max_relative_target: float | None = None

    # Reachy 2 机器人的 IP 地址
    ip_address: str | None = "localhost"
    # Reachy 2 机器人的端口
    port: int = 50065

    # 如果为 True，断开连接前会向机器人发送 turn_off_smoothly()。
    disable_torque_on_disconnect: bool = False

    # 外部命令控制的标志
    # 如果你使用外部命令系统来控制机器人，请设为 True，
    # 例如官方遥操作应用：https://github.com/pollen-robotics/Reachy2Teleoperation
    # 如果为 True，robot.send_action() 将不会向机器人发送命令。
    use_external_commands: bool = False

    # 机器人部件
    # 设为 False 可避免将对应部件的关节添加到机器人的关节列表中。
    # 默认情况下，所有部件都设为 True。
    with_mobile_base: bool = True
    with_l_arm: bool = True
    with_r_arm: bool = True
    with_neck: bool = True
    with_antennas: bool = True

    # 机器人相机
    # 如果你想在观测中使用对应的相机，请设为 True。
    # 默认情况下不使用任何相机。
    with_left_teleop_camera: bool = False
    with_right_teleop_camera: bool = False
    with_torso_camera: bool = False

    # 相机参数
    camera_width: int = 640
    camera_height: int = 480

    # 用于 3 个默认 Reachy 2 相机之外的相机。
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 添加与机器人使用相同 ip_address 的相机
        if self.with_left_teleop_camera:
            self.cameras["teleop_left"] = Reachy2CameraConfig(
                name="teleop",
                image_type="left",
                ip_address=self.ip_address,
                port=self.port,
                width=self.camera_width,
                height=self.camera_height,
                fps=30,  # Reachy 2 相机不可配置
                color_mode=ColorMode.RGB,
            )
        if self.with_right_teleop_camera:
            self.cameras["teleop_right"] = Reachy2CameraConfig(
                name="teleop",
                image_type="right",
                ip_address=self.ip_address,
                port=self.port,
                width=self.camera_width,
                height=self.camera_height,
                fps=30,  # Reachy 2 相机不可配置
                color_mode=ColorMode.RGB,
            )
        if self.with_torso_camera:
            self.cameras["torso_rgb"] = Reachy2CameraConfig(
                name="depth",
                image_type="rgb",
                ip_address=self.ip_address,
                port=self.port,
                width=self.camera_width,
                height=self.camera_height,
                fps=30,  # Reachy 2 相机不可配置
                color_mode=ColorMode.RGB,
            )

        super().__post_init__()

        if not (
            self.with_mobile_base
            or self.with_l_arm
            or self.with_r_arm
            or self.with_neck
            or self.with_antennas
        ):
            raise ValueError(
                "No Reachy2Robot part used.\n"
                "At least one part of the robot must be set to True "
                "(with_mobile_base, with_l_arm, with_r_arm, with_neck, with_antennas)"
            )
