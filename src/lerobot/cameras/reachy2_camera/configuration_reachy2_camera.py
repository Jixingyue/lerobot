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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode

__all__ = ["CameraConfig", "ColorMode", "Reachy2CameraConfig"]


@CameraConfig.register_subclass("reachy2_camera")
@dataclass
class Reachy2CameraConfig(CameraConfig):
    """Reachy 2 相机设备的配置类。

    该类为 Reachy 2 相机提供配置选项，
    支持 teleop 和 depth 两种相机。它包含
    分辨率、帧率、颜色模式以及相机选择的设置。

    配置示例：
    ```python
    # 基本配置
    Reachy2CameraConfig(
        name="teleop",
        image_type="left",
        ip_address="192.168.0.200",  # 机器人的 IP 地址
        port=50065,  # 相机服务器的端口
        width=640,
        height=480,
        fps=30,  # Reachy 2 相机不可配置
        color_mode=ColorMode.RGB,
    )  # 左侧 teleop 相机，640x480 @ 30FPS
    ```

    属性：
        name: 相机设备名称。可以是 "teleop" 或 "depth"。
        image_type: 图像流类型。对于 "teleop" 相机，可以是 "left" 或 "right"。
                    对于 "depth" 相机，可以是 "rgb" 或 "depth"。（depth 尚不支持）
        fps: 请求的彩色流每秒帧数。Reachy 2 相机不可配置。
        width: 请求的彩色流帧宽度（像素）。
        height: 请求的彩色流帧高度（像素）。
        color_mode: 图像输出的颜色模式（RGB 或 BGR）。默认为 RGB。
        ip_address: 机器人的 IP 地址。默认为 "localhost"。
        port: 相机服务器的端口号。默认为 50065。

    注意：
        - 目前仅支持 3 通道彩色输出（RGB/BGR）。
    """

    name: str
    image_type: str
    color_mode: ColorMode = ColorMode.RGB
    ip_address: str | None = "localhost"
    port: int = 50065

    def __post_init__(self) -> None:
        if self.name not in ["teleop", "depth"]:
            raise ValueError(f"`name` is expected to be 'teleop' or 'depth', but {self.name} is provided.")
        if (self.name == "teleop" and self.image_type not in ["left", "right"]) or (
            self.name == "depth" and self.image_type not in ["rgb", "depth"]
        ):
            raise ValueError(
                f"`image_type` is expected to be 'left' or 'right' for teleop camera, and 'rgb' or 'depth' for depth camera, but {self.image_type} is provided."
            )

        self.color_mode = ColorMode(self.color_mode)
