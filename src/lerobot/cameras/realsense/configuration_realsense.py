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

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class RealSenseCameraConfig(CameraConfig):
    """Intel RealSense 相机的配置类。

    该类为 Intel RealSense 相机提供专门的配置选项，
    包括深度感知支持以及通过序列号或名称进行设备标识。

    Intel RealSense D405 的配置示例：
    ```python
    # 基本配置
    RealSenseCameraConfig("0123456789", 30, 1280, 720)  # 1280x720 @ 30FPS
    RealSenseCameraConfig("0123456789", 60, 640, 480)  # 640x480 @ 60FPS

    # 高级配置
    RealSenseCameraConfig("0123456789", 30, 640, 480, use_depth=True)  # 带深度感知
    RealSenseCameraConfig("0123456789", 30, 640, 480, rotation=Cv2Rotation.ROTATE_90)  # 带 90° 旋转
    ```

    属性：
        fps: 请求的彩色流每秒帧数。
        width: 请求的彩色流帧宽度（像素）。
        height: 请求的彩色流帧高度（像素）。
        serial_number_or_name: 用于标识相机的唯一序列号或人类可读名称。
        color_mode: 图像输出的颜色模式（RGB 或 BGR）。默认为 RGB。
        use_rgb: 是否启用彩色流。默认为 True。
        use_depth: 是否启用深度流。默认为 False。
        rotation: 图像旋转设置（0°、90°、180° 或 270°）。默认不旋转。
        warmup_s: connect 返回前读取帧的时间（秒）
        exposure: 彩色传感器的手动曝光值。设置后会禁用自动曝光，
            并使用此固定值。有效范围因相机型号而异，
            如果值被拒绝会报告范围。默认为 None（保持不变）。
        gain: 彩色传感器的手动增益值。设置后会禁用自动曝光，
            并使用此固定增益；当未配置 exposure 时，曝光也会
            冻结在当前值。有效范围因相机型号而异，
            如果值被拒绝会报告范围。默认为 None（保持不变）。
        white_balance: 彩色传感器的手动白平衡值。设置后会禁用
            自动白平衡，并使用此固定值。有效范围
            因相机型号而异，如果值被拒绝会报告范围。默认为 None
            （保持不变）。

    注意：
        - 必须指定名称或序列号之一。
        - `use_rgb` 或 `use_depth` 至少需要启用一个。
        - 深度流配置（如果启用）将使用与彩色流相同的 FPS。
        - 实际分辨率和 FPS 可能被相机调整为最接近的支持模式。
        - 对于 `fps`、`width` 和 `height`，要么全部设置，要么全都不设置。
    """

    serial_number_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    use_rgb: bool = True
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    exposure: int | None = None
    gain: int | None = None
    white_balance: int | None = None

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of `use_rgb` or `use_depth` must be enabled.")

        manual_color_options = {
            "exposure": self.exposure,
            "gain": self.gain,
            "white_balance": self.white_balance,
        }
        configured_color_options = [name for name, value in manual_color_options.items() if value is not None]
        if configured_color_options and not self.use_rgb:
            raise ValueError(
                "Manual color sensor options require `use_rgb=True`. "
                f"Configured options: {configured_color_options}."
            )

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
