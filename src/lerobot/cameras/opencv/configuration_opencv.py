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
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Backends, Cv2Rotation

__all__ = ["OpenCVCameraConfig", "ColorMode", "Cv2Rotation", "Cv2Backends"]


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """基于 OpenCV 的相机设备或视频文件的配置类。

    该类为通过 OpenCV 访问的相机提供配置选项，
    支持物理相机设备和视频文件。它包含
    分辨率、帧率、颜色模式和图像旋转的设置。

    配置示例：
    ```python
    # 基本配置
    OpenCVCameraConfig(0, 30, 1280, 720)   # 1280x720 @ 30FPS
    OpenCVCameraConfig(/dev/video4, 60, 640, 480)   # 640x480 @ 60FPS

    # 带 FOURCC 格式的高级配置
    OpenCVCameraConfig(128422271347, 30, 640, 480, rotation=Cv2Rotation.ROTATE_90, fourcc="MJPG")     # 带 90° 旋转和 MJPG 格式
    OpenCVCameraConfig(0, 30, 1280, 720, fourcc="YUYV")     # 带 YUYV 格式
    ```

    属性：
        index_or_path: 表示相机设备索引的整数，
                      或指向视频文件的 Path 对象。
        fps: 请求的彩色流每秒帧数。
        width: 请求的彩色流帧宽度（像素）。
        height: 请求的彩色流帧高度（像素）。
        color_mode: 图像输出的颜色模式（RGB 或 BGR）。默认为 RGB。
        rotation: 图像旋转设置（0°、90°、180° 或 270°）。默认不旋转。
        warmup_s: connect 返回前读取帧的时间（秒）
        fourcc: 视频格式的 FOURCC 代码（如 "MJPG"、"YUYV"、"I420"）。默认为 None（自动检测）。
        backend: OpenCV 后端标识符 (https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html)。默认为 ANY。

    注意：
        - 目前仅支持 3 通道彩色输出（RGB/BGR）。
        - FOURCC 代码必须是 4 字符字符串（如 "MJPG"、"YUYV"）。一些常见的 FOURCC 代码：https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs#fourcc-constants
        - 设置 FOURCC 有助于在某些相机上获得更高的帧率。
    """

    index_or_path: int | Path
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    fourcc: str | None = None
    backend: Cv2Backends = Cv2Backends.ANY

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.backend = Cv2Backends(self.backend)

        if self.fourcc is not None and (not isinstance(self.fourcc, str) or len(self.fourcc) != 4):
            raise ValueError(
                f"`fourcc` must be a 4-character string (e.g., 'MJPG', 'YUYV'), but '{self.fourcc}' is provided."
            )
