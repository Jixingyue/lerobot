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

"""
查找系统中可用的相机设备的辅助工具。

示例：

```shell
lerobot-find-cameras
```
"""

# 注意（Steven）：RealSense 也可以被识别/打开为 OpenCV 相机。如果你知道相机是 RealSense，请使用 `lerobot-find-cameras realsense` 参数以避免混淆。
# 注意（Steven）：macOS 相机在初始化时有时会报告不同的 FPS，在这里不是问题，因为我们打开相机时不指定 FPS，但显示的信息可能不准确。

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lerobot.cameras import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def find_all_opencv_cameras() -> list[dict[str, Any]]:
    """
    查找插入系统的所有可用 OpenCV 相机。

    返回：
        所有可用 OpenCV 相机及其元数据的列表。
    """
    all_opencv_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for OpenCV cameras...")
    try:
        opencv_cameras = OpenCVCamera.find_cameras()
        for cam_info in opencv_cameras:
            all_opencv_cameras_info.append(cam_info)
        logger.info(f"Found {len(opencv_cameras)} OpenCV cameras.")
    except Exception as e:
        logger.error(f"Error finding OpenCV cameras: {e}")

    return all_opencv_cameras_info


def find_all_realsense_cameras() -> list[dict[str, Any]]:
    """
    查找插入系统的所有可用 RealSense 相机。

    返回：
        所有可用 RealSense 相机及其元数据的列表。
    """
    all_realsense_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for RealSense cameras...")
    try:
        realsense_cameras = RealSenseCamera.find_cameras()
        for cam_info in realsense_cameras:
            all_realsense_cameras_info.append(cam_info)
        logger.info(f"Found {len(realsense_cameras)} RealSense cameras.")
    except ImportError:
        logger.warning("Skipping RealSense camera search: pyrealsense2 library not found or not importable.")
    except Exception as e:
        logger.error(f"Error finding RealSense cameras: {e}")

    return all_realsense_cameras_info


def find_and_print_cameras(camera_type_filter: str | None = None) -> list[dict[str, Any]]:
    """
    根据可选的过滤条件查找可用相机并打印其信息。

    参数：
        camera_type_filter: 用于过滤相机的可选字符串（"realsense" 或 "opencv"）。
                            若为 None，则列出所有相机。

    返回：
        所有匹配过滤条件的可用相机及其元数据的列表。
    """
    all_cameras_info: list[dict[str, Any]] = []

    if camera_type_filter:
        camera_type_filter = camera_type_filter.lower()

    if camera_type_filter is None or camera_type_filter == "opencv":
        all_cameras_info.extend(find_all_opencv_cameras())
    if camera_type_filter is None or camera_type_filter == "realsense":
        all_cameras_info.extend(find_all_realsense_cameras())

    if not all_cameras_info:
        if camera_type_filter:
            logger.warning(f"No {camera_type_filter} cameras were detected.")
        else:
            logger.warning("No cameras (OpenCV or RealSense) were detected.")
    else:
        print("\n--- Detected Cameras ---")
        for i, cam_info in enumerate(all_cameras_info):
            print(f"Camera #{i}:")
            for key, value in cam_info.items():
                if key == "default_stream_profile" and isinstance(value, dict):
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key.capitalize()}: {sub_value}")
                else:
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
            print("-" * 20)
    return all_cameras_info


def save_image(
    img_array: np.ndarray,
    camera_identifier: str | int,
    images_dir: Path,
    camera_type: str,
) -> None:
    """
    使用 Pillow 将单张图像保存到磁盘。如有需要会处理颜色转换。
    """
    try:
        img = Image.fromarray(img_array, mode="RGB")

        safe_identifier = str(camera_identifier).replace("/", "_").replace("\\", "_")
        filename_prefix = f"{camera_type.lower()}_{safe_identifier}"
        filename = f"{filename_prefix}.png"

        path = images_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        logger.info(f"Saved image: {path}")
    except Exception as e:
        logger.error(f"Failed to save image for camera {camera_identifier} (type {camera_type}): {e}")


def create_camera_instance(cam_meta: dict[str, Any], *, warmup_s: int = 1) -> dict[str, Any] | None:
    """根据元数据创建并连接相机实例。"""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    instance = None

    logger.info(f"Preparing {cam_type} ID {cam_id} with default profile")

    try:
        if cam_type == "OpenCV":
            cv_config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
                warmup_s=warmup_s,
            )
            instance = OpenCVCamera(cv_config)
        elif cam_type == "RealSense":
            rs_config = RealSenseCameraConfig(
                serial_number_or_name=cam_id,
                color_mode=ColorMode.RGB,
                warmup_s=warmup_s,
            )
            instance = RealSenseCamera(rs_config)
        else:
            logger.warning(f"Unknown camera type: {cam_type} for ID {cam_id}. Skipping.")
            return None

        if instance:
            logger.info(f"Connecting to {cam_type} camera: {cam_id}...")
            instance.connect(warmup=True)
            return {"instance": instance, "meta": cam_meta}
    except Exception as e:
        logger.error(f"Failed to connect or configure {cam_type} camera {cam_id}: {e}")
        if instance and instance.is_connected:
            instance.disconnect()
        return None


def process_camera_image(cam_dict: dict[str, Any], output_dir: Path, current_time: float) -> None:
    """从单个相机捕获并处理一张图像。"""
    cam = cam_dict["instance"]
    meta = cam_dict["meta"]
    cam_type_str = str(meta.get("type", "unknown"))
    cam_id_str = str(meta.get("id", "unknown"))

    try:
        image_data = cam.read()

        save_image(
            image_data,
            cam_id_str,
            output_dir,
            cam_type_str,
        )
    except TimeoutError:
        logger.warning(
            f"Timeout reading from {cam_type_str} camera {cam_id_str} at time {current_time:.2f}s."
        )
    except Exception as e:
        logger.error(f"Error reading from {cam_type_str} camera {cam_id_str}: {e}")
    return None


def cleanup_camera(cam_dict: dict[str, Any]) -> None:
    """断开所有相机的连接。"""
    logger.info(f"Disconnecting camera with ID {cam_dict['meta'].get('id')}...")
    try:
        if cam_dict["instance"] and cam_dict["instance"].is_connected:
            cam_dict["instance"].disconnect()
    except Exception as e:
        logger.error(f"Error disconnecting camera {cam_dict['meta'].get('id')}: {e}")


def save_images_from_all_cameras(
    output_dir: Path,
    record_time_s: float = 2.0,
    camera_type: str | None = None,
    warmup_s: int = 1,
):
    """
    连接检测到的相机（可选按类型过滤）并保存每个相机的图像。
    宽度、高度和 FPS 使用默认的流配置文件。

    参数：
        output_dir: 保存图像的目录。
        record_time_s: 录制图像的时长（秒）。
        camera_type: 用于过滤相机的可选字符串（"realsense" 或 "opencv"）。
                            若为 None，则使用所有检测到的相机。
        warmup_s: 录制图像前预热相机的时长（秒）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving images to {output_dir}")
    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)

    if not all_camera_metadata:
        logger.warning("No cameras detected matching the criteria. Cannot save images.")
        return

    logger.info(
        f"Starting image capture for {record_time_s} seconds from {len(all_camera_metadata)} cameras."
    )

    try:
        for cam_meta in all_camera_metadata:
            cam_dict = create_camera_instance(cam_meta, warmup_s=warmup_s)
            if cam_dict is None:
                continue
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < record_time_s:
                current_capture_time = time.perf_counter()
                process_camera_image(cam_dict, output_dir, current_capture_time)
            cleanup_camera(cam_dict)
    except KeyboardInterrupt:
        logger.info("Capture interrupted by user.")
    finally:
        print(f"Image capture finished. Images saved to {output_dir}")


def main():
    init_logging()

    parser = argparse.ArgumentParser(
        description="Unified camera utility script for listing cameras and capturing images."
    )
    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv"],
        help="Specify camera type to capture from (e.g., 'realsense', 'opencv'). Captures from all if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/captured_images",
        help="Directory to save images. Default: outputs/captured_images",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Time duration to attempt capturing frames. Default: 2 seconds.",
    )
    parser.add_argument(
        "--warmup-s",
        type=int,
        default=1,
        help="Time duration to warmup camera before attempting to capture frames. Default: 1 second.",
    )
    args = parser.parse_args()
    save_images_from_all_cameras(**vars(args))


if __name__ == "__main__":
    main()
