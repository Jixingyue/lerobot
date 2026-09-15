#!/usr/bin/env python

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

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as F  # type: ignore  # noqa: N812
from tqdm import tqdm  # type: ignore

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import DONE, REWARD


def select_rect_roi(img):
    """
    允许用户在图像上绘制一个矩形 ROI。

    用户必须点击并拖动来绘制矩形。
    - 拖动过程中，矩形会被动态绘制。
    - 松开鼠标按键后，矩形被固定。
    - 按 'c' 确认选择。
    - 按 'r' 重置选择。
    - 按 ESC 取消。

    Returns:
        表示矩形 ROI 的元组 (top, left, height, width)，
        如果未选择有效的 ROI，则返回 None。
    """
    # 创建图像的工作副本
    clone = img.copy()
    working_img = clone.copy()

    roi = None  # 将以 (top, left, height, width) 存储最终的 ROI
    drawing = False
    index_x, index_y = -1, -1  # 初始点击坐标

    def mouse_callback(event, x, y, flags, param):
        nonlocal index_x, index_y, drawing, roi, working_img

        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始绘制：记录起始坐标
            drawing = True
            index_x, index_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # 无论拖动方向如何，都计算左上角和右下角
                top = min(index_y, y)
                left = min(index_x, x)
                bottom = max(index_y, y)
                right = max(index_x, x)
                # 显示绘制了当前矩形的临时图像
                temp = working_img.copy()
                cv2.rectangle(temp, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.imshow("Select ROI", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            # 完成绘制
            drawing = False
            top = min(index_y, y)
            left = min(index_x, x)
            bottom = max(index_y, y)
            right = max(index_x, x)
            height = bottom - top
            width = right - left
            roi = (top, left, height, width)  # (top, left, height, width)
            # 在工作图像上绘制最终矩形并显示
            working_img = clone.copy()
            cv2.rectangle(working_img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow("Select ROI", working_img)

    # 创建窗口并设置回调
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)
    cv2.imshow("Select ROI", working_img)

    print("Instructions for ROI selection:")
    print("  - Click and drag to draw a rectangular ROI.")
    print("  - Press 'c' to confirm the selection.")
    print("  - Press 'r' to reset and draw again.")
    print("  - Press ESC to cancel the selection.")

    # 等待用户按 'c' 确认、按 'r' 重置或按 ESC 取消
    while True:
        key = cv2.waitKey(1) & 0xFF
        # 如果已绘制 ROI，则确认
        if key == ord("c") and roi is not None:
            break
        # 重置：清除 ROI 并恢复原始图像
        elif key == ord("r"):
            working_img = clone.copy()
            roi = None
            cv2.imshow("Select ROI", working_img)
        # 取消对这张图像的选择
        elif key == 27:  # ESC 键
            roi = None
            break

    cv2.destroyWindow("Select ROI")
    return roi


def select_square_roi_for_images(images: dict) -> dict:
    """
    针对所提供字典中的每张图像，打开一个窗口让用户选择一个矩形 ROI。
    返回一个字典，将每个键映射到表示 ROI 的元组 (top, left, height, width)。

    Parameters:
        images (dict): 键为标识符、值为 OpenCV 图像的字典。

    Returns:
        dict: 图像键到所选矩形 ROI 的映射。
    """
    selected_rois = {}

    for key, img in images.items():
        if img is None:
            print(f"Image for key '{key}' is None, skipping.")
            continue

        print(f"\nSelect rectangular ROI for image with key: '{key}'")
        roi = select_rect_roi(img)

        if roi is None:
            print(f"No valid ROI selected for '{key}'.")
        else:
            selected_rois[key] = roi
            print(f"ROI for '{key}': {roi}")

    return selected_rois


def get_image_from_lerobot_dataset(dataset: LeRobotDataset):
    """
    找到数据集中的第一行并提取图像，以便用于裁剪。
    """
    row = dataset[0]
    image_dict = {}
    for k in row:
        if "image" in k:
            image_dict[k] = deepcopy(row[k])
    return image_dict


def convert_lerobot_dataset_to_cropped_lerobot_dataset(
    original_dataset: LeRobotDataset,
    crop_params_dict: dict[str, tuple[int, int, int, int]],
    new_repo_id: str,
    new_dataset_root: str,
    resize_size: tuple[int, int] = (128, 128),
    push_to_hub: bool = False,
    task: str = "",
) -> LeRobotDataset:
    """
    通过遍历现有 LeRobotDataset 的回合和帧来转换它，
    对图像观测应用裁剪和缩放，并将转换后的数据保存为新数据集。

    Args:
        original_dataset (LeRobotDataset): 源数据集。
        crop_params_dict (dict[str, Tuple[int, int, int, int]]):
            将观测键映射到裁剪参数 (top, left, height, width) 的字典。
        new_repo_id (str): 新数据集的仓库 id。
        new_dataset_root (str): 新数据集写入的根目录。
        resize_size (tuple[int, int], optional): 裁剪后的目标尺寸 (height, width)。
            默认为 (128, 128)。

    Returns:
        LeRobotDataset: 一个对指定图像观测进行了裁剪和缩放的新 LeRobotDataset。
    """
    # 1. 创建一个新的（空的）LeRobotDataset 用于写入。
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=int(original_dataset.fps),
        root=new_dataset_root,
        robot_type=original_dataset.meta.robot_type,
        features=original_dataset.meta.info.features,
        use_videos=len(original_dataset.meta.video_keys) > 0,
    )

    # 更新每个将被裁剪的图像键的元数据：
    # （这里我们直接将形状设置为最终的 resize_size。）
    for key in crop_params_dict:
        if key in new_dataset.meta.info.features:
            new_dataset.meta.info.features[key]["shape"] = (3, *resize_size)

    # TODO: 直接修改 mp4 视频 + 元信息特征，而不是重新创建数据集
    prev_episode_index = 0
    for frame_idx in tqdm(range(len(original_dataset))):
        frame = original_dataset[frame_idx]

        # 创建帧的副本以添加到新数据集
        new_frame = {}
        for key, value in frame.items():
            if key in ("task_index", "timestamp", "episode_index", "frame_index", "index", "task"):
                continue
            if key in (DONE, REWARD):
                # if not isinstance(value, str) and len(value.shape) == 0:
                value = value.unsqueeze(0)

            if key in crop_params_dict:
                top, left, height, width = crop_params_dict[key]
                # 先应用裁剪，再缩放。
                cropped = F.crop(value, top, left, height, width)
                value = F.resize(cropped, resize_size)
                value = value.clamp(0, 1)
            if key.startswith("complementary_info") and isinstance(value, torch.Tensor) and value.dim() == 0:
                value = value.unsqueeze(0)
            new_frame[key] = value

        new_frame["task"] = task
        new_dataset.add_frame(new_frame)

        if frame["episode_index"].item() != prev_episode_index:
            # 保存该回合
            new_dataset.save_episode()
            prev_episode_index = frame["episode_index"].item()

    # 保存最后一个回合
    new_dataset.save_episode()

    if push_to_hub:
        new_dataset.push_to_hub()

    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop rectangular ROIs from a LeRobot dataset.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot",
        help="The repository id of the LeRobot dataset to process.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="The root directory of the LeRobot dataset.",
    )
    parser.add_argument(
        "--crop-params-path",
        type=str,
        default=None,
        help="The path to the JSON file containing the ROIs.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Whether to push the new dataset to the hub.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="The natural language task to describe the dataset.",
    )
    parser.add_argument(
        "--new-repo-id",
        type=str,
        default=None,
        help="The repository id for the new cropped and resized dataset. If not provided, it defaults to `repo_id` + '_cropped_resized'.",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    images = get_image_from_lerobot_dataset(dataset)
    images = {k: v.cpu().permute(1, 2, 0).numpy() for k, v in images.items()}
    images = {k: (v * 255).astype("uint8") for k, v in images.items()}

    if args.crop_params_path is None:
        rois = select_square_roi_for_images(images)
    else:
        with open(args.crop_params_path) as f:
            rois = json.load(f)

    # 打印所选的矩形 ROI
    print("\nSelected Rectangular Regions of Interest (top, left, height, width):")
    for key, roi in rois.items():
        print(f"{key}: {roi}")

    new_repo_id = args.new_repo_id if args.new_repo_id else args.repo_id + "_cropped_resized"

    if args.new_repo_id:
        new_dataset_name = args.new_repo_id.split("/")[-1]
        # 上级目录 1：HF 用户，上级目录 2：HF LeRobot 主目录
        new_dataset_root = dataset.root.parent.parent / new_dataset_name
    else:
        new_dataset_root = Path(str(dataset.root) + "_cropped_resized")

    cropped_resized_dataset = convert_lerobot_dataset_to_cropped_lerobot_dataset(
        original_dataset=dataset,
        crop_params_dict=rois,
        new_repo_id=new_repo_id,
        new_dataset_root=new_dataset_root,
        resize_size=(128, 128),
        push_to_hub=args.push_to_hub,
        task=args.task,
    )

    meta_dir = new_dataset_root / "meta"
    meta_dir.mkdir(exist_ok=True)

    with open(meta_dir / "crop_params.json", "w") as f:
        json.dump(rois, f, indent=4)
