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

"""
本脚本为已有的 LeRobot 数据集增补分位数统计信息。

在添加分位数功能之前创建的大多数数据集，其元数据中不包含
分位数统计信息（q01、q10、q50、q90、q99）。本脚本会：

1. 加载 v3.0 格式的已有 LeRobot 数据集
2. 检查它是否已包含分位数统计信息
3. 若缺失，则为所有特征计算分位数统计信息
4. 用新的分位数统计信息更新数据集元数据

统计信息在所有 episode 中累积到每个特征一个滚动直方图中，
而不是聚合每个 episode 的分位数摘要。
得到的分位数是直方图近似值，会受到离散化和区间重分箱误差的影响；
图像/视频帧默认采用采样方式。

用法：

```bash
python src/lerobot/scripts/augment_dataset_quantile_stats.py \
    --repo-id=lerobot/pusht \
```
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from requests import HTTPError
from tqdm import tqdm

from lerobot.datasets import (
    CODEBASE_VERSION,
    DEFAULT_QUANTILES,
    LeRobotDataset,
    get_feature_stats,
    write_stats,
)
from lerobot.datasets.compute_stats import RunningQuantileStats, sample_indices
from lerobot.utils.utils import init_logging


def has_quantile_stats(stats: dict[str, dict] | None, quantile_list_keys: list[str] | None = None) -> bool:
    """检查数据集统计信息是否已包含分位数信息。

    参数：
        stats: 数据集统计信息字典

    返回：
        若存在分位数统计信息则返回 True，否则返回 False
    """
    if quantile_list_keys is None:
        quantile_list_keys = [f"q{int(q * 100):02d}" for q in DEFAULT_QUANTILES]

    if stats is None:
        return False

    for feature_stats in stats.values():
        if any(q_key in feature_stats for q_key in quantile_list_keys):
            return True

    return False


def collect_episode_arrays(
    dataset: LeRobotDataset,
    episode_idx: int,
    use_sampling: bool = True,
    skip_images: bool = False,
) -> dict[str, tuple[np.ndarray, int]]:
    """按特征收集单个 episode 的帧，并展平为 (num_samples, dim)。

    参数：
        dataset: LeRobot 数据集
        episode_idx: 要读取的 episode 的索引
        use_sampling: 若为 True，则对图像/视频帧进行子采样以限制内存占用。
            若为 False，则使用每一帧（内存占用更高）。
        skip_images: 若为 True，则完全跳过图像/视频特征。

    返回：
        从特征名到该 episode 的值及其来源帧数的映射
        （对于图像特征，帧数与行数不同）。
    """
    start_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    end_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]

    episode_len = end_idx - start_idx

    # 图像/视频是内存大户，因此对每个 episode 的这些帧进行子采样；
    # 数值列开销小，因此完整读取（精确）。
    image_keys = [k for k in dataset.features if dataset.features[k]["dtype"] in ("image", "video")]
    numeric_keys = [
        k
        for k in dataset.features
        if dataset.features[k]["dtype"] not in ("image", "video", "string", "language")
    ]

    collected_data: dict[str, list] = {}

    # 数值特征：每一帧都直接从底层表中读取。
    if numeric_keys:
        numeric_cols = dataset.hf_dataset.select_columns(numeric_keys)[start_idx:end_idx]
        for key in numeric_keys:
            collected_data[key] = [torch.as_tensor(v) for v in numeric_cols[key]]

    # 图像/视频特征：只解码采样子集的帧。
    if image_keys and not skip_images:
        sampled_offsets = sample_indices(episode_len) if use_sampling else list(range(episode_len))
        for offset in sampled_offsets:
            item = dataset[start_idx + offset]
            for key in image_keys:
                if key in item:
                    collected_data.setdefault(key, []).append(item[key])

    episode_arrays: dict[str, tuple[np.ndarray, int]] = {}
    for key, data_list in collected_data.items():
        data = torch.stack(data_list).cpu().numpy()
        if dataset.features[key]["dtype"] in ["image", "video"]:
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0
            # (N, C, H, W) -> (N * H * W, C)，以便按通道计算分位数。
            channels = data.shape[1]
            values = data.transpose(0, 2, 3, 1).reshape(-1, channels)
        else:
            values = data.reshape(-1, data.shape[-1]) if data.ndim > 1 else data.reshape(-1, 1)
        episode_arrays[key] = (values, len(data_list))

    return episode_arrays


def compute_quantile_stats_for_dataset(
    dataset: LeRobotDataset,
    use_sampling: bool = True,
    skip_images: bool = False,
) -> dict[str, dict]:
    """使用每个特征一个滚动直方图来计算整个数据集的统计信息。

    参数：
        dataset: 要计算统计信息的 LeRobot 数据集
        use_sampling: 若为 True，则对每个 episode 的图像/视频帧进行子采样
            以限制内存占用。若为 False，则使用每一帧（内存占用更高）。
        skip_images: 若为 True，则跳过图像/视频特征并保持其统计信息不变。

    返回：
        包含基于直方图的全局分位数估计的统计信息字典

    注意：
        由于滚动累加器在所有 episode 间共享，
        因此各 episode 按顺序累积。
    """
    logging.info(f"Computing quantile statistics for dataset with {dataset.num_episodes} episodes")

    running_stats: dict[str, RunningQuantileStats] = {}
    frame_counts: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    # 仅当特征只有一行时保留，以便仍能完成收尾处理。
    single_row_arrays: dict[str, np.ndarray] = {}

    for episode_idx in tqdm(range(dataset.num_episodes), desc="Processing episodes"):
        episode_arrays = collect_episode_arrays(
            dataset, episode_idx, use_sampling=use_sampling, skip_images=skip_images
        )
        for key, (array, num_frames) in episode_arrays.items():
            running_stats.setdefault(key, RunningQuantileStats()).update(array)
            frame_counts[key] = frame_counts.get(key, 0) + num_frames
            row_counts[key] = row_counts.get(key, 0) + len(array)
            if row_counts[key] < 2:
                single_row_arrays[key] = array
            else:
                single_row_arrays.pop(key, None)

    if not running_stats:
        raise ValueError("No episode data found for computing statistics")

    aggregated_stats: dict[str, dict] = {}
    for key, accumulator in running_stats.items():
        if row_counts[key] < 2:
            # 直方图至少需要两个样本；与 get_feature_stats 的基础统计路径保持一致。
            stats = get_feature_stats(single_row_arrays[key], axis=0, keepdims=False)
        else:
            stats = accumulator.get_statistics()
        if dataset.features[key]["dtype"] in ["image", "video"]:
            # 图像统计信息以 (C, 1, 1) 存储，以便在高度和宽度上广播。
            stats = {k: v if k == "count" else v[:, np.newaxis, np.newaxis] for k, v in stats.items()}
        # `get_feature_stats` 统计的是帧数，而不是累加器看到的按通道行数。
        stats["count"] = np.array([frame_counts[key]])
        aggregated_stats[key] = stats

    logging.info(f"Computed global histogram statistics for {len(aggregated_stats)} features")
    return aggregated_stats


def augment_dataset_with_quantile_stats(
    repo_id: str,
    root: str | Path | None = None,
    overwrite: bool = False,
    use_sampling: bool = True,
    skip_images: bool = False,
) -> None:
    """若数据集缺少分位数统计信息，则为其增补。

    参数：
        repo_id: 数据集的仓库 ID
        root: 数据集的本地根目录
        overwrite: 若分位数统计信息已存在则覆盖
        use_sampling: 若为 True，则对每个 episode 的图像/视频帧进行子采样
            以限制内存占用。若为 False，则使用每一帧（内存占用更高）。
        skip_images: 若为 True，则跳过图像/视频特征并保留其已有统计信息
    """
    logging.info(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        download_videos=not skip_images,
    )

    if not overwrite and has_quantile_stats(dataset.meta.stats):
        logging.info("Dataset already contains quantile statistics. No action needed.")
        return

    logging.info("Dataset does not contain quantile statistics. Computing them now...")

    new_stats = compute_quantile_stats_for_dataset(
        dataset, use_sampling=use_sampling, skip_images=skip_images
    )

    if skip_images and dataset.meta.stats:
        for key, feature_stats in dataset.meta.stats.items():
            new_stats.setdefault(key, feature_stats)

    logging.info("Updating dataset metadata with new quantile statistics")
    dataset.meta.stats = new_stats

    write_stats(new_stats, dataset.meta.root)

    logging.info("Successfully updated dataset with quantile statistics")
    dataset.push_to_hub()

    hub_api = HfApi()
    try:
        hub_api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    except HTTPError as e:
        logging.info(f"tag={CODEBASE_VERSION} probably doesn't exist. Skipping exception ({e})")
        pass
    hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=None, repo_type="dataset")


def main():
    """运行增补脚本的主函数。"""
    parser = argparse.ArgumentParser(description="Augment LeRobot dataset with quantile statistics")

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID of the dataset (e.g., 'lerobot/pusht')",
    )

    parser.add_argument(
        "--root",
        type=str,
        help="Local root directory for the dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing quantile statistics if they already exist",
    )
    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help=(
            "Compute stats over every frame (higher memory). By default, "
            "image/video frames are sub-sampled per episode to bound memory."
        ),
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image/video features and preserve their existing stats",
    )

    args = parser.parse_args()
    root = Path(args.root) if args.root else None

    init_logging()

    augment_dataset_with_quantile_stats(
        repo_id=args.repo_id,
        root=root,
        overwrite=args.overwrite,
        use_sampling=not args.no_sampling,
        skip_images=args.skip_images,
    )


if __name__ == "__main__":
    main()
