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

"""LeRobotDataset 的数据集工具函数。

本模块提供以下工具：
- 从数据集中删除 episode
- 将数据集拆分为多个更小的数据集
- 向数据集添加/移除特征
- 合并数据集（聚合功能的封装）
"""

import logging
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.configs import (
    DepthEncoderConfig,
    RGBEncoderConfig,
    VideoEncoderConfig,
    depth_encoder_defaults,
    encoder_config_from_video_info,
    rgb_encoder_defaults,
)
from lerobot.configs.video import DEPTH_ENCODER_INFO_FIELD_NAMES
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_IMAGE, OBS_STATE
from lerobot.utils.utils import flatten_dict

from .aggregate import aggregate_datasets
from .compute_stats import (
    aggregate_stats,
    compute_episode_stats,
    compute_relative_action_stats,
)
from .dataset_metadata import LeRobotDatasetMetadata
from .image_writer import write_image
from .io_utils import (
    get_parquet_file_size_in_mb,
    load_episodes,
    write_info,
    write_stats,
    write_tasks,
)
from .lerobot_dataset import LeRobotDataset
from .utils import (
    DATA_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEPTH_FILE_PATTERN,
    IMAGE_FILE_PATTERN,
    VIDEO_DIR,
    update_chunk_file_indices,
)
from .video_utils import (
    encode_video_frames,
    reencode_video,
)


def _load_episode_with_stats(src_dataset: LeRobotDataset, episode_idx: int) -> dict:
    """从 parquet 文件加载单个 episode 的元数据（包括统计信息）。

    Args:
        src_dataset: 源数据集
        episode_idx: 要加载的 episode 索引

    Returns:
        包含 episode 元数据和统计信息的字典
    """
    ep_meta = src_dataset.meta.episodes[episode_idx]
    chunk_idx = ep_meta["meta/episodes/chunk_index"]
    file_idx = ep_meta["meta/episodes/file_index"]

    parquet_path = src_dataset.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    df = pd.read_parquet(parquet_path)

    episode_row = df[df["episode_index"] == episode_idx].iloc[0]

    return episode_row.to_dict()


def delete_episodes(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """从 LeRobotDataset 中删除 episode 并创建一个新数据集。

    需要重新编码的视频片段（因为源文件混合了保留和删除的 episode）
    会使用源数据集现有的编码器设置重新编码——从 ``meta/info.json``
    读回——从而使输出数据集与其自身的元数据保持一致。

    Args:
        dataset: 源 LeRobotDataset。
        episode_indices: 要删除的 episode 索引列表。
        output_dir: 编辑后数据集的存储根目录。如果未指定，默认为 $HF_LEROBOT_HOME/repo_id。等同于 EditDatasetConfig 中的 new_root。
        repo_id: 编辑后的数据集标识符。等同于 EditDatasetConfig 中的 new_repo_id。
    """
    if not episode_indices:
        raise ValueError("No episodes to delete")

    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = set(episode_indices) - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    logging.info(f"Deleting {len(episode_indices)} episodes from dataset")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    episodes_to_keep = [i for i in range(dataset.meta.total_episodes) if i not in episode_indices]
    if not episodes_to_keep:
        raise ValueError("Cannot delete all episodes from dataset")

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(episodes_to_keep)}

    video_metadata = None
    if dataset.meta.video_keys:
        video_metadata = _copy_and_reindex_videos(dataset, new_meta, episode_mapping)

    data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)

    _copy_and_reindex_episodes_metadata(dataset, new_meta, episode_mapping, data_metadata, video_metadata)

    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    logging.info(f"Created new dataset with {len(episodes_to_keep)} episodes")
    return new_dataset


def split_dataset(
    dataset: LeRobotDataset,
    splits: dict[str, float | list[int]],
    output_dir: str | Path | None = None,
) -> dict[str, LeRobotDataset]:
    """将 LeRobotDataset 拆分为多个更小的数据集。

    需要重新编码的视频片段（因为源文件混合了落入不同划分的
    episode）会使用源数据集现有的编码器设置重新编码——从
    ``meta/info.json`` 读回——从而使每个输出划分与其自身的
    元数据保持一致。

    Args:
        dataset: 要拆分的源 LeRobotDataset。
        splits: 将划分名称映射到 episode 索引的字典，或者将划分名称
                映射到比例（总和必须 <= 1.0）的字典。
        output_dir: 拆分后数据集的存储根目录。如果未指定，默认为 $HF_LEROBOT_HOME/repo_id。

    Examples:
      按特定 episode 拆分
        splits = {"train": [0, 1, 2], "val": [3, 4]}
        datasets = split_dataset(dataset, splits)

      按比例拆分
        splits = {"train": 0.8, "val": 0.2}
        datasets = split_dataset(dataset, splits)
    """
    if not splits:
        raise ValueError("No splits provided")

    if all(isinstance(v, float) for v in splits.values()):
        splits = _fractions_to_episode_indices(dataset.meta.total_episodes, splits)

    all_episodes = set()
    for split_name, episodes in splits.items():
        if not episodes:
            raise ValueError(f"Split '{split_name}' has no episodes")
        episode_set = set(episodes)
        if episode_set & all_episodes:
            raise ValueError("Episodes cannot appear in multiple splits")
        all_episodes.update(episode_set)

    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = all_episodes - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    if output_dir is not None:
        output_dir = Path(output_dir)

    result_datasets = {}

    for split_name, episodes in splits.items():
        logging.info(f"Creating split '{split_name}' with {len(episodes)} episodes")

        split_repo_id = f"{dataset.repo_id}_{split_name}"

        split_output_dir = (
            output_dir / split_name if output_dir is not None else HF_LEROBOT_HOME / split_repo_id
        )

        episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(episodes))}

        new_meta = LeRobotDatasetMetadata.create(
            repo_id=split_repo_id,
            fps=dataset.meta.fps,
            features=dataset.meta.features,
            robot_type=dataset.meta.robot_type,
            root=split_output_dir,
            use_videos=len(dataset.meta.video_keys) > 0,
            chunks_size=dataset.meta.chunks_size,
            data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
            video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
        )

        video_metadata = None
        if dataset.meta.video_keys:
            video_metadata = _copy_and_reindex_videos(dataset, new_meta, episode_mapping)

        data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)

        _copy_and_reindex_episodes_metadata(dataset, new_meta, episode_mapping, data_metadata, video_metadata)

        new_dataset = LeRobotDataset(
            repo_id=split_repo_id,
            root=split_output_dir,
            image_transforms=dataset.image_transforms,
            delta_timestamps=dataset.delta_timestamps,
            tolerance_s=dataset.tolerance_s,
        )

        result_datasets[split_name] = new_dataset

    return result_datasets


def merge_datasets(
    datasets: list[LeRobotDataset],
    output_repo_id: str,
    output_dir: str | Path | None = None,
    concatenate_videos: bool = True,
    concatenate_data: bool = True,
) -> LeRobotDataset:
    """将多个 LeRobotDataset 合并为一个数据集。

    这是对 aggregate_datasets 功能的封装，提供更简洁的 API。

    Args:
        datasets: 要合并的 LeRobotDataset 列表。
        output_repo_id: 合并后的数据集标识符。
        output_dir: 合并后数据集的存储根目录。如果未指定，默认为 $HF_LEROBOT_HOME/output_repo_id。
        concatenate_videos: 为 False 时，每个源文件保留一个 mp4，而不是打包成分片。
        concatenate_data: 为 False 时，每个源文件保留一个 parquet，而不是打包成分片。
    """
    if not datasets:
        raise ValueError("No datasets to merge")

    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / output_repo_id

    repo_ids = [ds.repo_id for ds in datasets]
    roots = [ds.root for ds in datasets]

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_repo_id,
        roots=roots,
        aggr_root=output_dir,
        concatenate_videos=concatenate_videos,
        concatenate_data=concatenate_data,
    )

    merged_dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_dir,
        image_transforms=datasets[0].image_transforms,
        delta_timestamps=datasets[0].delta_timestamps,
        tolerance_s=datasets[0].tolerance_s,
    )

    return merged_dataset


def modify_features(
    dataset: LeRobotDataset,
    add_features: dict[str, tuple[np.ndarray | torch.Tensor | Callable, dict]] | None = None,
    remove_features: str | list[str] | None = None,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """通过一次遍历添加和/或移除特征来修改 LeRobotDataset。

    这是修改特征最高效的方式，因为无论添加或移除多少个特征，
    它都只复制数据集一次。

    Args:
        dataset: 源 LeRobotDataset。
        add_features: 可选字典，将特征名映射到 (feature_values, feature_info) 元组。
        remove_features: 可选的要移除的特征名。可以是单个字符串或列表。
        output_dir: 编辑后数据集的存储根目录。如果未指定，默认为 $HF_LEROBOT_HOME/repo_id。等同于 EditDatasetConfig 中的 new_root。
        repo_id: 编辑后的数据集标识符。等同于 EditDatasetConfig 中的 new_repo_id。

    Returns:
        特征已修改的新数据集。

    Example:
        new_dataset = modify_features(
            dataset,
            add_features={
                "reward": (reward_array, {"dtype": "float32", "shape": [1], "names": None}),
            },
            remove_features=["old_feature"],
            output_dir="./output",
        )
    """
    if add_features is None and remove_features is None:
        raise ValueError("Must specify at least one of add_features or remove_features")

    remove_features_list: list[str] = []
    if remove_features is not None:
        remove_features_list = [remove_features] if isinstance(remove_features, str) else remove_features

    if add_features:
        required_keys = {"dtype", "shape"}
        for feature_name, (_, feature_info) in add_features.items():
            if feature_name in dataset.meta.features:
                raise ValueError(f"Feature '{feature_name}' already exists in dataset")

            if not required_keys.issubset(feature_info.keys()):
                raise ValueError(f"feature_info for '{feature_name}' must contain keys: {required_keys}")

    if remove_features_list:
        for name in remove_features_list:
            if name not in dataset.meta.features:
                raise ValueError(f"Feature '{name}' not found in dataset")

        required_features = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        if any(name in required_features for name in remove_features_list):
            raise ValueError(f"Cannot remove required features: {required_features}")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    new_features = dataset.meta.features.copy()

    if remove_features_list:
        for name in remove_features_list:
            new_features.pop(name, None)

    if add_features:
        for feature_name, (_, feature_info) in add_features.items():
            new_features[feature_name] = feature_info

    video_keys_to_remove = [name for name in remove_features_list if name in dataset.meta.video_keys]
    remaining_video_keys = [k for k in dataset.meta.video_keys if k not in video_keys_to_remove]

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(remaining_video_keys) > 0,
    )

    _copy_data_with_feature_changes(
        dataset=dataset,
        new_meta=new_meta,
        add_features=add_features,
        remove_features=remove_features_list if remove_features_list else None,
    )

    if new_meta.video_keys:
        _copy_videos(dataset, new_meta, exclude_keys=video_keys_to_remove if video_keys_to_remove else None)

    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    return new_dataset


def add_features(
    dataset: LeRobotDataset,
    features: dict[str, tuple[np.ndarray | torch.Tensor | Callable, dict]],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """通过一次遍历向 LeRobotDataset 添加多个特征。

    这比多次调用 add_feature() 更高效，因为无论添加多少个特征，
    它都只复制数据集一次。

    Args:
        dataset: 源 LeRobotDataset。
        features: 将特征名映射到 (feature_values, feature_info) 元组的字典。
        output_dir: 编辑后数据集的存储根目录。如果未指定，默认为 $HF_LEROBOT_HOME/repo_id。等同于 EditDatasetConfig 中的 new_root。
        repo_id: 编辑后的数据集标识符。等同于 EditDatasetConfig 中的 new_repo_id。

    Returns:
        已添加所有特征的新数据集。

    Example:
        features = {
            "task_embedding": (task_emb_array, {"dtype": "float32", "shape": [384], "names": None}),
            "cam1_embedding": (cam1_emb_array, {"dtype": "float32", "shape": [768], "names": None}),
            "cam2_embedding": (cam2_emb_array, {"dtype": "float32", "shape": [768], "names": None}),
        }
        new_dataset = add_features(dataset, features, output_dir="./output", repo_id="my_dataset")
    """
    if not features:
        raise ValueError("No features provided")

    return modify_features(
        dataset=dataset,
        add_features=features,
        remove_features=None,
        output_dir=output_dir,
        repo_id=repo_id,
    )


def remove_feature(
    dataset: LeRobotDataset,
    feature_names: str | list[str],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """从 LeRobotDataset 中移除特征。

    Args:
        dataset: 源 LeRobotDataset。
        feature_names: 要移除的特征名。可以是单个字符串或列表。
        output_dir: 编辑后数据集的存储根目录。如果未指定，默认为 $HF_LEROBOT_HOME/repo_id。等同于 EditDatasetConfig 中的 new_root。
        repo_id: 编辑后的数据集标识符。等同于 EditDatasetConfig 中的 new_repo_id。

    Returns:
        已移除特征的新数据集。
    """
    return modify_features(
        dataset=dataset,
        add_features=None,
        remove_features=feature_names,
        output_dir=output_dir,
        repo_id=repo_id,
    )


def _fractions_to_episode_indices(
    total_episodes: int,
    splits: dict[str, float],
) -> dict[str, list[int]]:
    """将划分比例转换为 episode 索引。

    每个 episode 恰好被分配到一个划分，并且每个比例为正的划分
    至少获得一个 episode，因此较小的比例不会再向下取整为零
    而同时丢失该划分及其 episode。
    """
    for name, fraction in splits.items():
        if fraction < 0:
            raise ValueError(f"Split fraction for '{name}' must be non-negative, got {fraction}.")
    if sum(splits.values()) > 1.0:
        raise ValueError("Split fractions must sum to <= 1.0")

    for name, fraction in splits.items():
        if fraction == 0:
            logging.warning(f"Split '{name}' has a fraction of 0 and will be skipped.")

    positive_splits = [name for name, fraction in splits.items() if fraction > 0]
    if not positive_splits:
        raise ValueError("At least one split must have a positive fraction.")
    if total_episodes < len(positive_splits):
        raise ValueError(
            f"Cannot split {total_episodes} episodes into {len(positive_splits)} non-empty splits: "
            "there are fewer episodes than requested splits."
        )

    counts = {name: int(total_episodes * fraction) for name, fraction in splits.items()}
    counts[positive_splits[-1]] += total_episodes - sum(counts.values())

    for name in positive_splits:
        if counts[name] == 0:
            donor = max(positive_splits, key=lambda n: counts[n])
            counts[donor] -= 1
            counts[name] = 1

    indices = list(range(total_episodes))
    result = {}
    start_idx = 0
    for name in positive_splits:
        end_idx = start_idx + counts[name]
        result[name] = indices[start_idx:end_idx]
        start_idx = end_idx

    return result


def _copy_and_reindex_data(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> dict[int, dict]:
    """复制并过滤数据文件，仅修改包含被删除 episode 的文件。

    Args:
        src_dataset: 要复制的源数据集
        dst_meta: 目标元数据对象
        episode_mapping: 从旧 episode 索引到新索引的映射

    Returns:
        将 episode 索引映射到其数据文件元数据（chunk_index、file_index 等）的字典
    """
    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    file_to_episodes: dict[Path, set[int]] = {}
    for old_idx in episode_mapping:
        file_path = src_dataset.meta.get_data_file_path(old_idx)
        if file_path not in file_to_episodes:
            file_to_episodes[file_path] = set()
        file_to_episodes[file_path].add(old_idx)

    global_index = 0
    episode_data_metadata: dict[int, dict] = {}

    if dst_meta.tasks is None:
        all_task_indices = set()
        for src_path in file_to_episodes:
            df = pd.read_parquet(src_dataset.root / src_path)
            mask = df["episode_index"].isin(list(episode_mapping.keys()))
            task_series: pd.Series = df[mask]["task_index"]
            all_task_indices.update(task_series.unique().tolist())
        tasks = [src_dataset.meta.tasks.iloc[idx].name for idx in all_task_indices]
        dst_meta.save_episode_tasks(list(set(tasks)))

    task_mapping = {}
    for old_task_idx in range(len(src_dataset.meta.tasks)):
        task_name = src_dataset.meta.tasks.iloc[old_task_idx].name
        new_task_idx = dst_meta.get_task_index(task_name)
        if new_task_idx is not None:
            task_mapping[old_task_idx] = new_task_idx

    for src_path in tqdm(sorted(file_to_episodes.keys()), desc="Processing data files"):
        df = pd.read_parquet(src_dataset.root / src_path)

        all_episodes_in_file = set(df["episode_index"].unique())
        episodes_to_keep = file_to_episodes[src_path]

        if all_episodes_in_file == episodes_to_keep:
            df["episode_index"] = df["episode_index"].replace(episode_mapping)
            df["index"] = range(global_index, global_index + len(df))
            df["task_index"] = df["task_index"].replace(task_mapping)

            first_ep_old_idx = min(episodes_to_keep)
            src_ep = src_dataset.meta.episodes[first_ep_old_idx]
            chunk_idx = src_ep["data/chunk_index"]
            file_idx = src_ep["data/file_index"]
        else:
            mask = df["episode_index"].isin(list(episode_mapping.keys()))
            df = df[mask].copy().reset_index(drop=True)

            if len(df) == 0:
                continue

            df["episode_index"] = df["episode_index"].replace(episode_mapping)
            df["index"] = range(global_index, global_index + len(df))
            df["task_index"] = df["task_index"].replace(task_mapping)

            first_ep_old_idx = min(episodes_to_keep)
            src_ep = src_dataset.meta.episodes[first_ep_old_idx]
            chunk_idx = src_ep["data/chunk_index"]
            file_idx = src_ep["data/file_index"]

        dst_path = dst_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, dst_path, dst_meta)

        for ep_old_idx in episodes_to_keep:
            ep_new_idx = episode_mapping[ep_old_idx]
            ep_df = df[df["episode_index"] == ep_new_idx]
            episode_data_metadata[ep_new_idx] = {
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": int(ep_df["index"].min()),
                "dataset_to_index": int(ep_df["index"].max() + 1),
            }

        global_index += len(df)

    return episode_data_metadata


def _keep_episodes_from_video_with_av(
    input_path: Path,
    output_path: Path,
    episodes_to_keep: list[tuple[int, int]],
    fps: float,
    video_encoder: VideoEncoderConfig,
) -> None:
    """使用 PyAV 仅从视频文件中保留指定的 episode。

    此函数从指定的帧范围解码帧，并使用正确重置的时间戳重新编码它们，
    以确保单调递增。

    Args:
        input_path: 源视频文件路径。
        output_path: 目标视频文件路径。
        episodes_to_keep: 要保留的 episode 的 (start_frame, end_frame) 元组列表。
            范围是半开区间：[start_frame, end_frame)，其中 start_frame
            包含，end_frame 不包含。
        fps: 视频的帧率。
        video_encoder: 用于重新编码保留帧的视频编码器设置。
    """
    from fractions import Fraction

    import av

    if not episodes_to_keep:
        raise ValueError("No episodes to keep")

    in_container = av.open(str(input_path))

    # 检查视频流是否存在。
    if not in_container.streams.video:
        raise ValueError(
            f"No video streams found in {input_path}. "
            "The video file may be corrupted or empty. "
            "Try re-downloading the dataset or checking the video file."
        )

    v_in = in_container.streams.video[0]

    out = av.open(str(output_path), mode="w")

    # 将 fps 转换为 Fraction 以兼容 PyAV。
    fps_fraction = Fraction(fps).limit_denominator(1000)
    codec_options = video_encoder.get_codec_options(as_strings=True)
    v_out = out.add_stream(video_encoder.vcodec, rate=fps_fraction, options=codec_options)

    # PyAV 类型存根不区分视频流与音频/字幕流。
    v_out.width = v_in.codec_context.width
    v_out.height = v_in.codec_context.height
    v_out.pix_fmt = video_encoder.pix_fmt

    # 设置 time_base 以匹配帧率，从而正确处理时间戳。
    v_out.time_base = Fraction(1, int(fps))

    out.start_encoding()

    # 创建 (start, end) 范围集合以便快速查找。
    # 转换为排序列表以便高效检查。
    frame_ranges = sorted(episodes_to_keep)

    # 跟踪用于设置 PTS 的帧索引以及当前正在处理的范围。
    src_frame_count = 0
    frame_count = 0
    range_idx = 0

    # 一次性读取整个视频并过滤帧。
    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue

            # 检查帧是否位于我们期望的帧范围内。
            # 跳过已经经过的范围。
            while range_idx < len(frame_ranges) and src_frame_count >= frame_ranges[range_idx][1]:
                range_idx += 1

            # 如果已经经过所有范围，停止处理。
            if range_idx >= len(frame_ranges):
                break

            # 检查帧是否位于当前范围内。
            start_frame = frame_ranges[range_idx][0]

            if src_frame_count < start_frame:
                src_frame_count += 1
                continue

            # 帧位于范围内——创建一个重置了时间戳的新帧。
            # 我们需要创建一个副本以避免修改原始帧。
            new_frame = frame.reformat(width=v_out.width, height=v_out.height, format=v_out.pix_fmt)
            new_frame.pts = frame_count
            new_frame.time_base = Fraction(1, int(fps))

            # 编码并复用（mux）该帧。
            for pkt in v_out.encode(new_frame):
                out.mux(pkt)

            src_frame_count += 1
            frame_count += 1

    # 刷新编码器。
    for pkt in v_out.encode():
        out.mux(pkt)

    out.close()
    in_container.close()


def _copy_and_reindex_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> dict[int, dict]:
    """复制并过滤视频文件，仅重新编码包含被删除 episode 的文件。

    对于仅包含保留 episode 的视频文件，我们直接复制。
    对于混合了保留/删除 episode 的文件，我们使用 PyAV 滤镜高效地
    仅重新编码所需的片段。用于重新编码的编码器按视频键从源数据集的
    ``meta/info.json`` 推导而来，从而使目标元数据继续准确地描述视频。

    Args:
        src_dataset: 要复制的源数据集
        dst_meta: 目标元数据对象
        episode_mapping: 从旧 episode 索引到新索引的映射

    Returns:
        将 episode 索引映射到其视频元数据（chunk_index、file_index、时间戳）的字典
    """
    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    episodes_video_metadata: dict[int, dict] = {new_idx: {} for new_idx in episode_mapping.values()}

    for video_key in src_dataset.meta.video_keys:
        logging.info(f"Processing videos for {video_key}")
        video_encoder = encoder_config_from_video_info(
            src_dataset.meta.info.features.get(video_key, {}).get("info")
        )

        if dst_meta.video_path is None:
            raise ValueError("Destination metadata has no video_path defined")

        file_to_episodes: dict[tuple[int, int], list[int]] = {}
        for old_idx in episode_mapping:
            src_ep = src_dataset.meta.episodes[old_idx]
            chunk_idx = src_ep[f"videos/{video_key}/chunk_index"]
            file_idx = src_ep[f"videos/{video_key}/file_index"]
            file_key = (chunk_idx, file_idx)
            if file_key not in file_to_episodes:
                file_to_episodes[file_key] = []
            file_to_episodes[file_key].append(old_idx)

        for (src_chunk_idx, src_file_idx), episodes_in_file in tqdm(
            sorted(file_to_episodes.items()), desc=f"Processing {video_key} video files"
        ):
            all_episodes_in_file = [
                ep_idx
                for ep_idx in range(src_dataset.meta.total_episodes)
                if src_dataset.meta.episodes[ep_idx].get(f"videos/{video_key}/chunk_index") == src_chunk_idx
                and src_dataset.meta.episodes[ep_idx].get(f"videos/{video_key}/file_index") == src_file_idx
            ]

            episodes_to_keep_set = set(episodes_in_file)
            all_in_file_set = set(all_episodes_in_file)

            if all_in_file_set == episodes_to_keep_set:
                assert src_dataset.meta.video_path is not None
                src_video_path = src_dataset.root / src_dataset.meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path = dst_meta.root / dst_meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_video_path, dst_video_path)

                for old_idx in episodes_in_file:
                    new_idx = episode_mapping[old_idx]
                    src_ep = src_dataset.meta.episodes[old_idx]
                    episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = src_chunk_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = src_file_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = src_ep[
                        f"videos/{video_key}/from_timestamp"
                    ]
                    episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = src_ep[
                        f"videos/{video_key}/to_timestamp"
                    ]
            else:
                # 按排序顺序构建要保留的帧范围列表。
                sorted_keep_episodes = sorted(episodes_in_file, key=lambda x: episode_mapping[x])
                episodes_to_keep_ranges: list[tuple[int, int]] = []
                for old_idx in sorted_keep_episodes:
                    src_ep = src_dataset.meta.episodes[old_idx]
                    from_frame = round(src_ep[f"videos/{video_key}/from_timestamp"] * src_dataset.meta.fps)
                    to_frame = round(src_ep[f"videos/{video_key}/to_timestamp"] * src_dataset.meta.fps)
                    assert src_ep["length"] == to_frame - from_frame, (
                        f"Episode length mismatch: {src_ep['length']} vs {to_frame - from_frame}"
                    )
                    episodes_to_keep_ranges.append((from_frame, to_frame))

                # 使用 PyAV 滤镜高效地仅重新编码所需的片段。
                assert src_dataset.meta.video_path is not None
                src_video_path = src_dataset.root / src_dataset.meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path = dst_meta.root / dst_meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path.parent.mkdir(parents=True, exist_ok=True)

                logging.info(
                    f"Re-encoding {video_key} (chunk {src_chunk_idx}, file {src_file_idx}) "
                    f"with {len(episodes_to_keep_ranges)} episodes"
                )
                _keep_episodes_from_video_with_av(
                    src_video_path,
                    dst_video_path,
                    episodes_to_keep_ranges,
                    src_dataset.meta.fps,
                    video_encoder,
                )

                cumulative_ts = 0.0
                for old_idx in sorted_keep_episodes:
                    new_idx = episode_mapping[old_idx]
                    src_ep = src_dataset.meta.episodes[old_idx]
                    ep_length = src_ep["length"]
                    ep_duration = ep_length / src_dataset.meta.fps

                    episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = src_chunk_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = src_file_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = cumulative_ts
                    episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = (
                        cumulative_ts + ep_duration
                    )

                    cumulative_ts += ep_duration

    return episodes_video_metadata


def _copy_and_reindex_episodes_metadata(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
    data_metadata: dict[int, dict],
    video_metadata: dict[int, dict] | None = None,
) -> None:
    """使用提供的数据和视频元数据复制并重新索引 episode 元数据。

    Args:
        src_dataset: 要复制的源数据集
        dst_meta: 目标元数据对象
        episode_mapping: 从旧 episode 索引到新索引的映射
        data_metadata: 将新 episode 索引映射到其数据文件元数据的字典
        video_metadata: 可选字典，将新 episode 索引映射到其视频元数据
    """
    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    all_stats = []
    total_frames = 0

    for old_idx, new_idx in tqdm(
        sorted(episode_mapping.items(), key=lambda x: x[1]), desc="Processing episodes metadata"
    ):
        src_episode_full = _load_episode_with_stats(src_dataset, old_idx)

        src_episode = src_dataset.meta.episodes[old_idx]

        episode_meta = data_metadata[new_idx].copy()

        if video_metadata and new_idx in video_metadata:
            episode_meta.update(video_metadata[new_idx])

        # 从 parquet 元数据中提取 episode 统计信息。
        # 当 pandas/pyarrow 将形状为 (C, 1, 1) 的 numpy 数组序列化到 parquet 时，
        # 它们会被反序列化为嵌套的对象数组，例如：
        #   array([array([array([0.])]), array([array([0.])]), array([array([0.])])])
        # 这种情况尤其出现在图像/视频统计信息中。我们需要检测这些嵌套结构
        # 并将其展平回正确的 (C, 1, 1) 数组，以便 aggregate_stats 能处理它们。
        episode_stats = {}
        for key in src_episode_full:
            if key.startswith("stats/"):
                stat_key = key.replace("stats/", "")
                parts = stat_key.split("/")
                if len(parts) == 2:
                    feature_name, stat_name = parts
                    if feature_name not in episode_stats:
                        episode_stats[feature_name] = {}

                    value = src_episode_full[key]

                    if feature_name in src_dataset.meta.features:
                        feature_dtype = src_dataset.meta.features[feature_name]["dtype"]
                        if feature_dtype in ["image", "video"] and stat_name != "count":
                            # 统计信息是通道优先的 (C, 1, 1)
                            if isinstance(value, np.ndarray) and value.dtype == object:
                                flat_values = []
                                for item in value:
                                    while isinstance(item, np.ndarray):
                                        item = item.flatten()[0]
                                    flat_values.append(item)
                                value = np.array(flat_values, dtype=np.float64).reshape(-1, 1, 1)
                            elif isinstance(value, np.ndarray) and value.ndim == 1:
                                value = value.reshape(-1, 1, 1)

                    episode_stats[feature_name][stat_name] = value

        all_stats.append(episode_stats)

        episode_dict = {
            "episode_index": new_idx,
            "tasks": src_episode["tasks"],
            "length": src_episode["length"],
        }
        episode_dict.update(episode_meta)
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        dst_meta._save_episode_metadata(episode_dict)

        total_frames += src_episode["length"]

    dst_meta.finalize()

    dst_meta.info.total_episodes = len(episode_mapping)
    dst_meta.info.total_frames = total_frames
    dst_meta.info.total_tasks = len(dst_meta.tasks) if dst_meta.tasks is not None else 0
    dst_meta.info.splits = {"train": f"0:{len(episode_mapping)}"}
    write_info(dst_meta.info, dst_meta.root)

    if not all_stats:
        logging.warning("No statistics found to aggregate")
        return

    logging.info(f"Aggregating statistics for {len(all_stats)} episodes")
    aggregated_stats = aggregate_stats(all_stats)
    filtered_stats = {k: v for k, v in aggregated_stats.items() if k in dst_meta.features}
    write_stats(filtered_stats, dst_meta.root)


def _write_parquet(df: pd.DataFrame, path: Path, meta: LeRobotDatasetMetadata) -> None:
    """将 DataFrame 写入 parquet

    这可确保图像被正确嵌入，并且文件能被 HF datasets 正确加载。
    """
    from .feature_utils import get_hf_features_from_features
    from .io_utils import embed_images

    hf_features = get_hf_features_from_features(meta.features)
    ep_dataset = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features, split="train")

    if len(meta.image_keys) > 0:
        ep_dataset = embed_images(ep_dataset)

    table = ep_dataset.with_format("arrow")[:]
    writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def _save_data_chunk(
    df: pd.DataFrame,
    meta: LeRobotDatasetMetadata,
    chunk_idx: int = 0,
    file_idx: int = 0,
) -> tuple[int, int, dict[int, dict]]:
    """保存数据块并返回更新后的索引和 episode 元数据。

    Returns:
        tuple: (next_chunk_idx, next_file_idx, episode_metadata_dict)
            其中 episode_metadata_dict 将 episode_index 映射到其数据文件元数据
    """
    path = meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)

    _write_parquet(df, path, meta)

    episode_metadata = {}
    for ep_idx in df["episode_index"].unique():
        ep_df = df[df["episode_index"] == ep_idx]
        episode_metadata[ep_idx] = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": int(ep_df["index"].min()),
            "dataset_to_index": int(ep_df["index"].max() + 1),
        }

    file_size = get_parquet_file_size_in_mb(path)
    if file_size >= DEFAULT_DATA_FILE_SIZE_IN_MB * 0.9:
        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

    return chunk_idx, file_idx, episode_metadata


def _copy_data_with_feature_changes(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    add_features: dict[str, tuple] | None = None,
    remove_features: list[str] | None = None,
) -> None:
    """在添加或移除特征的同时复制数据。"""
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    frame_idx = 0

    for src_path in tqdm(parquet_files, desc="Processing data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        relative_path = src_path.relative_to(dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]

        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])

        if remove_features:
            df = df.drop(columns=remove_features, errors="ignore")

        if add_features:
            end_idx = frame_idx + len(df)
            for feature_name, (values, _) in add_features.items():
                if callable(values):
                    feature_values = []
                    for _, row in df.iterrows():
                        ep_idx = row["episode_index"]
                        frame_in_ep = row["frame_index"]
                        value = values(row.to_dict(), ep_idx, frame_in_ep)
                        if isinstance(value, np.ndarray) and value.size == 1:
                            value = value.item()
                        feature_values.append(value)
                    df[feature_name] = feature_values
                else:
                    feature_slice = values[frame_idx:end_idx]
                    if feature_slice.ndim == 1:
                        df[feature_name] = feature_slice
                    elif feature_slice.ndim == 2 and feature_slice.shape[1] == 1:
                        df[feature_name] = feature_slice.flatten()
                    else:
                        df[feature_name] = list(feature_slice)
            frame_idx = end_idx

        # 使用与源相同的块/文件结构写入
        dst_path = new_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, dst_path, new_meta)

    _copy_episodes_metadata_and_stats(dataset, new_meta)


def _copy_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    exclude_keys: list[str] | None = None,
) -> None:
    """复制视频文件，可选择排除某些键。"""
    if exclude_keys is None:
        exclude_keys = []

    for video_key in src_dataset.meta.video_keys:
        if video_key in exclude_keys:
            continue

        video_files = set()
        for ep_idx in range(len(src_dataset.meta.episodes)):
            try:
                video_files.add(src_dataset.meta.get_video_file_path(ep_idx, video_key))
            except KeyError:
                continue

        for src_path in tqdm(sorted(video_files), desc=f"Copying {video_key} videos"):
            dst_path = dst_meta.root / src_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_dataset.root / src_path, dst_path)


def _copy_episodes_metadata_and_stats(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
) -> None:
    """复制 episode 元数据并重新计算统计信息。"""
    if src_dataset.meta.tasks is not None:
        write_tasks(src_dataset.meta.tasks, dst_meta.root)
        dst_meta.tasks = src_dataset.meta.tasks.copy()

    episodes_dir = src_dataset.root / "meta/episodes"
    dst_episodes_dir = dst_meta.root / "meta/episodes"
    if episodes_dir.exists():
        shutil.copytree(episodes_dir, dst_episodes_dir, dirs_exist_ok=True)

    dst_meta.info.total_episodes = src_dataset.meta.total_episodes
    dst_meta.info.total_frames = src_dataset.meta.total_frames
    dst_meta.info.total_tasks = src_dataset.meta.total_tasks
    # 如果可用则保留原始划分，否则创建默认划分
    dst_meta.info.splits = (
        src_dataset.meta.info.splits
        if src_dataset.meta.info.splits
        else {"train": f"0:{src_dataset.meta.total_episodes}"}
    )

    if dst_meta.video_keys and src_dataset.meta.video_keys:
        for key in dst_meta.video_keys:
            if key in src_dataset.meta.features:
                dst_meta.info.features[key]["info"] = deepcopy(
                    src_dataset.meta.info.features[key].get("info", {})
                )

    write_info(dst_meta.info, dst_meta.root)

    if set(dst_meta.features.keys()) != set(src_dataset.meta.features.keys()):
        logging.info("Recalculating dataset statistics...")
        if src_dataset.meta.stats:
            new_stats = {}
            for key in dst_meta.features:
                if key in src_dataset.meta.stats:
                    new_stats[key] = src_dataset.meta.stats[key]
            write_stats(new_stats, dst_meta.root)
    else:
        if src_dataset.meta.stats:
            write_stats(src_dataset.meta.stats, dst_meta.root)


def _save_episode_images_for_video(
    dataset: LeRobotDataset,
    imgs_dir: Path,
    img_key: str,
    episode_index: int,
    num_workers: int = 4,
) -> None:
    """将特定 episode 和相机的图像保存到磁盘，用于视频编码。

    Args:
        dataset: 要从中提取图像的 LeRobot 数据集
        imgs_dir: 保存图像的目录
        img_key: 要提取的图像键（相机）
        episode_index: 要保存的 episode 索引
        num_workers: 并行保存图像的线程数
    """
    # 创建目录
    imgs_dir.mkdir(parents=True, exist_ok=True)

    # 获取不带 torch 格式的数据集以便访问 PIL 图像
    hf_dataset = dataset.hf_dataset.with_format(None)

    # 仅选择此相机的图像
    imgs_dataset = hf_dataset.select_columns(img_key)

    # 获取 episode 的起始和结束索引
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]

    # 获取此 episode 的所有条目
    episode_dataset = imgs_dataset.select(range(from_idx, to_idx))

    is_depth = img_key in dataset.meta.depth_keys
    frame_pattern = DEPTH_FILE_PATTERN if is_depth else IMAGE_FILE_PATTERN

    # 定义保存单张图像的函数
    def save_single_image(i_item_tuple):
        i, item = i_item_tuple
        write_image(item[img_key], imgs_dir / frame_pattern.format(frame_index=i))
        return i

    items = list(enumerate(episode_dataset))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(save_single_image, item) for item in items]
        for future in as_completed(futures):
            future.result()  # 这将抛出已发生的任何异常


def _save_batch_episodes_images(
    dataset: LeRobotDataset,
    imgs_dir: Path,
    img_key: str,
    episode_indices: list[int],
    num_workers: int = 4,
) -> list[float]:
    """将多个 episode 的图像保存到磁盘，用于批量视频编码。

    Args:
        dataset: 要从中提取图像的 LeRobot 数据集
        imgs_dir: 保存图像的目录
        img_key: 要提取的图像键（相机）
        episode_indices: 要保存的 episode 索引列表
        num_workers: 并行保存图像的线程数

    Returns:
        以秒为单位的 episode 时长列表
    """
    imgs_dir.mkdir(parents=True, exist_ok=True)
    hf_dataset = dataset.hf_dataset.with_format(None)
    imgs_dataset = hf_dataset.select_columns(img_key)

    is_depth = img_key in dataset.meta.depth_keys
    frame_pattern = DEPTH_FILE_PATTERN if is_depth else IMAGE_FILE_PATTERN

    # 定义使用全局帧索引保存单张图像的函数
    # 在循环外只定义一次，避免重复创建闭包
    def save_single_image(i_item_tuple, base_frame_idx, img_key_param):
        i, item = i_item_tuple
        write_image(item[img_key_param], imgs_dir / frame_pattern.format(frame_index=base_frame_idx + i))
        return i

    episode_durations = []
    frame_idx = 0

    for ep_idx in episode_indices:
        # 获取 episode 范围
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
        episode_length = to_idx - from_idx
        episode_durations.append(episode_length / dataset.fps)

        # 获取 episode 图像
        episode_dataset = imgs_dataset.select(range(from_idx, to_idx))

        # 保存图像
        items = list(enumerate(episode_dataset))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(save_single_image, item, frame_idx, img_key) for item in items]
            for future in as_completed(futures):
                future.result()

        frame_idx += episode_length

    return episode_durations


def _iter_episode_batches(
    episode_indices: list[int],
    episode_lengths: dict[int, int],
    size_per_frame_mb: float,
    video_file_size_limit: float,
    max_episodes: int | None,
    max_frames: int | None,
):
    """生成用于视频编码的 episode 索引批次的生成器。

    将 episode 分组为满足大小和内存约束的批次：
    - 保持在视频文件大小限制之下
    - 遵守每批最大 episode 数（如果指定）
    - 遵守每批最大帧数（如果指定）

    Args:
        episode_indices: 要分批的 episode 索引列表
        episode_lengths: 将 episode 索引映射到 episode 长度的字典
        size_per_frame_mb: 每帧的估计大小（MB）
        video_file_size_limit: 最大视频文件大小（MB）
        max_episodes: 每批最大 episode 数（None = 无限制）
        max_frames: 每批最大帧数（None = 无限制）

    Yields:
        每批的 episode 索引列表
    """
    batch_episodes = []
    estimated_size = 0.0
    total_frames = 0

    for ep_idx in episode_indices:
        ep_length = episode_lengths[ep_idx]
        ep_estimated_size = ep_length * size_per_frame_mb

        # 检查添加此 episode 是否会超出任何约束
        would_exceed_size = estimated_size > 0 and estimated_size + ep_estimated_size >= video_file_size_limit
        would_exceed_episodes = max_episodes is not None and len(batch_episodes) >= max_episodes
        would_exceed_frames = max_frames is not None and total_frames + ep_length > max_frames

        if batch_episodes and (would_exceed_size or would_exceed_episodes or would_exceed_frames):
            # 在添加此 episode 之前先产出当前批次
            yield batch_episodes
            # 以当前 episode 开始新批次
            batch_episodes = [ep_idx]
            estimated_size = ep_estimated_size
            total_frames = ep_length
        else:
            # 添加到当前批次
            batch_episodes.append(ep_idx)
            estimated_size += ep_estimated_size
            total_frames += ep_length

    # 如果最后一个批次非空则产出
    if batch_episodes:
        yield batch_episodes


def _estimate_frame_size_via_calibration(
    dataset: LeRobotDataset,
    img_key: str,
    episode_indices: list[int],
    temp_dir: Path,
    fps: int,
    video_encoder: VideoEncoderConfig,
    num_calibration_frames: int = 30,
) -> float:
    """通过编码一个小的校准样本来估计每帧的 MB 数。

    使用精确的编解码器参数编码具有代表性的帧样本，以测量实际压缩率，
    这比启发式方法更准确。

    Args:
        dataset: 包含图像的源数据集。
        img_key: 要校准的图像键（例如 "observation.images.top"）。
        episode_indices: 正在处理的 episode 索引列表。
        temp_dir: 校准文件的临时目录。
        fps: 视频编码的每秒帧数。
        video_encoder: 用于校准编码的视频编码器设置。
        num_calibration_frames: 用于校准的帧数（默认：30）。

    Returns:
        基于实际编码的每帧估计大小（MB）。
    """
    calibration_dir = temp_dir / "calibration" / img_key
    calibration_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 选择一个有代表性的 episode（如果可能，优先选择中间的 episode）
        calibration_ep_idx = episode_indices[len(episode_indices) // 2]

        # 获取 episode 范围
        from_idx = dataset.meta.episodes["dataset_from_index"][calibration_ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][calibration_ep_idx]
        episode_length = to_idx - from_idx

        # 从此 episode 中最多使用 num_calibration_frames 帧
        num_frames = min(num_calibration_frames, episode_length)

        # 从数据集获取帧
        hf_dataset = dataset.hf_dataset.with_format(None)
        sample_indices = range(from_idx, from_idx + num_frames)

        # 使用编码器期望的后缀/格式保存校准帧。
        is_depth = img_key in dataset.meta.depth_keys
        frame_pattern = DEPTH_FILE_PATTERN if is_depth else IMAGE_FILE_PATTERN
        for i, idx in enumerate(sample_indices):
            write_image(hf_dataset[idx][img_key], calibration_dir / frame_pattern.format(frame_index=i))

        # 编码校准视频
        calibration_video_path = calibration_dir / "calibration.mp4"
        encode_video_frames(
            imgs_dir=calibration_dir,
            video_path=calibration_video_path,
            fps=fps,
            video_encoder=video_encoder,
            overwrite=True,
        )

        # 测量实际压缩后的大小
        video_size_bytes = calibration_video_path.stat().st_size
        video_size_mb = video_size_bytes / BYTES_PER_MIB
        size_per_frame_mb = video_size_mb / num_frames

        logging.info(
            f"  Calibration: {num_frames} frames -> {video_size_mb:.2f} MB "
            f"= {size_per_frame_mb:.4f} MB/frame for {img_key}"
        )

        return size_per_frame_mb

    finally:
        # 清理校准文件
        if calibration_dir.exists():
            shutil.rmtree(calibration_dir)


def _copy_data_without_images(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_indices: list[int],
    img_keys: list[str],
) -> None:
    """复制不含图像列的数据文件。

    Args:
        src_dataset: 源数据集
        dst_meta: 目标元数据
        episode_indices: 要包含的 episode
        img_keys: 要移除的图像键
    """
    from .utils import DATA_DIR

    data_dir = src_dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    episode_set = set(episode_indices)

    for src_path in tqdm(parquet_files, desc="Processing data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        # 过滤以仅包含所选的 episode
        df = df[df["episode_index"].isin(episode_set)].copy()

        if len(df) == 0:
            continue

        # 移除图像列
        columns_to_drop = [col for col in img_keys if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        # 从路径获取块和文件索引
        relative_path = src_path.relative_to(src_dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]
        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])

        # 写入目标，不带 pandas 索引
        dst_path = dst_meta.root / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_path, index=False)


# 视频转换常量
BYTES_PER_KIB = 1024
BYTES_PER_MIB = BYTES_PER_KIB * BYTES_PER_KIB


def modify_tasks(
    dataset: LeRobotDataset,
    new_task: str | None = None,
    episode_tasks: dict[int, str] | None = None,
    task_replacements: dict[str, str] | None = None,
) -> LeRobotDataset:
    """修改 LeRobotDataset 中的任务。

    此函数允许你：
    1. 为整个数据集设置单个任务（使用 `new_task`）
    2. 为特定 episode 设置特定任务（使用 `episode_tasks`）
    3. 替换现有任务字符串（无论它们出现在哪里）（使用 `task_replacements`）

    对每个 episode，任务按以下优先级解析：
    `episode_tasks` > `task_replacements` > `new_task` > 原始任务。如果某个
    episode 最终没有任务（以上均不适用且它没有原始任务），则抛出错误。

    数据集会被原地修改，仅更新与任务相关的文件：
    - meta/tasks.parquet
    - data/**/*.parquet（task_index 列）
    - meta/episodes/**/*.parquet（tasks 列）
    - meta/info.json（total_tasks）

    Args:
        dataset: 要修改的源 LeRobotDataset。
        new_task: 应用于任何未被 `episode_tasks` 或匹配的 `task_replacements`
            条目覆盖的 episode 的默认任务。
        episode_tasks: 可选字典，将 episode 索引映射到任务字符串。优先级高于
            `task_replacements` 和 `new_task`。
        task_replacements: 可选字典，将现有任务字符串映射到新任务。应用于当前
            任务与键匹配的 episode。每个键都必须是已存在的任务。

    必须至少提供 `new_task`、`episode_tasks` 或 `task_replacements` 之一。

    Examples:
        为所有 episode 设置单个任务：
            dataset = modify_tasks(dataset, new_task="Pick up the cube")

        为特定 episode 设置不同任务：
            dataset = modify_tasks(
                dataset,
                episode_tasks={0: "Task A", 1: "Task B", 2: "Task A"}
            )

        设置带覆盖的默认任务：
            dataset = modify_tasks(
                dataset,
                new_task="Default task",
                episode_tasks={5: "Special task for episode 5"}
            )

        原地替换现有任务字符串：
            dataset = modify_tasks(
                dataset,
                task_replacements={"Pick up the cube": "Lift the cube"}
            )
    """
    if not new_task and not episode_tasks and not task_replacements:
        raise ValueError("Must specify at least one of new_task, episode_tasks, or task_replacements")

    if episode_tasks:
        valid_indices = set(range(dataset.meta.total_episodes))
        invalid = set(episode_tasks.keys()) - valid_indices
        if invalid:
            raise ValueError(f"Invalid episode indices: {invalid}")

    # 确保已加载 episode 元数据
    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    if task_replacements:
        current_tasks = set(dataset.meta.tasks.index)
        invalid_tasks = set(task_replacements) - current_tasks
        if invalid_tasks:
            raise ValueError(f"Task replacements reference unknown tasks: {sorted(invalid_tasks)}")

    # 构建从 episode 索引到任务字符串的映射
    episode_to_task: dict[int, str] = {}
    for ep_idx in range(dataset.meta.total_episodes):
        original_tasks = dataset.meta.episodes[ep_idx]["tasks"]
        original_task = original_tasks[0] if original_tasks else None

        if episode_tasks and ep_idx in episode_tasks:
            episode_to_task[ep_idx] = episode_tasks[ep_idx]
        elif task_replacements and original_task in task_replacements:
            episode_to_task[ep_idx] = task_replacements[original_task]
        elif new_task:
            episode_to_task[ep_idx] = new_task
        elif original_task:
            # 如果未被覆盖且未提供默认任务，则保留原始任务
            episode_to_task[ep_idx] = original_task
        else:
            raise ValueError(f"Episode {ep_idx} has no task; provide new_task or episode_tasks")

    # 收集所有唯一任务并创建新的任务映射
    unique_tasks = sorted(set(episode_to_task.values()))
    new_task_df = pd.DataFrame(
        {"task_index": list(range(len(unique_tasks)))}, index=pd.Index(unique_tasks, name="task")
    )
    task_to_index = {task: idx for idx, task in enumerate(unique_tasks)}

    logging.info(f"Modifying tasks in {dataset.repo_id}")
    logging.info(f"New tasks: {unique_tasks}")

    root = dataset.root

    # 更新数据文件——修改 task_index 列
    logging.info("Updating data files...")
    data_dir = root / DATA_DIR

    for parquet_path in tqdm(sorted(data_dir.rglob("*.parquet")), desc="Updating data"):
        df = pd.read_parquet(parquet_path)

        # 为此文件中的行构建从 episode_index 到新 task_index 的映射
        episode_indices_in_file = df["episode_index"].unique()
        ep_to_new_task_idx = {
            ep_idx: task_to_index[episode_to_task[ep_idx]] for ep_idx in episode_indices_in_file
        }

        # 更新 task_index 列
        df["task_index"] = df["episode_index"].map(ep_to_new_task_idx)
        df.to_parquet(parquet_path, index=False)

    # 更新 episode 元数据——修改 tasks 列
    logging.info("Updating episodes metadata...")
    episodes_dir = root / "meta" / "episodes"

    for parquet_path in tqdm(sorted(episodes_dir.rglob("*.parquet")), desc="Updating episodes"):
        df = pd.read_parquet(parquet_path)

        # 更新 tasks 列
        df["tasks"] = df["episode_index"].apply(lambda ep_idx: [episode_to_task[ep_idx]])
        df.to_parquet(parquet_path, index=False)

    # 写入新的 tasks.parquet
    write_tasks(new_task_df, root)

    # 更新 info.json
    dataset.meta.info.total_tasks = len(unique_tasks)
    write_info(dataset.meta.info, root)

    # 重新加载元数据以反映更改
    dataset.meta.tasks = new_task_df
    dataset.meta.episodes = load_episodes(root)

    logging.info(f"Tasks: {unique_tasks}")

    return dataset


def recompute_stats(
    dataset: LeRobotDataset,
    skip_image_video: bool = True,
    relative_action: bool = False,
    relative_exclude_joints: list[str] | None = None,
    chunk_size: int = 50,
    num_workers: int = 0,
) -> LeRobotDataset:
    """通过遍历所有 episode 从头重新计算 stats.json。

    Args:
        dataset: 要重新计算统计信息的 LeRobotDataset。
        skip_image_video: 如果为 True（默认），仅重新计算数值特征
            （action、state 等）的统计信息，并保持现有的图像/视频统计信息不变。
        relative_action: 如果为 True，通过遍历所有有效的动作块并减去
            当前状态，在相对空间中计算动作统计信息。这与模型在
            ``use_relative_actions=True`` 训练期间看到的归一化分布一致。
        relative_exclude_joints: 当 relative_action=True 时，要从相对转换中
            排除的关节名称。这些维度保留绝对统计信息。
        chunk_size: 用于相对统计信息计算的动作块大小。应与
            ``policy.chunk_size`` 匹配。仅在 ``relative_action=True`` 时使用。
        num_workers: 用于相对动作统计信息计算的并行线程数。
            值 ≤1 表示单线程。仅在 ``relative_action=True`` 时使用。

    Returns:
        统计信息已更新的同一数据集。
    """
    features = dataset.meta.features
    meta_keys = {"index", "episode_index", "task_index", "frame_index", "timestamp"}
    numeric_features = {
        k: v
        for k, v in features.items()
        if v["dtype"] not in ["image", "video", "string"] and k not in meta_keys
    }

    if skip_image_video:
        features_to_compute = numeric_features
    else:
        features_to_compute = {
            k: v for k, v in features.items() if v["dtype"] != "string" and k not in meta_keys
        }

    # 当启用 relative_action 时，通过基于块的采样计算动作统计信息
    # （与模型在训练期间看到的一致），并在下面按 episode 的
    # 遍历中跳过 action。
    relative_action_stats = None
    if relative_action and ACTION in features and OBS_STATE in features:
        if relative_exclude_joints is None:
            relative_exclude_joints = ["gripper"]
        relative_action_stats = compute_relative_action_stats(
            hf_dataset=dataset.hf_dataset,
            features=features,
            chunk_size=chunk_size,
            exclude_joints=relative_exclude_joints,
            num_workers=num_workers,
        )
        features_to_compute.pop(ACTION, None)

    logging.info(f"Recomputing stats for features: {list(features_to_compute.keys())}")

    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    all_episode_stats = []
    # TODO: 启用图像和视频统计信息的重新计算
    numeric_keys = [k for k, v in features_to_compute.items() if v["dtype"] not in ["image", "video"]]

    for parquet_path in tqdm(parquet_files, desc="Computing stats from data files"):
        df = pd.read_parquet(parquet_path)

        for ep_idx in sorted(df["episode_index"].unique()):
            ep_df = df[df["episode_index"] == ep_idx]
            episode_data = {}
            for key in numeric_keys:
                if key in ep_df.columns:
                    values = ep_df[key].values
                    if hasattr(values[0], "__len__"):
                        episode_data[key] = np.stack(values)
                    else:
                        episode_data[key] = np.array(values)

            ep_stats = compute_episode_stats(episode_data, features_to_compute)
            all_episode_stats.append(ep_stats)

    if features_to_compute and not all_episode_stats:
        logging.warning("No episode stats computed")
        return dataset

    new_stats = aggregate_stats(all_episode_stats) if all_episode_stats else {}

    if relative_action_stats is not None:
        new_stats[ACTION] = relative_action_stats

    # 合并：为未重新计算的特征保留现有统计信息
    if dataset.meta.stats:
        for key, value in dataset.meta.stats.items():
            if key not in new_stats:
                new_stats[key] = value

    write_stats(new_stats, dataset.root)
    dataset.meta.stats = new_stats

    logging.info("Stats recomputed successfully")
    return dataset


def convert_image_to_video_dataset(
    dataset: LeRobotDataset,
    output_dir: Path | None = None,
    repo_id: str | None = None,
    rgb_encoder: RGBEncoderConfig | None = None,
    depth_encoder: DepthEncoderConfig | None = None,
    episode_indices: list[int] | None = None,
    num_workers: int = 4,
    max_episodes_per_batch: int | None = None,
    max_frames_per_batch: int | None = None,
) -> LeRobotDataset:
    """转换图像到视频数据集。

    创建一个新的 LeRobotDataset，其中图像被编码为视频，遵循正确的
    LeRobot 数据集结构，视频存储在分块的 MP4 文件中。

    Args:
        dataset: 包含图像的源 LeRobot 数据集。
        output_dir: 转换后数据集的存储根目录。为 ``None`` 时，默认为
            ``$HF_LEROBOT_HOME/repo_id``。等同于 ``EditDatasetConfig`` 中的
            ``new_root``。
        repo_id: 转换后的数据集标识符。等同于 ``EditDatasetConfig`` 中的
            ``new_repo_id``。
        rgb_encoder: 应用于 RGB 相机的视频编码器设置。为 ``None`` 时，
            使用 :func:`~lerobot.configs.video.rgb_encoder_defaults`。
        depth_encoder: 应用于深度图相机的视频编码器设置，包括持久化到
            数据集元数据的量化参数。为 ``None`` 时，使用
            :func:`~lerobot.configs.video.depth_encoder_defaults`。
        episode_indices: 要转换的 episode 索引。为 ``None`` 时，转换所有
            episode。
        num_workers: 并行处理的线程数。
        max_episodes_per_batch: 每个视频批次的最大 episode 数，用于限制内存使用。
            ``None`` 表示无限制。
        max_frames_per_batch: 每个视频批次的最大帧数，用于限制内存使用。
            ``None`` 表示无限制。

    Returns:
        图像已编码为视频的新 :class:`LeRobotDataset`。
    """
    if rgb_encoder is None:
        rgb_encoder = rgb_encoder_defaults()
    if depth_encoder is None:
        depth_encoder = depth_encoder_defaults()

    # 检查这是一个图像数据集
    if len(dataset.meta.video_keys) > 0:
        raise ValueError(
            f"This operation is for image datasets only. Video dataset provided: {dataset.repo_id}"
        )

    # 获取所有图像键
    hf_dataset = dataset.hf_dataset.with_format(None)
    img_keys = [key for key in hf_dataset.features if key.startswith(OBS_IMAGE)]

    if len(img_keys) == 0:
        raise ValueError(f"No image keys found in dataset {dataset.repo_id}")

    # 确定要处理哪些 episode
    if episode_indices is None:
        episode_indices = list(range(dataset.meta.total_episodes))

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_video"

    logging.info(
        f"Converting {len(episode_indices)} episodes with {len(img_keys)} cameras from {dataset.repo_id}"
    )
    logging.info(f"RGB video encoder: {rgb_encoder}, depth video encoder: {depth_encoder}")

    # 创建新的特征字典，将图像特征转换为视频特征
    new_features = {}
    for key, value in dataset.meta.features.items():
        if key not in img_keys:
            new_features[key] = value
        else:
            # 将图像键转换为视频格式
            new_features[key] = value.copy()
            new_features[key]["dtype"] = "video"  # 将 dtype 从 "image" 改为 "video"
            # 视频信息将在 episode 编码完成后更新

    # 为视频数据集创建新的元数据
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=True,
        chunks_size=dataset.meta.chunks_size,
        data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
    )

    # 创建用于图像提取的临时目录
    temp_dir = output_dir / "temp_images"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 处理所有 episode 并批量编码视频
    # 使用字典实现 O(1) 的 episode 元数据查找，而不是 O(n) 的线性搜索
    all_episode_metadata = {}
    fps = int(dataset.fps)

    try:
        # 首先构建 episode 元数据条目
        logging.info("Building episode metadata...")
        cumulative_frame_idx = 0
        for ep_idx in episode_indices:
            src_episode = dataset.meta.episodes[ep_idx]
            ep_length = src_episode["length"]
            ep_meta = {
                "episode_index": ep_idx,
                "length": ep_length,
                "dataset_from_index": cumulative_frame_idx,
                "dataset_to_index": cumulative_frame_idx + ep_length,
            }
            if "data/chunk_index" in src_episode:
                ep_meta["data/chunk_index"] = src_episode["data/chunk_index"]
                ep_meta["data/file_index"] = src_episode["data/file_index"]
            all_episode_metadata[ep_idx] = ep_meta
            cumulative_frame_idx += ep_length

        # 处理每个相机，并将多个 episode 一起批量编码
        video_file_size_limit = new_meta.video_files_size_in_mb

        # 预先计算 episode 长度以便分批
        episode_lengths = {ep_idx: dataset.meta.episodes["length"][ep_idx] for ep_idx in episode_indices}

        for img_key in tqdm(img_keys, desc="Processing cameras"):
            target_encoder = depth_encoder if img_key in dataset.meta.depth_keys else rgb_encoder

            # 通过编码一个小的校准样本来估计每帧大小
            # 这为特定的编解码器参数提供了准确的压缩率
            size_per_frame_mb = _estimate_frame_size_via_calibration(
                dataset=dataset,
                img_key=img_key,
                episode_indices=episode_indices,
                temp_dir=temp_dir,
                fps=fps,
                video_encoder=target_encoder,
            )

            logging.info(f"Processing camera: {img_key}")
            chunk_idx, file_idx = 0, 0
            cumulative_timestamp = 0.0

            # 分批处理 episode 以保持在大小限制之下
            for batch_episodes in _iter_episode_batches(
                episode_indices=episode_indices,
                episode_lengths=episode_lengths,
                size_per_frame_mb=size_per_frame_mb,
                video_file_size_limit=video_file_size_limit,
                max_episodes=max_episodes_per_batch,
                max_frames=max_frames_per_batch,
            ):
                total_frames_in_batch = sum(episode_lengths[idx] for idx in batch_episodes)
                logging.info(
                    f"  Encoding batch of {len(batch_episodes)} episodes "
                    f"({batch_episodes[0]}-{batch_episodes[-1]}) = {total_frames_in_batch} frames"
                )

                # 保存此批次中所有 episode 的图像
                imgs_dir = temp_dir / f"batch_{chunk_idx}_{file_idx}" / img_key
                episode_durations = _save_batch_episodes_images(
                    dataset=dataset,
                    imgs_dir=imgs_dir,
                    img_key=img_key,
                    episode_indices=batch_episodes,
                    num_workers=num_workers,
                )

                # 将批次中的所有 episode 编码为单个视频
                video_path = new_meta.root / new_meta.video_path.format(
                    video_key=img_key, chunk_index=chunk_idx, file_index=file_idx
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)

                encode_video_frames(
                    imgs_dir=imgs_dir,
                    video_path=video_path,
                    fps=fps,
                    video_encoder=target_encoder,
                    overwrite=True,
                )

                # 清理临时图像
                shutil.rmtree(imgs_dir)

                # 更新批次中每个 episode 的元数据
                for ep_idx, duration in zip(batch_episodes, episode_durations, strict=True):
                    from_timestamp = cumulative_timestamp
                    to_timestamp = cumulative_timestamp + duration
                    cumulative_timestamp = to_timestamp

                    # 查找 episode 元数据条目并添加视频元数据（O(1) 字典查找）
                    ep_meta = all_episode_metadata[ep_idx]
                    ep_meta[f"videos/{img_key}/chunk_index"] = chunk_idx
                    ep_meta[f"videos/{img_key}/file_index"] = file_idx
                    ep_meta[f"videos/{img_key}/from_timestamp"] = from_timestamp
                    ep_meta[f"videos/{img_key}/to_timestamp"] = to_timestamp

                # 为下一批次移动到下一个视频文件
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, new_meta.chunks_size)
                cumulative_timestamp = 0.0

        # 复制并转换数据文件（移除图像列）
        _copy_data_without_images(dataset, new_meta, episode_indices, img_keys)

        # 保存 episode 元数据
        episodes_df = pd.DataFrame(list(all_episode_metadata.values()))
        episodes_path = new_meta.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_df.to_parquet(episodes_path, index=False)

        # 更新元数据信息
        new_meta.info.total_episodes = len(episode_indices)
        new_meta.info.total_frames = sum(ep["length"] for ep in all_episode_metadata.values())
        new_meta.info.total_tasks = dataset.meta.total_tasks
        new_meta.info.splits = {"train": f"0:{len(episode_indices)}"}

        # 更新所有图像键（现在是视频）的视频信息。它们在上方已注册为
        # 视频特征，因此 update_video_info 会填充它们（仍为空的）信息。
        for img_key in img_keys:
            target_encoder = depth_encoder if img_key in dataset.meta.depth_keys else rgb_encoder
            new_meta.update_video_info(video_key=img_key, video_encoder=target_encoder)

        write_info(new_meta.info, new_meta.root)

        # 复制统计信息和任务
        if dataset.meta.stats is not None:
            # 移除图像统计信息
            new_stats = {k: v for k, v in dataset.meta.stats.items() if k not in img_keys}
            write_stats(new_stats, new_meta.root)

        if dataset.meta.tasks is not None:
            write_tasks(dataset.meta.tasks, new_meta.root)

    finally:
        # 清理临时目录
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    logging.info(f"Completed converting {dataset.repo_id} to video format")
    logging.info(f"New dataset saved to: {output_dir}")

    # 返回新数据集
    return LeRobotDataset(repo_id=repo_id, root=output_dir)


def _reencode_video_worker(args: tuple) -> Path:
    """:func:`reencode_dataset` 进程池的可 pickle 工作函数。"""
    video_path, video_encoder, encoder_threads = args
    reencode_video(
        input_video_path=video_path,
        output_video_path=video_path,
        video_encoder=video_encoder,
        encoder_threads=encoder_threads,
        overwrite=True,
    )
    return video_path


def reencode_dataset(
    dataset: LeRobotDataset,
    rgb_encoder: RGBEncoderConfig | None = None,
    depth_encoder: DepthEncoderConfig | None = None,
    encoder_threads: int | None = None,
    num_workers: int | None = None,
) -> LeRobotDataset:
    """使用一组新的编码参数重新编码数据集中的每个视频。

    视频会被原地重新编码，并且 ``info.json`` 中的视频信息会被刷新。

    Args:
        dataset: 一个现有的 :class:`LeRobotDataset`，其视频将被
            重新编码。
        rgb_encoder: 应用于每个 RGB 视频文件的目标编码器配置。
            如果为 ``None``，则跳过 RGB 视频的重新编码。
        depth_encoder: 应用于每个深度视频文件的目标编码器配置。
            如果为 ``None``，则跳过深度视频的重新编码。
            量化参数不会覆盖当前数据集中的参数。
        encoder_threads: 转发给 :func:`reencode_video` 的每个编码器的
            线程数。``None`` 表示由编解码器决定。
        num_workers: 并行进程数。``None`` 或 ``0`` 表示顺序执行
            （不使用多进程）；``1+`` 会启动一个
            :class:`~concurrent.futures.ProcessPoolExecutor`。

    Returns:
        元数据已在磁盘上更新的同一 :class:`LeRobotDataset` 实例。
    """
    meta = dataset.meta
    video_keys_encoders_dict = {}
    video_keys_paths_dict = {}

    if rgb_encoder is None and depth_encoder is None:
        raise ValueError("Either rgb_encoder or depth_encoder must be provided")

    # 仅当视频尚未使用给定的视频编码参数编码时才重新编码
    for video_key in meta.video_keys:
        current_info = meta.info.features[video_key].get("info", {})
        current_encoder = encoder_config_from_video_info(current_info)
        target_encoder = depth_encoder if video_key in meta.depth_keys else rgb_encoder
        if target_encoder is None:
            logging.info(f"No encoder provided for {video_key} video. Skipping re-encoding.")
        elif current_encoder != target_encoder:
            video_keys_paths_dict[video_key] = list((meta.root / VIDEO_DIR / video_key).rglob("*.mp4"))
            video_keys_encoders_dict[video_key] = target_encoder
        else:
            logging.info(f"{video_key} videos are already encoded with {target_encoder}. Nothing to do.")

    if len(video_keys_paths_dict) == 0:
        logging.warning("Dataset has no videos to re-encode.")
        return dataset
    logging.info(f"Re-encoding {sum(len(paths) for paths in video_keys_paths_dict.values())} video file(s).")

    worker_args = [
        (path, encoder, encoder_threads)
        for video_key, encoder in video_keys_encoders_dict.items()
        for path in video_keys_paths_dict[video_key]
    ]
    if num_workers and num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_reencode_video_worker, args) for args in worker_args]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Re-encoding videos",
            ):
                future.result()
    else:
        for args in tqdm(worker_args, desc="Re-encoding videos"):
            _reencode_video_worker(args)

    # 为每个重新编码的键刷新元数据中的视频信息。重新编码只改变
    # 编解码器/容器参数，因此对于深度视频，我们保留 ``is_depth_map``
    # 和深度量化参数（``video.depth_min`` / ``video.depth_max`` / ...），
    # 它们描述的是数据而非编解码器，必须在转码后继续存在。
    # RGB 视频传入空集合：仍然会刷新，但没有需要保留的内容。
    depth_preserve_keys = {"is_depth_map", *(f"video.{n}" for n in DEPTH_ENCODER_INFO_FIELD_NAMES)}
    for video_key, encoder in video_keys_encoders_dict.items():
        preserve_keys = depth_preserve_keys if video_key in meta.depth_keys else set()
        meta.update_video_info(video_key=video_key, video_encoder=encoder, preserve_keys=preserve_keys)

    write_info(meta.info, meta.root)
    logging.info("Dataset metadata updated.")

    return dataset
