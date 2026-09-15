#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

import copy
import logging
import shutil
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import datasets
import numpy as np
import pandas as pd
import tqdm

from lerobot.configs import VIDEO_ENCODER_INFO_KEYS

from .compute_stats import aggregate_stats
from .dataset_metadata import LeRobotDatasetMetadata
from .feature_utils import canonicalize_depth_marker, features_equal_for_merge, get_hf_features_from_features
from .io_utils import (
    get_file_size_in_mb,
    get_parquet_file_size_in_mb,
    to_parquet_one_row_group_per_episode,
    to_parquet_with_hf_images,
    write_info,
    write_stats,
    write_tasks,
)
from .utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    update_chunk_file_indices,
)
from .video_utils import concatenate_video_files, get_video_duration_in_s

logger = logging.getLogger(__name__)

type FeatureDict = dict[str, dict[str, Any]]
type ChunkFile = tuple[int, int]


class IndexState(TypedDict):
    chunk: int
    file: int
    src_to_dst: NotRequired[dict[ChunkFile, ChunkFile]]


class VideoIndex(TypedDict):
    chunk: int
    file: int
    latest_duration: float
    episode_duration: float
    src_to_offset: NotRequired[dict[ChunkFile, float]]
    src_to_dst: NotRequired[dict[ChunkFile, ChunkFile]]
    dst_file_durations: NotRequired[dict[ChunkFile, float]]


type VideoIndexState = dict[str, VideoIndex]


def merge_video_feature_info_for_aggregate(all_metadata: list[LeRobotDatasetMetadata]) -> FeatureDict:
    """为聚合创建合并后的视频特征信息字典。视频编码器信息按字段合并：仅当所有来源一致时才保留每个键；否则该键被设为 ``null``（``video.extra_options`` 则为 ``{}``）并记录警告。

    Args:
        all_metadata: 要合并的 LeRobotDatasetMetadata 对象列表。

    Returns:
        dict: 合并后的视频特征信息字典。
    """
    merged_info: FeatureDict = copy.deepcopy(all_metadata[0].features)
    video_keys = [k for k in merged_info if merged_info[k].get("dtype") == "video"]

    for vk in video_keys:
        video_infos = [m.features.get(vk, {}).get("info") or {} for m in all_metadata]
        base_video_info = video_infos[0]

        merged_encoder_info: dict[str, Any] = {}
        fallback_keys: list[str] = []
        for info_key in VIDEO_ENCODER_INFO_KEYS:
            values = [info.get(info_key, None) for info in video_infos]
            first_value = values[0]
            all_match = all(v == first_value for v in values[1:])

            if all_match:
                merged_encoder_info[info_key] = first_value
            else:
                fallback_keys.append(info_key)
                merged_encoder_info[info_key] = {} if info_key == "video.extra_options" else None

        if fallback_keys:
            logger.warning(
                f"Merging heterogeneous or incomplete video encoder metadata for feature {vk}. "
                f"Setting these keys to null: {fallback_keys}.",
            )

        merged_info[vk]["info"] = {**base_video_info, **merged_encoder_info}
        # TODO(CarolinePascal): 一旦我们支持其他视频后端，就把这里改为变量。
        merged_info[vk]["info"]["video.video_backend"] = "pyav"
        # 即使某个来源使用了旧版键，也要持久化规范的深度标记。
        canonicalize_depth_marker(merged_info[vk])

    return merged_info


def validate_all_metadata(all_metadata: list[LeRobotDatasetMetadata]) -> tuple[int, str | None, FeatureDict]:
    """验证所有数据集元数据具有一致的属性。

    确保所有数据集具有相同的 fps、robot_type 和 features，以保证
    将它们聚合为单个数据集时的兼容性。
    验证时不考虑视频编码器信息，但会在聚合过程中通过 ``merge_video_feature_info_for_aggregate`` 进行合并。

    Args:
        all_metadata: 要验证的 LeRobotDatasetMetadata 对象列表。

    Returns:
        tuple: 包含来自第一个元数据的 (fps, robot_type, features) 的元组。

    Raises:
        ValueError: 如果任何元数据的 fps、robot_type 或 features
                   与列表中第一个元数据不同。
    """

    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features

    for meta in tqdm.tqdm(all_metadata, desc="Validate all meta data"):
        if fps != meta.fps:
            raise ValueError(f"Same fps is expected, but got fps={meta.fps} instead of {fps}.")
        if robot_type != meta.robot_type:
            raise ValueError(
                f"Same robot_type is expected, but got robot_type={meta.robot_type} instead of {robot_type}."
            )
        if not features_equal_for_merge(features, meta.features):
            raise ValueError(
                f"Same features is expected, but got features={meta.features} instead of {features}."
            )

    return fps, robot_type, features


def update_data_df(
    df: pd.DataFrame, src_meta: LeRobotDatasetMetadata, dst_meta: LeRobotDatasetMetadata
) -> pd.DataFrame:
    """为聚合更新数据 DataFrame 的索引和任务映射。

    调整 episode 索引、帧索引和任务索引，以考虑
    目标数据集中先前已聚合的数据。

    Args:
        df: 包含要更新数据的 DataFrame。
        src_meta: 源数据集元数据。
        dst_meta: 目标数据集元数据。

    Returns:
        pd.DataFrame: 索引已调整的更新后 DataFrame。
    """

    df["episode_index"] = df["episode_index"] + dst_meta.info.total_episodes
    df["index"] = df["index"] + dst_meta.info.total_frames

    src_task_names = src_meta.tasks.index.take(df["task_index"].to_numpy())
    df["task_index"] = dst_meta.tasks.loc[src_task_names, "task_index"].to_numpy()

    return df


def update_meta_data(
    df: pd.DataFrame,
    dst_meta: LeRobotDatasetMetadata,
    meta_idx: IndexState,
    data_idx: IndexState,
    videos_idx: VideoIndexState,
) -> pd.DataFrame:
    """使用新的 chunk、file 和时间戳索引更新元数据 DataFrame。

    调整所有索引和时间戳，以考虑目标数据集中先前已聚合的
    数据和视频。

    对于数据文件索引，使用来自 aggregate_data() 的 'src_to_dst' 映射，
    以正确地将源文件索引映射到其目标位置。

    Args:
        df: 包含要更新元数据的 DataFrame。
        dst_meta: 目标数据集元数据。
        meta_idx: 包含当前元数据 chunk 和 file 索引的字典。
        data_idx: 包含当前数据 chunk 和 file 索引的字典。
        videos_idx: 包含当前视频索引和时间戳的字典。

    Returns:
        pd.DataFrame: 索引和时间戳已调整的更新后 DataFrame。
    """

    df["meta/episodes/chunk_index"] = df["meta/episodes/chunk_index"] + meta_idx["chunk"]
    df["meta/episodes/file_index"] = df["meta/episodes/file_index"] + meta_idx["file"]

    # 使用源到目标的映射更新数据文件索引
    # 这对于处理已经是合并结果的数据集至关重要
    data_src_to_dst = data_idx.get("src_to_dst", {})
    if data_src_to_dst:
        # 保存原始索引以供查找
        df["_orig_data_chunk"] = df["data/chunk_index"].copy()
        df["_orig_data_file"] = df["data/file_index"].copy()

        # 从 (src_chunk, src_file) 到 (dst_chunk, dst_file) 的向量化映射
        # 对于大型元数据表，这比逐行迭代快得多
        mapping_index = pd.MultiIndex.from_tuples(
            list(data_src_to_dst.keys()),
            names=["chunk_index", "file_index"],
        )
        mapping_values = list(data_src_to_dst.values())
        mapping_df = pd.DataFrame(
            mapping_values,
            index=mapping_index,
            columns=["dst_chunk", "dst_file"],
        )

        # 基于原始数据索引为每一行构造 MultiIndex
        row_index = pd.MultiIndex.from_arrays(
            [df["_orig_data_chunk"], df["_orig_data_file"]],
            names=["chunk_index", "file_index"],
        )

        # 将映射与行对齐；缺失的键回退到默认目标
        reindexed = mapping_df.reindex(row_index)
        reindexed[["dst_chunk", "dst_file"]] = reindexed[["dst_chunk", "dst_file"]].fillna(
            {"dst_chunk": data_idx["chunk"], "dst_file": data_idx["file"]}
        )

        # 将映射后的目标索引赋值回 DataFrame
        df["data/chunk_index"] = reindexed["dst_chunk"].to_numpy()
        df["data/file_index"] = reindexed["dst_file"].to_numpy()

        # 清理临时列
        df = df.drop(columns=["_orig_data_chunk", "_orig_data_file"])
    else:
        # 回退到简单偏移（对单文件来源的向后兼容）
        df["data/chunk_index"] = df["data/chunk_index"] + data_idx["chunk"]
        df["data/file_index"] = df["data/file_index"] + data_idx["file"]
    for key, video_idx in videos_idx.items():
        # 在更新之前保存原始视频文件索引
        orig_chunk_col = f"videos/{key}/chunk_index"
        orig_file_col = f"videos/{key}/file_index"
        df["_orig_chunk"] = df[orig_chunk_col].copy()
        df["_orig_file"] = df[orig_file_col].copy()

        # 获取此视频键的映射
        src_to_offset = video_idx.get("src_to_offset", {})
        src_to_dst = video_idx.get("src_to_dst", {})

        # 应用按源文件的映射
        if src_to_dst:
            # 将每个 episode 映射到其正确的目标文件并应用偏移
            for idx in df.index:
                src_key = (df.at[idx, "_orig_chunk"], df.at[idx, "_orig_file"])

                # 获取此源文件的目标 chunk/file
                dst_chunk, dst_file = src_to_dst.get(src_key, (video_idx["chunk"], video_idx["file"]))
                df.at[idx, orig_chunk_col] = dst_chunk
                df.at[idx, orig_file_col] = dst_file

                # 应用时间戳偏移
                offset = src_to_offset.get(src_key, 0)
                df.at[idx, f"videos/{key}/from_timestamp"] += offset
                df.at[idx, f"videos/{key}/to_timestamp"] += offset
        elif src_to_offset:
            # 回退：对所有文件使用相同的目标，但应用按文件的偏移
            df[orig_chunk_col] = video_idx["chunk"]
            df[orig_file_col] = video_idx["file"]
            for idx in df.index:
                src_key = (df.at[idx, "_orig_chunk"], df.at[idx, "_orig_file"])
                offset = src_to_offset.get(src_key, 0)
                df.at[idx, f"videos/{key}/from_timestamp"] += offset
                df.at[idx, f"videos/{key}/to_timestamp"] += offset
        else:
            # 回退到简单偏移（为了向后兼容）
            df[orig_chunk_col] = video_idx["chunk"]
            df[orig_file_col] = video_idx["file"]
            df[f"videos/{key}/from_timestamp"] = (
                df[f"videos/{key}/from_timestamp"] + video_idx["latest_duration"]
            )
            df[f"videos/{key}/to_timestamp"] = df[f"videos/{key}/to_timestamp"] + video_idx["latest_duration"]

        # 清理临时列
        df = df.drop(columns=["_orig_chunk", "_orig_file"])

    df["dataset_from_index"] = df["dataset_from_index"] + dst_meta.info.total_frames
    df["dataset_to_index"] = df["dataset_to_index"] + dst_meta.info.total_frames
    df["episode_index"] = df["episode_index"] + dst_meta.info.total_episodes

    # 每个 episode 的统计信息仍描述上面重新索引的簿记列的合并前值。
    # index/episode_index 按常量偏移；task_index 被重新标记，
    # 因此通过统一的 tasks 表，从 episode 的（稳定的）任务字符串重新计算它。
    shift_stat_keys = ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")
    for name, offset in (
        ("episode_index", dst_meta.info.total_episodes),
        ("index", dst_meta.info.total_frames),
    ):
        for stat in shift_stat_keys:
            col = f"stats/{name}/{stat}"
            if col in df.columns:
                df[col] = df[col] + offset

    if any(c.startswith("stats/task_index/") for c in df.columns):
        quantiles = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
        ids_per_row = [
            np.array([dst_meta.tasks.loc[t, "task_index"] for t in tasks], dtype=np.float64)
            for tasks in df["tasks"]
        ]

        def _task_stat(ids, stat):
            if stat == "min":
                return ids.min()
            if stat == "max":
                return ids.max()
            if stat == "std":
                return ids.std()
            if stat in quantiles:
                return np.quantile(ids, quantiles[stat])
            return ids.mean()

        for stat in ("min", "max", "mean", "std", *quantiles):
            col = f"stats/task_index/{stat}"
            if col in df.columns:
                # np.full_like 保留每个单元格的容器和 dtype，因此 parquet 模式保持不变。
                df[col] = [
                    np.full_like(orig, _task_stat(ids, stat))
                    for orig, ids in zip(df[col], ids_per_row, strict=True)
                ]

    return df


def aggregate_datasets(
    repo_ids: list[str],
    aggr_repo_id: str,
    roots: list[Path] | None = None,
    aggr_root: Path | None = None,
    data_files_size_in_mb: int | None = None,
    video_files_size_in_mb: int | None = None,
    chunk_size: int | None = None,
    concatenate_videos: bool = True,
    concatenate_data: bool = True,
) -> None:
    """将多个 LeRobot 数据集聚合为单个统一的数据集。

    这是编排聚合过程的主函数，步骤如下：
    1. 加载并验证所有源数据集的元数据
    2. 创建具有统一任务的新目标数据集
    3. 聚合所有源数据集的视频、数据和元数据
    4. 以正确的统计信息完成聚合数据集

    Args:
        repo_ids: 要聚合的数据集的仓库 ID 列表。
        aggr_repo_id: 聚合输出数据集的仓库 ID。
        roots: 可选列表，包含每个源数据集的一个根路径。
        aggr_root: 聚合数据集的可选根路径。
        data_files_size_in_mb: 数据文件的最大大小（MB）（默认为 DEFAULT_DATA_FILE_SIZE_IN_MB）
        video_files_size_in_mb: 视频文件的最大大小（MB）（默认为 DEFAULT_VIDEO_FILE_SIZE_IN_MB）
        chunk_size: 每个 chunk 的最大文件数（默认为 DEFAULT_CHUNK_SIZE）
        concatenate_videos: 为 False 时，每个源文件保留一个 mp4，而不是打包成分片。
        concatenate_data: 为 False 时，每个源文件保留一个 parquet，而不是打包成分片。
    """
    logger.info("Start aggregate_datasets")

    if roots is not None and len(roots) != len(repo_ids):
        raise ValueError("repo_ids and roots must have the same length")

    if data_files_size_in_mb is None:
        data_files_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_files_size_in_mb is None:
        video_files_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    all_metadata = (
        [LeRobotDatasetMetadata(repo_id) for repo_id in repo_ids]
        if roots is None
        else [
            LeRobotDatasetMetadata(repo_id, root=root) for repo_id, root in zip(repo_ids, roots, strict=True)
        ]
    )
    fps, robot_type, _ = validate_all_metadata(all_metadata)
    features = merge_video_feature_info_for_aggregate(all_metadata)
    video_keys = [key for key in features if features[key]["dtype"] == "video"]

    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=aggr_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=aggr_root,
        use_videos=len(video_keys) > 0,
        chunks_size=chunk_size,
        data_files_size_in_mb=data_files_size_in_mb,
        video_files_size_in_mb=video_files_size_in_mb,
    )

    logger.info("Find all tasks")
    unique_tasks = pd.concat([m.tasks for m in all_metadata]).index.unique()
    dst_meta.tasks = pd.DataFrame(
        {"task_index": range(len(unique_tasks))}, index=pd.Index(unique_tasks, name="task")
    )

    meta_idx: IndexState = {"chunk": 0, "file": 0}
    data_idx: IndexState = {"chunk": 0, "file": 0}
    videos_idx: VideoIndexState = {
        key: {"chunk": 0, "file": 0, "latest_duration": 0, "episode_duration": 0} for key in video_keys
    }

    dst_meta.episodes = {}

    for src_meta in tqdm.tqdm(all_metadata, desc="Copy data and videos"):
        videos_idx = aggregate_videos(
            src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size, concatenate_videos
        )
        data_idx = aggregate_data(
            src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size, concatenate_data
        )

        meta_idx = aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx)

        # 在处理完每个源数据集后清除 src_to_dst 映射，
        # 以避免不同源数据集之间的干扰
        data_idx.pop("src_to_dst", None)

        dst_meta.info.total_episodes += src_meta.total_episodes
        dst_meta.info.total_frames += src_meta.total_frames

    finalize_aggregation(dst_meta, all_metadata)
    logger.info("Aggregation complete.")


def aggregate_videos(
    src_meta: LeRobotDatasetMetadata,
    dst_meta: LeRobotDatasetMetadata,
    videos_idx: VideoIndexState,
    video_files_size_in_mb: float,
    chunk_size: int,
    concatenate_videos: bool = True,
) -> VideoIndexState:
    """将源数据集的视频块聚合到目标数据集。

    根据文件大小限制处理视频文件的拼接和轮换。
    当超出大小限制时创建新的视频文件。

    Args:
        src_meta: 源数据集元数据。
        dst_meta: 目标数据集元数据。
        videos_idx: 跟踪视频 chunk 和 file 索引的字典。
        video_files_size_in_mb: 视频文件的最大大小（MB）（默认为 DEFAULT_VIDEO_FILE_SIZE_IN_MB）
        chunk_size: 每个 chunk 的最大文件数（默认为 DEFAULT_CHUNK_SIZE）
        concatenate_videos: 为 False 时，每个源文件保留一个 mp4，而不是打包成分片。
    Returns:
        dict: 更新后的 videos_idx，包含当前的 chunk 和 file 索引。
    """
    for key in videos_idx:
        videos_idx[key]["episode_duration"] = 0
        # 跟踪每个源 (chunk, file) 对的偏移
        videos_idx[key]["src_to_offset"] = {}
        # 跟踪每个源 (chunk, file) 对的目标 (chunk, file)
        videos_idx[key]["src_to_dst"] = {}
        # 如果不存在则初始化 dst_file_durations
        # dst_file_durations 跟踪每个目标文件的时长
        if "dst_file_durations" not in videos_idx[key]:
            videos_idx[key]["dst_file_durations"] = {}

    for key, video_idx in videos_idx.items():
        unique_chunk_file_pairs: list[ChunkFile] = sorted(
            {
                (chunk, file)
                for chunk, file in zip(
                    src_meta.episodes[f"videos/{key}/chunk_index"],
                    src_meta.episodes[f"videos/{key}/file_index"],
                    strict=False,
                )
            }
        )

        chunk_idx = video_idx["chunk"]
        file_idx = video_idx["file"]
        dst_file_durations = video_idx["dst_file_durations"]

        for src_chunk_idx, src_file_idx in unique_chunk_file_pairs:
            src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=src_chunk_idx,
                file_index=src_file_idx,
            )

            dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )

            src_duration = get_video_duration_in_s(src_path)
            dst_key = (chunk_idx, file_idx)

            if not dst_path.exists():
                # 新的目标文件：偏移为 0
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                # 跟踪此目标文件的时长
                dst_file_durations[dst_key] = src_duration
                videos_idx[key]["episode_duration"] += src_duration
                continue

            # 在追加之前检查文件大小
            src_size = get_file_size_in_mb(src_path)
            dst_size = get_file_size_in_mb(dst_path)

            if not concatenate_videos or dst_size + src_size >= video_files_size_in_mb:
                # 轮换到新文件——偏移为 0
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                dst_key = (chunk_idx, file_idx)
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
                dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                # 跟踪此新目标文件的时长
                dst_file_durations[dst_key] = src_duration
            else:
                # 追加到现有的目标文件
                # 偏移是此目标文件的当前时长
                current_dst_duration = dst_file_durations.get(dst_key, 0)
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = current_dst_duration
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
                # TODO(CarolinePascal): 将检查移到循环之前以避免中途失败 + 如果检查失败则增加重新编码视频的可能性
                concatenate_video_files(
                    [dst_path, src_path],
                    dst_path,
                    compatibility_check=True,
                )
                # 更新此目标文件的时长
                dst_file_durations[dst_key] = current_dst_duration + src_duration

            videos_idx[key]["episode_duration"] += src_duration

        videos_idx[key]["chunk"] = chunk_idx
        videos_idx[key]["file"] = file_idx

    return videos_idx


def aggregate_data(
    src_meta: LeRobotDatasetMetadata,
    dst_meta: LeRobotDatasetMetadata,
    data_idx: IndexState,
    data_files_size_in_mb: float,
    chunk_size: int,
    concatenate_data: bool = True,
) -> IndexState:
    """将源数据集的数据块聚合到目标数据集。

    读取源数据文件，更新索引以匹配聚合后的数据集，
    并通过适当的文件轮换将它们写入目标位置。

    跟踪从源 (chunk, file) 到目标 (chunk, file) 的 `src_to_dst` 映射，
    当源数据集具有多个数据文件（例如来自先前的合并操作）时，
    这对于正确更新 episode 元数据至关重要。

    Args:
        src_meta: 源数据集元数据。
        dst_meta: 目标数据集元数据。
        data_idx: 跟踪数据 chunk 和 file 索引的字典。
        data_files_size_in_mb: 数据文件的最大大小（MB）。
        chunk_size: 每个 chunk 的最大文件数。
        concatenate_data: 为 False 时，每个源文件保留一个 parquet，而不是打包成分片。

    Returns:
        dict: 更新后的 data_idx，包含当前的 chunk 和 file 索引。
    """
    unique_chunk_file_ids: list[ChunkFile] = sorted(
        {
            (c, f)
            for c, f in zip(
                src_meta.episodes["data/chunk_index"],
                src_meta.episodes["data/file_index"],
                strict=False,
            )
        }
    )
    contains_images = len(dst_meta.image_keys) > 0

    # 获取 features 模式以便在 parquet 中正确进行图像类型标注
    hf_features = get_hf_features_from_features(dst_meta.features) if contains_images else None

    # 跟踪源到目标文件的映射，用于更新元数据
    # 这对于处理已经是合并结果的数据集至关重要
    src_to_dst: dict[ChunkFile, ChunkFile] = {}

    for src_chunk_idx, src_file_idx in unique_chunk_file_ids:
        src_path = src_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=src_chunk_idx, file_index=src_file_idx
        )
        if contains_images:
            # 使用 HuggingFace datasets 读取源数据以保留图像格式
            src_ds = datasets.Dataset.from_parquet(str(src_path))
            df = src_ds.to_pandas()
        else:
            df = pd.read_parquet(src_path)
        df = update_data_df(df, src_meta, dst_meta)

        # 写入数据并获取其实际写入的目标文件
        # 这避免了在此处重复轮换逻辑
        data_idx, (dst_chunk, dst_file) = append_or_create_parquet_file(
            df,
            src_path,
            data_idx,
            data_files_size_in_mb,
            chunk_size,
            DEFAULT_DATA_PATH,
            contains_images=contains_images,
            aggr_root=dst_meta.root,
            hf_features=hf_features,
            concatenate=concatenate_data,
            one_row_group_per_episode=True,
        )

        # 记录从源到实际目标的映射
        src_to_dst[(src_chunk_idx, src_file_idx)] = (dst_chunk, dst_file)

    # 将映射添加到 data_idx，供元数据更新时使用
    data_idx["src_to_dst"] = src_to_dst

    return data_idx


def aggregate_metadata(
    src_meta: LeRobotDatasetMetadata,
    dst_meta: LeRobotDatasetMetadata,
    meta_idx: IndexState,
    data_idx: IndexState,
    videos_idx: VideoIndexState,
) -> IndexState:
    """将源数据集的元数据聚合到目标数据集。

    读取源元数据文件，更新所有索引和时间戳，
    并通过适当的文件轮换将它们写入目标位置。

    Args:
        src_meta: 源数据集元数据。
        dst_meta: 目标数据集元数据。
        meta_idx: 跟踪元数据 chunk 和 file 索引的字典。
        data_idx: 跟踪数据 chunk 和 file 索引的字典。
        videos_idx: 跟踪视频索引和时间戳的字典。

    Returns:
        dict: 更新后的 meta_idx，包含当前的 chunk 和 file 索引。
    """
    chunk_file_ids: list[ChunkFile] = sorted(
        {
            (c, f)
            for c, f in zip(
                src_meta.episodes["meta/episodes/chunk_index"],
                src_meta.episodes["meta/episodes/file_index"],
                strict=False,
            )
        }
    )
    for chunk_idx, file_idx in chunk_file_ids:
        src_path = src_meta.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        df = pd.read_parquet(src_path)
        df = update_meta_data(
            df,
            dst_meta,
            meta_idx,
            data_idx,
            videos_idx,
        )

        meta_idx, _ = append_or_create_parquet_file(
            df,
            src_path,
            meta_idx,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_CHUNK_SIZE,
            DEFAULT_EPISODES_PATH,
            contains_images=False,
            aggr_root=dst_meta.root,
        )

    # 将 latest_duration 增加此源数据集所添加的总时长
    for k in videos_idx:
        videos_idx[k]["latest_duration"] += videos_idx[k]["episode_duration"]

    return meta_idx


def append_or_create_parquet_file(
    df: pd.DataFrame,
    src_path: Path,
    idx: IndexState,
    max_mb: float,
    chunk_size: int,
    default_path: str,
    contains_images: bool = False,
    aggr_root: Path | None = None,
    hf_features: datasets.Features | None = None,
    concatenate: bool = True,
    one_row_group_per_episode: bool = False,
) -> tuple[IndexState, ChunkFile]:
    """根据大小约束将数据追加到现有的 parquet 文件或创建新文件。

    当超出大小限制时管理文件轮换，以防止单个文件
    变得过大。同时处理普通 parquet 文件和包含图像的文件。

    Args:
        df: 要写入 parquet 文件的 DataFrame。
        src_path: 源文件路径（用于大小估算）。
        idx: 包含当前 'chunk' 和 'file' 索引的字典。
        max_mb: 轮换前允许的最大文件大小（MB）。
        chunk_size: 递增 chunk 索引之前每个 chunk 的最大文件数。
        default_path: 用于生成文件路径的格式字符串。
        contains_images: 数据是否包含需要特殊处理的图像。
        aggr_root: 聚合数据集的根路径。
        hf_features: 可选的 HuggingFace Features 模式，用于正确的图像类型标注。
        concatenate: 为 False 时，总是轮换到新文件而不是追加到当前文件。
        one_row_group_per_episode: 对 DATA parquet 为 True（每个 episode 输出一个行组）；
            对 episodes 元数据 parquet 为 False（已经是每个 episode 一行）。

    Returns:
        tuple: (updated_idx, (dst_chunk, dst_file))，其中 updated_idx 是索引字典，
               (dst_chunk, dst_file) 是数据实际写入的目标文件。

    Raises:
        ValueError: 如果未提供 aggr_root。
    """
    if aggr_root is None:
        raise ValueError("aggr_root must be provided.")

    dst_chunk, dst_file = idx["chunk"], idx["file"]
    dst_path = aggr_root / default_path.format(chunk_index=dst_chunk, file_index=dst_file)

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if contains_images:
            to_parquet_with_hf_images(df, dst_path, features=hf_features)
        elif one_row_group_per_episode:
            to_parquet_one_row_group_per_episode(df, dst_path)
        else:
            df.to_parquet(dst_path)
        return idx, (dst_chunk, dst_file)

    src_size = get_parquet_file_size_in_mb(src_path)
    dst_size = get_parquet_file_size_in_mb(dst_path)

    if not concatenate or dst_size + src_size >= max_mb:
        idx["chunk"], idx["file"] = update_chunk_file_indices(idx["chunk"], idx["file"], chunk_size)
        dst_chunk, dst_file = idx["chunk"], idx["file"]
        new_path = aggr_root / default_path.format(chunk_index=dst_chunk, file_index=dst_file)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        final_df = df
        target_path = new_path
    else:
        if contains_images:
            # 使用 HuggingFace datasets 读取现有数据以保留图像格式
            existing_ds = datasets.Dataset.from_parquet(str(dst_path))
            existing_df = existing_ds.to_pandas()
        else:
            existing_df = pd.read_parquet(dst_path)
        final_df = pd.concat([existing_df, df], ignore_index=True)
        target_path = dst_path

    if contains_images:
        to_parquet_with_hf_images(final_df, target_path, features=hf_features)
    elif one_row_group_per_episode:
        to_parquet_one_row_group_per_episode(final_df, target_path)
    else:
        final_df.to_parquet(target_path)

    return idx, (dst_chunk, dst_file)


def finalize_aggregation(
    aggr_meta: LeRobotDatasetMetadata, all_metadata: list[LeRobotDatasetMetadata]
) -> None:
    """通过写入摘要文件和统计信息来完成数据集聚合。

    写入 tasks 文件、包含总计数和 splits 的 info 文件，以及
    来自所有源数据集的聚合统计信息。

    Args:
        aggr_meta: 聚合后的数据集元数据。
        all_metadata: 所有源数据集元数据对象的列表。
    """
    logger.info("write tasks")
    write_tasks(aggr_meta.tasks, aggr_meta.root)

    logger.info("write info")
    aggr_meta.info.total_tasks = len(aggr_meta.tasks)
    aggr_meta.info.total_episodes = sum(m.total_episodes for m in all_metadata)
    aggr_meta.info.total_frames = sum(m.total_frames for m in all_metadata)
    aggr_meta.info.splits = {"train": f"0:{sum(m.total_episodes for m in all_metadata)}"}
    write_info(aggr_meta.info, aggr_meta.root)

    logger.info("write stats")
    aggr_meta.stats = aggregate_stats([m.stats for m in all_metadata])
    write_stats(aggr_meta.stats, aggr_meta.root)
