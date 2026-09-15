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
import contextlib
import logging
from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
import packaging.version
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download, sync_bucket
from huggingface_hub.utils import WeakFileLock

from lerobot.configs import DEPTH_METER_UNIT, VideoEncoderConfig, is_depth_map
from lerobot.utils.constants import DEFAULT_FEATURES, HF_LEROBOT_HOME, HF_LEROBOT_HUB_CACHE
from lerobot.utils.feature_utils import _validate_feature_names
from lerobot.utils.utils import flatten_dict

from .compute_stats import aggregate_stats
from .depth_utils import MM_PER_METRE
from .feature_utils import canonicalize_depth_marker, create_empty_dataset_info
from .io_utils import (
    get_file_size_in_mb,
    load_episodes,
    load_info,
    load_stats,
    load_tasks,
    write_info,
    write_stats,
    write_tasks,
)
from .language import DEFAULT_TOOLS, LANGUAGE_COLUMNS
from .storage import DEFAULT_STORAGE_FORMAT
from .utils import (
    DEFAULT_EPISODES_PATH,
    check_version_compatibility,
    get_safe_version,
    has_legacy_hub_download_metadata,
    is_valid_version,
    update_chunk_file_indices,
)
from .video_utils import get_video_info

CODEBASE_VERSION = "v3.0"


class LeRobotDatasetMetadata:
    """LeRobot 数据集的元数据容器。

    管理描述数据集结构、内容和统计信息的
    ``info.json``、``stats.json``、``tasks.parquet`` 以及
    ``episodes/`` parquet 文件。
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        metadata_buffer_size: int = 10,
        *,
        repo_type: Literal["dataset", "bucket"] = "dataset",
        token: str | bool | None = None,
    ):
        """加载或下载现有 LeRobot 数据集的元数据。

        尝试从本地磁盘加载元数据。如果文件缺失或
        ``force_cache_sync`` 为 ``True``，则从 Hub 下载 ``meta/`` 目录。

        Args:
            repo_id: 仓库标识（例如 ``'lerobot/aloha_sim'``）。
            root: 数据集的本地目录。提供时，Hub 下载的内容
                会直接写入此目录。省略时，现有的本地数据集
                仍在 ``$HF_LEROBOT_HOME/{repo_id}`` 下查找，
                但 Hub 下载使用位于 ``$HF_LEROBOT_HOME/hub``
                下的版本安全快照缓存。
            revision: Git 修订（分支、标签或提交哈希）。默认为
                当前代码库版本。
            force_cache_sync: 如果为 ``True``，即使本地文件存在
                也会从 Hub 重新下载元数据。
            metadata_buffer_size: 刷新到 parquet 之前在内存中缓冲的
                episode 元数据记录数。
            repo_type: 仓库类型："dataset"（默认）或 "bucket"，后者表示
                通过 hf://buckets/ 流式传输的 HF 存储桶。
            token: 用于 Hub 请求的认证令牌。可传入字符串令牌，
                传入 ``True`` 表示要求使用本地存储的令牌，``False``
                表示禁用认证，``None`` 表示使用 Hugging Face Hub
                的默认行为。
        """
        if repo_type not in ("dataset", "bucket"):
            raise ValueError(f"repo_type must be 'dataset' or 'bucket', got {repo_type!r}")

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision if revision else CODEBASE_VERSION
        self._requested_root = Path(root) if root is not None else None
        if self._requested_root is not None:
            self.root = self._requested_root
        elif self.repo_type == "bucket":
            self.root = HF_LEROBOT_HUB_CACHE / ("buckets--" + self.repo_id.replace("/", "--"))
        else:
            self.root = HF_LEROBOT_HOME / repo_id
        self._pq_writer = None
        self.latest_episode = None
        self._metadata_buffer: list[dict] = []
        self._metadata_buffer_size = metadata_buffer_size
        self._finalized = False

        metadata_lock = contextlib.nullcontext()
        if self.repo_type == "bucket":
            self.root.parent.mkdir(parents=True, exist_ok=True)
            metadata_lock = WeakFileLock(self.root.parent / f".{self.root.name}.lock")

        with metadata_lock:
            try:
                if force_cache_sync or (
                    self._requested_root is None and has_legacy_hub_download_metadata(self.root)
                ):
                    raise FileNotFoundError
                self._load_metadata()
            except (FileNotFoundError, NotADirectoryError):
                if self.repo_type != "bucket" and is_valid_version(self.revision):
                    if token is None:
                        self.revision = get_safe_version(self.repo_id, self.revision)
                    else:
                        self.revision = get_safe_version(self.repo_id, self.revision, token=token)

                self._pull_from_repo(allow_patterns="meta/", token=token)
                self._load_metadata()

    def _flush_metadata_buffer(self) -> None:
        """将所有缓冲的 episode 元数据写入 parquet 文件。"""
        if not hasattr(self, "_metadata_buffer") or len(self._metadata_buffer) == 0:
            return

        combined_dict = {}
        for episode_dict in self._metadata_buffer:
            for key, value in episode_dict.items():
                if key not in combined_dict:
                    combined_dict[key] = []
                # 提取值并序列化 numpy 数组，
                # 因为 PyArrow 的 from_pydict 函数不支持 numpy 数组
                val = value[0] if isinstance(value, list) else value
                combined_dict[key].append(val.tolist() if isinstance(val, np.ndarray) else val)

        first_ep = self._metadata_buffer[0]
        chunk_idx = first_ep["meta/episodes/chunk_index"][0]
        file_idx = first_ep["meta/episodes/file_index"][0]

        table = pa.Table.from_pydict(combined_dict)

        if not self._pq_writer:
            path = Path(self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx))
            path.parent.mkdir(parents=True, exist_ok=True)
            self._pq_writer = pq.ParquetWriter(
                path, schema=table.schema, compression="snappy", use_dictionary=True
            )
        else:
            # `combined_dict` 中的列顺序遵循源 episode 字典的插入顺序，
            # 不同批次之间可能不同（例如原本存储在不同 parquet 分片中、
            # 列顺序不同的 episode）。重新对齐到写入器已确立的模式，
            # 以免 `write_table` 因列顺序不同而拒绝写入。
            table = table.select(self._pq_writer.schema.names)

        self._pq_writer.write_table(table)

        self.latest_episode = self._metadata_buffer[-1]
        self._metadata_buffer.clear()

    def _close_writer(self) -> None:
        """关闭并清理 parquet 写入器（如果存在）。"""
        self._flush_metadata_buffer()

        writer = getattr(self, "_pq_writer", None)
        if writer is not None:
            writer.close()
            self._pq_writer = None

    def finalize(self) -> None:
        """刷新元数据缓冲区并关闭 parquet 写入器。

        幂等——可以安全地多次调用。
        """
        if getattr(self, "_finalized", False):
            return
        self._close_writer()
        self._finalized = True

    def __del__(self):
        """安全网：在垃圾回收时刷新并关闭 parquet 写入器。"""
        # 在解释器关闭期间，被引用的对象可能已经被回收。
        with contextlib.suppress(Exception):
            self.finalize()

    def _load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks = load_tasks(self.root) if self.total_tasks > 0 else None
        self.episodes = load_episodes(self.root) if self.total_episodes > 0 else None
        self.stats = load_stats(self.root)

    def ensure_readable(self) -> None:
        """确保元数据已完全加载以进行读取操作。

        幂等——当元数据已在内存中时，这只是一次
        ``is None`` 检查。在同一实例从写入模式
        切换到读取模式之前调用此方法。
        """
        if self.episodes is None:
            self._load_metadata()

    def filter_episodes(
        self,
        predicate: Callable[[dict], bool],
        candidates: list[int] | None = None,
    ) -> list[int]:
        """筛选元数据满足给定谓词的 episode。

        Args:
            predicate: 作用于每个 episode 元数据行、用于选择 episode 的谓词。
            candidates: 可选的 episode 索引列表，用于限制求值范围。

        Returns:
            满足谓词的已排序 episode 索引列表。
        """
        self.ensure_readable()
        if candidates is not None:
            candidate_set = set(candidates)
            combined = lambda ep: ep["episode_index"] in candidate_set and predicate(ep)  # noqa: E731
        else:
            combined = predicate
        filtered = self.episodes.filter(combined, keep_in_memory=True, load_from_cache_file=False)
        return sorted(int(idx) for idx in filtered["episode_index"])

    def _pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        *,
        token: str | bool | None = None,
    ) -> None:
        if self.repo_type == "bucket":
            self.root.mkdir(parents=True, exist_ok=True)
            sync_bucket(
                f"hf://buckets/{self.repo_id}/meta",
                str(self.root / "meta"),
                delete=True,
                quiet=True,
                token=token,
            )
            return
        token_kwargs = {} if token is None else {"token": token}
        if self._requested_root is None:
            self.root = Path(
                snapshot_download(
                    self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    cache_dir=HF_LEROBOT_HUB_CACHE,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    **token_kwargs,
                )
            )
            return

        self._requested_root.mkdir(exist_ok=True, parents=True)
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self._requested_root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            **token_kwargs,
        )
        self.root = self._requested_root

    @property
    def url_root(self) -> str:
        """此数据集的 Hugging Face Hub URL 根。"""
        if self.repo_type == "bucket":
            return f"hf://buckets/{self.repo_id}"
        return f"hf://datasets/{self.repo_id}"

    @property
    def _version(self) -> packaging.version.Version:
        """创建此数据集所用的代码库版本。"""
        return packaging.version.parse(self.info.codebase_version)

    def get_data_file_path(self, ep_index: int) -> Path:
        """返回给定 episode 索引对应的相对 parquet 文件路径。

        Args:
            ep_index: 从零开始的 episode 索引。

        Returns:
            包含此 episode 数据的 parquet 文件路径。

        Raises:
            IndexError: 如果 ``ep_index`` 超出范围。
        """
        if self.episodes is None:
            self.episodes = load_episodes(self.root)
        if ep_index >= len(self.episodes):
            raise IndexError(
                f"Episode index {ep_index} out of range. Episodes: {len(self.episodes) if self.episodes else 0}"
            )
        ep = self.episodes[ep_index]
        chunk_idx = ep["data/chunk_index"]
        file_idx = ep["data/file_index"]
        fpath = self.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        """返回给定 episode 和视频键对应的相对视频文件路径。

        Args:
            ep_index: 从零开始的 episode 索引。
            vid_key: 标识视频流的特征键
                （例如 ``'observation.images.laptop'``）。

        Returns:
            包含此 episode 帧的视频文件路径。

        Raises:
            IndexError: 如果 ``ep_index`` 超出范围。
        """
        if self.episodes is None:
            self.episodes = load_episodes(self.root)
        if ep_index >= len(self.episodes):
            raise IndexError(
                f"Episode index {ep_index} out of range. Episodes: {len(self.episodes) if self.episodes else 0}"
            )
        ep = self.episodes[ep_index]
        chunk_idx = ep[f"videos/{vid_key}/chunk_index"]
        file_idx = ep[f"videos/{vid_key}/file_index"]
        fpath = self.video_path.format(video_key=vid_key, chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    @property
    def storage_format(self) -> str:
        """保存底层数据文件的格式（默认为 ``"lerobot"``）。"""
        return self.info.storage_format or DEFAULT_STORAGE_FORMAT

    @property
    def data_path(self) -> str:
        """用于 parquet 文件的可格式化字符串。"""
        return self.info.data_path

    @property
    def video_path(self) -> str | None:
        """用于视频文件的可格式化字符串。"""
        return self.info.video_path

    @property
    def robot_type(self) -> str | None:
        """录制此数据集时使用的机器人类型。"""
        return self.info.robot_type

    @property
    def fps(self) -> int:
        """数据采集期间使用的帧率（每秒帧数）。"""
        return self.info.fps

    @property
    def features(self) -> dict[str, dict]:
        """数据集中包含的所有特征。"""
        return self.info.features

    @property
    def image_keys(self) -> list[str]:
        """访问以图像形式存储的视觉模态的键。"""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """访问以视频形式存储的视觉模态的键。"""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def depth_keys(self) -> list[str]:
        """访问以视频或图像形式存储的深度图模态的键。

        深度键是其 ``info`` 字典带有 ``"is_depth_map": True`` 的特征
        （或 ``info`` 或 ``video_info`` 中遗留的 ``"video.is_depth_map"``）。
        """

        return [key for key, ft in self.features.items() if is_depth_map(ft)]

    def rescale_depth_stats(self, output_unit: str) -> None:
        """就地将深度特征统计量从其记录单位重新缩放到 ``output_unit``。

        深度统计量以帧被记录时的单位存储
        （``features[key]["info"]["depth_unit"]``），而帧在读取时以
        ``output_unit`` 返回。本方法会转换带单位的统计条目，
        使统计量与消费者看到的帧保持一致。
        """
        missing_unit_keys = [
            key for key in self.depth_keys if (self.features[key].get("info") or {}).get("depth_unit") is None
        ]
        if missing_unit_keys:
            logging.warning(
                f"Depth feature(s) {missing_unit_keys} have no recorded 'depth_unit' in their info. "
                f"Depth maps and stats for these keys will be returned AS IS, with no unit conversion "
                f"to the requested output unit {output_unit!r}. Re-record the dataset or set 'depth_unit' "
                f"in the feature info (meta/info.json) to enable conversion."
            )
        if self.stats is None:
            return
        for key in self.depth_keys:
            stored_unit = (self.features[key].get("info") or {}).get("depth_unit")
            if stored_unit is None or stored_unit == output_unit or key not in self.stats:
                continue
            factor = MM_PER_METRE if stored_unit == DEPTH_METER_UNIT else 1.0 / MM_PER_METRE
            self.stats[key] = {
                stat: value if stat == "count" else value * factor for stat, value in self.stats[key].items()
            }

    @property
    def camera_keys(self) -> list[str]:
        """访问视觉模态的键（无论其存储方式如何）。"""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def has_language_columns(self) -> bool:
        """如果数据集声明了任何语言列则返回 ``True``。

        用于门控语言感知的代码路径（collate、渲染步骤），
        使未标注的数据集保持 PyTorch 的默认 collate 行为。
        """
        return any(col in self.features for col in LANGUAGE_COLUMNS)

    @property
    def tools(self) -> list[dict]:
        """此数据集声明的 OpenAI 风格工具 schema。

        从 ``meta/info.json["tools"]`` 读取。返回副本，
        因此调用者可以安全地修改结果。当数据集未声明任何工具时，
        回退到 :data:`lerobot.datasets.language.DEFAULT_TOOLS`
        （规范的 ``say`` schema）——这样未标注的数据集和
        chat-template 消费者（``apply_chat_template(messages, tools=meta.tools)``）
        都能开箱即用地正常工作。

        实现位于 :mod:`lerobot.tools` 下（每个工具一个文件）；
        编写指南见 ``docs/source/tools.mdx``。
        """
        declared = self.info.tools
        if declared:
            return [dict(t) for t in declared]
        return [dict(t) for t in DEFAULT_TOOLS]

    @tools.setter
    def tools(self, value: list[dict] | None) -> None:
        """将工具目录持久化到 ``meta/info.json`` 并重新加载元数据。

        将 ``value`` 写入磁盘上的 ``info.json``（当 ``value`` 为
        ``None`` 或为空时清除 ``tools`` 键），然后重新加载
        ``self.info``，使内存中的元数据与磁盘上的内容保持一致。
        省去了调用者手动编辑 ``info.json`` 并重新实例化
        元数据对象的麻烦。
        """
        self.info.tools = [dict(t) for t in value] if value else None
        write_info(self.info, self.root)
        self.info = load_info(self.root)

    @property
    def names(self) -> dict[str, list | dict]:
        """向量模态各维度的名称。"""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """不同特征的形状。"""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """可用的 episode 总数。"""
        return self.info.total_episodes

    @property
    def total_frames(self) -> int:
        """此数据集中保存的帧总数。"""
        return self.info.total_frames

    @property
    def total_tasks(self) -> int:
        """此数据集中执行的不同任务总数。"""
        return self.info.total_tasks

    @property
    def chunks_size(self) -> int:
        """每个 chunk 目录中的最大文件数。"""
        return self.info.chunks_size

    @property
    def data_files_size_in_mb(self) -> int:
        """数据文件的最大大小（兆字节）。"""
        return self.info.data_files_size_in_mb

    @property
    def video_files_size_in_mb(self) -> int:
        """视频文件的最大大小（兆字节）。"""
        return self.info.video_files_size_in_mb

    def get_task_index(self, task: str) -> int | None:
        """
        给定一个自然语言任务，如果该任务已存在于数据集中则返回其 task_index，
        否则返回 None。
        """
        if task in self.tasks.index:
            return int(self.tasks.loc[task].task_index)
        else:
            return None

    def save_episode_tasks(self, tasks: list[str]):
        """为当前 episode 注册任务并持久化到磁盘。

        数据集中尚不存在的新任务会被分配连续的任务索引，
        并追加到 tasks parquet 文件中。

        Args:
            tasks: 自然语言描述的唯一任务列表。

        Raises:
            ValueError: 如果 ``tasks`` 包含重复项。
        """
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"Tasks are not unique: {tasks}")

        if self.tasks is None:
            new_tasks = tasks
            task_indices = range(len(tasks))
            self.tasks = pd.DataFrame({"task_index": task_indices}, index=pd.Index(tasks, name="task"))
        else:
            new_tasks = [task for task in tasks if task not in self.tasks.index]
            new_task_indices = range(len(self.tasks), len(self.tasks) + len(new_tasks))
            for task_idx, task in zip(new_task_indices, new_tasks, strict=False):
                self.tasks.loc[task] = task_idx

        if len(new_tasks) > 0:
            # 更新到磁盘
            write_tasks(self.tasks, self.root)

    def _save_episode_metadata(self, episode_dict: dict) -> None:
        """缓冲 episode 元数据并批量写入 parquet 以提高效率。

        本函数将 episode 元数据累积在缓冲区中，并在缓冲区达到
        配置的大小时刷新。这通过一次写入多个 episode 而不是
        一次只写一行来减少 I/O 开销。

        说明：我们既需要更新 parquet 文件，也需要更新 HF 数据集：
        - ``pandas`` 将 parquet 文件加载到内存中
        - ``datasets`` 依赖 pyarrow 的内存映射（不占内存）。它要么将 parquet 文件转换为磁盘上的 pyarrow 缓存，
          要么直接从 pyarrow 缓存加载。
        """
        # 将每个值转换为列表格式
        episode_dict = {key: [value] for key, value in episode_dict.items()}
        num_frames = episode_dict["length"][0]

        if self.latest_episode is None:
            # 为由第一个 episode 数据组成的新数据集初始化索引和帧数
            chunk_idx, file_idx = 0, 0
            if self.episodes is not None and len(self.episodes) > 0:
                # 这意味着我们正在恢复录制，因此需要加载最新的 episode
                # 更新索引以避免覆盖最新的 episode
                chunk_idx = self.episodes[-1]["meta/episodes/chunk_index"]
                file_idx = self.episodes[-1]["meta/episodes/file_index"]
                latest_num_frames = self.episodes[-1]["dataset_to_index"]
                episode_dict["dataset_from_index"] = [latest_num_frames]
                episode_dict["dataset_to_index"] = [latest_num_frames + num_frames]

                # 恢复录制时，移动到下一个文件
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.chunks_size)
            else:
                episode_dict["dataset_from_index"] = [0]
                episode_dict["dataset_to_index"] = [num_frames]

            episode_dict["meta/episodes/chunk_index"] = [chunk_idx]
            episode_dict["meta/episodes/file_index"] = [file_idx]
        else:
            chunk_idx = self.latest_episode["meta/episodes/chunk_index"][0]
            file_idx = self.latest_episode["meta/episodes/file_index"][0]

            latest_path = (
                self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
                if self._pq_writer is None
                else self._pq_writer.where
            )

            if Path(latest_path).exists():
                latest_size_in_mb = get_file_size_in_mb(Path(latest_path))
                latest_num_frames = self.latest_episode["episode_index"][0]

                av_size_per_frame = latest_size_in_mb / latest_num_frames if latest_num_frames > 0 else 0.0

                if latest_size_in_mb + av_size_per_frame * num_frames >= self.data_files_size_in_mb:
                    # 达到大小限制，刷新缓冲区并准备新的 parquet 文件
                    self._flush_metadata_buffer()
                    chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.chunks_size)
                    self._close_writer()

            # 用新行更新现有的 pandas 数据帧
            episode_dict["meta/episodes/chunk_index"] = [chunk_idx]
            episode_dict["meta/episodes/file_index"] = [file_idx]
            episode_dict["dataset_from_index"] = [self.latest_episode["dataset_to_index"][0]]
            episode_dict["dataset_to_index"] = [self.latest_episode["dataset_to_index"][0] + num_frames]

        # 加入缓冲区
        self._metadata_buffer.append(episode_dict)
        self.latest_episode = episode_dict

        if len(self._metadata_buffer) >= self._metadata_buffer_size:
            self._flush_metadata_buffer()

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        episode_metadata: dict,
    ) -> None:
        """持久化 episode 元数据，更新数据集信息，并聚合统计量。

        将 episode 的元数据写入缓冲的 parquet 写入器，递增
        ``info.json`` 中的 episode/帧总数计数器，并将该 episode 的
        统计量合并到正在运行的数据集统计量中。

        Args:
            episode_index: 正在保存的 episode 的从零开始的索引。
            episode_length: 此 episode 中的帧数。
            episode_tasks: 此 episode 的任务描述列表。
            episode_stats: 此 episode 中每个特征的统计量。
            episode_metadata: 附加元数据（chunk/file 索引、帧
                范围、视频时间戳等）。
        """
        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        episode_dict.update(episode_metadata)
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        self._save_episode_metadata(episode_dict)

        # 更新 info
        self.info.total_episodes += 1
        self.info.total_frames += episode_length
        self.info.total_tasks = len(self.tasks)
        self.info.splits = {"train": f"0:{self.info.total_episodes}"}

        write_info(self.info, self.root)

        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats is not None else episode_stats
        write_stats(self.stats, self.root)

    def update_video_info(
        self,
        video_key: str | None = None,
        video_encoder: VideoEncoderConfig | None = None,
        preserve_keys: Iterable[str] | None = None,
    ) -> None:
        """填充或刷新 ``info.json`` 中每个特征的视频信息。

        警告：本函数从第一个 episode 的视频写入信息，隐含假设所有视频
        都以相同方式编码。同时这意味着它假设第一个 episode 存在。

        始终重新探测视频，并覆盖每个重新计算的键的现有信息。
        ``preserve_keys`` 列出必须保留其现有值的键（例如
        数据内在的条目如 ``is_depth_map`` 和深度量化参数），
        而不是重新计算。

        Args:
            video_key: 若提供，则只更新此视频键。否则更新
                数据集中的所有视频键。
            video_encoder: 用于生成视频的编码器配置。提供时，
                其字段会与从流派生的 ``video.*`` 条目一起记录为
                ``video.<field>`` 条目（参见 :func:`get_video_info`）。
            preserve_keys: 其现有值被保留而非重新计算的键。
                ``None``（默认）会重新计算每个键。
        """
        if video_key is not None and video_key not in self.video_keys:
            raise ValueError(f"Video key {video_key} not found in dataset")

        video_keys = [video_key] if video_key is not None else self.video_keys
        preserve_set = set(preserve_keys or ())
        for key in video_keys:
            feature = self.info.features[key]
            existing = feature.get("info") or {}
            video_path = self.root / self.video_path.format(video_key=key, chunk_index=0, file_index=0)
            new_info = get_video_info(video_path, video_encoder=video_encoder)
            # 丢弃要保留的键，以便合并时现有值胜出。
            new_info = {k: v for k, v in new_info.items() if k not in preserve_set}
            feature["info"] = {**existing, **new_info}
            # 将任何遗留的深度标记（在 ``info`` 或单独的 ``video_info`` 字典中）
            # 迁移到规范的 ``is_depth_map`` 键。
            video_info = feature.get("video_info")
            had_legacy = "video.is_depth_map" in feature["info"] or (
                isinstance(video_info, dict) and "video.is_depth_map" in video_info
            )
            canonicalize_depth_marker(feature)
            if had_legacy:
                logging.warning(f"Migrated legacy depth marker to 'is_depth_map' for feature {key!r}.")

    def update_chunk_settings(
        self,
        chunks_size: int | None = None,
        data_files_size_in_mb: int | None = None,
        video_files_size_in_mb: int | None = None,
    ) -> None:
        """在数据集创建后更新 chunk 和文件大小设置。

        这允许用户在不修改构造函数的情况下自定义存储组织。
        这些设置控制 episode 如何分块，以及文件在创建新文件
        之前可以增长到多大。

        Args:
            chunks_size: 每个 chunk 目录中的最大文件数。若为 None，则保持当前值。
            data_files_size_in_mb: 数据 parquet 文件的最大大小（MB）。若为 None，则保持当前值。
            video_files_size_in_mb: 视频文件的最大大小（MB）。若为 None，则保持当前值。
        """
        if chunks_size is not None:
            if chunks_size <= 0:
                raise ValueError(f"chunks_size must be positive, got {chunks_size}")
            self.info.chunks_size = chunks_size

        if data_files_size_in_mb is not None:
            if data_files_size_in_mb <= 0:
                raise ValueError(f"data_files_size_in_mb must be positive, got {data_files_size_in_mb}")
            self.info.data_files_size_in_mb = data_files_size_in_mb

        if video_files_size_in_mb is not None:
            if video_files_size_in_mb <= 0:
                raise ValueError(f"video_files_size_in_mb must be positive, got {video_files_size_in_mb}")
            self.info.video_files_size_in_mb = video_files_size_in_mb

        # 更新磁盘上的 info 文件
        write_info(self.info, self.root)

    def get_chunk_settings(self) -> dict[str, int]:
        """获取当前的 chunk 和文件大小设置。

        Returns:
            包含 chunks_size、data_files_size_in_mb 和 video_files_size_in_mb 的字典。
        """
        return {
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": self.data_files_size_in_mb,
            "video_files_size_in_mb": self.video_files_size_in_mb,
        }

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
        metadata_buffer_size: int = 10,
        chunks_size: int | None = None,
        data_files_size_in_mb: int | None = None,
        video_files_size_in_mb: int | None = None,
    ) -> "LeRobotDatasetMetadata":
        """从零开始为新的 LeRobot 数据集创建元数据。

        使用提供的特征 schema 和数据集设置初始化磁盘上的
        ``info.json`` 文件。尚未写入任何 episode 数据。

        Args:
            repo_id: 仓库标识（例如 ``'user/my_dataset'``）。
            fps: 数据采集期间使用的帧率。
            features: 特征规范字典，将特征名称映射到其类型/形状元数据。
            robot_type: 可选的存储在元数据中的机器人类型字符串。
            root: 数据集的本地目录。默认为
                ``$HF_LEROBOT_HOME/{repo_id}``。必须不存在。
            use_videos: 若为 ``True``，视觉模态编码为 MP4 视频。
            metadata_buffer_size: 刷新到 parquet 之前在内存中缓冲的
                episode 元数据记录数。
            chunks_size: 每个 chunk 目录中的最大文件数。``None`` 使用
                默认值。
            data_files_size_in_mb: parquet 文件的最大大小（MB）。``None`` 使用
                默认值。
            video_files_size_in_mb: 视频文件的最大大小（MB）。``None`` 使用
                默认值。

        Returns:
            一个新的 :class:`LeRobotDatasetMetadata` 实例。
        """
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj._requested_root = Path(root) if root is not None else None
        obj.root = obj._requested_root if obj._requested_root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        features = {**deepcopy(features), **DEFAULT_FEATURES}
        _validate_feature_names(features)

        obj.tasks = None
        obj.episodes = None
        obj.stats = None
        obj.info = create_empty_dataset_info(
            CODEBASE_VERSION,
            fps,
            features,
            use_videos,
            robot_type,
            chunks_size,
            data_files_size_in_mb,
            video_files_size_in_mb,
        )
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError(
                f"Features contain video keys {obj.video_keys}, but 'use_videos' is set to False. "
                "Either remove video features from the features dict, or set 'use_videos=True'."
            )
        write_info(obj.info, obj.root)
        obj.revision = None
        obj._pq_writer = None
        obj.latest_episode = None
        obj._metadata_buffer = []
        obj._metadata_buffer_size = metadata_buffer_size
        obj._finalized = False
        return obj
