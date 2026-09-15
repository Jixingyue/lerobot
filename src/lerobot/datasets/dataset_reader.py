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
"""LeRobotDataset 的私有读取器组件。负责随机访问读取（HF 数据集、delta 索引、视频解码）。"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets
import torch

from lerobot.configs import (
    DEFAULT_DEPTH_UNIT,
    DEPTH_METER_UNIT,
    DepthEncoderConfig,
)

from .dataset_metadata import LeRobotDatasetMetadata
from .depth_utils import MM_PER_METRE, dequantize_depth
from .feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from .io_utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)
from .utils import resolve_episode_indices
from .video_utils import decode_video_frames


class BaseDatasetReader(ABC):
    """:class:`LeRobotDataset` 的读取侧数据访问契约。

    读取器负责某种存储格式的行获取与视频解码，并返回完整组装好的
    帧字典——表格特征、delta 时间戳窗口、填充掩码、解码后的视频帧——
    从而使每种格式都产出相同的条目。``LeRobotDataset`` 将
    ``__getitem__`` 和 ``__getitems__`` 委托给它，并保留其余部分
    （元数据、episode 选择、公共 API）。子类定义各自的构造函数
    （它们的输入合理地有所不同），并且必须可 pickle，以便
    ``DataLoader`` 工作进程能重新打开各自的连接。

    子类必须在构造期间设置 :attr:`episodes`（所选的 episode 索引，
    或 ``None`` 表示全部）。
    """

    episodes: list[int] | None

    @property
    @abstractmethod
    def num_frames(self) -> int:
        """所选 episode 中的帧数。"""

    @property
    @abstractmethod
    def num_episodes(self) -> int:
        """所选的 episode 数量。"""

    @property
    @abstractmethod
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        """从绝对帧索引到相对行位置的映射。

        仅对于经过 episode 过滤的数据集为非 None，此时（来自元数据的）
        绝对索引与过滤后视图中的位置不同。
        """

    @abstractmethod
    def get_item(self, idx: int) -> dict:
        """为一个相对索引返回一个完整组装好的帧字典。"""

    def get_items(self, indices: list[int]) -> list[dict]:
        """为一批相对索引返回帧字典。

        子类可以用批量实现覆盖此方法。
        """
        return [self.get_item(idx) for idx in indices]

    def __len__(self) -> int:
        return self.num_frames

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """替换应用于视觉观测的变换。"""
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self._image_transforms = image_transforms

    def clear_image_transforms(self) -> None:
        """移除应用于视觉观测的变换。"""
        self._image_transforms = None


class DatasetReader(BaseDatasetReader):
    """服务于 parquet/mp4 存储格式的默认读取器。

    拥有：hf_dataset、_absolute_to_relative_idx、delta_indices。
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        return_uint8: bool = False,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
    ):
        """使用元数据、过滤和变换配置初始化读取器。

        HF 数据集不会在此处加载——之后请调用 :meth:`try_load` 或
        :meth:`load_and_activate`。

        Args:
            meta: 数据集元数据实例。
            root: 本地数据集根目录。
            episodes: 要选择的 episode 索引的可选列表。``None``
                表示所有 episode。
            tolerance_s: 时间戳同步容差（秒）。
            video_backend: 视频解码后端标识符。
            delta_timestamps: 可选字典，将特征键映射到用于时间上下文窗口的
                相对时间戳偏移列表。
            image_transforms: 应用于视觉特征的可选 torchvision v2 变换。
            return_uint8: 如果为 True，则返回原始 uint8 张量形式的
                RGB 视频帧，而非归一化的 float32。
            depth_output_unit: 深度图反量化后的物理单位
                （``"m"`` 或 ``"mm"``）。默认为 ``"mm"``。
        """
        self._meta = meta
        self.root = root
        self.episodes = resolve_episode_indices(episodes, meta.total_episodes)
        self._tolerance_s = tolerance_s
        self._video_backend = video_backend
        self.set_image_transforms(image_transforms)
        self._return_uint8 = return_uint8
        self._depth_output_unit = depth_output_unit

        self.hf_dataset: datasets.Dataset | None = None
        self._absolute_to_relative_idx: dict[int, int] | None = None
        self._column_views: dict[str, datasets.Dataset] = {}
        self._column_views_source: datasets.Dataset | None = None
        self._column_views_transform: Callable | None = None

        # 设置 delta_indices（不依赖 hf_dataset）
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)

        self._depth_encoder_configs: dict[str, DepthEncoderConfig] = {
            vid_key: DepthEncoderConfig.from_video_info(self._meta.features[vid_key].get("info"))
            for vid_key in self._meta.depth_keys
        }

        # 获取以原始图像形式存储的每个深度特征的输入单位。
        self._image_depth_units: dict[str, str | None] = {
            key: (self._meta.features[key].get("info") or {}).get("depth_unit")
            for key in self._meta.depth_keys
            if key in self._meta.image_keys
        }

    def try_load(self) -> bool:
        """尝试从本地缓存加载。如果数据足够则返回 True。"""
        try:
            self.hf_dataset = self._load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError):
            self.hf_dataset = None
            return False
        if not self._check_cached_episodes_sufficient():
            self.hf_dataset = None
            return False
        self._build_index_mapping()
        return True

    def load_and_activate(self) -> None:
        """从磁盘加载 HF 数据集并构建索引映射。在数据已在磁盘上之后调用。"""
        self.hf_dataset = self._load_hf_dataset()
        self._build_index_mapping()

    def _build_index_mapping(self) -> None:
        """从已加载的 hf_dataset 构建绝对索引到相对索引的映射。"""
        self._absolute_to_relative_idx = None
        if self.episodes is not None and self.hf_dataset is not None:
            indices = self.hf_dataset.data.column("index").to_numpy()
            self._absolute_to_relative_idx = dict(zip(indices.tolist(), range(len(indices)), strict=True))

    @property
    def num_frames(self) -> int:
        """所选 episode 中的帧数。"""
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self._meta.total_frames

    @property
    def num_episodes(self) -> int:
        """所选的 episode 数量。"""
        return len(self.episodes) if self.episodes is not None else self._meta.total_episodes

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        """从绝对帧索引到 HF 数据集行位置的映射。"""
        if self.hf_dataset is None:
            self.load_and_activate()
        return self._absolute_to_relative_idx

    def _load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset 包含所有的观测、状态、动作、奖励等。"""
        features = get_hf_features_from_features(self._meta.features)
        self._validate_language_columns_declared(features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _validate_language_columns_declared(self, features: datasets.Features) -> None:
        """要求存储在 Parquet 中的语言列必须在元数据中声明。"""
        # 让空数据集通过正常的加载路径失败。
        try:
            sample = next((self.root / "data").glob("*/*.parquet"))
        except StopIteration:
            return

        from pyarrow import parquet as _pq  # noqa: PLC0415

        # LeRobot 分片的模式是统一的，因此一个模式即可代表整个数据集。
        schema_names = set(_pq.read_schema(sample).names)
        from .language import LANGUAGE_COLUMNS  # noqa: PLC0415

        missing = sorted(set(LANGUAGE_COLUMNS) & schema_names - set(features))
        if missing:
            raise ValueError(
                f"Dataset Parquet files contain language feature(s) missing from metadata: {missing}. "
                "Metadata must describe the stored data; add the entries returned by "
                "lerobot.datasets.language.language_feature_info() to meta/info.json['features'] "
                "or rerun the annotation pipeline's metadata synchronization."
            )

    def _check_cached_episodes_sufficient(self) -> bool:
        """检查缓存的数据集是否包含所有请求的 episode 及其视频文件。"""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        if self.episodes is None:
            requested_episodes = set(range(self._meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        if not requested_episodes.issubset(available_episodes):
            return False

        if len(self._meta.video_keys) > 0:
            for ep_idx in requested_episodes:
                for vid_key in self._meta.video_keys:
                    video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def get_episodes_file_paths(self) -> list[Path]:
        """返回所选 episode 的去重文件路径（数据 + 视频）。

        用于为 ``snapshot_download`` 构建 ``allow_patterns`` 列表。
        """
        episodes = self.episodes if self.episodes is not None else list(range(self._meta.total_episodes))
        fpaths = [str(self._meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self._meta.video_keys) > 0:
            video_files = [
                str(self._meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self._meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episode 存储在同一文件中，因此仅返回唯一路径
        fpaths = list(set(fpaths))
        return fpaths

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """计算 delta 时间戳的查询索引。"""
        ep = self._meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self._meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = self._column_view("timestamp")[relative_indices]["timestamp"]
                else:
                    timestamps = self._column_view("timestamp")[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _column_view(self, key: str) -> datasets.Dataset:
        """返回 ``hf_dataset`` 的缓存单列视图。

        ``select_columns`` 是零拷贝的模式投影：对该视图的行查询只会
        获取并解码 ``key``。相比之下，``hf_dataset[indices]``
        （以及在 ``datasets`` >= 4.4 中，由于自定义变换禁用了惰性
        ``Column`` 快速路径，``hf_dataset[key][indices]`` 也是如此）
        会获取并解码整行。在图像数据集上，仅仅为了读取像 ``action``
        这样的低维列，也会解码每个被查询行中嵌入的所有相机图像
        （#2895）。该视图保留了 ``hf_transform_to_torch`` 变换，
        而该变换是按列进行的，因此输出与普通的行查询完全相同。
        """
        transform = self.hf_dataset.format["format_kwargs"].get("transform")
        if self._column_views_source is not self.hf_dataset or self._column_views_transform is not transform:
            # hf_dataset 被（重新）加载，或其变换已更改：丢弃过期的视图
            self._column_views = {}
            self._column_views_source = self.hf_dataset
            self._column_views_transform = transform
        if key not in self._column_views:
            self._column_views[key] = self.hf_dataset.select_columns(key)
        return self._column_views[key]

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """跨各键按索引查询数据集，跳过视频键。"""
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self._meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            result[key] = torch.stack(self._column_view(key)[relative_indices][key])
        return result

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """注意：在使用数据工作进程时（例如 num_workers>0 的 DataLoader），
        不要在主进程中调用此函数（例如使用第二个 num_workers=0 的
        DataLoader）。这会导致段错误（Segmentation Fault）。
        """
        ep = self._meta.episodes[ep_idx]

        def _decode_single(vid_key: str, query_ts: list[float]) -> tuple[str, torch.Tensor]:
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(
                video_path,
                shifted_query_ts,
                self._tolerance_s,
                self._video_backend,
                return_uint8=self._return_uint8,
                is_depth=vid_key in self._meta.depth_keys,
            )
            if vid_key in self._meta.depth_keys:
                depth_encoder = self._depth_encoder_configs[vid_key]
                frames = dequantize_depth(
                    frames,
                    depth_min=depth_encoder.depth_min,
                    depth_max=depth_encoder.depth_max,
                    shift=depth_encoder.shift,
                    use_log=depth_encoder.use_log,
                    output_unit=self._depth_output_unit,
                )
            return vid_key, frames.squeeze(0)

        items = list(query_timestamps.items())

        # 单相机：无线程开销
        if len(items) <= 1:
            return {vid_key: _decode_single(vid_key, query_ts)[1] for vid_key, query_ts in items}

        # 多相机：并行解码（视频解码会释放 GIL）
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            futures = [pool.submit(_decode_single, k, ts) for k, ts in items]
            return dict(f.result() for f in futures)

    def get_item(self, idx) -> dict:
        """核心 __getitem__ 逻辑。在首次访问时加载 hf_dataset。

        ``idx`` 是（可能经过 episode 过滤的）HF 数据集中的*相对*索引，
        **而不是**存储在 ``index`` 列中的绝对帧索引。绝对索引从行本身获取。
        """
        if self.hf_dataset is None:
            # finalize() 之后的一次性加载
            self.load_and_activate()
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self._meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self._image_transforms is not None:
            for cam in self._meta.camera_keys:
                if cam in self._meta.depth_keys:
                    continue
                item[cam] = self._image_transforms(item[cam])

        # 将深度特征转换为输出单位。
        for key, stored_unit in self._image_depth_units.items():
            if key in item and stored_unit is not None and stored_unit != self._depth_output_unit:
                item[key] = (
                    item[key] * MM_PER_METRE if stored_unit == DEPTH_METER_UNIT else item[key] / MM_PER_METRE
                )

        # 以字符串形式添加任务
        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name

        return item
