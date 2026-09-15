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
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from pathlib import Path
from typing import Literal

import datasets
import numpy as np
import torch
from datasets import load_dataset

from lerobot.configs import DEFAULT_DEPTH_UNIT, DEPTH_METER_UNIT, DepthEncoderConfig
from lerobot.utils.constants import HF_LEROBOT_HOME, LOOKAHEAD_BACKTRACKTABLE, LOOKBACK_BACKTRACKTABLE

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .depth_utils import MM_PER_METRE, dequantize_depth
from .feature_utils import get_delta_indices
from .io_utils import item_to_torch
from .utils import (
    check_version_compatibility,
    find_float_index,
    is_float_in_list,
    safe_shard,
)
from .video_utils import (
    VideoDecoderCache,
    decode_video_frames,
    decode_video_frames_torchcodec,
)


class LookBackError(Exception):
    """
    尝试在 Backtrackable 对象的历史记录中向后回看时抛出的异常。
    """

    pass


class LookAheadError(Exception):
    """
    尝试在 Backtrackable 对象的未来记录中向前预看时抛出的异常。
    """

    pass


class _ShardExhaustedError(Exception):
    """当流式数据集分片已没有更多条目时抛出。"""


class Backtrackable[T]:
    """
    包装任意迭代器/可迭代对象，使你最多可以向后回退 `history` 个
    条目，并最多向前预看 `lookahead` 个条目。

    这对流式数据集很有用：你需要访问之前和未来的条目，
    但又无法将整个数据集加载到内存中。

    示例：
    -------
    ```python
    ds = load_dataset("c4", "en", streaming=True, split="train")
    rev = Backtrackable(ds, history=3, lookahead=2)

    x0 = next(rev)  # 向前
    x1 = next(rev)
    x2 = next(rev)

    # 向前预看
    x3_peek = rev.peek_ahead(1)  # 下一个条目，不移动游标
    x4_peek = rev.peek_ahead(2)  # 向前两个条目

    # 向后回看
    x1_again = rev.peek_back(1)  # 上一个条目，不移动游标
    x0_again = rev.peek_back(2)  # 向后两个条目

    # 向后移动
    x1_back = rev.prev()  # 向后退一步
    next(rev)  # 返回 x2，从之前的位置继续向前
    ```
    """

    __slots__ = ("_source", "_back_buf", "_ahead_buf", "_cursor", "_history", "_lookahead")

    def __init__(self, iterable: Iterable[T], *, history: int = 1, lookahead: int = 0):
        if history < 1:
            raise ValueError("history must be >= 1")
        if lookahead <= 0:
            raise ValueError("lookahead must be > 0")

        self._source: Iterator[T] = iter(iterable)
        self._back_buf: deque[T] = deque(maxlen=history)
        self._ahead_buf: deque[T] = deque(maxlen=lookahead) if lookahead > 0 else deque()
        self._cursor: int = 0
        self._history = history
        self._lookahead = lookahead

    def __iter__(self) -> "Backtrackable[T]":
        return self

    def __next__(self) -> T:
        # 如果我们已经向后回退过，先从回退缓冲区取数据
        if self._cursor < 0:  # -1 表示“上一个条目”，以此类推
            self._cursor += 1
            return self._back_buf[self._cursor]

        # 如果预看缓冲区中有条目，优先使用它们
        item = self._ahead_buf.popleft() if self._ahead_buf else next(self._source)

        # 将当前条目加入回退缓冲区并重置游标
        self._back_buf.append(item)
        self._cursor = 0
        return item

    def prev(self) -> T:
        """
        在历史记录中向后回退一个条目并返回它。
        如果已经处于缓冲的最旧条目处，则抛出 IndexError。
        """
        if len(self._back_buf) + self._cursor <= 1:
            raise LookBackError("At start of history")

        self._cursor -= 1
        return self._back_buf[self._cursor]

    def peek_back(self, n: int = 1) -> T:
        """
        在不移动游标的情况下向后查看 `n` 个条目（n=1 即上一个条目）。
        """
        if n < 0 or n + 1 > len(self._back_buf) + self._cursor:
            raise LookBackError("peek_back distance out of range")

        return self._back_buf[self._cursor - (n + 1)]

    def peek_ahead(self, n: int = 1) -> T:
        """
        在不移动游标的情况下向前预看 `n` 个条目（n=1 即下一个条目）。
        必要时填充预看缓冲区。
        """
        if n < 1:
            raise LookAheadError("peek_ahead distance must be 1 or more")
        elif n > self._lookahead:
            raise LookAheadError("peek_ahead distance exceeds lookahead limit")

        # 如果条目数不够，则填充预看缓冲区
        while len(self._ahead_buf) < n:
            try:
                item = next(self._source)
                self._ahead_buf.append(item)

            except StopIteration as err:
                raise LookAheadError("peek_ahead: not enough items in source") from err

        return self._ahead_buf[n - 1]

    def history(self) -> list[T]:
        """
        返回缓冲历史记录的副本（最新的在最后）。
        列表长度 ≤ 构造时传入的 `history` 参数。
        """
        if self._cursor == 0:
            return list(self._back_buf)

        # 当 cursor<0 时进行切片，以保持时间先后顺序
        return list(self._back_buf)[: self._cursor or None]

    def can_peek_back(self, steps: int = 1) -> bool:
        """
        检查是否可以向后回退 `steps` 个条目而不抛出 IndexError。
        """
        return steps < len(self._back_buf) + self._cursor

    def can_peek_ahead(self, steps: int = 1) -> bool:
        """
        检查是否可以向前预看 `steps` 个条目。
        这可能会尝试填充预看缓冲区。
        """
        if self._lookahead > 0 and steps > self._lookahead:
            return False

        # 尝试填充预看缓冲区，以检查能否预看到那么远
        try:
            while len(self._ahead_buf) < steps:
                if self._lookahead > 0 and len(self._ahead_buf) >= self._lookahead:
                    return False
                item = next(self._source)
                self._ahead_buf.append(item)
            return True
        except StopIteration:
            return False


class StreamingLeRobotDataset(torch.utils.data.IterableDataset):
    """具备流式能力的 LeRobotDataset。

    本类扩展了 LeRobotDataset，增加了流式功能，允许以流式方式
    获取数据，而不是将数据全部加载到内存中。这对于因数据集过大
    而无法放入内存，或希望在不完整下载数据集的情况下快速浏览
    数据集时尤其有用。

    关键创新在于使用 Backtrackable 迭代器，它维护一个有界的
    近期条目缓冲区，使我们无需将整个数据集加载到内存即可
    访问用于增量时间戳的历史帧。

    示例：
        基本用法：
        ```python
        from lerobot.common.datasets.streaming_dataset import StreamingLeRobotDataset

        # 创建一个带增量时间戳的流式数据集
        delta_timestamps = {
            "observation.image": [-1.0, -0.5, 0.0],  # 1 秒前、0.5 秒前、当前
            "action": [0.0, 0.1, 0.2],  # 当前、0.1 秒后、0.2 秒后
        }

        dataset = StreamingLeRobotDataset(
            repo_id="your-dataset-repo-id",
            delta_timestamps=delta_timestamps,
            streaming=True,
            buffer_size=1000,
        )

        # 遍历数据集
        for i, item in enumerate(dataset):
            print(f"Sample {i}: Episode {item['episode_index']} Frame {item['frame_index']}")
            # item 将包含根据 delta_timestamps 堆叠的帧
            if i >= 10:
                break
        ```
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        streaming: bool = True,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        rng: np.random.Generator | None = None,
        shuffle: bool = True,
        return_uint8: bool = False,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        *,
        repo_type: Literal["dataset", "bucket"] = "dataset",
        token: str | bool | None = None,
    ):
        """初始化一个 StreamingLeRobotDataset。

        Args:
            repo_id (str): 将用于获取数据集的 repo id。
            root (Path | None, optional): 用于本地数据集的本地目录。在 bucket
                模式下，这是可选的本地元数据缓存目录；parquet 和视频数据仍保留在远端。
                省略时，Hub 元数据通过 ``$HF_LEROBOT_HOME/hub`` 下的缓存解析。
            episodes (list[int] | None, optional): 若指定，则只加载本列表中以
                episode_index 指定的 episode。
            image_transforms (Callable | None, optional): 应用于图像数据的变换。
            tolerance_s (float, optional): 时间戳匹配的容差（秒）。
            revision (str, optional): Git 版本 id（分支名、标签或提交哈希）。
            force_cache_sync (bool, optional): 优先同步并刷新本地文件的标志。
            streaming (bool, optional): 是流式获取数据集还是全部加载。默认为 True。
            buffer_size (int, optional): 流式模式下用于打乱的缓冲区大小。默认为 1000。
            max_num_shards (int, optional): 将输入数据集重新分片后的分片数量。默认为 16。
            seed (int, optional): 可复现性的随机种子。
            rng (np.random.Generator | None, optional): 随机数生成器。
            shuffle (bool, optional): 是否在多次遍历之间打乱数据集。默认为 True。
            depth_output_unit (str, optional): 深度图反量化后的物理单位（"m" 或 "mm"）。
                默认为 "mm"。
            repo_type: "dataset"（默认）或 "bucket"，表示通过
                ``hf://buckets/`` 从 HF Storage Bucket 流式获取。
            token: 从 Hub 流式获取本数据集时使用的认证令牌。
                可传入字符串令牌；``True`` 表示要求使用本地
                存储的令牌；``False`` 表示禁用认证；``None``
                表示使用 Hugging Face Hub 的默认行为。初始化之后，
                令牌不会保留在数据集实例上。
        """
        super().__init__()
        if repo_type not in ("dataset", "bucket"):
            raise ValueError(f"repo_type must be 'dataset' or 'bucket', got {repo_type!r}")

        self.repo_id = repo_id
        self.repo_type = repo_type
        self._requested_root = Path(root) if root is not None else None
        self.root = self._requested_root if self._requested_root is not None else HF_LEROBOT_HOME / repo_id
        self.streaming_from_local = root is not None and self.repo_type == "dataset"

        self.image_transforms = image_transforms
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.shuffle = shuffle

        self.streaming = streaming
        self.buffer_size = buffer_size
        self._return_uint8 = return_uint8
        self._depth_output_unit = depth_output_unit

        # 缓存视频解码器，避免每一帧都重新初始化（可避免约 10 倍的减速）
        self.video_decoder_cache = None

        if self._requested_root is not None:
            self.root.mkdir(exist_ok=True, parents=True)

        # 加载元数据
        self.meta = LeRobotDatasetMetadata(
            self.repo_id,
            self._requested_root,
            self.revision,
            force_cache_sync=force_cache_sync,
            repo_type=self.repo_type,
            token=token,
        )
        self.root = self.meta.root
        self.revision = self.meta.revision
        self.meta.rescale_depth_stats(self._depth_output_unit)
        # 检查版本
        check_version_compatibility(self.repo_id, self.meta._version, CODEBASE_VERSION)

        self._depth_encoder_configs: dict[str, DepthEncoderConfig] = {
            vid_key: DepthEncoderConfig.from_video_info(self.meta.features[vid_key].get("info"))
            for vid_key in self.meta.depth_keys
        }

        # 每个以原始图像形式存储的深度特征的输入单位（与视频分开反量化）。
        self._image_depth_units: dict[str, str | None] = {
            key: (self.meta.features[key].get("info") or {}).get("depth_unit")
            for key in self.meta.depth_keys
            if key in self.meta.image_keys
        }

        self.delta_timestamps = None
        self.delta_indices = None

        if delta_timestamps is not None:
            self._validate_delta_timestamp_keys(delta_timestamps)  # 无效时抛出 ValueError
            self.delta_timestamps = delta_timestamps
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        token_kwargs = {} if token is None else {"token": token}
        if self.repo_type == "bucket":
            self.hf_dataset: datasets.IterableDataset = load_dataset(
                "parquet",
                data_files=f"hf://buckets/{self.repo_id}/data/*/*.parquet",
                split="train",
                streaming=self.streaming,
                **token_kwargs,
            )
        else:
            if self.streaming_from_local:
                token_kwargs = {}
            self.hf_dataset: datasets.IterableDataset = load_dataset(
                self.repo_id if not self.streaming_from_local else str(self.root),
                split="train",
                streaming=self.streaming,
                data_files="data/*/*.parquet",
                revision=self.revision,
                **token_kwargs,
            )

        self.num_shards = min(self.hf_dataset.num_shards, max_num_shards)

    @property
    def num_frames(self):
        return self.meta.total_frames

    @property
    def num_episodes(self):
        return self.meta.total_episodes

    @property
    def fps(self):
        return self.meta.fps

    @property
    def depth_output_unit(self) -> str:
        """读取时深度图所使用的物理单位（``"m"`` 或 ``"mm"``）。"""
        return self._depth_output_unit

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator, buffer_size: int, random_batch_size=100
    ) -> Iterator[int]:
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    @staticmethod
    def _infinite_generator_over_elements(rng: np.random.Generator, elements: list[int]) -> Iterator[int]:
        while True:
            yield rng.choice(elements)

    # TODO(fracapuano): 实现多线程预取以加速数据加载。
    # 当前的顺序迭代是一个瓶颈。可以采用生产者-消费者模式，
    # 配合 ThreadPoolExecutor 并行运行 `make_frame`（尤其是视频解码），
    # 将处理后的条目送入队列，本迭代器再从该队列中产出条目。
    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.video_decoder_cache is None:
            self.video_decoder_cache = VideoDecoderCache()

        # 如果 shuffle 为 False，则在多次遍历之间保持相同的种子；否则在多次遍历之间打乱数据
        rng = np.random.default_rng(self.seed) if not self.shuffle else self.rng

        buffer_indices_generator = self._iter_random_indices(rng, self.buffer_size)

        idx_to_backtrack_dataset = {
            idx: self._make_backtrackable_dataset(safe_shard(self.hf_dataset, idx, self.num_shards))
            for idx in range(self.num_shards)
        }

        # 该缓冲区在遍历数据集分片时填充，
        # 其逻辑是引入两个层次的随机性：
        # (1) 从可用分片中随机抽取一个分片，
        # (2) 从第 (1) 步抽中的分片中随机抽取一帧
        frames_buffer = []
        while available_shards := list(idx_to_backtrack_dataset.keys()):
            shard_key = next(self._infinite_generator_over_elements(rng, available_shards))
            backtrack_dataset = idx_to_backtrack_dataset[shard_key]  # 选择要迭代的分片

            try:
                for frame in self.make_frame(backtrack_dataset):
                    if len(frames_buffer) == self.buffer_size:
                        i = next(buffer_indices_generator)  # 从缓冲区中抽取一个元素
                        yield frames_buffer[i]
                        frames_buffer[i] = frame
                    else:
                        frames_buffer.append(frame)
                    break  # 已随机抽过分片，切换分片
            except _ShardExhaustedError:
                del idx_to_backtrack_dataset[shard_key]  # 移除已耗尽的分片，转去另一个分片

        # 所有分片都耗尽后，打乱缓冲区并产出剩余的帧
        rng.shuffle(frames_buffer)
        yield from frames_buffer

    def _get_window_steps(
        self, delta_timestamps: dict[str, list[float]] | None = None, dynamic_bounds: bool = False
    ) -> tuple[int, int]:
        if delta_timestamps is None:
            return 1, 1

        if not dynamic_bounds:
            # 固定窗口
            lookback = LOOKBACK_BACKTRACKTABLE
            lookahead = LOOKAHEAD_BACKTRACKTABLE
        else:
            # 根据给定的 delta_timesteps 动态调整窗口
            all_timestamps = sum(delta_timestamps.values(), [])
            lookback = min(all_timestamps) * self.fps
            lookahead = max(all_timestamps) * self.fps

            # 当 lookback >= 0 时，说明没有提供负的时间步长
            lookback = 0 if lookback >= 0 else (lookback * -1)

        return lookback, lookahead

    def _make_backtrackable_dataset(self, dataset: datasets.IterableDataset) -> Backtrackable:
        lookback, lookahead = self._get_window_steps(self.delta_timestamps)
        return Backtrackable(dataset, history=lookback, lookahead=lookahead)

    def _make_timestamps_from_indices(
        self, start_ts: float, indices: dict[str, list[int]] | None = None
    ) -> dict[str, list[float]]:
        if indices is not None:
            return {
                key: (
                    start_ts + torch.tensor(indices[key]) / self.fps
                ).tolist()  # 注意：为什么不直接使用 delta_timestamps？
                for key in self.delta_timestamps
            }
        else:
            return dict.fromkeys(self.meta.video_keys, [start_ts])

    def _make_padding_camera_frame(self, camera_key: str):
        """给定相机键对应的可变形状填充帧，以 (H, W, C) 给出"""
        return torch.zeros(self.meta.info.features[camera_key]["shape"]).permute(-1, 0, 1)

    def _get_video_frame_padding_mask(
        self,
        video_frames: dict[str, torch.Tensor],
        query_timestamps: dict[str, list[float]],
        original_timestamps: dict[str, list[float]],
    ) -> dict[str, torch.BoolTensor]:
        padding_mask = {}

        for video_key, timestamps in original_timestamps.items():
            if video_key not in video_frames:
                continue  # 只对可用的视频键进行填充
            frames = []
            mask = []
            padding_frame = self._make_padding_camera_frame(video_key)
            for ts in timestamps:
                if is_float_in_list(ts, query_timestamps[video_key]):
                    idx = find_float_index(ts, query_timestamps[video_key])
                    frames.append(video_frames[video_key][idx, :])
                    mask.append(False)
                else:
                    frames.append(padding_frame)
                    mask.append(True)

            padding_mask[f"{video_key}_is_pad"] = torch.BoolTensor(mask)

        return padding_mask

    def make_frame(self, dataset_iterator: Backtrackable) -> Generator:
        """从数据集迭代器开始构造一个帧"""
        try:
            item = next(dataset_iterator)
        except StopIteration as e:
            # 在这里转译耗尽异常，以免 PEP 479 将其变成无法区分的 RuntimeError。
            raise _ShardExhaustedError from e
        item = item_to_torch(item)

        updates = []  # 要应用到从 hf_dataset 获取的条目上的“更新”列表（不含相机特征）

        # 从条目中获取 episode 索引
        ep_idx = item["episode_index"]

        # "timestamp" 在每个 episode 都从 0 重新开始，而我们需要的是单个 .mp4 文件内的全局时间步（由 index/fps 给出）
        current_ts = item["index"] / self.fps

        episode_boundaries_ts = {
            key: (
                self.meta.episodes[ep_idx][f"videos/{key}/from_timestamp"],
                self.meta.episodes[ep_idx][f"videos/{key}/to_timestamp"],
            )
            for key in self.meta.video_keys
        }

        # 必要时应用增量查询逻辑
        if self.delta_indices is not None:
            query_result, padding = self._get_delta_frames(dataset_iterator, item)
            updates.append(query_result)
            updates.append(padding)

        # 需要时加载视频帧
        if len(self.meta.video_keys) > 0:
            original_timestamps = self._make_timestamps_from_indices(current_ts, self.delta_indices)

            # 考虑到 episode 的边界，某些时间戳可能不可用
            query_timestamps = self._get_query_timestamps(
                current_ts, self.delta_indices, episode_boundaries_ts
            )
            video_frames = self._query_videos(query_timestamps, ep_idx)

            if self.image_transforms is not None:
                image_keys = self.meta.camera_keys
                for cam in image_keys:
                    video_frames[cam] = self.image_transforms(video_frames[cam])

            updates.append(video_frames)

            if self.delta_indices is not None:
                # 我们返回的帧数始终相同。不可用的帧会被填充。
                padding_mask = self._get_video_frame_padding_mask(
                    video_frames, query_timestamps, original_timestamps
                )
                updates.append(padding_mask)

        result = item.copy()
        for update in updates:
            result.update(update)

        # 将原始图像深度特征转换为输出单位（视频深度已经转换过）。
        for key, stored_unit in self._image_depth_units.items():
            if key in result and stored_unit is not None and stored_unit != self._depth_output_unit:
                result[key] = (
                    result[key] * MM_PER_METRE
                    if stored_unit == DEPTH_METER_UNIT
                    else result[key] / MM_PER_METRE
                )

        result["task"] = self.meta.tasks.iloc[item["task_index"]].name

        yield result

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
        episode_boundaries_ts: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        keys_to_timestamps = self._make_timestamps_from_indices(current_ts, query_indices)
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = keys_to_timestamps[key]
                # 将超出 episode 边界的时间步钳制到边界内
                query_timestamps[key] = torch.clamp(
                    torch.tensor(timestamps), *episode_boundaries_ts[key]
                ).tolist()

            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict:
        """注意：当使用数据 worker 时（例如 num_workers>0 的 DataLoader），不要在主进程中
        调用本函数（例如再使用一个 num_workers=0 的 DataLoader）。否则会导致
        段错误（Segmentation Fault）。这很可能是因为视频加载器的内存引用是在
        主进程中创建的，而子进程无法访问它。
        """

        item = {}
        for video_key, query_ts in query_timestamps.items():
            root = self.meta.url_root if self.streaming and not self.streaming_from_local else self.root
            video_path = f"{root}/{self.meta.get_video_file_path(ep_idx, video_key)}"
            if video_key in self.meta.depth_keys:
                # 深度图是 12 位量化的，只能通过 pyav 解码；再反量化
                # 回物理单位，以与非流式读取器保持一致。
                frames = decode_video_frames(
                    video_path,
                    query_ts,
                    self.tolerance_s,
                    backend="pyav",
                    return_uint8=False,
                    is_depth=True,
                )
                depth_encoder = self._depth_encoder_configs[video_key]
                frames = dequantize_depth(
                    frames,
                    depth_min=depth_encoder.depth_min,
                    depth_max=depth_encoder.depth_max,
                    shift=depth_encoder.shift,
                    use_log=depth_encoder.use_log,
                    output_unit=self._depth_output_unit,
                )
            else:
                frames = decode_video_frames_torchcodec(
                    video_path,
                    query_ts,
                    self.tolerance_s,
                    decoder_cache=self.video_decoder_cache,
                    return_uint8=self._return_uint8,
                )

            item[video_key] = frames.squeeze(0) if len(query_ts) == 1 else frames

        return item

    def _get_delta_frames(self, dataset_iterator: Backtrackable, current_item: dict):
        # TODO(fracapuano): 将本函数模块化，重构代码
        """使用可回退迭代器获取带增量偏移的帧。

        Args:
            current_item (dict): 来自迭代器的当前条目。
            ep_idx (int): Episode 索引。

        Returns:
            tuple: (query_result, padding) —— 增量偏移处的帧和填充信息。
        """
        current_episode_idx = current_item["episode_index"]

        # 准备结果
        query_result = {}
        padding = {}

        for key, delta_indices in self.delta_indices.items():
            if key in self.meta.video_keys:
                continue  # 视觉帧单独解码

            target_frames = []
            is_pad = []

            # 创建一个结果字典，按处理顺序存储帧，然后再重建原始顺序以便堆叠
            delta_results = {}

            # 按难度将增量分开并排序（先做较容易的操作）
            negative_deltas = sorted([d for d in delta_indices if d < 0], reverse=True)  # [-1, -2, -3, ...]
            positive_deltas = sorted([d for d in delta_indices if d > 0])  # [1, 2, 3, ...]
            zero_deltas = [d for d in delta_indices if d == 0]

            # 处理零增量（当前帧）
            for delta in zero_deltas:
                delta_results[delta] = (
                    current_item[key],
                    False,
                )

            # 按难度递增的顺序处理负增量
            lookback_failed = False

            last_successful_frame = current_item[key]

            for delta in negative_deltas:
                if lookback_failed:
                    delta_results[delta] = (last_successful_frame, True)
                    continue

                try:
                    steps_back = abs(delta)
                    if dataset_iterator.can_peek_back(steps_back):
                        past_item = dataset_iterator.peek_back(steps_back)
                        past_item = item_to_torch(past_item)

                        if past_item["episode_index"] == current_episode_idx:
                            delta_results[delta] = (past_item[key], False)
                            last_successful_frame = past_item[key]

                        else:
                            raise LookBackError("Retrieved frame is from different episode!")
                    else:
                        raise LookBackError("Cannot go back further than the history buffer!")

                except LookBackError:
                    delta_results[delta] = (last_successful_frame, True)
                    lookback_failed = True  # 后续所有负增量也都会失败

            # 按难度递增的顺序处理正增量
            lookahead_failed = False
            last_successful_frame = current_item[key]

            for delta in positive_deltas:
                if lookahead_failed:
                    delta_results[delta] = (last_successful_frame, True)
                    continue

                try:
                    if dataset_iterator.can_peek_ahead(delta):
                        future_item = dataset_iterator.peek_ahead(delta)
                        future_item = item_to_torch(future_item)

                        if future_item["episode_index"] == current_episode_idx:
                            delta_results[delta] = (future_item[key], False)
                            last_successful_frame = future_item[key]

                        else:
                            raise LookAheadError("Retrieved frame is from different episode!")
                    else:
                        raise LookAheadError("Cannot go ahead further than the lookahead buffer!")

                except LookAheadError:
                    delta_results[delta] = (last_successful_frame, True)
                    lookahead_failed = True  # 后续所有正增量也都会失败

            # 重建原始顺序以便堆叠
            for delta in delta_indices:
                frame, is_padded = delta_results[delta]

                # 为堆叠添加批量维度
                target_frames.append(frame)  # frame.unsqueeze(0))
                is_pad.append(is_padded)

            # 堆叠帧并加入结果
            if target_frames:
                query_result[key] = torch.stack(target_frames)
                padding[f"{key}_is_pad"] = torch.BoolTensor(is_pad)

        return query_result, padding

    def _validate_delta_timestamp_keys(self, delta_timestamps: dict[list[float]]) -> None:
        """
        校验 delta_timestamps 中的所有键都对应数据集中实际存在的特征。

        Raises:
            ValueError: 当任意 delta timestamp 键不对应数据集特征时。
        """
        if delta_timestamps is None:
            return

        # 从数据集元数据中获取所有可用的特征键
        available_features = set(self.meta.features.keys())

        # 获取 delta_timestamps 中的所有键
        delta_keys = set(delta_timestamps.keys())

        # 找出所有不对应特征的键
        invalid_keys = delta_keys - available_features

        if invalid_keys:
            raise ValueError(
                f"The following delta_timestamp keys do not correspond to dataset features: {invalid_keys}. "
                f"Available features are: {sorted(available_features)}"
            )
