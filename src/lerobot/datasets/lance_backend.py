#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""供 :class:`~lerobot.datasets.lerobot_dataset.LeRobotDataset` 使用的 Lance 存储后端。

服务于 ``meta/info.json`` 中声明 ``"storage_format": "lance"`` 的数据集：
表格特征存放在 ``frames.lance`` 表中，mp4 文件存放在经过 blob 编码的
``videos.lance`` 表中，两者与标准的 ``meta/`` 目录并列。表采用就地读取
方式——可以从本地、Hub（``hf://``）或任意对象存储读取。

此处的所有内容都是 ``"lance"`` 存储格式的实现细节；
公共入口是 ``LeRobotDataset``（参见 :mod:`lerobot.datasets.storage`）。
"""

from __future__ import annotations

import contextlib
import importlib
import json
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from lerobot.configs.video import DEFAULT_DEPTH_UNIT, DepthEncoderConfig
from lerobot.utils.import_utils import _lancedb_available, require_package

if TYPE_CHECKING or _lancedb_available:
    from lancedb.permutation import Permutation

from .dataset_metadata import LeRobotDatasetMetadata
from .dataset_reader import BaseDatasetReader
from .depth_utils import dequantize_depth
from .feature_utils import check_delta_timestamps, get_delta_indices

# 重新导出：storage.py 会在本模块上解析 localize_root；其余的是
# 该格式的公共辅助工具（转换器、DataLoader worker 上下文）。
from .lance_utils import (  # noqa: F401
    _OPEN_PROBE_BYTES,
    _RANGE_SLACK,
    FRAMES_TABLE,
    VIDEO_BLOB_COLUMN,
    VIDEO_INDEX_COLUMNS,
    VIDEOS_TABLE,
    _connect,
    _merge_spans,
    _SparseBlobSource,
    _VideoDecoderLRU,
    build_video_byte_index,
    lance_mp_context,
    localize_root,
    resolve_lance_root,
    to_lance_column,
)
from .utils import resolve_episode_indices
from .video_utils import FrameTimestampError, decode_video_frames_pyav


class LanceDatasetReader(BaseDatasetReader):
    """服务于 Lance 格式 LeRobot 数据集的数据集读取器。

    由 :class:`LeRobotDataset` 通过读取器查找表实例化
    （参见 :mod:`lerobot.datasets.storage`）；返回与默认
    parquet/mp4 流水线相同的条目字典。

    Args:
        meta: 已加载的数据集元数据（此处绝不会重建或修改）。
        root: 包含 ``meta/`` 和 ``.lance`` 表的本地目录，或具有相同
            布局的对象存储 URI。``None`` 表示通过 ``hf://`` 从
            Hub 仓库 ``meta.repo_id`` 流式读取表。
        episodes: 要选择的 episode 索引。``None`` 表示全部。
        image_transforms: 可选的、用于相机帧的 torchvision v2 变换。
        delta_timestamps: 特征键 -> 相对时间戳偏移量（秒），含义同
            :class:`LeRobotDataset`。
        tolerance_s: 时间戳同步容差（秒）。
        revision: Hub 版本，与传给 :class:`LeRobotDataset` 的一致。
        return_uint8: 以原始 uint8 而非归一化的 float32 返回 RGB 帧。
        depth_output_unit: 深度特征反量化后的单位（``'mm'`` 或 ``'m'``）。
            深度视频通过 pyav 解码（torchcodec 无法输出 16 位平面）。
        storage_options: 转发给 ``lancedb.connect`` 的额外选项。
        video_decoder_cache_size: 每个 worker 的最大解码器数量（默认 16，
            同时受每个 worker 2 GiB 字节预算限制）。
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        return_uint8: bool = False,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        storage_options: dict | None = None,
        video_decoder_cache_size: int | None = None,
        token: str | bool | None = None,
    ):
        require_package("lancedb", extra="lancedb")
        self.meta = meta
        self.repo_id = meta.repo_id
        self.tolerance_s = tolerance_s
        self._storage_options = storage_options
        self._token = token

        self._db_uri, _ = resolve_lance_root(self.repo_id, root, self._storage_options, revision, token)
        self._hub_revision = meta.revision if self._db_uri == f"hf://datasets/{self.repo_id}" else None

        if self.meta.image_keys:
            raise NotImplementedError(
                f"Image-backed features are not supported by the lance backend: {self.meta.image_keys}. "
                "Re-encode them as video."
            )
        if self.meta.video_keys:
            try:
                importlib.import_module("torchcodec")
            except Exception as error:
                raise ImportError(
                    "The lance reader decodes RGB video with torchcodec, which is not available "
                    "on this platform. Video lance datasets cannot be read here; the default "
                    "storage format supports video_backend='pyav'."
                ) from error
        # 深度视频通过 pyav 在相同的预取数据源上解码。
        self._depth_output_unit = depth_output_unit
        self._depth_encoder_configs = {
            key: DepthEncoderConfig.from_video_info(self.meta.features[key].get("info"))
            for key in self.meta.depth_keys
        }
        self.set_image_transforms(image_transforms)
        self.return_uint8 = return_uint8

        episodes = resolve_episode_indices(episodes, meta.total_episodes)
        if episodes is not None and not episodes:
            raise ValueError("None of the requested episodes are in the dataset.")
        self.episodes = episodes
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)

        self._ep_from = self._episode_numpy("dataset_from_index", np.int64)
        self._ep_to = self._episode_numpy("dataset_to_index", np.int64)
        # episode 范围必须恰好铺满 [0, total_frames)
        if len(self._ep_from) and (
            int(self._ep_from[0]) != 0
            or int(self._ep_to[-1]) != self.meta.total_frames
            or (self._ep_from[1:] != self._ep_to[:-1]).any()
        ):
            raise ValueError(
                "Episode boundaries in meta do not tile [0, total_frames) contiguously; "
                "the dataset metadata is inconsistent."
            )

        if self.episodes is not None:
            # 无论 episodes 列表的顺序如何，行都按存储顺序提供，
            # 与默认读取器的 parquet 谓词下推保持一致。
            self._rel_to_abs = np.concatenate(
                [np.arange(self._ep_from[ep], self._ep_to[ep]) for ep in sorted(self.episodes)]
            )
            self._absolute_to_relative_idx = {
                int(abs_idx): rel_idx for rel_idx, abs_idx in enumerate(self._rel_to_abs)
            }
        else:
            self._rel_to_abs = None
            self._absolute_to_relative_idx = None

        self._task_names = list(self.meta.tasks.index)

        self._tabular_keys = [
            key
            for key in self.meta.features
            if key not in self.meta.video_keys and key not in self.meta.image_keys
        ]
        self._fetch_columns = [to_lance_column(key) for key in self._tabular_keys]
        self._feature_shapes = {
            key: tuple(self.meta.features[key].get("shape") or ()) for key in self._tabular_keys
        }
        # 字符串特征以 Python 字符串形式透传，语言列
        # （list<struct>，lerobot#3467）以 Python 字典列表形式透传，与上游一致。
        self._string_keys = {
            key for key in self._tabular_keys if self.meta.features[key].get("dtype") == "string"
        }
        self._language_keys = {
            key for key in self._tabular_keys if self.meta.features[key].get("dtype") == "language"
        }

        # 每个 episode 对应的 mp4 位于哪个 (chunk, file)，以及它在该文件内的
        # 起始位置（v3.0 中多个 episode 共享文件；时间戳会按 from_timestamp 偏移）。
        self._video_locator = {
            key: (
                self._episode_numpy(f"videos/{key}/chunk_index", np.int64),
                self._episode_numpy(f"videos/{key}/file_index", np.int64),
                self._episode_numpy(f"videos/{key}/from_timestamp", np.float64),
            )
            for key in self.meta.video_keys
        }

        self._frames_perm = None
        self._videos_table = None
        self._video_row_ids: dict[tuple, int] | None = None
        self._file_meta: OrderedDict[tuple, dict] = OrderedDict()
        self._prefetch_pool: ThreadPoolExecutor | None = None
        self._decode_pool: ThreadPoolExecutor | None = None
        if video_decoder_cache_size is None:
            video_decoder_cache_size = 16
        self._decoder_cache = _VideoDecoderLRU(video_decoder_cache_size, byte_budget=2 << 30)  # 上限 2GB

    def _episode_numpy(self, name: str, dtype: type[np.generic]) -> np.ndarray:
        # 直接从底层 Arrow 列读取，而不是通过 HF Dataset 的 __getitem__
        column = self.meta.episodes.data.column(name).to_numpy(zero_copy_only=False)
        return column.astype(dtype, copy=False)

    def _ensure_open(self) -> None:
        if self._frames_perm is not None:
            return
        if self.meta.video_keys:
            self._prefetch_pool = ThreadPoolExecutor(max_workers=1)
            self._decode_pool = ThreadPoolExecutor(max_workers=16)
        db = _connect(self._db_uri, self._storage_options, revision=self._hub_revision, token=self._token)
        table = db.open_table(FRAMES_TABLE)
        n_rows = table.count_rows()
        if n_rows != self.meta.total_frames:
            raise ValueError(
                f"frames table has {n_rows} rows but meta declares "
                f"{self.meta.total_frames} frames; the dataset is truncated or corrupt."
            )
        self._frames_perm = (
            Permutation.identity(table).select_columns(self._fetch_columns).with_format("arrow")
        )
        if self.meta.video_keys:
            self._videos_table = db.open_table(VIDEOS_TABLE)
            # 未来的 TODO：按批次惰性解析 row id。
            index = (
                self._videos_table.search()
                .select(["video_key", "chunk_index", "file_index"])
                .with_row_id(True)
                .to_arrow()
            )
            self._video_row_ids = {
                (row["video_key"], row["chunk_index"], row["file_index"]): row["_rowid"]
                for row in index.to_pylist()
            }
            referenced = {
                (key, int(chunk), int(file))
                for key, (chunks, files, _) in self._video_locator.items()
                for chunk, file in zip(chunks, files, strict=True)
            }
            missing = referenced - self._video_row_ids.keys()
            if missing:
                raise ValueError(
                    f"videos table is missing {len(missing)} file(s) referenced by episode "
                    f"metadata, e.g. {sorted(missing)[:3]}. The dataset is incomplete or "
                    "was converted against different metadata."
                )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_frames_perm"] = None
        state["_videos_table"] = None
        state["_video_row_ids"] = None
        state["_file_meta"] = OrderedDict()
        state["_prefetch_pool"] = None
        state["_decode_pool"] = None
        state["_decoder_cache"] = _VideoDecoderLRU(
            self._decoder_cache.capacity, byte_budget=self._decoder_cache.byte_budget
        )
        return state

    def close(self) -> None:
        """关闭工作线程并释放表句柄；读取时会惰性地重新打开。"""
        for name in ("_prefetch_pool", "_decode_pool"):
            pool = getattr(self, name, None)
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            setattr(self, name, None)
        self._frames_perm = None
        self._videos_table = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    @property
    def num_frames(self) -> int:
        return len(self._rel_to_abs) if self._rel_to_abs is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        return self._absolute_to_relative_idx

    def get_item(self, idx: int) -> dict:
        return self.get_items([idx])[0]

    def get_items(self, indices: list[int]) -> list[dict]:
        """批量获取：每个批次执行一次去重的 frames 表读取和一次 blob 获取。"""
        self._ensure_open()
        plans = self._plan_batch(indices)
        rows, row_pos = self._batch_rows(plans)

        # 视频准备工作（字节索引、头部范围、解码器创建、帧窗口获取）
        # 只需要本批次涉及的文件，因此可以与 frames 表的获取并行。错误的
        # 推测范围只会触发范围读取回退来重新获取，绝不会返回错误的帧。
        prepared_future = None
        if self.meta.video_keys:
            windows = self._plan_file_windows(plans)
            prepared_future = self._prefetch_pool.submit(self._prepare_files, sorted(windows), windows)

        columns = self._fetch_rows(rows)
        items = [self._build_item(plan, columns, row_pos) for plan in plans]

        if self.meta.video_keys:
            decoded = self._decode_videos(plans, columns, row_pos, prepared_future.result())
            for item, frames in zip(items, decoded, strict=True):
                item.update(frames)
        if self._image_transforms is not None:
            for item in items:
                for cam_key in self.meta.camera_keys:
                    if cam_key in self.meta.depth_keys:
                        continue
                    item[cam_key] = self._image_transforms(item[cam_key])
        return items

    def _plan_file_windows(self, plans: list[dict]) -> dict[tuple, list[tuple[int, int]]]:
        """将批次中每个样本的视频窗口映射为（文件键 -> 帧跨度）。

        位置仅来自 episode 元数据；阶段 2 会根据真实时间戳
        重新推导范围，并获取所有遗漏的部分。
        """
        fps = float(self.meta.fps)
        windows: dict[tuple, list[tuple[int, int]]] = {}
        for plan in plans:
            ep_idx = plan["ep_idx"]
            for key in self.meta.video_keys:
                file_key, span = self._planned_file_window(key, ep_idx, plan, fps)
                windows.setdefault(file_key, []).append(span)
        return windows

    def _plan_batch(self, indices: list[int]) -> list[dict]:
        """将每个样本解析为它所需的绝对行及其填充掩码。"""
        plans = []
        for idx in indices:
            abs_idx = self._resolve_abs_idx(idx)
            ep_idx = self._episode_index_for_abs_idx(abs_idx)
            start, end = self._episode_bounds(ep_idx)
            plan = {"abs_idx": abs_idx, "ep_idx": ep_idx, "rows": {abs_idx}, "windows": {}, "padding": {}}
            if self.delta_indices is not None:
                for key, deltas in self.delta_indices.items():
                    window = [min(max(abs_idx + delta, start), end - 1) for delta in deltas]
                    plan["windows"][key] = window
                    plan["rows"].update(window)
                    plan["padding"][f"{key}_is_pad"] = torch.BoolTensor(
                        [not (start <= abs_idx + delta < end) for delta in deltas]
                    )
            plans.append(plan)
        return plans

    def _resolve_abs_idx(self, idx: int) -> int:
        idx = int(idx)
        if idx < 0:
            idx += self.num_frames
        if not 0 <= idx < self.num_frames:
            raise IndexError(f"Index {idx} is out of range for a dataset of {self.num_frames} frames.")
        return int(self._rel_to_abs[idx]) if self._rel_to_abs is not None else idx

    def _episode_index_for_abs_idx(self, abs_idx: int) -> int:
        return int(np.searchsorted(self._ep_from, abs_idx, side="right") - 1)

    def _episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        return int(self._ep_from[ep_idx]), int(self._ep_to[ep_idx])

    def _batch_rows(self, plans: list[dict]) -> tuple[list[int], dict[int, int]]:
        rows = sorted({row for plan in plans for row in plan["rows"]})
        return rows, {row: pos for pos, row in enumerate(rows)}

    def _planned_file_window(
        self, key: str, ep_idx: int, plan: dict, fps: float
    ) -> tuple[tuple[str, int, int], tuple[int, int]]:
        ep_start, _ = self._episode_bounds(ep_idx)
        _, _, from_ts_arr = self._video_locator[key]
        window = plan["windows"].get(key, [plan["abs_idx"]])
        base = round(float(from_ts_arr[ep_idx]) * fps) - ep_start
        first = base + min(window) - (1 if key in self.meta.depth_keys else 0)
        return self._video_file_key(key, ep_idx), (first, base + max(window))

    def _fetch_rows(self, rows: list[int]) -> dict[str, np.ndarray]:
        batch = self._frames_perm.__getitems__(rows)
        columns = {}
        for key, lance_name in zip(self._tabular_keys, self._fetch_columns, strict=True):
            array = batch.column(lance_name)
            if hasattr(array, "combine_chunks"):
                array = array.combine_chunks()
            if key in self._language_keys:
                language_rows = array.to_pylist()
                for row in language_rows:
                    for msg in row or ():
                        if msg.get("tool_calls"):
                            msg["tool_calls"] = [
                                json.loads(call) if isinstance(call, str) else call
                                for call in msg["tool_calls"]
                            ]
                columns[key] = language_rows
            elif hasattr(array, "flatten") and hasattr(array.type, "value_type"):
                values = array.flatten().to_numpy(zero_copy_only=False)
                columns[key] = values.reshape(len(array), -1)
            else:
                columns[key] = array.to_numpy(zero_copy_only=False)
        return columns

    def _decode_videos(
        self,
        plans: list[dict],
        columns: dict[str, np.ndarray],
        row_pos: dict[int, int],
        prepared: dict[tuple, tuple],
    ) -> list[dict[str, torch.Tensor]]:
        """解码一个批次所需的全部相机帧，每个文件执行一次 blob 获取和一遍解码。"""
        requests = self._build_video_requests(plans, columns["timestamp"], row_pos)
        entries = prepared

        results: list[dict[str, torch.Tensor]] = [{} for _ in plans]

        def _decode_file(file_key: tuple, file_requests: list[tuple[int, list[float]]]) -> None:
            key, chunk_idx, file_idx = file_key
            decoder, source = entries[file_key]
            if key in self.meta.depth_keys:
                for sample_idx, shifted_ts in file_requests:
                    results[sample_idx][key] = self._decode_depth_window(source, shifted_ts, file_key)
                return
            fps = decoder.metadata.average_fps
            for sample_idx, shifted_ts in file_requests:
                indices = [round(ts * fps) for ts in shifted_ts]
                batch = decoder.get_frames_at(indices=indices)
                distance = (torch.tensor(shifted_ts, dtype=torch.float64) - batch.pts_seconds).abs()
                if (distance > self.tolerance_s).any():
                    raise FrameTimestampError(
                        f"Query timestamps violate tolerance_s={self.tolerance_s} for video "
                        f"'{key}' (chunk {chunk_idx}, file {file_idx}): queried {shifted_ts}, "
                        f"loaded {batch.pts_seconds.tolist()}."
                    )
                frames = batch.data
                if not self.return_uint8:
                    frames = (frames / 255.0).type(torch.float32)
                results[sample_idx][key] = frames.squeeze(0)

        futures = [self._decode_pool.submit(_decode_file, k, r) for k, r in requests.items()]
        for future in futures:
            future.result()
        return results

    def _build_video_requests(
        self,
        plans: list[dict],
        timestamps: np.ndarray,
        row_pos: dict[int, int],
    ) -> dict[tuple, list[tuple[int, list[float]]]]:
        requests: dict[tuple, list[tuple[int, list[float]]]] = defaultdict(list)
        for sample_idx, plan in enumerate(plans):
            ep_idx = plan["ep_idx"]
            for key in self.meta.video_keys:
                window = plan["windows"].get(key, [plan["abs_idx"]])
                requests[self._video_file_key(key, ep_idx)].append(
                    (sample_idx, self._shifted_video_timestamps(key, ep_idx, window, timestamps, row_pos))
                )
        return requests

    def _prepare_files(
        self, file_keys: list[tuple], windows: dict[tuple, list[tuple[int, int]]] | None = None
    ) -> dict[tuple, tuple]:
        """视频解码的阶段 1：所有不需要时间戳的工作"""
        # 惰性加载 torchcodec
        from torchcodec.decoders import VideoDecoder

        self._load_file_meta([key for key in file_keys if key not in self._file_meta])

        prepared: dict[tuple, tuple] = {}
        new_files = []
        for key in file_keys:
            if key in self._decoder_cache:
                prepared[key] = self._decoder_cache.get(key)
            else:
                new_files.append(key)

        if new_files:
            handles = self._videos_table.fetch_blob_files(
                VIDEO_BLOB_COLUMN, [self._video_row_ids[key] for key in new_files]
            )
            sources = {
                key: _SparseBlobSource(self._file_meta[key]["file_size"], handle)
                for key, handle in zip(new_files, handles, strict=True)
            }
            spans_by_key: dict[tuple, list[tuple[int, int]]] = {}
            for key in new_files:
                meta = self._file_meta[key]
                spans = [
                    (0, min(_OPEN_PROBE_BYTES, meta["file_size"])),
                    # moov 之后的余量用于覆盖 ffmpeg 接下来要读取的 box 头部。
                    (
                        meta["moov_offset"],
                        min(meta["moov_offset"] + meta["moov_size"] + _RANGE_SLACK, meta["file_size"]),
                    ),
                ]
                if len(meta["kf_positions"]):
                    first_packet = int(meta["kf_positions"][0])
                    spans.append((first_packet, min(first_packet + _OPEN_PROBE_BYTES, meta["file_size"])))
                spans_by_key[key] = spans
            self._fetch_spans(spans_by_key, sources)

            rgb_files = [key for key in new_files if key[0] not in self.meta.depth_keys]
            created = self._decode_pool.map(
                lambda key: VideoDecoder(sources[key], seek_mode="approximate"), rgb_files
            )
            for key, decoder in zip(rgb_files, created, strict=True):
                prepared[key] = (decoder, sources[key])
            for key in new_files:
                if key[0] in self.meta.depth_keys:
                    prepared[key] = (None, sources[key])

        if windows:
            window_spans: dict[tuple, list[tuple[int, int]]] = {}
            for key, frame_windows in windows.items():
                meta = self._file_meta[key]
                source = prepared[key][1]
                spans = [
                    span
                    for first, last in frame_windows
                    for span in [self._window_byte_range(key[0], meta, first, last)]
                    if not source.covers(*span)
                ]
                if spans:
                    window_spans[key] = spans
            self._fetch_spans(window_spans, {key: prepared[key][1] for key in window_spans})

        # 将每个文件插入解码器缓存或刷新其缓存状态。
        for key, (decoder, source) in prepared.items():
            self._decoder_cache.put(key, (decoder, source), nbytes=source.buffered)
        return prepared

    def _video_file_key(self, key: str, ep_idx: int) -> tuple[str, int, int]:
        chunk_arr, file_arr, _ = self._video_locator[key]
        return key, int(chunk_arr[ep_idx]), int(file_arr[ep_idx])

    def _shifted_video_timestamps(
        self,
        key: str,
        ep_idx: int,
        window: list[int],
        timestamps: np.ndarray,
        row_pos: dict[int, int],
    ) -> list[float]:
        _, _, from_ts_arr = self._video_locator[key]
        base = float(from_ts_arr[ep_idx])
        return [base + float(timestamps[row_pos[row]]) for row in window]

    def _decode_depth_window(self, source, shifted_ts: list[float], file_key: tuple) -> torch.Tensor:
        """在我们的稀疏数据源上，使用上游的 pyav 解码器解码一个深度窗口。"""
        source.seek(0)
        frames = decode_video_frames_pyav(
            source, shifted_ts, self.tolerance_s, return_uint8=False, is_depth=True
        )
        config = self._depth_encoder_configs[file_key[0]]
        return dequantize_depth(
            frames,
            depth_min=config.depth_min,
            depth_max=config.depth_max,
            shift=config.shift,
            use_log=config.use_log,
            output_unit=self._depth_output_unit,
        ).squeeze(0)

    def _fetch_spans(
        self, spans_by_key: dict[tuple, list[tuple[int, int]]], sources: dict[tuple, _SparseBlobSource]
    ) -> None:
        range_requests: list[tuple[int, int, int]] = []
        range_targets: list[tuple[tuple, int]] = []
        for key, spans in spans_by_key.items():
            for start, end in _merge_spans(spans):
                range_requests.append((self._video_row_ids[key], start, end - start))
                range_targets.append((key, start))
        if not range_requests:
            return
        payloads = self._videos_table.fetch_blob_ranges(VIDEO_BLOB_COLUMN, range_requests)
        for (key, offset), payload in zip(range_targets, payloads, strict=True):
            sources[key].add(offset, payload.as_py())

    def _load_file_meta(self, missing: list[tuple]) -> None:
        """为尚未进入各 worker 缓存的文件获取字节索引列。"""
        if not missing:
            return
        row_ids = [self._video_row_ids[file_key] for file_key in missing]
        batch = (
            self._videos_table.take_row_ids(row_ids)
            .select(["video_key", "chunk_index", "file_index", *VIDEO_INDEX_COLUMNS])
            .to_arrow()
        )
        if batch.num_rows != len(missing):
            raise ValueError(
                "videos table is missing byte-index columns or rows "
                f"({batch.num_rows} matches for {len(missing)} files). Re-convert the dataset "
                f"with a converter that writes {VIDEO_INDEX_COLUMNS}."
            )
        scalars = {
            name: batch.column(name).to_pylist()
            for name in ("video_key", "chunk_index", "file_index", "file_size", "moov_offset", "moov_size")
        }
        kf_index_column = batch.column("kf_indices").combine_chunks()
        kf_position_column = batch.column("kf_positions").combine_chunks()
        index_offsets = kf_index_column.offsets.to_numpy(zero_copy_only=False)
        index_values = kf_index_column.values.to_numpy(zero_copy_only=False)
        position_offsets = kf_position_column.offsets.to_numpy(zero_copy_only=False)
        position_values = kf_position_column.values.to_numpy(zero_copy_only=False)
        for i in range(batch.num_rows):
            file_key = (scalars["video_key"][i], scalars["chunk_index"][i], scalars["file_index"][i])
            self._file_meta[file_key] = {
                "file_size": scalars["file_size"][i],
                "moov_offset": scalars["moov_offset"][i],
                "moov_size": scalars["moov_size"][i],
                "kf_indices": index_values[index_offsets[i] : index_offsets[i + 1]],
                "kf_positions": position_values[position_offsets[i] : position_offsets[i + 1]],
            }
        while len(self._file_meta) > 2048:
            self._file_meta.popitem(last=False)

    def _window_byte_range(self, key: str, meta: dict, first_frame: int, last_frame: int) -> tuple[int, int]:
        """覆盖帧 [first, last] 的字节范围：从前一个关键帧到下一个关键帧。"""
        kf_indices, kf_positions = meta["kf_indices"], meta["kf_positions"]
        start_idx = max(int(np.searchsorted(kf_indices, first_frame, side="right")) - 1, 0)
        end_idx = int(np.searchsorted(kf_indices, last_frame, side="right"))
        end = int(kf_positions[end_idx]) if end_idx < len(kf_positions) else meta["file_size"]
        slack = _RANGE_SLACK * 4 if key in self.meta.depth_keys else _RANGE_SLACK
        return int(kf_positions[start_idx]), min(end + slack, meta["file_size"])

    def _build_item(self, plan: dict, columns: dict[str, np.ndarray], row_pos: dict[int, int]) -> dict:
        base = row_pos[plan["abs_idx"]]
        item = {}
        for key in self._tabular_keys:
            data = columns[key]
            shape = self._feature_shapes[key]
            if key in self._language_keys:
                item[key] = data[base]
            elif key in self._string_keys:
                value = data[base]
                item[key] = value if isinstance(value, str) or value is None else str(value)
            elif key in plan["windows"]:
                window = data[[row_pos[row] for row in plan["windows"][key]]]
                if len(shape) > 1:
                    window = window.reshape(len(window), *shape)
                item[key] = torch.from_numpy(np.ascontiguousarray(window))
            elif data.ndim > 1:
                value = data[base]
                if len(shape) > 1:
                    value = value.reshape(shape)
                item[key] = torch.from_numpy(value.copy())
            else:
                item[key] = torch.tensor(data[base])
        item.update(plan["padding"])
        item["task"] = self._task_names[int(item["task_index"].item())]
        return item

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  repo_id={self.repo_id},\n"
            f"  uri={self._db_uri},\n"
            f"  episodes={self.num_episodes} selected / {self.meta.total_episodes} total,\n"
            f"  frames={self.num_frames},\n"
            f")"
        )


# lerobot.datasets.storage 在 storage_format 为 "lance" 时实例化的类。
DATASET_READER = LanceDatasetReader
