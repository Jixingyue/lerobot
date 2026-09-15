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
from collections.abc import Callable
from pathlib import Path

import datasets
import torch
import torch.utils
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.configs import DEFAULT_DEPTH_UNIT, DepthEncoderConfig, RGBEncoderConfig
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .dataset_reader import BaseDatasetReader, DatasetReader
from .dataset_writer import DatasetWriter
from .storage import (
    DEFAULT_STORAGE_FORMAT,
    is_remote_uri,
    localize_remote_root,
    make_dataset_reader,
)
from .utils import (
    create_lerobot_dataset_card,
    get_safe_version,
    is_valid_version,
)
from .video_utils import (
    StreamingVideoEncoder,
    get_safe_default_video_backend,
)

logger = logging.getLogger(__name__)


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        episode_filter: Callable[[dict], bool] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        return_uint8: bool = False,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        batch_encoding_size: int = 1,
        rgb_encoder: RGBEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        encoder_threads: int | None = None,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        *,
        repo_type: str = "dataset",
        token: str | bool | None = None,
    ):
        """
        根据两种不同的使用场景，实例化本类有 2 种可用模式：

        1. 你的数据集已经存在：
            - 位于本地磁盘的 'root' 文件夹中。通常当你在本地录制了
              数据集、但可能尚未推送到 hub 时属于这种情况。使用 'root'
              实例化本类会直接从磁盘加载数据集。这可以在离线（无
              网络连接）状态下完成。

            - 位于 Hugging Face Hub，地址为 https://huggingface.co/datasets/{repo_id}，而不在
              本地磁盘的 'root' 文件夹中。使用该 'repo_id' 实例化本类会
              从该地址下载并加载数据集，前提是你的数据集符合
              codebase_version v3.0。如果你的数据集创建于这种新格式之前，系统会
              提示你使用我们提供的从 v2.1 转换到 v3.0 的转换脚本，脚本位于
              lerobot/scripts/convert_dataset_v21_to_v30.py。


        2. 你的数据集尚不存在（本地磁盘和 Hub 上都没有）：你可以使用 'create' 类方法
           创建一个空的 LeRobotDataset。这可用于录制数据集，或将已有
           数据集移植为 LeRobotDataset 格式。


        从文件角度看，LeRobotDataset 封装了 3 个主要部分：
            - 元数据：
                - info 包含数据集的各种信息，如形状、键、fps 等。
                - stats 存储不同模态的数据集统计信息，用于归一化。
                - tasks 包含数据集中每个任务的提示，可用于
                  任务条件化训练。
            - data（由 datasets.Dataset 支持），从 parquet 文件中读取值。
            - videos（可选），从中加载帧，使其与来自 parquet 文件的数据保持同步。

        一个典型的 LeRobotDataset 在其根路径下的结构如下：
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes
        │   │   ├── chunk-000
        │   │   │   ├── file-000.parquet
        │   │   │   ├── file-001.parquet
        │   │   │   └── ...
        │   │   ├── chunk-001
        │   │   │   └── ...
        │   │   └── ...
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.parquet
        └── videos
            ├── observation.images.laptop
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            ├── observation.images.phone
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            └── ...

        请注意，这种基于文件的结构被设计得尽可能通用。多个 episode
        被合并到分片文件中，从而提升存储效率和加载性能。数据集的
        结构完全由 info.json 文件描述，在下载任何实际数据之前就可以
        轻松下载该文件或直接在 hub 上查看。所使用的文件类型非常
        简单，不需要复杂的工具即可读取，只用到 .parquet、.json 和 .mp4 文件（README 则为 .md
        文件）。

        Args:
            repo_id (str): 将用于获取数据集的 repo id。
            root (Path | None, optional): 读取数据集或将数据集下载到其中的
                本地目录。若设置，所有数据集文件都会直接物化到该路径下。
                若不设置，已有的本地数据集仍会在 ``$HF_LEROBOT_HOME/{repo_id}``
                下查找，但从 Hub 下载时会使用
                ``$HF_LEROBOT_HOME/hub`` 下对版本安全的快照缓存。
                也可以是对象存储 URI（例如 ``hf://datasets/{repo_id}``），
                用于支持就地读取数据的存储格式；此时只有 ``meta/`` 会被物化到本地。
            episodes (list[int] | None, optional): 若指定，则只加载本列表中
                以 episode_index 指定的 episode。默认为 None。
            episode_filter (Callable[[dict], bool] | None, optional): 作用于
                每个 episode 元数据行、用于选择 episode 的谓词。针对不含 ``stats`` 键
                的 ``meta/`` 行求值（例如 ``task_index``、``episode_index``、
                ``length``、``from_timestamp``、``to_timestamp``）。
                当同时设置 ``episodes`` 时取两者的交集。示例：``lambda ep: ep["length"] >= 100``。
                默认为 None。
            image_transforms (Callable | None, optional):
                在 `__getitem__` 中、图像解码/张量转换之后应用于视觉模态的
                变换。对基于图像和基于视频的观测都有效，之后可以通过
                `set_image_transforms()` 更新，或通过 `clear_image_transforms()` 清除。
                默认为 None。
            delta_timestamps (dict[list[float]] | None, optional): _description_。默认为 None。
            tolerance_s (float, optional): 以秒为单位的容差，用于确保数据时间戳确实
                与 fps 值同步。在数据集初始化时用它来确保每个时间戳与
                下一个之间相隔 1/fps +/- tolerance_s。这同样适用于从视频文件
                解码出的帧。它还用于检查 `delta_timestamps`（在提供时）是否
                为 1/fps 的整数倍。默认为 1e-4。
            revision (str, optional): 可选的 Git 版本 id，可以是分支名、标签或
                提交哈希。默认为当前代码库版本标签。
            force_cache_sync (bool, optional): 优先同步并刷新本地文件的标志。若为 True
                且文件已存在于本地缓存中，速度会更快。但是，加载的文件可能
                与 hub 上的版本不同步，尤其是在你指定了 'revision' 的情况下。默认为
                False。
            download_videos (bool, optional): 下载视频的标志。请注意，当设为 True 但
                视频文件已存在于本地磁盘时，不会重复下载。默认为
                True。
            video_backend (str | None, optional): 用于解码视频的视频后端。当平台上可用时默认为
                torchcodec；否则默认为 'pyav'。
                你也可以使用 Torchvision 使用的 'pyav' 解码器（它曾经是默认选项），
                或 Torchvision 的另一个解码器 'video_reader'。
            batch_encoding_size (int, optional): 批量编码视频之前累积的
                episode 数量。设为 1 表示立即编码（默认），设为更大的值表示批量编码。默认为 1。
            rgb_encoder (RGBEncoderConfig | None, optional): 相机的视频编码器设置
                （编解码器、质量等）。为 ``None`` 时，写入器使用
                :func:`~lerobot.configs.video.rgb_encoder_defaults`。
            depth_encoder (DepthEncoderConfig | None, optional): 深度相机的视频编码器设置
                （编解码器、质量等）。为 ``None`` 时，写入器使用
                :func:`~lerobot.configs.video.depth_encoder_defaults`。
            encoder_threads (int | None, optional): 编码器线程数（全局）。``None`` 表示
                由编解码器决定。
            streaming_encoding (bool, optional): 若为 True，则在采集期间实时编码视频帧，
                而不是先写入 PNG 图像。这使得 save_episode() 几乎瞬时完成。默认为 False。
            encoder_queue_maxsize (int, optional): 使用流式编码时每个相机缓冲的
                最大帧数。默认为 30（30fps 下约 1 秒）。
            repo_type (str, optional): "dataset"（默认）或 "bucket"，后者表示 HF
                Storage Bucket。使用 "bucket" 且不指定 ``root`` 时，数据集从
                ``hf://buckets/{repo_id}`` 就地读取（map 风格访问需要
                非默认的存储格式；默认格式在 bucket 上仅支持流式）。
                显式指定的 ``root`` 始终优先于 ``repo_type``。
            token: 从 Hub 下载本数据集时使用的认证令牌。
                可传入字符串令牌；``True`` 表示要求使用本地存储的
                令牌；``False`` 表示禁用认证；
                ``None`` 表示使用 Hugging Face Hub 的默认行为。初始化之后，
                令牌不会保留在数据集实例上。

        Note:
            传给 ``__init__`` 的写入模式参数（``streaming_encoding``、
            ``batch_encoding_size``）已弃用。创建新数据集请使用
            :meth:`create`，向已有数据集追加内容请使用 :meth:`resume`。
        """
        super().__init__()
        self.repo_id = repo_id
        if repo_type not in ("dataset", "bucket"):
            raise ValueError(f"repo_type must be 'dataset' or 'bucket', got {repo_type!r}")
        if root is None and repo_type == "bucket":
            root = f"hf://buckets/{repo_id}"
        # 数据集可以位于对象存储根路径（例如 ``hf://datasets/...``）：
        # 非默认读取器会就地读取数据，只有 ``meta/`` 会被本地化。
        self._storage_root = root if root is not None and is_remote_uri(root) else None
        if self._storage_root is not None:
            root = localize_remote_root(
                repo_id, self._storage_root, revision, token=token, force_cache_sync=force_cache_sync
            )
        self._requested_root = Path(root) if root else None
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self._video_backend = video_backend if video_backend else get_safe_default_video_backend()
        self._return_uint8 = return_uint8
        self._depth_output_unit = depth_output_unit
        self._batch_encoding_size = batch_encoding_size
        self._encoder_threads = encoder_threads

        if self._requested_root is not None:
            self._requested_root.mkdir(exist_ok=True, parents=True)

        # 加载元数据（从解析出的元数据根路径一次性设置 self.root）
        self.meta = LeRobotDatasetMetadata(
            self.repo_id,
            self._requested_root,
            self.revision,
            # 对象存储根路径在本地化时已经刷新过其 meta/
            force_cache_sync=force_cache_sync and self._storage_root is None,
            token=token,
        )
        self.root = self.meta.root
        self.revision = self.meta.revision
        self.meta.rescale_depth_stats(self._depth_output_unit)

        if episodes is not None and any(
            episode >= self.meta.total_episodes or episode < 0 for episode in episodes
        ):
            logger.warning(
                f"Some episodes in the provided episodes list are out of range for this dataset ({self.meta.total_episodes})."
            )

        if episode_filter is not None:
            resolved = self.meta.filter_episodes(episode_filter, candidates=episodes)
            if not resolved:
                raise ValueError(
                    "The episode filter did not match any episode. Make sure the filter and episodes list are valid and compatible."
                )
            logger.info(f"The episode filter matched {len(resolved)} episode(s).")
            episodes = resolved
        self.episodes = episodes

        if self._storage_root is not None and self.meta.storage_format == DEFAULT_STORAGE_FORMAT:
            raise ValueError(
                f"The dataset at {self._storage_root!r} has the default {DEFAULT_STORAGE_FORMAT!r} "
                "storage format, which cannot be read in place from an object store. For HF Storage "
                "Buckets, use repo_type='bucket' with dataset.streaming=true."
            )

        is_default_format = self.meta.storage_format == DEFAULT_STORAGE_FORMAT
        reader_kwargs = {
            "meta": self.meta,
            "episodes": episodes,
            "delta_timestamps": delta_timestamps,
            "image_transforms": image_transforms,
            "tolerance_s": tolerance_s,
            "return_uint8": return_uint8,
            "depth_output_unit": depth_output_unit,
        }
        if is_default_format:
            reader_kwargs.update(root=self.root, video_backend=self._video_backend)
        else:
            # 非默认格式在其根路径处就地读取数据
            reader_kwargs.update(root=self._storage_root or root, revision=revision, token=token)
        self.reader: BaseDatasetReader | None = make_dataset_reader(self.meta.storage_format, **reader_kwargs)
        self.image_transforms = image_transforms
        if not is_default_format:
            self.episodes = self.reader.episodes
            self.writer = None
            self._is_finalized = False
            return

        # 加载实际数据
        if force_cache_sync or not self.reader.try_load():
            if is_valid_version(self.revision):
                if token is None:
                    self.revision = get_safe_version(self.repo_id, self.revision)
                else:
                    self.revision = get_safe_version(self.repo_id, self.revision, token=token)
            self._download(download_videos, token=token)
            self.reader.load_and_activate()

        # 检测写入模式参数，以保持向后兼容
        _has_write_params = streaming_encoding or batch_encoding_size != 1
        if _has_write_params:
            import warnings

            warnings.warn(
                "Passing write-mode parameters (streaming_encoding, batch_encoding_size) to "
                "LeRobotDataset.__init__() is deprecated. Use LeRobotDataset.resume() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            streaming_enc = None
            if streaming_encoding and len(self.meta.video_keys) > 0:
                streaming_enc = self._build_streaming_encoder(
                    self.meta.fps,
                    rgb_encoder,
                    depth_encoder,
                    encoder_queue_maxsize,
                    encoder_threads,
                )
            self.writer = DatasetWriter(
                meta=self.meta,
                root=self.root,
                rgb_encoder=rgb_encoder,
                depth_encoder=depth_encoder,
                encoder_threads=encoder_threads,
                batch_encoding_size=batch_encoding_size,
                streaming_encoder=streaming_enc,
                initial_frames=self.meta.total_frames,
            )
        else:
            self.writer = None

        self._is_finalized = False

    # ── 写入器守卫 ──────────────────────────────────────────────────

    def _require_writer(self, method_name: str) -> None:
        if self.writer is None:
            raise RuntimeError(
                f"Cannot call '{method_name}()' on a read-only dataset. "
                f"Use LeRobotDataset.create() for new recording or "
                f"LeRobotDataset.resume() for resume recording."
            )
        if self._is_finalized:
            raise RuntimeError(
                f"Cannot call '{method_name}()' after finalize(). "
                f"Use LeRobotDataset.resume() to append more episodes."
            )

    # ── 读取器守卫 ──────────────────────────────────────────────────

    def _ensure_reader(self) -> BaseDatasetReader:
        """返回读取器，在首次访问时惰性创建默认读取器。

        ``self.reader`` 仅在写入模式（create/resume）下为 ``None``，
        而写入模式只存在于默认格式——非默认格式会在 ``__init__``
        中构造自己的读取器。
        """
        if self.writer is not None and not self._is_finalized:
            raise RuntimeError(
                "Cannot read from a dataset that is being recorded. Call finalize() first, then access items."
            )
        if self.reader is None:
            self.meta.ensure_readable()
            self.reader = DatasetReader(
                meta=self.meta,
                root=self.root,
                episodes=self.episodes,
                tolerance_s=self.tolerance_s,
                video_backend=self._video_backend,
                delta_timestamps=self.delta_timestamps,
                image_transforms=self.image_transforms,
                return_uint8=self._return_uint8,
                depth_output_unit=self._depth_output_unit,
            )
        return self.reader

    @staticmethod
    def _build_streaming_encoder(
        fps: int,
        rgb_encoder: RGBEncoderConfig | None,
        depth_encoder: DepthEncoderConfig | None,
        encoder_queue_maxsize: int,
        encoder_threads: int | None,
    ) -> StreamingVideoEncoder:
        return StreamingVideoEncoder(
            fps=fps,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )

    # ── 元数据属性 ───────────────────────────────────────────────────

    @property
    def fps(self) -> int:
        """数据采集时使用的每秒帧数。"""
        return self.meta.fps

    @property
    def depth_output_unit(self) -> str:
        """读取时深度图和统计信息所使用的物理单位（``"m"`` 或 ``"mm"``）。"""
        return self._depth_output_unit

    @property
    def num_frames(self) -> int:
        """所选 episode 中的帧数。"""
        # 直接检查而不使用 _ensure_reader()：在只写模式
        # （create/resume）下，我们依赖元数据而非初始化读取器。
        if self.reader is None:
            return self.meta.total_frames
        return self.reader.num_frames

    @property
    def num_episodes(self) -> int:
        """所选 episode 的数量。"""
        # 直接检查而不使用 _ensure_reader()：在只写模式
        # （create/resume）下，我们依赖元数据而非初始化读取器。
        if self.reader is None:
            return len(self.episodes) if self.episodes is not None else self.meta.total_episodes
        return self.reader.num_episodes

    @property
    def features(self) -> dict[str, dict]:
        """特征规范字典，将特征名映射到其类型/形状元数据。"""
        return self.meta.features

    @property
    def hf_dataset(self) -> datasets.Dataset:
        """底层的 Hugging Face Dataset 对象"""
        reader = self._ensure_reader()
        if not isinstance(reader, DatasetReader):
            raise AttributeError(
                f"hf_dataset is not available for storage_format={self.meta.storage_format!r}: "
                "data is read through its own dataset reader."
            )
        if reader.hf_dataset is None:
            reader.load_and_activate()
        return reader.hf_dataset

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        """从绝对帧索引到相对行位置的映射。

        仅对经过 episode 过滤、且绝对索引（来自元数据）
        与过滤后视图中的行位置不同的数据集为非 None。
        """
        return self._ensure_reader().absolute_to_relative_idx

    # ── 委托给写入器的方法 ──────────────────────────────────────────

    def add_frame(self, frame: dict) -> None:
        """向当前 episode 缓冲区添加单帧数据。

        委托给 :meth:`DatasetWriter.add_frame`。数据集必须处于
        写入模式（通过 :meth:`create` 或 :meth:`resume` 创建）。

        Args:
            frame: 将特征名映射到本帧各特征值的字典。
                必须包含 ``'task'`` 键。Torch 张量会被转换为 numpy。

        Raises:
            RuntimeError: 当数据集为只读（没有写入器）时。
        """
        self._require_writer("add_frame")
        self.writer.add_frame(frame)

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
        """将当前 episode 缓冲区保存到磁盘。

        委托给 :meth:`DatasetWriter.save_episode`。负责编码视频、写入
        parquet 数据并更新元数据。之后 episode 缓冲区会被重置。

        Args:
            episode_data: 可选的、预先构建好的 episode 字典。若为 ``None``，
                则使用由 :meth:`add_frame` 填充的内部 episode 缓冲区。
            parallel_encoding: 若为 ``True`` 且存在多个相机，则使用
                进程池并行编码视频。

        Raises:
            RuntimeError: 当数据集为只读（没有写入器）时。
        """
        self._require_writer("save_episode")
        self.writer.save_episode(episode_data, parallel_encoding)

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        """不保存，直接丢弃当前 episode 缓冲区。

        委托给 :meth:`DatasetWriter.clear_episode_buffer`。适用于
        丢弃失败或被中断的录制 episode。

        Args:
            delete_images: 若为 ``True``，同时删除为当前 episode
                写入磁盘的临时图像文件。

        Raises:
            RuntimeError: 当数据集为只读（没有写入器）时。
        """
        self._require_writer("clear_episode_buffer")
        self.writer.clear_episode_buffer(delete_images)

    def has_pending_frames(self) -> bool:
        """检查 episode 缓冲区中是否有未保存的帧。"""
        if self.writer is None:
            return False
        # save_episode 会弹出 "size"，因此在保存中途被遗弃的缓冲区没有挂起内容。
        return self.writer.episode_buffer is not None and self.writer.episode_buffer.get("size", 0) > 0

    def finalize(self):
        """刷新所有挂起的工作并关闭写入器。

        必须在数据采集/转换完成后调用，否则页脚元数据
        不会被写入 parquet 文件，数据集将无效。

        幂等——可安全地多次调用。如果从未显式调用，
        DatasetWriter.__del__ 会作为安全兜底。
        """
        if self._is_finalized:
            return
        if self.writer is not None:
            self.writer.finalize()
        self._is_finalized = True

    # ── Dataset 核心方法 ──────────────────────────────────────────

    def __len__(self):
        """返回所选 episode 中的帧数。"""
        return self.num_frames

    def __getitem__(self, idx: int | slice) -> dict | list[dict]:
        """返回一帧或一个切片的多帧，并应用所有变换。

        从底层 HF 数据集加载帧，展开增量时间戳窗口，
        解码视频帧，并应用图像变换。核心逻辑委托给
        :class:`DatasetReader`。

        Args:
            idx: 对可能经过 episode 过滤的数据集的整数索引或切片。

        Returns:
            整数索引时返回一个帧字典；切片时返回
            帧字典的列表。

        Raises:
            RuntimeError: 当数据集正在录制且尚未调用
                :meth:`finalize` 时。
        """
        if isinstance(idx, slice):
            return [self[item_idx] for item_idx in range(*idx.indices(len(self)))]

        return self._ensure_reader().get_item(idx)

    def __getitems__(self, indices: list[int]) -> list[dict]:
        return self._ensure_reader().get_items(list(indices))

    def select_columns(self, column_names: str | list[str]):
        """从底层数据集中选择指定的列。

        适用于在回放时提取动作序列而无需加载全部特征。
        返回仅包含所请求列的 ``datasets.Dataset``。
        """
        return self.hf_dataset.select_columns(column_names)

    def get_raw_item(self, idx) -> dict:
        """获取未应用图像变换的原始帧。

        与 ``__getitem__`` 不同，本方法返回给定索引处原始的
        HF 数据集行，不做增量时间戳展开、视频解码或图像变换。
        """
        return self.hf_dataset[idx]

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            f"}})"
        )

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """替换应用于视觉观测的变换。"""
        self._ensure_reader().set_image_transforms(image_transforms)
        self.image_transforms = image_transforms

    def clear_image_transforms(self) -> None:
        """移除应用于视觉观测的变换。"""
        if self.reader is not None:
            self.reader.set_image_transforms(None)
        self.image_transforms = None

    # ── Hub 方法（保留在门面类上） ──────────────────────────────────

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool | None = None,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        """将数据集上传到 Hugging Face Hub。

        如果仓库不存在则创建，上传所有数据集文件
        （可选择排除视频），生成数据集卡片，并使用当前
        代码库版本为该版本打标签。

        Args:
            branch: 可选的推送目标分支。若不存在则基于当前
                版本创建。
            tags: 数据集卡片的可选标签列表。
            license: 数据集卡片的许可证标识符。
            tag_version: 若为 ``True``，为当前代码库
                版本创建 Git 标签。
            push_videos: 若为 ``False``，跳过上传 ``videos/`` 目录。
            private: 若为 ``True``，创建私有仓库。若为 ``None``
                （默认），则遵从 Hub 上组织的默认设置（仅影响组织）。
            allow_patterns: 限制上传文件范围的 glob 模式。
            upload_large_folder: 若为 ``True``，对于超大型数据集使用
                ``upload_large_folder`` 而非 ``upload_folder``。
            **card_kwargs: 转发给数据集卡片创建过程的额外
                关键字参数。
        """
        if self.meta.storage_format != DEFAULT_STORAGE_FORMAT:
            raise NotImplementedError(
                f"push_to_hub is not supported for storage_format={self.meta.storage_format!r}: "
                "the data files are not managed by LeRobotDataset."
            )
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        card = create_lerobot_dataset_card(
            tags=tags, dataset_info=self.meta.info, license=license, repo_id=self.repo_id, **card_kwargs
        )
        card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def _download(self, download_videos: bool = True, *, token: str | bool | None = None) -> None:
        """按给定版本从指定的 'repo_id' 下载数据集。"""
        ignore_patterns = None if download_videos else "videos/"
        files = None
        token_kwargs = {} if token is None else {"token": token}
        if self.episodes is not None:
            # 此处保证读取器已存在（在 __init__ 中、_download 之前创建）
            files = self.reader.get_episodes_file_paths()

        if self._requested_root is None:
            self.meta.root = Path(
                snapshot_download(
                    self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    cache_dir=HF_LEROBOT_HUB_CACHE,
                    allow_patterns=files,
                    ignore_patterns=ignore_patterns,
                    **token_kwargs,
                )
            )
        else:
            self._requested_root.mkdir(exist_ok=True, parents=True)
            snapshot_download(
                self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                local_dir=self._requested_root,
                allow_patterns=files,
                ignore_patterns=ignore_patterns,
                **token_kwargs,
            )
            self.meta.root = self._requested_root

        # 传播从元数据解析出的根路径（唯一事实来源）
        self.root = self.meta.root
        self.reader.root = self.meta.root

    # ── 类构造函数 ────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        rgb_encoder: RGBEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        metadata_buffer_size: int = 10,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        encoder_threads: int | None = None,
        video_files_size_in_mb: int | None = None,
        data_files_size_in_mb: int | None = None,
    ) -> "LeRobotDataset":
        """从头创建一个新的 LeRobotDataset，用于录制数据。

        返回一个处于写入模式、带有活动 :class:`DatasetWriter` 的
        数据集。使用 :meth:`add_frame` / :meth:`save_episode`
        填充数据，完成后调用 :meth:`finalize`。

        Args:
            repo_id: 仓库标识符，通常为 ``'{hf_user}/{dataset_name}'``。
            fps: 数据采集时使用的每秒帧数。
            features: 特征规范字典，将特征名映射到其
                类型/形状元数据。
            root: 数据集存储的本地目录。默认为
                ``$HF_LEROBOT_HOME/{repo_id}``。
            robot_type: 可选的机器人类型字符串，存储在元数据中。
            use_videos: 若为 ``True``，视觉模态存储为 MP4 视频。
                若为 ``False``，则存储为图像。
            tolerance_s: 时间戳同步容差（秒）。
            image_writer_processes: 异步图像写入使用的子进程数。
                ``0`` 表示仅使用线程。
            image_writer_threads: 异步图像写入使用的线程数。
            video_backend: 视频解码后端（回读时使用）。
            batch_encoding_size: 批量编码视频之前累积的
                episode 数量。``1`` 表示立即编码。
            rgb_encoder: 相机的视频编码器设置（编解码器、质量等）。
                为 ``None`` 时使用 :func:`~lerobot.configs.video.rgb_encoder_defaults`。
            depth_encoder: 深度相机的视频编码器设置（编解码器、质量等）。
                为 ``None`` 时使用 :func:`~lerobot.configs.video.depth_encoder_defaults`。
            encoder_threads: 编码器线程数（全局）。``None``
                表示由编解码器决定。
            metadata_buffer_size: 刷新到 parquet 之前缓冲的
                episode 元数据记录数量。
            streaming_encoding: 若为 ``True``，则在采集期间实时
                编码视频帧，而不是先写入图像。
            encoder_queue_maxsize: 使用流式编码时每个相机
                缓冲的最大帧数。

        Returns:
            一个处于写入模式的新 :class:`LeRobotDataset`。
        """
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
            metadata_buffer_size=metadata_buffer_size,
            video_files_size_in_mb=video_files_size_in_mb,
            data_files_size_in_mb=data_files_size_in_mb,
        )
        obj.repo_id = obj.meta.repo_id
        obj._requested_root = obj.meta.root
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.episodes = None
        obj._video_backend = video_backend if video_backend is not None else get_safe_default_video_backend()
        obj._return_uint8 = False
        obj._depth_output_unit = DEFAULT_DEPTH_UNIT
        obj._batch_encoding_size = batch_encoding_size
        obj._encoder_threads = encoder_threads
        obj._storage_root = None

        # 读取器在首次访问时惰性创建（只写模式）
        obj.reader = None

        streaming_enc = None
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            streaming_enc = cls._build_streaming_encoder(
                fps, rgb_encoder, depth_encoder, encoder_queue_maxsize, encoder_threads
            )
        obj.writer = DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            encoder_threads=encoder_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoder=streaming_enc,
        )

        if image_writer_processes or image_writer_threads:
            obj.writer.start_image_writer(image_writer_processes, image_writer_threads)

        obj._is_finalized = False

        return obj

    @classmethod
    def resume(
        cls,
        repo_id: str,
        root: str | Path | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        rgb_encoder: RGBEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        encoder_threads: int | None = None,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        *,
        token: str | bool | None = None,
    ) -> "LeRobotDataset":
        """在已有数据集上恢复录制。

        从已有数据集（本地或 Hub）加载元数据，并创建一个
        :class:`DatasetWriter` 用于追加新 episode。底层 HF
        数据集在调用 :meth:`finalize` 之后、随后读取数据时
        才会加载。

        Args:
            repo_id: 已有数据集的仓库标识符。
            root: 数据集的本地目录。提供时，从 Hub 下载的内容会
                直接物化到该目录中。省略时，从 Hub 下载会使用
                ``$HF_LEROBOT_HOME/hub`` 下对版本安全的快照缓存。
            tolerance_s: 时间戳同步容差（秒）。
            revision: Git 版本（分支、标签或提交哈希）。默认为
                当前代码库版本标签。
            force_cache_sync: 若为 ``True``，即使存在本地缓存
                也从 Hub 重新下载元数据。
            video_backend: 回读数据时使用的视频解码后端。
            batch_encoding_size: 批量编码视频之前累积的
                episode 数量。
            rgb_encoder: 相机的视频编码器设置（编解码器、质量等）。
                为 ``None`` 时使用 :func:`~lerobot.configs.video.rgb_encoder_defaults`。
            depth_encoder: 深度相机的视频编码器设置（编解码器、质量等）。
                为 ``None`` 时使用 :func:`~lerobot.configs.video.depth_encoder_defaults`。
            encoder_threads: 编码器线程数（全局）。``None``
                表示由编解码器决定。
            image_writer_processes: 异步图像写入使用的子进程数。
            image_writer_threads: 异步图像写入使用的线程数。
            streaming_encoding: 若为 ``True``，则在采集期间实时
                编码视频。
            encoder_queue_maxsize: 流式编码时每个相机缓冲的最大帧数。
            token: 当需要从 Hub 下载元数据时使用的认证令牌。
                令牌不会保留在数据集实例上。

        Returns:
            一个处于写入模式、可随时追加 episode 的
            :class:`LeRobotDataset`。
        """
        if not root:
            raise ValueError(
                "resume() requires an explicit 'root' directory because it creates a DatasetWriter. "
                "Writing into the revision-safe Hub snapshot cache (used when root=None) would corrupt "
                "the shared cache. Please provide a local directory path."
            )
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj._requested_root = Path(root)
        obj.revision = revision if revision else CODEBASE_VERSION
        obj.tolerance_s = tolerance_s
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.episodes = None
        obj._video_backend = video_backend if video_backend else get_safe_default_video_backend()
        obj._return_uint8 = False
        obj._depth_output_unit = DEFAULT_DEPTH_UNIT
        obj._batch_encoding_size = batch_encoding_size

        if obj._requested_root is not None:
            obj._requested_root.mkdir(exist_ok=True, parents=True)

        # 加载元数据（未提供 root 时对版本安全）
        obj.meta = LeRobotDatasetMetadata(
            obj.repo_id,
            obj._requested_root,
            obj.revision,
            force_cache_sync=force_cache_sync,
            token=token,
        )

        obj._encoder_threads = encoder_threads
        obj._storage_root = None
        obj.root = obj.meta.root

        # 读取器在首次访问时惰性创建（只写模式）
        obj.reader = None

        streaming_enc = None
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            streaming_enc = cls._build_streaming_encoder(
                obj.meta.fps, rgb_encoder, depth_encoder, encoder_queue_maxsize, encoder_threads
            )
        obj.writer = DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            encoder_threads=encoder_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoder=streaming_enc,
            initial_frames=obj.meta.total_frames,
        )

        if image_writer_processes or image_writer_threads:
            obj.writer.start_image_writer(image_writer_processes, image_writer_threads)

        obj._is_finalized = False

        return obj
