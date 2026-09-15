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
import glob
import importlib
import logging
import os
import queue
import shutil
import tempfile
import threading
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO, ClassVar

import av
import fsspec
import numpy as np
import pyarrow as pa
import torch
from datasets.features.features import register_feature
from PIL import Image

from lerobot.configs import (
    DepthEncoderConfig,
    RGBEncoderConfig,
    VideoEncoderConfig,
    depth_encoder_defaults,
    rgb_encoder_defaults,
)
from lerobot.utils.import_utils import get_safe_default_video_backend

from .depth_utils import quantize_depth
from .pyav_utils import get_pix_fmt_channels

logger = logging.getLogger(__name__)


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
    return_uint8: bool = False,
    is_depth: bool = False,
) -> torch.Tensor:
    """
    使用指定后端解码视频帧。

    Args:
        video_path (Path): 视频文件的路径。
        timestamps (list[float]): 要提取帧的时间戳列表。
        tolerance_s (float): 取帧时允许的偏差（秒）。
        backend (str, optional): 用于解码的后端。平台上可用时默认为
            "torchcodec"；否则默认为 "pyav"。遗留值 "video_reader"
            会作为 "pyav" 的别名被接受一个版本，并将在未来版本中移除。
        return_uint8 (bool): 对于 RGB 视频，若为 True 则返回原始 uint8 帧，不做 float32 归一化。
            这可以减少 DataLoader IPC 的内存占用；之后可在 GPU 上进行归一化。
        is_depth (bool): 如果视频是深度图（1 通道，uint12），设为 True。

    Returns:
        torch.Tensor: 解码后的帧（RGB：默认为 [0,1] 范围的 float32，若 return_uint8=True 则为 uint8；深度：uint12）。

    目前支持 CPU 上的 torchcodec 以及 pyav。
    """
    if backend != "pyav" and is_depth:
        logger.debug("Decoding depth maps is only supported with the 'pyav' backend, falling back to pyav.")
        # 这里实际上并不返回 uint8，但避免了 255 归一化步骤。
        return decode_video_frames_pyav(
            video_path, timestamps, tolerance_s, return_uint8=False, is_depth=True
        )

    if backend is None:
        backend = get_safe_default_video_backend()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s, return_uint8=return_uint8)
    elif backend == "pyav":
        return decode_video_frames_pyav(
            video_path, timestamps, tolerance_s, return_uint8=return_uint8, is_depth=is_depth
        )
    elif backend == "video_reader":
        logger.warning("backend='video_reader' is deprecated and now aliases to 'pyav'.")
        return decode_video_frames_pyav(
            video_path, timestamps, tolerance_s, return_uint8=return_uint8, is_depth=is_depth
        )
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_pyav(
    video_path: Path | str | BinaryIO,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    return_uint8: bool = False,
    is_depth: bool = False,
) -> torch.Tensor:
    """使用 PyAV 加载视频中与所请求时间戳对应的帧。

    这是 torchcodec 没有提供 wheel 的平台上的回退解码器（目前包括
    macOS x86_64 和 linux armv7l——完整矩阵见 pyproject.toml 中的
    torchcodec 部分）。在受支持的平台上，优先使用
    `decode_video_frames_torchcodec`，它速度更快且支持精确寻址。

    PyAV 不支持精确寻址：我们会寻址到最近的前一个关键帧，然后
    向前解码，直到覆盖所请求的时间戳范围。视频中的关键帧数量
    可以在编码时调整，以在解码速度和文件大小之间权衡。

    Args:
        video_path: 视频文件的路径，或支持寻址的二进制类文件对象
            （支持 ``read``/``seek``）——例如带缓冲的远端数据源。
        timestamps: 要提取帧的时间戳列表（秒）。
        tolerance_s: 查询时间戳与最近的解码帧之间允许的偏差（秒）。
        log_loaded_timestamps: 为 True 时，以 INFO 级别记录每个解码帧的时间戳。
        return_uint8: 对于 RGB 视频，若为 True 则返回原始 uint8 帧（C, H, W）。
            否则返回 [0, 1] 范围的 float32。
        is_depth: 如果视频是深度图（1 通道，uint12），设为 True。

    Returns:
        形状为 (len(timestamps), C, H, W) 的 torch.Tensor。
    """
    # TODO(rcadene): 同时加载音频流
    if isinstance(video_path, (str, Path)):
        video_path = str(video_path)
    # else：类文件对象（例如带缓冲的远端数据源）原样传给 av.open。

    # 设置首个和最后一个请求的时间戳
    # 注意：之前的时间戳通常也会被加载，因为我们需要访问前一个关键帧
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []

    # 寻址 + 解码。不带 `stream` 参数的 `container.seek(offset)` 要求 offset
    # 以 av.time_base 为单位（微秒）。`backward=True` 会让我们落到 `first_ts`
    # 处或之前最近的关键帧上，这样随后就可以向前解码直到覆盖 `last_ts`。参见：
    # https://pyav.basswood-io.com/docs/stable/api/container.html#av.container.InputContainer.seek
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        # 寻址到 `first_ts` 处或之前最近的关键帧，留有 1 帧的余量
        container.seek(
            round(first_ts / stream.time_base) - 1,
            backward=True,
            any_frame=False,
            stream=stream,
        )

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * stream.time_base)
            if log_loaded_timestamps:
                logger.info(f"frame loaded at timestamp={current_ts:.4f}")
            if is_depth:
                arr = frame.to_ndarray(format="gray12le")  # (H, W) uint12
                loaded_frames.append(torch.from_numpy(arr).unsqueeze(0).contiguous())
            else:
                arr = frame.to_ndarray(format="rgb24")  # (H, W, 3)
                # 转换为 CHW 的 uint8，以匹配 torchcodec 的输出布局。
                loaded_frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

    if not loaded_frames:
        raise FrameTimestampError(
            f"No frames could be decoded from {video_path} in the timestamp range [{first_ts}, {last_ts}]."
        )

    # float64：小时级时间戳在 float32 下的量化误差会超过 tolerance_s。
    query_ts = torch.tensor(timestamps, dtype=torch.float64)
    loaded_ts_t = torch.tensor(loaded_ts, dtype=torch.float64)

    # 计算每个查询时间戳与所有已加载帧时间戳之间的距离
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ <= tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} >= {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts_t}"
            f"\nvideo: {video_path}"
            f"\nbackend: pyav"
        )

    # 获取与查询时间戳最接近的帧
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts_t[argmin_]

    if log_loaded_timestamps:
        logger.info(f"{closest_ts=}")

    if len(timestamps) != len(closest_frames):
        raise FrameTimestampError(
            f"Number of retrieved frames ({len(closest_frames)}) does not match "
            f"number of queried timestamps ({len(timestamps)})"
        )

    if return_uint8 or is_depth:
        return closest_frames

    # 转换为 pytorch 格式，即 [0,1] 范围的 float32（通道优先）
    closest_frames = closest_frames.type(torch.float32) / 255
    return closest_frames


DEFAULT_DECODER_CACHE_SIZE = 100
""":class:`VideoDecoderCache` 的默认 LRU 容量。

该大小足以从容容纳一个小规模滚动窗口内、相当于若干 episode 的解码器
（典型配置：每个 episode 2-4 个相机 × 同时在处理的数十个 episode），
同时限制主机内存的增长。每个缓存条目都会持有一个 torchcodec
``VideoDecoder`` 以及一个打开的 ``fsspec`` 文件句柄——每个条目
大约几 MB。可通过 ``LEROBOT_VIDEO_DECODER_CACHE_SIZE`` 环境变量
覆盖，也可向构造函数传入 ``max_size``
（``None`` 恢复为旧的无界行为）。
"""


def _default_max_cache_size() -> int | None:
    raw = os.environ.get("LEROBOT_VIDEO_DECODER_CACHE_SIZE")
    if raw is None:
        return DEFAULT_DECODER_CACHE_SIZE
    raw = raw.strip().lower()
    if raw in ("", "none", "unbounded", "-1"):
        return None
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"LEROBOT_VIDEO_DECODER_CACHE_SIZE must be an integer, 'none', or '-1'; got {raw!r}"
        ) from e
    if value <= 0:
        raise ValueError(f"LEROBOT_VIDEO_DECODER_CACHE_SIZE must be positive; got {value}")
    return value


class VideoDecoderCache:
    """用于 torchcodec ``VideoDecoder`` 实例的线程安全 LRU 缓存。

    缓存条目持有一个 ``VideoDecoder`` 以及支撑它的、已打开的
    ``fsspec`` 文件句柄。当缓存已满又请求新路径时，最近最少使用的
    条目会被驱逐，其文件句柄会被关闭。这可以在迭代包含大量不同
    视频文件的数据集时限制主机内存的增长（否则每个 ``DataLoader``
    worker 会一直占用它曾打开过的每个解码器，直到进程退出）。

    Args:
        max_size: 保留的最大解码器数量。``None`` 禁用
            驱逐并恢复旧的无界行为。默认取
            ``LEROBOT_VIDEO_DECODER_CACHE_SIZE`` 的值（若已设置），
            否则取 :data:`DEFAULT_DECODER_CACHE_SIZE`。
    """

    _SENTINEL: ClassVar[object] = object()

    def __init__(self, max_size: int | None | object = _SENTINEL):
        if max_size is VideoDecoderCache._SENTINEL:
            max_size = _default_max_cache_size()
        if max_size is not None and max_size <= 0:
            raise ValueError(f"max_size must be positive or None; got {max_size}")
        self.max_size: int | None = max_size  # type: ignore[assignment]
        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._lock = Lock()

    def __contains__(self, video_path: object) -> bool:
        with self._lock:
            return str(video_path) in self._cache

    def get_decoder(self, video_path: str):
        """获取缓存的解码器或创建新的；达到容量时按 LRU 驱逐。"""
        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError(
                "'torchcodec' is required but not installed. "
                "Install it with: pip install 'lerobot[dataset]' (or uv pip install 'lerobot[dataset]')"
            )

        video_path = str(video_path)

        with self._lock:
            entry = self._cache.get(video_path)
            if entry is not None:
                self._cache.move_to_end(video_path)
                return entry[0]

            file_handle = fsspec.open(video_path).__enter__()
            try:
                decoder = VideoDecoder(file_handle, seek_mode="approximate")
            except Exception:
                file_handle.close()
                raise
            self._cache[video_path] = (decoder, file_handle)

            # 驱逐 LRU 条目，直到回到容量上限以下。我们立即关闭
            # 被驱逐条目的文件句柄；关联的 ``VideoDecoder``
            # 会在其最后一个引用消失时交由 GC 回收。
            if self.max_size is not None:
                while len(self._cache) > self.max_size:
                    _evicted_path, (_evicted_decoder, evicted_handle) = self._cache.popitem(last=False)
                    with contextlib.suppress(Exception):
                        evicted_handle.close()

            return decoder

    def clear(self):
        """清空缓存并关闭所有文件句柄。"""
        with self._lock:
            for _, file_handle in self._cache.values():
                with contextlib.suppress(Exception):
                    file_handle.close()
            self._cache.clear()

    def size(self) -> int:
        """返回已缓存解码器的数量。"""
        with self._lock:
            return len(self._cache)


class FrameTimestampError(ValueError):
    """辅助错误，用于表示取回的时间戳超出查询的时间戳"""

    pass


_default_decoder_cache = VideoDecoderCache()


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    decoder_cache: VideoDecoderCache | None = None,
    return_uint8: bool = False,
) -> torch.Tensor:
    """使用 torchcodec 加载视频中与所请求时间戳对应的帧。

    Args:
        video_path: 视频文件的路径。
        timestamps: 要提取帧的时间戳列表。
        tolerance_s: 取帧时允许的偏差（秒）。
        log_loaded_timestamps: 是否记录已加载的时间戳。
        decoder_cache: 可选的解码器缓存实例。为 None 时使用默认缓存。

    Note: 在主进程之外（例如数据加载器 worker 中）设置 device="cuda" 会导致 CUDA 初始化错误。

    Note: 视频受益于帧间压缩。编码器并不会单独存储每一帧，
    而是存储一个参考帧（即关键帧），后续帧以相对于该关键帧的
    差异形式存储。因此，要访问所请求的帧，我们需要加载其前面的
    关键帧，以及之后直到所请求帧的所有帧。视频中的关键帧数量
    可以在编码时调整，以兼顾解码时间和视频的字节大小。
    """
    if decoder_cache is None:
        decoder_cache = _default_decoder_cache

    # 使用缓存的解码器，而不是每次都创建新的
    decoder = decoder_cache.get_decoder(str(video_path))

    loaded_ts = []
    loaded_frames = []

    # 获取元数据以了解帧信息
    metadata = decoder.metadata
    average_fps = metadata.average_fps
    # 将时间戳转换为帧索引
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    # 根据索引取回帧
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=True):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logger.info(f"Frame loaded at timestamp={pts:.4f}")

    # float64：小时级时间戳在 float32 下的量化误差会超过 tolerance_s。
    query_ts = torch.tensor(timestamps, dtype=torch.float64)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float64)

    # 计算每个查询时间戳与已加载时间戳之间的距离
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ <= tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} >= {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
        )

    # 获取与查询时间戳最接近的帧
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logger.info(f"{closest_ts=}")

    if not len(timestamps) == len(closest_frames):
        raise FrameTimestampError(
            f"Retrieved timestamps differ from queried {set(closest_frames) - set(timestamps)}"
        )

    if return_uint8:
        return closest_frames

    # 转换为 [0,1] 范围的 float32
    closest_frames = (closest_frames / 255.0).type(torch.float32)
    return closest_frames


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    video_encoder: VideoEncoderConfig | None = None,
    encoder_threads: int | None = None,
    *,
    log_level: int | None = av.logging.WARNING,
    overwrite: bool = False,
) -> None:
    """将一个图像帧目录编码为 MP4 视频。

    当 ``video_encoder`` 是
    :class:`~lerobot.configs.video.DepthEncoderConfig` 时，帧会从
    ``.tiff`` 文件读取，并使用编码器的 ``depth_min`` / ``depth_max`` /
    ``shift`` / ``use_log`` 量化为 12 位深度码；否则直接编码
    ``.png`` RGB 帧。

    Args:
        imgs_dir: 包含待编码帧的目录，帧从 ``frame-000000``
            开始命名（RGB 为 ``.png``，深度为 ``.tiff``）。
        video_path: 编码后 ``.mp4`` 文件的输出路径。
        fps: 输出视频的帧率。
        video_encoder: 编码器设置（编解码器、像素格式、质量……）。为
            ``None`` 时使用 :func:`rgb_encoder_defaults`。传入
            :class:`~lerobot.configs.video.DepthEncoderConfig` 可编码深度帧。
        encoder_threads: 转发给编解码器的单编码器线程数。``None``
            表示由编解码器决定。
        log_level: 编码期间设置的 libav 日志级别；``None`` 表示保持
            当前日志配置不变。
        overwrite: 为 ``False`` 且 ``video_path`` 已存在时，跳过编码并
            记录一条警告。为 ``True`` 时重新编码并替换已有文件。
    """
    if video_encoder is None:
        video_encoder = rgb_encoder_defaults()
    vcodec = video_encoder.vcodec
    pix_fmt = video_encoder.pix_fmt

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    if video_path.exists() and not overwrite:
        logger.warning(f"Video file already exists: {video_path}. Skipping encoding.")
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取输入帧
    is_depth = isinstance(video_encoder, DepthEncoderConfig)
    suffix = ".png" if not is_depth else ".tiff"
    template = "frame-" + ("[0-9]" * 6) + suffix
    input_list = sorted(
        glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("-")[-1].split(".")[0])
    )

    if len(input_list) == 0:
        raise FileNotFoundError(f"No images with suffix {suffix} found in {imgs_dir}.")
    with Image.open(input_list[0]) as dummy_image:
        width, height = dummy_image.size

    video_options = video_encoder.get_codec_options(encoder_threads, as_strings=True)

    # 设置日志级别
    if log_level is not None:
        # “虽然效率较低，但通常更推荐使用 Python 的 logging 来修改日志配置”
        logging.getLogger("libav").setLevel(log_level)

    # 创建并打开输出文件（默认覆盖）
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # 遍历输入帧并进行编码
        for input_data in input_list:
            with Image.open(input_data) as input_image:
                if is_depth:
                    input_frame = quantize_depth(
                        np.array(input_image),
                        depth_min=video_encoder.depth_min,
                        depth_max=video_encoder.depth_max,
                        shift=video_encoder.shift,
                        use_log=video_encoder.use_log,
                        pix_fmt=video_encoder.pix_fmt,
                        video_backend="pyav",
                    )
                else:
                    input_image = input_image.convert("RGB")
                    input_frame = av.VideoFrame.from_image(input_image)
                packet = output_stream.encode(input_frame)
                if packet:
                    output.mux(packet)

        # 刷新编码器
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # 重置日志级别
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def reencode_video(
    input_video_path: Path | str,
    output_video_path: Path | str,
    video_encoder: VideoEncoderConfig | None = None,
    encoder_threads: int | None = None,
    log_level: int | None = av.logging.WARNING,
    overwrite: bool = False,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> None:
    """重新编码视频文件，可选地将其裁剪到 ``[start_time_s, end_time_s)``。

    Args:
        input_video_path: 要读取的已有视频文件。
        output_video_path: 重新编码后文件的路径。
        video_encoder: 编码器配置。默认为 :func:`rgb_encoder_defaults`。
        encoder_threads: 可选的线程数，转发给 :meth:`VideoEncoderConfig.get_codec_options`。
        log_level: 编码期间的 libav 日志级别；``None`` 表示保持日志配置不变。默认为 WARNING。
        overwrite: 为 ``False`` 且 ``output_video_path`` 已存在时，跳过并记录警告。
        start_time_s: 设置后，将输出裁剪为从该时间戳（秒）开始。
        end_time_s: 设置后，将输出裁剪为在该时间戳（秒，不含）结束。
    """

    video_encoder = video_encoder or rgb_encoder_defaults()

    if (start_time_s is not None and start_time_s < 0) or (end_time_s is not None and end_time_s < 0):
        raise ValueError(f"Trim times must be non-negative, got start={start_time_s}, end={end_time_s}.")
    if start_time_s is not None and end_time_s is not None and end_time_s <= start_time_s:
        raise ValueError(f"end_time_s ({end_time_s}) must be greater than start_time_s ({start_time_s}).")

    output_video_path = Path(output_video_path)

    if output_video_path.exists() and not overwrite:
        logger.warning(f"Video file already exists: {output_video_path}. Skipping re-encode.")
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    video_options = video_encoder.get_codec_options(encoder_threads, as_strings=True)
    vcodec = video_encoder.vcodec
    pix_fmt = video_encoder.pix_fmt

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_named_file:
        tmp_output_video_path = tmp_named_file.name

    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)

    try:
        with av.open(input_video_path, mode="r") as src:
            try:
                in_stream = src.streams.video[0]
            except IndexError as e:
                raise ValueError(f"No video stream in {input_video_path}") from e

            fps = (
                in_stream.base_rate
            )  # 我们允许分数 fps，尽管 LeRobotDataset 只支持整数 fps
            width = int(in_stream.width)
            height = int(in_stream.height)

            # 寻址到 start_time_s 处或之前的关键帧，避免从头开始读取。
            if start_time_s is not None:
                src.seek(int(start_time_s * av.time_base), backward=True)

            with av.open(
                tmp_output_video_path,
                mode="w",
                options={
                    "movflags": "faststart"
                },  # faststart 用于将元数据移到文件开头，以加快加载速度
            ) as dst:
                out_stream = dst.add_stream(vcodec, fps, options=video_options)
                out_stream.pix_fmt = pix_fmt
                out_stream.width = width
                out_stream.height = height

                for frame in src.decode(in_stream):
                    frame_time_s = frame.time
                    if start_time_s is not None and frame_time_s < start_time_s:
                        continue
                    if end_time_s is not None and frame_time_s >= end_time_s:
                        break
                    frame = frame.reformat(width=width, height=height, format=pix_fmt)
                    if start_time_s is not None:
                        frame.pts = None  # 重置时间戳，使裁剪后的输出从 t=0 开始
                    packet = out_stream.encode(frame)
                    if packet:
                        dst.mux(packet)

                packet = out_stream.encode()
                if packet:
                    dst.mux(packet)

        shutil.move(tmp_output_video_path, output_video_path)
    except Exception:
        Path(tmp_output_video_path).unlink(missing_ok=True)
        raise
    finally:
        if log_level is not None:
            av.logging.restore_default_callback()

    if not output_video_path.exists():
        raise OSError(f"Video re-encoding did not work. File not found: {output_video_path}.")


def concatenate_video_files(
    input_video_paths: list[Path | str],
    output_video_path: Path,
    overwrite: bool = True,
    compatibility_check: bool = False,
):
    """
    使用 pyav 将多个视频文件拼接为单个视频文件。

    本函数接收一个视频输入文件路径列表，并将它们拼接为单个
    输出视频文件。它使用 ffmpeg 的 concat demuxer 和流拷贝模式进行
    快速拼接，无需重新编码。

    Args:
        input_video_paths: 待拼接的输入视频文件路径的有序列表。
        output_video_path: 输出视频文件的路径。
        overwrite: 输出视频文件已存在时是否覆盖。默认为 True。
        compatibility_check: 是否检查输入视频之间是否兼容。默认为 False。

    Note:
        - 会创建一个临时目录存放中间文件，使用后会清理。
        - 使用 ffmpeg 的 concat demuxer，要求所有输入视频具有相同的
          编解码器、分辨率和帧率，才能正确拼接。
    """

    output_video_path = Path(output_video_path)

    if output_video_path.exists() and not overwrite:
        logger.warning(f"Video file already exists: {output_video_path}. Skipping concatenation.")
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if len(input_video_paths) == 0:
        raise FileNotFoundError("No input video paths provided.")

    # 录制时可以跳过此检查，因为视频都是用相同的编码器配置编码的。
    if compatibility_check:
        reference_video_info = get_video_info(input_video_paths[0])
        for input_path in input_video_paths[1:]:
            video_info = get_video_info(input_path)
            if (
                video_info["video.height"] != reference_video_info["video.height"]
                or video_info["video.width"] != reference_video_info["video.width"]
                or video_info["video.fps"] != reference_video_info["video.fps"]
                or video_info["video.codec"] != reference_video_info["video.codec"]
                or video_info["video.pix_fmt"] != reference_video_info["video.pix_fmt"]
            ):
                raise ValueError(
                    f"Input video {input_path} is not compatible with the reference video {input_video_paths[0]}."
                )

    # 创建临时 .ffconcat 文件以列出输入视频路径
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as tmp_concatenate_file:
        tmp_concatenate_file.write("ffconcat version 1.0\n")
        for input_path in input_video_paths:
            tmp_concatenate_file.write(f"file '{str(input_path.resolve())}'\n")
        tmp_concatenate_file.flush()
        tmp_concatenate_path = tmp_concatenate_file.name

    # 创建输入和输出容器
    input_container = av.open(
        tmp_concatenate_path, mode="r", format="concat", options={"safe": "0"}
    )  # safe = 0 同时允许绝对路径和相对路径

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_named_file:
        tmp_output_video_path = tmp_named_file.name

    output_container = av.open(
        tmp_output_video_path, mode="w", options={"movflags": "faststart"}
    )  # faststart 用于将元数据移到文件开头，以加快加载速度

    # 在输出容器中复制输入流
    stream_map = {}
    for input_stream in input_container.streams:
        if input_stream.type in ("video", "audio", "subtitle"):  # 只复制兼容的流
            stream_map[input_stream.index] = output_container.add_stream_from_template(
                template=input_stream, opaque=True
            )

            # 将时间基设置为输入流的时间基（编解码器上下文中缺少该信息）
            stream_map[input_stream.index].time_base = input_stream.time_base

    # 解复用 + 重新复用数据包（不重新编码）
    for packet in input_container.demux():
        # 跳过来自未映射流的数据包
        if packet.stream.index not in stream_map:
            continue

        # 跳过解复用刷新时的数据包
        if packet.dts is None:
            continue

        output_stream = stream_map[packet.stream.index]
        packet.stream = output_stream
        output_container.mux(packet)

    input_container.close()
    output_container.close()
    shutil.move(tmp_output_video_path, output_video_path)
    Path(tmp_concatenate_path).unlink()


class _CameraEncoderThread(threading.Thread):
    """一个将通过队列流式传入的视频帧编码为 MP4 文件的线程。

    每个相机、每个 episode 创建一个实例。帧以 numpy 数组的形式
    从主线程接收，使用 PyAV 实时编码（编码期间会释放 GIL），
    并写入磁盘。统计信息使用
    RunningQuantileStats 增量计算，并通过 result_queue 返回。
    """

    def __init__(
        self,
        video_path: Path,
        fps: int,
        video_encoder: VideoEncoderConfig,
        frame_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
        encoder_threads: int | None = None,
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.fps = fps
        self.video_encoder = video_encoder
        self.is_depth = isinstance(video_encoder, DepthEncoderConfig)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.encoder_threads = encoder_threads

    def run(self) -> None:
        from .compute_stats import RunningQuantileStats, auto_downsample_height_width

        container = None
        output_stream = None
        stats_tracker = RunningQuantileStats()
        frame_count = 0

        try:
            logging.getLogger("libav").setLevel(av.logging.WARNING)

            while True:
                try:
                    frame_data = self.frame_queue.get(timeout=1)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue

                if frame_data is None:
                    # 哨兵值：刷新并关闭
                    break

                # 确保是 HWC（RGB 或深度）的 numpy 数组，RGB 为 uint8
                if isinstance(frame_data, np.ndarray):
                    if frame_data.ndim == 3 and frame_data.shape[0] in (1, 3):
                        # CHW -> HWC
                        frame_data = frame_data.transpose(1, 2, 0)
                    if not self.is_depth and frame_data.dtype != np.uint8:
                        frame_data = (frame_data * 255).astype(np.uint8)

                # 在第一帧时打开容器（以获取宽度/高度）
                if container is None:
                    height, width = frame_data.shape[:2]
                    Path(self.video_path).parent.mkdir(parents=True, exist_ok=True)
                    container = av.open(str(self.video_path), "w")
                    output_stream = container.add_stream(
                        self.video_encoder.vcodec,
                        self.fps,
                        options=self.video_encoder.get_codec_options(self.encoder_threads, as_strings=True),
                    )
                    output_stream.pix_fmt = self.video_encoder.pix_fmt
                    output_stream.width = width
                    output_stream.height = height
                    output_stream.time_base = Fraction(1, self.fps)

                # 使用显式时间戳编码帧
                if not self.is_depth:
                    pil_img = Image.fromarray(frame_data)
                    video_frame = av.VideoFrame.from_image(pil_img)
                else:
                    video_frame = quantize_depth(
                        frame_data,
                        depth_min=self.video_encoder.depth_min,
                        depth_max=self.video_encoder.depth_max,
                        shift=self.video_encoder.shift,
                        use_log=self.video_encoder.use_log,
                        video_backend=self.video_encoder.video_backend,
                    )
                video_frame.pts = frame_count
                video_frame.time_base = Fraction(1, self.fps)
                packet = output_stream.encode(video_frame)
                if packet:
                    container.mux(packet)

                # 用下采样后的帧更新统计信息（逐通道统计，与 compute_episode_stats 一致）
                img_chw = frame_data.transpose(2, 0, 1)  # HWC -> CHW
                img_downsampled = auto_downsample_height_width(img_chw)
                # 将 CHW 重塑为 (H*W, C)，用于逐通道统计
                channels = img_downsampled.shape[0]
                img_for_stats = img_downsampled.transpose(1, 2, 0).reshape(-1, channels)
                stats_tracker.update(img_for_stats)

                frame_count += 1

            # 刷新编码器
            if output_stream is not None:
                packet = output_stream.encode()
                if packet:
                    container.mux(packet)

            if container is not None:
                container.close()

            av.logging.restore_default_callback()

            # 获取统计信息并放入结果队列
            if frame_count >= 2:
                stats = stats_tracker.get_statistics()
                self.result_queue.put(("ok", stats))
            else:
                self.result_queue.put(("ok", None))

        except Exception as e:
            logger.error(f"Encoder thread error: {e}")
            if container is not None:
                with contextlib.suppress(Exception):
                    container.close()
            self.result_queue.put(("error", str(e)))


class StreamingVideoEncoder:
    """管理每个相机的编码器线程，用于录制期间的实时视频编码。

    本类不再先将帧写为 PNG 图像、在 episode 结束时再编码为
    MP4，而是将帧直接流式发送给编码器线程，省去了 PNG 的
    往返过程，使 save_episode() 几乎瞬时完成。

    使用线程而非多进程，以避免通过 multiprocessing.Queue
    对大型 numpy 数组进行 pickle 的开销。PyAV 的 encode() 会释放
    GIL，因此编码可以与主录制循环并行执行。
    """

    def __init__(
        self,
        fps: int,
        rgb_encoder: RGBEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ):
        """
        Args:
            fps: 输出视频的每秒帧数。
            rgb_encoder: 应用于所有 RGB 相机的视频编码器设置。
                为 ``None`` 时使用 :func:`rgb_encoder_defaults`。
            depth_encoder: 应用于所有深度相机的视频编码器设置，
                包括深度量化参数。为 ``None`` 时使用
                :func:`depth_encoder_defaults`。
            queue_maxsize: 背压机制丢弃帧之前，每个相机
                最多缓冲的帧数。
            encoder_threads: 编码器线程数（全局设置）。
                ``None`` 表示由编解码器决定。
        """
        self.fps = fps
        self._rgb_encoder = rgb_encoder or rgb_encoder_defaults()
        self._depth_encoder = depth_encoder or depth_encoder_defaults()
        self._encoder_threads = encoder_threads
        self.queue_maxsize = queue_maxsize

        self._frame_queues: dict[str, queue.Queue] = {}
        self._result_queues: dict[str, queue.Queue] = {}
        self._threads: dict[str, _CameraEncoderThread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._video_paths: dict[str, Path] = {}
        self._dropped_frames: dict[str, int] = {}
        self._episode_active = False
        self._closed = False

    def start_episode(
        self, video_keys: list[str], temp_dir: Path, depth_video_keys: list[str] | None = None
    ) -> None:
        """为新 episode 启动编码器线程。

        Args:
            video_keys: 视频特征键列表（例如 ["observation.images.laptop"]）
            temp_dir: 临时 MP4 文件的根目录
            depth_video_keys: 携带深度图的视频或图像特征键列表（例如
                ["observation.images.laptop_depth"]）。默认为 ``[]``（无深度键）。
        """
        if self._episode_active:
            self.cancel_episode()

        self._dropped_frames.clear()

        if depth_video_keys is None:
            depth_video_keys = []

        for video_key in video_keys:
            frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_maxsize)
            result_queue: queue.Queue = queue.Queue(maxsize=1)
            stop_event = threading.Event()

            temp_video_dir = Path(tempfile.mkdtemp(dir=temp_dir))
            video_path = temp_video_dir / f"{video_key.replace('/', '_')}_streaming.mp4"

            encoder = self._depth_encoder if video_key in depth_video_keys else self._rgb_encoder
            encoder_thread = _CameraEncoderThread(
                video_path=video_path,
                fps=self.fps,
                video_encoder=encoder,
                frame_queue=frame_queue,
                result_queue=result_queue,
                stop_event=stop_event,
                encoder_threads=self._encoder_threads,
            )
            encoder_thread.start()

            self._frame_queues[video_key] = frame_queue
            self._result_queues[video_key] = result_queue
            self._threads[video_key] = encoder_thread
            self._stop_events[video_key] = stop_event
            self._video_paths[video_key] = video_path

        self._episode_active = True

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        """向特定相机的编码器送入一帧。

        入队前会复制图像，以防止与可能复用缓冲区的相机驱动
        发生竞态条件。如果编码器队列已满（编码器跟不上），
        会丢弃该帧并发出警告，而不是导致录制会话崩溃。

        Args:
            video_key: 视频特征键
            image: (H,W,C) 或 (C,H,W) 格式的 numpy 数组，uint8 或 float

        Raises:
            RuntimeError: 当编码器线程已崩溃时
        """
        if not self._episode_active:
            raise RuntimeError("No active episode. Call start_episode() first.")

        thread = self._threads[video_key]
        if not thread.is_alive():
            # 检查是否有错误
            try:
                status, msg = self._result_queues[video_key].get_nowait()
                if status == "error":
                    raise RuntimeError(f"Encoder thread for {video_key} crashed: {msg}")
            except queue.Empty:
                pass
            raise RuntimeError(f"Encoder thread for {video_key} is not alive")

        try:
            self._frame_queues[video_key].put(image.copy(), timeout=0.1)
        except queue.Full:
            self._dropped_frames[video_key] = self._dropped_frames.get(video_key, 0) + 1
            count = self._dropped_frames[video_key]
            # 定期记录日志以避免刷屏（第 1 次，之后每 10 次）
            if count == 1 or count % 10 == 0:
                logger.warning(
                    f"Encoder queue full for {video_key}, dropped {count} frame(s). "
                    f"Consider using vcodec='auto' for hardware encoding or increasing encoder_queue_maxsize."
                )

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        """完成当前 episode 的编码。

        发送哨兵值，等待编码器线程完成，
        并收集结果。

        Returns:
            将 video_key 映射到 (mp4_path, stats_dict_or_None) 的字典
        """
        if not self._episode_active:
            raise RuntimeError("No active episode to finish.")

        results = {}

        # 上报丢弃的帧
        for video_key, count in self._dropped_frames.items():
            if count > 0:
                logger.warning(f"Episode finished with {count} dropped frame(s) for {video_key}.")

        # 向所有队列发送哨兵值
        for video_key in self._frame_queues:
            self._frame_queues[video_key].put(None)

        # 等待所有线程并收集结果
        for video_key in self._threads:
            self._threads[video_key].join(timeout=120)
            if self._threads[video_key].is_alive():
                logger.error(f"Encoder thread for {video_key} did not finish in time")
                self._stop_events[video_key].set()
                self._threads[video_key].join(timeout=5)
                results[video_key] = (self._video_paths[video_key], None)
                continue

            try:
                status, data = self._result_queues[video_key].get(timeout=5)
                if status == "error":
                    raise RuntimeError(f"Encoder thread for {video_key} failed: {data}")
                results[video_key] = (self._video_paths[video_key], data)
            except queue.Empty:
                logger.error(f"No result from encoder thread for {video_key}")
                results[video_key] = (self._video_paths[video_key], None)

        self._cleanup()
        self._episode_active = False
        return results

    def cancel_episode(self) -> None:
        """取消当前 episode，停止编码器线程并进行清理。"""
        if not self._episode_active:
            return

        # 通知所有线程停止
        for video_key in self._stop_events:
            self._stop_events[video_key].set()

        # 等待线程结束
        for video_key in self._threads:
            self._threads[video_key].join(timeout=5)

            # 清理临时 MP4 文件
            video_path = self._video_paths.get(video_key)
            if video_path is not None and video_path.exists():
                shutil.rmtree(str(video_path.parent), ignore_errors=True)

        self._cleanup()
        self._episode_active = False

    def close(self) -> None:
        """关闭编码器，取消任何正在进行的 episode。"""
        if self._closed:
            return
        if self._episode_active:
            self.cancel_episode()
        self._closed = True

    def _cleanup(self) -> None:
        """清理队列和线程跟踪字典。"""
        for q in self._frame_queues.values():
            with contextlib.suppress(Exception):
                while not q.empty():
                    q.get_nowait()
        self._frame_queues.clear()
        self._result_queues.clear()
        self._threads.clear()
        self._stop_events.clear()
        self._video_paths.clear()


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): 迁移到 Hugging Face `datasets` 仓库
    """
    为包含视频帧的数据集提供一个类型。

    示例：

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # 使 VideoFrame 在 HuggingFace `datasets` 中可用
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    # 设置日志级别
    logging.getLogger("libav").setLevel(av.logging.WARNING)

    # 获取音频流信息
    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:
            # 重置日志级别
            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name
        # 理想的无损情况下：位深 × 采样率 × 通道数 = 比特率。
        # 实际的压缩情况下：比特率根据压缩级别设定——比特率越低，压缩程度越高。
        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate  # 每秒采样数
        # 理想的无损情况下：每个采样占用固定的比特数。
        # 实际的压缩情况下：每个采样的比特数可变（通常会降低以匹配给定的比特率）。
        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True

    # 重置日志级别
    av.logging.restore_default_callback()

    return audio_info


def get_video_info(
    video_path: Path | str,
    video_encoder: VideoEncoderConfig | None = None,
) -> dict:
    """构建持久化到 ``info.json`` 中的 ``video.*`` / ``audio.*`` 信息字典。

    Args:
        video_path: 要探测的已编码视频文件的路径。
        video_encoder: 如果提供，则记录用于编码该视频的确切编码器
            设置。从流派生的值优先级更高——编码器字段只写入那些
            尚未从视频文件本身获取的键。当传入
            :class:`~lerobot.configs.video.DepthEncoderConfig` 时，会记录
            深度量化参数（``depth_min`` / ``depth_max`` / ``shift`` /
            ``use_log``），以便在读取时对帧反量化。

    Returns:
        ``video.*`` / ``audio.*`` 信息字典，包含 ``is_depth_map``，该字段
        仅当 ``video_encoder`` 是
        :class:`~lerobot.configs.video.DepthEncoderConfig` 时为
        ``True``。
    """
    logging.getLogger("libav").setLevel(av.logging.WARNING)

    # 获取视频流信息
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            # 重置日志级别
            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt

        # 根据 r_frame_rate 计算 fps
        video_info["video.fps"] = int(video_stream.base_rate)
        video_info["video.channels"] = get_pix_fmt_channels(video_stream.pix_fmt)

    # 重置日志级别
    av.logging.restore_default_callback()

    # 添加音频流信息
    video_info.update(**get_audio_info(video_path))

    # 如果提供了额外的编码器配置，则添加
    if video_encoder is not None:
        for field_name, field_value in asdict(video_encoder).items():
            # vcodec 已经从视频流中获取
            if field_name == "vcodec":
                continue
            video_info.setdefault(f"video.{field_name}", field_value)

    video_info["is_depth_map"] = isinstance(video_encoder, DepthEncoderConfig)

    return video_info


def get_video_duration_in_s(video_path: Path | str) -> float:
    """
    使用 PyAV 获取视频文件的时长（秒）。

    Args:
        video_path: 视频文件的路径。

    Returns:
        视频时长（秒）。
    """
    with av.open(str(video_path)) as container:
        # 获取第一个视频流
        video_stream = container.streams.video[0]
        # 计算时长：stream.duration * stream.time_base 即得到以秒为单位的时长
        if video_stream.duration is not None:
            duration = float(video_stream.duration * video_stream.time_base)
        else:
            # 当流时长不可用时，回退到容器时长
            duration = float(container.duration / av.time_base)
    return duration


class VideoEncodingManager:
    """
    上下文管理器，确保即使发生异常也能正确完成视频编码和数据清理。

    该管理器负责：
    - 录制中断时，对所有剩余 episode 进行批量编码
    - 清理被中断 episode 的临时图像文件
    - 删除空的图像目录

    Args:
        dataset: LeRobotDataset 实例
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        writer = self.dataset.writer
        if writer is not None:
            if exc_type is not None and writer._streaming_encoder is not None:
                writer.cancel_pending_videos()

            # finalize() 负责处理 flush_pending_videos + parquet + 元数据
            self.dataset.finalize()

            # 录制中断时清理 episode 图像（仅非流式模式）
            if exc_type is not None and writer._streaming_encoder is None:
                writer.cleanup_interrupted_episode(self.dataset.num_episodes)
        else:
            self.dataset.finalize()

        # 如果 images 目录仍然存在且为空，则清理它
        img_dir = self.dataset.root / "images"
        if img_dir.exists():
            png_files = list(img_dir.rglob("*.png"))
            tiff_files = list(img_dir.rglob("*.tiff"))
            if len(png_files) == 0 and len(tiff_files) == 0:
                shutil.rmtree(img_dir)
                logger.debug("Cleaned up empty images directory")
            else:
                logger.debug(
                    f"Images directory is not empty, containing {len(png_files)} PNG and {len(tiff_files)} TIFF files"
                )

        return False  # 不抑制原始异常
