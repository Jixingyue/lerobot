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
"""标注流水线的关键帧提取。

模块将解码后的摄像头帧附加到其 VLM 提示词，以便模型可以
基于实际视觉内容进行子任务分解、插入语场景和 VQA。
流水线在模块之间共享一个提供者，并且一次处理一个回合，
带有小型的按回合缓存，以便多个模块查询相同时间戳时只需支付一次解码成本。
"""

from __future__ import annotations

import io
import logging
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import PIL.Image
import torch

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.video_utils import decode_video_frames, reencode_video

from .reader import EpisodeRecord, snap_to_frame

logger = logging.getLogger(__name__)


class FrameProvider(Protocol):
    """在回合相对时间戳处解码摄像头帧。"""

    @property
    def camera_keys(self) -> list[str]:
        """此提供者可以解码的所有 ``observation.images.*`` 特征键。"""

    def frames_at(
        self,
        record: EpisodeRecord,
        timestamps: list[float],
        camera_key: str | None = None,
    ) -> list[Any]:
        """从 ``camera_key``（或默认）返回每个时间戳一个解码帧。

        帧是 ``torch.Tensor``（``C, H, W`` uint8）——
        :func:`lerobot.datasets.video_utils.decode_video_frames` 返回的形状。
        :func:`to_image_blocks` 仅在 VLM 消息边界处将它们转换为 PIL。

        如果摄像头不可用则返回空列表。``camera_key=None`` 回退到提供者的默认摄像头，
        以便现有的单摄像头调用者（``plan`` 和 ``interjections`` 模块）保持不变地工作。
        """

    def video_for_episode(
        self,
        record: EpisodeRecord,
        max_frames: int,
        camera_key: str | None = None,
    ) -> list[Any]:
        """返回覆盖整个回合的最多 ``max_frames`` 个解码帧。

        采样在回合持续时间内均匀分布。帧是
        ``torch.Tensor``（``C, H, W`` uint8）；:func:`to_video_block` 将它们
        包装成一个 ``{"type":"video", "video":<list>}`` 块，供自身进行时间池化的
        Qwen-VL 兼容模型使用。如果没有可用的摄像头则返回空列表。
        """


@dataclass
class _NullProvider:
    """当数据集没有视频键时或在测试中使用的空操作提供者。"""

    @property
    def camera_keys(self) -> list[str]:
        return []

    def frames_at(
        self,
        record: EpisodeRecord,
        timestamps: list[float],
        camera_key: str | None = None,
    ) -> list[Any]:
        return []

    def video_for_episode(
        self,
        record: EpisodeRecord,
        max_frames: int,
        camera_key: str | None = None,
    ) -> list[Any]:
        return []


def null_provider() -> FrameProvider:
    return _NullProvider()


@dataclass
class VideoFrameProvider:
    """从数据集的 ``observation.images.*`` 流中解码帧。

    默认情况下，*第一个*摄像头键用于 ``plan`` 模块（子任务分解）和
    ``interjections`` 模块（插入语场景）——这些提示词关心的是*正在发生什么*，
    而不是哪个角度。``vqa`` 模块则遍历 :attr:`camera_keys` 中的每个摄像头，
    以便每帧的基础答案（bbox/keypoint/...）都标记有它所基于的摄像头。

    ``camera_key`` 覆盖默认摄像头选择，但不限制 :attr:`camera_keys`。
    显式传递 ``camera_key`` 给 ``frames_at`` / ``video_for_episode`` 以读取非默认流。

    每个进程缓存最多 ``cache_size`` 个解码帧，以保持同时间戳的
    ``interjections`` + ``plan`` 计划更新调用的低成本。
    """

    root: Path
    camera_key: str | None = None
    tolerance_s: float = 1e-2
    cache_size: int = 256
    # 转发到 :func:`lerobot.datasets.video_utils.decode_video_frames` 的关键帧解码后端。
    # ``None`` 使用库默认值（torchcodec 可用时使用，否则 PyAV）。
    video_backend: str | None = None
    _meta: Any = field(default=None, init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)
    _camera_keys: list[str] = field(default_factory=list, init=False, repr=False)
    # 流水线在线程池执行器下运行三个模块阶段（参见
    # ``ExecutorConfig.episode_parallelism``）；保护字典缓存和一次性警告标志
    # 免受工作线程的并发更新。
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    # 序列化 decode_video_frames 调用：torchcodec 从进程级缓存中为每个文件分发一个
    # ``VideoDecoder``，而解码器不适合从多个线程同时驱动。
    _decode_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _warned_decode_fail: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: PLC0415

        self._meta = LeRobotDatasetMetadata(repo_id="local", root=self.root)
        # 这里只有 ``video_keys`` 是可解码的：剪辑/解码路径从回合元数据读取
        # ``videos/<key>/from_timestamp``，这仅存在于视频存储的摄像头中。
        # 图像存储的摄像头（也在 ``camera_keys`` 中）会引发 KeyError，因此将列表——
        # 以及默认值——限制为视频键。
        # 深度摄像头目前被排除在标注流水线之外。
        depth_keys = set(self._meta.depth_keys)
        keys = [key for key in self._meta.video_keys if key not in depth_keys]
        # 最后手段回退：如果元数据没有显示任何视频键，但调用者显式命名了一个摄像头
        # （``--vlm.camera_key=...``），则信任他们——该键根据定义已知存在于数据集上。
        if not keys and self.camera_key:
            keys = [self.camera_key]
        self._camera_keys = keys
        if self.camera_key is None:
            self.camera_key = keys[0] if keys else None

    @property
    def camera_keys(self) -> list[str]:
        """此数据集上可用的所有 ``observation.images.*`` 键。"""
        return list(self._camera_keys)

    def frames_at(
        self,
        record: EpisodeRecord,
        timestamps: list[float],
        camera_key: str | None = None,
    ) -> list[Any]:
        target = camera_key if camera_key is not None else self.camera_key
        if not timestamps or target is None:
            return []
        # 将每个请求对齐到最近的真实帧时间戳：调用者采样均匀网格，其点落在帧中间，
        # 而 ``decode_video_frames`` 拒绝距离可解码帧超过 ``tolerance_s`` 的查询。
        # 对齐还通过缓存对重复查询去重。
        if record.frame_timestamps:
            timestamps = [snap_to_frame(float(ts), record.frame_timestamps) for ts in timestamps]

        out: list[Any] = []
        misses: list[float] = []
        miss_indices: list[int] = []
        with self._lock:
            for i, ts in enumerate(timestamps):
                key = (record.episode_index, target, round(float(ts), 6))
                cached = self._cache.get(key)
                if cached is not None:
                    out.append(cached)
                else:
                    out.append(None)
                    misses.append(float(ts))
                    miss_indices.append(i)

        if misses:
            decoded = self._decode(record.episode_index, misses, target)
            # ``_decode`` 为每个请求的时间戳恰好返回一帧，或者如果解码完全失败则返回空列表。
            # 部分列表意味着帧/时间戳不对齐，因此仅在计数匹配时才配对
            # （``strict=True`` 随后防止回归）。
            if len(decoded) == len(miss_indices):
                with self._lock:
                    for i, frame in zip(miss_indices, decoded, strict=True):
                        out[i] = frame
                        key = (record.episode_index, target, round(float(timestamps[i]), 6))
                        if len(self._cache) >= self.cache_size:
                            self._cache.pop(next(iter(self._cache)))
                        self._cache[key] = frame
        # 过滤掉解码失败留下的任何 None
        return [frame for frame in out if frame is not None]

    def video_for_episode(
        self,
        record: EpisodeRecord,
        max_frames: int,
        camera_key: str | None = None,
    ) -> list[Any]:
        """返回在回合中均匀采样的最多 ``max_frames`` 帧。

        覆盖整个回合持续时间；模型从其内部进行的时间池化中选择子任务边界。
        帧是 ``torch.Tensor``（参见 :meth:`frames_at`）。
        """
        target = camera_key if camera_key is not None else self.camera_key
        if max_frames <= 0 or target is None or not record.frame_timestamps:
            return []
        n_frames = min(max_frames, len(record.frame_timestamps))
        if n_frames == len(record.frame_timestamps):
            timestamps = list(record.frame_timestamps)
        else:
            t0 = record.frame_timestamps[0]
            t_last = record.frame_timestamps[-1]
            if t_last <= t0:
                timestamps = [float(t0)] * n_frames
            else:
                step = (t_last - t0) / (n_frames - 1) if n_frames > 1 else 0.0
                timestamps = [float(t0 + i * step) for i in range(n_frames)]
        return self.frames_at(record, timestamps, camera_key=target)

    def episode_clip_path(self, record: EpisodeRecord, cache_dir: Path) -> Path | None:
        """将回合的子剪辑提取到 ``cache_dir/ep_{idx:06d}.mp4``。

        如果数据集没有视频轨道或提取失败，则返回 ``None``。
        当缓存的剪辑已存在时跳过重新提取。
        通过 :func:`lerobot.datasets.video_utils.reencode_video` 重新编码为 H.264，
        以便生成的 mp4 可被每个下游视频处理器解码——流复制会继承源编解码器
        （现代 LeRobot 数据集中通常是 AV1），而 vllm 的 libav 构建无法解码它。
        """
        if self.camera_key is None:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"ep_{record.episode_index:06d}.mp4"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        ep = self._meta.episodes[record.episode_index]
        from_timestamp = float(ep[f"videos/{self.camera_key}/from_timestamp"])
        to_timestamp = float(ep[f"videos/{self.camera_key}/to_timestamp"])
        src = self.root / self._meta.get_video_file_path(record.episode_index, self.camera_key)
        encoder = RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=None, crf=23, preset="ultrafast")
        try:
            reencode_video(
                src,
                out_path,
                video_encoder=encoder,
                overwrite=True,
                start_time_s=from_timestamp,
                end_time_s=to_timestamp,
            )
        except Exception:
            logger.warning(
                "clip extraction failed for episode %s (%s)", record.episode_index, src, exc_info=True
            )
            return None
        return out_path if out_path.exists() and out_path.stat().st_size > 0 else None

    def _decode(self, episode_index: int, timestamps: list[float], camera_key: str) -> list[Any]:
        """将回合视频中的 ``timestamps`` 解码为 ``(C, H, W)`` 张量。

        委托给 :func:`lerobot.datasets.video_utils.decode_video_frames`
        （torchcodec 可用时使用，否则 PyAV；``video_backend`` 显式指定一个）。
        为每个请求的时间戳返回一帧，如果解码失败则返回 ``[]``——
        调用者将 ``[]`` 视为"无可用帧"。
        """
        ep = self._meta.episodes[episode_index]
        from_timestamp = ep[f"videos/{camera_key}/from_timestamp"]
        shifted = [from_timestamp + ts for ts in timestamps]
        video_path = self.root / self._meta.get_video_file_path(episode_index, camera_key)

        try:
            # 模块阶段在线程池执行器下解码（参见
            # ``ExecutorConfig.episode_parallelism``），但 torchcodec 的缓存的
            # 每文件解码器是单线程的，因此在专用锁上串行化解码。
            # 帧提取是回合墙钟时间的一小部分（VLM 调用占主导），因此争用成本很低。
            with self._decode_lock:
                # 堆叠的 ``(N, C, H, W)`` uint8 张量；每个时间戳一行。
                decoded = decode_video_frames(
                    video_path, shifted, self.tolerance_s, backend=self.video_backend, return_uint8=True
                )
            return list(decoded)
        except Exception as exc:
            # 第一次大声记录，以便静默的 vqa 模块空操作（每个提示词都被跳过，
            # 因为 frames_at 返回 []）可以从作业日志而不是事后 parquet 检查中调试。
            # 后续失败保持安静。
            with self._lock:
                already_warned = self._warned_decode_fail
                if not already_warned:
                    self._warned_decode_fail = True
            if not already_warned:
                logger.warning(
                    "VideoFrameProvider._decode failed for episode=%s camera=%s video_path=%s backend=%s: %s",
                    episode_index,
                    camera_key,
                    video_path,
                    self.video_backend,
                    exc,
                    exc_info=exc,
                )
            return []


def make_frame_provider(
    root: Path, camera_key: str | None = None, video_backend: str | None = None
) -> FrameProvider:
    """如果存在视频则构建 :class:`VideoFrameProvider`，否则为空。"""
    try:
        provider = VideoFrameProvider(root=root, camera_key=camera_key, video_backend=video_backend)
    except Exception:
        return null_provider()
    if provider.camera_key is None:
        return null_provider()
    return provider


def _frame_to_pil(frame: Any) -> Any:
    """将解码帧具体化为 ``PIL.Image`` 以供 VLM 消息使用。

    帧作为 ``torch.Tensor``（``C, H, W`` uint8，直接来自
    :func:`decode_video_frames``）流经提供者；PIL 仅在这里，在 VLM 消息边界处创建，
    因为聊天后端期望 PIL 图像 / 数据 URL。非张量输入（例如测试存根）原样通过。
    """
    if not isinstance(frame, torch.Tensor):
        return frame
    array = frame.detach().cpu()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    return PIL.Image.fromarray(array.to(torch.uint8).numpy())


def to_image_blocks(frames: list[Any]) -> list[dict[str, Any]]:
    """将解码帧转换为 Qwen-VL 兼容的图像内容块。"""
    return [{"type": "image", "image": _frame_to_pil(frame)} for frame in frames]


def to_video_block(frames: list[Any]) -> list[dict[str, Any]]:
    """将解码帧列表包装为一个 Qwen-VL 视频块。

    当列表为空时返回 ``[]``，以便调用者可以将结果展平到内容数组中，
    而无需单独的空检查。
    """
    if not frames:
        return []
    return [{"type": "video", "video": [_frame_to_pil(frame) for frame in frames]}]


def to_video_url_block(url: str | None, fps: float = 2.0) -> list[dict[str, Any]]:
    """将视频文件 URL 包装为一个 ``video_url`` 块。

    由 ``openai`` 后端（transformers serve / vllm serve / ktransformers serve）使用，
    其中服务器处理帧采样。当 ``url`` 为 ``None`` 时返回 ``[]``，以便调用者可以展平。
    """
    if not url:
        return []
    return [{"type": "video_url", "video_url": {"url": url}, "fps": fps}]


def _draw_timestamp_badge(image: PIL.Image.Image, timestamp: float) -> PIL.Image.Image:
    """将 ``timestamp``（秒）烧录到 ``image`` 的左上角。

    一个带有白色文本的纯黑色徽章，这样读取联系表的 VLM 可以直接引用
    每个图块的确切源时间（例如 ``012.50s``），而不是调用者必须将图块位置映射回时间。
    镜像 macrodata/refiner 联系表约定。
    """
    from PIL import ImageDraw, ImageFont

    result = image.copy()
    draw = ImageDraw.Draw(result)
    # 将时间戳缩放到图块，以便在模型将整张表下采样到 768px 图块后保持可读——
    # 微小的位图字体在联系表分辨率下会模糊，VLM 无法再读取确切的源时间，
    # 而这正是边界分数所依赖的。``size=`` 自 10.1 起由 Pillow 的位图默认值支持；
    # 否则回退。
    badge_px = max(14, round(image.height * 0.12))
    try:
        font = ImageFont.load_default(size=badge_px)
    except TypeError:
        font = ImageFont.load_default()
    label = f"{timestamp:06.2f}s"
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = right - left, bottom - top
    pad = max(3, round(min(image.width, image.height) * 0.018))
    draw.rectangle((0, 0, text_w + pad * 2, text_h + pad * 2), fill=(0, 0, 0))
    draw.text((pad - left, pad - top), label, fill=(255, 255, 255), font=font)
    return result


def to_contact_sheet_blocks(
    frames: Sequence[Any],
    timestamps: Sequence[float],
    *,
    columns: int = 5,
    frames_per_sheet: int = 20,
    frame_width: int = 224,
    quality: int = 84,
) -> list[dict[str, Any]]:
    """将解码帧打包成带时间戳的 JPEG 联系表图像块。

    每帧被调整为 ``frame_width`` 宽，盖上其回合相对时间戳，
    并按行优先平铺成 ``frames_per_sheet``（``columns`` 宽）的网格。
    每个网格返回一个 ``{"type":"image", ...}`` 块；许多帧折叠成几张图像，
    因此长回合的时间覆盖保持密集，而视觉 token 仅为 N 个单独帧的一小部分。
    ``frames`` 和 ``timestamps`` 必须对齐且长度相等。空输入返回 ``[]``。
    """
    from PIL import Image

    if not frames:
        return []
    columns = max(1, columns)
    frames_per_sheet = max(1, frames_per_sheet)
    rows_per_sheet = math.ceil(frames_per_sheet / columns)

    tiles: list[PIL.Image.Image] = []
    for ts, frame in zip(timestamps, frames, strict=False):
        img = _frame_to_pil(frame)
        if not isinstance(img, PIL.Image.Image):
            continue
        img = img.convert("RGB")
        if img.width != frame_width:
            height = max(1, round(img.height * frame_width / img.width))
            img = img.resize((frame_width, height), resample=Image.Resampling.BILINEAR)
        tiles.append(_draw_timestamp_badge(img, float(ts)))
    if not tiles:
        return []

    blocks: list[dict[str, Any]] = []
    for start in range(0, len(tiles), frames_per_sheet):
        chunk = tiles[start : start + frames_per_sheet]
        cell_w = max(tile.width for tile in chunk)
        cell_h = max(tile.height for tile in chunk)
        sheet = Image.new("RGB", (cell_w * columns, cell_h * rows_per_sheet), color=(0, 0, 0))
        for i, tile in enumerate(chunk):
            x = (i % columns) * cell_w
            y = (i // columns) * cell_h
            sheet.paste(tile, (x, y))
        # 在 ``quality`` 处进行 JPEG 往返以匹配 refiner 约定并缩小传输负载；
        # 视觉 token 计数由分辨率决定，因此真正的节省是网格打包，而不是编解码器。
        buf = io.BytesIO()
        sheet.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        blocks.append({"type": "image", "image": Image.open(buf).convert("RGB")})
    return blocks
