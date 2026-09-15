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
# 注意：我们继承 str，这样序列化就很直接了
# https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json

"""视频编码器配置。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import numpy as np

from lerobot.utils.import_utils import require_package

logger = logging.getLogger(__name__)

# 自动选择时要探测的硬件编码器列表。可用性取决于平台和所选的视频后端。
# 当使用 vcodec="auto" 时，决定自动选择的优先顺序。
HW_VIDEO_CODECS = [
    "h264_videotoolbox",  # macOS
    "hevc_videotoolbox",  # macOS
    "h264_nvenc",  # NVIDIA GPU
    "hevc_nvenc",  # NVIDIA GPU
    "h264_vaapi",  # Linux Intel/AMD
    "h264_qsv",  # Intel Quick Sync
]
VALID_VIDEO_CODECS: frozenset[str] = frozenset(
    {"h264", "hevc", "libsvtav1", "libaom-av1", "auto", *HW_VIDEO_CODECS}
)
# 旧版视频编解码器名称的别名。
VIDEO_CODECS_ALIASES: dict[str, str] = {"av1": "libsvtav1"}

LIBSVTAV1_DEFAULT_PRESET: int = 12

# 以 ``video.<name>`` 形式持久化到 ``features[*]["info"]`` 下的键（来自 :class:`VideoEncoderConfig`）。
# ``vcodec``` 和 ``pix_fmt`` 直接从视频流中推导。
VIDEO_ENCODER_INFO_FIELD_NAMES: frozenset[str] = frozenset(
    {"g", "crf", "preset", "fast_decode", "extra_options", "video_backend"}
)
VIDEO_ENCODER_INFO_KEYS: frozenset[str] = frozenset(
    f"video.{name}" for name in VIDEO_ENCODER_INFO_FIELD_NAMES
)

# 默认的量化和编码参数。
DEPTH_QUANT_BITS: int = 12
DEPTH_QMAX: int = (1 << DEPTH_QUANT_BITS) - 1  # 4095

DEFAULT_DEPTH_MIN: float = 0.01
DEFAULT_DEPTH_MAX: float = 10.0
DEFAULT_DEPTH_SHIFT: float = 3.5
DEFAULT_DEPTH_USE_LOG: bool = True
DEFAULT_DEPTH_PIX_FMT: str = "gray12le"

DEPTH_METER_UNIT: str = "m"
DEPTH_MILLIMETER_UNIT: str = "mm"
DEFAULT_DEPTH_UNIT: str = DEPTH_MILLIMETER_UNIT


def infer_depth_unit(dtype: np.dtype | type) -> str:
    """根据原始帧的 dtype 推断其物理单位。

    浮点帧假定为米，整数帧假定为毫米。
    """
    return DEPTH_METER_UNIT if np.issubdtype(np.dtype(dtype), np.floating) else DEPTH_MILLIMETER_UNIT


# 深度专用的调优字段，以 ``video.<name>`` 形式持久化到 ``features[*]["info"]`` 下。
DEPTH_ENCODER_INFO_FIELD_NAMES: frozenset[str] = frozenset({"depth_min", "depth_max", "shift", "use_log"})


@dataclass
class VideoEncoderConfig:
    """视频编码器配置。"""

    vcodec: str = "libsvtav1"  # 视频编解码器名称。"auto" 会在可用时选择硬件编解码器，否则使用 libsvtav1。
    pix_fmt: str = "yuv420p"  # 像素格式（例如 yuv420p）。
    g: int | None = 2  # GOP 大小（关键帧间隔）。
    crf: int | float | None = 30  # 质量级别。越低表示质量越好、文件越大。
    preset: int | str | None = None  # 速度/质量预设。可接受的值取决于具体编解码器。
    fast_decode: int = 0  # 快速解码调优。可接受的值取决于具体编解码器，0 表示禁用。
    # TODO(CarolinePascal): 添加 torchcodec 支持 + 找到统一两个后端
    # （编码和解码）的方法。
    video_backend: str = "pyav"  # 编码后端。目前仅支持 "pyav"。
    # 最后合并的额外编解码器选项，例如 {"tune": "film"}。
    extra_options: dict[str, Any] = field(default_factory=dict)

    # 该编码器预期处理的源数据通道数。``None`` 会禁用
    # pix_fmt 的通道数检查；具体子类会设置它
    # （RGB 为 3，深度为 1，等等）。
    _DEFAULT_CHANNELS: ClassVar[int | None] = None

    def __post_init__(self) -> None:
        self.resolve_vcodec()
        # 空构造函数的易用性：``VideoEncoderConfig()`` 必须"开箱即用"。
        if self.preset is None and self.vcodec == "libsvtav1":
            self.preset = LIBSVTAV1_DEFAULT_PRESET
        self.validate()

    @classmethod
    def _kwargs_from_video_info(cls, video_info: dict | None) -> dict[str, Any]:
        """将特征 ``info`` 块中的 ``video.*`` 键解析为
        构造函数 kwargs。
        """
        video_info = video_info or {}
        kwargs: dict[str, Any] = {}

        for src_key, dst_field in (("video.codec", "vcodec"), ("video.pix_fmt", "pix_fmt")):
            value = video_info.get(src_key)
            if value is not None:
                kwargs[dst_field] = value

        for field_name in VIDEO_ENCODER_INFO_FIELD_NAMES:
            value = video_info.get(f"video.{field_name}")
            if value is None:
                continue
            # 与来源不一致合并后持久化为 ``{}`` —— 按默认值处理。
            if field_name == "extra_options" and not value:
                continue
            kwargs[field_name] = value

        return kwargs

    @classmethod
    def from_video_info(cls, video_info: dict | None) -> Self:
        """从视频特征的 ``info`` 块重建编码器配置。

        缺失或为 ``None`` 的值回退到类的默认值。
        """
        return cls(**cls._kwargs_from_video_info(video_info))

    def detect_available_encoders(self, encoders: list[str] | str) -> list[str]:
        """根据指定的视频后端返回可用编码器的子集。

        Args:
            encoders: 要检测的编码器名称列表。如果是字符串，会被转换为列表。
        Returns:
            可用编码器名称的列表。如果视频后端不是 "pyav"，返回空列表。
        """
        if self.video_backend == "pyav":
            require_package("av", extra="dataset")
            from lerobot.datasets import detect_available_encoders_pyav

            return detect_available_encoders_pyav(encoders)
        return []

    def validate(self) -> None:
        """验证视频编码器配置。"""
        if self.video_backend == "pyav":
            require_package("av", extra="dataset")
            from lerobot.datasets import check_video_encoder_parameters_pyav

            check_video_encoder_parameters_pyav(
                self.vcodec, self.pix_fmt, self.get_codec_options(), channels=self._DEFAULT_CHANNELS
            )

    def resolve_vcodec(self) -> None:
        """检查 ``vcodec``，当其值为 ``"auto"`` 时选择一个具体的编码器。

        对于 ``"auto"``，会选择优先列表中第一个可用的硬件编码器；如果没有可用的，则使用 ``libsvtav1``。如果解析出的编解码器（显式指定或自动选择后）不可用，则抛出 ``ValueError``。

        :data:`VIDEO_CODECS_ALIASES` 中列出的从视频流推导的规范编解码器名称
        会被重写为对应的编码器名称（例如 ``"av1"`` → ``"libsvtav1"``）。
        """
        self.vcodec = VIDEO_CODECS_ALIASES.get(self.vcodec, self.vcodec)
        if self.vcodec not in VALID_VIDEO_CODECS:
            raise ValueError(f"Invalid vcodec '{self.vcodec}'. Must be one of: {sorted(VALID_VIDEO_CODECS)}")
        if self.vcodec == "auto":
            available = self.detect_available_encoders(HW_VIDEO_CODECS)
            for encoder in HW_VIDEO_CODECS:
                if encoder in available:
                    logger.info(f"Auto-selected video codec: {encoder}")
                    self.vcodec = encoder
                    return
            logger.warning("No hardware encoder available, falling back to software encoder 'libsvtav1'")
            self.vcodec = "libsvtav1"

        if self.detect_available_encoders(self.vcodec):
            logger.info(f"Using video codec: {self.vcodec}")
            return
        raise ValueError(f"Unsupported video codec: {self.vcodec} with video backend {self.video_backend}")

    def get_codec_options(
        self, encoder_threads: int | None = None, as_strings: bool = False
    ) -> dict[str, Any]:
        """将调优字段转换为特定编解码器的选项。

        ``VideoEncoderConfig.extra_options`` 最后合并，但不会覆盖结构化字段。

        Args:
            encoder_threads: 为所有 VideoEncoderConfig 全局设置的编码器线程数。
                对于 libsvtav1，通过 ``svtav1-params`` 映射为 ``lp``。
                对于 h264/hevc，映射为 ``threads``。
                硬件编码器忽略此参数。
            as_strings: 如果为 ``True``，将值转换为字符串。
        """
        opts: dict[str, Any] = {}

        def set_if(key: str, value: Any) -> None:
            if value is not None:
                opts[key] = value if not as_strings else str(value)

        # GOP 大小不是编解码器特定的选项，因此始终设置。
        set_if("g", self.g)

        if self.vcodec == "libsvtav1":
            set_if("crf", self.crf)
            set_if("preset", self.preset)
            svtav1_parts: list[str] = []
            if self.fast_decode is not None:
                svtav1_parts.append(f"fast-decode={max(0, min(2, self.fast_decode))}")
            if encoder_threads is not None:
                svtav1_parts.append(f"lp={encoder_threads}")
            if svtav1_parts:
                set_if("svtav1-params", ":".join(svtav1_parts))
        elif self.vcodec in ("h264", "hevc"):
            set_if("crf", self.crf)
            set_if("preset", self.preset)
            if self.fast_decode:
                set_if("tune", "fastdecode")
            set_if("threads", encoder_threads)
        elif self.vcodec == "libaom-av1":
            set_if("crf", self.crf)
            set_if("preset", self.preset)
            if encoder_threads is not None:
                set_if("threads", encoder_threads)
                set_if("row-mt", 1)
        elif self.vcodec in ("h264_videotoolbox", "hevc_videotoolbox"):
            if self.crf is not None:
                set_if("q:v", max(1, min(100, 100 - self.crf * 2)))
        elif self.vcodec in ("h264_nvenc", "hevc_nvenc"):
            set_if("rc", 0)
            set_if("qp", self.crf)
            set_if("preset", self.preset)
        elif self.vcodec == "h264_vaapi":
            set_if("qp", self.crf)
        elif self.vcodec == "h264_qsv":
            set_if("global_quality", self.crf)
            set_if("preset", self.preset)
        else:
            set_if("crf", self.crf)
            set_if("preset", self.preset)

        # 额外选项最后合并，但不会覆盖结构化字段（值按给定保留）。
        for k, v in self.extra_options.items():
            if k not in opts:
                set_if(k, v)

        return opts


@dataclass
class RGBEncoderConfig(VideoEncoderConfig):
    """RGB 相机流的编码器配置。

    与 :class:`VideoEncoderConfig` 相同，但声明了 3 通道的
    源数据布局，因此 ``pix_fmt`` 会针对 RGB 输入进行验证。
    """

    _DEFAULT_CHANNELS: ClassVar[int] = 3


def rgb_encoder_defaults() -> RGBEncoderConfig:
    """返回带有 RGB 相机默认值的 :class:`RGBEncoderConfig`。"""
    return RGBEncoderConfig()


@dataclass
class DepthEncoderConfig(VideoEncoderConfig):
    """深度图流的编码器配置。

    继承完整的 :class:`VideoEncoderConfig` 接口（编解码器、GOP、CRF、
    preset、``extra_options``…），并添加量化器的参数。
    默认值将 ``vcodec`` 切换为 ``"hevc"``（Main 12 profile），将 ``pix_fmt`` 切换为
    ``"gray12le"``。
    """

    vcodec: str = "hevc"  # 视频编解码器名称。默认为 HEVC Main 12（支持 12 位的编解码器）。
    pix_fmt: str = "gray12le"  # 像素格式。默认为 12 位灰度。
    extra_options: dict[str, Any] = field(default_factory=lambda: {"x265-params": "lossless=1"})

    depth_min: float = DEFAULT_DEPTH_MIN  # 最小深度（米），映射到最低量化值。
    depth_max: float = DEFAULT_DEPTH_MAX  # 最大深度（米），映射到最高量化值。
    shift: float = DEFAULT_DEPTH_SHIFT  # 接近零时为保证数值稳定的对数前偏移（米）。
    use_log: bool = DEFAULT_DEPTH_USE_LOG  # 使用对数量化（True）还是线性量化（False）。

    _DEFAULT_CHANNELS: ClassVar[int] = 1

    @classmethod
    def _kwargs_from_video_info(cls, video_info: dict | None) -> dict[str, Any]:
        """在基础解析器之上叠加深度专用调优（``depth_min`` / ``depth_max`` /
        ``shift`` / ``use_log``）。缺失的键回退到类的默认值。
        """
        kwargs = super()._kwargs_from_video_info(video_info)
        video_info = video_info or {}
        for name in DEPTH_ENCODER_INFO_FIELD_NAMES:
            value = video_info.get(f"video.{name}")
            if value is not None:
                kwargs[name] = value
        return kwargs


def depth_encoder_defaults() -> DepthEncoderConfig:
    """返回带有相机默认值的 :class:`DepthEncoderConfig`。"""
    return DepthEncoderConfig()


def is_depth_map(feature: dict | None) -> bool:
    """返回一个特征是否被标记为深度图。

    深度图通过 ``feature['info']['is_depth_map']`` 进行规范标记，或通过
    旧版的 ``video.is_depth_map`` 键标记，后者可能位于 ``feature['info']`` 中，
    也可能位于单独的 ``feature['video_info']`` 字典中。
    """
    feature = feature or {}
    info = feature.get("info") or {}
    video_info = feature.get("video_info")
    return bool(
        info.get("is_depth_map")
        or info.get("video.is_depth_map")
        or (isinstance(video_info, dict) and video_info.get("video.is_depth_map"))
    )


def encoder_config_from_video_info(video_info: dict | None) -> VideoEncoderConfig:
    """根据特征的 ``info`` 块构建合适的编码器配置。

    当字典将特征标记为深度图时分派到 :class:`DepthEncoderConfig`，
    否则分派到 :class:`RGBEncoderConfig`。

    Args:
        video_info: 持久化在 ``info.json`` 中的特征 ``info`` 字典，
            或 ``None``（视为空字典）。

    Returns:
        深度特征返回 :class:`DepthEncoderConfig`，否则返回
        :class:`RGBEncoderConfig`。
    """
    video_info = video_info or {}
    cls: type[VideoEncoderConfig] = (
        DepthEncoderConfig if is_depth_map({"info": video_info}) else RGBEncoderConfig
    )
    return cls.from_video_info(video_info)
