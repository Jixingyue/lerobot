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
"""基于 PyAV 的 :class:`VideoEncoderConfig` 兼容性检查。

集中处理对捆绑的 FFmpeg 构建的所有 :mod:`av` 内省。
当目标编解码器在本地不可用时，检查会退化为空操作。
"""

import functools
import logging
from typing import Any

import av
import numpy as np

logger = logging.getLogger(__name__)

FFMPEG_NUMERIC_OPTION_TYPES = ("INT", "INT64", "UINT64", "FLOAT", "DOUBLE")
FFMPEG_INTEGER_OPTION_TYPES = ("INT", "INT64", "UINT64")


def write_u16_plane(plane: av.video.plane.VideoPlane, src: np.ndarray, fill_value: int | None = None) -> None:
    """将 2D ``uint16`` 图像逐行复制到该平面的内存缓冲区中。

    为了提高速度，每一行都会被填充到比 ``width`` 更宽的宽度，因此内存中真实的行宽是
    ``plane.line_size``（字节），而不是 ``width``。如果作为一条连续的流来复制，
    会使图像发生错位，因此我们只写入每行的前 ``width`` 列，
    并保持填充部分不变。

    Args:
        plane: 目标 16 位平面。
        src: 源图像，形状 ``(height, width)``，dtype ``uint16``。
        fill_value: 如果给定，则首先将每个像素（包括填充部分）设置为此值，
            使填充部分保存干净的数据而不是垃圾数据。
    """
    height, width = src.shape
    stride_u16 = plane.line_size // np.dtype(np.uint16).itemsize
    dst = np.frombuffer(plane, dtype=np.uint16).reshape(height, stride_u16)
    if fill_value is not None:
        dst.fill(fill_value)
    dst[:, :width] = src


@functools.cache
def get_pix_fmt_channels(pix_fmt: str) -> int:
    """返回 *pix_fmt* 的分量（通道）数量。"""
    return len(av.VideoFormat(pix_fmt).components)


@functools.cache
def get_codec(vcodec: str) -> av.codec.Codec | None:
    """返回 *vcodec* 的 PyAV 写入模式 ``Codec``，如果不可用则返回 ``None``。"""
    try:
        return av.codec.Codec(vcodec, "w")
    except Exception:
        return None


@functools.cache
def _get_codec_options_by_name(vcodec: str) -> dict[str, av.option.Option]:
    """*vcodec* 的私有选项名称 → PyAV ``Option`` 映射（不可用时为空）。"""
    codec = get_codec(vcodec)
    if codec is None:
        return {}
    return {opt.name: opt for opt in codec.descriptor.options}


@functools.cache
def _get_codec_video_formats(vcodec: str) -> tuple[str, ...]:
    """*vcodec* 按 PyAV 首选顺序接受的像素格式（未知时为空）。"""
    codec = get_codec(vcodec)
    if codec is None:
        return ()
    return tuple(fmt.name for fmt in (codec.video_formats or []))


def detect_available_encoders_pyav(encoders: list[str] | str) -> list[str]:
    """返回 *encoders* 中在本地 FFmpeg 构建中可用作视频编码器的子集。

    每个名称都通过 :func:`get_codec` 直接探测；保留输入顺序。
    """
    if isinstance(encoders, str):
        encoders = [encoders]

    available: list[str] = []
    for name in encoders:
        codec = get_codec(name)
        if codec is not None and codec.type == "video":
            available.append(name)
        else:
            logger.debug("encoder '%s' not available as video encoder", name)
    return available


def _check_option_value(vcodec: str, label: str, value: Any, opt: av.option.Option) -> None:
    """对数值型 *value* 做范围检查，对字符串型 *value* 做选项检查，依据是 *opt*。"""
    type_name = opt.type.name
    if type_name in FFMPEG_NUMERIC_OPTION_TYPES:
        if isinstance(value, bool):
            raise ValueError(
                f"{label}={value!r} is not numeric; codec {vcodec!r} expects a number for this option."
            )
        elif isinstance(value, str):
            try:
                num_val = float(value)
            except ValueError as e:
                raise ValueError(
                    f"{label}={value!r} is not numeric; codec {vcodec!r} expects a number for this option."
                ) from e
        elif isinstance(value, (float, int)):
            num_val = float(value)
        else:
            raise ValueError(
                f"{label}={value!r} is not numeric; codec {vcodec!r} expects a number for this option."
            )

        # 检查整数类型兼容性
        if type_name in FFMPEG_INTEGER_OPTION_TYPES and not num_val.is_integer():
            raise ValueError(
                f"{label}={num_val!r} must be an integer for codec {vcodec!r} "
                f"(FFmpeg option {opt.name!r} is {type_name}); float values are not allowed."
            )

        # 检查数值范围兼容性
        lo, hi = float(opt.min), float(opt.max)
        if lo < hi and not (lo <= num_val <= hi):
            raise ValueError(
                f"{label}={num_val} is out of range for codec {vcodec!r}; must be in [{lo}, {hi}]"
            )

    elif type_name == "STRING":
        if isinstance(value, bool):
            raise ValueError(f"{label}={value!r} is not a valid string value for codec {vcodec!r}.")
        if isinstance(value, str):
            str_val = value
        elif isinstance(value, (int, float)):
            str_val = str(value)
        else:
            raise ValueError(f"{label}={value!r} has unsupported type for STRING option on codec {vcodec!r}")

        # 检查字符串选项兼容性
        choices = [c.name for c in (opt.choices or [])]
        if choices and str_val not in choices:
            raise ValueError(
                f"{label}={str_val!r} is not a supported choice for codec "
                f"{vcodec!r}; valid choices: {choices}"
            )
    else:
        return


def _check_pixel_format(vcodec: str, pix_fmt: str) -> None:
    formats = _get_codec_video_formats(vcodec)
    if formats and pix_fmt not in formats:
        raise ValueError(
            f"pix_fmt={pix_fmt!r} is not supported by codec {vcodec!r}; "
            f"supported pixel formats: {list(formats)}"
        )


def _check_pix_fmt_channels(pix_fmt: str, channels: int) -> None:
    """确保 *pix_fmt* 至少能承载 *channels* 个分量。"""
    pix_fmt_channels = get_pix_fmt_channels(pix_fmt)
    if pix_fmt_channels < channels:
        raise ValueError(
            f"pix_fmt={pix_fmt!r} carries only {pix_fmt_channels} component(s) "
            f"but the source data has {channels} channel(s)."
        )


def _check_codec_options(vcodec: str, codec_options: dict[str, Any]) -> None:
    """根据编解码器公开的 AVOptions 验证（带类型的）合并后的编码器选项。"""
    supported_options = _get_codec_options_by_name(vcodec)
    for key, value in codec_options.items():
        # GOP 大小不是编解码器特定的选项，必须单独验证。
        if key == "g":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"g={value!r} must be a positive integer for codec {vcodec!r}")
            continue
        if key not in supported_options:
            continue
        _check_option_value(vcodec, key, value, supported_options[key])


def check_video_encoder_parameters_pyav(
    vcodec: str,
    pix_fmt: str,
    codec_options: dict[str, Any],
    channels: int | None = None,
) -> None:
    """验证 *config* 与捆绑的 FFmpeg 构建兼容。

    根据 PyAV 检查像素格式、抽象调优字段兼容性，以及来自
    :meth:`~lerobot.configs.video.VideoEncoderConfig.get_codec_options` 的每个合并后的
    编码器选项（包括该字典中存在的数值型 ``extra_options``）。
    如果给定，还会验证 *pix_fmt* 承载的分量数量与源数据通道数一致。
    当 ``config.vcodec`` 不在本地 FFmpeg 构建中时为空操作。

    Raises:
        ValueError: 遇到第一个不兼容项时抛出。
    """
    options = _get_codec_options_by_name(vcodec)
    if not options:
        raise ValueError(f"Codec {vcodec!r} is not available in the bundled FFmpeg build")
    _check_pixel_format(vcodec, pix_fmt)
    if channels is not None:
        _check_pix_fmt_channels(pix_fmt, channels)
    _check_codec_options(vcodec, codec_options)
