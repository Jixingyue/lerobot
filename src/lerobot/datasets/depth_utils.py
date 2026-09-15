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
"""
:class:`DepthEncoderConfig` 的深度编码/解码辅助函数。
"""

import math
from typing import Literal

import av
import numpy as np
import torch
from numpy.typing import NDArray

from lerobot.configs.video import (
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_DEPTH_PIX_FMT,
    DEFAULT_DEPTH_SHIFT,
    DEFAULT_DEPTH_USE_LOG,
    DEPTH_METER_UNIT,
    DEPTH_MILLIMETER_UNIT,
    DEPTH_QMAX,
    infer_depth_unit,
)

from .image_writer import squeeze_single_channel
from .pyav_utils import write_u16_plane

MM_PER_METRE = 1000.0
_UINT16_MAX = 65535


def _validate_log_quant_params(depth_min: float, shift: float) -> None:
    """确保 ``log(depth_min + shift)`` 是有限值。"""
    if depth_min + shift <= 0:
        raise ValueError(
            f"depth_min + shift must be positive for logarithmic quantization, "
            f"got depth_min={depth_min} + shift={shift} = {depth_min + shift}"
        )


def _depth_input_to_float32_and_unit(
    depth: NDArray[np.integer] | NDArray[np.floating],
    input_unit: Literal["auto", DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT],
) -> tuple[NDArray[np.float32], Literal[DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT]]:
    """将深度转换为所选单位的 float32，并返回解析出的单位。"""
    resolved_unit = infer_depth_unit(depth.dtype) if input_unit == "auto" else input_unit
    return depth.astype(np.float32, order="K"), resolved_unit


def quantize_depth(
    depth: NDArray[np.uint16] | NDArray[np.float32] | torch.Tensor,
    depth_min: float = DEFAULT_DEPTH_MIN,
    depth_max: float = DEFAULT_DEPTH_MAX,
    shift: float = DEFAULT_DEPTH_SHIFT,
    use_log: bool = DEFAULT_DEPTH_USE_LOG,
    pix_fmt: str = DEFAULT_DEPTH_PIX_FMT,
    video_backend: str | None = "pyav",
    input_unit: Literal["auto", DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT] = "auto",
) -> NDArray[np.uint16] | av.VideoFrame:
    """将深度量化为 12 位编码（``uint16``，取值 ``0…DEPTH_QMAX``）。

    深度图被打包为 12 位整数帧，以便适配标准的
    高位深像素格式（例如 ``yuv420p12le`` / ``gray12le``），
    并可被广泛支持的视频编解码器（例如 HEVC Main 12）编码。
    对数量化是默认方式，因为它为近距离深度分配更多量化级，
    这符合典型深度传感器的 (1/depth) 误差特性。
    数学实现移植自 BEHAVIOR-1K 的 ``obs_utils.py``。

    **输入单位**：

    - ``input_unit="auto"``（默认）：根据 dtype 推断（浮点 = m，非浮点 = mm）。
    - ``input_unit="mm"``：将输入值解释为毫米。
    - ``input_unit="m"``：将输入值解释为米。

    量化数学运算在**解析出的输入单位**下进行。

    ``depth_min``、``depth_max`` 和 ``shift`` 始终以**米**为单位。

    Args:
        depth: 深度图；``torch.Tensor`` 会被移到 CPU 上进行转换。
        depth_min: 量化级 ``0`` 对应的深度（米）。
        depth_max: 量化级 :data:`DEPTH_QMAX` 对应的深度（米）。
        shift: 深度偏移量（米）；用于 log 模式。必须满足 ``depth_min + shift > 0``。
        use_log: 如果为 ``True``（默认），在 log 空间中量化。
        video_backend: 用于编码的视频后端。默认为 "pyav"。
        input_unit: 输入单位策略（``"auto"``、``"mm"``、``"m"``）。

    Returns:
        ``numpy.ndarray``，``dtype=uint16``，与 ``depth`` 形状相同，取值在
        ``[0, DEPTH_QMAX]`` 范围内。

    Raises:
        ValueError: 如果 ``input_unit`` 不是 ``"auto"``、``"mm"`` 或 ``"m"``。
        ValueError: 如果 ``use_log=True`` 且 ``depth_min + shift <= 0``。
    """
    if input_unit not in ("auto", DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT):
        raise ValueError(
            f"input_unit must be 'auto', '{DEPTH_METER_UNIT}', or '{DEPTH_MILLIMETER_UNIT}', got {input_unit!r}"
        )

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    # 去除单通道维度：(H, W, 1) 或 (1, H, W) → (H, W)
    depth = squeeze_single_channel(depth)

    depth_f, resolved_unit = _depth_input_to_float32_and_unit(depth, input_unit=input_unit)

    # 将 depth_min、depth_max 和 shift 转换为解析出的输入单位。
    depth_min_u = (
        np.float32(depth_min) if resolved_unit == DEPTH_METER_UNIT else np.float32(depth_min * MM_PER_METRE)
    )
    depth_max_u = (
        np.float32(depth_max) if resolved_unit == DEPTH_METER_UNIT else np.float32(depth_max * MM_PER_METRE)
    )
    shift_u = np.float32(shift) if resolved_unit == DEPTH_METER_UNIT else np.float32(shift * MM_PER_METRE)

    # 归一化和量化在解析出的输入单位下执行。
    if use_log:
        _validate_log_quant_params(depth_min, shift)
        log_min = math.log(float(depth_min_u + shift_u))
        log_max = math.log(float(depth_max_u + shift_u))
        norm = (np.log(depth_f + shift_u) - log_min) / (log_max - log_min)
    else:
        norm = (depth_f - depth_min_u) / (depth_max_u - depth_min_u)

    quantized = np.rint(norm * DEPTH_QMAX).clip(0, DEPTH_QMAX).astype(np.uint16, copy=False)

    if video_backend == "pyav":
        frame = av.VideoFrame.from_ndarray(quantized, format=pix_fmt)
        write_u16_plane(frame.planes[0], quantized)
        return frame
    else:
        return quantized


def dequantize_depth(
    quantized: NDArray[np.uint16] | av.VideoFrame | torch.Tensor,
    depth_min: float = DEFAULT_DEPTH_MIN,
    depth_max: float = DEFAULT_DEPTH_MAX,
    shift: float = DEFAULT_DEPTH_SHIFT,
    use_log: bool = DEFAULT_DEPTH_USE_LOG,
    pix_fmt: str = DEFAULT_DEPTH_PIX_FMT,
    output_unit: Literal[DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT] = DEPTH_MILLIMETER_UNIT,
    output_tensor: bool = True,
    output_channel_last: bool = False,
) -> NDArray[np.uint16] | NDArray[np.float32] | torch.Tensor:
    """:func:`quantize_depth` 的逆操作。

    解码时使用 ``depth_min`` / ``depth_max`` / ``shift``（单位为米）
    反转与 :func:`quantize_depth` 相同的归一化编码映射，然后返回
    所请求的输出单位。调参**必须与** :func:`quantize_depth` 一致。

    接受的输入布局：

    - ``(H, W, 1)`` 或 ``(H, W)`` — 通道在后的单帧。
    - ``(..., 1, H, W)`` — 通道在前的批量帧。
    - ``(..., H, W, 1)`` — 通道在后的批量帧。
    输出布局由 ``output_channel_last`` 决定。

    Args:
        quantized: ``[0, DEPTH_QMAX]`` 范围内的 12 位编码。``np.ndarray``、
            ``av.VideoFrame`` 或 ``torch.Tensor``（任意整数或浮点 dtype）。
        depth_min, depth_max, shift, use_log: 与 :func:`quantize_depth` 相同（米）。
        pix_fmt: 用于从 ``av.VideoFrame`` 提取平面的像素格式。
        output_unit: ``"mm"`` 在返回 numpy 数组时返回 ``uint16`` 毫米值
            （rint、裁剪到 ``[0, 65535]``），或在 ``output_tensor=True`` 时
            返回 ``float32`` mm。``"m"`` 返回 ``[depth_min, depth_max]``
            范围内的 ``float32`` 米值。
        output_tensor: 如果为 True，返回 ``torch.Tensor`` 而不是 numpy 数组。

    Returns:
        以所请求单位和 dtype 表示的深度图。

    Raises:
        ValueError: 如果 ``output_unit`` 不是 ``"m"`` 或 ``"mm"``。
        ValueError: 如果 ``use_log=True`` 且 ``depth_min + shift <= 0``。
    """
    if output_unit not in (DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT):
        raise ValueError(
            f"output_unit must be '{DEPTH_METER_UNIT}' or '{DEPTH_MILLIMETER_UNIT}', got {output_unit!r}"
        )
    if use_log:
        _validate_log_quant_params(depth_min, shift)

    if isinstance(quantized, av.VideoFrame):
        quantized = quantized.to_ndarray(format=pix_fmt)

    # 先计算缩放系数和偏移量。
    depth_min_m = float(depth_min)
    depth_max_m = float(depth_max)
    shift_m = float(shift)
    if use_log:
        log_min = math.log(depth_min_m + shift_m)
        log_max = math.log(depth_max_m + shift_m)
        scale = (log_max - log_min) / DEPTH_QMAX
        offset = log_min
    else:
        scale = (depth_max_m - depth_min_m) / DEPTH_QMAX
        offset = depth_min_m

    # ── Torch 路径：保持在输入设备上，单次 fp32 分配。 ────────
    if isinstance(quantized, torch.Tensor):
        if quantized.ndim >= 3:
            # 去掉单通道维度，使数学运算在 (..., H, W) 上进行。
            quantized = quantized.squeeze(-3) if quantized.shape[-3] == 1 else quantized.squeeze(-1)

        # 由我们拥有的单次分配；其余操作均为原地执行。
        buf = quantized.to(dtype=torch.float32, copy=True)
        buf.mul_(scale).add_(offset)
        if use_log:
            buf.exp_().sub_(shift_m)
        buf.clamp_(depth_min_m, depth_max_m)
        buf.unsqueeze_(-1) if output_channel_last else buf.unsqueeze_(-3)

        if output_unit == DEPTH_METER_UNIT:
            return buf if output_tensor else buf.cpu().numpy()

        # mm 路径：在 float32 下做舍入 + 裁剪，返回张量时跳过
        # uint16 往返转换（torch.uint16 支持较差）。
        buf.mul_(MM_PER_METRE).round_().clamp_(0.0, _UINT16_MAX)
        if output_tensor:
            return buf
        return buf.cpu().numpy().astype(np.uint16, copy=False)

    # ── NumPy 路径：单次 fp32 分配，使用 ``out=`` 进行原地运算。 ─────
    arr = np.asarray(quantized)
    if arr.ndim >= 3:
        # 去掉单通道维度，使数学运算在 (..., H, W) 上进行。
        arr = np.squeeze(arr, axis=-3) if arr.shape[-3] == 1 else np.squeeze(arr, axis=-1)

    buf = np.empty(arr.shape, dtype=np.float32)
    np.multiply(arr, scale, out=buf)
    np.add(buf, offset, out=buf)
    if use_log:
        np.exp(buf, out=buf)
        np.subtract(buf, shift_m, out=buf)
    np.clip(buf, depth_min_m, depth_max_m, out=buf)
    buf = np.expand_dims(buf, axis=-1) if output_channel_last else np.expand_dims(buf, axis=-3)

    if output_unit == DEPTH_METER_UNIT:
        return torch.from_numpy(buf) if output_tensor else buf

    np.multiply(buf, MM_PER_METRE, out=buf)
    np.rint(buf, out=buf)
    np.clip(buf, 0.0, _UINT16_MAX, out=buf)
    if output_tensor:
        # torch.uint16 的支持非常有限；返回 float32 毫米值。
        return torch.from_numpy(buf)
    return buf.astype(np.uint16, copy=False)
