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

"""源自 openpi 的 VLA 策略（pi0、pi05、pi0_fast、smolvla、eo1、xvla）共享的辅助函数。

这些函数过去是按策略复制粘贴的，这里是它们的规范版本。它们是纯函数
（无参数、无模块状态），因此从这里导入而不是使用策略本地的副本，
对检查点没有任何影响。
"""

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE
from lerobot.utils.device_utils import get_safe_dtype
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import DynamicCache
else:
    DynamicCache = None


def create_sinusoidal_pos_embedding(  # 参见 openpi 的 `create_sinusoidal_pos_embedding`（完全一致的副本）
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """为标量或逐动作位置计算正弦-余弦嵌入。"""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim not in (1, 2):
        raise ValueError("The time tensor must have shape (batch_size,) or (batch_size, action_horizon).")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = time[..., None] * scaling_factor
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:  # 参见 openpi（完全一致的副本）
    """复制自 big_vision。

    token 可以关注累计 mask_ar 小于或等于自身的有效输入 token。这样
    `mask_ar` int[B, N] 就可以用来设置多种类型的注意力，例如：

      [[1 1 1 1 1 1]]：纯因果注意力。

      [[0 0 0 1 1 1]]：prefix-lm 注意力。前 3 个 token 可以相互关注，
          后 3 个 token 采用因果注意力。第一个元素取 1 也不会改变行为。

      [[1 0 1 0 1 0 0 1 0 0]]：4 个块之间的因果注意力。某个块内的
          token 可以关注所有前面的块以及同一块内的所有 token。

    Args:
      input_mask: bool[B, N]，属于输入则为 true，是填充则为 false。
      mask_ar: int32[B, N] 掩码，为 1 表示前面的 token 不能依赖它，
        为 0 表示它与前一个 token 共享相同的注意力掩码。
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def prepare_attention_masks_4d(att_2d_masks: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """将布尔 2D 注意力掩码扩展为 transformers 所期望的加性 4D 布局。

    有效位置变为 0.0，被掩蔽位置变为 openpi 的大负数常量。
    """
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    result = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


def clone_past_key_values(past_key_values):
    """克隆前缀预填充返回的 DynamicCache，用于编译后的去噪。"""
    if DynamicCache is None:
        require_package("transformers", extra="transformers-dep")

    return DynamicCache(
        tuple(
            (keys.clone(), values.clone(), sliding_window) for keys, values, sliding_window in past_key_values
        )
    )


def pad_vector(vector: Tensor, new_dim: int, *, truncate: bool = False) -> Tensor:
    """用零将向量的最后一维填充到 new_dim。

    可以是 (batch_size x sequence_length x features_dimension)
    或 (batch_size x features_dimension)

    当 ``truncate=False`` 时（openpi 行为），最后一维已经 >= new_dim 的向量
    原样返回。当 ``truncate=True`` 时（xVLA 行为），最后一维会被截断到
    恰好 ``new_dim``（可能为 0）。
    """
    if vector.shape[-1] == new_dim:
        return vector
    if not truncate:
        if vector.shape[-1] >= new_dim:
            return vector
        return F.pad(vector, (0, new_dim - vector.shape[-1]))
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = vector.new_zeros(*shape)
    length = min(current_dim, new_dim)
    new_vector[..., :length] = vector[..., :length]
    return new_vector


def resize_with_pad_torch(  # 参见 openpi 的 `resize_with_pad_torch`（完全一致的副本）
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """resize_with_pad 的 PyTorch 版本。通过填充黑色，将图像无失真地缩放到目标高度和宽度。
    如果图像是 float32，则其值必须在 [-1, 1] 范围内。

    填充是居中的（openpi 约定）。对于 smolvla/xvla 使用的左上角填充变体，
    参见 :func:`resize_with_pad`。

    Args:
        images: 形状为 [*b, h, w, c] 或 [*b, c, h, w] 的张量
        height: 目标高度
        width: 目标宽度
        mode: 插值模式（'bilinear'、'nearest' 等）

    Returns:
        与输入形状格式相同的缩放并填充后的张量
    """
    # 检查输入是通道后置格式 [*b, h, w, c] 还是通道前置格式 [*b, c, h, w]
    if images.shape[-1] <= 4:  # 假定为通道后置格式
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # 添加批次维度
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # 添加批次维度

    batch_size, channels, cur_height, cur_width = images.shape

    # 计算缩放比例
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # 缩放
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # 处理特定 dtype 的裁剪
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # 计算填充
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # 填充
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # 左、右、上、下
        mode="constant",
        value=constant_value,
    )

    # 如有需要，转换回原始格式
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


def resize_with_pad(img: torch.Tensor, height: int, width: int, *, pad_value: float) -> torch.Tensor:
    """无失真地缩放 (b, c, h, w) 图像，并在左侧和上侧填充。

    这是 smolvla/xvla 的约定。对于居中填充的 openpi 变体，参见
    :func:`resize_with_pad_torch`。``pad_value`` 特意设为仅限关键字参数：
    调用方过去使用不同的值（0、-1），必须显式声明其选择。
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img
