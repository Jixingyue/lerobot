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
import collections
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.io import decode_image, encode_jpeg
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)


class RandomSubsetApply(Transform):
    """从变换列表中随机应用 N 个变换的子集。

    Args:
        transforms: 变换列表。
        p: 表示用于采样变换的多项分布概率（无放回）。
            如果权重之和不为 1，将被归一化。如果为 ``None``（默认），所有变换
            具有相同的概率。
        n_subset: 要应用的变换数量。如果为 ``None``，则应用所有变换。
            必须在 [1, len(transforms)] 范围内。
        random_order: 以随机顺序应用变换。
    """

    def __init__(
        self,
        transforms: Sequence[Callable[..., Any]],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1.0] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms: list[Callable[..., Any]] = []

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """随机改变图像或视频的锐度。

    类似于 p=1 且随机采样 sharpness_factor 的 v2.RandomAdjustSharpness。
    v2.RandomAdjustSharpness 以给定概率对图像应用固定的 sharpness_factor，
    而 SharpnessJitter 每次都应用随机的 sharpness_factor。这样可以得到
    更多样化的增强结果。

    sharpness_factor 为 0 得到模糊图像，为 1 得到原始图像，为 2 则锐度
    提升 2 倍。

    如果输入是 :class:`torch.Tensor`，
    其形状应为 [..., 1 or 3, H, W]，其中 ... 表示任意数量的前导维度。

    Args:
        sharpness: 锐度抖动的幅度。sharpness_factor 从
            [max(0, 1 - sharpness), 1 + sharpness] 或给定的
            [min, max] 区间内均匀选取。应为非负数。
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness: float | Sequence[float]) -> tuple[float, float]:
        if isinstance(sharpness, (int | float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


class GaussianNoise(Transform):
    """添加高斯噪声以模拟相机传感器噪声。

    模拟 ADC 量化产生的读出噪声，该噪声在低光照条件下会增大。
    在腕部相机工作于次优光照环境的真实机器人场景中很常见。

    Args:
        std: 噪声标准差的范围 (min, max)，以像素值尺度 (0-255) 计。
    """

    def __init__(self, std: float | Sequence[float] = (5.0, 25.0)) -> None:
        super().__init__()
        if isinstance(std, (int, float)):
            self.std = (0.0, float(std))
        elif isinstance(std, Sequence) and len(std) == 2:
            self.std = (float(std[0]), float(std[1]))
        else:
            raise TypeError("std must be a number or a sequence with length 2.")
        if not 0.0 <= self.std[0] <= self.std[1]:
            raise ValueError(f"std must satisfy 0 <= min <= max, but got {self.std}.")

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {
            "std": torch.empty(1).uniform_(self.std[0], self.std[1]).item(),
            "seed": torch.randint(0, torch.iinfo(torch.int64).max, ()).item(),
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, torch.Tensor) and inpt.is_floating_point():
            generator = torch.Generator(device=inpt.device).manual_seed(params["seed"])
            noise = torch.randn(inpt.shape, device=inpt.device, dtype=inpt.dtype, generator=generator)
            return (inpt + noise * (params["std"] / 255.0)).clamp(0.0, 1.0)
        return inpt


class MotionBlur(Transform):
    """应用方向性运动模糊以模拟机器人或物体的快速移动。

    沿随机方向生成一维平均核，通过深度卷积应用。

    Args:
        kernel_size: 奇数核大小，或至少包含一个奇数核大小的范围。
    """

    def __init__(self, kernel_size: int | Sequence[int] = (3, 11)) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, Sequence) and len(kernel_size) == 2:
            self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        else:
            raise TypeError("kernel_size must be an int or a sequence with length 2.")
        if not 1 <= self.kernel_size[0] <= self.kernel_size[1]:
            raise ValueError(f"kernel_size must satisfy 1 <= min <= max, but got {self.kernel_size}.")
        self._first_odd_kernel_size = self.kernel_size[0] + (self.kernel_size[0] + 1) % 2
        if self._first_odd_kernel_size > self.kernel_size[1]:
            raise ValueError(f"kernel_size range must contain an odd value, but got {self.kernel_size}.")

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        num_odd_sizes = (self.kernel_size[1] - self._first_odd_kernel_size) // 2 + 1
        size_index = int(torch.randint(0, num_odd_sizes, ()).item())
        ks = self._first_odd_kernel_size + 2 * size_index
        angle = torch.empty(1).uniform_(0, 360).item()
        return {"kernel_size": ks, "angle": angle}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        if inpt.ndim < 3:
            raise ValueError(f"MotionBlur expects [..., C, H, W] input, but got shape {inpt.shape}.")

        kernel_size = params["kernel_size"]
        radius = kernel_size // 2
        angle = math.radians(params["angle"])
        positions = torch.linspace(-radius, radius, kernel_size, device=inpt.device)
        x_coords = (positions * math.cos(angle)).round().to(torch.long) + radius
        y_coords = (positions * math.sin(angle)).round().to(torch.long) + radius
        kernel = torch.zeros((kernel_size, kernel_size), device=inpt.device, dtype=inpt.dtype)
        kernel[y_coords, x_coords] = 1
        kernel /= kernel.sum()

        channels, height, width = inpt.shape[-3:]
        flat_input = inpt.reshape(-1, channels, height, width)
        depthwise_kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        padded = torch.nn.functional.pad(flat_input, (radius,) * 4, mode="replicate")
        output = torch.nn.functional.conv2d(padded, depthwise_kernel, groups=channels)
        return output.reshape(inpt.shape).clamp(0.0, 1.0)


class JPEGCompression(Transform):
    """模拟 JPEG 压缩伪影（块状伪影、色带）。

    模拟网络串流相机画面中视频压缩造成的质量下降。

    Args:
        quality: JPEG 质量因子的范围 (min, max)（越低 = 伪影越多）。
    """

    def __init__(self, quality: int | Sequence[int] = (15, 75)) -> None:
        super().__init__()
        if isinstance(quality, int):
            self.quality = (quality, quality)
        elif isinstance(quality, Sequence) and len(quality) == 2:
            self.quality = (int(quality[0]), int(quality[1]))
        else:
            raise TypeError("quality must be an int or a sequence with length 2.")
        if not 1 <= self.quality[0] <= self.quality[1] <= 100:
            raise ValueError(f"quality must satisfy 1 <= min <= max <= 100, but got {self.quality}.")

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"quality": int(torch.randint(self.quality[0], self.quality[1] + 1, (1,)).item())}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        if inpt.ndim < 3:
            raise ValueError(f"JPEGCompression expects [..., C, H, W] input, but got shape {inpt.shape}.")

        channels, height, width = inpt.shape[-3:]
        if channels not in (1, 3):
            raise ValueError(f"JPEGCompression expects 1 or 3 channels, but got {channels}.")

        flat_input = inpt.reshape(-1, channels, height, width)
        flat_uint8 = (flat_input.clamp(0.0, 1.0) * 255).round().to(torch.uint8).cpu()
        decoded_frames = [
            decode_image(encode_jpeg(frame, quality=params["quality"])) for frame in flat_uint8.unbind()
        ]
        output = torch.stack(decoded_frames).to(device=inpt.device, dtype=inpt.dtype) / 255.0
        return output.reshape(inpt.shape)


class GaussianPatchBrightness(Transform):
    """使用高斯斑块应用空间变化的亮度。

    模拟在具有多个光源的真实机器人工作空间中常见的
    不均匀顶部照明、聚光灯和阴影斑块。

    Args:
        num_patches: 亮度斑块数量的范围 (min, max)。
        sigma_range: 高斯 sigma 的范围，以图像尺寸的比例表示。
        factor_range: 亮度因子的范围（< 1 变暗，> 1 变亮）。
    """

    def __init__(
        self,
        num_patches: int | Sequence[int] = (1, 4),
        sigma_range: Sequence[float] = (0.05, 0.25),
        factor_range: Sequence[float] = (0.4, 1.6),
    ) -> None:
        super().__init__()
        if isinstance(num_patches, int):
            self.num_patches = (num_patches, num_patches)
        elif isinstance(num_patches, Sequence) and len(num_patches) == 2:
            self.num_patches = (int(num_patches[0]), int(num_patches[1]))
        else:
            raise TypeError("num_patches must be an int or a sequence with length 2.")
        if not 1 <= self.num_patches[0] <= self.num_patches[1]:
            raise ValueError(f"num_patches must satisfy 1 <= min <= max, but got {self.num_patches}.")
        if not isinstance(sigma_range, Sequence) or len(sigma_range) != 2:
            raise TypeError("sigma_range must be a sequence with length 2.")
        self.sigma_range = (float(sigma_range[0]), float(sigma_range[1]))
        if not 0.0 < self.sigma_range[0] <= self.sigma_range[1]:
            raise ValueError(f"sigma_range must satisfy 0 < min <= max, but got {self.sigma_range}.")
        if not isinstance(factor_range, Sequence) or len(factor_range) != 2:
            raise TypeError("factor_range must be a sequence with length 2.")
        self.factor_range = (float(factor_range[0]), float(factor_range[1]))
        if not 0.0 <= self.factor_range[0] <= self.factor_range[1]:
            raise ValueError(f"factor_range must satisfy 0 <= min <= max, but got {self.factor_range}.")

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        n = int(torch.randint(self.num_patches[0], self.num_patches[1] + 1, (1,)).item())
        return {
            "centers": torch.rand(n, 2).tolist(),
            "sigmas": torch.empty(n).uniform_(self.sigma_range[0], self.sigma_range[1]).tolist(),
            "factors": torch.empty(n).uniform_(self.factor_range[0], self.factor_range[1]).tolist(),
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        h, w = inpt.shape[-2:]
        mask = torch.ones(h, w, device=inpt.device, dtype=inpt.dtype)
        grid_y = torch.linspace(0, 1, h, device=inpt.device, dtype=inpt.dtype)
        grid_x = torch.linspace(0, 1, w, device=inpt.device, dtype=inpt.dtype)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        for (cy, cx), sigma, factor in zip(
            params["centers"], params["sigmas"], params["factors"], strict=True
        ):
            gauss = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
            mask = mask * (1.0 + (factor - 1.0) * gauss)
        broadcast_shape = (1,) * (inpt.ndim - 2) + (h, w)
        return (inpt * mask.reshape(broadcast_shape)).clamp(0.0, 1.0)


class RandomShadow(Transform):
    """添加边缘平滑的随机垂直条带阴影。

    模拟机器人工作空间附近物体或人员投射的阴影。
    对称性：随机变亮或变暗，以防止 BatchNorm 统计量偏移。

    Args:
        opacity: 阴影/高亮不透明度的范围 (min, max)。
    """

    def __init__(self, opacity: float | Sequence[float] = (0.3, 0.6)) -> None:
        super().__init__()
        if isinstance(opacity, (int, float)):
            self.opacity = (float(opacity), float(opacity))
        elif isinstance(opacity, Sequence) and len(opacity) == 2:
            self.opacity = (float(opacity[0]), float(opacity[1]))
        else:
            raise TypeError("opacity must be a number or a sequence with length 2.")
        if not 0.0 <= self.opacity[0] <= self.opacity[1] <= 1.0:
            raise ValueError(f"opacity must satisfy 0 <= min <= max <= 1, but got {self.opacity}.")

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {
            "opacity": torch.empty(1).uniform_(self.opacity[0], self.opacity[1]).item(),
            "start": torch.rand(1).item(),
            "width": torch.empty(1).uniform_(1 / 3, 2 / 3).item(),
            "direction": -1.0 if torch.rand(1).item() < 0.5 else 1.0,
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        if inpt.ndim < 3:
            raise ValueError(f"RandomShadow expects [..., C, H, W] input, but got shape {inpt.shape}.")

        h, w = inpt.shape[-2:]
        band_width = max(1, min(w, round(params["width"] * w)))
        x_start = round(params["start"] * (w - band_width))
        x_end = x_start + band_width
        mask = torch.ones(h, w, device=inpt.device, dtype=inpt.dtype)
        mask[:, x_start:x_end] = 1.0 + params["direction"] * params["opacity"]

        smoothing_size = min(8, h, w)
        if smoothing_size > 1:
            batched_mask = mask[None, None]
            small = torch.nn.functional.avg_pool2d(batched_mask, smoothing_size, stride=smoothing_size)
            mask = torch.nn.functional.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)[
                0, 0
            ]

        broadcast_shape = (1,) * (inpt.ndim - 2) + (h, w)
        return (inpt * mask.reshape(broadcast_shape)).clamp(0.0, 1.0)


class CoarseDropout(Transform):
    """丢弃随机矩形斑块以模拟部分遮挡。

    模拟机器人操作过程中物体、手部或线缆
    穿过相机视野的情况。

    Args:
        max_holes: 要丢弃的矩形斑块的最大数量。
        max_height_frac: 斑块最大高度，以图像高度的比例表示。
        max_width_frac: 斑块最大宽度，以图像宽度的比例表示。
        fill_value: 用于填充丢弃区域的值。
    """

    def __init__(
        self,
        max_holes: int = 8,
        max_height_frac: float = 0.07,
        max_width_frac: float = 0.07,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(max_holes, int):
            raise TypeError("max_holes must be an int.")
        if max_holes < 1:
            raise ValueError(f"max_holes must be at least 1, but got {max_holes}.")
        if not 0.0 < max_height_frac <= 1.0:
            raise ValueError(f"max_height_frac must be in (0, 1], but got {max_height_frac}.")
        if not 0.0 < max_width_frac <= 1.0:
            raise ValueError(f"max_width_frac must be in (0, 1], but got {max_width_frac}.")
        if not 0.0 <= fill_value <= 1.0:
            raise ValueError(f"fill_value must be in [0, 1], but got {fill_value}.")
        self.max_holes = max_holes
        self.max_height_frac = max_height_frac
        self.max_width_frac = max_width_frac
        self.fill_value = fill_value

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        n = int(torch.randint(1, self.max_holes + 1, (1,)).item())
        sizes = torch.rand(n, 2)
        sizes[:, 0] *= self.max_height_frac
        sizes[:, 1] *= self.max_width_frac
        return {"sizes": sizes.tolist(), "positions": torch.rand(n, 2).tolist()}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        if inpt.ndim < 3:
            raise ValueError(f"CoarseDropout expects [..., C, H, W] input, but got shape {inpt.shape}.")

        h, w = inpt.shape[-2:]
        result = inpt.clone()
        for (height_frac, width_frac), (y_frac, x_frac) in zip(
            params["sizes"], params["positions"], strict=True
        ):
            hole_h = max(1, min(h, round(height_frac * h)))
            hole_w = max(1, min(w, round(width_frac * w)))
            y = round(y_frac * (h - hole_h))
            x = round(x_frac * (w - hole_w))
            result[..., y : y + hole_h, x : x + hole_w] = self.fill_value
        return result


class GammaCorrection(Transform):
    """应用随机伽马校正以模拟曝光变化。

    模拟不同的相机自动曝光设置和传感器响应曲线。
    使用对数对称采样，使变亮和变暗的概率相等，
    防止 BatchNorm 统计量偏移。

    Args:
        gamma: 伽马值的范围 (min, max)。值 < 1 变亮，> 1 变暗。
    """

    def __init__(self, gamma: float | Sequence[float] = (0.5, 2.0)) -> None:
        super().__init__()
        if isinstance(gamma, (int, float)):
            gamma = float(gamma)
            if gamma <= 0:
                raise ValueError(f"gamma must be positive, but got {gamma}.")
            self.gamma = (min(gamma, 1.0 / gamma), max(gamma, 1.0 / gamma))
        elif isinstance(gamma, Sequence) and len(gamma) == 2:
            self.gamma = (float(gamma[0]), float(gamma[1]))
        else:
            raise TypeError("gamma must be a number or a sequence with length 2.")
        if not 0.0 < self.gamma[0] <= self.gamma[1]:
            raise ValueError(f"gamma must satisfy 0 < min <= max, but got {self.gamma}.")

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        log_lo = math.log(self.gamma[0])
        log_hi = math.log(self.gamma[1])
        gamma = math.exp(torch.empty(1).uniform_(log_lo, log_hi).item())
        return {"gamma": gamma}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, torch.Tensor) and inpt.is_floating_point():
            return inpt.pow(params["gamma"]).clamp(0.0, 1.0)
        return inpt


# 来自论文作者的 MIT 许可参考实现：
# https://github.com/TheZino/PlanckianJitter
_PLANCKIAN_BLACKBODY_COEFFICIENTS = (
    (0.6743, 0.4029, 0.0013),
    (0.6281, 0.4241, 0.1665),
    (0.5919, 0.4372, 0.2513),
    (0.5623, 0.4457, 0.3154),
    (0.5376, 0.4515, 0.3672),
    (0.5163, 0.4555, 0.4103),
    (0.4979, 0.4584, 0.4468),
    (0.4816, 0.4604, 0.4782),
    (0.4672, 0.4619, 0.5053),
    (0.4542, 0.4630, 0.5289),
    (0.4426, 0.4638, 0.5497),
    (0.4320, 0.4644, 0.5681),
    (0.4223, 0.4648, 0.5844),
    (0.4135, 0.4651, 0.5990),
    (0.4054, 0.4653, 0.6121),
    (0.3980, 0.4654, 0.6239),
    (0.3911, 0.4655, 0.6346),
    (0.3847, 0.4656, 0.6444),
    (0.3787, 0.4656, 0.6532),
    (0.3732, 0.4656, 0.6613),
    (0.3680, 0.4655, 0.6688),
    (0.3632, 0.4655, 0.6756),
    (0.3586, 0.4655, 0.6820),
    (0.3544, 0.4654, 0.6878),
    (0.3503, 0.4653, 0.6933),
)
_PLANCKIAN_MIN_TEMPERATURE = 3_000
_PLANCKIAN_MAX_TEMPERATURE = 15_000
_PLANCKIAN_TEMPERATURE_STEP = 500


class PlanckianJitter(Transform):
    """模拟沿普朗克轨迹的色温偏移。

    采样一个黑体温度，并应用相应的相关红通道和蓝通道缩放，
    同时保持绿通道不变。表格中 500 K 间隔之间的系数通过线性插值得到。

    参考文献：Zini et al., "Planckian Jitter", CVPR 2022 Workshop.

    Args:
        temperature: 固定的色温或以开尔文为单位的范围。支持的值
            在 3000 K 到 15000 K 之间。
    """

    def __init__(self, temperature: int | Sequence[int] = (3_000, 15_000)) -> None:
        super().__init__()
        if isinstance(temperature, int):
            self.temperature = (temperature, temperature)
        elif isinstance(temperature, Sequence) and len(temperature) == 2:
            self.temperature = (int(temperature[0]), int(temperature[1]))
        else:
            raise TypeError("temperature must be an int or a sequence with length 2.")
        if not (
            _PLANCKIAN_MIN_TEMPERATURE
            <= self.temperature[0]
            <= self.temperature[1]
            <= _PLANCKIAN_MAX_TEMPERATURE
        ):
            raise ValueError(
                "temperature must satisfy "
                f"{_PLANCKIAN_MIN_TEMPERATURE} <= min <= max <= {_PLANCKIAN_MAX_TEMPERATURE}, "
                f"but got {self.temperature}."
            )

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        temperature = int(torch.randint(self.temperature[0], self.temperature[1] + 1, ()).item())
        return {"temperature": temperature}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        if inpt.ndim < 3 or inpt.shape[-3] != 3:
            raise ValueError(f"PlanckianJitter expects [..., 3, H, W] input, but got shape {inpt.shape}.")

        table_position = (params["temperature"] - _PLANCKIAN_MIN_TEMPERATURE) / _PLANCKIAN_TEMPERATURE_STEP
        left_index = math.floor(table_position)
        right_index = min(left_index + 1, len(_PLANCKIAN_BLACKBODY_COEFFICIENTS) - 1)
        interpolation_weight = table_position - left_index

        left = torch.tensor(
            _PLANCKIAN_BLACKBODY_COEFFICIENTS[left_index],
            device=inpt.device,
            dtype=inpt.dtype,
        )
        right = torch.tensor(
            _PLANCKIAN_BLACKBODY_COEFFICIENTS[right_index],
            device=inpt.device,
            dtype=inpt.dtype,
        )
        coefficients = torch.lerp(left, right, interpolation_weight)
        scale = torch.stack(
            (
                coefficients[0] / coefficients[1],
                coefficients.new_tensor(1.0),
                coefficients[2] / coefficients[1],
            )
        )
        broadcast_shape = (1,) * (inpt.ndim - 3) + (3, 1, 1)
        return (inpt * scale.reshape(broadcast_shape)).clamp(0.0, 1.0)


_CUSTOM_TRANSFORMS: dict[str, type[Transform]] = {
    "SharpnessJitter": SharpnessJitter,
    "GaussianNoise": GaussianNoise,
    "MotionBlur": MotionBlur,
    "JPEGCompression": JPEGCompression,
    "GaussianPatchBrightness": GaussianPatchBrightness,
    "RandomShadow": RandomShadow,
    "CoarseDropout": CoarseDropout,
    "GammaCorrection": GammaCorrection,
    "PlanckianJitter": PlanckianJitter,
}


@dataclass
class ImageTransformConfig:
    """
    对于每个变换，以下参数可用：
      weight: 表示用于采样该变换的多项分布概率（无放回）。
            如果权重之和不为 1，将被归一化。
      type: 所用类的名称。可以是 torchvision.transforms.v2 下可用的类，
            也可以是此处定义的自定义变换。
      kwargs: 应用变换时分别用于采样变换参数（按均匀分布）的
            下界和上界。
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """
    这些变换都使用标准的 torchvision.transforms.v2
    你可以在这里查看这些变换对图像的影响：
    https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    我们使用自定义的 RandomSubsetApply 容器对它们进行采样。
    """

    # 将此标志设为 `true` 以在训练期间启用变换
    enable: bool = False
    # 这是将应用于每一帧的最大变换数量（从下方这些变换中采样）。
    # 它是区间 [1, number_of_available_transforms] 内的整数。
    max_num_transforms: int = 3
    # 默认情况下，变换按 Torchvision 建议的顺序（如下所示）应用。
    # 将此设为 True 则以随机顺序应用。
    random_order: bool = False
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            ),
        }
    )


def make_transform_from_config(cfg: ImageTransformConfig) -> Transform:
    if cfg.type in _CUSTOM_TRANSFORMS:
        return _CUSTOM_TRANSFORMS[cfg.type](**cfg.kwargs)

    transform_cls = getattr(v2, cfg.type, None)
    if isinstance(transform_cls, type) and issubclass(transform_cls, Transform):
        return transform_cls(**cfg.kwargs)

    valid_custom = ", ".join(sorted(_CUSTOM_TRANSFORMS.keys()))
    raise ValueError(
        f"Transform '{cfg.type}' is not valid. It must be a class in "
        f"torchvision.transforms.v2 or one of: {valid_custom}."
    )


class ImageTransforms(Transform):
    """根据配置组合图像变换的类。"""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg

        self.weights: list[float] = []
        self.transforms: dict[str, Transform] = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)
