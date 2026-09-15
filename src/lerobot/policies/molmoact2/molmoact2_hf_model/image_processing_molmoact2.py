# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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


"""MolmoAct2 的图像处理器类"""

import einops
import numpy as np
import torch
import torchvision.transforms
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor, get_size_dict
from transformers.image_transforms import convert_to_rgb
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.processing_utils import ImagesKwargs
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)


def normalize_image(
    image: np.ndarray,
    image_mean: list[float],
    image_std: list[float],
) -> np.ndarray:
    if np.allclose(image_mean, [0.5, 0.5, 0.5]) and np.allclose(image_std, [0.5, 0.5, 0.5]):
        return image * np.asarray(2.0, dtype=np.float32) - np.asarray(1.0, dtype=np.float32)
    image -= np.array(image_mean, dtype=np.float32)[None, None, :]
    image /= np.array(image_std, dtype=np.float32)[None, None, :]
    return image


def resize_image(
    image: np.ndarray,
    desired_output_size: list[int],
    resample: PILImageResampling,
) -> np.ndarray:
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    dtype = image.dtype
    if torch.is_floating_point(image):
        in_min = 0.0
        in_max = 1.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            resample,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(dtype)
    else:
        assert image.dtype == torch.uint8, (
            f"SigLIP expects float images or uint8 images, but got {image.dtype}"
        )
        in_min = 0.0
        in_max = 255.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            resample,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0, 255).to(dtype)

    resized = resized.to(torch.float32)
    resized = (resized - in_min) / (in_max - in_min)

    resized = torch.permute(resized, [1, 2, 0]).numpy()

    return resized


def select_tiling(h, w, patch_size, max_num_crops):
    """将尺寸为 [w, h] 的图像划分为最多 max_num_patches 个尺寸为 patch_size 的块"""
    original_size = np.stack([h, w])  # [1, 2]
    tilings = []
    for i in range(1, max_num_crops + 1):
        for j in range(1, max_num_crops + 1):
            if i * j <= max_num_crops:
                tilings.append((i, j))
    # 排序，使得出现并列时 argmin 和 argmax 倾向于更小的平铺方案
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)  # [n_resolutions, 2]
    candidate_resolutions = candidate_tilings * patch_size  # [n_resolutions, 2]

    # 要使图像恰好放入每种平铺方案所需的缩放比例
    original_size = np.stack([h, w], dtype=np.float32)  # [1, 2]

    # 在罕见情况下，如果图像比边距还小，原始尺寸可能为零
    # 在这些情况下，让缩放比例变为无穷大意味着平铺将基于
    # 另一条边，或者回退到最小的平铺方案
    with np.errstate(divide="ignore"):
        required_scale_d = (candidate_resolutions.astype(np.float32) / original_size,)
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if np.all(required_scale < 1):
        # 我们被迫缩小图像，因此尽量减少缩小的幅度
        ix = np.argmax(required_scale)
    else:
        # 选择所需放大倍数最小的分辨率，使其最贴近图像
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)
    return candidate_tilings[ix]


def build_resized_image(
    image: np.ndarray,
    base_image_input_size: list[int],
    resample: PILImageResampling,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    resized = resize_image(
        image,
        base_image_input_size,
        resample,
    )
    resized = normalize_image(resized, image_mean, image_std)
    if len(resized.shape) == 3:
        resized = np.expand_dims(resized, 0)
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    resize_idx = np.arange(crop_patch_w * crop_patch_h).reshape([crop_patch_h, crop_patch_w])
    return resized, resize_idx


def build_overlapping_crops(
    image: np.ndarray,
    max_crops: int,
    overlap_margins: list[int],
    base_image_input_size: list[int],
    resample: PILImageResampling,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """将图像分解为一组相互重叠的裁剪块

    :return crop_arr: [n_crops, h, w, 3] 裁剪块
    :return patch_idx: [overlap_patch_h, overlap_patch_w] 对于裁剪来源的缩放图像中的
                        每个块，记录它对应 `crop_arr` 中的哪个块
    """
    original_image_h, original_image_w = image.shape[:2]
    crop_size = base_image_input_size[0]
    assert base_image_input_size[0] == base_image_input_size[1]

    left_margin, right_margin = overlap_margins
    total_margin_pixels = image_patch_size * (right_margin + left_margin)  # 每个维度上移除的像素数
    crop_patches = base_image_input_size[0] // image_patch_size  # 每个裁剪维度上的块数
    crop_window_patches = crop_patches - (right_margin + left_margin)  # 可用的块数
    crop_window_size = crop_window_patches * image_patch_size
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    original_image_h, original_image_w = image.shape[:2]
    crop_size = base_image_input_size[0]

    # 决定如何平铺图像。为了处理重叠边距，我们按照“图像不含边距、
    # 且使用不含边距的裁剪尺寸”的方式来计算平铺
    tiling = select_tiling(
        original_image_h - total_margin_pixels,
        original_image_w - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    src = resize_image(
        image,
        [
            tiling[0] * crop_window_size + total_margin_pixels,
            tiling[1] * crop_window_size + total_margin_pixels,
        ],
        resample,
    )
    src = normalize_image(src, image_mean, image_std)

    # 现在需要将图像切分为多个裁剪块，并在 `patch_idx_arr` 中记录
    # 各个块来自哪里
    n_crops = tiling[0] * tiling[1]
    crop_arr = np.zeros([n_crops, crop_size, crop_size, 3], dtype=src.dtype)
    patch_idx_arr = np.zeros([n_crops, crop_patch_h, crop_patch_w], dtype=np.int32)
    on_crop = 0
    for i in range(tiling[0]):
        # 以 `crop_window_size` 为步长在 `src` 上滑动，但提取尺寸为 `crop_size`
        # 的裁剪块，从而产生重叠的裁剪窗口
        y0 = i * crop_window_size
        for j in range(tiling[1]):
            x0 = j * crop_window_size
            crop_arr[on_crop] = src[y0 : y0 + crop_size, x0 : x0 + crop_size]
            patch_idx = np.arange(crop_patch_w * crop_patch_h).reshape(crop_patch_h, crop_patch_w)
            patch_idx += on_crop * crop_patch_h * crop_patch_w

            # 屏蔽位于重叠区域内的索引
            if i != 0:
                patch_idx[:left_margin, :] = -1
            if j != 0:
                patch_idx[:, :left_margin] = -1
            if i != tiling[0] - 1:
                patch_idx[-right_margin:, :] = -1
            if j != tiling[1] - 1:
                patch_idx[:, -right_margin:] = -1
            patch_idx_arr[on_crop] = patch_idx
            on_crop += 1

    # `patch_idx_arr` 按裁剪块逐个排序，这里对 `patch_idx_arr` 做转置，
    # 使其按从左到右的顺序排列
    patch_idx_arr = np.reshape(patch_idx_arr, [tiling[0], tiling[1], crop_patch_h, crop_patch_w])
    patch_idx_arr = np.transpose(patch_idx_arr, [0, 2, 1, 3])
    patch_idx_arr = np.reshape(patch_idx_arr, [-1])

    # 现在取出不在重叠区域内的部分，这样它就能将 `src` 中的每个块
    # 映射到 `crop_arr` 中其应来源的正确块
    patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(
        src.shape[0] // image_patch_size,
        src.shape[1] // image_patch_size,
    )
    return crop_arr, patch_idx_arr


def batch_pixels_to_patches(array: np.ndarray, patch_size: int) -> np.ndarray:
    """将 [n_images, h, w, 3] 的图像重塑为 [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, h, w = array.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(array, [n_crops, h_patches * w_patches, patch_size * patch_size])
        return array
    else:
        n_crops, h, w, c = array.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
        array = np.transpose(array, [0, 1, 3, 2, 4, 5])
        array = np.reshape(array, [n_crops, h_patches * w_patches, patch_size * patch_size * c])
        return array


def arange_for_pooling(
    idx_arr: np.ndarray,
    pool_h: int,
    pool_w: int,
) -> np.ndarray:
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(
        idx_arr,
        [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
        mode="constant",
        constant_values=-1,
    )
    return einops.rearrange(idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)


def image_to_patches_and_grids(
    image: np.ndarray,
    max_crops: int,
    overlap_margins: list[int],
    base_image_input_size: list[int],
    resample: PILImageResampling,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
    image_pooling_w: int,
    image_pooling_h: int,
    crop_mode: str = "overlap-and-resize-c2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :return image_grids，每张（低分辨率、高分辨率）图像池化后的形状
    :return crops，要用 ViT 处理的图像裁剪块
    :return pooled_patch_idx，对于 `image_tokens` 中的每个 patch_id token，
                                为该 token 做池化时所用 `crops` 中块的索引，用 -1 屏蔽
    """
    if isinstance(base_image_input_size, int):
        base_image_input_size = (base_image_input_size, base_image_input_size)

    base_image_input_d = image_patch_size
    pooling_w = image_pooling_w
    pooling_h = image_pooling_h
    crop_patch_w = base_image_input_size[1] // base_image_input_d
    crop_patch_h = base_image_input_size[0] // base_image_input_d

    if crop_mode == "resize":
        resized, resize_idx = build_resized_image(
            image,
            base_image_input_size,
            resample,
            image_mean,
            image_std,
            image_patch_size,
        )
        resize_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
        resized_h, resized_w = resize_idx.shape[:2]
        resize_idx = resize_idx.reshape([-1, pooling_h * pooling_w])
        image_grid = [np.array([resized_h, resized_w, 0, 0])]
        return (
            np.stack(image_grid, 0),
            batch_pixels_to_patches(resized, image_patch_size),
            resize_idx,
        )

    if crop_mode not in {"overlap-and-resize-c2", "overlap-and-resize"}:
        raise ValueError(f"Unsupported MolmoAct2 image crop_mode {crop_mode!r}.")

    crop_arr, patch_idx_arr = build_overlapping_crops(
        image,
        max_crops,
        overlap_margins,
        base_image_input_size,
        resample,
        image_mean,
        image_std,
        image_patch_size,
    )
    pooling_idx = arange_for_pooling(patch_idx_arr, pooling_h, pooling_w)
    h, w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape([-1, pooling_h * pooling_w])

    # 最后对全局图像执行同样的操作
    resized, resize_idx = build_resized_image(
        image,
        base_image_input_size,
        resample,
        image_mean,
        image_std,
        image_patch_size,
    )
    crop_arr = np.concatenate([resized, crop_arr], 0)

    resize_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
    resized_h, resized_w = resize_idx.shape[:2]
    resize_idx = resize_idx.reshape([-1, pooling_h * pooling_w])

    # 全局图像排在最前面，因此前面各裁剪块中块的序号相应后移
    pooling_idx = np.where(pooling_idx >= 0, pooling_idx + crop_patch_h * crop_patch_w, -1)
    pooling_idx = np.concatenate([resize_idx, pooling_idx])
    image_grid = [np.array([resized_h, resized_w, h, w])]

    return (np.stack(image_grid, 0), batch_pixels_to_patches(crop_arr, image_patch_size), pooling_idx)


class MolmoAct2ImagesKwargs(ImagesKwargs, total=False):
    max_crops: int | None
    overlap_margins: list[int] | None
    crop_mode: str | None
    patch_size: int | None
    pooling_size: list[int] | None


class MolmoAct2ImageProcessor(BaseImageProcessor):
    r"""
    构造一个 MolmoAct2 图像处理器，用于为模型预处理图像。

    参数：
        size (`dict[str, int]`，*可选*，默认为 `{"height": 378, "width": 378}`)：
            图像缩放后的尺寸。
        resample (`PILImageResampling`，*可选*，默认为 `Resampling.BILINEAR`)：
            缩放图像时使用的重采样滤波器。
        image_mean (`float` 或 `list[float]`，*可选*，默认为 `[0.5, 0.5, 0.5]`)：
            归一化图像时使用的均值。这是一个浮点数，或对应图像每个通道的浮点数列表。
        image_std (`float` 或 `list[float]`，*可选*，默认为 `[0.5, 0.5, 0.5]`)：
            归一化图像时使用的标准差。这是一个浮点数，或对应图像每个通道的浮点数列表。
        do_convert_rgb (`bool`，*可选*，默认为 `True`)：
            是否将图像转换为 RGB。
        max_crops (`int`，*可选*，默认为 `8`)：
            每张图像最多使用的裁剪块数量。
        overlap_margins (`list[int]`，*可选*，默认为 `[4, 4]`)：
            使用的重叠边距。
        patch_size (`int`，*可选*，默认为 14)：
            视觉编码器的空间块尺寸。
        pooling_size (`list[int]`，*可选*，默认为 `[2, 2]`)：
            视觉适配器的池化尺寸。
    """

    model_input_names = ["pixel_values", "image_token_pooling", "image_grids", "image_num_crops"]

    def __init__(
        self,
        size: dict[str, int] | None = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,
        max_crops: int = 8,
        overlap_margins: list[int] | None = None,
        crop_mode: str = "overlap-and-resize-c2",
        patch_size: int = 14,
        pooling_size: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if overlap_margins is None:
            overlap_margins = [4, 4]
        if pooling_size is None:
            pooling_size = [2, 2]
        size = size if size is not None else {"height": 378, "width": 378}
        size = get_size_dict(size, default_to_square=True)
        self.size = size

        self.resample = resample
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_convert_rgb = do_convert_rgb

        self.max_crops = max_crops
        self.overlap_margins = overlap_margins
        self.crop_mode = crop_mode
        self.patch_size = patch_size
        self.pooling_size = pooling_size

    def preprocess(
        self,
        images: ImageInput,
        size: dict[str, int] | None = None,
        resample: PILImageResampling | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool | None = None,
        max_crops: int | None = None,
        overlap_margins: list[int] | None = None,
        crop_mode: str | None = None,
        patch_size: int | None = None,
        pooling_size: list[int] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        参数：
            images (`ImageInput`)：
                要预处理的图像。
            size (`dict[str, int]`，*可选*，默认为 `self.size`)：
                图像缩放后的尺寸。
            resample (`PILImageResampling`，*可选*，默认为 `self.resample`)：
                缩放图像时使用的重采样滤波器。可以是 `PILImageResampling` 枚举之一。
                仅在 `do_resize` 设为 `True` 时生效。
            image_mean (`float` 或 `list[float]`，*可选*，默认为 `self.image_mean`)：
                归一化使用的图像均值。仅在 `do_normalize` 设为 `True` 时生效。
            image_std (`float` 或 `list[float]`，*可选*，默认为 `self.image_std`)：
                归一化使用的图像标准差。仅在 `do_normalize` 设为
                `True` 时生效。
            do_convert_rgb (`bool`，*可选*，默认为 `self.do_convert_rgb`)：
                是否将图像转换为 RGB。
            max_crops (`int`，*可选*，默认为 `self.max_crops`)：
                每张图像最多使用的裁剪块数量。
            overlap_margins (`list[int]`，*可选*，默认为 `self.overlap_margins`)：
                使用的重叠边距。
            patch_size (`int`，*可选*，默认为 `self.patch_size`)：
                视觉编码器的空间块尺寸。
            pooling_size (`list[int]`，*可选*，默认为 `self.pooling_size`)：
                视觉适配器的池化尺寸。
            return_tensors (`str` 或 `TensorType`，*可选*)：
                要返回的张量类型。可以是以下之一：
                - 未设置：返回 `np.ndarray` 列表。
                - `TensorType.TENSORFLOW` 或 `'tf'`：返回 `tf.Tensor` 类型的批次。
                - `TensorType.PYTORCH` 或 `'pt'`：返回 `torch.Tensor` 类型的批次。
                - `TensorType.NUMPY` 或 `'np'`：返回 `np.ndarray` 类型的批次。
                - `TensorType.JAX` 或 `'jax'`：返回 `jax.numpy.ndarray` 类型的批次。

        返回：
            包含以下键的 `BatchFeature`：
                - `pixel_values`：预处理后的图像。
                - `image_token_pooling`：`image_tokens` 中每个 token 做池化时所用 `crops` 中块的索引。
                - `image_grids`：图像网格。
                - `image_num_crops`：每张图像的裁剪块数量。
        """
        if size is not None:
            if "height" not in size or "width" not in size:
                raise ValueError("size must contain 'height' and 'width' keys.")
        else:
            size = {**self.size}

        base_image_input_size = [size["height"], size["width"]]

        resample = resample or self.resample
        image_mean = image_mean or self.image_mean
        image_std = image_std or self.image_std
        do_convert_rgb = do_convert_rgb or self.do_convert_rgb

        max_crops = max_crops or self.max_crops
        overlap_margins = overlap_margins or self.overlap_margins
        crop_mode = crop_mode or self.crop_mode
        patch_size = patch_size or self.patch_size
        pooling_size = pooling_size or self.pooling_size

        image_pooling_h, image_pooling_w = pooling_size

        if images is not None:
            images = self.fetch_images(images)
            images = make_flat_list_of_images(images)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # 所有变换都期望输入 numpy 数组。
        images = [to_numpy_array(image) for image in images]

        data = {}
        if images is not None:
            batch_grids = []
            batch_crops = []
            batch_pooled_patches_idx = []
            batch_num_crops = []

            for image in images:
                image_grid, crops, pooled_idx = image_to_patches_and_grids(
                    image,
                    max_crops,
                    overlap_margins,
                    base_image_input_size,
                    resample,
                    image_mean,
                    image_std,
                    patch_size,
                    image_pooling_w,
                    image_pooling_h,
                    crop_mode,
                )
                batch_grids.append(image_grid)
                batch_crops.append(crops)
                batch_pooled_patches_idx.append(pooled_idx)
                batch_num_crops.append(crops.shape[0])

            pixel_values = np.concatenate(batch_crops, 0)
            image_token_pooling = np.concatenate(batch_pooled_patches_idx, 0)
            image_grids = np.concatenate(batch_grids, 0)
            image_num_crops = np.array(batch_num_crops)

            data.update(
                pixel_values=pixel_values,
                image_token_pooling=image_token_pooling,
                image_grids=image_grids,
                image_num_crops=image_num_crops,
            )

        return BatchFeature(data, tensor_type=return_tensors)


MolmoAct2ImageProcessor.register_for_auto_class()
