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
from __future__ import annotations

import logging

import numpy as np

from lerobot.configs import is_depth_map
from lerobot.processor import RelativeActionsProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

from .io_utils import load_image_as_numpy

DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]


class RunningQuantileStats:
    """
    维护向量批次的运行统计量，包括均值、标准差、
    最小值、最大值和近似分位数。

    统计量按特征维度计算，并随着观测到新批次而增量更新。
    分位数使用直方图估计，当观测到的数据范围扩大时
    会动态自适应。
    """

    def __init__(self, quantile_list: list[float] | None = None, num_quantile_bins: int = 5000):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = num_quantile_bins

        self._quantile_list = quantile_list
        if self._quantile_list is None:
            self._quantile_list = DEFAULT_QUANTILES
        self._quantile_keys = [f"q{int(q * 100):02d}" for q in self._quantile_list]

    def update(self, batch: np.ndarray) -> None:
        """使用一批向量更新运行统计量。

        Args:
            batch: 除最后一维外所有维度均为批次维度的数组。
        """
        batch = batch.reshape(-1, batch.shape[-1])
        # 在计算平方统计量之前，将整数和低精度输入提升精度。
        batch = batch.astype(np.result_type(batch.dtype, np.float32), copy=False)
        num_elements, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")

            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # 更新运行均值和平方均值
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch)

    def get_statistics(self) -> dict[str, np.ndarray]:
        """计算并返回到目前为止所处理向量的统计量。

        Args:
            quantiles: 要计算的分位数列表（例如 [0.01, 0.10, 0.50, 0.90, 0.99]）。如果为 None，则不计算分位数。

        Returns:
            包含所计算统计量的字典。
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2

        stddev = np.sqrt(np.maximum(0, variance))

        stats = {
            "min": self._min.copy(),
            "max": self._max.copy(),
            "mean": self._mean.copy(),
            "std": stddev,
            "count": np.array([self._count]),
        }

        quantile_results = self._compute_quantiles()
        for i, q in enumerate(self._quantile_keys):
            stats[q] = quantile_results[i]

        return stats

    def _adjust_histograms(self):
        """当最小值或最大值发生变化时调整直方图。"""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]

            # 创建带有少量填充的新边界以确保范围覆盖
            padding = (self._max[i] - self._min[i]) * 1e-10
            new_edges = np.linspace(
                self._min[i] - padding, self._max[i] + padding, self._num_quantile_bins + 1
            )

            # 将现有直方图计数重新分配到新的分箱
            # 需要将每个旧分箱中心映射到新的分箱
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2
            new_hist = np.zeros(self._num_quantile_bins)

            for old_center, count in zip(old_centers, old_hist, strict=False):
                if count > 0:
                    # 找出此旧中心属于哪个新分箱
                    bin_idx = np.searchsorted(new_edges, old_center) - 1
                    bin_idx = max(0, min(bin_idx, self._num_quantile_bins - 1))
                    new_hist[bin_idx] += count

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """使用新向量更新直方图。"""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self) -> list[np.ndarray]:
        """基于直方图计算分位数。"""
        results = []
        for q in self._quantile_list:
            target_count = q * self._count
            q_values = []

            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                q_value = self._compute_single_quantile(hist, edges, target_count)
                q_values.append(q_value)

            results.append(np.array(q_values))
        return results

    def _compute_single_quantile(self, hist: np.ndarray, edges: np.ndarray, target_count: float) -> float:
        """根据直方图和分箱边界计算单个分位数值。"""
        cumsum = np.cumsum(hist)
        idx = np.searchsorted(cumsum, target_count)

        if idx == 0:
            return edges[0]
        if idx >= len(cumsum):
            return edges[-1]

        # 如果不是边界情况，则在分箱内进行插值
        count_before = cumsum[idx - 1]
        count_in_bin = cumsum[idx] - count_before

        # 如果此分箱中没有样本，则使用分箱边界
        if count_in_bin == 0:
            return edges[idx]

        # 分箱内的线性插值
        fraction = (target_count - count_before) / count_in_bin
        return edges[idx] + fraction * (edges[idx + 1] - edges[idx])


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    """根据数据集大小估计样本数量的启发式方法。
    power 控制样本数量相对于数据集大小的增长速度。
    降低 power 可以减少样本数量。

    对于默认参数，有：
    - 从 1 到约 500，num_samples=100
    - 在 1000 时，num_samples=177
    - 在 2000 时，num_samples=299
    - 在 5000 时，num_samples=594
    - 在 10000 时，num_samples=1000
    - 在 20000 时，num_samples=1681
    """
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    _, height, width = img.shape

    if max(width, height) < max_size_threshold:
        # 无需降采样
        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def sample_images(image_paths: list[str]) -> np.ndarray:
    sampled_indices = sample_indices(len(image_paths))

    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        # 我们将 RGB 图像加载为 uint8 以减少内存占用；深度图保持其原生 dtype
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=img.dtype)

        images[i] = img

    return images


def _reshape_stats_by_axis(
    stats: dict[str, np.ndarray],
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    original_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """重塑所有统计量以匹配 NumPy 的输出约定。

    根据 axis 和 keepdims 参数，对所有统计量（'count' 除外）应用一致的
    重塑。这确保统计量具有与原始数据进行广播的正确形状。

    Args:
        stats: 已计算统计量的字典
        axis: 计算统计量时所沿的轴
        keepdims: 是否将被归约的维度保留为大小为 1 的维度
        original_shape: 原始数组的形状

    Returns:
        重塑后统计量的字典

    Note:
        'count' 统计量永远不会被重塑，因为它表示的是元数据
        而不是按特征的统计量。
    """
    if axis == (1,) and not keepdims:
        return stats

    result = {}
    for key, value in stats.items():
        if key == "count":
            result[key] = value
        else:
            result[key] = _reshape_single_stat(value, axis, keepdims, original_shape)

    return result


def _reshape_for_image_stats(value: np.ndarray, keepdims: bool) -> np.ndarray:
    """重塑图像数据的统计量（axis=(0,2,3)）。"""
    if keepdims and value.ndim == 1:
        return value.reshape(1, -1, 1, 1)
    return value


def _reshape_for_vector_stats(
    value: np.ndarray, keepdims: bool, original_shape: tuple[int, ...]
) -> np.ndarray:
    """重塑向量数据的统计量（axis=0 或 axis=(0,)）。"""
    if not keepdims:
        return value

    if len(original_shape) == 1 and value.ndim > 0:
        return value.reshape(1)
    elif len(original_shape) >= 2 and value.ndim == 1:
        return value.reshape(1, -1)
    return value


def _reshape_for_feature_stats(value: np.ndarray, keepdims: bool) -> np.ndarray:
    """重塑按特征计算的统计量（axis=(1,)）。"""
    if not keepdims:
        return value

    if value.ndim == 0:
        return value.reshape(1, 1)
    elif value.ndim == 1:
        return value.reshape(-1, 1)
    return value


def _reshape_for_global_stats(
    value: np.ndarray, keepdims: bool, original_shape: tuple[int, ...]
) -> np.ndarray | float:
    """重塑全局归约的统计量（axis=None）。"""
    if keepdims:
        target_shape = tuple(1 for _ in original_shape)
        return value.reshape(target_shape)
    # 至少保持一维数组以满足验证器的要求
    return np.atleast_1d(value)


def _reshape_single_stat(
    value: np.ndarray, axis: int | tuple[int, ...] | None, keepdims: bool, original_shape: tuple[int, ...]
) -> np.ndarray | float:
    """对单个统计量数组应用适当的重塑。

    该函数根据 axis 配置和 keepdims 参数，将统计量数组变换为
    期望的输出形状。

    Args:
        value: 要重塑的统计量数组
        axis: 计算过程中被归约的轴
        keepdims: 是否将被归约的维度保留为大小为 1 的维度
        original_shape: 归约前原始数据的形状

    Returns:
        遵循 NumPy 广播约定的重塑后数组

    """
    if axis == (0, 2, 3):
        return _reshape_for_image_stats(value, keepdims)

    if axis in [0, (0,)]:
        return _reshape_for_vector_stats(value, keepdims, original_shape)

    if axis == (1,):
        return _reshape_for_feature_stats(value, keepdims)

    if axis is None:
        return _reshape_for_global_stats(value, keepdims, original_shape)

    return value


def _prepare_array_for_stats(array: np.ndarray, axis: int | tuple[int, ...] | None) -> tuple[np.ndarray, int]:
    """根据 axis 重塑数组，为统计量计算做准备。

    Args:
        array: 输入数据数组
        axis: 计算统计量时所沿的轴

    Returns:
        (reshaped_array, sample_count) 元组
    """
    if axis == (0, 2, 3):  # 图像数据
        batch_size, channels, height, width = array.shape
        reshaped = array.transpose(0, 2, 3, 1).reshape(-1, channels)
        return reshaped, batch_size

    if axis == 0 or axis == (0,):  # 向量数据
        reshaped = array
        if array.ndim == 1:
            reshaped = array.reshape(-1, 1)
        return reshaped, array.shape[0]

    if axis == (1,):  # 按特征的统计量
        return array.T, array.shape[1]

    if axis is None:  # 全局统计量
        reshaped = array.reshape(-1, 1)
        # 为了向后兼容，count 表示第一维的大小
        return reshaped, array.shape[0] if array.ndim > 0 else 1

    raise ValueError(f"Unsupported axis configuration: {axis}")


def _compute_basic_stats(
    array: np.ndarray, sample_count: int, quantile_list: list[float] | None = None
) -> dict[str, np.ndarray]:
    """为样本数不足以计算分位数的数组计算基本统计量。

    Args:
        array: 已重塑、可用于统计量计算的数组
        sample_count: 数据所表示的样本数量

    Returns:
        包含基本统计量且分位数设为均值的字典
    """
    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES
    quantile_list_keys = [f"q{int(q * 100):02d}" for q in quantile_list]

    stats = {
        "min": np.min(array, axis=0),
        "max": np.max(array, axis=0),
        "mean": np.mean(array, axis=0),
        "std": np.std(array, axis=0),
        "count": np.array([sample_count]),
    }

    for q in quantile_list_keys:
        stats[q] = stats["mean"].copy()

    return stats


def get_feature_stats(
    array: np.ndarray,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    quantile_list: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """沿指定轴计算数组特征的综合统计量。

    该函数沿指定轴为输入数组计算 min、max、mean、std 和分位数（1%、10%、50%、90%、99%）。
    它处理不同的数据布局：
    - 图像数据：axis=(0,2,3) 计算按通道的统计量
    - 向量数据：axis=0 计算按特征的统计量
    - 按特征：axis=1 计算跨特征的统计量
    - 全局：axis=None 计算整个数组的统计量

    Args:
        array: 输入数据数组，形状适合指定的 axis
        axis: 计算统计量时所沿的轴
            - (0, 2, 3)：用于图像数据（批次、通道、高度、宽度）
            - 0 或 (0,)：用于向量/表格数据（样本、特征）
            - (1,)：用于跨特征计算
            - None：用于整个数组的全局统计量
        keepdims: 如果为 True，被归约的轴保留为大小为 1 的维度

    Returns:
        包含以下内容的字典：
            - 'min'：最小值
            - 'max'：最大值
            - 'mean'：均值
            - 'std'：标准差
            - 'count'：样本数量（形状始终为 (1,)）
            - 'q01'、'q10'、'q50'、'q90'、'q99'：分位数值

    """
    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES

    original_shape = array.shape
    reshaped, sample_count = _prepare_array_for_stats(array, axis)

    if reshaped.shape[0] < 2:
        stats = _compute_basic_stats(reshaped, sample_count, quantile_list)
    else:
        running_stats = RunningQuantileStats()
        running_stats.update(reshaped)
        stats = running_stats.get_statistics()
        stats["count"] = np.array([sample_count])

    stats = _reshape_stats_by_axis(stats, axis, keepdims, original_shape)
    return stats


def compute_episode_stats(
    episode_data: dict[str, list[str] | np.ndarray],
    features: dict,
    quantile_list: list[float] | None = None,
) -> dict:
    """计算一个 episode 中所有特征的综合统计量。

    适当处理不同的数据类型：
    - 图像/视频：从路径中采样，计算按通道的统计量，归一化到 [0,1]
    - 数值数组：计算按特征的统计量
    - 字符串：跳过（不计算统计量）

    Args:
        episode_data: 将特征名映射到数据的字典
            - 对于图像/视频：文件路径列表
            - 对于数值数据：numpy 数组
        features: 描述每个特征的 dtype 和 shape 的字典

    Returns:
        将特征名映射到其统计量字典的字典。
        每个统计量字典包含 min、max、mean、std、count 和分位数。

    Note:
        对于 'image'/'video' 特征，统计量按通道计算，并保留前导的通道轴
        （例如 RGB 的形状为 (3, 1, 1)）。RGB 统计量除以 255 以落入 [0, 1]；
        深度图（标记为 ``is_depth_map`` 的特征）跳过此缩放，
        保持其存储单位（存储在 ``depth_unit`` 中）。
    """
    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES

    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] in {"string", "language"}:
            continue

        if features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            ep_ft_array = data
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        ep_stats[key] = get_feature_stats(
            ep_ft_array, axis=axes_to_reduce, keepdims=keepdims, quantile_list=quantile_list
        )

        if features[key]["dtype"] in ["image", "video"]:
            normalization_factor = 1.0 if is_depth_map(features[key]) else 255.0
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / normalization_factor, axis=0)
                for k, v in ep_stats[key].items()
            }

    return ep_stats


def _validate_stat_value(value: np.ndarray, key: str, feature_key: str) -> None:
    """验证单个统计量值。"""
    if not isinstance(value, np.ndarray):
        raise ValueError(
            f"Stats must be composed of numpy array, but key '{key}' of feature '{feature_key}' "
            f"is of type '{type(value)}' instead."
        )

    if value.ndim == 0:
        raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")

    if key == "count" and value.shape != (1,):
        raise ValueError(f"Shape of 'count' must be (1), but is {value.shape} instead.")

    if "image" in feature_key and key != "count" and value.shape not in ((3, 1, 1), (1, 1, 1)):
        raise ValueError(
            f"Shape of quantile '{key}' must be (3,1,1) or (1,1,1) but is {value.shape} instead."
        )


def _assert_type_and_shape(stats_list: list[dict[str, dict]]):
    """验证所有统计量具有正确的类型和形状。

    Args:
        stats_list: 要验证的统计量字典列表

    Raises:
        ValueError: 如果任何统计量的类型或形状不正确
    """
    for stats in stats_list:
        for feature_key, feature_stats in stats.items():
            for stat_key, stat_value in feature_stats.items():
                _validate_stat_value(stat_value, stat_key, feature_key)


def aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """聚合单个特征的统计量。"""
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    # 通过匹配维度数来准备加权均值
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # 计算加权均值
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # 使用并行算法计算方差
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    aggregated = {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }

    if stats_ft_list:
        quantile_keys = [k for k in stats_ft_list[0] if k.startswith("q") and k[1:].isdigit()]

        for q_key in quantile_keys:
            if all(q_key in s for s in stats_ft_list):
                quantile_values = np.stack([s[q_key] for s in stats_ft_list])
                # 无法从分位数摘要中恢复精确的全局分位数。
                # 保留可用估计的保守包络：下分位数取 min，
                # 上分位数取 max。所得值是各输入的边界，
                # 而不是全局分位数估计。
                q_percent = int(q_key[1:])
                if q_percent <= 50:
                    aggregated[q_key] = np.min(quantile_values, axis=0)
                else:
                    aggregated[q_key] = np.max(quantile_values, axis=0)

    return aggregated


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """将多个 compute_stats 输出的统计量聚合为单组统计量。

    最终统计量将包含所有统计量字典中数据键的并集。

    例如：
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = （所有数据的均值，按 count 加权）
    - new_std = （所有数据的标准差）
    """

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats


def _get_valid_chunk_starts(episode_indices: np.ndarray, chunk_size: int) -> np.ndarray:
    """返回所有起始索引，使得长度为 ``chunk_size`` 的块保持在单个 episode 内。"""
    total = len(episode_indices)
    if total < chunk_size:
        return np.array([], dtype=np.int64)
    max_start = total - chunk_size
    starts = np.arange(max_start + 1)
    valid = episode_indices[starts] == episode_indices[starts + chunk_size - 1]
    return starts[valid]


def _compute_relative_chunk_batch(
    start_indices: np.ndarray,
    all_actions: np.ndarray,
    all_states: np.ndarray,
    chunk_size: int,
    relative_mask: np.ndarray,
) -> np.ndarray:
    """对一批起始索引进行向量化的相对动作计算。

    返回一个 ``(N * chunk_size, action_dim)`` 的 float32 数组。
    """
    if len(start_indices) == 0:
        return np.empty((0, all_actions.shape[1]), dtype=np.float32)
    offsets = np.arange(chunk_size)
    frame_idx = start_indices[:, None] + offsets[None, :]
    chunks = all_actions[frame_idx].copy()
    states = all_states[start_indices]
    mask_dim = len(relative_mask)
    chunks[:, :, :mask_dim] -= states[:, None, :mask_dim] * relative_mask[None, None, :]
    return chunks.reshape(-1, all_actions.shape[1])


def compute_relative_action_stats(
    hf_dataset,
    features: dict,
    chunk_size: int,
    exclude_joints: list[str] | None = None,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    """在整个数据集上计算相对动作的归一化统计量。

    遍历*所有*有效的动作块（位于单个 episode 内），将它们转换为
    相对动作（action − current_state），并计算适合归一化的
    按维度统计量。

    Args:
        hf_dataset: 底层 HuggingFace 数据集，包含 "action"、
            "observation.state" 和 "episode_index" 列。
        features: 数据集特征元数据（必须包含带有 "shape"
            以及可选 "names" 的 "action"）。
        chunk_size: 每个动作块的连续帧数。
        exclude_joints: 应保持绝对值（不转换为相对动作）的关节名称。
        num_workers: 用于计算的并行线程数。值 ≤1 表示单线程。
            Numpy 会释放 GIL，因此线程在这里可以实现真正的并行。

    Returns:
        包含 "mean"、"std"、"min"、"max"、"q01"、……、"q99" 键的统计量字典。

    Raises:
        ValueError: 如果数据集的帧数少于 ``chunk_size``。
        RuntimeError: 如果找不到有效的（单 episode）块。
    """
    if exclude_joints is None:
        exclude_joints = []

    action_dim = features[ACTION]["shape"][0]
    action_names = features.get(ACTION, {}).get("names")
    mask_step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=exclude_joints,
        action_names=action_names,
    )
    relative_mask = np.array(mask_step._build_mask(action_dim), dtype=np.float32)

    logging.info("Loading action/state data for relative action stats...")
    all_actions = np.array(hf_dataset[ACTION], dtype=np.float32)
    all_states = np.array(hf_dataset[OBS_STATE], dtype=np.float32)
    episode_indices = np.array(hf_dataset["episode_index"])

    valid_starts = _get_valid_chunk_starts(episode_indices, chunk_size)
    if len(valid_starts) == 0:
        raise RuntimeError(
            f"No valid chunks found (total_frames={len(episode_indices)}, chunk_size={chunk_size})"
        )

    effective_workers = max(num_workers, 1)
    logging.info(
        f"Computing relative action stats from {len(valid_starts)} chunks "
        f"(chunk_size={chunk_size}, workers={effective_workers})"
    )

    batch_size = 50_000
    batches = [valid_starts[i : i + batch_size] for i in range(0, len(valid_starts), batch_size)]

    running_stats = RunningQuantileStats()

    if num_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [
                pool.submit(
                    _compute_relative_chunk_batch,
                    batch,
                    all_actions,
                    all_states,
                    chunk_size,
                    relative_mask,
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                running_stats.update(future.result())
    else:
        for batch in batches:
            running_stats.update(
                _compute_relative_chunk_batch(batch, all_actions, all_states, chunk_size, relative_mask)
            )

    stats = running_stats.get_statistics()

    excluded_dims = int(len(relative_mask) - relative_mask.sum())
    total_frames = len(valid_starts) * chunk_size
    logging.info(
        f"Relative action stats ({len(valid_starts)} chunks, {total_frames} frames): "
        f"relative_dims={int(relative_mask.sum())}/{len(relative_mask)} (excluded={excluded_dims}), "
        f"mean={np.abs(stats['mean']).mean():.4f}, std={stats['std'].mean():.4f}, "
        f"q01={stats['q01'].mean():.4f}, q99={stats['q99'].mean():.4f}"
    )

    return stats
