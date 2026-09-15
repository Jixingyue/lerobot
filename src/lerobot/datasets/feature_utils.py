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
import logging
from copy import deepcopy
from pprint import pformat

import datasets
import numpy as np
from PIL import Image as PILImage

from lerobot.configs import VIDEO_ENCODER_INFO_KEYS, is_depth_map
from lerobot.utils.constants import DEFAULT_FEATURES, LANGUAGE_PERSISTENT
from lerobot.utils.utils import is_valid_numpy_dtype_string

from .language import is_language_column, language_events_column_feature, language_persistent_column_feature
from .utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    DatasetInfo,
)


def get_hf_features_from_features(features: dict) -> datasets.Features:
    """将 LeRobot 特征字典转换为 `datasets.Features` 对象。

    Args:
        features (dict): LeRobot 风格的特征字典。

    Returns:
        datasets.Features: 对应的 Hugging Face `datasets.Features` 对象。

    Raises:
        ValueError: 当某个特征的形状不受支持时。
    """
    hf_features = {}
    for key, ft in features.items():
        if is_language_column(key):
            hf_features[key] = (
                language_persistent_column_feature()
                if key == LANGUAGE_PERSISTENT
                else language_events_column_feature()
            )
        elif ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
        else:
            raise ValueError(f"Corresponding feature is not valid: {ft}")

    return datasets.Features(hf_features)


def create_empty_dataset_info(
    codebase_version: str,
    fps: int,
    features: dict,
    use_videos: bool,
    robot_type: str | None = None,
    chunks_size: int | None = None,
    data_files_size_in_mb: int | None = None,
    video_files_size_in_mb: int | None = None,
) -> DatasetInfo:
    """为新数据集的 ``meta/info.json`` 创建一个模板 ``DatasetInfo`` 对象。

    Args:
        codebase_version (str): LeRobot 代码库的版本。
        fps (int): 数据的每秒帧数。
        features (dict): 数据集的 LeRobot 特征字典。
        use_videos (bool): 数据集是否将存储视频。
        robot_type (str | None): 所使用机器人的类型（如果有）。
        chunks_size (int | None): 每个分片目录的最大文件数。默认为 ``DEFAULT_CHUNK_SIZE``。
        data_files_size_in_mb (int | None): parquet 文件的最大大小（MB）。默认为 ``DEFAULT_DATA_FILE_SIZE_IN_MB``。
        video_files_size_in_mb (int | None): 视频文件的最大大小（MB）。默认为 ``DEFAULT_VIDEO_FILE_SIZE_IN_MB``。

    Returns:
        DatasetInfo: 包含初始元数据的带类型数据集信息对象。
    """
    return DatasetInfo(
        codebase_version=codebase_version,
        fps=fps,
        features=features,
        robot_type=robot_type,
        chunks_size=chunks_size or DEFAULT_CHUNK_SIZE,
        data_files_size_in_mb=data_files_size_in_mb or DEFAULT_DATA_FILE_SIZE_IN_MB,
        video_files_size_in_mb=video_files_size_in_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        data_path=DEFAULT_DATA_PATH,
        video_path=DEFAULT_VIDEO_PATH if use_videos else None,
    )


def canonicalize_depth_marker(feature: dict) -> None:
    """将特征中遗留的深度图标记统一归入规范的 ``info['is_depth_map']``。

    在不同数据集版本中，深度图可以通过三种方式标记：规范的
    ``feature['info']['is_depth_map']``、遗留的 ``feature['info']['video.is_depth_map']``，
    或位于独立字典中的遗留 ``feature['video_info']['video.is_depth_map']``。本函数会原地
    修改 ``feature``，并删除遗留的键。
    """
    info = feature.get("info") or {}
    feature["info"] = info
    video_info = feature.get("video_info")
    depth = is_depth_map(feature)
    info.pop("video.is_depth_map", None)
    if isinstance(video_info, dict):
        video_info.pop("video.is_depth_map", None)
    if not video_info:
        feature.pop("video_info", None)
    info["is_depth_map"] = depth


def features_equal_for_merge(features_a: dict[str, dict], features_b: dict[str, dict]) -> bool:
    """判断两个 LeRobotDatasetMetadata 的 ``features`` 字典是否兼容、可以聚合。

    对于视频特征，比较时会忽略 ``info`` 下与视频编码参数相关的键，因为它们
    不会妨碍聚合。遗留的深度图标记（位于 ``info`` 或独立 ``video_info`` 字典中的
    ``video.is_depth_map``）会被规范化，从而使仅在深度图标记方式上有差异的
    数据集仍然可以合并。
    """

    def _normalized(feature: dict) -> dict:
        normalized = deepcopy(feature)
        canonicalize_depth_marker(normalized)
        normalized["info"] = {
            info_key: info_value
            for info_key, info_value in normalized["info"].items()
            if info_key not in VIDEO_ENCODER_INFO_KEYS
        }
        return normalized

    if set(features_a) != set(features_b):
        return False
    for key in features_a:
        fa_key = features_a[key]
        fb_key = features_b[key]
        if fa_key.get("dtype") != fb_key.get("dtype"):
            return False
        if _normalized(fa_key) != _normalized(fb_key):
            return False
    return True


def check_delta_timestamps(
    delta_timestamps: dict[str, list[float]], fps: int, tolerance_s: float, raise_value_error: bool = True
) -> bool:
    """检查增量时间戳是否为 1/fps 的整数倍（允许给定容差）。

    这可以确保将这些增量时间戳与数据集中任意已有时间戳相加后，
    所得值仍与数据集的帧率对齐。

    Args:
        delta_timestamps (dict): 一个字典，其值为以秒为单位的
            时间增量列表。
        fps (int): 数据集的每秒帧数。
        tolerance_s (float): 允许的容差（秒）。
        raise_value_error (bool): 为 True 时，校验失败会抛出错误。

    Returns:
        bool: 所有增量都有效时返回 True，否则返回 False。

    Raises:
        ValueError: 当存在超出容差的增量且 `raise_value_error` 为 True 时。
    """
    outside_tolerance = {}
    for key, delta_ts in delta_timestamps.items():
        within_tolerance = [abs(ts * fps - round(ts * fps)) / fps <= tolerance_s for ts in delta_ts]
        if not all(within_tolerance):
            outside_tolerance[key] = [
                ts for ts, is_within in zip(delta_ts, within_tolerance, strict=True) if not is_within
            ]

    if len(outside_tolerance) > 0:
        if raise_value_error:
            raise ValueError(
                f"""
                The following delta_timestamps are found outside of tolerance range.
                Please make sure they are multiples of 1/{fps} +/- tolerance and adjust
                their values accordingly.
                \n{pformat(outside_tolerance)}
                """
            )
        return False

    return True


def get_delta_indices(delta_timestamps: dict[str, list[float]], fps: int) -> dict[str, list[int]]:
    """将以秒为单位的增量时间戳转换为以帧为单位的增量索引。

    Args:
        delta_timestamps (dict): 以秒为单位的时间增量字典。
        fps (int): 数据集的每秒帧数。

    Returns:
        dict: 帧增量索引字典。
    """
    delta_indices = {}
    for key, delta_ts in delta_timestamps.items():
        delta_indices[key] = [round(d * fps) for d in delta_ts]

    return delta_indices


def validate_frame(frame: dict, features: dict) -> None:
    # DEFAULT_FEATURES（timestamp、frame_index、episode_index、index、task_index）由
    # 录制流水线（add_frame / save_episode）自动填充，调用方不得
    # 提供。在这里排除它们意味着任何包含这些键的帧字典都会
    # 因多余特征而被拒绝。
    expected_features = set(features) - set(DEFAULT_FEATURES)
    actual_features = set(frame)

    # task 是特殊的必填字段，不属于常规特征
    if "task" not in actual_features:
        raise ValueError("Feature mismatch in `frame` dictionary:\nMissing features: {'task'}\n")

    # 从 actual_features 中移除 task，以便进行常规特征校验
    actual_features_for_validation = actual_features - {"task"}

    error_message = validate_features_presence(actual_features_for_validation, expected_features)

    common_features = actual_features_for_validation & expected_features
    for name in common_features:
        error_message += validate_feature_dtype_and_shape(name, features[name], frame[name])

    if error_message:
        raise ValueError(error_message)


def validate_features_presence(actual_features: set[str], expected_features: set[str]) -> str:
    """检查帧中是否存在缺失或多余的特征。

    Args:
        actual_features (set[str]): 帧中实际存在的特征名集合。
        expected_features (set[str]): 帧中预期存在的特征名集合。

    Returns:
        str: 不匹配时返回错误消息字符串，否则返回空字符串。
    """
    error_message = ""
    missing_features = expected_features - actual_features
    extra_features = actual_features - expected_features

    if missing_features or extra_features:
        error_message += "Feature mismatch in `frame` dictionary:\n"
        if missing_features:
            error_message += f"Missing features: {missing_features}\n"
        if extra_features:
            error_message += f"Extra features: {extra_features}\n"

    return error_message


def validate_feature_dtype_and_shape(
    name: str, feature: dict, value: np.ndarray | PILImage.Image | str
) -> str:
    """校验单个特征值的 dtype 和形状。

    Args:
        name (str): 特征名。
        feature (dict): 来自 LeRobot 特征字典的特征规范。
        value: 待校验的特征值。

    Returns:
        str: 校验失败时返回错误消息，否则返回空字符串。

    Raises:
        NotImplementedError: 当该特征 dtype 尚不支持校验时。
    """
    expected_dtype = feature["dtype"]
    expected_shape = feature["shape"]
    if is_valid_numpy_dtype_string(expected_dtype):
        return validate_feature_numpy_array(name, expected_dtype, expected_shape, value)
    elif expected_dtype in ["image", "video"]:
        return validate_feature_image_or_video(name, expected_shape, value)
    elif expected_dtype == "string":
        return validate_feature_string(name, value)
    elif expected_dtype == "language":
        return validate_feature_language(name, value)
    else:
        raise NotImplementedError(f"The feature dtype '{expected_dtype}' is not implemented yet.")


def validate_feature_numpy_array(
    name: str, expected_dtype: str, expected_shape: list[int], value: np.ndarray
) -> str:
    """校验一个预期为 numpy 数组的特征。

    Args:
        name (str): 特征名。
        expected_dtype (str): 预期的 numpy dtype（字符串形式）。
        expected_shape (list[int]): 预期的形状。
        value (np.ndarray): 待校验的 numpy 数组。

    Returns:
        str: 校验失败时返回错误消息，否则返回空字符串。
    """
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_dtype = value.dtype
        actual_shape = value.shape

        if actual_dtype != np.dtype(expected_dtype):
            error_message += f"The feature '{name}' of dtype '{actual_dtype}' is not of the expected dtype '{expected_dtype}'.\n"

        if actual_shape != expected_shape:
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{expected_shape}'.\n"
    else:
        error_message += f"The feature '{name}' is not a 'np.ndarray'. Expected type is '{expected_dtype}', but type '{type(value)}' provided instead.\n"

    return error_message


def validate_feature_image_or_video(
    name: str, expected_shape: list[str], value: np.ndarray | PILImage.Image
) -> str:
    """校验一个预期为图像或视频帧的特征。

    接受 `np.ndarray`（通道优先或通道最后）或 `PIL.Image.Image`。

    Args:
        name (str): 特征名。
        expected_shape (list[str]): 预期的形状，例如 (C, H, W) 或 (H, W, C)。
        value: 待校验的图像数据。

    Returns:
        str: 校验失败时返回错误消息，否则返回空字符串。
    """
    # 注意：像素范围的检查（浮点数为 [0,1]，uint8 为 [0,255]）由图像写入器线程完成。
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_shape = value.shape
        c, h, w = expected_shape
        if len(actual_shape) != 3 or (actual_shape != (c, h, w) and actual_shape != (h, w, c)):
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{(c, h, w)}' or '{(h, w, c)}'.\n"
    elif isinstance(value, PILImage.Image):
        pass
    else:
        error_message += f"The feature '{name}' is expected to be of type 'PIL.Image' or 'np.ndarray' channel first or channel last, but type '{type(value)}' provided instead.\n"

    return error_message


def validate_feature_string(name: str, value: str) -> str:
    """校验一个预期为字符串的特征。

    Args:
        name (str): 特征名。
        value (str): 待校验的值。

    Returns:
        str: 校验失败时返回错误消息，否则返回空字符串。
    """
    if not isinstance(value, str):
        return f"The feature '{name}' is expected to be of type 'str', but type '{type(value)}' provided instead.\n"
    return ""


def validate_feature_language(name: str, value) -> str:
    """校验一个预期存放语言标注的特征。

    语言列（``language_persistent`` / ``language_events``）是在录制之后
    由标注流水线填充的，而非在录制时填充。此处提供的任何值都会在
    帧写入之前被丢弃，因此非空值几乎肯定意味着出错了。这里选择发出
    警告而非直接失败，以保持录制过程的健壮性。

    Args:
        name (str): 特征名。
        value: 待校验的值。

    Returns:
        str: 始终为空字符串——语言值问题不属于致命错误。
    """
    if value is not None:
        logging.warning(
            f"The feature '{name}' is a 'language' column populated by the annotation pipeline, "
            f"not at record time. The provided value will be dropped."
        )
    return ""


def validate_episode_buffer(episode_buffer: dict, total_episodes: int, features: dict) -> None:
    """在 episode 缓冲区写入磁盘之前对其进行校验。

    确保缓冲区包含必需的键、至少有一帧，并且其特征
    与数据集规范一致。

    Args:
        episode_buffer (dict): 包含单个 episode 数据的缓冲区。
        total_episodes (int): 数据集中当前的 episode 总数。
        features (dict): 数据集的 LeRobot 特征字典。

    Raises:
        ValueError: 当缓冲区无效时。
        NotImplementedError: 当手动设置了不匹配的 episode 索引时。
    """
    if "size" not in episode_buffer:
        raise ValueError("size key not found in episode_buffer")

    if "task" not in episode_buffer:
        raise ValueError("task key not found in episode_buffer")

    if episode_buffer["episode_index"] != total_episodes:
        # TODO(aliberts): 增加使用已有 episode_index 的选项
        raise NotImplementedError(
            "You might have manually provided the episode_buffer with an episode_index that doesn't "
            "match the total number of episodes already in the dataset. This is not supported for now."
        )

    if episode_buffer["size"] == 0:
        raise ValueError("You must add one or several frames with `add_frame` before calling `add_episode`.")

    buffer_keys = set(episode_buffer.keys()) - {"task", "size"}
    if not buffer_keys == set(features):
        raise ValueError(
            f"Features from `episode_buffer` don't match the ones in `features`."
            f"In episode_buffer not in features: {buffer_keys - set(features)}"
            f"In features not in episode_buffer: {set(features) - buffer_keys}"
        )
