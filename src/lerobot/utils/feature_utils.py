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
"""轻量级的特征操作工具。

这些函数有意保持不依赖重型依赖（例如
HuggingFace ``datasets`` 库），以便可以从代码库的任何位置
导入它们——包括属于*最小*安装一部分的模块——
而不会触发 ``lerobot.datasets`` 包的守卫。
"""

from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PolicyFeature

from .constants import ACTION, DEFAULT_FEATURES, OBS_ENV_STATE, OBS_STR


def _validate_feature_names(features: dict[str, dict]) -> None:
    """校验特征名称中不包含非法字符。

    参数:
        features (dict): LeRobot 特征字典。

    异常:
        ValueError: 如果任何特征名称包含 '/'。
    """
    invalid_features = {name: ft for name, ft in features.items() if "/" in name}
    if invalid_features:
        raise ValueError(f"Feature names should not contain '/'. Found '/' in '{invalid_features}'.")


def hw_to_dataset_features(
    hw_features: dict[str, type | tuple], prefix: str, use_video: bool = True
) -> dict[str, dict]:
    """将硬件专属特征转换为 LeRobot 数据集特征字典。

    此函数接收一个描述硬件输出的字典（例如关节状态
    或相机图像形状），并将其格式化为标准的 LeRobot 特征
    规范。单通道相机（形状为 ``(H, W, 1)``）会通过
    ``info["is_depth_map"] = True`` 标记为深度图；三通道相机 ``(H, W, 3)``
    被视为 RGB。

    参数:
        hw_features (dict): 将特征名称映射到其类型（关节为
            float）或形状（图像为 tuple）的字典。
        prefix (str): 要添加到特征键上的前缀（例如 "observation"
            或 "action"）。
        use_video (bool)：为 True 时图像特征标记为 "video"，否则标记为 "image"。

    返回:
        dict：LeRobot 特征字典。深度相机带有 ``info["is_depth_map"] = True``。
    """
    features = {}
    joint_fts = {
        key: ftype
        for key, ftype in hw_features.items()
        if ftype is float or (isinstance(ftype, PolicyFeature) and ftype.type != FeatureType.VISUAL)
    }
    # TODO(CarolinePascal)：我们不应依赖形状来判断一个特征是否为相机！
    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}

    if joint_fts and prefix == ACTION:
        features[prefix] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    if joint_fts and prefix == OBS_STR:
        features[f"{prefix}.state"] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    for key, shape in cam_fts.items():
        dtype = "video" if use_video else "image"
        if len(shape) == 3 and shape[2] in (1, 3):
            features[f"{prefix}.images.{key}"] = {
                "dtype": dtype,
                "shape": shape,
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": shape[2] == 1},
            }
        else:
            raise ValueError(
                f"Camera feature '{key}' has shape {shape}. "
                f"Expected a 3-tuple (H, W, C), e.g. (480, 640, 3) for RGB or (480, 640, 1) for depth."
            )

    _validate_feature_names(features)
    return features


def build_dataset_frame(
    ds_features: dict[str, dict], values: dict[str, Any], prefix: str
) -> dict[str, np.ndarray]:
    """根据数据集特征，用原始值构造单个数据帧。

    “帧”是一个包含单个时间步全部数据的字典，
    其中的数据按特征规范格式化为 numpy 数组。

    参数:
        ds_features (dict): LeRobot 数据集特征字典。
        values (dict): 来自硬件/环境的原始值字典。
        prefix (str): 用于筛选特征的前缀（例如 "observation"
            或 "action"）。

    返回:
        dict：表示单帧数据的字典。
    """
    frame = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]

    return frame


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    """将数据集特征转换为策略特征。

    此函数把数据集的特征规范转换为
    策略可使用的格式，按类型对特征进行分类（例如视觉、状态、
    动作），并确保形状正确（例如图像采用通道优先）。

    参数:
        features (dict): LeRobot 数据集特征字典。

    返回:
        dict：将特征键映射到 `PolicyFeature` 对象的字典。

    异常:
        ValueError：如果某个图像特征不具有三维形状。
    """
    # TODO(aliberts)：在数据集特征中实现 "type" 并简化此处
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video"]:
            type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
            else:
                names = ft["names"]
                # 对 "channel" 的向后兼容：这是 LeRobotDataset v2.0 中针对移植数据集引入的一个错误。
                if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
                    shape = (shape[2], shape[0], shape[1])
        elif key == OBS_ENV_STATE:
            type = FeatureType.ENV
        elif key.startswith(OBS_STR):
            type = FeatureType.STATE
        elif key.startswith(ACTION):
            type = FeatureType.ACTION
        else:
            continue

        policy_features[key] = PolicyFeature(
            type=type,
            shape=shape,
        )

    return policy_features


def combine_feature_dicts(*dicts: dict) -> dict:
    """合并多个 LeRobot 分组特征字典。

    - 对于带有 "names" 的一维数值规范（dtype 不是 image/video/string）：合并 names 并重新计算形状。
    - 对于其他特征（例如 `observation.images.*`）：后一个覆盖前一个（前提是它们完全相同）。

    参数:
        *dicts: 任意数量的、待合并的 LeRobot 特征字典。

    返回:
        dict：单个合并后的特征字典。

    异常:
        ValueError：如果被合并的某个特征存在 dtype 不匹配。
    """
    out: dict = {}
    for d in dicts:
        for key, value in d.items():
            if not isinstance(value, dict):
                out[key] = value
                continue

            dtype = value.get("dtype")
            shape = value.get("shape")
            is_vector = (
                dtype not in ("image", "video", "string")
                and isinstance(shape, tuple)
                and len(shape) == 1
                and "names" in value
            )

            if is_vector:
                # 初始化或获取该特征键的累积字典
                target = out.setdefault(key, {"dtype": dtype, "names": [], "shape": (0,)})
                # 确保各合并条目的数据类型一致
                if "dtype" in target and dtype != target["dtype"]:
                    raise ValueError(f"dtype mismatch for '{key}': {target['dtype']} vs {dtype}")

                # 合并特征名称：仅追加新名称，以保持顺序且不产生重复
                seen = set(target["names"])
                for n in value["names"]:
                    if n not in seen:
                        target["names"].append(n)
                        seen.add(n)
                # 重新计算形状，以反映更新后的特征数量
                target["shape"] = (len(target["names"]),)
            else:
                # 对于图像/视频以及非一维条目：用最新的定义覆盖
                out[key] = value
    return out
