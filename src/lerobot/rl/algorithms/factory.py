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

from __future__ import annotations

import torch

from .base import RLAlgorithm
from .configs import RLAlgorithmConfig


def make_algorithm_config(algorithm_type: str, **kwargs) -> RLAlgorithmConfig:
    """根据注册的算法类型名称实例化 `RLAlgorithmConfig`。

    Args:
        algorithm_type: 算法的注册表键（例如 ``"sac"``）。
        **kwargs: 转发给配置类构造函数的关键字参数。

    Returns:
        匹配的 ``RLAlgorithmConfig`` 子类的实例。

    Raises:
        ValueError: 如果 ``algorithm_type`` 未注册。
    """
    try:
        cls = RLAlgorithmConfig.get_choice_class(algorithm_type)
    except KeyError as err:
        raise ValueError(
            f"Algorithm type '{algorithm_type}' is not registered. "
            f"Available: {list(RLAlgorithmConfig.get_known_choices().keys())}"
        ) from err
    return cls(**kwargs)


def get_algorithm_class(name: str) -> type[RLAlgorithm]:
    """
    根据注册的名称获取 RL 算法类。

    该函数使用动态导入，以避免一次性将所有算法类加载到
    内存中，从而缩短启动时间并减少依赖。

    Args:
        name: 算法名称。支持的名称为 "sac"。

    Returns:
        与给定名称对应的算法类。

    Raises:
        ValueError: 如果算法名称无法识别。
    """
    if name == "sac":
        from .sac.sac_algorithm import SACAlgorithm

        return SACAlgorithm
    raise ValueError(
        f"Algorithm type '{name}' is not available. "
        f"Known: {list(RLAlgorithmConfig.get_known_choices().keys())}"
    )


def make_algorithm(cfg: RLAlgorithmConfig, policy: torch.nn.Module) -> RLAlgorithm:
    """
    实例化一个 RL 算法。

    该工厂函数查找与 ``cfg.type`` 匹配的 :class:`RLAlgorithm` 子类，
    并使用提供的策略实例化它。它还强制要求在构造之前
    ``cfg.policy_config`` 已被填充（这通常由
    :meth:`TrainRLServerPipelineConfig.validate` 处理）。

    Args:
        cfg: 算法配置。必须已设置 ``policy_config``。
        policy: 算法将要训练的策略模块。

    Returns:
        实例化后的 :class:`RLAlgorithm`。

    Raises:
        ValueError: 如果 ``cfg.policy_config`` 为 ``None``，或 ``cfg.type``
            未注册。
    """
    if getattr(cfg, "policy_config", None) is None:
        raise ValueError(
            f"{type(cfg).__name__}.policy_config is None. "
            "It must be populated (typically by TrainRLServerPipelineConfig.validate) "
            "before calling make_algorithm()."
        )
    cls = get_algorithm_class(cfg.type)
    return cls(policy=policy, config=cfg)
