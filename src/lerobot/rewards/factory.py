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

import importlib
import logging
from typing import Any

import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .classifier.configuration_classifier import RewardClassifierConfig
from .pretrained import PreTrainedRewardModel
from .robometer.configuration_robometer import RobometerConfig
from .sarm.configuration_sarm import SARMConfig
from .topreward.configuration_topreward import TOPRewardConfig


def get_reward_model_class(name: str) -> type[PreTrainedRewardModel]:
    """
    根据注册的名称获取奖励模型类。

    此函数使用动态导入，避免一次性将所有奖励模型类加载到内存中，
    从而加快启动速度并减少依赖。

    Args:
        name: 奖励模型的名称。支持的名称有 "reward_classifier"、
              "sarm"、"robometer"、"topreward"。

    Returns:
        与给定名称对应的奖励模型类。

    Raises:
        ValueError: 如果奖励模型名称无法识别。
    """
    if name == "reward_classifier":
        from lerobot.rewards.classifier.modeling_classifier import Classifier

        return Classifier
    elif name == "sarm":
        from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel

        return SARMRewardModel
    elif name == "robometer":
        from lerobot.rewards.robometer.modeling_robometer import RobometerRewardModel

        return RobometerRewardModel
    elif name == "topreward":
        from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

        return TOPRewardModel
    else:
        try:
            return _get_reward_model_cls_from_name(name=name)
        except Exception as e:
            raise ValueError(f"Reward model type '{name}' is not available.") from e


def make_reward_model_config(reward_type: str, **kwargs) -> RewardModelConfig:
    """
    根据奖励类型实例化奖励模型配置对象。

    此工厂函数通过将字符串标识符映射到对应的配置类，
    简化了奖励模型配置对象的创建。

    Args:
        reward_type: 奖励模型的类型。支持的类型包括
                     "reward_classifier"、"sarm"、"robometer"、"topreward"。
        **kwargs: 传递给配置类构造函数的关键字参数。

    Returns:
        `RewardModelConfig` 子类的实例。

    Raises:
        ValueError: 如果 `reward_type` 无法识别。
    """
    if reward_type == "reward_classifier":
        return RewardClassifierConfig(**kwargs)
    elif reward_type == "sarm":
        return SARMConfig(**kwargs)
    elif reward_type == "robometer":
        return RobometerConfig(**kwargs)
    elif reward_type == "topreward":
        return TOPRewardConfig(**kwargs)
    else:
        try:
            config_cls = RewardModelConfig.get_choice_class(reward_type)
            return config_cls(**kwargs)
        except Exception as e:
            raise ValueError(f"Reward model type '{reward_type}' is not available.") from e


def make_reward_model(cfg: RewardModelConfig, **kwargs) -> PreTrainedRewardModel:
    """
    根据配置实例化奖励模型。

    Args:
        cfg: 待创建奖励模型的配置。如果设置了 `cfg.pretrained_path`，
             模型将从该路径加载权重。
        **kwargs: 转发给模型构造函数的其他关键字参数
            （例如 ``dataset_stats``、``dataset_meta``）。

    Returns:
        已实例化并放置到设备上的奖励模型。
    """
    reward_cls = get_reward_model_class(cfg.type)

    kwargs["config"] = cfg

    if cfg.pretrained_path:
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        kwargs["revision"] = cfg.pretrained_revision
        reward_model = reward_cls.from_pretrained(**kwargs)
    else:
        reward_model = reward_cls(**kwargs)

    reward_model.to(cfg.device)
    assert isinstance(reward_model, torch.nn.Module)

    return reward_model


def make_reward_pre_post_processors(
    reward_cfg: RewardModelConfig,
    **kwargs,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为给定的奖励模型创建前处理器和后处理器流水线。

    每种奖励模型类型都有其专用的处理器工厂函数。

    Args:
        reward_cfg: 要为其创建处理器的奖励模型配置。
        **kwargs: 传递给处理器工厂的其他关键字参数
            （例如 ``dataset_stats``、``dataset_meta``）。

    Returns:
        包含输入（前处理器）和输出（后处理器）流水线的元组。

    Raises:
        ValueError: 如果给定的奖励模型配置类型没有实现对应的处理器工厂。
    """
    # 根据奖励模型类型创建新的处理器
    if isinstance(reward_cfg, RewardClassifierConfig):
        from lerobot.rewards.classifier.processor_classifier import make_classifier_processor

        return make_classifier_processor(
            config=reward_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(reward_cfg, SARMConfig):
        from lerobot.rewards.sarm.processor_sarm import make_sarm_pre_post_processors

        return make_sarm_pre_post_processors(
            config=reward_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
            dataset_meta=kwargs.get("dataset_meta"),
        )
    elif isinstance(reward_cfg, RobometerConfig):
        from lerobot.rewards.robometer.processor_robometer import make_robometer_pre_post_processors

        return make_robometer_pre_post_processors(
            config=reward_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(reward_cfg, TOPRewardConfig):
        from lerobot.rewards.topreward.processor_topreward import make_topreward_pre_post_processors

        return make_topreward_pre_post_processors(
            config=reward_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    else:
        try:
            processors = _make_processors_from_reward_model_config(
                config=reward_cfg,
                dataset_stats=kwargs.get("dataset_stats"),
            )
        except Exception as e:
            raise ValueError(
                f"Processor for reward model type '{reward_cfg.type}' is not implemented."
            ) from e
        return processors


def _get_reward_model_cls_from_name(name: str) -> type[PreTrainedRewardModel]:
    """使用动态导入，根据注册的名称获取奖励模型类。

    此函数作为辅助函数，用于从第三方 lerobot 插件导入奖励模型。

    Args:
        name: 奖励模型的名称。

    Returns:
        与给定名称对应的奖励模型类。
    """
    if name not in RewardModelConfig.get_known_choices():
        raise ValueError(
            f"Unknown reward model name '{name}'. "
            f"Available reward models: {RewardModelConfig.get_known_choices()}"
        )

    config_cls = RewardModelConfig.get_choice_class(name)
    config_cls_name = config_cls.__name__

    model_name = config_cls_name.removesuffix("Config")
    if model_name == config_cls_name:
        raise ValueError(
            f"The config class name '{config_cls_name}' does not follow the expected naming convention. "
            f"Make sure it ends with 'Config'!"
        )

    cls_name = model_name + "RewardModel"
    module_path = config_cls.__module__.replace("configuration_", "modeling_")

    module = importlib.import_module(module_path)
    reward_cls = getattr(module, cls_name)
    return reward_cls


def _make_processors_from_reward_model_config(
    config: RewardModelConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """使用动态导入，根据奖励模型配置创建前处理器和后处理器。

    此函数作为辅助函数，用于从第三方 lerobot 奖励模型插件导入处理器工厂。

    Args:
        config: 奖励模型配置对象。
        dataset_stats: 用于归一化的数据集统计信息。

    Returns:
        包含输入（前处理器）和输出（后处理器）流水线的元组。
    """
    reward_type = config.type
    function_name = f"make_{reward_type}_pre_post_processors"
    module_path = config.__class__.__module__.replace("configuration_", "processor_")
    logging.debug(
        f"Instantiating reward pre/post processors using function '{function_name}' "
        f"from module '{module_path}'"
    )
    module = importlib.import_module(module_path)
    function = getattr(module, function_name)
    return function(config, dataset_stats=dataset_stats)
