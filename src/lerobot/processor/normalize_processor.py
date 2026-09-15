#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, PolicyAction, TransitionKey

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset

from lerobot.utils.constants import ACTION

from .converters import from_tensor_to_numpy, to_tensor
from .pipeline import PolicyProcessorPipeline, ProcessorStep, ProcessorStepRegistry, RobotObservation


@dataclass
class _NormalizationMixin:
    """
    提供归一化和反归一化核心功能的混入（mixin）类。

    该类管理归一化统计量（`stats`），将其转换为张量以进行高效计算，
    处理设备放置，并实现应用归一化变换（mean/std 和 min/max）的逻辑。
    它设计为由具体的 `ProcessorStep` 实现继承，不应直接使用。

    **统计量覆盖保留：**
    当在构造期间显式提供了统计量时（例如通过
    `DataProcessorPipeline.from_pretrained()` 中的 overrides），即使调用了
    `load_state_dict()`，这些统计量也会被保留。这允许用户覆盖已保存模型的
    归一化统计量，同时保持模型的其余状态不变。

    Examples:
        ```python
        # 常见用例：使用数据集统计量进行覆盖
        from lerobot.datasets import LeRobotDataset

        dataset = LeRobotDataset("my_dataset")
        pipeline = DataProcessorPipeline.from_pretrained(
            "model_path", overrides={"normalizer_processor": {"stats": dataset.meta.stats}}
        )
        # 将使用 dataset.meta.stats，而不是已保存模型中的统计量

        # 自定义统计量覆盖
        custom_stats = {"action": {"mean": [0.0], "std": [1.0]}}
        pipeline = DataProcessorPipeline.from_pretrained(
            "model_path", overrides={"normalizer_processor": {"stats": custom_stats}}
        )
        ```

    Attributes:
        features: 将特征名称映射到 `PolicyFeature` 对象的字典，定义
            要处理的数据结构。
        norm_map: 将 `FeatureType` 映射到 `NormalizationMode` 的字典，指定
            每种特征类型使用哪种归一化方法。
        stats: 包含每个特征的归一化统计量（例如 mean、std、
            min、max）的字典。
        device: 存储和执行张量运算的 PyTorch 设备。
        eps: 用于防止归一化计算中出现除零的小 epsilon 值。
        normalize_observation_keys: 用于选择性地对特定观测特征应用
            归一化的可选键集合。
        _tensor_stats: 以 PyTorch 张量形式保存归一化统计量的内部字典。
        _stats_explicitly_provided: 记录构造期间是否显式提供了统计量的
            内部标志（用于覆盖保留）。
    """

    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]
    stats: dict[str, dict[str, Any]] | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    eps: float = 1e-8
    normalize_observation_keys: set[str] | None = None

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)
    _stats_explicitly_provided: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """
        在 dataclass 构造之后初始化混入类。

        该方法处理 `features` 和 `norm_map` 从 JSON 兼容格式的健壮反序列化
        （其中枚举变为字符串、元组变为列表），并将提供的 `stats` 字典
        转换为指定设备上的张量字典（`_tensor_stats`）。
        """
        # 记录统计量是否被显式提供（不为 None 且不为空）
        self._stats_explicitly_provided = self.stats is not None and bool(self.stats)
        # 健壮的 JSON 反序列化处理（防护空映射）。
        if self.features:
            first_val = next(iter(self.features.values()))
            if isinstance(first_val, dict):
                reconstructed = {}
                for key, ft_dict in self.features.items():
                    reconstructed[key] = PolicyFeature(
                        type=FeatureType(ft_dict["type"]), shape=tuple(ft_dict["shape"])
                    )
                self.features = reconstructed

        # 如果键是字符串（JSON），则重建枚举映射
        if self.norm_map and all(isinstance(k, str) for k in self.norm_map):
            reconstructed = {}
            for ft_type_str, norm_mode_str in self.norm_map.items():
                reconstructed[FeatureType(ft_type_str)] = NormalizationMode(norm_mode_str)
            self.norm_map = reconstructed

        # 将统计量转换为张量，并在初始化期间一次性移动到目标设备。
        self.stats = self.stats or {}
        if self.dtype is None:
            self.dtype = torch.float32
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        self._reshape_visual_stats()

    def _reshape_visual_stats(self) -> None:
        """将扁平的 ``(C,)`` 视觉统计量重塑为 ``(C, 1, 1)``，以便图像广播。

        对于来自 :func:`~lerobot.datasets.compute_stats.compute_stats` 的统计量
        此操作为空操作（已经是 ``(C, 1, 1)``）。RL 训练需要此操作，因为它可以
        在没有数据集的情况下开始，并通过 JSON 配置手动提供统计量。
        """
        for key, feature in self.features.items():
            if feature.type != FeatureType.VISUAL:
                continue
            if key not in self._tensor_stats:
                continue
            for stat_name, stat_tensor in self._tensor_stats[key].items():
                if not isinstance(stat_tensor, Tensor) or stat_tensor.ndim != 1:
                    continue
                self._tensor_stats[key][stat_name] = stat_tensor.reshape(-1, 1, 1)

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> _NormalizationMixin:
        """
        将处理器的归一化统计量移动到指定设备。

        Args:
            device: 目标 PyTorch 设备。

        Returns:
            类的实例，支持方法链式调用。
        """
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        self._reshape_visual_stats()
        return self

    def state_dict(self) -> dict[str, Tensor]:
        """
        以扁平状态字典的形式返回归一化统计量。

        所有张量在返回之前都会移动到 CPU，这是保存状态字典的标准做法。

        Returns:
            从 `'feature_name.stat_name'` 映射到 CPU 上对应统计量张量的
            扁平字典。
        """
        flat: dict[str, Tensor] = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor.cpu()  # 始终保存到 CPU
        return flat

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """
        从状态字典加载归一化统计量。

        加载的张量会被移动到处理器配置的设备上。

        **统计量覆盖保留：**
        如果在构造期间显式提供了统计量（例如通过
        `DataProcessorPipeline.from_pretrained()` 中的 overrides），它们会被保留，
        状态字典会被忽略。这允许用户在仍然加载模型其余状态的同时
        覆盖归一化统计量。

        此行为对于用户想要将预训练模型适配到具有不同统计量的新数据集、
        而无需重新训练整个模型的场景至关重要。

        Args:
            state: 键格式为 `'feature_name.stat_name'` 的扁平状态字典。

        Note:
            当由于显式提供而保留统计量时，只会更新张量表示，
            以确保与当前设备和 dtype 设置的一致性。
        """
        # 如果统计量在构造期间被显式提供，则保留它们
        if self._stats_explicitly_provided and self.stats is not None:
            # 不从 state_dict 加载，保留显式提供的统计量
            # 但确保 _tensor_stats 被正确初始化
            self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)  # type: ignore[assignment]
            self._reshape_visual_stats()
            return

        # 正常行为：从 state_dict 加载统计量
        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            # 加载到处理器配置的设备上。
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor.to(
                dtype=torch.float32, device=self.device
            )
        self._reshape_visual_stats()

        # 从张量统计量重建原始统计量字典，以兼容 to() 方法
        # 以及其他依赖 self.stats 的函数
        self.stats = {}
        for key, tensor_dict in self._tensor_stats.items():
            self.stats[key] = {}
            for stat_name, tensor in tensor_dict.items():
                # 将张量转换回 python/numpy 格式
                self.stats[key][stat_name] = from_tensor_to_numpy(tensor)

    def get_config(self) -> dict[str, Any]:
        """
        返回处理器配置的可序列化字典。

        该方法在将处理器保存到磁盘时使用，确保其配置
        之后可以被重建。

        Returns:
            包含配置的 JSON 可序列化字典。
        """
        config = {
            "eps": self.eps,
            "features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.features.items()
            },
            "norm_map": {ft_type.value: norm_mode.value for ft_type, norm_mode in self.norm_map.items()},
        }
        if self.normalize_observation_keys is not None:
            config["normalize_observation_keys"] = sorted(self.normalize_observation_keys)
        return config

    def _normalize_observation(self, observation: RobotObservation, inverse: bool) -> dict[str, Tensor]:
        """
        对观测字典中所有相关特征应用（反）归一化。

        Args:
            observation: 要处理的观测字典。
            inverse: 如果为 `True`，则应用反归一化；否则应用归一化。

        Returns:
            具有转换后张量值的新观测字典。
        """
        new_observation = dict(observation)
        for key, feature in self.features.items():
            if self.normalize_observation_keys is not None and key not in self.normalize_observation_keys:
                continue
            if feature.type != FeatureType.ACTION and key in new_observation:
                # 转换为张量，但保留原始 dtype 以供适配逻辑使用
                tensor = torch.as_tensor(new_observation[key])
                new_observation[key] = self._apply_transform(tensor, key, feature.type, inverse=inverse)
        return new_observation

    def _normalize_action(self, action: Tensor, inverse: bool) -> Tensor:
        # 转换为张量，但保留原始 dtype 以供适配逻辑使用
        """
        对动作张量应用（反）归一化。

        Args:
            action: 要处理的动作张量。
            inverse: 如果为 `True`，则应用反归一化；否则应用归一化。

        Returns:
            转换后的动作张量。
        """
        processed_action = self._apply_transform(action, ACTION, FeatureType.ACTION, inverse=inverse)
        return processed_action

    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> Tensor:
        """
        对张量应用归一化或反归一化变换的核心逻辑。

        该方法根据特征类型选择合适的归一化模式，
        并应用相应的数学运算。

        归一化模式：
          - MEAN_STD: 将数据中心化为零均值、单位方差。
          - MIN_MAX: 使用实际的 min/max 值将数据缩放到 [-1, 1] 范围。
          - QUANTILES: 使用第 1 和第 99 百分位数（q01/q99）将数据缩放到 [-1, 1] 范围。
          - QUANTILE10: 使用第 10 和第 90 百分位数（q10/q90）将数据缩放到 [-1, 1] 范围。

        Args:
            tensor: 要转换的输入张量。
            key: 与该张量对应的特征键。
            feature_type: 张量的 `FeatureType`。
            inverse: 如果为 `True`，则应用逆变换（反归一化）。

        Returns:
            转换后的张量。

        Raises:
            ValueError: 如果遇到不支持的归一化模式。
        """
        norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor

        if norm_mode not in (
            NormalizationMode.MEAN_STD,
            NormalizationMode.MIN_MAX,
            NormalizationMode.QUANTILES,
            NormalizationMode.QUANTILE10,
        ):
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        # 为了兼容 Accelerate：确保统计量与输入张量位于相同的设备并具有相同的 dtype
        if self._tensor_stats and key in self._tensor_stats:
            first_stat = next(iter(self._tensor_stats[key].values()))
            if first_stat.device != tensor.device or first_stat.dtype != tensor.dtype:
                self.to(device=tensor.device, dtype=tensor.dtype)

        stats = self._tensor_stats[key]

        if norm_mode == NormalizationMode.MEAN_STD:
            mean = stats.get("mean", None)
            std = stats.get("std", None)
            if mean is None or std is None:
                raise ValueError(
                    "MEAN_STD normalization mode requires mean and std stats, please update the dataset with the correct stats"
                )

            mean, std = stats["mean"], stats["std"]
            # 通过加上一个小 epsilon 来避免除零。
            denom = std + self.eps
            if inverse:
                return tensor * std + mean
            return (tensor - mean) / denom

        if norm_mode == NormalizationMode.MIN_MAX:
            min_val = stats.get("min", None)
            max_val = stats.get("max", None)
            if min_val is None or max_val is None:
                raise ValueError(
                    "MIN_MAX normalization mode requires min and max stats, please update the dataset with the correct stats"
                )

            min_val, max_val = stats["min"], stats["max"]
            denom = max_val - min_val
            # 当 min_val == max_val 时，用小 epsilon 替换分母
            # 以防止除零。这会将等于 min_val 的输入一致地映射到 -1，
            # 确保变换的稳定性。
            denom = torch.where(
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            )
            if inverse:
                # 从 [-1, 1] 映射回 [min, max]
                return (tensor + 1) / 2 * denom + min_val
            # 从 [min, max] 映射到 [-1, 1]
            return 2 * (tensor - min_val) / denom - 1

        if norm_mode == NormalizationMode.QUANTILES:
            q01 = stats.get("q01", None)
            q99 = stats.get("q99", None)
            if q01 is None or q99 is None:
                raise ValueError(
                    "QUANTILES normalization mode requires q01 and q99 stats, please update the dataset with the correct stats using the `augment_dataset_quantile_stats.py` script"
                )

            denom = q99 - q01
            # 当分位数相同时，通过加上 epsilon 来避免除零
            denom = torch.where(
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            )
            if inverse:
                return (tensor + 1.0) * denom / 2.0 + q01
            return 2.0 * (tensor - q01) / denom - 1.0

        if norm_mode == NormalizationMode.QUANTILE10:
            q10 = stats.get("q10", None)
            q90 = stats.get("q90", None)
            if q10 is None or q90 is None:
                raise ValueError(
                    "QUANTILE10 normalization mode requires q10 and q90 stats, please update the dataset with the correct stats using the `augment_dataset_quantile_stats.py` script"
                )

            denom = q90 - q10
            # 当分位数相同时，通过加上 epsilon 来避免除零
            denom = torch.where(
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            )
            if inverse:
                return (tensor + 1.0) * denom / 2.0 + q10
            return 2.0 * (tensor - q10) / denom - 1.0

        # 如果缺少必要的统计量，则原样返回输入。
        return tensor


@dataclass
@ProcessorStepRegistry.register(name="normalizer_processor")
class NormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    """
    对转移中的观测和动作应用归一化的处理步骤。

    该类使用 `_NormalizationMixin` 中的逻辑执行正向归一化
    （例如将数据缩放为零均值、单位方差，或缩放到 [-1, 1] 范围）。
    它通常在将数据送入策略之前的预处理流水线中使用。
    """

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
        *,
        normalize_observation_keys: set[str] | None = None,
        eps: float = 1e-8,
        device: torch.device | str | None = None,
    ) -> NormalizerProcessorStep:
        """
        使用 `LeRobotDataset` 中的统计量创建 `NormalizerProcessorStep` 实例。

        Args:
            dataset: 从中提取归一化统计量的数据集。
            features: 处理器的特征定义。
            norm_map: 从特征类型到归一化模式的映射。
            normalize_observation_keys: 要归一化的观测键的可选集合。
            eps: 用于数值稳定性的小 epsilon 值。
            device: 处理器的目标设备。

        Returns:
            `NormalizerProcessorStep` 的新实例。
        """
        return cls(
            features=features,
            norm_map=norm_map,
            stats=dataset.meta.stats,
            normalize_observation_keys=normalize_observation_keys,
            eps=eps,
            device=device,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        # 处理观测归一化。
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(
                observation, inverse=False
            )

        # 处理动作归一化。
        action = new_transition.get(TransitionKey.ACTION)

        if action is None:
            return new_transition

        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=False)

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="unnormalizer_processor")
class UnnormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    """
    对观测和动作应用反归一化的处理步骤。

    该类逆转归一化过程，将数据缩放回其原始范围。
    它通常在后处理流水线中使用，将策略的归一化动作输出
    转换为机器人或环境可以执行的格式。
    """

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
        *,
        device: torch.device | str | None = None,
    ) -> UnnormalizerProcessorStep:
        """
        使用 `LeRobotDataset` 中的统计量创建 `UnnormalizerProcessorStep`。

        Args:
            dataset: 从中提取归一化统计量的数据集。
            features: 处理器的特征定义。
            norm_map: 从特征类型到归一化模式的映射。
            device: 处理器的目标设备。

        Returns:
            `UnnormalizerProcessorStep` 的新实例。
        """
        return cls(features=features, norm_map=norm_map, stats=dataset.meta.stats, device=device)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        # 处理观测反归一化。
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(observation, inverse=True)

        # 处理动作反归一化。
        action = new_transition.get(TransitionKey.ACTION)

        if action is None:
            return new_transition
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=True)

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def hotswap_stats(
    policy_processor: PolicyProcessorPipeline, stats: dict[str, dict[str, Any]]
) -> PolicyProcessorPipeline:
    """
    替换现有 `PolicyProcessorPipeline` 实例中的归一化统计量。

    该函数创建所提供流水线的深拷贝，并更新其中包含的任何
    `NormalizerProcessorStep` 或 `UnnormalizerProcessorStep` 的统计量。
    这对于将训练好的策略适配到具有不同数据分布的新环境或数据集很有用，
    而无需重建整个流水线。

    Args:
        policy_processor: 要修改的策略处理器流水线。
        stats: 要应用的新归一化统计量字典。

    Returns:
        具有更新后统计量的新 `PolicyProcessorPipeline` 实例。
    """
    rp = deepcopy(policy_processor)
    for step in rp.steps:
        if isinstance(step, _NormalizationMixin):
            step.stats = stats
            # 在正确的设备上重新初始化 tensor_stats。
            step._tensor_stats = to_tensor(stats, device=step.device, dtype=step.dtype)  # type: ignore[assignment]
    return rp
