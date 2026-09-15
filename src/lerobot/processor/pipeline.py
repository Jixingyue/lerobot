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

"""
本模块定义了一个通用的顺序数据处理流水线框架，主要用于
转换机器人数据（观测、动作、奖励等）。

核心组件包括：
- ProcessorStep：单个数据转换操作的抽象基类。
- ProcessorStepRegistry：按名称注册和获取 ProcessorStep 类的机制。
- DataProcessorPipeline：将多个 ProcessorStep 实例链接起来构成完整
  数据处理工作流的类。它与 Hugging Face Hub 集成，便于共享和版本化
  流水线（包括其配置和状态）。
- 专用的 ProcessorStep 抽象子类（例如 ObservationProcessorStep、ActionProcessorStep），
  用于简化针对数据 transition 特定部分的步骤的创建。
"""

from __future__ import annotations

import importlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, TypeVar, cast

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import (
    EnvAction,
    EnvTransition,
    PolicyAction,
    RobotAction,
    RobotObservation,
    TransitionKey,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.hub import HubMixin

from .converters import batch_to_transition, create_transition, transition_to_batch

# 流水线输入和输出的通用类型变量。
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class ProcessorStepRegistry:
    """ProcessorStep 类的注册表，支持通过字符串名称实例化。

    本类提供字符串标识符到 `ProcessorStep` 类的映射，
    适用于从配置文件反序列化流水线，而无需
    硬编码类的导入。
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str | None = None):
        """注册 ProcessorStep 的类装饰器。

        Args:
            name: 注册该类时使用的名称。若为 None，则使用类的 `__name__`。

        Returns:
            注册该类并将其返回的装饰器函数。

        Raises:
            ValueError: 当同名步骤已经注册时。
        """

        def decorator(step_class: type) -> type:
            """执行注册的实际装饰器。"""
            registration_name = name if name is not None else step_class.__name__

            if registration_name in cls._registry:
                raise ValueError(
                    f"Processor step '{registration_name}' is already registered. "
                    f"Use a different name or unregister the existing one first."
                )

            cls._registry[registration_name] = step_class
            # 将注册名称存储在类上，便于序列化时查找。
            step_class._registry_name = registration_name
            return step_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """按名称从注册表中获取处理器步骤类。

        Args:
            name: 要获取的步骤名称。

        Returns:
            与给定名称对应的处理器步骤类。

        Raises:
            KeyError: 当注册表中找不到该名称时。
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Processor step '{name}' not found in registry. "
                f"Available steps: {available}. "
                f"Make sure the step is registered using @ProcessorStepRegistry.register()"
            )
        return cls._registry[name]

    @classmethod
    def unregister(cls, name: str) -> None:
        """从注册表中移除一个处理器步骤。

        Args:
            name: 要注销的步骤名称。
        """
        cls._registry.pop(name, None)

    @classmethod
    def list(cls) -> list[str]:
        """返回所有已注册处理器步骤名称的列表。"""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """清空注册表中的所有处理器步骤。"""
        cls._registry.clear()


class ProcessorStep(ABC):
    """数据处理流水线中单个步骤的抽象基类。

    每个步骤必须实现 `__call__` 方法以对数据
    transition 执行转换，并实现 `transform_features` 方法以描述它
    如何改变数据特征的形状或类型。

    子类可以通过实现 `state_dict` 和 `load_state_dict` 来选择性地保持状态。
    """

    _current_transition: EnvTransition | None = None

    @property
    def transition(self) -> EnvTransition:
        """提供对当前正在处理的最近一个 transition 的访问。

        适用于需要访问 transition 中主要目标之外其他部分的步骤
        （例如需要查看观测的动作处理步骤）。

        Raises:
            ValueError: 在步骤尚未使用 transition 调用时进行访问。
        """
        if self._current_transition is None:
            raise ValueError("Transition is not set. Make sure to call the step with a transition first.")
        return self._current_transition

    @abstractmethod
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """处理一个环境 transition。

        本方法应包含处理步骤的核心逻辑。

        Args:
            transition: 待处理的输入数据 transition。

        Returns:
            处理后的 transition。
        """
        return transition

    def get_config(self) -> dict[str, Any]:
        """返回本步骤的配置，用于序列化。

        Returns:
            可 JSON 序列化的配置参数字典。
        """
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """返回本步骤的状态（例如学习到的参数、运行均值）。

        Returns:
            将状态名称映射到张量的字典。
        """
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """从状态字典加载本步骤的状态。

        Args:
            state: 状态张量组成的字典。
        """
        return None

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        """保存非张量资产，并将构造函数参数映射到相对路径。"""
        return {}

    def reset(self) -> None:
        """重置处理器步骤的内部状态（如果有）。"""
        return None

    @abstractmethod
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """定义本步骤如何修改流水线特征的描述。

        本方法用于在数据流经流水线时跟踪数据形状、dtype
        或模态的变化，而无需处理实际数据。

        Args:
            features: 描述观测、动作等输入特征的字典。

        Returns:
            描述经过本步骤转换后输出特征的字典。
        """
        return features


class ProcessorKwargs(TypedDict, total=False):
    """流水线构造中使用的可选关键字参数的 TypedDict。"""

    to_transition: Callable[[dict[str, Any]], EnvTransition] | None
    to_output: Callable[[EnvTransition], Any] | None
    name: str | None
    before_step_hooks: list[Callable[[int, EnvTransition], None]] | None
    after_step_hooks: list[Callable[[int, EnvTransition], None]] | None


class ProcessorMigrationError(Exception):
    """当模型需要迁移到处理器格式时抛出。"""

    def __init__(self, model_path: str | Path, migration_command: str, original_error: str):
        self.model_path = model_path
        self.migration_command = migration_command
        self.original_error = original_error
        super().__init__(
            f"Model '{model_path}' requires migration to processor format. "
            f"Run: {migration_command}\n\nOriginal error: {original_error}"
        )


@dataclass
class DataProcessorPipeline[TInput, TOutput](HubMixin):
    """用于处理数据的顺序流水线，与 Hugging Face Hub 集成。

    本类将多个 `ProcessorStep` 实例链接起来，构成完整的
    数据处理工作流。它是泛型的，允许自定义输入和输出类型，
    这些类型由 `to_transition` 和 `to_output` 转换器处理。

    Attributes:
        steps: 组成流水线的 `ProcessorStep` 对象序列。
        name: 流水线的描述性名称。
        to_transition: 将原始输入数据转换为标准化 `EnvTransition` 格式的函数。
        to_output: 将最终的 `EnvTransition` 转换为所需输出格式的函数。
        before_step_hooks: 每个步骤执行前调用的函数列表。
        after_step_hooks: 每个步骤执行后调用的函数列表。
    """

    steps: Sequence[ProcessorStep] = field(default_factory=list)
    name: str = "DataProcessorPipeline"

    to_transition: Callable[[TInput], EnvTransition] = field(
        default_factory=lambda: cast(Callable[[TInput], EnvTransition], batch_to_transition), repr=False
    )
    to_output: Callable[[EnvTransition], TOutput] = field(
        default_factory=lambda: cast(Callable[[EnvTransition], TOutput], transition_to_batch),
        repr=False,
    )

    before_step_hooks: list[Callable[[int, EnvTransition], None]] = field(default_factory=list, repr=False)
    after_step_hooks: list[Callable[[int, EnvTransition], None]] = field(default_factory=list, repr=False)
    _serialized_state_filenames: tuple[str | None, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __call__(self, data: TInput) -> TOutput:
        """通过整条流水线处理输入数据。

        Args:
            data: 要处理的输入数据。

        Returns:
            指定输出格式下处理后的数据。
        """
        transition = self.to_transition(data)
        transformed_transition = self._forward(transition)
        return self.to_output(transformed_transition)

    def _forward(self, transition: EnvTransition) -> EnvTransition:
        """按顺序执行所有处理步骤和钩子。

        Args:
            transition: 初始的 `EnvTransition` 对象。

        Returns:
            应用所有步骤之后的最终 `EnvTransition`。
        """
        for idx, processor_step in enumerate(self.steps):
            # 执行前置钩子
            for hook in self.before_step_hooks:
                hook(idx, transition)

            transition = processor_step(transition)

            # 执行后置钩子
            for hook in self.after_step_hooks:
                hook(idx, transition)
        return transition

    def step_through(self, data: TInput) -> Iterable[EnvTransition]:
        """逐步处理数据，在每个阶段产出 transition。

        这是一个生成器方法，适用于调试和检查数据
        流经流水线时的中间状态。

        Args:
            data: 输入数据。

        Yields:
            `EnvTransition` 对象，从初始状态开始，随后是
            每个处理步骤之后的状态。
        """
        transition = self.to_transition(data)

        # 在任何处理之前先产出初始状态。
        yield transition

        for processor_step in self.steps:
            transition = processor_step(transition)
            yield transition

    def _get_sanitized_name(self) -> str:
        """返回流水线名称的文件名安全版本。

        Returns:
            小写的流水线名称，非字母数字字符替换为下划线。
        """
        return re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())

    @staticmethod
    def _get_state_filename(
        *,
        step_index: int,
        registry_name: str | None,
        sanitized_name: str,
    ) -> str:
        """返回一个有状态处理器步骤的 safetensors 文件名。

        Args:
            step_index: 处理器步骤在本流水线中的索引。
            registry_name: 已注册的处理器步骤名称（如果有）。
            sanitized_name: 文件名安全的流水线名称。

        Returns:
            现有磁盘序列化格式所使用的状态文件名。
        """
        if registry_name:
            return f"{sanitized_name}_step_{step_index}_{registry_name}.safetensors"

        return f"{sanitized_name}_step_{step_index}.safetensors"

    @staticmethod
    def _get_state_key(state_filename: str) -> str:
        """根据序列化的状态文件名返回内存中的状态键。

        Args:
            state_filename: 序列化配置中的 `.safetensors` 文件名。

        Returns:
            内存流水线状态字典使用的状态键。
        """
        return state_filename.removesuffix(".safetensors")

    @staticmethod
    def _get_state_filenames_from_config(loaded_config: dict[str, Any]) -> tuple[str | None, ...]:
        """按步骤顺序返回序列化的状态文件名。

        Args:
            loaded_config: 经过校验的处理器流水线配置。

        Returns:
            一个元组，包含每个步骤的序列化状态文件名；无状态步骤为 None。
        """
        return tuple(step_entry.get("state_file") for step_entry in loaded_config["steps"])

    def _get_state_filenames_for_loading(self) -> tuple[str | None, ...]:
        """返回 `load_state_dict()` 所需的、按步骤顺序的状态文件名。

        Returns:
            可用时保留序列化时的状态文件名，否则根据
            当前非空步骤状态派生文件名。
        """
        if self._serialized_state_filenames is not None and len(self._serialized_state_filenames) == len(
            self.steps
        ):
            return self._serialized_state_filenames

        sanitized_name = self._get_sanitized_name()
        state_filenames: list[str | None] = []

        for step_index, processor_step in enumerate(self.steps):
            step_state_dict = processor_step.state_dict()
            if not step_state_dict:
                state_filenames.append(None)
                continue

            registry_name = getattr(processor_step.__class__, "_registry_name", None)
            state_filenames.append(
                self._get_state_filename(
                    step_index=step_index,
                    registry_name=registry_name,
                    sanitized_name=sanitized_name,
                )
            )

        return tuple(state_filenames)

    def get_config(self) -> dict[str, Any]:
        """返回可 JSON 序列化的流水线配置。

        Returns:
            与 `save_pretrained()` 写为 JSON 的内容相同的字典。
        """
        sanitized_name = self._get_sanitized_name()
        pipeline_config: dict[str, Any] = {
            "name": self.name,
            "steps": [],
        }

        for step_index, processor_step in enumerate(self.steps):
            registry_name = getattr(processor_step.__class__, "_registry_name", None)
            step_entry: dict[str, Any] = {}

            if registry_name:
                step_entry["registry_name"] = registry_name
            else:
                step_entry["class"] = (
                    f"{processor_step.__class__.__module__}.{processor_step.__class__.__name__}"
                )

            step_entry["config"] = processor_step.get_config()

            step_state_dict = processor_step.state_dict()
            if step_state_dict:
                step_entry["state_file"] = self._get_state_filename(
                    step_index=step_index,
                    registry_name=registry_name,
                    sanitized_name=sanitized_name,
                )

            pipeline_config["steps"].append(step_entry)

        return pipeline_config

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """按状态键分组返回流水线状态张量。

        Returns:
            将无后缀状态键映射到各步骤状态字典（已克隆）的字典。
        """
        sanitized_name = self._get_sanitized_name()
        pipeline_state_dict: dict[str, dict[str, torch.Tensor]] = {}

        for step_index, processor_step in enumerate(self.steps):
            step_state_dict = processor_step.state_dict()
            if not step_state_dict:
                continue

            registry_name = getattr(processor_step.__class__, "_registry_name", None)
            state_filename = self._get_state_filename(
                step_index=step_index,
                registry_name=registry_name,
                sanitized_name=sanitized_name,
            )
            state_key = self._get_state_key(state_filename)
            pipeline_state_dict[state_key] = {
                tensor_name: tensor.clone() for tensor_name, tensor in step_state_dict.items()
            }

        return pipeline_state_dict

    def load_state_dict(
        self,
        state_dict: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """将流水线状态张量加载到现有步骤中。

        Args:
            state_dict: 将无后缀状态键映射到各步骤状态字典的字典。

        Raises:
            KeyError: 加载时发现缺少预期状态或出现意外的额外状态。
        """
        expected_state_filenames = self._get_state_filenames_for_loading()
        used_state_keys: set[str] = set()

        for step_index, (processor_step, state_filename) in enumerate(
            zip(self.steps, expected_state_filenames, strict=True)
        ):
            if state_filename is None:
                continue

            state_key = self._get_state_key(state_filename)
            if state_key not in state_dict:
                raise KeyError(
                    f"Missing state key '{state_key}' for processor step {step_index}. "
                    f"Available state keys: {sorted(state_dict.keys())}"
                )

            processor_step.load_state_dict(state_dict[state_key])
            used_state_keys.add(state_key)

        unexpected_state_keys = set(state_dict) - used_state_keys
        if unexpected_state_keys:
            expected_state_key_set = {
                self._get_state_key(state_filename)
                for state_filename in expected_state_filenames
                if state_filename is not None
            }
            raise KeyError(
                f"Unexpected processor state keys: {sorted(unexpected_state_keys)}. "
                f"Expected state keys: {sorted(expected_state_key_set)}"
            )

    def _save_pretrained(self, save_directory: Path, **kwargs) -> None:
        """遵循 `HubMixin` 保存机制的内部方法。

        本方法执行实际的保存工作，由 HubMixin.save_pretrained 调用。
        """
        config_filename = kwargs.pop("config_filename", None)
        sanitized_name = self._get_sanitized_name()

        if config_filename is None:
            config_filename = f"{sanitized_name}.json"

        pipeline_config = self.get_config()
        pipeline_state_dict = self.state_dict()

        for processor_step, step_entry in zip(self.steps, pipeline_config["steps"], strict=True):
            artifacts = processor_step.save_artifacts(save_directory)
            if artifacts:
                for config_key, relative_path in artifacts.items():
                    artifact_path = Path(relative_path)
                    if artifact_path.is_absolute() or ".." in artifact_path.parts:
                        raise ValueError(
                            f"Processor artifact path must be relative to the checkpoint: {relative_path!r}"
                        )
                    if not (save_directory / artifact_path).exists():
                        raise FileNotFoundError(
                            f"Processor step did not save declared artifact '{relative_path}'"
                        )
                    step_entry["config"][config_key] = artifact_path.as_posix()
                step_entry["artifacts"] = artifacts

        for state_key, step_state_dict in pipeline_state_dict.items():
            state_filename = f"{state_key}.safetensors"
            save_file(step_state_dict, save_directory / state_filename)

        with open(save_directory / config_filename, "w") as file_pointer:
            json.dump(pipeline_config, file_pointer, indent=2)

    def save_pretrained(
        self,
        save_directory: str | Path | None = None,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        card_kwargs: dict[str, Any] | None = None,
        config_filename: str | None = None,
        **push_to_hub_kwargs,
    ):
        """将流水线的配置和状态保存到目录中。

        本方法会创建一个定义流水线结构（名称和步骤）的
        JSON 配置文件。对于每个有状态步骤，还会保存一个包含其
        状态字典的 `.safetensors` 文件。

        Args:
            save_directory: 流水线的保存目录。若为 None，则保存到
                HF_LEROBOT_HOME/processors/{sanitized_pipeline_name}。
            repo_id: 你在 Hub 上的仓库 ID。仅在 `push_to_hub=true` 时使用。
            push_to_hub: 保存后是否将对象推送到 Hugging Face Hub。
            card_kwargs: 传给卡片模板以自定义卡片的额外参数。
            config_filename: JSON 配置文件的名称。若为 None，则根据
                流水线的 `name` 属性生成名称。
            **push_to_hub_kwargs: 转发给 push_to_hub 方法的额外关键字参数。
        """
        if save_directory is None:
            # 使用 HF_LEROBOT_HOME 中的默认目录
            sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())
            save_directory = HF_LEROBOT_HOME / "processors" / sanitized_name

        # 对于直接保存（不通过 hub），处理 config_filename
        if not push_to_hub and config_filename is not None:
            # 直接调用 _save_pretrained 并传入 config_filename
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            self._save_pretrained(save_directory, config_filename=config_filename)
            return None

        # 使用 hub 时，通过 kwargs 传递 config_filename 给 _save_pretrained
        if config_filename is not None:
            push_to_hub_kwargs["config_filename"] = config_filename

        # 调用父类的 save_pretrained，它会进一步调用我们的 _save_pretrained
        return super().save_pretrained(
            save_directory=save_directory,
            repo_id=repo_id,
            push_to_hub=push_to_hub,
            card_kwargs=card_kwargs,
            **push_to_hub_kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        config_filename: str,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[str, str] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        overrides: dict[str, Any] | None = None,
        to_transition: Callable[[TInput], EnvTransition] | None = None,
        to_output: Callable[[EnvTransition], TOutput] | None = None,
        **kwargs,
    ) -> DataProcessorPipeline[TInput, TOutput]:
        """从本地目录、单个文件或 Hugging Face Hub 仓库加载流水线。

        本方法实现了简化的加载流水线，并带有智能迁移检测：

        **简化加载策略**：
        1. **配置加载**（_load_config）：
           - **目录**：从目录中加载指定的 config_filename
           - **单个文件**：直接加载文件（忽略 config_filename）
           - **Hub 仓库**：从 Hub 下载指定的 config_filename

        2. **配置校验**（_validate_loaded_config）：
           - 格式校验：确保配置是有效的处理器格式
           - 迁移检测：引导用户迁移旧的 LeRobot 模型
           - 清晰的错误：提供可操作的错误消息

        3. **步骤构造**（_build_steps_with_overrides）：
           - 类解析：注册表查找或动态导入
           - 覆盖合并：用户参数覆盖已保存的配置
           - 状态加载：为有状态步骤加载 .safetensors 文件

        4. **覆盖校验**（_validate_overrides_used）：
           - 确保所有用户覆盖都已应用（捕获拼写错误）
           - 提供带有可用键的有用错误消息

        **迁移检测**：
        - **智能检测**：分析 JSON 文件以检测旧的 LeRobot 模型
        - **精确定向**：避免在其他 HuggingFace 模型上误报
        - **清晰指引**：提供要运行的确切迁移命令
        - **错误模式**：始终抛出 ProcessorMigrationError，以促使用户明确处理

        **加载示例**：
        ```python
        # Directory loading
        pipeline = DataProcessorPipeline.from_pretrained("/models/my_model", config_filename="processor.json")

        # Single file loading
        pipeline = DataProcessorPipeline.from_pretrained(
            "/models/my_model/processor.json", config_filename="processor.json"
        )

        # Hub loading
        pipeline = DataProcessorPipeline.from_pretrained("user/repo", config_filename="processor.json")

        # Multiple configs (preprocessor/postprocessor)
        preprocessor = DataProcessorPipeline.from_pretrained(
            "model", config_filename="policy_preprocessor.json"
        )
        postprocessor = DataProcessorPipeline.from_pretrained(
            "model", config_filename="policy_postprocessor.json"
        )
        ```

        **覆盖系统**：
        - **键匹配**：使用注册表名称或类名作为覆盖键
        - **配置合并**：用户覆盖优先于已保存的配置
        - **校验**：确保所有覆盖键都与实际步骤匹配（捕获拼写错误）
        - **示例**：overrides={"NormalizeStep": {"device": "cuda"}}

        Args:
            pretrained_model_name_or_path: Hugging Face Hub 上的仓库标识符、
                本地目录路径或单个配置文件路径。
            config_filename: 流水线 JSON 配置文件的名称。始终必填，
                以避免存在多个配置时产生歧义（例如预处理器与后处理器）。
            force_download: 是否强制（重新）下载文件。
            resume_download: 是否恢复之前中断的下载。
            proxies: 要使用的代理服务器字典。
            token: 用于私有 Hub 仓库的 HTTP bearer 授权令牌。
            cache_dir: 存储下载文件的特定缓存文件夹路径。
            local_files_only: 若为 True，则不从 Hub 下载文件。
            revision: 要使用的特定模型版本（例如分支名、标签名或提交 id）。
            overrides: 用于覆盖特定步骤配置的字典。键应
                与步骤的类名或注册表名称匹配。
            to_transition: 将输入数据转换为 `EnvTransition` 的自定义函数。
            to_output: 将最终 `EnvTransition` 转换为输出格式的自定义函数。
            **kwargs: 额外参数（未使用）。

        Returns:
            使用指定配置和状态加载的 `DataProcessorPipeline` 实例。

        Raises:
            FileNotFoundError: 找不到配置文件时。
            ValueError: 配置有歧义或实例化失败时。
            ImportError: 无法导入某个步骤的类时。
            KeyError: 覆盖键与流水线中任何步骤都不匹配时。
            ProcessorMigrationError: 模型需要迁移到处理器格式时。
        """
        model_id = str(pretrained_model_name_or_path)
        model_path = Path(model_id)
        is_local_source = model_path.is_dir() or model_path.is_file()
        hub_download_kwargs = {
            "force_download": force_download,
            "resume_download": resume_download,
            "proxies": proxies,
            "token": token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        # 1. 使用简化的三分支逻辑加载配置
        loaded_config, base_path = cls._load_config(model_id, config_filename, hub_download_kwargs)

        # 2. 校验配置并处理迁移
        cls._validate_loaded_config(model_id, loaded_config, config_filename)

        # 3. 带覆盖地构建步骤
        steps, validated_overrides = cls._build_steps_with_overrides(
            loaded_config,
            overrides or {},
            model_id,
            base_path,
            config_filename,
            hub_download_kwargs,
            is_local_source,
        )

        # 4. 校验所有覆盖都已被使用
        cls._validate_overrides_used(validated_overrides, loaded_config)

        # 5. 构造并返回最终的流水线实例
        pipeline = cls(
            steps=steps,
            name=loaded_config.get("name", "DataProcessorPipeline"),
            to_transition=to_transition or cast(Callable[[TInput], EnvTransition], batch_to_transition),
            to_output=to_output or cast(Callable[[EnvTransition], TOutput], transition_to_batch),
        )
        pipeline._serialized_state_filenames = cls._get_state_filenames_from_config(loaded_config)
        return pipeline

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        state_dict: dict[str, dict[str, torch.Tensor]] | None = None,
        overrides: dict[str, Any] | None = None,
        to_transition: Callable[[TInput], EnvTransition] | None = None,
        to_output: Callable[[EnvTransition], TOutput] | None = None,
    ) -> DataProcessorPipeline[TInput, TOutput]:
        """从内存中的配置和可选的状态张量构建流水线。

        Args:
            config: 与已保存处理器 JSON 结构相同的配置字典。
            state_dict: 可选的内存流水线状态，按无后缀状态键分组。
            overrides: 可选的构造函数覆盖，以注册表名或类名为键。
            to_transition: 可选的从输入数据到 `EnvTransition` 的转换器。
            to_output: 可选的从 `EnvTransition` 到输出数据的转换器。

        Returns:
            根据配置和可选状态构建的处理器流水线。
        """
        cls._validate_loaded_config("<in-memory config>", config, "<in-memory config>")

        steps, remaining_override_keys = cls._build_steps_from_config(config, overrides or {})
        cls._validate_overrides_used(remaining_override_keys, config)

        pipeline = cls(
            steps=steps,
            name=config.get("name", "DataProcessorPipeline"),
            to_transition=to_transition or cast(Callable[[TInput], EnvTransition], batch_to_transition),
            to_output=to_output or cast(Callable[[EnvTransition], TOutput], transition_to_batch),
        )
        pipeline._serialized_state_filenames = cls._get_state_filenames_from_config(config)

        if state_dict is not None:
            pipeline.load_state_dict(state_dict)

        return pipeline

    @classmethod
    def _load_config(
        cls,
        model_id: str,
        config_filename: str,
        hub_download_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], Path]:
        """从本地文件或 Hugging Face Hub 加载配置。

        本方法实现了极简的三分支加载策略：

        1. **本地目录**：从目录中加载 config_filename
           - 示例：model_id="/models/my_model"，config_filename="processor.json"
           - 加载："/models/my_model/processor.json"

        2. **单个文件**：直接加载文件（忽略 config_filename）
           - 示例：model_id="/models/my_model/processor.json"
           - 加载："/models/my_model/processor.json"（忽略 config_filename）

        3. **Hub 仓库**：从 Hub 下载 config_filename
           - 示例：model_id="user/repo"，config_filename="processor.json"
           - 下载并加载：Hub 仓库中的 config_filename

        **显式 config_filename 的好处**：
        - 没有自动检测的复杂性和边界情况
        - 没有加载错误配置的风险（预处理器与后处理器）
        - 本地和 Hub 使用时行为一致
        - 错误清晰、可预测

        Args:
            model_id: 模型标识符（Hub 仓库 ID、本地目录或文件路径）
            config_filename: 要加载的显式配置文件名（始终必填）
            hub_download_kwargs: hf_hub_download 的参数（令牌、缓存等）

        Returns:
            (loaded_config, base_path) 元组
            - loaded_config：解析后的 JSON 配置字典（始终已加载，永不为 None）
            - base_path：包含配置文件的目录（用于解析状态文件）

        Raises:
            FileNotFoundError: 在本地或 Hub 上都找不到配置文件时
        """
        model_path = Path(model_id)

        if model_path.is_dir():
            # 目录：从目录中加载指定的配置
            config_path = model_path / config_filename
            if not config_path.exists():
                # 在给出明确错误之前先检查是否需要迁移
                if cls._should_suggest_migration(model_path):
                    cls._suggest_processor_migration(model_id, f"Config file '{config_filename}' not found")
                raise FileNotFoundError(
                    f"Config file '{config_filename}' not found in directory '{model_id}'"
                )

            with open(config_path) as f:
                return json.load(f), model_path

        elif model_path.is_file():
            # 文件：直接加载（单个文件时忽略 config_filename）
            with open(model_path) as f:
                return json.load(f), model_path.parent

        else:
            # Hub：下载指定的配置
            try:
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename=config_filename,
                    repo_type="model",
                    **hub_download_kwargs,
                )

                with open(config_path) as f:
                    return json.load(f), Path(config_path).parent

            except Exception as e:
                if cls._hub_model_requires_migration(model_id, hub_download_kwargs):
                    revision = hub_download_kwargs.get("revision")
                    cls._suggest_processor_migration(
                        model_id,
                        f"Config file '{config_filename}' not found on the Hugging Face Hub",
                        revision=revision if isinstance(revision, str) else None,
                    )
                raise FileNotFoundError(
                    f"Could not find '{config_filename}' on the HuggingFace Hub at '{model_id}'"
                ) from e

    @classmethod
    def _validate_loaded_config(cls, model_id: str, loaded_config: Any, config_filename: str) -> None:
        """校验配置已加载且是有效的处理器配置。

        本方法带智能迁移检测地校验处理器配置格式：

        **配置格式校验**：
        - 使用 _is_processor_config() 校验结构
          - 必须有 "steps" 字段，且是步骤配置的列表
          - 每个步骤需要有 "class" 或 "registry_name"
        - 校验失败且为本地目录时：检查是否需要迁移
        - 需要迁移时：抛出带命令的 ProcessorMigrationError
        - 不需要迁移时：抛出带有用错误消息的 ValueError

        **迁移检测逻辑**：
        - 仅对本地目录触发（不对 Hub 仓库）
        - 分析目录中的所有 JSON 文件以检测旧的 LeRobot 模型
        - 提供带模型路径的确切迁移命令

        Args:
            model_id: 模型标识符（用于迁移检测）
            loaded_config: 加载的待校验配置值（可能不是字典）
            config_filename: 已加载的配置文件名（用于错误消息）

        Raises:
            ValueError: 配置格式无效时
            ProcessorMigrationError: 模型需要迁移到处理器格式时
        """
        # 校验它确实是一个处理器配置
        if not cls._is_processor_config(loaded_config):
            if Path(model_id).is_dir() and cls._should_suggest_migration(Path(model_id)):
                cls._suggest_processor_migration(
                    model_id,
                    f"Config file '{config_filename}' is not a valid processor configuration",
                )
            loaded_config_description = (
                list(loaded_config.keys())
                if isinstance(loaded_config, dict)
                else type(loaded_config).__name__
            )
            raise ValueError(
                f"Config file '{config_filename}' is not a valid processor configuration. "
                f"Expected a config with 'steps' field, but got: {loaded_config_description}"
            )

    @classmethod
    def _build_steps_with_overrides(
        cls,
        loaded_config: dict[str, Any],
        overrides: dict[str, Any],
        model_id: str,
        base_path: Path | None,
        config_filename: str,
        hub_download_kwargs: dict[str, Any],
        is_local_source: bool = False,
    ) -> tuple[list[ProcessorStep], set[str]]:
        """带覆盖和状态加载地构建所有处理器步骤。

        本方法编排完整的步骤构造流水线：

        **对于 loaded_config["steps"] 中的每个步骤**：

        0. **资产解析**（通过 _resolve_artifact_paths）：
           - 针对本地检查点解析声明的相对资产路径
           - 从 Hub 加载流水线时下载声明的资产
           - 在构造步骤之前拒绝绝对路径和路径穿越

        1. **类解析**（通过 _resolve_step_class）：
           - **如果存在 "registry_name"**：在 ProcessorStepRegistry 中查找
             示例：{"registry_name": "normalize_step"} -> 获取已注册的类
           - **否则使用 "class" 字段**：从完整模块路径动态导入
             示例：{"class": "lerobot.processor.normalize.NormalizeStep"}
           - **结果**：(step_class, step_key)，其中 step_key 用于覆盖

        2. **步骤实例化**（通过 _instantiate_step）：
           - **合并配置**：saved_config + user_overrides
           - **覆盖优先级**：用户覆盖优先于已保存的配置
           - **示例**：saved={"mean": 0.0}，override={"mean": 1.0} -> final={"mean": 1.0}
           - **结果**：实例化的 ProcessorStep 对象

        3. **状态加载**（通过 _load_step_state）：
           - **如果步骤有 "state_file"**：从 .safetensors 加载张量状态
           - **本地优先**：检查 base_path/state_file.safetensors
           - **Hub 回退**：流水线从 Hub 加载时下载状态文件
           - **可选**：仅在步骤有 load_state_dict 方法时加载

        4. **覆盖跟踪**：
           - **跟踪已使用的覆盖**：从剩余集合中移除 step_key
           - **目的**：校验所有用户覆盖都已应用（检测拼写错误）

        **错误处理**：
        - 类解析错误 -> 带有用消息的 ImportError
        - 实例化错误 -> 带配置详情的 ValueError
        - 状态加载错误 -> 由 load_state_dict 传播

        Args:
            loaded_config: 加载的处理器配置（必须有 "steps" 字段）
            overrides: 用户提供的参数覆盖（以类名/注册表名为键）
            model_id: 模型标识符（Hub 状态文件下载时需要）
            base_path: 用于查找状态文件的本地目录路径
            config_filename: 处理器配置路径，用作状态文件和声明资产
                相对于仓库的根路径。
            hub_download_kwargs: hf_hub_download 的参数（令牌、缓存等）
            is_local_source: model_id 是否解析为本地目录或配置文件。

        Returns:
            (instantiated_steps_list, unused_override_keys) 元组
            - instantiated_steps_list：开箱即用的 ProcessorStep 实例列表
            - unused_override_keys：未匹配任何步骤的覆盖键（用于校验）

        Raises:
            ImportError: 无法从注册表或导入路径加载步骤类时
            ValueError: 步骤无法用其配置实例化时
        """
        loaded_config = deepcopy(loaded_config)
        cls._resolve_artifact_paths(
            loaded_config,
            model_id,
            base_path,
            config_filename,
            hub_download_kwargs,
        )
        steps, remaining_override_keys = cls._build_steps_from_config(loaded_config, overrides)

        for step_instance, step_entry in zip(steps, loaded_config["steps"], strict=True):
            cls._load_step_state(
                step_instance,
                step_entry,
                model_id,
                base_path,
                config_filename,
                hub_download_kwargs,
                is_local_source,
            )

        return steps, remaining_override_keys

    @classmethod
    def _resolve_artifact_paths(
        cls,
        loaded_config: dict[str, Any],
        model_id: str,
        base_path: Path | None,
        config_filename: str,
        hub_download_kwargs: dict[str, Any],
    ) -> None:
        """在构造步骤之前解析声明的相对处理器资产。

        Args:
            loaded_config: 包含步骤资产声明的可变处理器配置。
            model_id: 本地检查点路径或 Hub 模型标识符。
            base_path: 包含已解析处理器配置的本地目录。
            config_filename: 处理器配置路径，其父目录是 Hub 上的资产根目录。
            hub_download_kwargs: Hub 下载的认证、版本和缓存参数。

        Raises:
            ValueError: 当声明的资产路径是绝对路径或逃逸出检查点时。
            FileNotFoundError: 当声明的资产在本地找不到且无法下载时。
        """
        is_local = Path(model_id).is_dir() or Path(model_id).is_file()

        for step_entry in loaded_config["steps"]:
            artifacts = step_entry.get("artifacts", {})
            for config_key, relative_path in artifacts.items():
                artifact_path = Path(relative_path)
                if artifact_path.is_absolute() or ".." in artifact_path.parts:
                    raise ValueError(
                        f"Processor artifact path must be relative to the checkpoint: {relative_path!r}"
                    )

                resolved_path = base_path / artifact_path if base_path is not None else artifact_path
                if not resolved_path.exists() and not is_local:
                    repository_path = Path(config_filename).parent / artifact_path
                    snapshot_download(
                        repo_id=model_id,
                        repo_type="model",
                        allow_patterns=f"{repository_path.as_posix()}/**",
                        **hub_download_kwargs,
                    )

                if not resolved_path.exists():
                    step_name = step_entry.get("registry_name", step_entry.get("class", "unknown"))
                    raise FileNotFoundError(
                        f"Missing processor artifact '{relative_path}' for step '{step_name}' "
                        f"next to '{config_filename}'. Checkpoint artifacts are incomplete."
                    )
                step_entry["config"][config_key] = str(resolved_path)

    @classmethod
    def _build_steps_from_config(
        cls,
        loaded_config: dict[str, Any],
        overrides: dict[str, Any],
    ) -> tuple[list[ProcessorStep], set[str]]:
        """从配置构建处理器步骤，不加载张量状态。

        Args:
            loaded_config: 加载的处理器配置。
            overrides: 用户提供的构造函数覆盖，以步骤键为键。

        Returns:
            一个元组，包含已实例化的步骤以及未匹配任何步骤的覆盖键。
        """
        processor_steps: list[ProcessorStep] = []
        remaining_override_keys = set(overrides.keys())

        for step_entry in loaded_config["steps"]:
            step_class, step_key = cls._resolve_step_class(step_entry)
            processor_step = cls._instantiate_step(step_entry, step_class, step_key, overrides)

            if step_key in remaining_override_keys:
                remaining_override_keys.discard(step_key)

            processor_steps.append(processor_step)

        return processor_steps, remaining_override_keys

    @classmethod
    def _resolve_step_class(cls, step_entry: dict[str, Any]) -> tuple[type[ProcessorStep], str]:
        """从注册表或导入路径解析步骤类。

        本方法实现两层解析策略：

        **第 1 层：基于注册表的解析**（首选）：
        - **如果 step_entry 中有 "registry_name"**：在 ProcessorStepRegistry 中查找
          - **优点**：更快、无需导入、保证兼容性
          - **示例**：{"registry_name": "normalize_step"} -> 获取预注册的类
          - **错误**：找不到 registry_name 时的 KeyError -> 转换为 ImportError

        **第 2 层：动态导入回退**：
        - **否则使用 "class" 字段**：完整的 module.ClassName 导入路径
          - **过程**：将 "module.path.ClassName" 拆分为模块和类两部分
          - **导入**：使用 importlib.import_module() + getattr()
          - **示例**："lerobot.processor.normalize.NormalizeStep"
            a. 导入模块："lerobot.processor.normalize"
            b. 获取类：getattr(module, "NormalizeStep")
          - **step_key**：使用类名（"NormalizeStep"）作为覆盖键

        **覆盖键策略**：
        - 注册表步骤：使用 registry_name（"normalize_step"）
        - 导入步骤：使用类名（"NormalizeStep"）
        - 允许用户使用 {"normalize_step": {...}} 或 {"NormalizeStep": {...}} 进行覆盖

        **错误处理**：
        - 注册表 KeyError -> 带注册表上下文的 ImportError
        - 导入/属性错误 -> 带有用建议的 ImportError
        - 所有错误都包含排查指引

        Args:
            step_entry: 步骤配置字典（必须有 "registry_name" 或 "class"）

        Returns:
            (step_class, step_key) 元组
            - step_class：解析出的 ProcessorStep 类（可直接实例化）
            - step_key：用于用户覆盖的键（registry_name 或类名）

        Raises:
            ImportError: 无法从注册表或导入路径加载步骤类时
        """
        if "registry_name" in step_entry:
            try:
                step_class = ProcessorStepRegistry.get(step_entry["registry_name"])
                return step_class, step_entry["registry_name"]
            except KeyError as e:
                raise ImportError(f"Failed to load processor step from registry. {str(e)}") from e
        else:
            # 回退为使用完整类路径动态导入
            full_class_path = step_entry["class"]
            module_path, class_name = full_class_path.rsplit(".", 1)

            try:
                module = importlib.import_module(module_path)
                step_class = getattr(module, class_name)
                return step_class, class_name
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Failed to load processor step '{full_class_path}'. "
                    f"Make sure the module '{module_path}' is installed and contains class '{class_name}'. "
                    f"Consider registering the step using @ProcessorStepRegistry.register() for better portability. "
                    f"Error: {str(e)}"
                ) from e

    @classmethod
    def _instantiate_step(
        cls,
        step_entry: dict[str, Any],
        step_class: type[ProcessorStep],
        step_key: str,
        overrides: dict[str, Any],
    ) -> ProcessorStep:
        """带配置覆盖地实例化单个处理器步骤。

        本方法处理配置合并和实例化逻辑：

        **配置合并策略**：
        1. **提取已保存配置**：从已保存流水线中获取 step_entry.get("config", {})
           - 示例：{"config": {"mean": 0.0, "std": 1.0}}
        2. **提取用户覆盖**：为该步骤获取 overrides.get(step_key, {})
           - 示例：overrides = {"NormalizeStep": {"mean": 2.0, "device": "cuda"}}
        3. **按优先级合并**：{**saved_cfg, **step_overrides}
           - **覆盖优先级**：用户值覆盖已保存的值
           - **结果**：{"mean": 2.0, "std": 1.0, "device": "cuda"}

        **实例化过程**：
        - **调用构造函数**：step_class(**merged_cfg)
        - **示例**：NormalizeStep(mean=2.0, std=1.0, device="cuda")

        **错误处理**：
        - **实例化期间的任何异常**：转换为 ValueError
        - **包含上下文**：步骤名称、尝试的配置、原始错误
        - **目的**：帮助用户调试配置问题
        - **常见原因**：
          a. 参数类型无效（用了 str 而非 float）
          b. 缺少必需参数
          c. 参数组合不兼容

        Args:
            step_entry: 来自已保存配置的步骤配置（包含 "config" 字典）
            step_class: 要实例化的步骤类（已解析）
            step_key: 用于覆盖的键（"registry_name" 或类名）
            overrides: 用户提供的参数覆盖（以 step_key 为键）

        Returns:
            实例化的处理器步骤（可直接使用）

        Raises:
            ValueError: 步骤无法实例化时，附带详细错误上下文
        """
        try:
            saved_cfg = step_entry.get("config", {})
            step_overrides = overrides.get(step_key, {})
            merged_cfg = {**saved_cfg, **step_overrides}
            return step_class(**merged_cfg)
        except Exception as e:
            step_name = step_entry.get("registry_name", step_entry.get("class", "Unknown"))
            raise ValueError(
                f"Failed to instantiate processor step '{step_name}' with config: {step_entry.get('config', {})}. "
                f"Error: {str(e)}"
            ) from e

    @classmethod
    def _load_step_state(
        cls,
        step_instance: ProcessorStep,
        step_entry: dict[str, Any],
        model_id: str,
        base_path: Path | None,
        config_filename: str,
        hub_download_kwargs: dict[str, Any],
        is_local_source: bool = False,
    ) -> None:
        """如果处理器步骤有状态字典则加载。

        本方法实现带本地/Hub 回退的条件状态加载：

        **前置条件检查**（不满足时提前返回）：
        1. **step_entry 中有 "state_file"**：步骤配置指定了状态文件
           - **若缺失**：该步骤没有已保存状态（例如无状态变换）
        2. **hasattr(step_instance, "load_state_dict")**：步骤支持状态加载
           - **若缺失**：该步骤未实现状态加载（很少见）

        **状态文件解析策略**：
        1. **本地文件优先**：检查 base_path/state_filename 是否存在
           - **优点**：更快，无需网络调用
           - **示例**："/models/my_model/normalize_step_0.safetensors"
           - **使用场景**：从本地保存的模型目录加载

        2. **Hub 下载回退**：从仓库下载状态文件
           - **触发时机**：本地文件未找到且流水线来源是 Hub 仓库
           - **过程**：使用与配置相同的参数调用 hf_hub_download
           - **示例**：从 "user/repo" 下载 "normalize_step_0.safetensors"
           - **结果**：下载到本地缓存并返回路径

        **状态加载过程**：
        - **加载张量**：使用 safetensors.torch.load_file()
        - **应用到步骤**：调用 step_instance.load_state_dict(tensor_dict)
        - **原地修改**：更新步骤的内部张量状态

        **常见状态文件示例**：
        - "normalize_step_0.safetensors"——归一化统计量
        - "custom_step_1.safetensors"——学习到的参数
        - "tokenizer_step_2.safetensors"——词表嵌入

        Args:
            step_instance: 要加载状态的步骤实例（必须有 load_state_dict）
            step_entry: 步骤配置字典（可能包含 "state_file"）
            model_id: 模型标识符（需要时用于 Hub 下载）
            base_path: 查找状态文件的本地目录路径（仅 Hub 时为 None）
            config_filename: 处理器配置路径，其父目录用于解析
                Hub 上相对于仓库的状态文件。
            hub_download_kwargs: hf_hub_download 的参数（令牌、缓存等）
            is_local_source: model_id 是否解析为本地目录或配置文件。

        Note:
            本方法原地修改 step_instance 并返回 None。
            如果状态加载失败，load_state_dict 抛出的异常会向上传播。
        """
        if "state_file" not in step_entry or not hasattr(step_instance, "load_state_dict"):
            return

        state_filename = step_entry["state_file"]

        # 先尝试本地文件
        if base_path and (base_path / state_filename).exists():
            state_path = str(base_path / state_filename)
        elif is_local_source:
            state_path = base_path / state_filename if base_path else Path(state_filename)
            raise FileNotFoundError(
                f"State file '{state_filename}' was not found for local processor pipeline "
                f"'{model_id}' at '{state_path}'."
            )
        else:
            # 从 Hub 下载
            state_path = hf_hub_download(
                repo_id=model_id,
                filename=(Path(config_filename).parent / state_filename).as_posix(),
                repo_type="model",
                **hub_download_kwargs,
            )

        step_instance.load_state_dict(load_file(state_path))

    @classmethod
    def _validate_overrides_used(
        cls, remaining_override_keys: set[str], loaded_config: dict[str, Any]
    ) -> None:
        """校验所有提供的覆盖都已被使用。

        本方法确保用户覆盖有效，以捕获拼写错误和配置错误：

        **校验逻辑**：
        1. **如果 remaining_override_keys 为空**：所有覆盖都已使用 -> 成功
           - **提前返回**：无需校验
           - **正常情况**：用户提供了正确的覆盖键

        2. **如果 remaining_override_keys 有条目**：部分覆盖未使用 -> 错误
           - **根本原因**：用户提供的键与任何步骤都不匹配
           - **常见问题**：
             a. 步骤名拼写错误（"NormalizStep" 与 "NormalizeStep"）
             b. 使用了错误的键类型（类名与注册表名）
             c. 已保存流水线中不存在该步骤

        **生成有用的错误**：
        - **提取可用键**：从配置构建有效覆盖键列表
          a. **注册表步骤**：直接使用 "registry_name"
          b. **导入步骤**：从 "class" 字段提取类名
          - 示例："lerobot.processor.normalize.NormalizeStep" -> "NormalizeStep"
        - **错误消息包含**：
          a. 用户提供的无效键
          b. 他们可以使用的有效键列表
          c. 关于注册表名与类名的指引

        **覆盖键解析规则**：
        - 有 "registry_name" 的步骤：覆盖时使用 registry_name
        - 有 "class" 的步骤：覆盖时使用最终类名
        - 用户必须在其覆盖字典中使用这些确切的键

        Args:
            remaining_override_keys: 未匹配到任何步骤的覆盖键
            loaded_config: 加载的处理器配置（包含 "steps" 列表）

        Raises:
            KeyError: 当有覆盖键未被使用时，附带有用的错误消息
        """
        if not remaining_override_keys:
            return

        available_keys = [
            step.get("registry_name") or step["class"].rsplit(".", 1)[1] for step in loaded_config["steps"]
        ]

        raise KeyError(
            f"Override keys {list(remaining_override_keys)} do not match any step in the saved configuration. "
            f"Available step keys: {available_keys}. "
            f"Make sure override keys match exact step class names or registry names."
        )

    @classmethod
    def _should_suggest_migration(cls, model_path: Path) -> bool:
        """检查目录中是否有 JSON 文件但没有处理器配置。

        本方法实现智能迁移检测以避免误报：

        **判定逻辑**：
        1. **未找到 JSON 文件**：返回 False
           - **原因**：空目录或只有非配置文件
           - **示例**：只包含 .safetensors、.md 文件的目录
           - **动作**：无需迁移

        2. **存在 JSON 文件**：逐个分析
           - **目标**：确定是否有任意文件是有效的处理器配置
           - **过程**：
             a. 尝试解析每个 .json 文件
             b. 跳过有 JSON 解析错误的文件（格式错误）
             c. 检查解析出的配置是否通过 _is_processor_config()
           - **若找到任意有效处理器配置**：返回 False（无需迁移）
           - **若没有任何有效处理器配置**：返回 True（需要迁移）

        **示例**：
        - **无需迁移**：["processor.json", "config.json"]，其中 processor.json 有效
        - **需要迁移**：["config.json", "train.json"]，两者都是模型配置
        - **无需迁移**：[]（空目录）
        - **需要迁移**：["old_model_config.json"]，为旧 LeRobot 格式

        **为什么有效**：
        - **精确检测**：仅对真正的旧 LeRobot 模型建议迁移
        - **避免误报**：不会在其他 HuggingFace 模型类型上触发
        - **优雅处理**：忽略格式错误的 JSON 文件

        Args:
            model_path: 要分析的本地目录路径

        Returns:
            目录有 JSON 配置但没有一个是处理器配置（需要迁移）时返回 True
            没有 JSON 文件或至少存在一个有效处理器配置时返回 False
        """
        json_files = list(model_path.glob("*.json"))
        if len(json_files) == 0:
            return False

        # 检查是否有任意 JSON 文件是处理器配置
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    config = json.load(f)

                if cls._is_processor_config(config):
                    return False  # 找到至少一个处理器配置，无需迁移

            except (json.JSONDecodeError, OSError):
                # 跳过无法解析为 JSON 的文件
                continue

        # 有 JSON 文件但没有处理器配置——建议迁移
        return True

    @classmethod
    def _hub_model_requires_migration(cls, model_id: str, hub_download_kwargs: dict[str, Any]) -> bool:
        """检查 Hub 仓库是否包含旧版 LeRobot 策略配置。

        缺少处理器文件本身并不足以作为证据：该仓库
        可能是私有的、不可用的，或与 LeRobot 无关。因此本方法会
        获取策略的 ``config.json`` 并检查那些可标识
        LeRobot 策略检查点的特征声明。任何查找或解析失败都会被
        忽略，以使原始的处理器文件错误保持可见。

        Args:
            model_id: Hugging Face Hub 模型仓库 ID。
            hub_download_kwargs: 原始处理器查找所使用的认证、缓存和版本参数。

        Returns:
            当仓库具有旧版 LeRobot 策略配置时返回 True。
        """
        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                repo_type="model",
                **hub_download_kwargs,
            )
            with open(config_path) as f:
                config = json.load(f)
        except Exception:
            # 这是在处理原始处理器查找失败时进行的尽力而为诊断，
            # 原始错误必须保持为可见错误。
            return False

        feature_types = {feature_type.value for feature_type in FeatureType}

        def is_policy_feature_mapping(features: Any) -> bool:
            return (
                isinstance(features, dict)
                and bool(features)
                and all(
                    isinstance(name, str)
                    and isinstance(feature, dict)
                    and feature.get("type") in feature_types
                    and isinstance(feature.get("shape"), list)
                    and all(isinstance(dimension, int) for dimension in feature["shape"])
                    for name, feature in features.items()
                )
            )

        return (
            isinstance(config, dict)
            and isinstance(config.get("type"), str)
            and bool(config["type"])
            and is_policy_feature_mapping(config.get("input_features"))
            and is_policy_feature_mapping(config.get("output_features"))
        )

    @classmethod
    def _is_processor_config(cls, config: Any) -> bool:
        """检查配置是否遵循 DataProcessorPipeline 格式。

        本方法校验处理器配置结构：

        **必需结构校验**：
        1. **"steps" 字段存在**：必须有顶层 "steps" 键
           - **若缺失**：不是处理器配置（例如模型配置、训练配置）
           - **无效示例**：{"type": "act", "hidden_dim": 256}

        2. **"steps" 字段类型**：必须是列表，不能是其他类型
           - **若不是列表**：格式无效
           - **无效示例**：{"steps": "some_string"} 或 {"steps": {"key": "value"}}

        3. **空步骤校验**：空列表有效
           - **若 len(steps) == 0**：立即返回 True
           - **使用场景**：空处理器流水线（无操作）
           - **有效示例**：{"name": "EmptyProcessor", "steps": []}

        **单个步骤校验**（对于非空 steps）：
        对 steps 列表中的每个步骤：
        1. **步骤类型**：必须是字典
           - **若不是字典**：步骤格式无效
           - **无效示例**：["string_step", 123, true]

        2. **步骤标识符**：必须有 "class" 或 "registry_name"
           - **"registry_name"**：已注册步骤（首选）
             示例：{"registry_name": "normalize_step", "config": {...}}
           - **"class"**：完整导入路径
             示例：{"class": "lerobot.processor.normalize.NormalizeStep"}
           - **两者都没有**：步骤无效（无法解析类）
           - **两者都有**：同样有效（registry_name 优先）

        **有效处理器配置示例**：
        - {"steps": []}——空处理器
        - {"steps": [{"registry_name": "normalize"}]}——注册表步骤
        - {"steps": [{"class": "my.module.Step"}]}——导入步骤
        - {"name": "MyProcessor", "steps": [...]}——带名称

        **无效配置示例**：
        - {"type": "act"}——缺少 "steps"
        - {"steps": "normalize"}——steps 不是列表
        - {"steps": [{}]}——步骤缺少 class/registry_name
        - {"steps": ["string"]}——步骤不是字典

        Args:
            config: 要校验的配置字典

        Returns:
            配置遵循有效的 DataProcessorPipeline 格式时返回 True，否则返回 False
        """
        if not isinstance(config, dict):
            return False

        # 必须有一个 "steps" 字段，且值为步骤配置列表
        if not isinstance(config.get("steps"), list):
            return False

        steps = config["steps"]
        if len(steps) == 0:
            return True  # 空处理器是有效的

        # 每个步骤必须是字典，且包含 "class" 或 "registry_name"
        for step in steps:
            if not isinstance(step, dict):
                return False
            if not ("class" in step or "registry_name" in step):
                return False

        return True

    @classmethod
    def _suggest_processor_migration(
        cls,
        model_path: str | Path,
        original_error: str,
        *,
        revision: str | None = None,
    ) -> None:
        """当检测到有 JSON 文件但没有处理器配置时抛出迁移错误。

        当迁移检测判定某个模型目录包含配置文件、
        但没有一个是有效处理器配置时调用本方法。
        这通常表示该模型是需要迁移的旧 LeRobot 模型。

        **调用时机**：
        - 用户尝试从本地目录加载 DataProcessorPipeline
        - 目录包含 JSON 配置文件
        - 没有任何 JSON 文件遵循处理器配置格式
        - _should_suggest_migration() 返回了 True

        **迁移命令生成**：
        - 构造用户需要运行的确切命令
        - 使用迁移脚本：migrate_policy_normalization.py
        - 自动包含模型路径
        - 示例："python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path /models/old_model"

        **错误结构**：
        - **始终抛出**：ProcessorMigrationError（绝不正常返回）
        - **包含**：model_path、migration_command、original_error
        - **目的**：强制用户注意迁移需求
        - **用户体验**：清晰、可操作的错误，并给出要运行的确切命令

        **迁移过程**：
        建议的命令将：
        1. 从旧模型中提取归一化统计量
        2. 创建新的处理器配置（预处理器 + 后处理器）
        3. 从模型中移除归一化层
        4. 保存带处理器流水线的迁移后模型

        Args:
            model_path: 需要迁移的模型目录路径
            original_error: 触发迁移检测的错误（用于提供上下文）
            revision: 包含旧检查点的可选 Hub 版本。

        Raises:
            ProcessorMigrationError: 始终抛出（本方法不会正常返回）
        """
        migration_command = (
            f"python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path {model_path}"
        )
        if revision is not None:
            migration_command += f" --revision {revision}"

        raise ProcessorMigrationError(model_path, migration_command, original_error)

    def __len__(self) -> int:
        """返回流水线中的步骤数量。"""
        return len(self.steps)

    def __getitem__(self, idx: int | slice) -> ProcessorStep | DataProcessorPipeline[TInput, TOutput]:
        """按索引或切片获取一个步骤或一个子流水线。

        Args:
            idx: 整数索引或切片对象。

        Returns:
            如果 `idx` 是整数则返回一个 `ProcessorStep`，如果是切片
            则返回一个包含所切出步骤的新 `DataProcessorPipeline`。
        """
        if isinstance(idx, slice):
            # 返回一个包含所切出步骤的新流水线实例。
            return DataProcessorPipeline(
                steps=self.steps[idx],
                name=self.name,
                to_transition=self.to_transition,
                to_output=self.to_output,
                before_step_hooks=self.before_step_hooks.copy(),
                after_step_hooks=self.after_step_hooks.copy(),
            )
        return self.steps[idx]

    def register_before_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """注册一个在每个步骤之前调用的函数。

        Args:
            fn: 接收步骤索引和当前 transition 的可调用对象。
        """
        self.before_step_hooks.append(fn)

    def unregister_before_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """注销一个 'before_step' 钩子。

        Args:
            fn: 之前注册时使用的确切函数对象。

        Raises:
            ValueError: 在列表中找不到该钩子时。
        """
        try:
            self.before_step_hooks.remove(fn)
        except ValueError:
            raise ValueError(
                f"Hook {fn} not found in before_step_hooks. Make sure to pass the exact same function reference."
            ) from None

    def register_after_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """注册一个在每个步骤之后调用的函数。

        Args:
            fn: 接收步骤索引和当前 transition 的可调用对象。
        """
        self.after_step_hooks.append(fn)

    def unregister_after_step_hook(self, fn: Callable[[int, EnvTransition], None]):
        """注销一个 'after_step' 钩子。

        Args:
            fn: 之前注册时使用的确切函数对象。

        Raises:
            ValueError: 在列表中找不到该钩子时。
        """
        try:
            self.after_step_hooks.remove(fn)
        except ValueError:
            raise ValueError(
                f"Hook {fn} not found in after_step_hooks. Make sure to pass the exact same function reference."
            ) from None

    def reset(self):
        """重置流水线中所有有状态步骤的状态。"""
        for step in self.steps:
            if hasattr(step, "reset"):
                step.reset()

    def __repr__(self) -> str:
        """提供流水线的简洁字符串表示。"""
        step_names = [step.__class__.__name__ for step in self.steps]

        if not step_names:
            steps_repr = "steps=0: []"
        elif len(step_names) <= 3:
            steps_repr = f"steps={len(step_names)}: [{', '.join(step_names)}]"
        else:
            # 对于很长的流水线，显示第一个、第二个和最后一个步骤。
            displayed = f"{step_names[0]}, {step_names[1]}, ..., {step_names[-1]}"
            steps_repr = f"steps={len(step_names)}: [{displayed}]"

        parts = [f"name='{self.name}'", steps_repr]

        return f"DataProcessorPipeline({', '.join(parts)})"

    def __post_init__(self):
        """校验所有提供的步骤都是 `ProcessorStep` 的实例。"""
        for i, step in enumerate(self.steps):
            if not isinstance(step, ProcessorStep):
                raise TypeError(f"Step {i} ({type(step).__name__}) must inherit from ProcessorStep")

    def transform_features(
        self, initial_features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """依次应用所有步骤的特征变换。

        本方法将特征描述字典依次传递给每个步骤的
        `transform_features` 方法，使流水线能够静态地确定
        输出特征规格，而无需处理任何真实数据。

        Args:
            initial_features: 描述初始特征的字典。

        Returns:
            所有变换之后的最终特征描述。
        """
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = deepcopy(initial_features)

        for _, step in enumerate(self.steps):
            out = step.transform_features(features)
            features = out
        return features

    # 处理 transition 各独立部分的便捷方法。
    def process_observation(self, observation: RobotObservation) -> RobotObservation:
        """仅将 transition 的观测部分通过流水线处理。

        Args:
            observation: 观测字典。

        Returns:
            处理后的观测字典。
        """
        transition: EnvTransition = create_transition(observation=observation)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.OBSERVATION]

    def process_action(
        self, action: PolicyAction | RobotAction | EnvAction
    ) -> PolicyAction | RobotAction | EnvAction:
        """仅将 transition 的动作部分通过流水线处理。

        Args:
            action: 动作数据。

        Returns:
            处理后的动作。
        """
        transition: EnvTransition = create_transition(action=action)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.ACTION]

    def process_reward(self, reward: float | torch.Tensor) -> float | torch.Tensor:
        """仅将 transition 的奖励部分通过流水线处理。

        Args:
            reward: 奖励值。

        Returns:
            处理后的奖励。
        """
        transition: EnvTransition = create_transition(reward=reward)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.REWARD]

    def process_done(self, done: bool | torch.Tensor) -> bool | torch.Tensor:
        """仅将 transition 的 done 标志通过流水线处理。

        Args:
            done: done 标志。

        Returns:
            处理后的 done 标志。
        """
        transition: EnvTransition = create_transition(done=done)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.DONE]

    def process_truncated(self, truncated: bool | torch.Tensor) -> bool | torch.Tensor:
        """仅将 transition 的 truncated 标志通过流水线处理。

        Args:
            truncated: truncated 标志。

        Returns:
            处理后的 truncated 标志。
        """
        transition: EnvTransition = create_transition(truncated=truncated)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.TRUNCATED]

    def process_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """仅将 transition 的 info 字典通过流水线处理。

        Args:
            info: info 字典。

        Returns:
            处理后的 info 字典。
        """
        transition: EnvTransition = create_transition(info=info)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.INFO]

    def process_complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        """仅将 transition 的 complementary data 部分通过流水线处理。

        Args:
            complementary_data: complementary data 字典。

        Returns:
            处理后的 complementary data 字典。
        """
        transition: EnvTransition = create_transition(complementary_data=complementary_data)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.COMPLEMENTARY_DATA]


# 用于语义清晰的类型别名。
RobotProcessorPipeline = DataProcessorPipeline[TInput, TOutput]
PolicyProcessorPipeline = DataProcessorPipeline[TInput, TOutput]


class ObservationProcessorStep(ProcessorStep, ABC):
    """专门针对 transition 中观测的抽象 `ProcessorStep`。"""

    @abstractmethod
    def observation(self, observation: RobotObservation) -> RobotObservation:
        """处理观测字典。子类必须实现本方法。

        Args:
            observation: 来自 transition 的输入观测字典。

        Returns:
            处理后的观测字典。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `observation` 方法应用于 transition 的观测。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            raise ValueError("ObservationProcessorStep requires an observation in the transition.")

        processed_observation = self.observation(observation.copy())
        new_transition[TransitionKey.OBSERVATION] = processed_observation
        return new_transition


class ActionProcessorStep(ProcessorStep, ABC):
    """专门针对 transition 中动作的抽象 `ProcessorStep`。"""

    @abstractmethod
    def action(
        self, action: PolicyAction | RobotAction | EnvAction
    ) -> PolicyAction | RobotAction | EnvAction:
        """处理动作。子类必须实现本方法。

        Args:
            action: 来自 transition 的输入动作。

        Returns:
            处理后的动作。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `action` 方法应用于 transition 的动作。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("ActionProcessorStep requires an action in the transition.")

        processed_action = self.action(action)
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class RobotActionProcessorStep(ProcessorStep, ABC):
    """用于处理 `RobotAction`（字典）的抽象 `ProcessorStep`。"""

    @abstractmethod
    def action(self, action: RobotAction) -> RobotAction:
        """处理 `RobotAction`。子类必须实现本方法。

        Args:
            action: 输入的 `RobotAction` 字典。

        Returns:
            处理后的 `RobotAction`。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `action` 方法应用于 transition 的动作，并确保其为 `RobotAction`。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None or not isinstance(action, dict):
            raise ValueError(f"Action should be a RobotAction type (dict), but got {type(action)}")

        processed_action = self.action(action.copy())
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class PolicyActionProcessorStep(ProcessorStep, ABC):
    """用于处理 `PolicyAction`（张量或张量字典）的抽象 `ProcessorStep`。"""

    @abstractmethod
    def action(self, action: PolicyAction) -> PolicyAction:
        """处理 `PolicyAction`。子类必须实现本方法。

        Args:
            action: 输入的 `PolicyAction`。

        Returns:
            处理后的 `PolicyAction`。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `action` 方法应用于 transition 的动作，并确保其为 `PolicyAction`。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type (tensor), but got {type(action)}")

        processed_action = self.action(action)
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class RewardProcessorStep(ProcessorStep, ABC):
    """专门针对 transition 中奖励的抽象 `ProcessorStep`。"""

    @abstractmethod
    def reward(self, reward) -> float | torch.Tensor:
        """处理奖励。子类必须实现本方法。

        Args:
            reward: 来自 transition 的输入奖励。

        Returns:
            处理后的奖励。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `reward` 方法应用于 transition 的奖励。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        reward = new_transition.get(TransitionKey.REWARD)
        if reward is None:
            raise ValueError("RewardProcessorStep requires a reward in the transition.")

        processed_reward = self.reward(reward)
        new_transition[TransitionKey.REWARD] = processed_reward
        return new_transition


class DoneProcessorStep(ProcessorStep, ABC):
    """专门针对 transition 中 'done' 标志的抽象 `ProcessorStep`。"""

    @abstractmethod
    def done(self, done) -> bool | torch.Tensor:
        """处理 'done' 标志。子类必须实现本方法。

        Args:
            done: 来自 transition 的输入 'done' 标志。

        Returns:
            处理后的 'done' 标志。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `done` 方法应用于 transition 的 'done' 标志。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        done = new_transition.get(TransitionKey.DONE)
        if done is None:
            raise ValueError("DoneProcessorStep requires a done flag in the transition.")

        processed_done = self.done(done)
        new_transition[TransitionKey.DONE] = processed_done
        return new_transition


class TruncatedProcessorStep(ProcessorStep, ABC):
    """专门针对 transition 中 'truncated' 标志的抽象 `ProcessorStep`。"""

    @abstractmethod
    def truncated(self, truncated) -> bool | torch.Tensor:
        """处理 'truncated' 标志。子类必须实现本方法。

        Args:
            truncated: 来自 transition 的输入 'truncated' 标志。

        Returns:
            处理后的 'truncated' 标志。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `truncated` 方法应用于 transition 的 'truncated' 标志。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        truncated = new_transition.get(TransitionKey.TRUNCATED)
        if truncated is None:
            raise ValueError("TruncatedProcessorStep requires a truncated flag in the transition.")

        processed_truncated = self.truncated(truncated)
        new_transition[TransitionKey.TRUNCATED] = processed_truncated
        return new_transition


class InfoProcessorStep(ProcessorStep, ABC):
    """专门针对 transition 中 'info' 字典的抽象 `ProcessorStep`。"""

    @abstractmethod
    def info(self, info) -> dict[str, Any]:
        """处理 'info' 字典。子类必须实现本方法。

        Args:
            info: 来自 transition 的输入 'info' 字典。

        Returns:
            处理后的 'info' 字典。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `info` 方法应用于 transition 的 'info' 字典。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        info = new_transition.get(TransitionKey.INFO)
        if info is None or not isinstance(info, dict):
            raise ValueError("InfoProcessorStep requires an info dictionary in the transition.")

        processed_info = self.info(info.copy())
        new_transition[TransitionKey.INFO] = processed_info
        return new_transition


class ComplementaryDataProcessorStep(ProcessorStep, ABC):
    """针对 transition 中 'complementary_data' 的抽象 `ProcessorStep`。"""

    @abstractmethod
    def complementary_data(self, complementary_data) -> dict[str, Any]:
        """处理 'complementary_data' 字典。子类必须实现本方法。

        Args:
            complementary_data: 来自 transition 的输入 'complementary_data'。

        Returns:
            处理后的 'complementary_data' 字典。
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将 `complementary_data` 方法应用于 transition 的数据。"""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None or not isinstance(complementary_data, dict):
            raise ValueError("ComplementaryDataProcessorStep requires complementary data in the transition.")

        processed_complementary_data = self.complementary_data(complementary_data.copy())
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = processed_complementary_data
        return new_transition


class IdentityProcessorStep(ProcessorStep):
    """无操作处理器步骤，原样返回输入的 transition 和特征。

    可用作占位符或用于调试。
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """原样返回 transition，不做修改。"""
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """原样返回特征，不做修改。"""
        return features
