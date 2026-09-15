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

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    LANGUAGE_EVENTS,
    LANGUAGE_PERSISTENT,
    MESSAGES_RENDERED,
    QUERY_KIND,
    QUERY_TEXT,
)
from lerobot.utils.utils import unwrap_scalar

from .pipeline import ComplementaryDataProcessorStep, ProcessorStep, ProcessorStepRegistry

if TYPE_CHECKING:
    from lerobot.datasets.recipe import TrainingRecipe


@dataclass
@ProcessorStepRegistry.register(name="render_training_messages_processor")
class RenderTrainingMessagesStep(ProcessorStep):
    """渲染训练标注，并在过滤样本时保持观测/动作的对齐。

    这是一个通用的 ProcessorStep，因为稀疏样本可能会从整个转移中被丢弃，
    而不仅仅是从补充数据中丢弃。
    如果没有 recipe，输入将原样透传。否则，仅对原始标注或动作目标运行，
    并跳过运行时查询。
    """

    recipe: TrainingRecipe | None = None
    dataset_ctx: Any | None = None

    def __post_init__(self) -> None:
        if isinstance(self.recipe, dict):
            # 仅在 recipe 场景下导入：datasets 包需要可选的 extras。
            from lerobot.datasets.recipe import TrainingRecipe

            self.recipe = TrainingRecipe.from_dict(self.recipe)

    def get_config(self) -> dict[str, Any]:
        return {
            "recipe": asdict(self.recipe) if self.recipe is not None else None,
        }

    def __call__(self, transition: EnvTransition) -> EnvTransition | None:
        """渲染单个样本或一个批次的训练标注。"""
        if self.recipe is None:
            return transition

        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        kind = complementary_data.get(QUERY_KIND)
        has_raw_language = LANGUAGE_PERSISTENT in complementary_data or LANGUAGE_EVENTS in complementary_data

        # 两个渲染器位于同一个已保存的流水线中。显式文本查询
        # 属于运行时渲染器；普通观测没有目标。
        if kind is not None:
            return transition
        if not has_raw_language and transition.get(TransitionKey.ACTION) is None:
            return transition

        if MESSAGES_RENDERED in complementary_data and not has_raw_language:
            return transition

        persistent = complementary_data.get(LANGUAGE_PERSISTENT) or []
        events = complementary_data.get(LANGUAGE_EVENTS) or []

        if not persistent and not events:
            # 没有语言标注的数据集仍然可用：将其 task 渲染为
            # 低层级监督，或在不存在 task 时直接透传。
            rendered = _fallback_low_level_render(complementary_data.get("task"))
            if rendered is None:
                return transition
            new_transition = transition.copy()
            new_complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
            new_complementary_data.update(rendered)
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
            return new_transition

        if _is_batched_language(persistent) or _is_batched_language(events):
            return self._call_batch(transition, complementary_data, persistent, events)

        timestamp = complementary_data.get("timestamp")
        if timestamp is None:
            raise KeyError("RenderTrainingMessagesStep requires sample timestamp in complementary data.")

        sample_idx = complementary_data.get("index", 0)
        rendered = _render_sample(
            recipe=self.recipe,
            persistent=persistent,
            events=events,
            t=unwrap_scalar(timestamp),
            sample_idx=int(unwrap_scalar(sample_idx)),
            task=complementary_data.get("task"),
            dataset_ctx=self.dataset_ctx,
        )
        if rendered is None:
            # 存在语言，但该稀疏帧没有适用的 recipe 分支。
            # 仅当可以进行任务级动作监督时才保留该样本。
            rendered = _fallback_low_level_render(complementary_data.get("task"))
            if rendered is None:
                return None

        new_transition = transition.copy()
        new_complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        new_complementary_data.pop(LANGUAGE_PERSISTENT, None)
        new_complementary_data.pop(LANGUAGE_EVENTS, None)
        new_complementary_data.update(rendered)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return new_transition

    def _call_batch(
        self,
        transition: EnvTransition,
        complementary_data: dict[str, Any],
        persistent_batch: list,
        events_batch: list,
    ) -> EnvTransition | None:
        """渲染语言批次。

        非空的 persistent 和 events 批次必须具有相同的大小。
        当批次中不存在某个语言列时，对应的列表可以为空。
        """
        timestamp = complementary_data.get("timestamp")
        if timestamp is None:
            raise KeyError("RenderTrainingMessagesStep requires sample timestamp in complementary data.")

        non_empty_batch_sizes = {len(batch) for batch in (persistent_batch, events_batch) if batch}
        if len(non_empty_batch_sizes) > 1:
            raise ValueError(
                "Batched language columns must have equal lengths when both are non-empty, "
                f"got persistent={len(persistent_batch)} and events={len(events_batch)}."
            )
        batch_size = next(iter(non_empty_batch_sizes), 0)
        messages: list[list[dict[str, Any]]] = []
        message_streams: list[list[str | None]] = []
        target_message_indices: list[list[int]] = []
        keep_indices: list[int] = []

        for i in range(batch_size):
            rendered = _render_sample(
                recipe=self.recipe,
                persistent=persistent_batch[i] if i < len(persistent_batch) else [],
                events=events_batch[i] if i < len(events_batch) else [],
                t=_batch_value(timestamp, i),
                sample_idx=int(_batch_value(complementary_data.get("index", 0), i)),
                task=_batch_value(complementary_data.get("task"), i),
                dataset_ctx=self.dataset_ctx,
            )
            if rendered is None:
                rendered = _fallback_low_level_render(_batch_value(complementary_data.get("task"), i))
                if rendered is None:
                    continue
            keep_indices.append(i)
            messages.append(rendered[MESSAGES_RENDERED])
            message_streams.append(rendered["message_streams"])
            target_message_indices.append(rendered["target_message_indices"])

        if not messages:
            raise ValueError(
                "Recipe-backed training produced no renderable samples in this batch. "
                "Provide the recipe's target annotations or a non-empty task fallback."
            )

        new_transition = (
            _select_batch_indices(transition, keep_indices, batch_size)
            if len(keep_indices) != batch_size
            else transition.copy()
        )
        new_complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        new_complementary_data.pop(LANGUAGE_PERSISTENT, None)
        new_complementary_data.pop(LANGUAGE_EVENTS, None)
        new_complementary_data[MESSAGES_RENDERED] = messages
        new_complementary_data["message_streams"] = message_streams
        new_complementary_data["target_message_indices"] = target_message_indices
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """保留特征形状；过滤仅改变批次长度。"""
        return features


@dataclass
@ProcessorStepRegistry.register(name="render_runtime_messages_processor")
class RenderRuntimeMessagesStep(ComplementaryDataProcessorStep):
    """使用检查点中保存的 recipe 渲染 VQA 和下一子任务提示。

    普通动作输入和已渲染的对话将直接透传。
    聊天模板、图像和分词仍由策略负责。
    """

    recipe: TrainingRecipe | None = None

    def __post_init__(self) -> None:
        if isinstance(self.recipe, dict):
            # 仅在 recipe 场景下导入：datasets 包需要可选的 extras。
            from lerobot.datasets.recipe import TrainingRecipe

            self.recipe = TrainingRecipe.from_dict(self.recipe)

    def get_config(self) -> dict[str, Any]:
        return {"recipe": asdict(self.recipe) if self.recipe is not None else None}

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        kind = complementary_data.get(QUERY_KIND)
        if kind is None:
            return complementary_data
        if LANGUAGE_PERSISTENT in complementary_data or LANGUAGE_EVENTS in complementary_data:
            raise ValueError("Runtime rendering cannot be combined with raw training language columns.")
        text = complementary_data.get(QUERY_TEXT)
        if not isinstance(text, str):
            raise TypeError(f"Text generation requires complementary data {QUERY_TEXT!r} to be a string.")

        if kind == "vqa":
            messages = [{"role": "user", "content": text}]
        elif kind == "next_subtask":
            if self.recipe is None:
                raise ValueError(
                    "Subtask generation requires a checkpoint recipe with an assistant target "
                    "that supervises `${subtask}`."
                )
            from lerobot.datasets.recipe import render_message_turns

            bindings = dict.fromkeys(self.recipe.referenced_binding_names())
            bindings.update(complementary_data)
            bindings["task"] = text
            messages = render_message_turns(self.recipe.prompt_turns("subtask"), bindings)[MESSAGES_RENDERED]
        else:
            raise ValueError(f"Unsupported query kind: {kind!r}. Expected one of: 'vqa', 'next_subtask'.")

        complementary_data.pop(QUERY_KIND)
        complementary_data.pop(QUERY_TEXT)
        complementary_data[MESSAGES_RENDERED] = messages
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def _is_batched_language(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], list)


def _render_sample(**kwargs) -> dict[str, Any] | None:
    """仅在 recipe 训练使用数据集渲染时才导入。"""
    from lerobot.datasets.language_render import render_sample

    return render_sample(**kwargs)


def _batch_value(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return value[index]
    if hasattr(value, "ndim") and value.ndim > 0:
        return unwrap_scalar(value[index])
    return unwrap_scalar(value)


def _select_batch_indices(transition: EnvTransition, indices: list[int], batch_size: int) -> EnvTransition:
    selected = transition.copy()
    for key in (TransitionKey.OBSERVATION, TransitionKey.COMPLEMENTARY_DATA):
        data = selected.get(key)
        if isinstance(data, dict):
            selected[key] = {
                name: _select_value(value, indices, batch_size, f"{key}.{name}")
                for name, value in data.items()
            }
    action = selected.get(TransitionKey.ACTION)
    if action is not None:
        selected[TransitionKey.ACTION] = _select_value(action, indices, batch_size, str(TransitionKey.ACTION))
    return selected


def _select_value(value: Any, indices: list[int], batch_size: int, path: str) -> Any:
    if isinstance(value, dict):
        return {key: _select_value(item, indices, batch_size, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        if len(value) != batch_size:
            raise ValueError(
                f"Cannot filter batched field {path!r}: expected {batch_size} values, got {len(value)}."
            )
        return [value[i] for i in indices]
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return value[indices]
    if hasattr(value, "index_select") and hasattr(value, "new_tensor") and getattr(value, "ndim", 0) > 0:
        return value.index_select(0, value.new_tensor(indices).long())
    return value


def _fallback_low_level_render(task: Any) -> dict[str, Any] | None:
    """当没有匹配的 recipe 分支时，保持仅有动作的样本可训练。"""
    if hasattr(task, "item"):
        task = task.item()
    if isinstance(task, list):
        if not task:
            return None
        messages = []
        message_streams = []
        target_message_indices = []
        missing_indices = []
        for index, t in enumerate(task):
            rendered = _fallback_low_level_render(t)
            if rendered is None:
                missing_indices.append(index)
                continue
            messages.append(rendered[MESSAGES_RENDERED])
            message_streams.append(rendered["message_streams"])
            target_message_indices.append(rendered["target_message_indices"])
        if missing_indices:
            if len(missing_indices) == len(task):
                return None
            raise ValueError(
                "Batched low-level fallback requires a non-empty task for every sample; "
                f"missing task at indices {missing_indices}."
            )
        return {
            MESSAGES_RENDERED: messages,
            "message_streams": message_streams,
            "target_message_indices": target_message_indices,
        }
    if not isinstance(task, str) or not task:
        return None
    return {
        MESSAGES_RENDERED: [{"role": "user", "content": task}],
        "message_streams": ["low_level"],
        "target_message_indices": [],
    }
