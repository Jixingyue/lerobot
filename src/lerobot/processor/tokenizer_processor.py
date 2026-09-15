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
此脚本定义了一个处理器，用于对来自环境转移（transition）的自然语言指令进行分词（tokenize）。

它使用 Hugging Face `transformers` 库中的分词器，将任务描述（文本）转换为
token ID 和注意力掩码（attention mask），然后将它们添加到观测（observation）字典中。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, RobotObservation, TransitionKey
from lerobot.utils.constants import (
    ACTION_CODE_TOKEN_MASK,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import _transformers_available

from .pipeline import ActionProcessorStep, ObservationProcessorStep, ProcessorStepRegistry

# 用于类型检查和懒加载的条件导入
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, AutoTokenizer
else:
    AutoProcessor = None
    AutoTokenizer = None


@dataclass
@ProcessorStepRegistry.register(name="tokenizer_processor")
class TokenizerProcessorStep(ObservationProcessorStep):
    """
    用于对自然语言任务描述进行分词的处理器步骤。

    该步骤从 `EnvTransition` 的 `complementary_data` 中提取任务字符串，
    使用 Hugging Face `transformers` 分词器对其进行分词，并将生成的
    token ID 和注意力掩码添加到 `observation` 字典中。

    需要安装 `transformers` 库。

    属性:
        tokenizer_name: 来自 Hugging Face Hub 的预训练分词器名称（例如 "bert-base-uncased"）。
        tokenizer: 预先初始化好的分词器对象。如果提供，则忽略 `tokenizer_name`。
        max_length: 序列填充（pad）或截断（truncate）到的最大长度。
        task_key: `complementary_data` 中存储任务字符串的键。
        padding_side: 填充的一侧（'left' 或 'right'）。
        padding: 填充策略（'max_length'、'longest' 等）。
        truncation: 是否截断长度超过 `max_length` 的序列。
        input_tokenizer: 内部分词器实例，在初始化期间加载。
    """

    tokenizer_name: str | None = None
    tokenizer: Any | None = None  # 使用 `Any`，以便在没有硬依赖的情况下保持兼容
    max_length: int = 512
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    # 内部分词器实例（不属于配置的一部分）
    input_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        在数据类（dataclass）创建完成后初始化分词器。

        它会检查 `transformers` 库是否可用，并从提供的对象加载分词器，
        或者根据名称从 Hugging Face Hub 加载分词器。

        异常:
            ImportError: 如果未安装 `transformers` 库。
            ValueError: 如果既没有提供 `tokenizer`，也没有提供 `tokenizer_name`。
        """
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use TokenizerProcessorStep."
            )

        if self.tokenizer is not None:
            # 直接使用提供的分词器对象
            self.input_tokenizer = self.tokenizer
        elif self.tokenizer_name is not None:
            if AutoTokenizer is None:
                raise ImportError("AutoTokenizer is not available")
            self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            raise ValueError(
                "Either 'tokenizer' or 'tokenizer_name' must be provided. "
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            )

    def get_task(self, transition: EnvTransition) -> list[str] | None:
        """
        从转移数据的 complementary data 中提取任务描述。

        参数:
            transition: 环境转移数据。

        返回:
            任务字符串列表；如果未找到任务键或其值为 None，则返回 None。
        """
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is None so no task can be extracted from it")

        task = complementary_data[self.task_key]
        if task is None:
            raise ValueError("Task extracted from Complementary data is None")

        # 统一转换为字符串列表，供分词器使用
        if isinstance(task, str):
            return [task]
        elif isinstance(task, list | tuple) and all(isinstance(t, str) for t in task):
            return list(task)

        return None

    def get_subtask(self, transition: EnvTransition) -> list[str] | None:
        """
        从转移数据的 complementary data 中提取子任务（subtask）。

        参数:
            transition: 环境转移数据。

        返回:
            子任务字符串列表；如果未找到子任务键或其值为 None，则返回 None。
        """
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            return None

        subtask = complementary_data.get("subtask")
        if subtask is None:
            return None

        # 统一转换为字符串列表，供分词器使用
        if isinstance(subtask, str):
            return [subtask]
        elif isinstance(subtask, list) and all(isinstance(t, str) for t in subtask):
            return subtask

        return None

    def observation(self, observation: RobotObservation) -> RobotObservation:
        """
        对任务描述进行分词，并将其添加到观测字典中。

        该方法获取任务、对其进行分词，将生成的张量移动到与转移数据中
        其他数据相同的设备上，然后更新观测字典。

        参数:
            observation: 原始观测字典。

        返回:
            更新后的观测字典，包含 token ID 和注意力掩码。
        """
        task = self.get_task(self.transition)
        if task is None:
            raise ValueError("Task cannot be None")

        # 对任务进行分词（此时会创建 CPU 张量）
        tokenized_prompt = self._tokenize_text(task)

        # 从转移数据中已有的张量检测设备，以确保一致性
        target_device = self._detect_device(self.transition)

        # 将新分词得到的张量移动到检测到的设备上
        if target_device is not None:
            tokenized_prompt = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_prompt.items()
            }

        # 创建新的观测字典，避免原地修改原始字典
        new_observation = dict(observation)

        # 将分词后的数据添加到观测中
        new_observation[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)

        # 如果存在子任务，则对其进行分词
        subtask = self.get_subtask(self.transition)
        if subtask is not None:
            tokenized_subtask = self._tokenize_text(subtask)

            # 将新分词得到的张量移动到检测到的设备上
            if target_device is not None:
                tokenized_subtask = {
                    k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                    for k, v in tokenized_subtask.items()
                }

            # 将分词后的子任务添加到观测中
            new_observation[OBS_LANGUAGE_SUBTASK_TOKENS] = tokenized_subtask["input_ids"]
            new_observation[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = tokenized_subtask["attention_mask"].to(
                dtype=torch.bool
            )

        return new_observation

    def _detect_device(self, transition: EnvTransition) -> torch.device | None:
        """
        从转移数据中已有的张量检测 torch.device。

        它先检查观测字典中的张量，然后再检查动作（action）张量。

        参数:
            transition: 环境转移数据。

        返回:
            检测到的 `torch.device`；如果未找到任何张量，则返回 None。
        """
        # 先检查观测中的张量（最有可能找到张量的地方）
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation:
            for value in observation.values():
                if isinstance(value, torch.Tensor):
                    return value.device

        # 回退到检查动作张量
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            return action.device

        return None  # 未找到张量，默认将使用 CPU

    def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        """
        对分词器调用的一层封装。

        参数:
            text: 待分词的字符串或字符串列表。

        返回:
            一个字典，包含以 PyTorch 张量形式给出的分词结果 'input_ids' 和 'attention_mask'。
        """
        return self.input_tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )

    def get_config(self) -> dict[str, Any]:
        """
        返回处理器的可序列化配置。

        注意：分词器对象本身不会被序列化。如果处理器是使用分词器名称初始化的，
        则该名称会包含在配置中。

        返回:
            包含处理器配置参数的字典。
        """
        config = {
            "max_length": self.max_length,
            "task_key": self.task_key,
            "padding_side": self.padding_side,
            "padding": self.padding,
            "truncation": self.truncation,
        }

        # 仅当 tokenizer_name 曾被用于创建分词器时才保存它
        if self.tokenizer_name is not None and self.tokenizer is None:
            config["tokenizer_name"] = self.tokenizer_name

        return config

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        """保存分词器，以便通过对象传入的实例在重新加载时不会被覆盖。"""
        artifact_path = Path("tokenizer")
        save_pretrained = getattr(self.input_tokenizer, "save_pretrained", None)
        if save_pretrained is None:
            raise TypeError("Tokenizer must implement save_pretrained() to save a portable pipeline.")
        save_pretrained(save_directory / artifact_path)
        return {"tokenizer_name": artifact_path.as_posix()}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        为语言 token 和注意力掩码添加特征定义。

        该方法会更新策略特征字典，使其包含添加到观测中的新数据，
        从而确保下游组件知晓它们的形状和类型。

        参数:
            features: 现有的策略特征字典。

        返回:
            更新后的策略特征字典。
        """
        # 如果尚不存在 token ID 对应的特征，则添加一个
        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        # 如果尚不存在注意力掩码对应的特征，则添加一个
        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        return features


@dataclass
@ProcessorStepRegistry.register(name="action_tokenizer_processor")
class ActionTokenizerProcessorStep(ActionProcessorStep):
    """
    使用快速动作分词器（fast action tokenizer）对动作数据进行分词的处理器步骤。

    该步骤从 `EnvTransition` 中获取动作张量，使用 Hugging Face `transformers` 的
    AutoProcessor（例如 Physical Intelligence 的 "fast" 分词器）对其进行分词，
    并返回分词后的动作。

    需要安装 `transformers` 库。

    属性:
        tokenizer_name: 来自 Hugging Face Hub 的预训练处理器名称（例如 "lerobot/fast-action-tokenizer"）。
        tokenizer: 预先初始化好的处理器/分词器对象。如果提供，则忽略 `tokenizer_name`。
        trust_remote_code: 加载分词器时是否信任远程代码（某些分词器需要）。
        action_tokenizer: 内部分词器/处理器实例，在初始化期间加载。
        paligemma_tokenizer_name: 来自 Hugging Face Hub 的预训练 PaliGemma 分词器名称（例如 "google/paligemma-3b-pt-224"）。
    """

    action_tokenizer_name: str | None = None
    action_tokenizer_input_object: Any | None = None
    trust_remote_code: bool = True
    max_action_tokens: int = 256
    fast_skip_tokens: int = 128
    paligemma_tokenizer_name: str = "google/paligemma-3b-pt-224"
    allow_truncation: bool = True
    # 内部分词器实例（不属于配置的一部分）
    action_tokenizer: Any = field(default=None, init=False, repr=False)
    _paligemma_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        在数据类（dataclass）创建完成后初始化动作分词器。

        它会检查 `transformers` 库是否可用，并从提供的对象加载分词器，
        或者根据名称从 Hugging Face Hub 加载分词器。

        异常:
            ImportError: 如果未安装 `transformers` 库。
            ValueError: 如果既没有提供 `tokenizer`，也没有提供 `tokenizer_name`。
        """
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use ActionTokenizerProcessorStep."
            )

        if self.action_tokenizer_input_object is not None:
            self.action_tokenizer = self.action_tokenizer_input_object

        elif self.action_tokenizer_name is not None:
            if AutoProcessor is None:
                raise ImportError("AutoProcessor is not available")
            self.action_tokenizer = AutoProcessor.from_pretrained(
                self.action_tokenizer_name, trust_remote_code=self.trust_remote_code
            )
        else:
            raise ValueError(
                "Either 'action_tokenizer' or 'action_tokenizer_name' must be provided. "
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            )

        self._paligemma_tokenizer = AutoTokenizer.from_pretrained(
            self.paligemma_tokenizer_name,
            trust_remote_code=self.trust_remote_code,
            add_eos_token=True,
            add_bos_token=False,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        对转移数据执行动作分词。

        此方法重写了基类方法，以同时处理 token 和掩码。

        参数:
            transition: 包含动作数据的输入转移数据。

        返回:
            处理后的转移数据，其 complementary data 中包含分词后的动作和掩码。
        """
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            # 在推理期间没有可用的动作，跳过分词
            return new_transition

        # 进行分词，并获取完整格式化序列以及离散动作编码（code）各自的掩码。
        tokens, mask, code_mask = self._tokenize_action(action)

        # 将掩码存入 complementary data
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        if complementary_data is None:
            complementary_data = {}
        complementary_data[ACTION_TOKEN_MASK] = mask
        complementary_data[ACTION_CODE_TOKEN_MASK] = code_mask
        complementary_data[ACTION_TOKENS] = tokens
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def _act_tokens_to_paligemma_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        将动作 token 转换为 PaliGemma token。
        """
        return self._paligemma_tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens

    def _tokenize_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对动作张量进行分词并创建掩码。

        参数:
            action: 待分词的输入动作张量。形状: (B, H, action_dim) 或 (H, action_dim,)

        返回:
            一个 (tokens, mask) 元组，其中:
            - tokens: token ID 张量，形状为 (B, max_action_tokens)
            - mask: 布尔掩码，形状为 (B, max_action_tokens)，真实 token 为 True，填充部分为 False
        """
        if action is None:
            raise ValueError("Action cannot be None")

        # 获取输入动作的设备和数据类型
        device = action.device if isinstance(action, torch.Tensor) else None

        # 处理单个样本（添加批次维度）
        single_sample = action.dim() == 1
        if single_sample:
            action = action.unsqueeze(0)

        batch_size = action.shape[0]

        # 对动作批次进行分词
        # 快速分词器接收动作数据并返回 token ID
        tokens_list = []
        masks_list = []
        code_masks_list = []

        for i in range(batch_size):
            # 对单个动作进行分词（先移至 CPU，因为分词器使用需要 numpy 的 scipy）
            action_cpu = action[i : i + 1].cpu()
            tokens = self.action_tokenizer(action_cpu)

            # 如果是列表，则转换为张量
            if isinstance(tokens, list) or not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device=action.device)
            else:
                # 将 token 移回与输入动作相同的设备
                tokens = tokens.to(device=action.device)

            # 必要时展平为一维
            if tokens.dim() > 1:
                tokens = tokens.flatten()

            action_code_tokens = self._act_tokens_to_paligemma_tokens(tokens)
            bos_id = self._paligemma_tokenizer.bos_token_id
            prompt_tokens = torch.tensor(
                self._paligemma_tokenizer.encode("Action: ", add_special_tokens=False),
                device=action.device,
            )
            end_tokens = torch.tensor(self._paligemma_tokenizer.encode("|"), device=action.device)

            code_start = 1 + len(prompt_tokens)
            code_end = code_start + len(action_code_tokens)
            tokens = torch.cat(
                [
                    torch.tensor([bos_id], device=action.device),
                    prompt_tokens,
                    action_code_tokens,
                    end_tokens,
                ]
            )
            code_mask = torch.zeros(len(tokens), dtype=torch.bool, device=action.device)
            code_mask[code_start:code_end] = True

            # 截断或填充到 max_action_tokens
            if len(tokens) > self.max_action_tokens:
                if not self.allow_truncation:
                    raise ValueError(
                        f"FAST action sequence has {len(tokens)} tokens, exceeding "
                        f"max_action_tokens={self.max_action_tokens}."
                    )
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self.max_action_tokens}), truncating. "
                    "Consider increasing the `max_action_tokens` in your model config if this happens frequently."
                )
                tokens = tokens[: self.max_action_tokens]
                code_mask = code_mask[: self.max_action_tokens]
                mask = torch.ones(self.max_action_tokens, dtype=torch.bool, device=action.device)
            else:
                pad_len = self.max_action_tokens - len(tokens)
                mask = torch.cat(
                    [
                        torch.ones(len(tokens), dtype=torch.bool, device=action.device),
                        torch.zeros(pad_len, dtype=torch.bool, device=action.device),
                    ]
                )
                code_mask = torch.nn.functional.pad(code_mask, (0, pad_len), value=False)
                # 用零填充 token
                tokens = torch.nn.functional.pad(tokens, (0, pad_len), value=0)

            tokens_list.append(tokens)
            masks_list.append(mask)
            code_masks_list.append(code_mask)

        # 堆叠成批次张量
        tokens_batch = torch.stack(tokens_list, dim=0)  # (B, max_action_tokens)
        masks_batch = torch.stack(masks_list, dim=0)  # (B, max_action_tokens)
        code_masks_batch = torch.stack(code_masks_list, dim=0)  # (B, max_action_tokens)

        # 如果输入是单个样本，则移除批次维度
        if single_sample:
            tokens_batch = tokens_batch.squeeze(0)
            masks_batch = masks_batch.squeeze(0)
            code_masks_batch = code_masks_batch.squeeze(0)

        # 移动到与输入相同的设备
        if device is not None:
            tokens_batch = tokens_batch.to(device)
            masks_batch = masks_batch.to(device)
            code_masks_batch = code_masks_batch.to(device)

        return tokens_batch, masks_batch, code_masks_batch

    def action(self, action: torch.Tensor) -> torch.Tensor:
        """
        由于我们重写了 __call__，此方法不会被使用。
        它是 ActionProcessorStep 抽象基类（ABC）所要求的。
        """
        tokens, _, _ = self._tokenize_action(action)
        return tokens

    def get_config(self) -> dict[str, Any]:
        """
        返回处理器的可序列化配置。

        注意：分词器对象本身不会被序列化。如果处理器是使用分词器名称初始化的，
        则该名称会包含在配置中。

        返回:
            包含处理器配置参数的字典。
        """
        config = {
            "trust_remote_code": self.trust_remote_code,
            "max_action_tokens": self.max_action_tokens,
            "fast_skip_tokens": self.fast_skip_tokens,
            "paligemma_tokenizer_name": self.paligemma_tokenizer_name,
            "allow_truncation": self.allow_truncation,
        }

        # 仅当 tokenizer_name 曾被用于创建分词器时才保存它
        if self.action_tokenizer_name is not None and self.action_tokenizer_input_object is None:
            config["action_tokenizer_name"] = self.action_tokenizer_name

        return config

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        artifact_path = Path("action_tokenizer")
        save_pretrained = getattr(self.action_tokenizer, "save_pretrained", None)
        if save_pretrained is None:
            raise TypeError("Action tokenizer must implement save_pretrained() to save a portable pipeline.")
        save_pretrained(save_directory / artifact_path)
        return {"action_tokenizer_name": artifact_path.as_posix()}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        更新特征定义以反映分词后的动作。

        该方法会更新策略特征字典，以表明动作已被分词为
        形状为 (max_action_tokens,) 的 token ID 序列。

        参数:
            features: 现有的策略特征字典。

        返回:
            更新后的策略特征字典。
        """
        return features
