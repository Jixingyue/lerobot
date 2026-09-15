# Copyright 2026 Shirui Chen, Cole Harrison, Ying-Chun Lee, Angela Jin Yang,
# Zhongzheng Ren, Lillian J. Ratliff, Jiafei Duan, Dieter Fox, Ranjay Krishna
# and The HuggingFace Inc. team. All rights reserved.
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

"""TOPReward：将 Token 概率作为机器人领域的隐式零样本奖励。

论文：         https://arxiv.org/abs/2602.19313
项目：         https://topreward.github.io/webpage/
原始代码：     https://github.com/TOPReward/TOPReward
骨干模型：     https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct  (默认)

TOPReward 是一个**零样本**奖励模型：它自身没有微调权重。
给定一条视频轨迹和一条任务指令，它会询问一个现成的
VLM：在以视频为条件的情况下该指令成立的可能性有多大，
并将该对数似然作为奖励信号返回。

推理流程：

1. 处理器构建一个聊天式提示词，对其进行分词，并输出
   ``input_ids``、``attention_mask``、视觉张量和 ``labels``。
   处理器用 ``-100`` 对除末尾回答 token 之外的所有
   label 进行掩码。
2. 将完整的 token 序列前向传入 VLM。
3. 从 logits 中读取末尾回答 token 的对数概率，作为
   标量奖励。

使用默认的 ``prompt_suffix_template`` 时，唯一未被掩码的 token 是
末尾的字面量 ``"True"``——奖励即
``log P("True" | video + prompt + instruction)``。

本 LeRobot 移植版**仅支持推理且不可训练**——:meth:`forward`
有意继承自 :class:`PreTrainedRewardModel` 并抛出
``NotImplementedError``，使 :attr:`PreTrainedRewardModel.is_trainable`
返回 ``False``。

由于 VLM 权重以规范 id（如 ``Qwen/Qwen3-VL-8B-Instruct``）托管在
Hugging Face Hub 上，且 TOPReward 从不修改它们，
因此重写了 :meth:`_save_pretrained` 和 :meth:`from_pretrained`，
使 TOPReward 的 LeRobot "检查点"仅为单个 ``config.json``
（VLM 在加载时从 Hub 重新获取）。
"""

from __future__ import annotations

import builtins
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError
from torch import Tensor
from torch.nn.functional import cross_entropy

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig
from lerobot.rewards.topreward.processor_topreward import TOPREWARD_FEATURE_PREFIX, TOPREWARD_INPUT_KEYS
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import Qwen3VLForConditionalGeneration
else:
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="TOPRewardModel")


def _torch_dtype(name: str) -> torch.dtype | str:
    """解析 torch dtype 名称；``"auto"`` 原样透传。"""
    if name == "auto":
        return "auto"
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unknown torch dtype: {name!r}")


class TOPRewardModel(PreTrainedRewardModel):
    """TOPReward 零样本奖励模型。"""

    name = "topreward"
    config_class = TOPRewardConfig

    def __init__(self, config: TOPRewardConfig) -> None:
        require_package("transformers", extra="topreward")
        super().__init__(config)
        self.config = config

        torch_dtype = _torch_dtype(config.torch_dtype)
        model_kwargs: dict[str, Any] = {"dtype": torch_dtype, "trust_remote_code": True}
        if config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = config.attn_implementation

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(config.vlm_name, **model_kwargs)

    def compute_reward(self, batch: dict[str, Any]) -> Tensor:
        """为批次中的每个样本返回一个 log-prob 奖励。"""
        inputs: dict[str, Any] = {}
        for key in TOPREWARD_INPUT_KEYS:
            batch_key = f"{TOPREWARD_FEATURE_PREFIX}{key}"
            if batch_key not in batch:
                raise KeyError(
                    f"TOPReward batch missing `{batch_key}`. Make sure the "
                    "TOPRewardEncoderProcessorStep ran before `compute_reward`."
                )
            inputs[key] = batch[batch_key]

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        labels = inputs.pop("labels")
        inputs["logits_to_keep"] = 2

        self.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        rewards = -cross_entropy(logits[:, -2, :].float(), labels[:, -1], reduction="none")
        if np.isfinite(self.config.success_threshold):
            rewards = (rewards > self.config.success_threshold).float()
        return rewards.to(self.config.device or "cpu")

    def _save_pretrained(self, save_directory: Path) -> None:
        """仅保存 ``config.json``。"""
        self.config._save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: RewardModelConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,  # noqa: ARG003 — 为保持 API 一致性而接受；未使用（没有可加载的 safetensors）
        **kwargs: Any,
    ) -> T:
        """加载 TOPReward 配置并实例化其包装的 VLM。"""
        if config is None:
            config = RewardModelConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        if not isinstance(config, TOPRewardConfig):
            raise TypeError(
                f"Expected a TOPRewardConfig, got {type(config).__name__}. Make sure "
                f"`pretrained_name_or_path={pretrained_name_or_path!r}` points at a "
                "TOPReward checkpoint."
            )

        model_id = str(pretrained_name_or_path)
        if not os.path.isdir(model_id):
            try:
                hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        instance = cls(config, **kwargs)
        instance.to(config.device)
        instance.eval()
        return instance
