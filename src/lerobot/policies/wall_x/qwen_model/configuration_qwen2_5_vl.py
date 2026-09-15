#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""针对 Transformers 原生 Qwen2.5-VL 配置的 Wall-X 配置扩展。"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from huggingface_hub.dataclasses import strict

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig as TransformersQwen2_5_VLConfig,
        Qwen2_5_VLTextConfig as TransformersQwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )
else:

    @dataclass
    class _TransformersConfigFallback:
        """仅在 Transformers 不可用时使用的、导入安全的替代类。"""

    TransformersQwen2_5_VLConfig = _TransformersConfigFallback
    TransformersQwen2_5_VLTextConfig = _TransformersConfigFallback
    Qwen2_5_VLVisionConfig = None

# 0.6.0 之前的 Wall-X checkpoint 使用旧的、扁平的 Qwen2.5-VL 配置布局。原生的
# ``Qwen2_5_VLConfig`` 可以接受该布局，并会把文本模型字段移入其 ``text_config``
# 子配置中，因此这里只需要声明 Wall-X 特有的 MoE 字段。
_LEGACY_TEXT_ATTRIBUTES = {
    "attention_dropout",
    "attention_moe",
    "dim_inputs",
    "dof_config",
    "experts",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "max_window_layers",
    "mlp_moe",
    "noise_scheduler",
    "num_attention_heads",
    "num_experts",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "rms_norm_eps",
    "sliding_window",
    "use_cache",
    "use_sliding_window",
    "vocab_size",
}


@strict
class Qwen2_5_VLTextConfig(TransformersQwen2_5_VLTextConfig):  # noqa: N801
    """原生 Qwen2.5-VL 文本配置，加上 Wall-X 的硬路由 MoE 设置。"""

    num_experts: int = 4
    experts: list[dict] | None = None
    dof_config: dict | None = None
    noise_scheduler: dict | None = None
    dim_inputs: tuple[int, ...] | list[int] = (1536, 1536)
    attention_moe: bool = False
    mlp_moe: bool = False

    def __post_init__(self, **kwargs):
        self.dim_inputs = tuple(self.dim_inputs)
        super().__post_init__(**kwargs)


@strict
class Qwen2_5_VLConfig(TransformersQwen2_5_VLConfig):  # noqa: N801
    """带 Wall-X 文本子配置的原生复合 Qwen2.5-VL 配置。

    原生的复合加载器同时支持当前的嵌套配置和现有 ``wall-oss-flow``
    checkpoint 所使用的扁平布局。
    """

    sub_configs = {
        "vision_config": Qwen2_5_VLVisionConfig,
        "text_config": Qwen2_5_VLTextConfig,
    }

    def __getattr__(self, name):
        """保留对现归属于 ``text_config`` 的字段的旧式直接访问。

        Wall-X 过去使用扁平配置，并直接访问 ``hidden_size`` 和 ``num_experts``
        等字段。转发未知属性可以保留该 API，而无需重复原生配置。
        """
        text_config = self.__dict__.get("text_config")
        if name in _LEGACY_TEXT_ATTRIBUTES and text_config is not None and hasattr(text_config, name):
            return getattr(text_config, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
