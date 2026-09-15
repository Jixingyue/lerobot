# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from typing import TYPE_CHECKING

import torch
from torch import nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_layers import GradientCheckpointingLayer
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.gemma.modeling_gemma import (
        GemmaAttention,
        GemmaConfig,
        GemmaForCausalLM,
        GemmaMLP,
        GemmaModel,
    )
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaForConditionalGeneration,
        PaliGemmaModel,
    )
else:
    GemmaAttention = None
    GemmaConfig = None
    GemmaForCausalLM = None
    GemmaMLP = None
    GemmaModel = None
    PaliGemmaModel = None
    PaliGemmaForConditionalGeneration = None
    DynamicCache = None
    GradientCheckpointingLayer = None
    BaseModelOutputWithPast = None
    create_causal_mask = None


def _gated_residual(
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    gate: torch.Tensor | None,
) -> torch.Tensor | None:
    """门控残差：当 gate 为 None 时为 x + y，否则为 x + y * gate。"""
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


def layernorm_forward(
    layernorm: nn.Module,
    x: torch.Tensor,
    cond: torch.Tensor | None = None,
):
    """
    调用 layernorm 并返回隐藏状态和门控值
    如果 cond 不为 None，使用条件归一化
    否则使用普通的 gemma 归一化
    """
    if cond is not None:
        return layernorm(x, cond=cond)
    else:
        return layernorm(x)


class PiGemmaRMSNorm(nn.Module):
    """
    PI Gemma 的自适应 RMSNorm（AdaRMS）。
    当设置了 cond_dim 时，使用 cond 来调制缩放/偏移/门控；否则行为与标准 GemmaRMSNorm 相同。
    forward(x, cond=None) 返回 (output, gate)，供 _gated_residual 使用。

    ``cond`` 可以是 ``(batch_size, cond_dim)``——所有 token 共享一个调制——也可以是
    ``(batch_size, seq_len, cond_dim)``，这让缩放/偏移/门控可以按 token 不同。
    按 token 的形式是训练时 RTC 所需要的，用于以各自的流时间步标记干净的动作前缀
    （arXiv 2512.05964）；它不增加任何参数。
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x):
        # 以 float32 计算方差（与源实现一致）
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # 以 float32 计算归一化
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = x.dtype
        normed = self._norm(x)
        if cond is None or self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.type_as(x), None
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {cond.shape[-1]}")
        if cond.ndim not in (2, 3):
            raise ValueError(f"cond must be (B, cond_dim) or (B, T, cond_dim), got {tuple(cond.shape)}")
        if x.ndim == 3 and cond.ndim == 3 and cond.shape[1] != x.shape[1]:
            raise ValueError(
                f"Per-token cond has {cond.shape[1]} tokens but x has {x.shape[1]}; shapes must match."
            )
        modulation = self.dense(cond)
        if x.ndim == 3 and modulation.ndim == 2:
            # 每样本标量的 cond：添加 token 轴，使一个调制可以广播到
            # 所有 token。按 token 的 cond 已经带有该轴，不能再增加一个。
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)

    def extra_repr(self) -> str:
        if self.dense is not None:
            return f"dim={self.dim}, eps={self.eps}, adaptive=True, cond_dim={self.cond_dim}"
        return f"dim={self.dim}, eps={self.eps}"


def _get_pi_gemma_decoder_layer_base():
    """PiGemmaDecoderLayer 的基类"""

    class _PiGemmaDecoderLayerBase(GradientCheckpointingLayer):
        """使用 PiGemmaRMSNorm 和 _gated_residual 的解码器层，与 v5 Gemma 兼容。"""

        def __init__(self, config: GemmaConfig, layer_idx: int):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
            self.mlp = GemmaMLP(config)
            cond_dim = (
                getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
            )
            self.input_layernorm = PiGemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )
            self.post_attention_layernorm = PiGemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values=None,
            use_cache: bool = False,
            cache_position: torch.LongTensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            adarms_cond: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            residual = hidden_states
            hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)
            hidden_states, _ = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = _gated_residual(residual, hidden_states, gate)

            residual = hidden_states
            hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
            hidden_states = self.mlp(hidden_states)
            hidden_states = _gated_residual(residual, hidden_states, gate)
            return hidden_states

    return _PiGemmaDecoderLayerBase


class PiGemmaModel(GemmaModel):  # type: ignore[misc]
    """
    当 config.use_adarms 为 True 时，扩展了 AdaRMS（自适应 RMSNorm）和门控残差的 GemmaModel。
    """

    def __init__(self, config: GemmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        # 在替换之前释放父类分配的层/归一化，以避免约 2 倍的峰值内存。
        del self.layers
        del self.norm
        # if not getattr(config, "use_adarms", False):
        #     return
        cond_dim = getattr(config, "adarms_cond_dim", None)
        pi_gemma_decoder_layer_base = _get_pi_gemma_decoder_layer_base()
        self.layers = nn.ModuleList(
            [pi_gemma_decoder_layer_base(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PiGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        adarms_cond（形状为 `(batch_size, cond_dim)` 或
            `(batch_size, seq_len, cond_dim)` 的 `torch.Tensor`，*可选*）：
            ADARMS 的条件。按 token 的形式驱动训练时 RTC。
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            import logging

            logging.warning(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # 嵌入位置
        hidden_states = inputs_embeds
        # 如果第一层使用 bfloat16，则转换为 bfloat16
        if len(self.layers) > 0 and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        # 创建将在各解码器层之间共享的位置嵌入
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 归一化
        # Gemma 会将下面的值向下转换为 float16，导致 sqrt(3072)=55.4256 变成 55.5
        # 参见 https://github.com/huggingface/transformers/pull/29402

        # 解码器层
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
                **kwargs,
            )

            hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states, _ = self.norm(hidden_states, adarms_cond)

        # 添加来自最后一个解码器层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class PiGemmaForCausalLM(GemmaForCausalLM):  # type: ignore[misc]
    """
    使用 PiGemmaModel 作为骨干的因果 LM 包装器，与 GemmaForCausalLM
    以及 pi0_fast 中使用的语言模型保持一致。在 pi0/pi05 中将其用作动作专家。
    """

    def __init__(self, config: GemmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        del self.model
        self.model = PiGemmaModel(config)


class PaliGemmaModelWithPiGemma(PaliGemmaModel):
    """language_model 为 PiGemmaModel 的 PaliGemmaModel（使用 PiGemmaRMSNorm 和门控残差的自定义解码器）。"""

    def __init__(self, config):
        super().__init__(config)
        del self.language_model
        self.language_model = PiGemmaModel(config.text_config)


class PaliGemmaForConditionalGenerationWithPiGemma(PaliGemmaForConditionalGeneration):
    """语言模型使用 PiGemma 解码器的 PaliGemmaForConditionalGeneration。"""

    def __init__(self, config):
        super().__init__(config)
        del self.model
        self.model = PaliGemmaModelWithPiGemma(config)

    # 通过条件类公开模块以保持向后兼容
    @property
    def language_model(self):
        return self.model.language_model


__all__ = [
    "PiGemmaModel",
    "PiGemmaForCausalLM",
    "PiGemmaRMSNorm",
    "_gated_residual",
    "layernorm_forward",
    "PaliGemmaModelWithPiGemma",
    "PaliGemmaForConditionalGenerationWithPiGemma",
]
