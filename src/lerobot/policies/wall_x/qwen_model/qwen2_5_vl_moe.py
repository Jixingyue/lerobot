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

"""在 transformers 原生 Qwen2.5-VL 模型之上的 Wall-X 混合专家（Mixture-of-Experts）扩展。

它基于原生的 ``transformers.models.qwen2_5_vl`` 类进行重构，只保留 Wall-X 真正新增的内容：

- ``BlockSparseMLP`` / ``SparseMoeBlock``：硬路由（按 token 类型索引）的专家 MLP。
- ``Qwen2_5_VLDecoderLayer_with_MoE``：原生 decoder 层，其 MLP 被替换为稀疏 MoE
  块，且 forward 会把激活值转换为参数 dtype（Wall-X 将 layernorm 保持在 float32，
  而投影层在 bfloat16 下运行，参见 ``to_bfloat16_for_selected_params``）。
- ``Qwen2_5_VLMoEModel``：带有 MoE decoder 层的原生文本模型，以及一个感知
  ``moe_token_types`` 的因果掩码覆盖（类型为 1 的 token——即动作 token——彼此之间
  进行双向注意力，其他所有 token 仍保持因果注意力）。
- ``Qwen2_5_VLACausalLMOutputWithPast``：带有 Wall-X 额外损失字段的输出 dataclass。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.activations import ACT2FN
    from transformers.cache_utils import Cache, DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLDecoderLayer,
        Qwen2_5_VLTextModel,
    )
    from transformers.utils.generic import merge_with_config_defaults
    from transformers.utils.output_capturing import capture_outputs
else:
    ACT2FN = None
    Cache = None
    DynamicCache = None
    create_causal_mask = None
    create_sliding_window_causal_mask = None
    BaseModelOutputWithPast = object
    ModelOutput = object
    Qwen2_5_VLDecoderLayer = nn.Module
    Qwen2_5_VLTextModel = nn.Module

    def merge_with_config_defaults(func):
        return func

    def capture_outputs(func):
        return func


from .configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig


@dataclass
class Qwen2_5_VLACausalLMOutputWithPast(ModelOutput):  # noqa: N801
    loss: torch.FloatTensor | None = None
    flow_loss: torch.FloatTensor | None = None
    cross_entropy_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None

    channel_loss_dict: dict[torch.FloatTensor] | None = None
    channel_loss_count_dict: dict[torch.FloatTensor] | None = None


class BlockSparseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.hidden_act = config["hidden_act"]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[self.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class SparseMoeBlock(nn.Module):
    def __init__(self, config, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([BlockSparseMLP(config.experts[i]) for i in range(num_experts)])

        if not hasattr(config, "dim_inputs") or not config.dim_inputs:
            raise ValueError("Config must contain valid dim_inputs")

        self.dim_inputs = config.dim_inputs

    def forward(self, hidden_states: torch.Tensor, experts_indices: torch.Tensor) -> torch.Tensor:
        """
        将不同的 hidden_states 路由到相应的专家进行处理。

        Args:
            hidden_states (torch.Tensor): 形状为 (batch_size, seq_length, hidden_dim) 的张量。
            experts_indices (torch.Tensor): 形状为 (batch_size, seq_length) 的张量，
                表示分配给每个 token 的专家索引。

        Returns:
            output (torch.Tensor): 形状为 (batch_size, seq_length, hidden_dim) 的张量。
        """
        batch_size, seq_length, hidden_dim = hidden_states.size()
        output = torch.zeros_like(hidden_states)

        for expert_idx, expert in enumerate(self.experts):
            mask = experts_indices == expert_idx
            if mask.sum() == 0:
                continue
            dim_input = self.dim_inputs[expert_idx]

            selected_hidden = hidden_states[mask]
            processed_hidden = expert(selected_hidden[:, :dim_input])

            batch_indices, seq_indices = torch.where(mask)
            output[batch_indices, seq_indices, :dim_input] = processed_hidden

        return output


class Qwen2_5_VLDecoderLayer_with_MoE(Qwen2_5_VLDecoderLayer):  # noqa: N801
    """带可选硬路由稀疏 MoE MLP 的原生 Qwen2.5-VL decoder 层。

    与原生层 forward 的不同之处：
    - 当设置了 ``config.mlp_moe`` 时，将注意力后的隐藏状态以 ``token_types`` 为键
      路由通过 ``SparseMoeBlock``；
    - 在每个 block 之前将激活值转换为参数 dtype，因为 Wall-X 在同一模块中使用
      float32 的 layernorm 和 bfloat16 的投影层。
    """

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int, num_experts: int):
        super().__init__(config, layer_idx)
        if config.mlp_moe:
            del self.mlp
            self.mlp = None
            self.moe = SparseMoeBlock(config, num_experts=num_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        token_types: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = hidden_states.to(self.input_layernorm.weight.dtype)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states.to(self.self_attn.q_proj.weight.dtype)

        # 自注意力（Self Attention）
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # 全连接层（Fully Connected）
        residual = hidden_states
        hidden_states = hidden_states.to(self.post_attention_layernorm.weight.dtype)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.mlp is None:  # 使用 moe mlp
            hidden_states = hidden_states.to(self.moe.experts[0].down_proj.weight.dtype)
            hidden_states = self.moe(hidden_states, token_types)
        else:
            hidden_states = hidden_states.to(self.mlp.down_proj.weight.dtype)
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2_5_VLMoEModel(Qwen2_5_VLTextModel):  # noqa: N801
    """带混合专家（MoE）decoder 层的 Qwen2.5-VL 文本模型。

    在原生 ``Qwen2_5_VLTextModel`` 之上扩展了按 token 类型的专家路由，以及一个因果掩码
    覆盖：让动作 token 块（``moe_token_types == 1``）内部彼此进行双向注意力，
    同时其他所有位置仍保持因果注意力。
    """

    config_class = Qwen2_5_VLTextConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer_with_MoE"]

    def __init__(self, config: Qwen2_5_VLConfig | Qwen2_5_VLTextConfig):
        text_config = config.text_config if isinstance(config, Qwen2_5_VLConfig) else config
        self._require_eager_attention(text_config._attn_implementation)
        # 当未指定实现时，Transformers 会自动选择 SDPA。Wall-X 的动作 token 岛
        # 需要显式的 4D 掩码，因此在 PreTrainedModel 做出该选择之前先指定 eager。
        text_config._attn_implementation = "eager"
        super().__init__(text_config)
        # 在替换父类分配的稠密层之前先释放它们（pi_gemma.py 的先例）。
        del self.layers
        self.layers = nn.ModuleList(
            [
                Qwen2_5_VLDecoderLayer_with_MoE(text_config, layer_idx, text_config.num_experts)
                for layer_idx in range(text_config.num_hidden_layers)
            ]
        )
        # 初始化权重并执行最终处理
        self.post_init()

    @staticmethod
    def _require_eager_attention(attn_implementation: str | None) -> None:
        if attn_implementation not in {None, "eager"}:
            raise ValueError(
                "Wall-X currently supports only attn_implementation='eager'. "
                "Its bidirectional action-token islands cannot be represented "
                f"correctly by {attn_implementation!r}."
            )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    @merge_with_config_defaults
    # ``capture_outputs`` 会从 kwargs 中读取 output_hidden_states/output_attentions，
    # 并通过 decoder 层和注意力模块上的 hook 来填充 BaseModelOutputWithPast。
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        moe_token_types: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        self._require_eager_attention(self.config._attn_implementation)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if moe_token_types is None:
            raise ValueError("moe_token_types must be provided for MoE routing")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if moe_token_types.shape[-1] < inputs_embeds.shape[1]:
            raise ValueError(
                "moe_token_types must cover every input token; got "
                f"{moe_token_types.shape[-1]} types for {inputs_embeds.shape[1]} tokens"
            )
        routing_token_types = moe_token_types[:, -inputs_embeds.shape[1] :]

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # 原生 Qwen 使用第四个仅用于文本的位置 ID 行来描述 packed 序列。
        # 保留三行多模态行用于 RoPE，并把文本行同时传给掩码和 decoder 注意力。
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        full_token_types = self._prepare_action_token_types(
            moe_token_types,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        action_island_mask = self._action_island_mask(full_token_types)
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
            "or_mask_function": action_island_mask,
        }
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                token_types=routing_token_types,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    @staticmethod
    def _action_island_mask(token_types: torch.Tensor):
        action_tokens = token_types == 1

        def action_island(batch_idx, head_idx, query_idx, key_idx):
            del head_idx
            return action_tokens[batch_idx, query_idx] & action_tokens[batch_idx, key_idx]

        return action_island

    @staticmethod
    def _prepare_action_token_types(
        token_types: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        """将当前步的 token 类型与绝对掩码索引对齐。

        生成时只为新的 query token 传入 token 类型，而原生掩码回调接收的是
        绝对的 query/key 索引。被缓存的 token 默认为类型 0；调用方也可以改为
        传入完整历史的 token 类型。
        """
        query_length = inputs_embeds.shape[1]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if isinstance(past_length, torch.Tensor):
            past_length = int(past_length.item())

        token_types = token_types.to(device=inputs_embeds.device)
        if token_types.shape[-1] == query_length and past_length:
            token_types = torch.nn.functional.pad(token_types, (past_length, 0))

        required_length = past_length + query_length
        if attention_mask is not None and attention_mask.ndim == 2:
            required_length = max(required_length, attention_mask.shape[-1])
        if past_key_values is not None:
            kv_length, kv_offset = past_key_values.get_mask_sizes(query_length, 0)
            if isinstance(kv_length, torch.Tensor):
                kv_length = int(kv_length.item())
            if isinstance(kv_offset, torch.Tensor):
                kv_offset = int(kv_offset.item())
            required_length = max(required_length, kv_length + kv_offset)

        if token_types.shape[-1] < required_length:
            token_types = torch.nn.functional.pad(token_types, (0, required_length - token_types.shape[-1]))
        return token_types
