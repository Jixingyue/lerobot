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

"""来自 MEM（arXiv:2603.03596）的短时域视觉记忆与本体感知记忆。"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

# MEM 从 `SiglipEncoderLayer` 及其 `SiglipAttention` 上读取的属性。MEM
# 重新实现了注意力子块，以组合空间注意力和时间注意力的权重
# （MEM 附录 C，公式 3），因此它依赖这些内部实现，
# 而不是依赖 `SiglipAttention.forward`。逐层进行校验，
# 这样 transformers 版本升级时会明确报错，而不是悄无声息地改变架构。
_REQUIRED_LAYER_ATTRS = ("layer_norm1", "self_attn", "layer_norm2", "mlp")
_REQUIRED_ATTENTION_ATTRS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "num_heads",
    "head_dim",
    "scale",
    "dropout",
)


def _hidden_tensor(output, *, source: str) -> Tensor:
    """对受支持的 Transformers 层输出进行归一化，并在 API 变动时给出明确的失败信息。"""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError(
        f"{source} must return a Tensor or a tuple whose first item is a Tensor, got {type(output).__name__}"
    )


def _validate_siglip_layer(layer, *, layer_index: int) -> None:
    """校验 MEM 所使用的 SigLIP 编码器层的私有约定。"""
    missing = [name for name in _REQUIRED_LAYER_ATTRS if not hasattr(layer, name)]
    if missing:
        raise TypeError(
            "MEM visual memory requires SigLIP encoder layers exposing "
            f"{_REQUIRED_LAYER_ATTRS}; layer {layer_index} is missing {tuple(missing)}"
        )
    missing = [name for name in _REQUIRED_ATTENTION_ATTRS if not hasattr(layer.self_attn, name)]
    if missing:
        raise TypeError(
            "MEM visual memory requires SigLIP self-attention exposing "
            f"{_REQUIRED_ATTENTION_ATTRS}; layer {layer_index} is missing {tuple(missing)}"
        )


def sample_observation_history(
    history: list[Tensor], *, num_frames: int, stride: int, steps_seen: int
) -> tuple[Tensor, Tensor]:
    """对同质批次的推理队列进行子采样，并标记回合开始前的填充。

    帧以相对于 ``history[-1]``（最新观测）的年龄来寻址，
    因此无论队列长度如何，当前帧始终是最后采样的帧。

    ``steps_seen`` 适用于每个批次行。Pi05 在每个批量化 rollout 边界处
    清空队列；不支持独立地重置向量中的各行。
    """
    # 年龄降序排列，因此返回的帧顺序为 最旧 -> 当前。
    required_ages = list(range((num_frames - 1) * stride, -1, -stride))
    if len(history) <= required_ages[0]:
        raise ValueError(
            f"observation history holds {len(history)} frames, need at least "
            f"{required_ages[0] + 1} to sample {num_frames} frames at stride {stride}"
        )
    values = torch.stack([history[-1 - age] for age in required_ages], dim=1)
    valid = torch.tensor([steps_seen > age for age in required_ages], dtype=torch.bool, device=values.device)
    padding_mask = (~valid)[None, :].expand(values.shape[0], -1)
    return values, padding_mask


def temporal_sinusoidal_embedding(
    num_frames: int, hidden_size: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """返回固定的时间嵌入，其中当前位置恰好为零。"""
    if hidden_size % 2:
        raise ValueError(f"hidden_size must be even, got {hidden_size}")
    positions = torch.arange(1 - num_frames, 1, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, hidden_size, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / hidden_size)
    )[None, :]
    angles = positions * frequencies
    embedding = torch.zeros(num_frames, hidden_size, device=device, dtype=torch.float32)
    embedding[:, 0::2] = torch.sin(angles)
    embedding[:, 1::2] = torch.cos(angles) - 1.0
    return embedding.to(dtype=dtype)


def causal_temporal_mask(frame_mask: Tensor, *, dtype: torch.dtype, num_patches: int) -> Tensor:
    """构建用于时间注意力的加性因果掩码与 key 填充掩码。"""
    if frame_mask.ndim != 2:
        raise ValueError(f"frame_mask must have shape (batch, frames), got {tuple(frame_mask.shape)}")
    batch_size, num_frames = frame_mask.shape
    allowed = torch.ones(num_frames, num_frames, dtype=torch.bool, device=frame_mask.device).tril()
    allowed = allowed[None] & frame_mask[:, None, :].bool()
    # 保持填充的 query 行在数值上安全；有效的 query 仍然无法看到填充的 key。
    allowed |= torch.eye(num_frames, dtype=torch.bool, device=frame_mask.device)[None]
    mask = torch.zeros(batch_size, 1, num_frames, num_frames, dtype=dtype, device=frame_mask.device)
    mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
    return mask.repeat_interleave(num_patches, dim=0)


def space_time_attention(attention, hidden_states: Tensor, temporal_mask: Tensor) -> Tensor:
    """对 ``(B,T,P,D)`` 应用 MEM 的组合式时空注意力（附录 C，公式 3）。

    ``hidden_states`` 必须已经携带时间位置嵌入，这样 ViT 预训练的
    同一组 q/k/v 投影就能同时服务于两个阶段 —— MEM 不会给视觉塔
    增加任何可学习参数。

    公式 3 将两个注意力算子组合为 ``alpha_spatial[alpha_temporal[z]]``，
    然后"遵循 transformer 层的标准计算"。组合权重而非堆叠两个注意力
    子块有双重好处：``out_proj`` 只应用一次，且对单个时间步做 softmax
    是恒等映射，因此 ``T == 1`` 时按构造退化为原版 SigLIP 空间注意力。
    应用该组合等价于以时间混合后的 value 做空间注意力，这使两个阶段
    都能使用 SDPA —— 注意力矩阵始终不会被实例化。
    """
    batch_size, num_frames, num_patches, _ = hidden_states.shape
    heads, head_dim = attention.num_heads, attention.head_dim
    projected_shape = (batch_size, num_frames, num_patches, heads, head_dim)
    queries = attention.q_proj(hidden_states).view(projected_shape)
    keys = attention.k_proj(hidden_states).view(projected_shape)
    values = attention.v_proj(hidden_states).view(projected_shape)
    dropout = attention.dropout if attention.training else 0.0

    def as_temporal(tensor: Tensor) -> Tensor:
        # (B,T,P,h,d) -> (B*P,h,T,d)：每个 patch 一个序列，跨帧排列。
        return tensor.permute(0, 2, 3, 1, 4).reshape(batch_size * num_patches, heads, num_frames, head_dim)

    def as_spatial(tensor: Tensor) -> Tensor:
        # (B,T,P,h,d) -> (B*T,h,P,d)：每帧一个序列，跨 patch 排列。
        return tensor.permute(0, 1, 3, 2, 4).reshape(batch_size * num_frames, heads, num_patches, head_dim)

    temporal = F.scaled_dot_product_attention(
        as_temporal(queries),
        as_temporal(keys),
        as_temporal(values),
        attn_mask=temporal_mask,
        dropout_p=dropout,
        scale=attention.scale,
    )
    temporal = temporal.reshape(batch_size, num_patches, heads, num_frames, head_dim).permute(0, 3, 1, 2, 4)

    attended = F.scaled_dot_product_attention(
        as_spatial(queries),
        as_spatial(keys),
        as_spatial(temporal),
        dropout_p=dropout,
        scale=attention.scale,
    )
    attended = attended.reshape(batch_size, num_frames, heads, num_patches, head_dim).permute(0, 1, 3, 2, 4)
    return attention.out_proj(attended.reshape(batch_size, num_frames, num_patches, heads * head_dim))


def encode_video_with_mem(
    vision_model,
    pixel_values: Tensor,
    frame_mask: Tensor,
    *,
    temporal_attention_every: int,
) -> Tensor:
    """使用 MEM 时空可分离注意力对 ``(B,T,C,H,W)`` 进行编码。

    每隔 N 层将其注意力子块替换为 :func:`space_time_attention` 中的
    组合式时空注意力，复用预训练的 SigLIP 投影。一旦最后一个这样的层
    执行完毕，过去帧的 token 就会被丢弃 —— 其上方不可能再发生跨帧混合 ——
    因此其余层和下游的 VLM 前缀看到的恰好是单帧的 token 数量。

    时间层直接调用 SDPA，因此会忽略视觉塔配置的注意力实现；
    仅含空间注意力的层仍然使用配置的实现。
    """
    if pixel_values.ndim != 5:
        raise ValueError(f"pixel_values must have shape (B,T,C,H,W), got {tuple(pixel_values.shape)}")
    if temporal_attention_every < 1:
        raise ValueError("temporal_attention_every must be at least 1")

    batch_size, num_frames, channels, height, width = pixel_values.shape
    if frame_mask.shape != (batch_size, num_frames):
        raise ValueError(
            f"frame_mask must have shape {(batch_size, num_frames)}, got {tuple(frame_mask.shape)}"
        )
    if any(not hasattr(vision_model, name) for name in ("embeddings", "encoder", "post_layernorm")):
        raise TypeError("MEM visual memory requires a SigLIP-compatible vision transformer")
    if not hasattr(vision_model.encoder, "layers"):
        raise TypeError("MEM visual memory requires a SigLIP encoder exposing a layers collection")

    layers = vision_model.encoder.layers
    if temporal_attention_every > len(layers):
        raise ValueError(
            f"temporal_attention_every ({temporal_attention_every}) must not exceed the number of "
            f"SigLIP encoder layers ({len(layers)})"
        )
    temporal_layers = [i for i in range(len(layers)) if (i + 1) % temporal_attention_every == 0]
    last_temporal_index = temporal_layers[-1] if (temporal_layers and num_frames > 1) else -1

    flat_pixels = pixel_values.reshape(batch_size * num_frames, channels, height, width)
    hidden_states = vision_model.embeddings(flat_pixels)
    num_patches, hidden_size = hidden_states.shape[1:]
    hidden_states = hidden_states.reshape(batch_size, num_frames, num_patches, hidden_size)
    temporal_positions = temporal_sinusoidal_embedding(
        num_frames, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype
    )[None, :, None]
    temporal_mask = causal_temporal_mask(frame_mask, dtype=hidden_states.dtype, num_patches=num_patches)

    for layer_index, layer in enumerate(layers):
        _validate_siglip_layer(layer, layer_index=layer_index)
        active_frames = hidden_states.shape[1]
        if active_frames == 1 or (layer_index + 1) % temporal_attention_every:
            flat_hidden = hidden_states.reshape(batch_size * active_frames, num_patches, hidden_size)
            layer_output = _hidden_tensor(
                layer(flat_hidden, attention_mask=None),
                source=f"SigLIP encoder layer {layer_index}",
            )
            hidden_states = layer_output.reshape(batch_size, active_frames, num_patches, hidden_size)
            continue

        # 公式 1：两个阶段都从 z + e(t) 推导 q/k/v。残差保持为层的输入，
        # 这样位置信号不会在各层之间累积。
        attended = space_time_attention(
            layer.self_attn, layer.layer_norm1(hidden_states + temporal_positions), temporal_mask
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + layer.mlp(layer.layer_norm2(hidden_states))
        if layer_index == last_temporal_index:
            hidden_states = hidden_states[:, -1:]

    return vision_model.post_layernorm(hidden_states[:, -1])
