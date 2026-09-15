#!/usr/bin/env python

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

import builtins
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import _scipy_available, _transformers_available, require_package

# 用于类型检查和延迟加载的条件导入
if TYPE_CHECKING or _scipy_available:
    from scipy.fftpack import idct
else:
    idct = None

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.models.auto import CONFIG_MAPPING

    from ..pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaModel,
    )
else:
    CONFIG_MAPPING = None
    AutoProcessor = None
    AutoTokenizer = None
    PiGemmaModel = None
    PaliGemmaForConditionalGenerationWithPiGemma = None

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from ..common.vla_utils import pad_vector, prepare_attention_masks_4d, resize_with_pad_torch
from ..pretrained import PreTrainedPolicy, T
from ..rtc.modeling_rtc import RTCProcessor
from .configuration_pi0_fast import PI0FastConfig


class ActionSelectKwargs(TypedDict, total=False):
    temperature: float | None


class GemmaConfig:  # 参见 openpi `gemma.py: Config`
    """Gemma 模型变体的配置。"""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # 参见 openpi `gemma.py: get_config`
    """返回指定 gemma 变体的配置。"""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PI0FastPaliGemma(nn.Module):
    """用于 PI0Fast 的 PaliGemma 模型"""

    def __init__(
        self,
        vlm_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)

        # 当 use_adarms[0] 为 True 时，使用 PI Gemma（AdaRMS）作为语言模型，
        # 以支持 forward(..., adarms_cond=...)（与 pi0/pi05 相同）。
        if use_adarms[0]:
            text_config = self.paligemma.config.text_config
            del self.paligemma.model.language_model
            self.paligemma.model.language_model = PiGemmaModel(text_config)

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # 将完整的视觉路径保持为 float32，以避免来回切换（切换会导致
        # 优化器报 "same dtype" 错误）。与 PI05 保持一致。
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        # 视觉塔（vision tower）和 multi_modal_projector 保持为 float32（params_to_keep_float32）。与 PI05 保持一致。
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output
        norm = 2048**0.5
        features = features / norm * norm
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            # prefix_output 供语言头使用
            # 形状：[batch_size, seq_len, hidden_size]，其中 hidden_size = 2048
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        return [prefix_output, suffix_output], prefix_past_key_values


class PI0FastPytorch(nn.Module):  # 参见 openpi `PI0Pytorch`
    """PI0Fast 核心 PyTorch 模型。"""

    def __init__(
        self,
        config: PI0FastConfig,
        rtc_processor: RTCProcessor | None = None,
        paligemma_tokenizer: "AutoTokenizer | None" = None,
    ):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor
        self._paligemma_tokenizer = paligemma_tokenizer

        paligemma_config = get_gemma_config(config.paligemma_variant)

        self.paligemma_with_expert = PI0FastPaliGemma(
            paligemma_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        # 初始化梯度检查点标志
        self.gradient_checkpointing_enabled = False

        # 如有需要则编译模型
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions_fast = torch.compile(self.sample_actions_fast, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """启用梯度检查点以优化显存。"""
        self.gradient_checkpointing_enabled = True
        # 调用正确的 gradient_checkpointing_enable() 方法，并使用 use_reentrant=False 以获得更好的显存效率
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logging.info("Enabled gradient checkpointing for PI0FastPytorch model")

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点。"""
        self.gradient_checkpointing_enabled = False
        # 调用正确的 gradient_checkpointing_disable() 方法
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing_disable()
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing_disable()
        logging.info("Disabled gradient checkpointing for PI0FastPytorch model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """辅助方法：如果启用了梯度检查点则应用之。"""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def embed_prefix_fast(
        self,
        images,
        img_masks,
        tokens,
        masks,
        fast_action_tokens=None,
        fast_action_masks=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """嵌入图像、语言 token 和 FAST 动作 token。

        注意力模式：
        - 图像 + 语言：彼此之间双向关注
        - FAST：关注图像 + 语言，彼此之间为因果（causal）关注

        参数：
            images: 图像张量列表
            img_masks: 图像掩码列表
            tokens: 语言指令 token
            masks: token 的注意力掩码
            fast_action_tokens: FAST 动作 token（离散 token ID）
            fast_action_masks: FAST 动作 token 的填充掩码

        返回：
            embs: 拼接后的嵌入 [images, tokens, fast_action_tokens]
            pad_masks: 填充掩码
            att_masks: 2D 注意力掩码
            total_T_images: 图像 token 总数
            num_fast_embs: FAST 动作 token 嵌入的数量
        """
        embs = []
        pad_masks = []
        att_mask_segments = []
        total_t_images = 0
        num_fast_embs = 0

        # 处理图像
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_mask_segments.append(("image", num_img_embs))
            total_t_images += num_img_embs

        # 处理语言指令 token
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return lang_emb

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_mask_segments.append(("language", num_lang_embs))

        # 处理 FAST 动作 token（离散 token ID）
        if fast_action_tokens is not None:

            def fast_action_embed_func(fast_action_tokens):
                fast_emb = self.paligemma_with_expert.embed_language_tokens(fast_action_tokens)
                return fast_emb

            fast_action_emb = self._apply_checkpoint(fast_action_embed_func, fast_action_tokens)
            embs.append(fast_action_emb)

            num_fast_embs = fast_action_tokens.shape[1]
            pad_masks.append(fast_action_masks)
            att_mask_segments.append(("fast", num_fast_embs))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        # 创建自定义 2D 注意力掩码：
        # - 图像 + 语言：彼此之间双向关注
        # - FAST：关注图像 + 语言，彼此之间为因果（causal）关注
        att_masks = self._create_custom_attention_mask_fast(att_mask_segments, pad_masks, bsize)

        return embs, pad_masks, att_masks, total_t_images, num_fast_embs

    def _create_custom_attention_mask_fast(self, att_mask_segments, pad_masks, bsize):
        """创建自定义 2D 注意力掩码。

        注意力规则：
        - 图像 + 语言：彼此之间双向关注
        - FAST：关注图像 + 语言，彼此之间为因果（causal）关注
        """
        total_len = sum(length for _, length in att_mask_segments)
        device = pad_masks.device

        att_2d_masks = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)

        positions = []
        current_pos = 0
        for seg_type, seg_len in att_mask_segments:
            positions.append((seg_type, current_pos, current_pos + seg_len))
            current_pos += seg_len

        for _i, (query_type, query_start, query_end) in enumerate(positions):
            for _j, (key_type, key_start, key_end) in enumerate(positions):
                # 图像和语言可以双向地相互关注
                if (
                    query_type in ["image", "language"]
                    and key_type in ["image", "language"]
                    or query_type == "fast"
                    and key_type in ["image", "language"]
                ):
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = True

                # FAST token 以因果方式关注自身
                elif query_type == "fast" and key_type == "fast":
                    fast_len = query_end - query_start
                    causal_mask = torch.tril(torch.ones(fast_len, fast_len, dtype=torch.bool, device=device))
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = causal_mask[None, :, :]

        # 应用填充掩码
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        att_2d_masks = att_2d_masks & pad_2d_masks

        return att_2d_masks

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        fast_action_tokens,
        fast_action_masks,
    ) -> dict:
        """PI0Fast 的前向传播。

        实现了 Pi0FAST 训练目标：使用交叉熵损失预测下一个动作 token。

        参数：
            images: 图像张量列表
            img_masks: 图像掩码列表
            tokens: 语言指令 token
            masks: token 的注意力掩码
            fast_action_tokens: 离散动作 token ID [B, max_action_tokens]
            fast_action_masks: 快速动作 token 的填充掩码 [B, max_action_tokens]

        返回：
            包含 'fast_loss' 和 'loss' 键的字典
        """
        if fast_action_tokens is None or fast_action_masks is None:
            raise ValueError("fast_action_tokens and fast_action_masks are required for FAST-only mode")

        # 嵌入带 FAST token 的前缀
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, num_fast_embs = (
            self.embed_prefix_fast(
                images,
                img_masks,
                tokens,
                masks,
                fast_action_tokens=fast_action_tokens,
                fast_action_masks=fast_action_masks,
            )
        )

        # 如有需要，将嵌入转换为 bfloat16
        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 对于下一 token 预测，输入 token [0:T-1] 以预测 token [1:T]
        input_embs = prefix_embs
        input_pad_masks = prefix_pad_masks
        input_att_masks = prefix_att_masks

        position_ids = torch.cumsum(input_pad_masks, dim=1) - 1
        att_2d_4d = prepare_attention_masks_4d(input_att_masks, dtype=input_embs.dtype)

        # 通过 paligemma（语言模型）进行前向传播
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[input_embs, None],  # 无后缀/动作专家
            use_cache=False,
            adarms_cond=[None, None],
        )

        # 使用 FAST LM 头获取 FAST 动作 token 的 logits
        # 仅对预测 FAST token 的位置计算 logits
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        # 目标是 FAST 动作 token
        fast_targets = fast_action_tokens  # (B, num_fast_embs)

        # 提取用于 FAST token 预测的 logits
        fast_hidden = prefix_out[:, -fast_targets.shape[1] :, :]
        fast_logits_for_pred = lm_head(fast_hidden)  # (B, num_fast_embs, gemma_vocab_size)

        # 左移以进行下一步预测，并右移目标
        # logits[:, i] 预测 targets[:, i+1]
        fast_logits_for_pred = fast_logits_for_pred[:, :-1, :]  # logits 左移
        fast_targets = fast_targets[:, 1:]  # 目标右移
        fast_action_masks = fast_action_masks[:, 1:]  # 掩码右移以与目标对齐

        # 计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        fast_logits_flat = fast_logits_for_pred.reshape(-1, fast_logits_for_pred.size(-1))
        fast_targets_flat = fast_targets.reshape(-1)

        fast_loss_per_token = loss_fct(fast_logits_flat, fast_targets_flat)
        fast_loss_per_token = fast_loss_per_token.reshape(fast_targets.shape)

        # 应用掩码并计算平均损失
        masked_fast_loss = fast_loss_per_token * fast_action_masks.float()
        fast_loss = masked_fast_loss.sum() / fast_action_masks.sum().clamp(min=1)

        return {
            "ce_loss": fast_loss,
            "loss": fast_loss,
        }

    @torch.no_grad()
    def sample_actions_fast(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps=None,
        temperature=0.0,
    ) -> torch.Tensor:
        """
        FAST token 的低效但安全的自回归解码。
        与 _generate_subtask_tokens 的模式一致。
        TODO: jadechoghari，我们是否应该将此逻辑移到 PI0FastPolicy 类中？
        """
        if max_decoding_steps is None:
            max_decoding_steps = self.config.max_action_tokens

        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        # 在 token 之后添加 bos token
        bos_token = torch.full(
            (bsize, 1), self._paligemma_tokenizer.bos_token_id, dtype=torch.long, device=device
        )
        tokens = torch.cat([tokens, bos_token], dim=1)
        masks = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)

        # 1. 初始嵌入（与训练前缀一致）
        # prefix_embs 将包含 [图像、语言提示、BOS]
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, _ = self.embed_prefix_fast(
            images, img_masks, tokens, masks, fast_action_tokens=None, fast_action_masks=None
        )

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        generated_action_tokens = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)

        # 2. 解码循环（每一步都重新计算完整序列）
        for t in range(max_decoding_steps):
            # 始终根据当前的填充掩码重新计算位置 ID
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            att_4d = prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)

            # 完整的前向传播（无 kv 缓存）
            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                adarms_cond=[None, None],
            )

            # 从序列的最后一个位置预测下一个 token
            last_logits = lm_head(prefix_out[:, -1:, :])  # (B, 1, vocab_size)

            if temperature > 0:
                probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

            generated_action_tokens[:, t] = next_token.squeeze(-1)

            # 3. 为下一次迭代更新序列（除非是最后一步）
            if t < max_decoding_steps - 1:
                # 嵌入新生成的 token
                next_token_emb = self.paligemma_with_expert.embed_language_tokens(next_token)
                if prefix_embs.dtype == torch.bfloat16:
                    next_token_emb = next_token_emb.to(dtype=torch.bfloat16)

                # 追加到嵌入中
                prefix_embs = torch.cat([prefix_embs, next_token_emb], dim=1)

                # 更新填充掩码（新 token 始终有效/为 1）
                prefix_pad_masks = torch.cat(
                    [prefix_pad_masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1
                )

                # 更新 2D 注意力掩码：扩展矩阵
                old_len = prefix_att_masks.shape[1]
                new_len = old_len + 1
                new_att_masks = torch.zeros((bsize, new_len, new_len), dtype=torch.bool, device=device)
                new_att_masks[:, :old_len, :old_len] = prefix_att_masks
                # 新 token 关注更新后序列中所有非填充的 token
                new_att_masks[:, -1, :] = prefix_pad_masks
                prefix_att_masks = new_att_masks
        return generated_action_tokens

    @torch.no_grad()
    def sample_actions_fast_kv_cache(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps=None,
        temperature=0.0,
    ) -> torch.Tensor:
        """
        使用 KV 缓存优化的 FAST token 自回归解码。

        一旦所有序列都输出了动作结束标记，贪心解码就会停止。返回的
        张量保持其固定形状，批次级停止之后未生成的位置保持为零填充。
        随机解码始终运行到 ``max_decoding_steps``，这样提前停止就不会
        改变后续调用所使用的随机数状态。
        """
        if max_decoding_steps is None:
            max_decoding_steps = self.config.max_action_tokens

        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        # detokenize_actions() 会在第一个 "|" 处截断，因此一旦所有序列都输出了
        # 该标记，贪心解码就可以停止。随机解码保持不变，因为跳过
        # multinomial 调用会使后续调用的随机数状态发生偏移。
        end_of_action_token_id = self._paligemma_tokenizer.convert_tokens_to_ids("|")
        finished = torch.zeros(bsize, dtype=torch.bool, device=device) if temperature == 0 else None

        # --- 1. 预填充阶段 ---
        # 一次性处理 图像 + 文本提示 + BOS token，以填充 KV 缓存。

        # 向提示添加 BOS token
        bos_token = torch.full(
            (bsize, 1), self._paligemma_tokenizer.bos_token_id, dtype=torch.long, device=device
        )
        tokens_in = torch.cat([tokens, bos_token], dim=1)
        masks_in = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)

        # 嵌入前缀 [图像、语言、BOS]
        # fast_action_tokens=None 表示我们只是嵌入条件（图像+文本）
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, _ = self.embed_prefix_fast(
            images, img_masks, tokens_in, masks_in, fast_action_tokens=None, fast_action_masks=None
        )

        # 确保精度正确（bfloat16/float32）
        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 创建位置 ID（掩码的累加和减 1）
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # 为前缀创建 4D 掩码
        att_4d = prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)

        # 使用 use_cache=True 进行前向传播（预填充）
        # 我们只传入 [prefix_embs, None]，因为此时尚未使用后缀（专家）模型
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,  # 启用缓存
            adarms_cond=[None, None],
        )

        # 从前缀的最后一个 logit 采样第一个动作 token
        last_logits = lm_head(prefix_out[:, -1:, :])  # (B, 1, V)
        if temperature > 0:
            probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

        # 初始化生成 token 的存储
        generated_action_tokens = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)
        generated_action_tokens[:, 0] = next_token.squeeze(-1)
        if finished is not None:
            finished |= next_token.squeeze(-1) == end_of_action_token_id
            if bool(finished.all()):
                return generated_action_tokens

        # 跟踪有效 token 掩码（填充为 0，有效为 1）
        # 我们需要用它来告诉新 token 可以关注哪些内容（图像 + 文本 + 历史动作）
        current_pad_mask = prefix_pad_masks

        # --- 2. 解码阶段 ---
        # 使用缓存逐个生成剩余的 token。

        for t in range(1, max_decoding_steps):
            # 嵌入前一个单独的 token
            # 直接使用 embed_language_tokens，以避免完整前缀嵌入的开销
            next_token_emb = self.paligemma_with_expert.embed_language_tokens(next_token)
            if prefix_embs.dtype == torch.bfloat16:
                next_token_emb = next_token_emb.to(dtype=torch.bfloat16)

            # 更新填充掩码：为新的有效 token 追加 1
            new_column = torch.ones((bsize, 1), dtype=torch.bool, device=device)
            current_pad_mask = torch.cat([current_pad_mask, new_column], dim=1)

            # 更新单个新 token 的位置 ID
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()

            # 为单个新步骤创建注意力掩码
            # 新 token 关注历史中所有有效的 token（由 current_pad_mask 捕获）。
            # 形状变为 (B, 1, 1, Total_Len)，与 HF 的缓存逻辑兼容。
            step_att_mask = prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1), dtype=next_token_emb.dtype
            )

            # 前向传播（解码步骤）
            # input_embeds 仅为新 token (B, 1, D)
            (step_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,  # 传入更新后的缓存
                inputs_embeds=[next_token_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )

            # 采样下一个 token
            last_logits = lm_head(step_out[:, -1:, :])
            if temperature > 0:
                probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

            generated_action_tokens[:, t] = next_token.squeeze(-1)

            if finished is not None:
                finished |= next_token.squeeze(-1) == end_of_action_token_id
                if bool(finished.all()):
                    break

        return generated_action_tokens


class PI0FastPolicy(PreTrainedPolicy):
    """用于 LeRobot 的 PI0Fast 策略。"""

    config_class = PI0FastConfig
    name = "pi0_fast"

    def __init__(
        self,
        config: PI0FastConfig,
        **kwargs,
    ):
        """
        参数：
            config: 策略配置类实例。
        """
        require_package("transformers", extra="pi")
        require_package("scipy", extra="pi")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # 首先加载分词器
        try:
            # 加载 FAST 分词器
            self.action_tokenizer = AutoProcessor.from_pretrained(
                config.action_tokenizer_name, trust_remote_code=True
            )

            # 加载 PaliGemma 分词器用于 token 转换
            self._paligemma_tokenizer = AutoTokenizer.from_pretrained(
                config.text_tokenizer_name, trust_remote_code=True, add_eos_token=True, add_bos_token=False
            )

            logging.info("Loaded FAST tokenizer for action detokenization")
        except Exception as e:
            logging.error(f"Failed to load FAST tokenizer for action detokenization: {e}")
            logging.error("Tokenizer loading is required for proper policy initialization; aborting.")
            raise RuntimeError("Failed to load required tokenizers for PI0FastPolicy initialization") from e

        # 初始化 PI0Fast 核心模型
        self.init_rtc_processor()
        self.model = PI0FastPytorch(
            config, rtc_processor=self.rtc_processor, paligemma_tokenizer=self._paligemma_tokenizer
        )

        # 如有需要则启用梯度检查点
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """重写 from_pretrained 方法，以处理键重映射并显示重要声明。"""
        print(
            "The PI0Fast model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # 如果提供了配置则使用，否则创建默认配置
        if config is None:
            config = PreTrainedConfig.from_pretrained(
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

        # 初始化模型但不加载权重
        # 检查 kwargs 中是否提供了 dataset_stats
        model = cls(config, **kwargs)

        # 加载 state dict（期望键带有 "model." 前缀）
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # 首先，修复所有键差异（参见 openpi model.py, _fix_pytorch_state_dict_keys）
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # 然后，为所有尚未带 "model." 前缀的键添加该前缀
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # 将重映射后的 state dict 加载到模型中
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # 参见 openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """修复 state dict 键，使其与当前模型架构匹配。"""

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # 处理视觉塔嵌入层的潜在差异
            if "patch_embedding" in key:
                # 某些检查点可能包含此项，但当前模型期望不同的结构
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """重置内部状态——在环境重置时调用。"""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """如果配置中启用了 RTC，则初始化 RTC 处理器。"""
        self.rtc_processor = None

        # 如果提供了配置则创建处理器
        # 即使未启用 RTC，我们仍然可以跟踪去噪数据
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """为模型预处理图像。

        来自 LeRobot 的图像通常为 [B, C, H, W] 格式，并归一化到 [0, 1]。
        PaliGemma 期望图像为 [B, C, H, W] 格式，并归一化到 [-1, 1]。
        """
        images = []
        img_masks = []

        # 从模型参数获取设备
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # 预处理批次中存在的图像特征
        for key in present_img_keys:
            img = batch[key]

            # 确保张量与模型在同一设备上
            if img.device != device:
                img = img.to(device)

            # 为保持一致，确保数据类型为 float32
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # 来自 openpi preprocess_observation_pytorch：同时处理 [B, C, H, W] 和 [B, H, W, C] 两种格式
            is_channels_first = img.shape[1] == 3  # 检查通道是否位于维度 1

            if is_channels_first:
                # 将 [B, C, H, W] 转换为 [B, H, W, C] 以便处理
                img = img.permute(0, 2, 3, 1)

            # 来自 openpi preprocess_observation_pytorch：如有需要则带填充缩放
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # 按照 siglip 的预期，从 [0,1] 归一化到 [-1,1]
            img = img * 2.0 - 1.0

            # 来自 openpi preprocess_observation_pytorch：如果原本就是通道优先格式，则转换回 [B, C, H, W] 格式
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # 创建掩码（真实图像全为 1）
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # 将批次中不存在的图像特征创建为全填充 -1 的图像
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # 为 SigLIP 填充 -1
            mask = torch.zeros_like(mask)  # 空相机的掩码为零
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """填充动作"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _paligemma_tokens_to_act_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        将 PaliGemma token 转换回动作 token（_act_tokens_to_paligemma_tokens 的逆操作）。

        参数：
            tokens: PaliGemma token ID

        返回：
            动作 token ID
        """
        return self._paligemma_tokenizer.vocab_size - 1 - self.config.fast_skip_tokens - tokens

    def decode_actions_with_fast(
        self, token_ids: list[int], time_horizon: int, action_dim: int, relaxed_decoding: bool = True
    ) -> np.ndarray:
        """
        使用 FAST 分词器将动作 token ID 解码回连续动作值。

        参数：
            token_ids: 要解码的 token ID 列表。
            time_horizon: 动作的时间步数。
            action_dim: 每个动作的维度。
            relaxed_decoding: 是否使用宽松解码（允许部分序列）。

        返回：
            表示解码后动作的 numpy 数组。
        """
        decoded_actions = []

        for token in token_ids:
            try:
                decoded_tokens = self.action_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.action_tokenizer.min_token

                if relaxed_decoding:
                    # 期望的序列长度
                    expected_seq_len = time_horizon * action_dim
                    diff = expected_seq_len - decoded_dct_coeff.shape[0]

                    # 如果过长则应用截断
                    if diff < 0:
                        decoded_dct_coeff = decoded_dct_coeff[:expected_seq_len]  # 从右侧截断

                    # 如果过短则应用填充
                    elif diff > 0:
                        decoded_dct_coeff = np.pad(
                            decoded_dct_coeff, (0, diff), mode="constant", constant_values=0
                        )

                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
                assert decoded_dct_coeff.shape == (
                    time_horizon,
                    action_dim,
                ), (
                    f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({time_horizon}, {action_dim})"
                )

            except Exception as e:
                logging.warning(f"Error decoding tokens: {e}")
                logging.warning(f"Tokens: {token}")
                decoded_dct_coeff = np.zeros((time_horizon, action_dim))

            decoded_actions.append(
                idct(decoded_dct_coeff / self.action_tokenizer.scale, axis=0, norm="ortho")
            )

        return np.stack(decoded_actions)

    def detokenize_actions(self, tokens: torch.Tensor, action_horizon: int, action_dim: int) -> torch.Tensor:
        """
        将动作 token 反分词回连续动作。

        该方法使用 FAST 分词器将模型预测的动作 token 转换回连续动作值。
        它处理从 PaliGemma token 空间到动作 token 空间的转换，
        然后使用 DCT 解码将动作 token 解码为连续值。

        参数：
            tokens: 分词输出的输入张量。形状：(B, seq_len) 或 (seq_len,)
            action_horizon: 动作的时间步数。
            action_dim: 每个动作的维度。

        返回：
            连续动作张量。形状：(B, action_horizon, action_dim) 或 (action_horizon, action_dim)
        """
        if self.action_tokenizer is None or self._paligemma_tokenizer is None:
            raise ValueError(
                "Action tokenizer not initialized. Make sure fast_only=True in config and tokenizers loaded successfully."
            )

        # 处理单个样本（添加批次维度）
        single_sample = tokens.dim() == 1
        if single_sample:
            tokens = tokens.unsqueeze(0)

        # 将 token ID 转换为 token 字符串
        decoded_tokens = [self._paligemma_tokenizer.convert_ids_to_tokens(seq.tolist()) for seq in tokens]
        # 获取 "Action: " 的 token 序列以便将其移除
        action_prefix_ids = self._paligemma_tokenizer.encode("Action: ", add_special_tokens=False)
        action_prefix_tokens = self._paligemma_tokenizer.convert_ids_to_tokens(action_prefix_ids)
        action_prefix_len = len(action_prefix_tokens)

        # 通过移除第一个 "|"（动作结束标记）之后的所有内容
        # 以及移除所有出现的 "Action: " token 序列来清理 token
        # 断言开头包含 "Action: "
        if self.config.validate_action_token_prefix:
            for token_seq in decoded_tokens:
                assert len(token_seq) >= 2 and token_seq[0] == "Action" and token_seq[1] == ":", (
                    f"Token sequence does not start with ['Action', ':']: {token_seq}"
                )

        cleaned_tokens = []
        for token_seq in decoded_tokens:
            # 移除 "|" 之后的所有内容
            if "|" in token_seq:
                token_seq = token_seq[: token_seq.index("|")]

            # 移除所有出现的 "Action: " token 序列
            i = 0
            while i <= len(token_seq) - action_prefix_len:
                if token_seq[i : i + action_prefix_len] == action_prefix_tokens:
                    # 找到匹配项，将其移除
                    token_seq = token_seq[:i] + token_seq[i + action_prefix_len :]
                else:
                    i += 1

            cleaned_tokens.append(token_seq)

        # 将 token 字符串转换回 ID
        raw_action_tokens = [
            torch.tensor(
                self._paligemma_tokenizer.convert_tokens_to_ids(token_seq),
                dtype=torch.long,
                device=tokens.device,
            )
            for token_seq in cleaned_tokens
        ]

        # 将 PaliGemma token 转换为动作 token
        action_tokens = [
            self._paligemma_tokens_to_act_tokens(raw_action_token) for raw_action_token in raw_action_tokens
        ]

        # 将动作 token 解码为连续动作
        actions = self.decode_actions_with_fast(
            action_tokens, time_horizon=action_horizon, action_dim=action_dim
        )

        # 转换为张量并返回
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=tokens.device)

        # 如果输入是单个样本，则移除批次维度
        if single_sample:
            actions_tensor = actions_tensor.squeeze(0)

        return actions_tensor

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """根据环境观测选择单个动作。"""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # n_action_steps > 1 时的动作队列逻辑
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # 转置以得到形状 (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """根据环境观测预测一个动作块。"""
        self.eval()
        # 准备输入
        images, img_masks = self._preprocess_images(batch)

        # 仅 FAST 模式：使用自回归解码
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # 获取解码参数
        temperature = self.config.temperature
        max_decoding_steps = self.config.max_decoding_steps

        # 自回归地采样动作 token
        if self.config.use_kv_cache:
            action_tokens = self.model.sample_actions_fast_kv_cache(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
            )
        else:
            action_tokens = self.model.sample_actions_fast(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
            )

        # 将动作 token 反分词为连续动作
        action_horizon = self.config.n_action_steps
        action_dim = self.config.output_features[ACTION].shape[0]

        continuous_actions = self.detokenize_actions(
            action_tokens, action_horizon=action_horizon, action_dim=action_dim
        )

        return continuous_actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """将批次送入模型并计算训练损失。"""

        # 准备输入
        images, img_masks = self._preprocess_images(batch)

        # 从批次中获取 FAST 动作 token
        fast_action_tokens = batch.get(ACTION_TOKENS)  # (B, max_action_tokens)
        fast_action_masks = batch.get(ACTION_TOKEN_MASK)  # (B, max_action_tokens)

        # 使用完整的语言 token（不区分 high_level_task 和 subtask）
        tokens = batch.get(OBS_LANGUAGE_TOKENS)
        masks = batch.get(OBS_LANGUAGE_ATTENTION_MASK)

        if fast_action_tokens is None or fast_action_masks is None:
            raise ValueError(
                f"PI0Fast requires {ACTION_TOKENS} and {ACTION_TOKEN_MASK} to be present in the batch"
            )

        loss_dict = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            fast_action_tokens,
            fast_action_masks,
        )

        loss = loss_dict["loss"]
        detailed_loss_dict = {
            "loss": loss.item(),
            "ce_loss": loss_dict["ce_loss"].item(),
        }
        return loss, detailed_loss_dict
