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

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
else:
    AutoProcessor = None
    Qwen3VLForConditionalGeneration = None

from .configuration_vla_jepa import VLAJEPAConfig


class Qwen3VLInterface(torch.nn.Module):
    def __init__(self, config: VLAJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.qwen_model_name,
            torch_dtype=self._get_torch_dtype(config.torch_dtype),
        )
        self.processor = AutoProcessor.from_pretrained(config.qwen_model_name)
        self.processor.tokenizer.padding_side = config.tokenizer_padding_side
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    @staticmethod
    def _get_torch_dtype(dtype_name: str) -> torch.dtype:
        if dtype_name == "float32":
            return torch.float32
        if dtype_name == "float16":
            return torch.float16
        return torch.bfloat16

    def expand_tokenizer(self) -> tuple[list[str], list[int], int]:
        # starVLA/JEVLA checkpoint 将动作 token 扩展为 action_horizon * 4 个，
        # 与 vj2 的 num_action_tokens_per_timestep 无关。必须保持这一数量，
        # Qwen 的 embedding/lm_head 的 checkpoint 形状才能匹配。
        max_action_tokens = self.config.chunk_size * 4
        tokenizer = self.processor.tokenizer
        action_tokens = []
        action_token_ids = []
        for idx in range(max_action_tokens):
            token = self.config.special_action_token.format(idx)
            action_tokens.append(token)
            if token not in tokenizer.get_vocab():
                tokenizer.add_tokens([token], special_tokens=True)
            action_token_ids.append(tokenizer.convert_tokens_to_ids(token))

        embodied_action_token = self.config.embodied_action_token
        if embodied_action_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([embodied_action_token], special_tokens=True)
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        # Qwen3-VL-2B 自带 267 个备用嵌入行，因此在 chunk_size 不超过 66 时，新增的
        # `chunk_size * 4 + 1` 个 token 无需 resize 即可容纳。超过该值后，resize 会改变
        # `embed_tokens` / `lm_head` 的形状，除非这些前缀位于 `reinit_modules` 中，否则
        # checkpoint 将无法跨 chunk_size 加载——这里发出警告，而不是静默失败。
        current_rows = self.model.get_input_embeddings().weight.size(0)
        if current_rows < len(tokenizer):
            logging.warning(
                f"chunk_size={self.config.chunk_size} needs {max_action_tokens + 1} added tokens, "
                f"which exceeds the {current_rows} embedding rows of {self.config.qwen_model_name}. "
                f"Resizing to {len(tokenizer)} rows changes the shapes of "
                f"`model.qwen.model.model.language_model.embed_tokens` and "
                f"`model.qwen.model.lm_head`, so this model will not load from a checkpoint trained "
                f"with a different chunk_size unless those prefixes are in `reinit_modules`."
            )
            self.model.resize_token_embeddings(len(tokenizer))
        return action_tokens, action_token_ids, embodied_action_token_id

    def build_inputs(
        self,
        images: Sequence[Sequence[torch.Tensor]],
        instructions: Sequence[str],
        action_prompt: str,
        embodied_prompt: str,
    ) -> dict[str, torch.Tensor]:
        messages = []
        for sample_images, instruction in zip(images, instructions, strict=True):
            prompt = self.config.prompt_template.format(
                instruction=instruction,
                actions=action_prompt,
                e_actions=embodied_prompt,
            )
            content = [{"type": "image", "image": img} for img in sample_images]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        # Qwen 图像处理器是一个基于 torchvision 的快速处理器：将图像作为 GPU 张量传入
        # （配合 `device`）可让整条视觉流水线都留在设备上，避免 GPU->CPU->GPU 的往返。
        # 图像张量会经由 apply_chat_template 原样传递到 Qwen3VLProcessor.__call__。
        # do_rescale=False：图像到达时已经是 [0, 1] 的 float（数据集解码器产出 float32/255，
        # 且 VISUAL 归一化为 IDENTITY），因此跳过处理器的 /255 缩放，而不是绕道 uint8 再转回。
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "padding": True,
                "return_tensors": "pt",
                "device": self.model.device,
                "do_rescale": False,
            },
        )
        return batch_inputs.to(self.model.device)

    @staticmethod
    def to_pixel_values(image_tensor: torch.Tensor) -> torch.Tensor:
        """为快速处理器准备图像/视频张量（配合 do_rescale=False 使用）。

        数据集解码器产出 [0, 1] 的 float32（通道优先），且 VISUAL 归一化为 IDENTITY，
        因此张量到达时已经在 [0, 1]；这里直接以 float 透传，让处理器去做归一化
        （不 rescale，也不做 uint8 量化）。单通道会被扩展为 3 通道，以匹配 RGB 处理器。

        适用于任意通道优先的布局（通道维为 -3）：[C, H, W]、[B, C, H, W]、
        [T, C, H, W]、[B, V, T, C, H, W]……
        """
        image = image_tensor.detach().float()
        if image.shape[-3] == 1:
            repeats = [1] * image.ndim
            repeats[-3] = 3
            image = image.repeat(*repeats)
        return image
