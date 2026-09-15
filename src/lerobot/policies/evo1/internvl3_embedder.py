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
import torch.nn as nn
import torchvision.transforms.functional as tvf
from torchvision.transforms.functional import InterpolationMode

from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import is_flash_attn_2_available
else:
    AutoModel = None
    AutoTokenizer = None
    is_flash_attn_2_available = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"  # nosec B105
IMG_START_TOKEN = "<img>"  # nosec B105
IMG_END_TOKEN = "</img>"  # nosec B105

logger = logging.getLogger(__name__)


def _batched_resize_01(images: torch.Tensor, image_size: int) -> torch.Tensor:
    """在设备上将一批 ``[0, 1]`` 图像缩放到 ``(image_size, image_size)``。

    在数值上复刻 InternVL3 的参考 PIL 预处理
    （``to_pil_image`` -> ``Image.resize`` -> ``to_tensor``）：浮点输入会像
    ``to_pil_image`` 一样被精确量化为 uint8，然后使用双三次插值加抗锯齿进行缩放，
    这与 PIL 的默认重采样器一致。与参考实现逐像素保持一致，使该策略可以与
    上游 EVO1 预处理生成的检查点互换使用。

    Args:
        images: 形状为 ``(N, C, H, W)``、取值在 ``[0, 1]`` 内的浮点张量。

    Returns:
        形状为 ``(N, C, image_size, image_size)``、取值在 ``[0, 1]`` 内的 float32 张量。
    """
    # to_pil_image() 会将浮点 [0, 1] 量化为 uint8（x * 255，截断）；这里复现该行为，
    # 使双三次重采样看到的整数像素与 PIL 看到的相同。
    pixels_u8 = (images * 255.0).clamp(0, 255).to(torch.uint8)
    resized = tvf.resize(
        pixels_u8, [image_size, image_size], interpolation=InterpolationMode.BICUBIC, antialias=True
    )
    return resized.to(torch.float32) / 255.0


def _batched_pixel_values(
    camera_images: Sequence[torch.Tensor],
    max_views: int,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """在不离开设备的情况下，从各相机的 ``[0, 1]`` 图像批次构建 InternVL3 的 ``pixel_values``。

    每张图像会被缩放、转换为 ``dtype``，并进行 ImageNet 归一化（每张图像
    单个 tile），在整个小批次上进行批处理。缺失的视图（相机数量少于
    ``max_views``）用零图像填充；它们的占位符 token 会在下游通过
    ``_mask_absent_image_tokens`` 从注意力中掩蔽掉。

    Returns:
        形状为 ``(B * max_views, C, image_size, image_size)`` 的 ``pixel_values``，
        按 ``(sample, view)`` 行主序排列，以与提示词中逐视图的图像占位符对齐。
    """
    resized: list[torch.Tensor] = []
    for image in camera_images:
        resized.append(_batched_resize_01(image.to(device=device), image_size).to(dtype))

    batch_size = resized[0].shape[0]
    channels = resized[0].shape[1]
    while len(resized) < max_views:
        resized.append(torch.zeros(batch_size, channels, image_size, image_size, dtype=dtype, device=device))

    stacked = torch.stack(resized[:max_views], dim=1)  # (B, V, C, H, W)
    mean = mean.to(device=device, dtype=dtype).view(1, 1, -1, 1, 1)
    std = std.to(device=device, dtype=dtype).view(1, 1, -1, 1, 1)
    normalized = (stacked - mean) / std
    return normalized.reshape(batch_size * max_views, channels, image_size, image_size)


class InternVL3Embedder(nn.Module):
    """使用原生 HF InternVL3 模型的视觉-语言嵌入器（无需 trust_remote_code）。"""

    def __init__(
        self,
        model_name="OpenGVLab/InternVL3-1B-hf",
        image_size=448,
        device="cuda",
        num_language_layers: int | None = 14,
        model_dtype: str | torch.dtype = "bfloat16",
        use_flash_attn: bool = True,
        max_text_length: int = 1024,
        enable_gradient_checkpointing: bool = True,
        gradient_checkpointing_use_reentrant: bool = False,
        hub_kwargs: dict | None = None,
    ):
        super().__init__()
        self._requested_device = device
        self.image_size = image_size
        self.num_language_layers = num_language_layers
        self.max_text_length = max_text_length
        self.enable_gradient_checkpointing = bool(enable_gradient_checkpointing)
        self.gradient_checkpointing_use_reentrant = bool(gradient_checkpointing_use_reentrant)
        hub_kwargs = hub_kwargs or {}

        require_package("transformers", extra="evo1")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **hub_kwargs)
        if isinstance(model_dtype, str):
            try:
                model_dtype = getattr(torch, model_dtype)
            except AttributeError as exc:
                raise ValueError(f"Unsupported EVO1 vlm_dtype '{model_dtype}'") from exc
        self.model_dtype = model_dtype

        attn_implementation = (
            "flash_attention_2" if (use_flash_attn and is_flash_attn_2_available()) else "eager"
        )
        if use_flash_attn and attn_implementation == "eager":
            logger.warning(
                "Flash Attention 2 is unavailable on this runtime. Falling back to eager attention."
            )

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            **hub_kwargs,
        ).to(self._requested_device)

        checkpoint_image_size = getattr(self.model.config.vision_config, "image_size", None)
        if isinstance(checkpoint_image_size, (list, tuple)):
            checkpoint_image_size = checkpoint_image_size[0]
        if checkpoint_image_size is not None and int(checkpoint_image_size) != int(image_size):
            raise ValueError(
                f"EVO1 image_resolution ({image_size}) must match the InternVL checkpoint's native "
                f"image size ({checkpoint_image_size}): the checkpoint's image_seq_length assumes "
                "its native resolution, so other sizes would desync the image placeholder tokens "
                "from the vision features."
            )

        self.num_image_token = self.model.config.image_seq_length

        # 将语言模型截断到请求的层数
        layers = self.model.language_model.layers
        if self.num_language_layers is not None:
            layers = layers[: self.num_language_layers]
        self.model.language_model.layers = torch.nn.ModuleList(layers)

        self._configure_memory_features()
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def _configure_memory_features(self) -> None:
        checkpoint_kwargs = {"use_reentrant": self.gradient_checkpointing_use_reentrant}

        if not self.enable_gradient_checkpointing:
            language_model = self.model.language_model
            if hasattr(language_model, "gradient_checkpointing_disable"):
                language_model.gradient_checkpointing_disable()
            vision_tower = getattr(self.model, "vision_tower", None)
            if vision_tower is not None and hasattr(vision_tower, "encoder"):
                vision_tower.encoder.gradient_checkpointing = False
            return

        def _enable_ckpt(module: nn.Module | None) -> bool:
            if module is None:
                return False
            if hasattr(module, "gradient_checkpointing_enable"):
                try:
                    module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
                except TypeError:
                    module.gradient_checkpointing_enable()
                return True
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                return True
            return False

        enabled_any = _enable_ckpt(self.model)

        vision_tower = getattr(self.model, "vision_tower", None)
        if vision_tower is not None:
            enabled_any = _enable_ckpt(vision_tower) or enabled_any

        language_model = self.model.language_model
        enabled_any = _enable_ckpt(language_model) or enabled_any
        if hasattr(language_model, "config"):
            language_model.config.use_cache = False

        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        if enabled_any:
            logger.info("Gradient checkpointing enabled for InternVL3 embedder.")
        else:
            logger.warning(
                "Requested gradient checkpointing, but model does not expose checkpointing controls."
            )

    def _build_multimodal_prompts(
        self,
        batch_num_tiles_list: list[list[int]],
        text_prompts: Sequence[str],
    ) -> list[str]:
        prompts = []
        for num_tiles_list, text_prompt in zip(batch_num_tiles_list, text_prompts, strict=True):
            prompt_segments = []
            for i, tile_count in enumerate(num_tiles_list):
                token_count = self.num_image_token * tile_count
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
                prompt_segments.append(f"Image-{i + 1}: {image_tokens}\n")
            prompts.append("".join(prompt_segments) + text_prompt.strip())
        return prompts

    def get_fused_image_text_embedding_batched(
        self,
        camera_images: Sequence[torch.Tensor],
        image_masks: torch.Tensor,
        text_prompts: Sequence[str],
        return_cls_only: bool = True,
    ):
        """从各相机的 ``[0, 1]`` 图像批次获取融合的 VL 嵌入（无 PIL，无主机往返）。

        Args:
            camera_images: 逐相机张量的列表，每个形状为 ``(B, C, H, W)``，取值在 ``[0, 1]`` 内。
            image_masks: 布尔张量 ``(B, max_views)``，标记存在的视图。

        Returns:
            ``(embeddings, valid_mask)`` 元组。当 ``return_cls_only=False`` 时，``embeddings`` 为
            ``(B, L, H)``，``valid_mask`` 是标记下游注意力可关注的 token 的 ``(B, L)`` 布尔张量
            （填充和缺失视图的 token 为 False）。当 ``return_cls_only=True`` 时，
            ``embeddings`` 是池化后的 ``(B, H)`` 最后一个有效 token 状态，
            ``valid_mask`` 为 None。
        """
        max_views = int(image_masks.shape[1])
        batch_size = int(image_masks.shape[0])
        mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=self.model_dtype)
        std = torch.tensor(IMAGENET_STD, device=self.device, dtype=self.model_dtype)
        pixel_values = _batched_pixel_values(
            camera_images, max_views, self.image_size, mean, std, self.model_dtype, self.device
        )
        # InternVL3 预处理对每张图像使用单个 tile（max_num=1）。
        batch_num_tiles_list = [[1] * max_views for _ in range(batch_size)]
        return self._forward_vlm(
            pixel_values, batch_num_tiles_list, image_masks, text_prompts, return_cls_only
        )

    def _mask_absent_image_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_masks: torch.Tensor,
        batch_num_tiles_list: list[list[int]],
    ) -> torch.Tensor:
        """将缺失（零填充）视图的图像上下文 token 的注意力置零。

        完全向量化：运行时没有任何主机<->设备同步。
        """
        # 每张图像单个 tile（max_num=1），因此每张图像占据相同数量的
        # 上下文 token。
        tiles_per_image = (
            batch_num_tiles_list[0][0] if batch_num_tiles_list and batch_num_tiles_list[0] else 1
        )
        tokens_per_image = self.num_image_token * tiles_per_image

        image_masks = image_masks.to(device=input_ids.device).bool()
        img_token_mask = input_ids == self.img_context_token_id  # (B, L)
        # keep[b, k] 表示第 k 个图像上下文 token（按 view0、view1... 排序）是否保留。
        per_token_keep = image_masks.repeat_interleave(tokens_per_image, dim=1)  # (B, V * tokens_per_image)
        # 按每个上下文 token 在该行上下文 token 中的累计位置为其排序。
        ctx_index = img_token_mask.to(torch.long).cumsum(dim=1) - 1
        ctx_index = ctx_index.clamp(min=0, max=per_token_keep.shape[1] - 1)
        keep_here = torch.gather(per_token_keep, 1, ctx_index)  # (B, L)
        drop = img_token_mask & ~keep_here
        return attention_mask.masked_fill(drop, 0)

    def _forward_vlm(
        self,
        pixel_values: torch.Tensor,
        batch_num_tiles_list: list[list[int]],
        image_masks: torch.Tensor,
        text_prompts: Sequence[str],
        return_cls_only: bool,
    ):
        if pixel_values.shape[0] == 0:
            logger.warning("InternVL3 received an empty image batch after preprocessing.")
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
            if hidden_size is None:
                raise RuntimeError("Unable to infer hidden size for empty InternVL3 batch.")
            return torch.empty(0, hidden_size, device=self.device, dtype=torch.float32), None

        prompts = self._build_multimodal_prompts(batch_num_tiles_list, text_prompts)

        model_inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        ).to(self.device)
        input_ids = model_inputs["input_ids"]
        if input_ids.shape[1] >= self.max_text_length:
            # 截断从右侧切除，因此文本会在图像占位符之前被丢弃——但
            # 较大的 max_views * image_seq_length 预算仍可能侵占占位符。
            # 与其让 VLM 因占位符/视觉特征数量不匹配而崩溃，不如显式报错。
            expected_image_tokens = self.num_image_token * sum(batch_num_tiles_list[0])
            image_token_counts = (input_ids == self.img_context_token_id).sum(dim=1)
            if not bool((image_token_counts == expected_image_tokens).all()):
                raise ValueError(
                    f"Prompt truncation at max_text_length={self.max_text_length} cut into the "
                    f"image placeholder tokens ({expected_image_tokens} expected per sample). "
                    "Increase max_text_length or reduce max_views."
                )
        attention_mask = self._mask_absent_image_tokens(
            input_ids, model_inputs["attention_mask"], image_masks, batch_num_tiles_list
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        fused_hidden = outputs.hidden_states[-1].to(torch.float32)
        valid_mask = attention_mask.to(torch.bool)
        if return_cls_only:
            # 右填充的因果解码器：最后一个有效 token 是唯一关注过
            # 完整图像 + 文本提示词的 token。
            positions = torch.arange(valid_mask.shape[1], device=valid_mask.device)
            last_valid = (valid_mask.long() * positions).argmax(dim=1)
            batch_index = torch.arange(fused_hidden.shape[0], device=fused_hidden.device)
            return fused_hidden[batch_index, last_valid], None
        return fused_hidden, valid_mask

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
