#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from __future__ import annotations

import builtins
import logging
import os
import re
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from ..common.vla_utils import pad_vector, resize_with_pad
from ..pretrained import PreTrainedPolicy, T
from ..utils import populate_queues
from .action_hub import build_action_space
from .configuration_xvla import XVLAConfig
from .soft_transformer import SoftPromptedTransformer

# Florence2 的配置和建模依赖 transformers
if TYPE_CHECKING or _transformers_available:
    from transformers import Florence2Config, Florence2Model
else:
    Florence2Config = None
    Florence2Model = None


class XVLAModel(nn.Module):
    """
    XVLA 主干网络，将 Florence-2 的嵌入与时间/动作 transformer 头衔接起来。
    """

    def __init__(
        self,
        config: XVLAConfig,
        florence_config: Florence2Config,
        proprio_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.chunk_size: int = config.chunk_size
        self.use_proprio: bool = config.use_proprio

        # 在 "auto" 模式下构建带自动检测的动作空间
        if config.action_mode.lower() == "auto":
            # 从 config.action_feature 自动检测真实动作维度
            real_dim = (
                config.action_feature.shape[-1]
                if config.action_feature is not None
                else config.max_action_dim
            )
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=real_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())

        self.dim_action = self.action_space.dim_action
        self.dim_proprio = proprio_dim

        self.vlm = Florence2Model(florence_config)
        # XVLA 只使用 Florence-2 的编码器侧路径；完全移除文本解码器。
        del self.vlm.language_model.decoder

        projection_dim = getattr(florence_config.vision_config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=self.dim_action,
            dim_propio=self.dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )

        # 根据配置应用冻结
        self._apply_freezing()

        # 根据配置应用 dtype 转换
        self._apply_dtype()

    def _get_target_dtype(self) -> torch.dtype:
        """根据配置获取目标 dtype。"""
        if self.config.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _apply_dtype(self) -> None:
        """
        根据配置对模型组件进行 dtype 转换。
        """
        target_dtype = self._get_target_dtype()
        self.to(dtype=target_dtype)

    def _apply_freezing(self) -> None:
        """
        根据配置选项冻结 VLM 视觉编码器和语言编码器。
        只保留策略 transformer 和 soft prompt 可训练。
        """
        # 冻结视觉编码器
        if self.config.freeze_vision_encoder and hasattr(self.vlm, "vision_tower"):
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = False

        # 冻结语言编码器
        if self.config.freeze_language_encoder and hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            # 冻结编码器
            if hasattr(lm, "encoder"):
                for param in lm.encoder.parameters():
                    param.requires_grad = False
            # 冻结共享嵌入
            if hasattr(lm, "shared"):
                for param in lm.shared.parameters():
                    param.requires_grad = False

        # 冻结或解冻策略 transformer
        if not self.config.train_policy_transformer:
            for name, param in self.transformer.named_parameters():
                if "soft_prompt" not in name:
                    param.requires_grad = False

        # 冻结或解冻 soft prompt
        if not self.config.train_soft_prompts and hasattr(self.transformer, "soft_prompt_hub"):
            for param in self.transformer.soft_prompt_hub.parameters():
                param.requires_grad = False

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        通过 Florence2 编码器对文本和多视角图像进行编码。
        """
        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm.get_image_features(valid_images).pooler_output
        tokens_per_view, hidden_dim = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)

        # XVLA 将主视角的图像 token 前置到文本嵌入中，并对所有内容做注意力。
        merged_embeds = torch.cat([image_features[:, 0], inputs_embeds], dim=1)
        attention_mask = torch.ones(merged_embeds.shape[:2], dtype=torch.long, device=merged_embeds.device)

        enc_out = self.vlm.language_model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]

        aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        XVLA 模型的前向传播。
        """
        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        action = action.to(dtype=target_dtype)

        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        t = (
            torch.rand(1, device=input_ids.device, dtype=target_dtype)
            + torch.arange(batch_size, device=input_ids.device, dtype=target_dtype) / batch_size
        ) % (1 - 1e-5)

        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )
        return self.action_space.compute_loss(pred_action, action)

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        self.eval()

        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)

        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        action_dim = self.dim_action

        x1 = torch.randn(batch_size, self.chunk_size, action_dim, device=proprio.device, dtype=target_dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((batch_size,), i / steps, device=proprio.device, dtype=target_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        return self.action_space.postprocess(action)


class XVLAPolicy(PreTrainedPolicy):
    """围绕 XVLA 模型构建的、符合 LeRobot 规范的包装器。"""

    config_class = XVLAConfig
    name = "xvla"

    def __init__(self, config: XVLAConfig, **kwargs):
        require_package("transformers", extra="xvla")
        super().__init__(config)
        config.validate_features()
        florence_config = config.get_florence_config()
        proprio_dim = config.max_state_dim if config.use_proprio else 0
        self.model = XVLAModel(config=config, florence_config=florence_config, proprio_dim=proprio_dim)
        self.reset()

    def reset(self) -> None:
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self) -> dict:
        """返回用于优化的、带名称的可训练参数。

        返回一个包含所有可训练参数的 name -> param 字典。
        这使 xvla-adamw 优化器能够根据参数名称应用差异化的学习率
        （例如 VLM 组件使用 1/10 的学习率）。
        """
        return dict(filter(lambda kv: kv[1].requires_grad, self.named_parameters()))

    def _prepare_state(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        if not self.config.use_proprio or OBS_STATE not in batch:
            return torch.zeros(batch_size, 0, device=device)
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state[:, -1, :]
        return pad_vector(state, self.model.dim_proprio, truncate=True)

    def _prepare_images(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        present_img_keys = [key for key in self.config.image_features if key in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                "All image features are missing from the batch. "
                f"Batch keys: {list(batch.keys())}, expected at least one of {list(self.config.image_features)}."
            )

        images = []
        masks = []
        for key in present_img_keys:
            img = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0.0)
            images.append(img)
            masks.append(torch.ones(img.size(0), dtype=torch.bool, device=img.device))

        stacked_imgs = torch.stack(images, dim=1)
        stacked_masks = torch.stack(masks, dim=1)

        total_views = self.config.num_image_views or stacked_imgs.size(1)
        total_views = max(total_views, stacked_imgs.size(1))
        num_pad = total_views - stacked_imgs.size(1)
        if num_pad > 0:
            pad_shape = (stacked_imgs.size(0), num_pad, *stacked_imgs.shape[2:])
            pad_imgs = stacked_imgs.new_zeros(pad_shape)
            pad_masks = stacked_masks.new_zeros((stacked_masks.size(0), num_pad))
            stacked_imgs = torch.cat([stacked_imgs, pad_imgs], dim=1)
            stacked_masks = torch.cat([stacked_masks, pad_masks], dim=1)

        return stacked_imgs, stacked_masks

    def _get_domain_id(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        candidate = None
        if self.config.domain_feature_key and self.config.domain_feature_key in batch:
            candidate = batch[self.config.domain_feature_key]
        elif "domain_id" in batch:
            candidate = batch["domain_id"]

        if candidate is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if not isinstance(candidate, torch.Tensor):
            candidate = torch.as_tensor(candidate, device=device)
        else:
            candidate = candidate.to(device=device)

        if candidate.ndim == 0:
            candidate = candidate.expand(batch_size)
        if candidate.ndim > 1:
            candidate = candidate.view(candidate.shape[0], -1)[:, 0]
        if candidate.shape[0] != batch_size:
            candidate = candidate.expand(batch_size)
        return candidate.to(dtype=torch.long)

    def _prepare_action_targets(self, batch: dict[str, Tensor]) -> Tensor:
        if ACTION not in batch:
            raise ValueError("Batch is missing action targets required for training.")
        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = pad_tensor_along_dim(actions, self.config.chunk_size, dim=1)
        if actions.shape[-1] != self.model.dim_action:
            actions = pad_vector(actions, self.model.dim_action, truncate=True)
        return actions

    def _build_model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        batch_size = input_ids.shape[0]
        images, image_mask = self._prepare_images(batch)
        domain_id = self._get_domain_id(batch, batch_size, images.device)
        proprio = self._prepare_state(batch, batch_size, images.device)
        return {
            "input_ids": input_ids,
            "image_input": images,
            "image_mask": image_mask,
            "domain_id": domain_id,
            "proprio": proprio,
        }

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        inputs = self._build_model_inputs(batch)
        targets = self._prepare_action_targets(batch)
        losses = self.model(action=targets, **inputs)
        total_loss = sum(losses.values())

        log_dict = {k: v.detach().item() for k, v in losses.items()}
        log_dict["loss"] = total_loss.detach().item()
        return total_loss, log_dict

    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        inputs = self._build_model_inputs(batch)
        actions = self.model.generate_actions(**inputs, steps=self.config.num_denoising_steps)
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self._get_action_chunk(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

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
        strict: bool = False,
        **kwargs,
    ):
        """
        加载 XVLA 模型权重，具备：
        - 自动为所有键添加 'model.' 前缀
        - 针对应保持随机初始化的层的跳过列表
        """
        import safetensors.torch

        # 第 1 步：加载 config
        # TODO: jadechoghari，修复这个问题
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

        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        # 第 2 步：定位 model.safetensors
        if os.path.isdir(model_id):
            logging.info("Loading weights from local directory")
            model_file = os.path.join(model_id, "model.safetensors")
        else:
            try:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.utils import HfHubHTTPError

                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(f"model.safetensors not found on the Hub at {model_id}") from e

        logging.info(f"Loading checkpoint from {model_file}")
        # 第 3 步：加载 state dict，把以旧的内嵌 Florence-2 模块布局保存的
        # checkpoint 重映射为原生 transformers 布局
        # （openpi model.py 的 `_fix_pytorch_state_dict_keys` / pi0 中有相同模式）
        state_dict = safetensors.torch.load_file(model_file)
        if _is_vendored_florence_state_dict(state_dict):
            logging.info(
                "Detected XVLA checkpoint with the old vendored Florence-2 layout; "
                "remapping keys to the native transformers layout."
            )
            state_dict = _remap_vendored_florence_state_dict(state_dict)
        # safetensors 在保存时会对绑定张量去重：恢复共享/编码器 token 嵌入中
        # 缺失的那个别名
        shared_key = "model.vlm.language_model.shared.weight"
        embed_key = "model.vlm.language_model.encoder.embed_tokens.weight"
        if shared_key in state_dict and embed_key not in state_dict:
            state_dict[embed_key] = state_dict[shared_key]
        elif embed_key in state_dict and shared_key not in state_dict:
            state_dict[shared_key] = state_dict[embed_key]
        # 第 4 步：加载到实例中
        instance.load_state_dict(state_dict, strict=True)
        logging.info("Loaded XVLA checkpoint")
        # 第 5 步：收尾
        # 加载 state dict 后重新应用 dtype
        instance.model._apply_dtype()
        instance.to(config.device)
        instance.eval()
        return instance


def _is_vendored_florence_state_dict(state_dict: dict[str, Tensor], prefix: str = "model.vlm.") -> bool:
    """通过标志性键检测以旧的内嵌（Microsoft 远程代码版）Florence-2
    模块布局保存的 XVLA checkpoint。"""
    return f"{prefix}image_projection" in state_dict or any(
        key.startswith(f"{prefix}language_model.model.") for key in state_dict
    )


def _remap_vendored_florence_state_dict(
    state_dict: dict[str, Tensor], prefix: str = "model.vlm."
) -> dict[str, Tensor]:
    """将 state dict 从内嵌（Microsoft 远程代码版）Florence-2 布局重映射为
    原生 ``transformers.models.florence2`` 布局。

    只有 ``prefix`` 下的键会被改写；其他所有键原样透传。
    """
    vision = re.escape(prefix) + r"vision_tower\."
    block = vision + r"blocks\.(\d+)\.(\d+)\.(spatial_block|channel_block)\."
    new_block = prefix + r"vision_tower.blocks.\1.\2.\3."
    rules: list[tuple[str, str]] = [
        # DaViT stem：ConvEmbed.proj -> Florence2VisionConvEmbed.conv
        (vision + r"convs\.(\d+)\.proj\.", prefix + r"vision_tower.convs.\1.conv."),
        # DaViT blocks：PreNorm/Mlp 包装层在原生实现中被展平
        (block + r"conv1\.fn\.dw\.", new_block + r"conv1."),
        (block + r"conv2\.fn\.dw\.", new_block + r"conv2."),
        (block + r"(window_attn|channel_attn)\.norm\.", new_block + r"norm1."),
        (block + r"(window_attn|channel_attn)\.fn\.", new_block + r"\4."),
        (block + r"ffn\.norm\.", new_block + r"norm2."),
        (block + r"ffn\.fn\.net\.", new_block + r"ffn."),
        # 多模态投影层被移入专门的 projector 模块
        (re.escape(prefix) + r"image_proj_norm\.", prefix + r"multi_modal_projector.image_proj_norm."),
        (
            re.escape(prefix) + r"image_pos_embed\.",
            prefix + r"multi_modal_projector.image_position_embed.",
        ),
        (
            re.escape(prefix) + r"visual_temporal_embed\.",
            prefix + r"multi_modal_projector.visual_temporal_embed.",
        ),
        # 语言模型：Florence2LanguageForConditionalGeneration.model -> BartModel
        (re.escape(prefix) + r"language_model\.model\.", prefix + r"language_model."),
    ]

    remapped: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key == f"{prefix}language_model.final_logits_bias":
            # 内嵌语言模型中仅用于生成的 buffer；原生 BartModel 没有这个东西
            continue
        if key == f"{prefix}image_projection":
            # 内嵌版：形状为 (embed_dim, projection_dim) 的 nn.Parameter，按 `x @ p` 使用；
            # 原生版：nn.Linear(embed_dim, projection_dim, bias=False)，其权重是前者的转置
            remapped[f"{prefix}multi_modal_projector.image_projection.weight"] = value.transpose(
                0, 1
            ).contiguous()
            continue
        new_key = key
        for pattern, replacement in rules:
            new_key, count = re.subn(pattern, replacement, new_key, count=1)
            if count:
                break
        remapped[new_key] = value

    return remapped


def pad_tensor_along_dim(tensor: Tensor, target_len: int, dim: int = 1) -> Tensor:
    current_len = tensor.size(dim)
    if current_len == target_len:
        return tensor
    if current_len > target_len:
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(0, target_len)
        return tensor[tuple(slices)]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - current_len
    pad_tensor = tensor.new_zeros(pad_shape)
    return torch.cat([tensor, pad_tensor], dim=dim)
