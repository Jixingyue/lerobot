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

"""
SmolVLA:

[论文](https://huggingface.co/papers/2506.01844)

由 Hugging Face 设计。

安装 smolvla 的额外依赖：
```bash
pip install -e ".[smolvla]"
```

微调 smolvla 预训练模型（`smolvla_base`）的示例：
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

微调 SmolVLA 的示例。SmolVLA 由一个预训练 VLM
和一个动作专家（action expert）组成。
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

在 LeRobot 训练框架之外使用 smolvla 预训练模型的示例：
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package

from ..common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from ..common.vla_utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad,
)
from ..pretrained import PreTrainedPolicy
from ..rtc.modeling_rtc import RTCProcessor
from ..utils import (
    populate_queues,
)
from .configuration_smolvla import SmolVLAConfig
from .smolvlm_with_expert import SmolVLMWithExpertModel


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # 这可确保输入保持在
    # [−1,1] 范围内，以避免 arcsin 出现无效值
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha 将夹爪位置变换到一个线性空间。下面的代码
    # 会逆转该变换，以与在角度空间中预训练的 smolvla 保持一致。
    #
    # 这些数值来自 Aloha 代码：
    # PUPPET_GRIPPER_POSITION_OPEN、PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # 这是 Interbotix 代码中角度到线性变换的逆变换。
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # 这些常量取自 Interbotix 代码。
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # 归一化到 [0, 1]。
    # 数值 0.4 和 1.5 是在真实的 Trossen 机器人上测量得到的。
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # 将 smolvla 使用的夹爪位置转换为 Aloha 使用的夹爪位置。
    # 注意单位仍然是角度，只是取值范围不同。

    # 数值 0.4 和 1.5 是在真实的 Trossen 机器人上测量得到的。
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # 这些数值来自 Aloha 代码：
    # PUPPET_GRIPPER_JOINT_OPEN、PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # 直接求 gripper_from_angular 函数的逆。
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """对 VLAFlowMatching 模型的封装类，用于在 LeRobot 中进行训练和推理。"""

    config_class = SmolVLAConfig
    name = "smolvla"

    def supports_rtc(self) -> bool:
        return True

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: 策略配置类实例；若为 None，则使用该配置类的默认实例化结果。
        """

        require_package("transformers", extra="smolvla")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """每当环境重置时都应调用此方法。"""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """若配置中启用了 RTC，则初始化 RTC 处理器。"""
        self.rtc_processor = None

        # 如果提供了配置，就创建处理器
        # 即使未启用 RTC，我们仍然可以跟踪去噪数据
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # 处理在模型创建之后调用 init_rtc_processor 的情况
            # 此时需要把 rtc_processor 设置到模型上
            # 在正常的初始化流程中，模型此时尚未创建
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO：检查是否需要这个 for 循环。
        # 背景：实际上 self.queues 只包含 ACTION 字段，而推理时 batch 中并没有 action；
        # 在离线推理的情况下，batch 中才有 action。
        # 这就是为什么如果不加上 k != ACTION 的判断，会因为试图在空容器上 stack 而报错。
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        # 去除填充（unpad）动作
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """根据环境观测选择单个动作。

        该方法封装了 `select_actions`，以便每次返回一个动作供环境中执行。
        它通过队列来管理动作，并且只在队列为空时才调用 `select_actions`。
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` 返回形状为 (batch_size, n_action_steps, action_dim) 的张量，
            # 但队列实际采用的形状是 (n_action_steps, batch_size, *)，因此这里需要转置。
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """执行完整的训练前向传播以计算损失。

        Args:
            batch: 包含观测和动作的训练批次。
            noise: 可选的 flow matching 噪声张量。
            time: 可选的 flow matching 时间张量。
            reduction: 损失的归约方式。可选值：
                - "mean"：返回标量平均损失（默认，向后兼容）
                - "none"：返回形状为 (batch_size,) 的逐样本损失，用于 RA-BC 加权
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # 移除填充
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            # 通过对有效的 (time, action) 条目求平均，返回形状为 (B,) 的逐样本损失
            if actions_is_pad is None:
                per_sample_loss = losses.mean(dim=(1, 2))
            else:
                num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample_loss = losses.sum(dim=(1, 2)) / num_valid
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # 默认：对有效的 (time, action) 条目求平均，返回标量平均损失
            if actions_is_pad is None:
                loss = losses.mean()
            else:
                num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                loss = losses.sum() / num_valid
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def prepare_images(self, batch):
        """对图像应用 SmolVLA 预处理，例如将尺寸调整为 224x224 并通过填充保持宽高比，
        同时按照 SigLIP 的要求将像素范围从 [0.0, 1.0] 转换到 [-1.0, 1.0]。
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # 对批次中存在的图像特征进行预处理
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                # SmolVLA 以 (width, height) 存储目标尺寸，而共享的辅助函数要求 (height, width)。
                img = resize_with_pad(
                    img,
                    self.config.resize_imgs_with_padding[1],
                    self.config.resize_imgs_with_padding[0],
                    pad_value=0,
                )

            # 按 siglip 的要求将范围从 [0,1] 归一化到 [-1,1]
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # 将批次中不存在的图像特征创建为
        # 完全以 0 填充的图像。
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # 翻转关节。
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # 逆转 Aloha 运行时所应用的夹爪变换。
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # 翻转关节。
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # 逆转 Aloha 运行时所应用的夹爪变换。
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # 再次翻转关节。
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # 逆转 Aloha 运行时所应用的夹爪变换。
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """对状态进行填充"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """对动作进行填充"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """返回 SmolVLA 微调所用的默认 PEFT 目标模块。"""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """校验 SmolVLA 的 PEFT 配置。"""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    沿序列维度高效填充张量，使其长度达到 max_len。

    Args:
        tensor (torch.Tensor): 形状为 (B, L, ...) 或 (B, L)。
        max_len (int): 固定的序列长度。
        pad_value (int/float): 填充所用的值。

    Returns:
        torch.Tensor: 形状为 (B, max_len, ...) 或 (B, max_len)。
    """
    b, d = tensor.shape[:2]

    # 创建长度为 max_len 的填充张量，并把已有值拷贝进去
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # 高效的原地拷贝

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    由 Hugging Face 设计。
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # 按需编译模型
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        return sample_noise(shape, device)

    def sample_time(self, bsize, device):
        return sample_time_beta(bsize, device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """用 SigLIP 嵌入图像、用嵌入层嵌入语言 token，为
        SmolVLM transformer 的处理做准备。
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # 对图像嵌入进行归一化
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # 对语言嵌入进行归一化
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # 设置注意力掩码，使图像和语言输入不会关注状态或动作
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """嵌入状态、带噪动作和时间步，为 Expert Gemma 的处理做准备。"""
        embs = []
        pad_masks = []
        att_masks = []

        # 使用 MLP 融合时间步和动作信息
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # 使用正余弦位置编码嵌入时间步，敏感范围为 [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish 即 silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # 添加到输入 token 中
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # 设置注意力掩码，使图像、语言和状态输入不会关注动作 token
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """执行完整的训练前向传播并计算损失（batch_size x num_steps x num_motors）"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # 沿用原始 openpi 代码，将注意力输出提升为高精度类型
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """执行完整的推理前向传播并计算动作（batch_size x num_steps x num_motors）"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # 计算图像和语言的键值缓存
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
        )
        num_steps = self.config.num_steps

        return euler_integrate(
            lambda input_x_t, current_timestep: self.denoise_step(
                x_t=input_x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=current_timestep,
            ),
            noise,
            num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """在给定时间步对噪声 `x_t` 施加一次去噪步骤。"""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
        )
        if past_key_values is not None:
            # 自注意力层会原地追加后缀的 K/V；这里为下一步恢复前缀。
            past_key_values.crop(prefix_len)
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
