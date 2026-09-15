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
from collections import deque
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.utils import log_model_loading_keys, populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import get_autocast_context, resolve_safetensors_device
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModel, AutoVideoProcessor
else:
    AutoModel = None
    AutoVideoProcessor = None

from .action_head import VLAJEPAActionHead
from .configuration_vla_jepa import VLAJEPAConfig
from .qwen_interface import Qwen3VLInterface
from .world_model import ActionConditionedVideoPredictor

# ============================================================================
# 原生 VLA-JEPA 模型 - 遵循 starVLA 原始的 VLA_JEPA.py 实现
# ============================================================================


class VLAJEPAModel(nn.Module):
    """
    原生 VLA-JEPA 模型，遵循 starVLA 原始的 VLA_JEPA.py。

    组成部分：
      - Qwen3-VL：视觉-语言主干网络，用于生成融合嵌入
      - DiT-B：flow-matching 动作头，用于预测未来动作
      - V-JEPA：世界模型，用于预测视频帧

    输入为保留在模型设备上的批量张量
      - images: List[List[Tensor [C, H, W]]]（float，[0,1]）— 每个样本、每个视角一个（Qwen messages）
      - instructions: List[str]
      - videos: Tensor [B, V, T, C, H, W]（float，[0,1]，仅世界模型使用）
      - actions: Tensor [B, T, action_dim]（可选，仅训练时使用）
      - state: Tensor [B, 1, state_dim]（可选）
      - action_is_pad: Tensor [B, T]（可选）
    """

    def __init__(self, config: VLAJEPAConfig) -> None:
        super().__init__()
        require_package("transformers", extra="vla_jepa")
        self.config = config

        # 视觉-语言主干网络
        self.qwen = Qwen3VLInterface(config)

        # 扩展分词器以加入特殊动作 token
        self.action_tokens, self.action_token_ids, self.embodied_action_token_id = (
            self.qwen.expand_tokenizer()
        )
        self.register_buffer(
            "_action_token_ids_t",
            torch.tensor(self.action_token_ids, dtype=torch.long),
            persistent=False,
        )

        # 动作头（flow-matching DiT）
        self.action_model = VLAJEPAActionHead(config, cross_attention_dim=self.qwen.model.config.hidden_size)

        # JEPA 世界模型组件
        if config.enable_world_model:
            self.video_encoder = AutoModel.from_pretrained(
                config.jepa_encoder_name,
                torch_dtype=self.qwen._get_torch_dtype(config.torch_dtype),
            )
            self.video_processor = AutoVideoProcessor.from_pretrained(config.jepa_encoder_name)
            num_views = config.num_world_model_views
            tubelet_size = self.video_encoder.config.tubelet_size
            image_size = getattr(self.video_encoder.config, "image_size", None)
            if image_size is None:
                first_image_shape = next(iter(config.image_features.values())).shape
                image_size = first_image_shape[-1]
            self.video_predictor = ActionConditionedVideoPredictor(
                num_frames=config.num_video_frames // tubelet_size,
                img_size=(image_size, image_size),
                patch_size=16,
                tubelet_size=1,
                embed_dim=self.video_encoder.config.hidden_size * num_views,
                action_embed_dim=self.qwen.model.config.hidden_size,
                predictor_embed_dim=self.video_encoder.config.hidden_size,
                depth=config.predictor_depth,
                num_heads=config.predictor_num_heads,
                mlp_ratio=config.predictor_mlp_ratio,
                num_action_tokens_per_step=config.num_action_tokens_per_timestep,
                dropout=config.predictor_dropout,
            )
        else:
            self.video_encoder = None
            self.video_processor = None
            self.video_predictor = None

        if config.freeze_qwen:
            self.qwen.requires_grad_(False)

        # 构建 prompt 占位符。
        # 如果可以拿到编码器实际的 tubelet_size（已启用世界模型）就使用它，
        # 否则回退到 config 中的配置。
        _tubelet_size = (
            self.video_encoder.config.tubelet_size
            if config.enable_world_model
            else self.config.jepa_tubelet_size
        )
        num_action_prompt_steps = self.config.num_video_frames // _tubelet_size - 1
        self.replace_prompt = "".join(
            token * self.config.num_action_tokens_per_timestep
            for token in self.action_tokens[:num_action_prompt_steps]
        )
        self.embodied_replace_prompt = (
            self.config.embodied_action_token * self.config.num_embodied_action_tokens_per_instruction
        )

    def _qwen_last_decoder_hidden(self, qwen_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """返回最后一个 decoder 层在最终 RMSNorm 之前的隐藏状态。

        模型训练时使用的是最后一个 block 在 RMSNorm 之前的输出，但在 transformers 5.x 中
        `hidden_states[-1]` 是 norm 之后的结果，因此改为 hook `language_model.layers[-1]`。

        这里调用的是内部的 `Qwen3VLModel`，而不是 `Qwen3VLForConditionalGeneration` 包装器；
        后者的 forward 会针对 151936 大小的词表构建并丢弃整条序列的 logits（batch 8、bf16 时
        约 3.4 GB）。包装器仍保留为 `self.qwen.model`，以便 `lm_head` 保持其 checkpoint 键名；
        只有这条路径会跳过它。
        """
        captured: list[torch.Tensor] = []

        def _hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h)

        last_layer = self.qwen.model.model.language_model.layers[-1]
        handle = last_layer.register_forward_hook(_hook)
        try:
            self.qwen.model.model(**qwen_inputs)
        finally:
            handle.remove()

        return captured[0]  # [B, seq_len, H]

    # ---- 原生 VLA-JEPA forward（遵循原始 VLA_JEPA.py）----

    def _encode_qwen(
        self, images: list[list[Tensor]], instructions: list[str], *, need_action_tokens: bool
    ) -> tuple[Tensor, Tensor | None]:
        """运行 Qwen，并收集具身动作（embodied-action）token（以及可选的动作 token）的隐藏状态。"""
        qwen_inputs = self.qwen.build_inputs(
            images=images,
            instructions=instructions,
            action_prompt=self.replace_prompt,
            embodied_prompt=self.embodied_replace_prompt,
        )
        input_ids = qwen_inputs["input_ids"]
        embodied_idx = (input_ids == self.embodied_action_token_id).nonzero(as_tuple=True)
        action_idx = None
        if need_action_tokens:
            action_mask = torch.isin(input_ids, self._action_token_ids_t)
            action_idx = action_mask.nonzero(as_tuple=True)

        device_type = next(self.parameters()).device.type
        with get_autocast_context(device_type, torch.bfloat16):
            last_hidden = self._qwen_last_decoder_hidden(qwen_inputs)  # [B, seq_len, H]
            b, _, h = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_idx[0], embodied_idx[1], :].view(b, -1, h)
            action_tokens = (
                last_hidden[action_idx[0], action_idx[1], :].view(b, -1, h)
                if action_idx is not None
                else None
            )
        return embodied_action_tokens, action_tokens

    def _causal_video_embeddings(self, video_pixels: Tensor, tubelet_size: int, num_positions: int) -> Tensor:
        """仅依据各时间位置自身的原始帧前缀来编码前 `num_positions` 个时间位置。

        对完整 clip 只做一次 V-JEPA2 前向时，双向注意力会让未来帧泄漏到每个位置的嵌入中，
        包括用作预测器输入的上下文位置（#4153）。为每个上下文位置分别做一次仅前缀的前向，
        可以让位置 i 看不到它之后的帧，代价是需要调用 `num_positions` 次编码器而不是一次。
        """
        positions = []
        for position in range(num_positions):
            prefix = self.video_encoder.get_vision_features(
                pixel_values_videos=video_pixels[:, : (position + 1) * tubelet_size]
            )
            tokens_per_position = prefix.shape[1] // (position + 1)
            positions.append(prefix[:, -tokens_per_position:])
        return torch.cat(positions, dim=1)

    @staticmethod
    def _merge_views(embeddings: Tensor, b: int, v: int) -> Tensor:
        """合并各视角的特征：[B*V, N, H] -> [B, N, V*H]。

        行的排列以 view 为最快变化维度，因为 `videos.reshape(b * v, ...)` 按行优先（row-major）
        展平 (B, V)。而使用 `chunk(chunks=v, dim=0)` + `cat(dim=2)` 的合并方式假定 view 为最慢
        变化维度，因此会把*不同*样本的特征拼接在一起——形状上仍然合法，所以在 b > 1 时会静默出错。
        """
        n_tokens, hidden = embeddings.shape[1], embeddings.shape[2]
        return embeddings.reshape(b, v, n_tokens, hidden).permute(0, 2, 1, 3).reshape(b, n_tokens, v * hidden)

    def _world_model_loss(self, videos: Tensor, action_tokens: Tensor, reduction: str = "mean") -> Tensor:
        """JEPA 编码 + 预测器 L1 损失。`videos` 形状为 [B, V, T, C, H, W]，float，取值 [0, 1]。

        `reduction="none"` 返回逐样本损失 (B,)，用于样本加权（RA-BC）；
        "mean" 返回标量损失。
        """
        # 对齐世界模型所需的视角数：用第一个视角填充不足的部分，或裁掉多余的视角。
        num_views = self.config.num_world_model_views
        if videos.shape[1] < num_views:
            missing = num_views - videos.shape[1]
            videos = torch.cat([videos, videos[:, :1].repeat(1, missing, 1, 1, 1, 1)], dim=1)
        elif videos.shape[1] > num_views:
            videos = videos[:, :num_views]

        b, v, t_frames, c, h_img, w_img = videos.shape
        flat = videos.reshape(b * v, t_frames, c, h_img, w_img)
        # 在设备上使用快速（torchvision）视频处理器，do_rescale=False（帧已经在 [0, 1] 范围内）。
        video_pixels = self.video_processor(
            videos=list(flat),
            return_tensors="pt",
            device=self.video_encoder.device,
            do_rescale=False,
        )["pixel_values_videos"]  # [B*V, T, C, H, W]

        tubelet_size = self.video_encoder.config.tubelet_size
        with torch.no_grad():
            video_embeddings = self.video_encoder.get_vision_features(pixel_values_videos=video_pixels)
            video_embeddings = self._merge_views(video_embeddings, b, v)

        # num_video_frames 个原始帧 → 经 tubelet 压缩后得到 t_enc_total 个时间位置
        t_enc_total = self.config.num_video_frames // tubelet_size
        if t_enc_total < 2:
            zero_shape = (video_embeddings.shape[0],) if reduction == "none" else ()
            return torch.zeros(zero_shape, device=video_embeddings.device)

        # 错开一位的 JEPA 划分：input_states = 位置 0..T-2，gt_states = 位置 1..T-1
        t_enc_ctx = t_enc_total - 1
        tokens_per_frame = video_embeddings.shape[1] // t_enc_total
        if self.config.causal_world_model_context:
            # 上面那次共享前向会让双向注意力把未来帧泄漏到用作预测器输入的上下文位置中（#4153）。
            # 因此改为以因果方式重新计算 input_states；gt_states 仍保留全序列前向的嵌入
            # （目标编码器能看到完整上下文是没有问题的）。
            with torch.no_grad():
                input_states = self._causal_video_embeddings(video_pixels, tubelet_size, t_enc_ctx)
                input_states = self._merge_views(input_states, b, v)
        else:
            input_states = video_embeddings[:, : tokens_per_frame * t_enc_ctx, :]
        gt_states = video_embeddings[:, tokens_per_frame:, :]

        expected_actions = t_enc_ctx * self.config.num_action_tokens_per_timestep
        if action_tokens.shape[1] < expected_actions:
            pad = action_tokens[:, -1:].repeat(1, expected_actions - action_tokens.shape[1], 1)
            action_tokens = torch.cat([action_tokens, pad], dim=1)

        predicted_states = self.video_predictor(
            input_states.float(), action_tokens[:, :expected_actions].float()
        )
        if reduction == "none":
            # 逐样本损失 (B,)：对所有非 batch 维度（tokens、feature）取均值。
            elementwise = F.l1_loss(predicted_states, gt_states.float(), reduction="none")
            return elementwise.mean(dim=tuple(range(1, elementwise.ndim)))
        return F.l1_loss(predicted_states, gt_states.float(), reduction="mean")

    def _action_loss(
        self,
        embodied_action_tokens: Tensor,
        actions: Tensor,
        state: Tensor | None,
        action_is_pad: Tensor | None,
        reduction: str = "mean",
    ) -> Tensor:
        """Flow-matching 动作头损失，按 `repeated_diffusion_steps` 重复计算。

        `reduction="none"` 返回逐样本损失 (B,)——`repeated_diffusion_steps` 次独立的噪声采样
        会按原始样本平均回去——用于 RA-BC 加权。
        """
        device_type = next(self.parameters()).device.type
        with get_autocast_context(device_type, torch.float32):
            r = self.config.repeated_diffusion_steps
            horizon = self.config.chunk_size
            b = embodied_action_tokens.shape[0]
            actions_target = actions[:, -horizon:, :].to(torch.float32).repeat(r, 1, 1)
            embodied = embodied_action_tokens.repeat(r, 1, 1)
            state_rep = state.to(embodied_action_tokens.dtype).repeat(r, 1, 1) if state is not None else None
            pad_rep = action_is_pad[:, -horizon:].repeat(r, 1) if action_is_pad is not None else None
            loss = self.action_model(embodied, actions_target, state_rep, pad_rep, reduction=reduction)
            if reduction == "none":
                # `.repeat(r, 1, 1)` 的排布为 [rep0(b0..b_{B-1}), rep1(...), ...] → (r, B)；对各次重复取均值。
                return loss.view(r, b).mean(dim=0)
            return loss

    def forward(
        self,
        images: list[list[Tensor]],
        instructions: list[str],
        videos: Tensor | None = None,
        actions: Tensor | None = None,
        state: Tensor | None = None,
        action_is_pad: Tensor | None = None,
        reduction: str = "mean",
    ) -> dict[str, Tensor]:
        """原生 forward：Qwen 编码 → 可选的世界模型损失 → 可选的动作头损失。

        `reduction="none"` 会让两个损失项都变为逐样本形式 (B,)，用于 RA-BC 加权；"mean"
        返回标量损失。
        """
        embodied_action_tokens, action_tokens = self._encode_qwen(
            images, instructions, need_action_tokens=self.config.enable_world_model
        )

        if self.config.enable_world_model and videos is not None:
            wm_loss = self._world_model_loss(videos, action_tokens, reduction=reduction)
        else:
            zero_shape = (embodied_action_tokens.shape[0],) if reduction == "none" else ()
            wm_loss = torch.zeros(zero_shape, device=embodied_action_tokens.device)

        if actions is None:
            return {"wm_loss": wm_loss}

        action_loss = self._action_loss(
            embodied_action_tokens, actions, state, action_is_pad, reduction=reduction
        )
        return {"action_loss": action_loss, "wm_loss": wm_loss * self.config.world_model_loss_weight}

    # ---- 原生 predict_action（遵循原始 VLA_JEPA.predict_action）----

    @torch.no_grad()
    def predict_action(
        self,
        images: list[list[Tensor]],
        instructions: list[str],
        state: Tensor | None = None,
    ) -> Tensor:
        """预测一个动作块。`images` 为逐样本、逐视角的 float [0,1]、形状 [C, H, W] 的张量。"""
        if self.config.resize_images_to is not None:
            height, width = self.config.resize_images_to
            images = [
                [F.interpolate(img[None], size=(height, width), mode="area")[0] for img in views]
                for views in images
            ]

        embodied_action_tokens, _ = self._encode_qwen(images, instructions, need_action_tokens=False)
        return self.action_model.predict_action(
            embodied_action_tokens.float(), state.float() if state is not None else None
        )


# ============================================================================
# LeRobot 适配层 - 在 LeRobot 批格式与原生 VLA-JEPA 格式之间进行转换
# ============================================================================


class VLAJEPAPolicy(PreTrainedPolicy):
    """
    VLA-JEPA 的 LeRobot 适配器。

    将 LeRobot 的标准批格式（dict[str, Tensor]）转换为原生模型所需的批量张量
    （所有数据都保留在设备上），调用原生模型，再把输出转换回 LeRobot 格式。
    """

    config_class = VLAJEPAConfig
    name = "vla_jepa"

    def __init__(self, config: VLAJEPAConfig, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        # 数据集维度的推导位于 `VLAJEPAConfig.set_dataset_feature_metadata` 中（由 `make_policy`
        # 在此之前调用）：这样可以避免 `__init__` 修改一个不属于它的 config，同时也让推导出来的
        # 维度对 processor 工厂可见。
        self.model = VLAJEPAModel(config)
        self.reset()

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    # ---- 格式转换：LeRobot → 原生 ----

    def _prepare_model_inputs(self, batch: dict[str, Tensor], training=True) -> dict[str, Any]:
        """将 LeRobot 批转换为模型所需的、保留在设备上的批量输入。

        LeRobot 格式：
            batch = {
                "observation.images.<key>": Tensor [B, C, H, W] 或 [B, T, C, H, W],
                "observation.state": Tensor [B, state_dim] 或 [B, T, state_dim],
                "action": Tensor [B, chunk_size, action_dim],  （仅训练时）
                "task": str | List[str],  （可选的指令）
            }

        返回 `VLAJEPAModel.forward` / `.predict_action` 所需的 kwargs（所有数据都保留在批所在
        设备上，不做逐样本拆分）：`images`（供 Qwen messages 使用的逐样本、逐视角列表）、
        `instructions`，以及在存在时给出的批量 `videos` / `actions` / `state` /
        `action_is_pad`。
        """
        image_keys = list(self.config.image_features.keys())
        if not image_keys:
            raise ValueError("VLAJEPA requires at least one image feature.")
        batch_size = batch[image_keys[0]].shape[0]

        # 每个视角的当前帧图像（[B, C, H, W]）；按样本重新分组以供 Qwen messages 使用。与
        # `predict_action` 一样缩放到 `resize_images_to`，使训练和推理向 Qwen 输入相同的分辨率，
        # 避免原生帧（例如 720x1280）导致视觉塔的 patch 数量爆炸。
        resize_hw = tuple(self.config.resize_images_to) if self.config.resize_images_to else None
        frames = []
        for key in image_keys:
            t = batch[key]
            if t.ndim == 5:  # [B, T, C, H, W] -> 当前观测（delta=0）
                t = t[:, 0]
            px = self.model.qwen.to_pixel_values(t)  # [B, C, H, W]
            if resize_hw is not None and tuple(px.shape[-2:]) != resize_hw:
                px = F.interpolate(px.float(), size=resize_hw, mode="area")
            frames.append(px)
        images = [[frame[b] for frame in frames] for b in range(batch_size)]

        tasks = batch.get("task")
        if tasks is None:
            instructions = ["Execute the robot action."] * batch_size
        elif isinstance(tasks, str):
            instructions = [tasks] * batch_size
        else:
            instructions = list(tasks)

        inputs: dict[str, Any] = {"images": images, "instructions": instructions}

        # Videos [B, V, T, C, H, W] - 仅在训练期间、世界模型需要用到时才组装。
        if self.model.config.enable_world_model and training:
            views = [batch[k].unsqueeze(1) if batch[k].ndim == 4 else batch[k] for k in image_keys]
            # 单个堆叠而成的 [B, V, T, C, H, W] 张量要求所有视角具有相同的空间尺寸，而各个相机
            # 可能不同（基座 480x640 对比腕部 720x1280）。缩放到 `resize_images_to`，否则缩放到
            # 第一个视角的尺寸（对于单一分辨率数据集这是空操作）。vjepa 视频处理器会负责最终缩放到
            # 编码器所需的分辨率。
            cfg = self.model.config
            target_hw = tuple(cfg.resize_images_to) if cfg.resize_images_to else tuple(views[0].shape[-2:])
            resized = []
            for v in views:
                if tuple(v.shape[-2:]) != target_hw:
                    b, t, c = v.shape[0], v.shape[1], v.shape[2]
                    v = F.interpolate(
                        v.reshape(b * t, c, v.shape[3], v.shape[4]).float(),
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(b, t, c, target_hw[0], target_hw[1])
                resized.append(v)
            inputs["videos"] = self.model.qwen.to_pixel_values(torch.stack(resized, dim=1))

        actions = batch.get(ACTION)
        if actions is not None:
            inputs["actions"] = (actions.unsqueeze(1) if actions.ndim == 2 else actions).float()
            if (pad := batch.get("action_is_pad")) is not None:
                inputs["action_is_pad"] = pad

        state = batch.get(OBS_STATE)
        if state is not None:
            if state.ndim > 2:
                # 这里的 delta 是前向的，因此索引 0 才是当前观测，而不是 -1。
                state = state[:, 0, :]
            inputs["state"] = (state.unsqueeze(1) if state.ndim == 2 else state).float()  # [B, 1, dim]

        return inputs

    # ---- LeRobot Policy 接口 ----

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """LeRobot 训练 forward：转换 → 原生 forward → 聚合损失。"""
        native_output = self.model.forward(
            **self._prepare_model_inputs(batch, training=True), reduction=reduction
        )

        ref = next(iter(native_output.values()))
        zero = torch.zeros_like(ref)
        total_loss = native_output.get("action_loss", zero) + native_output.get("wm_loss", zero)
        logs = {k: v.detach().mean().item() for k, v in native_output.items()}
        logs["loss"] = total_loss.detach().mean().item()
        return total_loss, logs

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """LeRobot 推理：转换 → 原生预测 → 以 Tensor 形式返回。"""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        inputs = self._prepare_model_inputs(batch, training=False)
        actions = self.model.predict_action(inputs["images"], inputs["instructions"], inputs.get("state"))
        return actions.to(device=self.config.device, dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """带动作队列缓存的 LeRobot select_action。"""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        reinit_prefixes = model.config.reinit_modules
        if not reinit_prefixes:
            return super()._load_as_safetensor(model, model_file, map_location, strict)

        # 正是 `resolve_safetensors_device` 避免了每个 rank 都把整个 checkpoint 实例化到
        # GPU 0 上：safetensors 会把裸字符串 "cuda" 映射到 cuda:0，而不管
        # torch.cuda.current_device() 是什么，`config.device` 恰好就是这个裸字符串。
        state_dict = load_file(model_file, device=resolve_safetensors_device(map_location))
        current = model.state_dict()

        reinitialized: list[str] = []
        filtered: dict = {}
        for key, value in state_dict.items():
            if key in current and value.shape != current[key].shape:
                if not any(key.startswith(p) for p in reinit_prefixes):
                    raise ValueError(
                        f"Shape mismatch for '{key}' (checkpoint {tuple(value.shape)} vs model "
                        f"{tuple(current[key].shape)}) and its prefix is not in `reinit_modules`."
                    )
                reinitialized.append(
                    f"{key}: checkpoint {tuple(value.shape)} → model {tuple(current[key].shape)}"
                )
            else:
                filtered[key] = value

        if reinitialized:
            logging.warning(
                f"reinit_modules: skipping {len(reinitialized)} tensor(s) with mismatched shapes "
                f"(randomly re-initialised):\n  " + "\n  ".join(reinitialized)
            )

        # 这里有意使用 non-strict：上面那些被重新初始化的张量*本来就应该*缺失。
        # 但 `strict` 仍然需要有意义，因此对其他所有键强制严格检查。
        missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
        if strict:
            reinit_keys = {entry.split(":", 1)[0] for entry in reinitialized}
            unaccounted = [k for k in missing_keys if k not in reinit_keys]
            if unaccounted or unexpected_keys:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {type(model).__name__} with strict=True: "
                    f"missing keys not covered by `reinit_modules` {unaccounted}, "
                    f"unexpected keys {list(unexpected_keys)}."
                )
        log_model_loading_keys(missing_keys, unexpected_keys)
        return model
