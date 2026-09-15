# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
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

"""LingBot-VA 策略：构建在 Wan2.2 技术栈上的自回归视频-动作世界模型。

该采样循环是对上游流式服务器（``wan_va/wan_va_server.py``）和 LIBERO 客户端
（``evaluation/libero/client.py``）的忠实重新实现，并适配到 LeRobot 的
``select_action`` 接口：

  * 可训练的双流 transformer 作为子模块持有，并随单个
    ``model.safetensors`` 检查点一起保存/加载；
  * 冻结的 Wan VAE + UMT5 文本编码器 + tokenizer 从
    ``config.wan_pretrained_path`` *惰性拉取*（不打包在内），从而让 LeRobot
    检查点保持小巧；
  * ``predict_action_chunk`` 运行一个自回归分块（先视频流后动作流，各自带
    CFG 和独立的 flow-matching 调度器），并更新 KV cache；
  * ``select_action`` 逐步消费动作队列，并记录真实观测到的关键帧；当队列
    需要重新填充时，这些关键帧会被反馈回 KV cache。

注意：流式路径是为单环境评估（``--eval.batch_size=1``）编写的。
"""

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from .configuration_lingbot_va import LingBotVAConfig
from .utils import (
    FlowMatchScheduler,
    WanTransformer3DModel,
    WanVAEStreamingWrapper,
    _sample_timestep_id,
    _torch_dtype,
    clean_prompt,
    data_seq_to_patch,
    denormalize_latents,
    get_mesh_id,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)


class LingBotVAPolicy(PreTrainedPolicy):
    """LingBot-VA 自回归视频-动作世界模型的 LeRobot 封装。"""

    config_class = LingBotVAConfig
    name = "lingbot_va"

    def __init__(self, config: LingBotVAConfig, **kwargs):
        require_package("diffusers", extra="lingbot_va")
        require_package("transformers", extra="lingbot_va")
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.dtype = _torch_dtype(config.dtype)

        # 可训练的双流 transformer（LeRobot 检查点中唯一保存的子模块）。
        self.transformer = WanTransformer3DModel(
            patch_size=tuple(config.patch_size),
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            action_dim=config.action_dim,
            text_dim=config.text_dim,
            freq_dim=config.freq_dim,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            rope_max_seq_len=config.rope_max_seq_len,
            attn_mode=config.attn_mode,
        )
        # transformer 以 config.dtype（bf16）运行；norm/modulation 路径在内部上转为 fp32。
        self.transformer = self.transformer.to(self.dtype)

        # 冻结模块存放在 nn.Module 注册表之外（普通 dict），因此它们既不会被
        # 保存到 model.safetensors，也不会被 ``.to()`` 移动。它们在第一次运行
        # 推理时从 ``config.wan_pretrained_path`` 惰性加载。
        self._frozen: dict = {}

        self.last_predicted_frames: Tensor | None = None
        self.last_predicted_latents: Tensor | None = None
        self.reset()

    # 冻结模块的惰性加载（VAE + UMT5 + tokenizer）
    def _ensure_frozen_modules(self):
        if self._frozen:
            return
        path = self.config.wan_pretrained_path
        device = self.config.device

        # 冻结模块总是位于 ``vae/``、``text_encoder/`` 和 ``tokenizer/``
        # 子目录中——无论是已发布的 diffusers 风格 HF 仓库，还是本地的
        # ``--bundle-frozen`` 输出目录都是如此。``from_pretrained(path, subfolder=...)``
        # 对 HF 仓库 id 或本地目录都能正确解析它们。
        vae = load_vae(path, torch_dtype=self.dtype, torch_device=device, subfolder="vae")
        # UMT5-XXL 文本编码器（约 11 GB）每个 episode 只运行一次；将其放在
        # 独立的（默认为 CPU）设备上，以便 5B 的 transformer + VAE 能放进单张 GPU。
        text_encoder = load_text_encoder(
            path,
            torch_dtype=self.dtype,
            torch_device=self.config.text_encoder_device,
            subfolder="text_encoder",
        )
        tokenizer = load_tokenizer(path, subfolder="tokenizer")
        self._frozen = {
            "vae": vae.eval(),
            "streaming_vae": WanVAEStreamingWrapper(vae),
            "text_encoder": text_encoder.eval(),
            "tokenizer": tokenizer,
        }
        # RoboTwin 的 T 形布局会通过第二个流式 VAE（独立的因果缓存）编码
        # 半分辨率腕部相机，与全分辨率头部相机并行。
        if self.config.camera_layout == "robotwin_tshape":
            vae_half = load_vae(path, torch_dtype=self.dtype, torch_device=device, subfolder="vae")
            self._frozen["streaming_vae_half"] = WanVAEStreamingWrapper(vae_half.eval())

    @property
    def _vae(self):
        return self._frozen["vae"]

    @property
    def _streaming_vae(self):
        return self._frozen["streaming_vae"]

    # PreTrainedPolicy API
    def get_optim_params(self) -> dict:
        # 只有 transformer 可训练；VAE / 文本编码器保持冻结（存放在 nn.Module
        # 注册表之外）。使用 PEFT/LoRA 时，这里自然只返回 adapter 参数。
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def reset(self):
        """重置所有逐 episode 的流式状态（KV cache、队列、帧计数器）。"""
        cfg = self.config
        self._action_queue: deque = deque(maxlen=cfg.n_action_steps)
        self._obs_buffer: list = []  # 本分块内观测到的原始关键帧 obs（每个环境子步一个）
        self._executed_actions: Tensor | None = (
            None  # 上一分块的动作（模型归一化空间），用于 KV 反馈
        )
        self._started = False  # 第一次 select_action 调用将该 obs 作为条件帧
        self._exec_step = 0  # 当前分块内正在执行的动作索引
        self._prev_j = 0  # 上一个已执行动作在预测帧内的子步索引
        # 每执行 ``action_per_frame / temporal_downsample`` 个子步采样一个关键帧，
        # 使得每个分块恰好有 ``frame_chunk_size * temporal_downsample`` 帧被 VAE 编码
        # （Wan2.2 VAE 的时间下采样倍率为 4 -> 得到 ``frame_chunk_size`` 个潜变量帧）。
        self._keyframe_stride = max(1, cfg.action_per_frame // 4)
        self._frame_st_id = 0
        self._first_chunk = True
        self._prompt: str | None = None
        self._prompt_embeds = None
        self._negative_prompt_embeds = None
        self.last_predicted_frames = None
        self.last_predicted_latents = None
        self._use_cfg = (cfg.guidance_scale > 1) or (cfg.action_guidance_scale > 1)
        # 两个相互独立的 flow-matching 调度器（视频潜变量流 + 动作流）。
        self._scheduler = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)
        self._action_scheduler = FlowMatchScheduler(
            shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self._scheduler.set_timesteps(1000, training=True)
        self._action_scheduler.set_timesteps(1000, training=True)
        self._cache_initialised = False
        # 若 transformer 已构建，则清空其上的 KV cache。
        if hasattr(self, "transformer"):
            self.transformer.clear_cache("pos")
        # 在 episode 之间重置因果流式 VAE 的 feat 缓存（对应上游的 ``_reset``）。
        # 否则编码器会延续上一个 episode 的时序状态，导致下一个 episode 首次
        # 编码时潜变量帧的数量出错。
        if self._frozen:
            self._frozen["streaming_vae"].clear_cache()
            if "streaming_vae_half" in self._frozen:
                self._frozen["streaming_vae_half"].clear_cache()

    # 训练（flow-matching 双流损失）。需要 attn_mode="flex"。
    def _ensure_train_schedulers(self):
        if getattr(self, "_train_sched_latent", None) is None:
            cfg = self.config
            self._train_sched_latent = FlowMatchScheduler(
                shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True
            )
            self._train_sched_latent.set_timesteps(1000, training=True)
            self._train_sched_action = FlowMatchScheduler(
                shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True
            )
            self._train_sched_action.set_timesteps(1000, training=True)

    @torch.no_grad()
    def _add_noise_stream(self, latent, scheduler, action_mask, action_mode, noisy_cond_prob):
        """对单个流做 flow-matching 加噪（移植自上游 ``Trainer._add_noise``）。"""
        device = latent.device
        b, _c, f, _h, _w = latent.shape
        p = self.config.patch_size
        patch_f, patch_h, patch_w = (1, 1, 1) if action_mode else (p[0], p[1], p[2])

        ts_ids = _sample_timestep_id(f, num_train_timesteps=scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = scheduler.timesteps[ts_ids].to(device)
        noisy_latents = scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = scheduler.training_target(latent, noise, timesteps)

        grid_id = (
            get_mesh_id(
                latent.shape[-3] // patch_f,
                latent.shape[-2] // patch_h,
                latent.shape[-1] // patch_w,
                t=1 if action_mode else 0,
                f_w=1,
                f_shift=0,
                action=action_mode,
            )
            .to(device)[None]
            .repeat(b, 1, 1)
        )

        if torch.rand(1).item() < noisy_cond_prob:
            cond_ids = _sample_timestep_id(
                f, min_timestep_bd=0.5, max_timestep_bd=1.0, num_train_timesteps=scheduler.num_train_timesteps
            )
            cond_noise = torch.zeros_like(latent).normal_()
            cond_timesteps = scheduler.timesteps[cond_ids].to(device)
            latent = scheduler.add_noise(latent, cond_noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents = noisy_latents * action_mask.float()
            targets = targets * action_mask.float()
            latent = latent * action_mask.float()

        return {
            "timesteps": timesteps[None].repeat(b, 1),
            "noisy_latents": noisy_latents,
            "targets": targets,
            "latent": latent,
            "cond_timesteps": cond_timesteps[None].repeat(b, 1),
            "grid_id": grid_id,
        }

    def _flow_matching_loss(self, input_dict, pred):
        """双流 flow-matching 损失（移植自上游 ``Trainer.compute_loss``）。"""
        latent_pred, action_pred = pred
        ld, ad = input_dict["latent_dict"], input_dict["action_dict"]
        action_pred = rearrange(action_pred, "b (f n) c -> b c f n 1", f=ad["targets"].shape[-3])
        latent_pred = data_seq_to_patch(
            self.config.patch_size,
            latent_pred,
            ld["targets"].shape[-3],
            ld["targets"].shape[-2],
            ld["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )
        bn, fn = ld["timesteps"].shape
        lw = self._train_sched_latent.training_weight(ld["timesteps"].flatten()).reshape(bn, fn)
        aw = self._train_sched_action.training_weight(ad["timesteps"].flatten()).reshape(bn, fn)

        latent_loss = F.mse_loss(latent_pred.float(), ld["targets"].float().detach(), reduction="none")
        latent_loss = (
            (latent_loss * lw[:, None, :, None, None]).permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        )
        latent_loss = (latent_loss.sum(dim=1) / (torch.ones_like(latent_loss).sum(dim=1) + 1e-6)).mean()

        amask = ad["actions_mask"].float()
        action_loss = F.mse_loss(action_pred.float(), ad["targets"].float().detach(), reduction="none")
        action_loss = (
            (action_loss * aw[:, None, :, None, None] * amask).permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        )
        amask_f = amask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_loss = (action_loss.sum(dim=1) / (amask_f.sum(dim=1) + 1e-6)).mean()
        return latent_loss, action_loss

    def training_loss_from_streams(self, latents, actions, actions_mask, text_emb):
        """在已准备好的潜变量/动作/文本嵌入上计算核心双流训练损失。

        ``latents``：``[B, in_channels, F, h, w]``（归一化后的视频潜变量）。
        ``actions`` / ``actions_mask``：``[B, action_dim, F, action_per_frame, 1]``。
        ``text_emb``：``[B, seq_len, text_dim]``。返回
        ``(loss, {latent_loss, action_loss})``。
        """
        if self.config.attn_mode != "flex":
            raise ValueError(
                "LingBot-VA training requires attn_mode='flex' (block-causal flow-matching masks). "
                "Load/convert the policy with --policy.attn_mode=flex for training/fine-tuning."
            )
        self._ensure_train_schedulers()
        latent_dict = self._add_noise_stream(
            latents, self._train_sched_latent, action_mask=None, action_mode=False, noisy_cond_prob=0.5
        )
        action_dict = self._add_noise_stream(
            actions, self._train_sched_action, action_mask=actions_mask, action_mode=True, noisy_cond_prob=0.0
        )
        latent_dict["text_emb"] = text_emb
        action_dict["text_emb"] = text_emb
        action_dict["actions_mask"] = actions_mask
        input_dict = {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": int(torch.randint(1, 5, (1,)).item()),
            "window_size": int(torch.randint(4, 65, (1,)).item()),
        }
        pred = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self._flow_matching_loss(input_dict, pred)
        loss = latent_loss + action_loss
        return loss, {"latent_loss": latent_loss.item(), "action_loss": action_loss.item()}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """训练前向：双流 flow-matching 损失。

        从 LeRobot batch 构建（视频潜变量、动作、文本）训练流（对相机帧做
        VAE 编码、对任务做 UMT5 编码），然后计算 flow-matching 双流损失。
        要求策略以 ``attn_mode='flex'`` 构建。
        """
        self._ensure_frozen_modules()
        latents, actions, actions_mask, text_emb = self._build_training_streams(batch)
        return self.training_loss_from_streams(latents, actions, actions_mask, text_emb)

    @torch.no_grad()
    def _build_training_streams(self, batch):
        """从 LeRobot 训练 batch 构建（latents, actions, actions_mask, text_emb）。

        ``obs_cam_keys`` 对应的相机帧应为时序片段 ``[B, C, T, H, W]``（或
        ``[B, T, C, H, W]``）；它们经 VAE 编码为
        ``F = T / temporal_downsample`` 个潜变量帧。动作
        ``[B, F*action_per_frame, n_used]`` 会被散布到模型的 ``action_dim`` 空间中。
        """
        cfg = self.config
        device = cfg.device
        # 文本嵌入
        task = batch.get("task")
        if isinstance(task, str):
            task = [task]
        text_emb = self._get_t5_prompt_embeds(list(task), cfg.max_sequence_length)

        # 视频潜变量（对相机片段做 VAE 编码）
        latents = self._encode_training_latents(batch)

        # 动作 -> [B, action_dim, F, action_per_frame, 1]
        act = batch[ACTION].to(device)  # [B, F*apf, n_used]
        b = act.shape[0]
        used = cfg.used_action_channel_ids
        apf, fc = cfg.action_per_frame, cfg.frame_chunk_size
        act = act[:, : fc * apf].reshape(b, fc, apf, len(used)).permute(0, 3, 1, 2)  # [B, n_used, F, apf]
        full = act.new_zeros(b, cfg.action_dim, fc, apf)
        idx = torch.as_tensor(used, device=device)
        full[:, idx] = act
        actions = full.unsqueeze(-1).to(self.dtype)  # [B, action_dim, F, apf, 1]
        mask = torch.zeros(cfg.action_dim, device=device, dtype=self.dtype)
        mask[idx] = 1.0
        actions_mask = mask.view(1, -1, 1, 1, 1).expand_as(actions)
        return latents, actions, actions_mask, text_emb

    @torch.no_grad()
    def _encode_training_latents(self, batch) -> Tensor:
        """将逐相机的训练片段 VAE 编码为归一化视频潜变量 [B, C, F, h, w]。"""
        vae_device = next(self._vae.parameters()).device

        def _clip(key):
            x = batch[key].to(vae_device)
            if x.dim() == 4:  # [B, C, H, W] -> 单帧片段
                x = x.unsqueeze(2)
            elif x.shape[1] not in (1, 3) and x.shape[2] in (1, 3):  # [B, T, C, H, W] -> [B, C, T, H, W]
                x = x.permute(0, 2, 1, 3, 4)
            return x.contiguous()

        def _encode(x, size):
            b, c, t = x.shape[:3]
            x = F.interpolate(x.flatten(0, 1).float(), size=size, mode="bilinear", align_corners=False)
            x = (x.view(b, c, t, *size) * 2.0 - 1.0).to(self.dtype)
            mu = self._vae.encode(x).latent_dist.mode()  # [B, z_dim, F, h, w]
            mean = torch.tensor(self._vae.config.latents_mean).view(1, -1, 1, 1, 1).to(mu.device)
            inv_std = (1.0 / torch.tensor(self._vae.config.latents_std)).view(1, -1, 1, 1, 1).to(mu.device)
            return ((mu.float() - mean) * inv_std).to(mu)

        keys = self.config.obs_cam_keys
        if self.config.camera_layout == "robotwin_tshape":
            h, w = self.config.height, self.config.width
            head = _encode(_clip(keys[0]), (h, w))
            left = _encode(_clip(keys[1]), (h // 2, w // 2))
            right = _encode(_clip(keys[2]), (h // 2, w // 2))
            return torch.cat([torch.cat([left, right], dim=-1), head], dim=-2).to(self.config.device)
        per_cam = [_encode(_clip(k), (self.config.height, self.config.width)) for k in keys]
        return torch.cat(per_cam, dim=-1).to(self.config.device)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """返回单个动作，并在需要时重新填充分块（同时反馈观测到的关键帧）。

        对应上游 LIBERO 客户端循环（``evaluation/libero/client.py``）：第一个
        obs 是条件帧；此后产生的每个观测都作为关键帧缓存，当分块的动作全部
        执行完后，在预测下一个分块之前，将缓存的帧 + 已执行动作反馈回
        KV cache。
        """
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)

        if not self._started:
            # 第一次调用：该观测作为第一个分块的条件（它*不是*关键帧）。
            self._started = True
            actions = self.predict_action_chunk(batch)  # [B, chunk_size, n_used]
            self._action_queue.extend(actions.transpose(0, 1))  # [chunk_size, B, n_used]
            self._obs_buffer = []
            self._exec_step = 0
        else:
            # 该观测是上一个已执行动作的结果 -> 关键帧候选。
            # 在上游客户端采样关键帧的子步边界上将其缓存。
            if (self._prev_j + 1) % self._keyframe_stride == 0:
                self._obs_buffer.append(self._extract_raw_obs(batch))
            if len(self._action_queue) == 0:
                # 当前分块的所有动作都已执行；将观测到的关键帧 + 已执行
                # 动作反馈回去，并预测下一个分块。
                actions = self.predict_action_chunk(None)
                self._action_queue.extend(actions.transpose(0, 1))
                self._exec_step = 0

        self._prev_j = self._exec_step % self.config.action_per_frame
        self._exec_step += 1
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """运行一个自回归分块，返回动作 ``[B, chunk_size, n_used]``（已归一化）。"""
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)

        is_first = self._first_chunk
        if is_first:
            init_latent = self._encode_frames([self._extract_raw_obs(batch)])
            self._init_latent = init_latent
            self._init_streaming_cache(init_latent)
            self._obs_buffer = []  # 第 0 帧（初始 obs）作为分块条件；不反馈回去
            actions, latents = self._infer(init_latent, frame_st_id=0)
            self._first_chunk = False
        else:
            # 将真实观测到的关键帧 + 已执行动作反馈回 KV cache。
            self._compute_kv_cache(self._obs_buffer, self._executed_actions)
            self._obs_buffer = []
            actions, latents = self._infer(None, frame_st_id=self._frame_st_id)

        # actions: [B, action_dim, F, action_per_frame, 1]（模型归一化空间）。保留用于 KV 反馈。
        self._executed_actions = actions

        if self.config.save_predicted_video:
            # 与上游 LingBot-VA 的可视化一致：收集各分块潜变量，在 rollout
            # 结束后一次性解码拼接起来的潜变量序列。
            self.last_predicted_frames = None
            self.last_predicted_latents = latents.detach().to("cpu")

        # 在第一个分块中，第 0 帧是条件帧（已经是“已知”的）：上游 LIBERO
        # 客户端会跳过它（start_idx=1），因此这里丢弃第一帧对应的动作。
        used = self.config.used_action_channel_ids
        a = actions[:, used]  # [B, n_used, F, action_per_frame, 1]
        if is_first:
            a = a[:, :, 1:]  # 丢弃第 0 帧 -> (F-1) 帧的动作
        a = a.squeeze(-1).flatten(2)  # [B, n_used, n_steps]
        a = a.transpose(1, 2).contiguous()  # [B, n_steps, n_used]
        return a.to(torch.float32)

    # Prompt / 文本编码
    def _maybe_init_prompt(self, batch):
        if self._prompt_embeds is not None or batch is None:
            return
        task = batch.get("task")
        prompt = task[0] if isinstance(task, list | tuple) else task
        self._prompt = prompt or ""
        self._prompt_embeds, self._negative_prompt_embeds = self._encode_prompt(self._prompt)

    def _get_t5_prompt_embeds(self, prompt, max_sequence_length):
        tokenizer = self._frozen["tokenizer"]
        text_encoder = self._frozen["text_encoder"]
        device = self.config.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [clean_prompt(u) for u in prompt]

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        te_device = next(text_encoder.parameters()).device
        prompt_embeds = text_encoder(text_input_ids.to(te_device), mask.to(te_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens, strict=False)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )
        return prompt_embeds.to(device)

    def _encode_prompt(self, prompt):
        max_len = self.config.max_sequence_length
        prompt_embeds = self._get_t5_prompt_embeds(prompt, max_len)
        negative_prompt_embeds = None
        if self._use_cfg:
            negative_prompt_embeds = self._get_t5_prompt_embeds("", max_len)
        return prompt_embeds, negative_prompt_embeds

    # 观测（图像）编码 -> 归一化视频潜变量
    def _extract_raw_obs(self, batch) -> dict[str, Tensor]:
        """从 batch 中快照所配置的相机图像（保持原始形式，供后续 VAE 编码）。"""
        return {k: batch[k].detach() for k in self.config.obs_cam_keys}

    def _camera_frame(self, raw_obs, key, size=None) -> Tensor:
        """返回单帧相机张量 [1, C, 1, H, W]，已缩放尺寸并映射到 [-1, 1]。"""
        img = raw_obs[key]
        if img.dim() == 3:  # [C, H, W]
            img = img.unsqueeze(0)
        # LeRobot 的图像是取值在 [0, 1] 的 float，形状为 [B, C, H, W]。
        img = img.to(self.config.device, torch.float32)
        if self.config.image_hflip:
            img = torch.flip(img, dims=[-1])  # 撤销环境处理器的水平翻转
        if size is None:
            size = (self.config.height, self.config.width)
        img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
        img = img * 2.0 - 1.0
        return img.unsqueeze(2).to(self.dtype)  # [1, C, F=1, H, W]

    def _normalize_vae_latent(self, enc_out: Tensor) -> Tensor:
        """取 VAE 编码器输出的均值并按通道归一化（与上游一致）。"""
        mu, _logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self._vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self._vae.config.latents_std).to(mu.device)
        mean = latents_mean.view(1, -1, 1, 1, 1)
        inv_std = (1.0 / latents_std).view(1, -1, 1, 1, 1)
        return ((mu.float() - mean) * inv_std).to(mu)

    @torch.no_grad()
    def _encode_frames(self, raw_frames: list) -> Tensor:
        """对观测帧的时序片段做 VAE 编码，并沿宽度方向拼接逐相机潜变量。

        ``raw_frames`` 是逐帧 obs 字典组成的列表（每个环境子步一个）。每个所
        配置的相机会沿时间轴堆叠为 ``[1, C, F, H, W]`` 片段，并通过一次流式
        ``encode_chunk`` 调用完成编码，使 VAE 的时间下采样（x4）把 F 个输入帧
        压缩为 ``F / 4`` 个潜变量帧，同时因果 ``feat_cache`` 会跨分块保留
        （对应上游 ``_encode_obs``）。
        """
        vae_device = next(self._vae.parameters()).device
        if self.config.camera_layout == "robotwin_tshape":
            return self._encode_frames_tshape(raw_frames, vae_device)
        per_cam_videos = []
        for k in self.config.obs_cam_keys:
            frames = [self._camera_frame(fb, k) for fb in raw_frames]
            per_cam_videos.append(torch.cat(frames, dim=2))  # [1, C, F, H, W]
        videos = torch.cat(per_cam_videos, dim=0)  # [num_cam, C, F, H, W]
        enc_out = self._streaming_vae.encode_chunk(videos.to(vae_device).to(self.dtype))
        mu_norm = self._normalize_vae_latent(enc_out)
        # 沿宽度方向拼接逐相机潜变量。
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.config.device)

    @torch.no_grad()
    def _encode_frames_tshape(self, raw_frames: list, vae_device) -> Tensor:
        """RoboTWin T 形潜变量装配：全分辨率头部相机 + 半分辨率腕部相机（第二个流式 VAE）。

        两个腕部潜变量先沿宽度方向拼接，再（沿高度轴）堆叠到头部潜变量上方，
        对应上游 ``env_type='robotwin_tshape'`` 时的 ``_encode_obs``。
        """
        cfg = self.config
        h, w = cfg.height, cfg.width
        head_key, left_key, right_key = cfg.obs_cam_keys[0], cfg.obs_cam_keys[1], cfg.obs_cam_keys[2]
        head = torch.cat([self._camera_frame(fb, head_key, size=(h, w)) for fb in raw_frames], dim=2)
        left = torch.cat(
            [self._camera_frame(fb, left_key, size=(h // 2, w // 2)) for fb in raw_frames], dim=2
        )
        right = torch.cat(
            [self._camera_frame(fb, right_key, size=(h // 2, w // 2)) for fb in raw_frames], dim=2
        )
        wrists = torch.cat([left, right], dim=0)  # [2, C, F, H/2, W/2]
        enc_high = self._streaming_vae.encode_chunk(head.to(vae_device).to(self.dtype))
        enc_lr = self._frozen["streaming_vae_half"].encode_chunk(wrists.to(vae_device).to(self.dtype))
        # 两个腕部相机沿宽度并排放置，再沿高度轴堆叠到头部潜变量上方。
        enc_out = torch.cat([torch.cat(enc_lr.split(1, dim=0), dim=-1), enc_high], dim=-2)
        video_latent = self._normalize_vae_latent(enc_out)
        return video_latent.to(self.config.device)

    # KV cache 管理
    @property
    def _latent_hw(self):
        if self.config.camera_layout == "robotwin_tshape":
            # 头部相机（全分辨率）在底部，两个半分辨率腕部相机并排在上方 -> 高度为 1.5 倍。
            return ((self.config.height // 16) * 3) // 2, self.config.width // 16
        h = self.config.height // 16
        w = (self.config.width // 16) * len(self.config.obs_cam_keys)
        return h, w

    def _init_streaming_cache(self, init_latent):
        cfg = self.config
        latent_h, latent_w = self._latent_hw
        p = cfg.patch_size
        latent_token_per_chunk = (cfg.frame_chunk_size * latent_h * latent_w) // (p[0] * p[1] * p[2])
        action_token_per_chunk = cfg.frame_chunk_size * cfg.action_per_frame
        self.transformer.create_empty_cache(
            "pos",
            cfg.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            device=self.config.device,
            dtype=self.dtype,
            batch_size=2 if self._use_cfg else 1,
        )
        self._cache_initialised = True

    def _repeat_input_for_cfg(self, input_dict):
        if self._use_cfg:
            input_dict["noisy_latents"] = input_dict["noisy_latents"].repeat(2, 1, 1, 1, 1)
            input_dict["text_emb"] = torch.cat(
                [
                    self._prompt_embeds.to(self.dtype).clone(),
                    self._negative_prompt_embeds.to(self.dtype).clone(),
                ],
                dim=0,
            )
            input_dict["grid_id"] = input_dict["grid_id"][None].repeat(2, 1, 1)
            input_dict["timesteps"] = input_dict["timesteps"][None].repeat(2, 1)
        else:
            input_dict["grid_id"] = input_dict["grid_id"][None]
            input_dict["timesteps"] = input_dict["timesteps"][None]
        return input_dict

    def _prepare_latent_input(
        self,
        latent_model_input,
        action_model_input,
        latent_t=0,
        action_t=0,
        latent_cond=None,
        action_cond=None,
        frame_st_id=0,
    ):
        cfg = self.config
        device = self.config.device
        p = cfg.patch_size
        out = {}
        if latent_model_input is not None:
            out["latent_res_lst"] = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones([latent_model_input.shape[2]], dtype=torch.float32, device=device)
                * latent_t,
                "grid_id": get_mesh_id(
                    latent_model_input.shape[-3] // p[0],
                    latent_model_input.shape[-2] // p[1],
                    latent_model_input.shape[-1] // p[2],
                    0,
                    1,
                    frame_st_id,
                ).to(device),
                "text_emb": self._prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                out["latent_res_lst"]["noisy_latents"][:, :, 0:1] = latent_cond[:, :, 0:1]
                out["latent_res_lst"]["timesteps"][0:1] *= 0
        if action_model_input is not None:
            out["action_res_lst"] = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones([action_model_input.shape[2]], dtype=torch.float32, device=device)
                * action_t,
                "grid_id": get_mesh_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1,
                    1,
                    frame_st_id,
                    action=True,
                ).to(device),
                "text_emb": self._prompt_embeds.to(self.dtype).clone(),
            }
            if action_cond is not None:
                out["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[:, :, 0:1]
                out["action_res_lst"]["timesteps"][0:1] *= 0
            out["action_res_lst"]["noisy_latents"][:, ~self._action_mask] *= 0
        return out

    @property
    def _action_mask(self):
        mask = torch.zeros([self.config.action_dim], dtype=torch.bool)
        mask[self.config.used_action_channel_ids] = True
        return mask

    # 动作条件（已执行动作历史）的（反）归一化
    def _preprocess_action_state(self, action_norm: Tensor) -> Tensor:
        """从已经归一化的已执行动作构建动作条件张量。

        ``action_norm`` 是模型空间的动作分块
        ``[B, action_dim, F, action_per_frame, 1]``。上游通过分位数归一化从
        原始已执行动作重新推导条件；这里的已执行动作已经处于模型归一化空间，
        因此直接透传。
        """
        return action_norm.to(self.config.device, self.dtype)

    def _compute_kv_cache(self, obs_buffer, executed_actions):
        """将真实观测到的关键帧 + 已执行动作反馈回 KV cache。"""
        if not obs_buffer or executed_actions is None:
            return
        self.transformer.clear_pred_cache("pos")
        # 通过一次流式调用编码缓存的关键帧片段（携带因果 VAE 缓存）。
        latent_model_input = self._encode_frames(obs_buffer)
        # 第一次反馈时，在前面补上初始潜变量，使潜变量/动作的帧计数对齐
        # （上游在 frame_st_id == 0 时会把 ``init_latent`` 补到观测关键帧前面）。
        if self._frame_st_id == 0 and getattr(self, "_init_latent", None) is not None:
            latent_model_input = torch.cat([self._init_latent, latent_model_input], dim=2)
        action_model_input = self._preprocess_action_state(executed_actions)
        action_model_input = action_model_input.to(latent_model_input)
        input_dict = self._prepare_latent_input(
            latent_model_input, action_model_input, frame_st_id=self._frame_st_id
        )
        with torch.no_grad():
            self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=2,
                cache_name="pos",
                action_mode=False,
            )
            self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=2,
                cache_name="pos",
                action_mode=True,
            )
        self._frame_st_id += latent_model_input.shape[2]

    # 核心双流去噪循环（一个分块）
    @torch.no_grad()
    def _infer(self, init_latent, frame_st_id=0):
        cfg = self.config
        device = self.config.device
        latent_h, latent_w = self._latent_hw
        frame_chunk_size = cfg.frame_chunk_size

        latents = torch.randn(1, 48, frame_chunk_size, latent_h, latent_w, device=device, dtype=self.dtype)
        actions = torch.randn(
            1, cfg.action_dim, frame_chunk_size, cfg.action_per_frame, 1, device=device, dtype=self.dtype
        )

        self._scheduler.set_timesteps(cfg.num_inference_steps)
        self._action_scheduler.set_timesteps(cfg.action_num_inference_steps)
        timesteps = F.pad(self._scheduler.timesteps, (0, 1), mode="constant", value=0)
        if cfg.video_exec_step != -1:
            timesteps = timesteps[: cfg.video_exec_step]
        action_timesteps = F.pad(self._action_scheduler.timesteps, (0, 1), mode="constant", value=0)

        # 1. 视频潜变量去噪循环
        for i, t in enumerate(timesteps):
            last_step = i == len(timesteps) - 1
            latent_cond = (
                init_latent[:, :, 0:1].to(self.dtype)
                if frame_st_id == 0 and init_latent is not None
                else None
            )
            input_dict = self._prepare_latent_input(
                latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id
            )
            video_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name="pos",
                action_mode=False,
            )
            if not last_step or cfg.video_exec_step != -1:
                video_noise_pred = data_seq_to_patch(
                    cfg.patch_size,
                    video_noise_pred,
                    frame_chunk_size,
                    latent_h,
                    latent_w,
                    batch_size=2 if self._use_cfg else 1,
                )
                if cfg.guidance_scale > 1:
                    video_noise_pred = video_noise_pred[1:] + cfg.guidance_scale * (
                        video_noise_pred[:1] - video_noise_pred[1:]
                    )
                else:
                    video_noise_pred = video_noise_pred[:1]
                latents = self._scheduler.step(video_noise_pred, t, latents, return_dict=False)
            if frame_st_id == 0 and latent_cond is not None:
                latents[:, :, 0:1] = latent_cond

        # 2. 动作去噪循环
        for i, t in enumerate(action_timesteps):
            last_step = i == len(action_timesteps) - 1
            action_cond = (
                torch.zeros([1, cfg.action_dim, 1, cfg.action_per_frame, 1], device=device, dtype=self.dtype)
                if frame_st_id == 0
                else None
            )
            input_dict = self._prepare_latent_input(
                None, actions, t, t, None, action_cond, frame_st_id=frame_st_id
            )
            action_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name="pos",
                action_mode=True,
            )
            if not last_step:
                action_noise_pred = rearrange(action_noise_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size)
                if cfg.action_guidance_scale > 1:
                    action_noise_pred = action_noise_pred[1:] + cfg.action_guidance_scale * (
                        action_noise_pred[:1] - action_noise_pred[1:]
                    )
                else:
                    action_noise_pred = action_noise_pred[:1]
                actions = self._action_scheduler.step(action_noise_pred, t, actions, return_dict=False)
            if frame_st_id == 0 and action_cond is not None:
                actions[:, :, 0:1] = action_cond

        actions[:, ~self._action_mask] *= 0
        return actions, latents

    # 预测视频解码（可选启用）
    @torch.no_grad()
    def decode_predicted_latents(self, latents) -> Tensor:
        """将拼接的预测潜变量序列解码为 ``[T, H, W, 3]`` uint8 帧。"""
        return self._decode_predicted_video(latents)

    @torch.no_grad()
    def _decode_predicted_video(self, latents) -> Tensor:
        """在 CPU 上将预测潜变量经 VAE 解码为 uint8 帧栈 ``[T, H, W, 3]``。"""
        vae = self._vae
        z_dim = vae.config.z_dim
        vae_device = next(vae.parameters()).device
        latents = latents.to(device=vae_device, dtype=vae.dtype)
        latents = denormalize_latents(latents, vae.config.latents_mean, vae.config.latents_std, z_dim)
        video = vae.decode(latents, return_dict=False)[0]  # [B, C, F, H, W] in [-1, 1]
        video = (video.float().clamp(-1, 1) + 1.0) / 2.0
        video = (video[0].permute(1, 2, 3, 0) * 255.0).round().to(torch.uint8)  # [F, H, W, C]
        return video.cpu()
