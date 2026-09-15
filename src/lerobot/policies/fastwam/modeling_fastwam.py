# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_fastwam import FastWAMConfig
from .wan import (
    ActionDiT,
    FastWAM,
    MoT,
    WanVideoDiT,
    build_wan_tokenizer,
    load_pretrained_wan_text_encoder,
    load_pretrained_wan_vae,
)


class FastWAMPolicy(PreTrainedPolicy):
    """FastWAM 的 LeRobot 策略封装。

    注意力后端：FastWAM 的 DiT 对所有注意力均使用
    ``torch.nn.functional.scaled_dot_product_attention``（SDPA）。它不使用 FlashAttention，
    因为 MoT 路由需要任意的布尔 ``[query, key]`` 掩码，而 FlashAttention 的 varlen API
    无法表达；安装 ``flash-attn`` 对 FastWAM 路径没有影响。（SDPA 在内部仍可能调度到
    PyTorch 自带的 flash/mem-efficient/math 内核，与 ``flash-attn`` 包无关。）

    Args:
        config (FastWAMConfig): FastWAM 策略配置。
        dataset_stats (dict[str, dict[str, Tensor]] | None): 可选的、由训练/评估
            栈传入的 LeRobot 数据集统计信息。
    """

    config_class = FastWAMConfig
    name = "fastwam"
    # FSDP2 包装单元：MoTLayer 是每一层专家块的唯一 FSDP 持有者
    # （这些块被重新挂到它下面，正是为了让分片只有一个可挂钩的边界）。
    _fsdp_wrap_modules = ["MoTLayer"]

    def __init__(
        self,
        config: FastWAMConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs: Any,
    ):
        # FastWAM 的 Wan2.2 主干需要 transformers（UMT5 文本编码器/分词器）和
        # diffusers（Wan VAE），两者都位于 `fastwam` extra 之后。在基础安装中
        # 应尽早失败并给出可操作的提示，而不是在 Wan 组件构建深处才报错。
        require_package("transformers", extra="fastwam")
        require_package("diffusers", extra="fastwam")
        # `make_policy`/`from_pretrained` 会转发额外的 kwargs（例如 `dataset_meta`）；
        # 数据集特征元数据已经由上游的 make_policy 应用到 `config` 上，
        # 因此这里接收并忽略它们，与其他 LeRobot 策略保持一致。
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats
        self.model = self._build_core_model(config)
        if config.freeze_video_expert and getattr(self.model, "video_expert", None) is not None:
            # 冻结约 5B 的 Wan 视频专家；get_optim_params 按 requires_grad 过滤，
            # 因此它的参数会退出优化器（DDP 也会跳过它们）。
            self.model.video_expert.requires_grad_(False)
            # transformer 块被重新挂到 MoTLayers 下（单一 FSDP 持有者），因此
            # `video_expert.requires_grad_` 不再能触达它们——通过各层来冻结。
            mot = getattr(self.model, "mot", None)
            if mot is not None and getattr(mot, "layers", None) is not None:
                for layer in mot.layers:
                    if "video" in layer.blocks:
                        layer.blocks["video"].requires_grad_(False)
        self.reset()

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        """支持跨本体微调的形状感知加载。

        `safetensors.load_model(strict=False)` 会忽略缺失/多余的键，但对于共有键
        的形状不匹配仍会报错。当从一个在不同本体上训练的 checkpoint 进行微调时
        （例如把 LIBERO 7-DoF / 8 维的 checkpoint 适配到 6-DoF / 6 维机械臂），
        动作编码器/头和本体感知编码器的形状合理地存在差异。在 `strict=False` 下，
        我们只丢弃这些形状不匹配的张量——让它们保持新初始化的值——并加载所有
        兼容的张量。在 `strict=True` 下则使用标准的精确匹配加载器。
        """
        from safetensors import safe_open

        model_state_dict = model.state_dict()
        mismatched = []
        with safe_open(model_file, framework="pt") as f:
            checkpoint_keys = list(f.keys())
            for key in checkpoint_keys:
                if key in model_state_dict and tuple(model_state_dict[key].shape) != tuple(
                    f.get_slice(key).get_shape()
                ):
                    mismatched.append(key)

        if not mismatched:
            return super()._load_as_safetensor(model, model_file, map_location, strict)
        if strict:
            raise RuntimeError(
                f"FastWAM: {len(mismatched)} checkpoint tensors have a shape mismatch under "
                f"strict=True: {mismatched}"
            )

        from safetensors.torch import load_file

        logging.warning(
            "FastWAM cross-embodiment load: reinitializing %d shape-mismatched tensor(s), keeping "
            "every compatible weight: %s",
            len(mismatched),
            mismatched,
        )
        state_dict = load_file(model_file, device="cpu")
        for key in mismatched:
            state_dict.pop(key, None)
        model.load_state_dict(state_dict, strict=False)
        if map_location and map_location != "cpu":
            model.to(map_location)
        return model

    def get_optim_params(self) -> list[Tensor]:
        # 直接返回可训练的张量（单个参数组）。优化器构建器会把这些张量包装进
        # 参数组；如果改为返回裸的 {"params": [...]} 字典，`list(...)` 得到的
        # 将是键字符串 "params"。
        params = (
            list(self.model.dit.parameters()) if hasattr(self.model, "dit") else list(self.model.parameters())
        )
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            params.extend(list(proprio_encoder.parameters()))
        return [p for p in params if p.requires_grad]

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def _batch_to_training_sample(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """将标准 LeRobot batch 适配为 `FastWAM.build_inputs` 所消费的
        FastWAM 原生样本（`video`、`action`、`context`/`context_mask`、
        逐帧的 `proprio`）。

        LeRobot 训练循环传入原始的 `observation.images.*`、单步的
        `observation.state` `[B, D]`、`action` 和语言 `task` 字符串。我们只做
        `build_inputs` 无法完成的转换：把相机帧堆叠成视频、用（冻结的）文本编码器
        编码 prompt（与推理保持一致，因此语言条件数据集无需预计算的 context），
        并为 proprio 添加 `build_inputs` 索引所需的逐帧轴。所有形状/存在性校验
        都留给 `build_inputs`——该契约的唯一权威。
        """
        sample = dict(batch)
        if "video" not in sample:
            sample["video"] = _stack_video_from_images(batch, self.config)
        if "context" not in sample or "context_mask" not in sample:
            prompt = _prompt_from_batch(batch=batch, config=self.config)
            if prompt is None:
                raise KeyError(
                    "FastWAM training requires a `task`/`prompt` to encode text context, "
                    "or precomputed `context`/`context_mask` in the batch."
                )
            sample["context"], sample["context_mask"] = self.model.encode_prompt(prompt)
        if self.config.proprio_dim is not None and "proprio" not in sample:
            state = sample.get(OBS_STATE)
            if state is not None:
                # LeRobot 给出单步状态 [B, D]；build_inputs 期望逐帧的
                # [B, T, D] 并使用第 0 帧，因此添加一个 T=1 的轴。
                sample["proprio"] = state.unsqueeze(1) if state.ndim == 2 else state
        return sample

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        """为一个 LeRobot batch 计算 FastWAM 训练损失。

        Args:
            batch (dict[str, Tensor]): 包含 FastWAM 就绪键
                （`video`、`action`、`context`、`context_mask`）或可被适配的
                LeRobot 键（`observation.images.*`、`observation.state`、
                `action`、`action_is_pad`）的 batch。

        Returns:
            tuple[Tensor, dict[str, Any]]: 用于反向传播的标量损失，以及记录指标
            （例如 `loss_video`、`loss_action`）的字典——即 LeRobot 训练循环
            所期望的 `(loss, output_dict)` 契约。
        """

        sample = self._batch_to_training_sample(batch)
        loss, metrics = self.model.training_loss(sample)
        return loss, dict(metrics or {})

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        """从当前 FastWAM 观测预测一个动作块。

        Args:
            batch (dict[str, Tensor]): 包含 `input_image` 或图像观测键，
                以及 `context/context_mask` 或 `prompt` 的推理 batch。

        Returns:
            Tensor: 形状为 `[B, action_horizon, action_dim]` 的动作块。
        """

        self.eval()
        infer_kwargs = _batch_to_infer_kwargs(batch=batch, config=self.config)
        batch_size = _infer_kwargs_batch_size(infer_kwargs)
        if batch_size == 1:
            action = _action_from_model_output(self.model.infer_action(**infer_kwargs))
        else:
            action = torch.cat(
                [
                    _action_from_model_output(
                        self.model.infer_action(
                            **_slice_infer_kwargs(infer_kwargs, index=i, batch_size=batch_size)
                        )
                    )
                    for i in range(batch_size)
                ],
                dim=0,
            )
        return action.to(device=batch_device(batch), dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def _build_core_model(self, config: FastWAMConfig) -> FastWAM:
        """构建用于训练/推理的 FastWAM 核心。

        这里只实例化可训练部分（MoT DiT 和本体感知编码器）的空壳，随后由基类的
        `from_pretrained` 从策略的 `model.safetensors` 填充权重。*冻结的* Wan2.2 VAE
        和 UMT5 文本编码器从 `Wan-AI/Wan2.2-TI2V-5B-Diffusers` 仓库加载真实权重
        （缓存在 HF 缓存中，跨 checkpoint 共享），并被有意排除在 `model.safetensors`
        之外——参见 `FastWAM.__init__`。分词器来自 `google/umt5-xxl`。
        """
        dtype = _dtype_from_name(config.torch_dtype)
        device = config.device
        video_expert = WanVideoDiT(**config.video_dit_config).to(device=device, dtype=dtype)
        action_expert = ActionDiT(**config.action_dit_config).to(device=device, dtype=dtype)
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
        )
        text_encoder = (
            load_pretrained_wan_text_encoder(
                model_id=config.text_encoder_model_id, torch_dtype=dtype, device=device
            )
            if config.load_text_encoder
            else None
        )
        return FastWAM(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=load_pretrained_wan_vae(torch_dtype=dtype, device=device),
            text_encoder=text_encoder,
            tokenizer=build_wan_tokenizer(
                model_id=config.tokenizer_model_id, tokenizer_max_len=config.tokenizer_max_len
            ),
            text_dim=int(config.video_dit_config["text_dim"]),
            proprio_dim=config.proprio_dim,
            device=device,
            torch_dtype=dtype,
            video_train_shift=float(config.video_scheduler["train_shift"]),
            video_infer_shift=float(config.video_scheduler["infer_shift"]),
            video_num_train_timesteps=int(config.video_scheduler["num_train_timesteps"]),
            action_train_shift=float(config.action_scheduler["train_shift"]),
            action_infer_shift=float(config.action_scheduler["infer_shift"]),
            action_num_train_timesteps=int(config.action_scheduler["num_train_timesteps"]),
            loss_lambda_video=float(config.loss["lambda_video"]),
            loss_lambda_action=float(config.loss["lambda_action"]),
        )


def _scalar(value: Any) -> Any:
    """将 0 维/单元素张量（例如来自 DataLoader 的 collate）解包为 Python 标量。"""
    return value.item() if isinstance(value, Tensor) else value


def _batch_to_infer_kwargs(batch: dict[str, Tensor], config: FastWAMConfig) -> dict[str, Any]:
    return {
        "prompt": _prompt_from_batch(batch=batch, config=config),
        "input_image": _input_image_from_batch(batch, config),
        "action_horizon": config.action_horizon,
        "proprio": batch.get("proprio", batch.get(OBS_STATE)),
        "context": batch.get("context"),
        "context_mask": batch.get("context_mask"),
        "negative_prompt": batch.get("negative_prompt", config.negative_prompt),
        "text_cfg_scale": float(_scalar(batch.get("text_cfg_scale", config.text_cfg_scale))),
        "num_inference_steps": int(_scalar(batch.get("num_inference_steps", config.num_inference_steps))),
        "sigma_shift": batch.get("sigma_shift", config.sigma_shift),
        "seed": batch.get("seed", config.inference_seed),
        "rand_device": batch.get("rand_device", config.rand_device),
        "tiled": bool(batch.get("tiled", config.tiled)),
        "compile_action_infer": bool(batch.get("compile_action_infer", config.compile_action_infer)),
    }


def _prompt_from_batch(batch: dict[str, Tensor], config: FastWAMConfig) -> Any:
    prompt = batch.get("prompt")
    if prompt is not None:
        return prompt

    task = batch.get("task")
    if task is None:
        return None
    if isinstance(task, str):
        return config.prompt_template.format(task=task)
    if isinstance(task, (list, tuple)):
        return [config.prompt_template.format(task=str(item)) for item in task]
    return config.prompt_template.format(task=str(task))


def _action_from_model_output(output: Any) -> Tensor:
    action = output["action"] if isinstance(output, dict) else output
    if action.ndim == 2:
        action = action.unsqueeze(0)
    return action


def _infer_kwargs_batch_size(infer_kwargs: dict[str, Any]) -> int:
    image = infer_kwargs["input_image"]
    if not isinstance(image, Tensor):
        raise TypeError(f"`input_image` must be a tensor, got {type(image).__name__}.")
    if image.ndim == 3:
        return 1
    if image.ndim == 4:
        return int(image.shape[0])
    raise ValueError(f"`input_image` must be [B,C,H,W] or [C,H,W], got {tuple(image.shape)}.")


def _slice_infer_kwargs(infer_kwargs: dict[str, Any], *, index: int, batch_size: int) -> dict[str, Any]:
    return {
        key: _slice_infer_value(value, index=index, batch_size=batch_size)
        for key, value in infer_kwargs.items()
    }


def _slice_infer_value(value: Any, *, index: int, batch_size: int) -> Any:
    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[index : index + 1]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[index]
    return value


def _dtype_from_name(name: str) -> torch.dtype:
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype `{name}`.")
    return dtype_map[name]


def batch_device(batch: dict[str, Any]) -> torch.device:
    for value in batch.values():
        if isinstance(value, Tensor):
            return value.device
    return torch.device("cpu")


def _resize_frames(frames: Tensor, size: tuple[int, int]) -> Tensor:
    """将帧张量 resize 到 `size` (H, W)，容忍前置的时间/批次堆叠维度。

    `interpolate` 只接受单个前置批次维（`[N, C, H, W]`），但 FastWAM 的相机张量
    以 `[B, C, H, W]`（实时评估）或 `[B, T, C, H, W]`（时间堆叠）的形式到达，因此
    把所有前置维度展平进批次维、执行 resize，然后再还原。若已是 `size` 则不做任何操作。
    """
    if tuple(frames.shape[-2:]) == size:
        return frames
    lead = frames.shape[:-3]
    flat = frames.reshape(-1, *frames.shape[-3:])
    flat = torch.nn.functional.interpolate(
        flat, size=size, mode="bilinear", align_corners=False, antialias=True
    )
    return flat.reshape(*lead, *flat.shape[-3:])


def _stack_video_from_images(batch: dict[str, Tensor], config: FastWAMConfig) -> Tensor:
    # 排除 delta-timestamp 加载为每个相机附加的 `*_is_pad` 伴随张量（形状 [B, T]）；
    # 它们共享 `observation.images.` 前缀，但并不是帧。
    image_keys = sorted(k for k in batch if k.startswith("observation.images.") and not k.endswith("_is_pad"))
    if not image_keys:
        raise KeyError("FastWAM batch must contain `video` or `observation.images.*` keys.")
    per_cam = (int(config.image_size[0]), int(config.image_size[1]) // len(image_keys))
    images = [_resize_frames(batch[key], per_cam) for key in image_keys]
    # 无论是单帧还是时间序列情况，相机都沿宽度（最后一维）拼接。
    image = torch.cat(images, dim=-1) if len(images) > 1 else images[0]
    if image.ndim == 4:
        # [B, C, H, W]：单帧（例如实时评估的观测）-> 沿时间维重复。
        image = image.unsqueeze(2).repeat(1, 1, config.model_video_frames, 1, 1)
    elif image.ndim == 5:
        # [B, T, C, H, W]：来自 delta-timestamp 加载的时间堆叠 -> [B, C, T, H, W]。
        image = image.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Expected image batch [B,C,H,W] or temporal [B,T,C,H,W], got {tuple(image.shape)}.")
    return image


def _input_image_from_batch(batch: dict[str, Tensor], config: FastWAMConfig) -> Tensor:
    if "input_image" in batch:
        return _prepare_infer_image(batch["input_image"], config)
    video = batch.get("video")
    if video is None:
        video = _stack_video_from_images(batch, config)
    if video.ndim == 5:
        return _prepare_infer_image(video[:, :, 0], config)
    if video.ndim == 4:
        return _prepare_infer_image(video, config)
    raise ValueError(f"Cannot build input image from tensor with shape {tuple(video.shape)}.")


def _prepare_infer_image(image: Tensor, config: FastWAMConfig) -> Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor [B,C,H,W] or [C,H,W], got {tuple(image.shape)}.")

    # Resize 到完整配置的分辨率（如果视频路径已经生成了该尺寸则不做任何操作，但
    # 也覆盖了直接提供的 `input_image`）。模型拥有其输入分辨率的所有权——参见
    # `_stack_video_from_images`——因此我们在尺寸不匹配时执行 resize 而不是断言。
    target_h, target_w = int(config.image_size[0]), int(config.image_size[1])
    return _resize_frames(image, (target_h, target_w))
