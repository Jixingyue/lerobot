#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
用于 LeRobot 集成的 Groot 策略封装

这是一个最小化集成，尽可能委托给 Isaac-GR00T N1.7 的组件，而不移植其代码。
数据集加载和训练编排由 LeRobot 的标准训练栈处理。
"""

import builtins
import logging
import os
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from torch import Tensor

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.import_utils import _transformers_available, require_package

from ..pretrained import PreTrainedPolicy
from ..utils import get_device_from_parameters
from .configuration_groot import (
    GROOT_N1_5,
    GROOT_N1_5_REMOVAL_GUIDANCE,
    GROOT_N1_7,
    GrootConfig,
    infer_groot_model_version,
    infer_groot_n1_7_action_execution_horizon,
    infer_groot_n1_7_action_horizon,
)
from .groot_n1_7 import GR00TN17, _tie_unused_qwen_lm_head

if TYPE_CHECKING or _transformers_available:
    from transformers.trainer_pt_utils import get_parameter_names
else:
    get_parameter_names = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="GrootPolicy")


class GrootPolicy(PreTrainedPolicy):
    """对外部 Groot 模型的封装，用于 LeRobot 集成。"""

    name = "groot"
    config_class = GrootConfig

    def supports_rtc(self) -> bool:
        return True

    def __init__(self, config: GrootConfig, **kwargs):
        """初始化 Groot 策略封装。"""
        require_package("transformers", extra="groot")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # 使用移植过来的组件初始化 GR00T 模型
        self._groot_model = self._create_groot_model()
        self._action_queue_steps = self._resolve_action_queue_steps()
        self._warned_native_relative_rtc_prefix_disabled = False

        self.reset()

    def _create_groot_model(self):
        """使用移植过来的组件创建并初始化 GR00T N1.7 模型。"""
        model_kwargs = {
            "pretrained_model_name_or_path": self.config.base_model_path,
            "tune_llm": self.config.tune_llm,
            "tune_visual": self.config.tune_visual,
            "tune_projector": self.config.tune_projector,
            "tune_diffusion_model": self.config.tune_diffusion_model,
            # 作为 GR00TN17Config 的覆写项传入；由 set_trainable_parameters 读回。
            "tune_top_llm_layers": self.config.tune_top_llm_layers,
            "use_flash_attention": self.config.use_flash_attention,
        }
        # 仅当用户显式设置时，才把推理时的调节项透传到模型配置；
        # None 表示保留检查点中固化的值不动。
        if self.config.num_inference_timesteps is not None:
            model_kwargs["num_inference_timesteps"] = self.config.num_inference_timesteps
        if self.config.rtc_ramp_rate is not None:
            model_kwargs["rtc_ramp_rate"] = self.config.rtc_ramp_rate

        model = GR00TN17.from_pretrained(
            **model_kwargs,
            tune_vlln=self.config.tune_vlln,
            transformers_loading_kwargs={"trust_remote_code": True},
        )
        backbone = getattr(model, "backbone", None)
        qwen_model = getattr(backbone, "model", None)
        if qwen_model is not None:
            _tie_unused_qwen_lm_head(qwen_model)
        if self.config.model_params_fp32:
            self._cast_model_parameters_to_fp32(model)
        return model

    @staticmethod
    def _cast_model_parameters_to_fp32(model: torch.nn.Module) -> None:
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.to(torch.float32)

    @staticmethod
    def _build_weight_decay_parameter_groups(model: torch.nn.Module) -> list[dict[str, object]]:
        forbidden_name_patterns = [
            r"bias",
            r"layernorm",
            r"rmsnorm",
            r"(?:^|\.)norm(?:$|\.)",
            r"_norm(?:$|\.)",
        ]
        decay_names = set(get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns))
        decay_params = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name in decay_names
        ]
        no_decay_params = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name not in decay_names
        ]
        return [
            {"params": decay_params},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def reset(self):
        """在环境重置时重置策略状态。"""
        self._action_queue = deque([], maxlen=self._action_queue_steps)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: GrootConfig | None = None,
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
        """从预训练模型加载 Groot 策略。

        处理两种情况：
        1. 基础 GR00T N1.7 模型——直接加载原始模型
        2. 微调过的 LeRobot 检查点——从 safetensors 加载配置和权重

        Args:
            pretrained_name_or_path: GR00T 模型或微调检查点的路径
            config: 可选的 GrootConfig。若为 None，则从检查点加载或创建默认配置
            force_download: 即使已有缓存也强制下载
            resume_download: 恢复被中断的下载
            proxies: 代理设置
            token: HuggingFace 认证令牌
            cache_dir: 缓存目录路径
            local_files_only: 仅使用本地文件
            revision: 指定的模型版本
            strict: 是否严格加载 state dict
            **kwargs: 附加参数（传递给 config）

        Returns:
            已加载模型的、初始化完成的 GrootPolicy 实例
        """
        requested_version = infer_groot_model_version(str(pretrained_name_or_path)) or GROOT_N1_7
        logger.info(
            "The Groot policy wraps NVIDIA's GR00T %s model. Loading pretrained model from: %s",
            requested_version,
            pretrained_name_or_path,
        )

        model_id = str(pretrained_name_or_path)
        is_finetuned_checkpoint = False

        # 检查这是否是一个微调过的 LeRobot 检查点（包含 model.safetensors）
        try:
            if os.path.isdir(model_id):
                is_finetuned_checkpoint = os.path.exists(os.path.join(model_id, SAFETENSORS_SINGLE_FILE))
            else:
                # 尝试下载 safetensors 文件以检查它是否存在
                try:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=SAFETENSORS_SINGLE_FILE,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=False,  # 仅检查是否存在，不强制下载
                        proxies=proxies,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    is_finetuned_checkpoint = True
                except HfHubHTTPError:
                    is_finetuned_checkpoint = False
        except Exception:
            is_finetuned_checkpoint = False

        if is_finetuned_checkpoint:
            # 这是一个微调过的 LeRobot 检查点——使用父类的加载逻辑
            logger.info("Detected fine-tuned LeRobot checkpoint, loading with state dict...")
            return super().from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                config=config,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                strict=strict,
                **kwargs,
            )

        # 这是一个基础 GR00T 模型——全新加载
        logger.info("Detected base GR00T model, loading from HuggingFace...")

        if config is None:
            # 用预训练路径创建默认配置
            config = GrootConfig(
                base_model_path=str(pretrained_name_or_path),
            )

            # 添加校验所需的最小视觉特征
            # validate_features() 会自动添加 state 和 action 特征
            # 这些只是占位符——实际的机器人特征来自预处理器
            if not config.input_features:
                config.input_features = {
                    f"{OBS_IMAGES}.camera": PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, 224, 224),  # 配置中的默认图像尺寸
                    ),
                }
        else:
            # 用给定的路径覆写 base_model_path
            config.base_model_path = str(pretrained_name_or_path)

        # 透传 kwargs 中任何额外的配置覆写项
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        inferred_version = infer_groot_model_version(config.base_model_path)
        if inferred_version is not None and inferred_version != GROOT_N1_7:
            message = (
                f"GR00T model_version '{GROOT_N1_7}' does not match base_model_path "
                f"'{config.base_model_path}', which looks like '{inferred_version}'."
            )
            if inferred_version == GROOT_N1_5:
                message = f"{message} {GROOT_N1_5_REMOVAL_GUIDANCE}"
            raise ValueError(message)
        # 创建一个全新的策略实例——这会在 __init__ 中通过 _create_groot_model()
        # 自动加载 GR00T 模型
        policy = cls(config)

        policy.eval()
        return policy

    def get_optim_params(self):  # type: ignore[override]
        """Isaac-GR00T 将偏置和归一化参数排除在权重衰减之外。"""
        return self._build_weight_decay_parameter_groups(self)

    def _resolve_action_queue_steps(self) -> int:
        n_action_steps = int(self.config.n_action_steps)
        checkpoint_action_horizon = infer_groot_n1_7_action_horizon(
            self.config.base_model_path,
            self.config.embodiment_tag,
        )
        execution_horizon = infer_groot_n1_7_action_execution_horizon(
            self.config.base_model_path,
            self.config.embodiment_tag,
        )
        horizons = [n_action_steps]
        if checkpoint_action_horizon is not None:
            horizons.append(checkpoint_action_horizon)
        if execution_horizon is not None:
            horizons.append(execution_horizon)
        return min(horizons)

    def _resolve_prediction_horizon(self, actions: Tensor) -> int:
        """针对原生 GR00T 预测结果，返回面向策略的动作时域（action horizon）。"""

        horizons = [actions.shape[1]]
        checkpoint_action_horizon = infer_groot_n1_7_action_horizon(
            self.config.base_model_path,
            self.config.embodiment_tag,
        )
        if checkpoint_action_horizon is not None:
            horizons.append(checkpoint_action_horizon)

        for horizon in (self.config.chunk_size, self.config.n_action_steps):
            horizon = int(horizon)
            if horizon > 0:
                horizons.append(horizon)

        return max(1, min(horizons))

    def _filter_groot_inputs(self, batch: dict[str, Tensor], *, include_action: bool) -> dict[str, Tensor]:
        allowed_base = {"state", "state_mask", "action_mask", "embodiment_id"}
        if include_action:
            allowed_base.add("action")

        allowed_base.update(
            {
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "mm_token_type_ids",
                "pixel_values_videos",
                "video_grid_thw",
            }
        )

        return {
            k: v for k, v in batch.items() if k in allowed_base and not (k.startswith("next.") or k == "info")
        }

    def _prepare_n1_7_rtc_inputs(
        self,
        inputs: dict[str, Tensor],
        *,
        inference_delay: object,
        prev_chunk_left_over: object,
    ) -> tuple[dict[str, Tensor], dict[str, object] | None]:
        if prev_chunk_left_over is None:
            return inputs, None
        if getattr(self.config, "use_relative_actions", False):
            # 通用 RTC 只能提供上一分块归一化后的剩余动作。对于使用原生相对动作的
            # N1.7 检查点，这些行与旧的观测状态以及旧的逐时域统计量行绑定，因此将其
            # 作为下一个前缀可能会把策略引向错误方向。在 GR00T 专属的 RTC 路径能够
            # 传递重新锚定后的绝对剩余动作之前，先不使用原生 RTC 重叠引导来运行。
            if not getattr(self, "_warned_native_relative_rtc_prefix_disabled", False):
                logger.info("Disabling native GR00T RTC prefix for relative-action policy")
                self._warned_native_relative_rtc_prefix_disabled = True
            return inputs, None
        if not isinstance(prev_chunk_left_over, torch.Tensor):
            raise TypeError("prev_chunk_left_over must be a torch.Tensor for GR00T N1.7 RTC.")
        if prev_chunk_left_over.numel() == 0:
            return inputs, None

        prev_actions = prev_chunk_left_over
        if prev_actions.ndim == 2:
            prev_actions = prev_actions.unsqueeze(0)
        elif prev_actions.ndim != 3:
            raise ValueError("prev_chunk_left_over must have shape (T, A) or (B, T, A) for GR00T N1.7 RTC.")

        state = inputs.get("state")
        if state is None:
            raise ValueError("GR00T N1.7 RTC requires `state` in the preprocessed batch.")
        batch_size = state.shape[0]
        if prev_actions.shape[0] == 1 and batch_size > 1:
            prev_actions = prev_actions.expand(batch_size, -1, -1).clone()
        elif prev_actions.shape[0] != batch_size:
            raise ValueError("prev_chunk_left_over batch size must match the current GR00T N1.7 batch size.")

        # 通用的 LeRobot RTC 引擎为了定长形状的策略调用，会用全零行填充较短的剩余
        # 动作。原生 GR00T N1.7 RTC 会把提供的每个前缀行都当作真实的动作约束，因此
        # 在构造原生 overlap 选项之前要先去除这些填充。
        valid_prefix_rows = prev_actions.detach().abs().sum(dim=(0, 2)) > 0
        if valid_prefix_rows.any():
            valid_prefix_steps = int(valid_prefix_rows.nonzero()[-1].item()) + 1
            prev_actions = prev_actions[:, :valid_prefix_steps, :]
        else:
            return inputs, None

        model_action_horizon = int(
            getattr(self._groot_model.config, "action_horizon", self.config.chunk_size)
        )
        max_action_dim = int(getattr(self._groot_model.config, "max_action_dim", self.config.max_action_dim))
        if prev_actions.shape[1] > model_action_horizon:
            prev_actions = prev_actions[:, -model_action_horizon:, :]

        action_horizon = int(prev_actions.shape[1])
        if action_horizon <= 0:
            return inputs, None

        if prev_actions.shape[2] > max_action_dim:
            prev_actions = prev_actions[:, :, :max_action_dim]
        elif prev_actions.shape[2] < max_action_dim:
            pad = torch.zeros(
                prev_actions.shape[0],
                prev_actions.shape[1],
                max_action_dim - prev_actions.shape[2],
                dtype=prev_actions.dtype,
                device=prev_actions.device,
            )
            prev_actions = torch.cat([prev_actions, pad], dim=2)

        prev_actions = prev_actions.to(device=state.device, dtype=state.dtype)

        rtc_config = getattr(self.config, "rtc_config", None)
        execution_horizon = int(getattr(rtc_config, "execution_horizon", action_horizon))
        overlap_steps = max(0, min(action_horizon, execution_horizon))
        if overlap_steps == 0:
            return inputs, None

        try:
            frozen_steps = int(inference_delay or 0)
        except (TypeError, ValueError):
            frozen_steps = 0
        frozen_steps = max(0, min(frozen_steps, overlap_steps))

        options = {
            "action_horizon": action_horizon,
            "rtc_overlap_steps": overlap_steps,
            "rtc_frozen_steps": frozen_steps,
            "rtc_ramp_rate": float(getattr(self._groot_model.config, "rtc_ramp_rate", 6.0)),
        }

        inputs = dict(inputs)
        inputs["action"] = prev_actions
        return inputs, options

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """训练前向传播。

        当输入兼容时，委托给 Isaac-GR00T 的 model.forward。
        """
        groot_inputs = self._filter_groot_inputs(batch, include_action=True)

        # 从模型参数获取设备
        device = get_device_from_parameters(self)

        # 在启用时于 bf16 autocast 下运行 GR00T 前向，以降低激活值内存占用
        # 理由：与原始 GR00T 微调一致（bf16 计算、fp32 参数），并避免上转为 fp32。
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(groot_inputs)

        # Isaac-GR00T 返回一个 BatchFeature；loss 的键通常是 'loss'
        loss = outputs.get("loss")
        if loss is None:
            raise RuntimeError(
                "GR00T model.forward did not return a 'loss'. Training batches must include "
                "'action' and 'action_mask'; check the preprocessor output."
            )

        loss_dict = {"loss": loss.item()}

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: object) -> Tensor:
        """通过委托给 Isaac-GR00T 预测推理用的一个动作分块。

        返回形状为 (B, n_action_steps, action_dim) 的张量。

        对于 N1.7，在调用底层模型之前，LeRobot 的 RTC 剩余动作会被转换为
        原生 GR00T 的动作重叠（action-overlap）选项。
        """
        self.eval()

        # 预处理由处理器流水线完成，因此这里只需对 batch 做过滤。
        # 推理时不传 action，因为 action 是要被预测出来的。
        # N1.7 仍会携带来自其检查点处理器的二维动作时域掩码。
        groot_inputs = self._filter_groot_inputs(batch, include_action=False)
        groot_inputs, groot_options = self._prepare_n1_7_rtc_inputs(
            groot_inputs,
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
        )

        # 从模型参数获取设备
        device = get_device_from_parameters(self)

        # 推理时使用 bf16 autocast，以保持较低内存占用并与主干网络的 dtype 一致
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            if groot_options is not None:
                outputs = self._groot_model.get_action(groot_inputs, options=groot_options)
            else:
                outputs = self._groot_model.get_action(groot_inputs)

        actions = outputs.get("action_pred")

        prediction_horizon = self._resolve_prediction_horizon(actions)
        actions = actions[:, :prediction_horizon]

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """从动作队列中选取单个动作。"""
        if getattr(self.config, "use_relative_actions", False):
            raise NotImplementedError(
                "GrootPolicy.select_action does not support relative-action policies because cached "
                "relative chunk actions can be decoded against newer observation states. Use "
                "predict_action_chunk and postprocess the full chunk before queuing actions, or use "
                "the RTC/chunked rollout inference path."
            )

        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions[:, : self._action_queue_steps].transpose(0, 1))
        return self._action_queue.popleft()
