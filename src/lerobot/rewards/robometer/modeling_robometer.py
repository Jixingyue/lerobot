# Copyright 2026 Anthony Liang, Yigit Korkmaz, Stephen Tu, Erdem Bıyık, Jesse Zhang
# and The HuggingFace Inc. team. All rights reserved.
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

"""ROBOMETER: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons.

Paper:         https://arxiv.org/abs/2603.02115
Project:       https://robometer.github.io
Original code: https://github.com/aliang8/robometer
Model:         https://huggingface.co/robometer/Robometer-4B

Robometer 是一个通用的、以视频-语言为输入的奖励模型，构建在
``Qwen/Qwen3-VL-4B-Instruct`` 之上。它采用双重奖励预测目标进行训练：

- 帧级 progress 损失，在专家数据上锚定奖励幅度。
- 轨迹比较偏好损失，对共享同一指令的各轨迹施加全局排序约束。

为支持下游强化学习，它还会预测帧级的二元 success。训练提示词中
插入了三个可学习的 token：

- ``<|prog_token|>``：位于每帧之后，用于读取逐帧的 progress 和 success。
- ``<|pref_token|>``：位于末尾，用于读取成对偏好（仅训练时使用）。
- ``<|split_token|>``：位于偏好样本中两条轨迹之间（仅训练时使用）。

Progress 被建模为在 ``[0, 1]`` 区间内 ``progress_discrete_bins`` 个
均匀分布中心点上的分类分布（C51 风格），连续估计值则恢复为
这些中心点经 softmax 加权的均值 —— 参见
:func:`convert_bins_to_continuous`。

本 LeRobot 移植版**仅支持推理**：偏好头保留在 state dict 中，
以便与已发布的 ``Robometer-4B`` 检查点保持字节级一致，
但 :meth:`RobometerRewardModel.compute_reward` 不会查询它；
该方法根据 :attr:`RobometerConfig.reward_output` 的设置，
返回最后一帧的 progress（截断到 ``[0, 1]``）或经 sigmoid 处理的
success 概率。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
from lerobot.utils.constants import OBS_PREFIX
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModelForImageTextToText
else:
    AutoModelForImageTextToText = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Robometer 预编码 Qwen-VL 观测张量的命名空间。
ROBOMETER_FEATURE_PREFIX = f"{OBS_PREFIX}robometer."
ROBOMETER_QWEN_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
    "mm_token_type_ids",
)
ROBOMETER_METADATA_KEYS = (
    "prog_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "video_merge_size",
)
ROBOMETER_INPUT_KEYS = ROBOMETER_QWEN_INPUT_KEYS + ROBOMETER_METADATA_KEYS


def convert_bins_to_continuous(bin_logits: Tensor) -> Tensor:
    """将各分箱的 logits 折叠为 ``[0, 1]`` 内的单个值。

    离散 progress 头为每帧输出 ``num_bins`` 个 logits。各分箱是
    ``[0, 1]`` 内均匀分布的中心点；连续预测值是这些中心点
    经 softmax 加权的均值。
    """
    bin_probs = torch.softmax(bin_logits, dim=-1)
    num_bins = bin_logits.shape[-1]
    bin_centers = torch.linspace(0.0, 1.0, num_bins, device=bin_logits.device, dtype=bin_logits.dtype)
    return (bin_probs * bin_centers).sum(dim=-1)


def _squeeze_last_safe(x: Tensor) -> Tensor:
    """仅当末尾存在单元素维度时才将其去除。"""
    return x.squeeze(-1) if x.ndim > 1 and x.shape[-1] == 1 else x


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unknown torch dtype: {name!r}")


class RobometerPredictionHead(nn.Sequential):
    """用于 Robometer 的 progress / success / preference 输出的小型 MLP 头。"""

    def __init__(self, hidden_dim: int, output_size: int, *, dropout: float, with_sigmoid: bool) -> None:
        layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_size),
        ]
        if with_sigmoid:
            layers.append(nn.Sigmoid())
        super().__init__(*layers)


def decode_progress_outputs(
    progress_logits: Tensor | None,
    success_logits: Tensor | None,
    *,
    is_discrete_mode: bool,
) -> dict[str, list[list[float]]]:
    """将 RBM 头的输出解码为逐帧浮点数。

    Args:
        progress_logits: ``(B, T)``（连续）或 ``(B, T, num_bins)``（离散）。
        success_logits: ``(B, T)`` 原始 logits，经 ``sigmoid`` 转为概率。
        is_discrete_mode: 若为 True，则对 progress logits 在分箱上执行
            softmax，并通过 :func:`convert_bins_to_continuous` 投影到分箱中心。

    Returns:
        包含 ``progress_pred`` 和 ``success_probs`` 的字典，二者均为
        长度为 ``B`` 的列表，元素是逐帧浮点数列表。
    """
    progress_pred: list[list[float]] = []
    success_probs: list[list[float]] = []

    if progress_logits is not None:
        for sample_logits in progress_logits:
            if is_discrete_mode:
                continuous = convert_bins_to_continuous(sample_logits.detach().float().cpu())
                progress_pred.append(continuous.flatten().tolist())
            else:
                progress_pred.append(sample_logits.detach().float().cpu().flatten().tolist())

    if success_logits is not None:
        for sample_logits in success_logits:
            success_probs.append(torch.sigmoid(sample_logits.detach().float().cpu()).flatten().tolist())

    return {"progress_pred": progress_pred, "success_probs": success_probs}


class RobometerRewardModel(PreTrainedRewardModel):
    """Robometer (RBM) 奖励模型 —— 仅支持推理的 LeRobot 移植版。

    包装了一个 Qwen-VL 骨干（默认：``Qwen/Qwen3-VL-4B-Instruct``），
    并带有论文中的三个预测头（progress、success、preference）。推理时
    只查询 progress 和 success 头；preference 头保留在模块上，
    以便已发布的 ``Robometer-4B`` safetensors 可以原样加载。
    """

    name = "robometer"
    config_class = RobometerConfig

    def __init__(self, config: RobometerConfig, *, dropout: float = 0.1) -> None:
        require_package("transformers", extra="robometer")
        super().__init__(config)
        self.config = config

        # 两种骨干构建路径（EO-1 风格，根据 ``pretrained_path`` 分支）：
        #
        #   - 全新训练（``pretrained_path is None``）：下载基础 Qwen 权重，
        #     并调整嵌入表大小以匹配
        #     ``vlm_config.text_config.vocab_size`` —— 该值在
        #     ``RobometerConfig.__post_init__`` 中确定性地填充为
        #     ``len(tokenizer) + len(ROBOMETER_SPECIAL_TOKENS)``
        #
        #   - 加载已保存的检查点（设置了 ``pretrained_path``）：通过
        #     ``AutoModelForImageTextToText.from_config`` 根据 ``vlm_config``
        #     重建空架构，使后续加载 ``model.safetensors`` 时能直接以
        #     正确的形状填充 —— 无需重复下载 Qwen 权重。
        torch_dtype = _torch_dtype(config.torch_dtype)
        if config.pretrained_path is None:
            self.model = AutoModelForImageTextToText.from_pretrained(
                config.base_model_id,
                dtype=torch_dtype,
                trust_remote_code=True,
            )
            target_vocab = config.vlm_config["text_config"]["vocab_size"]
            self.model.resize_token_embeddings(target_vocab)
        else:
            self.model = AutoModelForImageTextToText.from_config(
                config.vlm_backbone_config,
                dtype=torch_dtype,
                trust_remote_code=True,
            )

        # Robometer 支持的所有 Qwen-VL 骨干都暴露 `text_config.hidden_size`。
        # 回退到顶层的 `hidden_size`，这样未来的非多模态变体也能正常解析。
        backbone_config = self.model.config
        text_config = getattr(backbone_config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None) if text_config is not None else None
        if hidden_size is None:
            hidden_size = getattr(backbone_config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError(
                f"Could not infer hidden_size from backbone config of {config.base_model_id}"
            )
        hidden_dim = int(hidden_size)

        # Robometer 的三个预测头 + 帧池化注意力。
        progress_output = config.progress_discrete_bins if config.use_discrete_progress else 1
        self.progress_head = RobometerPredictionHead(
            hidden_dim,
            progress_output,
            dropout=dropout,
            with_sigmoid=not config.use_discrete_progress,
        )
        self.preference_head = RobometerPredictionHead(hidden_dim, 1, dropout=dropout, with_sigmoid=False)
        self.success_head = RobometerPredictionHead(hidden_dim, 1, dropout=dropout, with_sigmoid=False)
        self.frame_pool_attn = nn.Linear(hidden_dim, 1, bias=False)

        # 匹配已加载基础模型的 dtype，使权重加载成为一次无操作式的类型转换。
        model_dtype = next(self.model.parameters()).dtype
        self.progress_head.to(dtype=model_dtype)
        self.preference_head.to(dtype=model_dtype)
        self.success_head.to(dtype=model_dtype)
        self.frame_pool_attn.to(dtype=model_dtype)

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        inputs = {
            key: batch[f"{ROBOMETER_FEATURE_PREFIX}{key}"]
            for key in ROBOMETER_INPUT_KEYS
            if f"{ROBOMETER_FEATURE_PREFIX}{key}" in batch
        }
        if "input_ids" not in inputs:
            raise KeyError(
                f"Robometer batch missing pre-encoded inputs (expected "
                f"`{ROBOMETER_FEATURE_PREFIX}input_ids`). Make sure the "
                "RobometerEncoderProcessorStep ran before `compute_reward`."
            )

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

        self.eval()
        with torch.no_grad():
            progress_logits, success_logits = self._compute_rbm_logits(inputs)

        decoded = decode_progress_outputs(
            progress_logits,
            success_logits,
            is_discrete_mode=self.config.use_discrete_progress,
        )
        values = (
            decoded["success_probs"] if self.config.reward_output == "success" else decoded["progress_pred"]
        )

        rewards = torch.stack([torch.as_tensor(seq, dtype=torch.float32)[-1] for seq in values])
        if self.config.reward_output == "success":
            rewards = (rewards > self.config.success_threshold).float()
        else:
            # 与上游 Robometer 的 ``extract_rewards_from_output`` 保持一致：
            # 逐帧 progress 预测在返回前会被截断到 ``[0, 1]``。
            rewards = rewards.clamp(0.0, 1.0)
        return rewards.to(self.config.device or "cpu")

    def _compute_rbm_logits(
        self,
        inputs: dict[str, Any],
    ) -> tuple[Tensor, Tensor]:
        """运行 Qwen3-VL 骨干并应用 Robometer 的各预测头。

        ``inputs`` 是 :class:`RobometerEncoderProcessorStep` 生成的编码后批次。
        它既携带 Qwen 张量，也携带 Robometer 特有的元数据
        （``prog_token_id``、``vision_start_token_id``、``vision_end_token_id``、
        ``video_merge_size``）—— 元数据会在此处被弹出，其余部分可以
        直接转发给 Qwen 模型。

        返回 ``(progress_logits, success_logits)``。形状：

        - ``progress_logits``：``(B, T)``（连续）或 ``(B, T, num_bins)``（离散）。
        - ``success_logits``：``(B, T)`` 原始 logits（解码时才执行 sigmoid）。
        """
        prog_token_id = inputs.pop("prog_token_id", None)
        vision_start_token_id = inputs.pop("vision_start_token_id", None)
        vision_end_token_id = inputs.pop("vision_end_token_id", None)
        video_merge_size = inputs.pop("video_merge_size", 14)

        # Qwen3-VL 不能可靠地填充 `last_hidden_state`；因此请求完整的
        # hidden-state 元组并取最后一层。这与上游 Robometer 的
        # `RBM.forward_qwen`（main 分支）中的 `is_qwen3` 路径一致。
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_state = (
            outputs.hidden_states[-1]
            if getattr(outputs, "hidden_states", None)
            else outputs.last_hidden_state
        )

        input_ids = inputs["input_ids"]
        if self.config.use_per_frame_progress_token:
            if prog_token_id is None:
                raise KeyError("`prog_token_id` missing in batch (run RobometerEncoderProcessorStep first)")
            return self._process_token_extraction(hidden_state, input_ids, prog_token_id=prog_token_id)
        if self.config.use_multi_image:
            if vision_start_token_id is None or vision_end_token_id is None:
                raise KeyError(
                    "`vision_start_token_id` / `vision_end_token_id` missing in batch "
                    "(run RobometerEncoderProcessorStep first)"
                )
            return self._process_multi_image_frames(
                hidden_state,
                input_ids,
                start_id=vision_start_token_id,
                end_id=vision_end_token_id,
            )
        video_grid_thw = inputs.get("video_grid_thw")
        if video_grid_thw is None:
            raise ValueError("video_grid_thw is required for video-mode Robometer inference")
        if vision_start_token_id is None:
            raise KeyError("`vision_start_token_id` missing in batch")
        return self._process_video_frames(
            hidden_state,
            input_ids,
            video_grid_thw,
            start_id=vision_start_token_id,
            merge_size=video_merge_size,
        )

    def _apply_heads_to_hidden_states(self, frame_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """将 progress + success 头应用于帧嵌入张量。"""
        progress_out = self.progress_head(frame_embeddings)
        progress = progress_out if self.config.use_discrete_progress else _squeeze_last_safe(progress_out)
        success = _squeeze_last_safe(self.success_head(frame_embeddings))
        return progress, success

    def _process_token_extraction(
        self,
        hidden_state: Tensor,
        input_ids: Tensor,
        *,
        prog_token_id: int,
    ) -> tuple[Tensor, Tensor]:
        """从 ``<|prog_token|>`` 位置提取逐帧的 progress/success。"""
        token_mask = input_ids == prog_token_id
        batch_indices, positions = token_mask.nonzero(as_tuple=True)
        if positions.numel() == 0:
            raise ValueError("`<|prog_token|>` not found in any sequence")

        per_sample_hidden = [
            hidden_state[i, positions[batch_indices == i]] for i in range(input_ids.shape[0])
        ]
        progress_list, success_list = [], []
        for embeddings in per_sample_hidden:
            if embeddings.shape[0] == 0:
                raise ValueError("`<|prog_token|>` missing in a sequence")
            progress, success = self._apply_heads_to_hidden_states(embeddings)
            progress_list.append(progress)
            success_list.append(success)

        return torch.stack(progress_list), torch.stack(success_list)

    def _process_multi_image_frames(
        self,
        hidden_state: Tensor,
        input_ids: Tensor,
        *,
        start_id: int,
        end_id: int,
    ) -> tuple[Tensor, Tensor]:
        """多图模式（Qwen-VL）下的逐帧 progress/success。"""
        progress_list, success_list = [], []
        for batch_idx in range(input_ids.shape[0]):
            seq_ids = input_ids[batch_idx]
            seq_hidden = hidden_state[batch_idx]
            frame_embeddings = self._extract_hidden_states_from_token_pairs(
                seq_hidden, seq_ids, start_id, end_id
            )
            progress, success = self._apply_heads_to_hidden_states(frame_embeddings)
            progress_list.append(progress)
            success_list.append(success)

        return torch.stack(progress_list), torch.stack(success_list)

    def _extract_hidden_states_from_token_pairs(
        self,
        hidden_state: Tensor,
        input_ids: Tensor,
        start_id: int,
        end_id: int,
    ) -> Tensor:
        start_positions = (input_ids == start_id).nonzero(as_tuple=True)[0]
        end_positions = (input_ids == end_id).nonzero(as_tuple=True)[0]
        if start_positions.numel() == 0:
            raise ValueError("`<|vision_start|>` not found in sequence")
        if start_positions.numel() != end_positions.numel():
            raise ValueError(
                f"Mismatched vision token counts: {start_positions.numel()} start vs "
                f"{end_positions.numel()} end"
            )

        frames: list[Tensor] = []
        for start, end in zip(start_positions.tolist(), end_positions.tolist(), strict=True):
            if start >= end:
                raise ValueError(f"Invalid vision token pair: start={start} end={end}")
            patch_tokens = hidden_state[start + 1 : end]
            if patch_tokens.shape[0] == 0:
                frames.append((hidden_state[start] + hidden_state[end]) / 2.0)
                continue

            pooling = self.config.frame_pooling
            if pooling == "mean":
                frames.append(patch_tokens.mean(dim=0))
            elif pooling == "boundary":
                frames.append(patch_tokens[-1])
            else:  # attention
                scores = (
                    self.frame_pool_attn(patch_tokens).squeeze(-1)
                    / self.config.frame_pooling_attn_temperature
                )
                weights = torch.softmax(scores, dim=0).unsqueeze(-1)
                frames.append((weights * patch_tokens).sum(dim=0))

        return torch.stack(frames)

    def _process_video_frames(
        self,
        hidden_state: Tensor,
        input_ids: Tensor,
        video_grid_thw: Tensor,
        *,
        start_id: int,
        merge_size: int,
    ) -> tuple[Tensor, Tensor]:
        """视频模式（Qwen-VL）下的逐帧 progress/success。"""
        progress_list, success_list = [], []
        for batch_idx in range(input_ids.shape[0]):
            seq_ids = input_ids[batch_idx]
            seq_hidden = hidden_state[batch_idx]
            start_positions = (seq_ids == start_id).nonzero(as_tuple=True)[0]
            if start_positions.numel() == 0:
                raise ValueError("`<|vision_start|>` not found in sequence")
            t_dim, h_dim, w_dim = (int(x) for x in video_grid_thw[batch_idx].tolist())
            tokens_per_frame = (h_dim * w_dim) // (merge_size**2)

            cursor = start_positions[0].item()
            frame_embeddings: list[Tensor] = []
            for _ in range(t_dim):
                if self.config.average_temporal_patches:
                    patch = seq_hidden[cursor : cursor + tokens_per_frame]
                    frame_embeddings.append(patch.mean(dim=0))
                else:
                    frame_embeddings.append(seq_hidden[cursor + tokens_per_frame])
                cursor += tokens_per_frame

            stacked = torch.stack(frame_embeddings)
            progress, success = self._apply_heads_to_hidden_states(stacked)
            progress_list.append(progress)
            success_list.append(success)

        return torch.stack(progress_list), torch.stack(success_list)
