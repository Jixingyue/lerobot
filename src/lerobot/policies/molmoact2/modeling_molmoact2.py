# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""LeRobot 的 MolmoAct2 策略。

MolmoAct2 是 Allen AI 推出的基于 VLM 的机器人策略，它将 Molmo 视觉-语言
骨干网络与逐层的 flow-matching action expert 相结合，用于连续动作生成，
并附带一个可选的离散动作 token 头。本模块将内置的 HF 模型实现
（``molmoact2_hf_model/``）封装为 LeRobot 的 ``PreTrainedPolicy`` 接口。

论文：https://allenai.org/blog/molmoact2
代码：https://github.com/allenai/molmoact2
"""

from __future__ import annotations

import json
import logging
import os
import re
import types
from collections import deque
from collections.abc import Iterator
from contextlib import nullcontext, suppress
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from torch.distributions import Beta

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import (
    _peft_available,
    _scipy_available,
    _transformers_available,
    require_package,
)

from ..rtc.modeling_rtc import RTCProcessor
from .configuration_molmoact2 import MolmoAct2Config
from .utils_molmoact2 import (
    hf_token as _hf_token,
    position_ids_from_attention_mask as _position_ids_from_attention_mask,
    resolve_checkpoint_location as _resolve_checkpoint_location,
)

if TYPE_CHECKING or _peft_available:
    from peft import LoraConfig, get_peft_model
else:
    LoraConfig = None
    get_peft_model = None

logger = logging.getLogger(__name__)


def _torch_dtype(dtype: str) -> torch.dtype:
    """将 dtype 名称字符串转换为 torch.dtype。"""
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _call_module_without_gradient_checkpointing_layer(
    module: torch.nn.Module,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """绕过子类的 ``__call__`` 包装，直接调用普通的 ``nn.Module`` 分发逻辑。

    Transformers 的 decoder 层通过重写 ``__call__`` 来实现激活检查点。
    MolmoAct2 的联合连续路径会把配对的文本层+动作层作为一个整体做检查点，
    因此在联合检查点内部调用那个重写方法会导致文本层被重复检查点两次。
    通过基类实现进行分发只会跳过 Transformers 的包装层，同时保留模块编译、
    hooks 和 autograd。
    """
    return torch.nn.Module.__call__(module, *args, **kwargs)


def _next_decode_position_ids_from_attention_mask(attention_mask: Tensor | None) -> Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError(
            "MolmoAct2 cached decoding requires a 2D attention mask, "
            f"got shape {tuple(attention_mask.shape)}."
        )
    return attention_mask.to(dtype=torch.long).sum(dim=-1, keepdim=True)


def _set_dynamo_lru_cache(enabled: bool) -> None:
    """设置 Dynamo 编译图缓存的排序策略。

    PyTorch 的激活检查点会在反向传播期间重放 eager 函数。对于重复出现的
    编译块，LRU 重排序可能使该重放选择一个与原始前向不同但仍有效的图，
    从而改变保存张量的签名。PyTorch 将这个窄开关作为
    pytorch/pytorch#166926 的短期修复方案暴露出来。
    """
    torch_c = getattr(torch, "_C", None)
    dynamo_c = getattr(torch_c, "_dynamo", None)
    eval_frame = getattr(dynamo_c, "eval_frame", None)
    set_lru_cache = getattr(eval_frame, "_set_lru_cache", None)
    if not callable(set_lru_cache):
        raise RuntimeError(
            "MolmoAct2 compile_model=true with gradient_checkpointing=true requires "
            "a PyTorch build that exposes torch._C._dynamo.eval_frame._set_lru_cache. "
            "Disable compile_model or upgrade PyTorch."
        )
    set_lru_cache(enabled)


def _disable_dynamo_ddp_optimizer() -> None:
    """使 DDP 前向与重计算阶段的块编译保持一致。

    Dynamo 的 DDP 优化器只在执行处于 DDP 前向上下文内部时才会改变图划分。
    而激活检查点的重计算发生在反向传播期间，位于该上下文之外。MolmoAct2
    已经对单个 transformer 块进行编译，因此禁用这个额外的图分割器可以保持
    块图的稳定，同时常规的 C++ DDP reducer 仍会同步梯度。
    """
    dynamo = getattr(torch, "_dynamo", None)
    dynamo_config = getattr(dynamo, "config", None)
    if dynamo_config is None or not hasattr(dynamo_config, "optimize_ddp"):
        raise RuntimeError(
            "MolmoAct2 compile_model=true with gradient_checkpointing=true requires "
            "a PyTorch build that exposes torch._dynamo.config.optimize_ddp. "
            "Disable compile_model or upgrade PyTorch."
        )
    dynamo_config.optimize_ddp = False


def _mark_action_context_dynamic(
    key_states: Tensor,
    value_states: Tensor,
    attention_mask: Tensor | None,
) -> None:
    """仅将可变的交叉注意力序列维度标记为动态。

    MolmoAct2 训练会固定每个 rank 的批大小、flow 时间步数量和动作时长。
    编码器上下文长度是唯一在不同批次之间变化的 action-block 输入维度。
    显式标记该维度可以让 Dynamo 构建一个可复用的图，而不必把其他本来
    静态的动作维度都变成符号化的。
    """
    dynamo = getattr(torch, "_dynamo", None)
    mark_dynamic = getattr(dynamo, "mark_dynamic", None)
    if not callable(mark_dynamic):
        raise RuntimeError(
            "MolmoAct2 compile_model=true with gradient_checkpointing=true requires "
            "a PyTorch build that exposes torch._dynamo.mark_dynamic. "
            "Disable compile_model or upgrade PyTorch."
        )
    mark_dynamic(key_states, 1)
    mark_dynamic(value_states, 1)
    if attention_mask is not None:
        mark_dynamic(attention_mask, attention_mask.ndim - 1)


if TYPE_CHECKING or _transformers_available:
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

    from .molmoact2_hf_model.configuration_molmoact2 import MolmoAct2Config as HFMolmoAct2Config
    from .molmoact2_hf_model.modeling_molmoact2 import (
        ActionExpertRMSNorm,
        MolmoAct2ForConditionalGeneration,
        MolmoAct2RMSNorm,
        MolmoAct2RotaryEmbedding,
    )
else:
    SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
    SAFE_WEIGHTS_NAME = "model.safetensors"
    ActionExpertRMSNorm = None
    HFMolmoAct2Config = None
    MolmoAct2ForConditionalGeneration = None
    MolmoAct2RMSNorm = None
    MolmoAct2RotaryEmbedding = None

if TYPE_CHECKING or (_transformers_available and _scipy_available):
    from .molmoact2_hf_model.action_tokenizer import UniversalActionProcessor
else:
    UniversalActionProcessor = None

_MODEL_INPUT_KEYS = {
    "input_ids",
    "pixel_values",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
    "pixel_values_videos",
    "video_token_pooling",
    "video_grids",
    "attention_mask",
    "position_ids",
    "past_key_values",
    "token_type_ids",
    "inputs_embeds",
}


def _load_hf_norm_metadata_for_tag(
    checkpoint_path: str,
    *,
    revision: str | None,
    force_download: bool,
    norm_tag: str | None,
    norm_stats_path: str | None = None,
) -> dict[str, Any]:
    """从检查点的 ``norm_stats.json`` 中读取按 tag 划分的元数据。"""
    norm_tag = str(norm_tag or "").strip()
    if not norm_tag:
        return {}
    from pathlib import Path

    norm_stats_filename = "norm_stats.json"
    if norm_stats_path is not None:
        stats_path = Path(norm_stats_path).expanduser().resolve(strict=True)
        norm_stats_filename = stats_path.name
        if not stats_path.is_file():
            raise ValueError(f"MolmoAct2 norm_stats_path must be a JSON file, got {stats_path}.")
    else:
        checkpoint_location = Path(
            _resolve_checkpoint_location(
                checkpoint_path,
                revision=revision,
                force_download=force_download,
            )
        )
        config_path = checkpoint_location / "config.json"
        if config_path.exists():
            with suppress(OSError, json.JSONDecodeError):
                norm_stats_filename = str(
                    json.loads(config_path.read_text()).get("norm_stats_filename") or norm_stats_filename
                )
        stats_path = checkpoint_location / norm_stats_filename
    if not stats_path.exists():
        raise FileNotFoundError(
            f"MolmoAct2 HF checkpoint is missing {norm_stats_filename!r}; cannot resolve norm_tag={norm_tag!r}."
        )
    payload = json.loads(stats_path.read_text())
    metadata_by_tag = payload.get("metadata_by_tag")
    if not isinstance(metadata_by_tag, dict):
        raise ValueError(f"MolmoAct2 norm stats file {stats_path} has no metadata_by_tag mapping.")
    metadata = metadata_by_tag.get(norm_tag)
    if not isinstance(metadata, dict):
        available = sorted(str(tag) for tag in metadata_by_tag)
        raise ValueError(f"Unknown MolmoAct2 norm_tag={norm_tag!r}. Available tags: {available}.")
    return metadata


def _apply_norm_tag_metadata(config: MolmoAct2Config) -> None:
    """根据检查点的 norm-tag 元数据填充配置字段。"""
    if config.pretrained_path is not None:
        # LeRobot 检查点持久化了已解析好的本体（embodiment）元数据，
        # 且其保存的 processor 自行管理归一化。通过 policy.path 重新打开
        # 检查点时，绝不能再查询原始的 HF/本地归一化统计来源。
        return
    if not str(config.norm_tag or "").strip():
        return
    metadata = _load_hf_norm_metadata_for_tag(
        config.checkpoint_path,
        revision=config.checkpoint_revision,
        force_download=bool(config.checkpoint_force_download),
        norm_tag=config.norm_tag,
        norm_stats_path=config.norm_stats_path,
    )
    if metadata.get("action_horizon") is not None:
        config.chunk_size = int(metadata["action_horizon"])
    if metadata.get("n_action_steps") is not None:
        config.n_action_steps = int(metadata["n_action_steps"])
    if not config.setup_type and metadata.get("setup_type") is not None:
        config.setup_type = str(metadata["setup_type"])
    if not config.control_mode and metadata.get("control_mode") is not None:
        config.control_mode = str(metadata["control_mode"])


def _saved_policy_action_mode(config: MolmoAct2Config) -> str | None:
    """从 LeRobot 保存的检查点的 ``config.json`` 中读取动作模式。"""
    from pathlib import Path

    pretrained_path = getattr(config, "pretrained_path", None)
    if pretrained_path is None:
        return None
    config_path = Path(pretrained_path) / "config.json"
    if not config_path.exists():
        return None
    try:
        mode = json.loads(config_path.read_text()).get("action_mode")
    except (OSError, json.JSONDecodeError):
        return None
    if mode in {"continuous", "discrete", "both"}:
        return str(mode)
    return None


def _training_action_mode(config: MolmoAct2Config, saved_policy_action_mode: str | None = None) -> str:
    return saved_policy_action_mode or config.action_mode


def _validate_inference_action_mode(
    config: MolmoAct2Config, saved_policy_action_mode: str | None = None
) -> None:
    """检查所请求的推理模式是否与训练模式兼容。"""
    requested_mode = config.inference_action_mode
    if requested_mode is None:
        return
    training_mode = _training_action_mode(config, saved_policy_action_mode)
    if requested_mode == "continuous" and training_mode == "discrete":
        raise ValueError(
            "MolmoAct2 checkpoint was trained with action_mode='discrete' and cannot run "
            "continuous inference."
        )
    if requested_mode == "discrete" and training_mode == "continuous":
        raise ValueError(
            "MolmoAct2 checkpoint was trained with action_mode='continuous' and cannot run "
            "discrete inference. Train with action_mode='both' or action_mode='discrete' first."
        )


def _validate_checkpoint_action_mode(
    config: MolmoAct2Config,
    checkpoint_action_mode: str,
    *,
    has_action_expert: bool,
) -> None:
    """检查检查点的动作模式是否与配置兼容。"""
    if config.action_mode == "both" and checkpoint_action_mode != "both":
        raise ValueError(
            f"action_mode='both' requires checkpoint action_mode='both', got {checkpoint_action_mode!r}."
        )
    if config.action_mode == "discrete" and checkpoint_action_mode not in {"discrete", "both"}:
        raise ValueError(
            f"action_mode='discrete' requires checkpoint action_mode in {{'discrete', 'both'}}, "
            f"got {checkpoint_action_mode!r}."
        )
    if config.action_mode in {"continuous", "both"} and not has_action_expert:
        raise ValueError("Continuous MolmoAct2 training requires an action expert checkpoint.")


def _resolve_inference_action_mode(
    config: MolmoAct2Config,
    requested_mode: str | None,
    saved_policy_action_mode: str | None = None,
) -> str:
    """解析最终的推理动作模式，并校验兼容性。"""
    training_mode = _training_action_mode(config, saved_policy_action_mode)
    if requested_mode is None:
        requested_mode = config.inference_action_mode
    if requested_mode is None:
        raise ValueError(
            "MolmoAct2 inference requires `inference_action_mode` to be set explicitly "
            "to either 'continuous' or 'discrete'."
        )
    if requested_mode not in {"continuous", "discrete"}:
        raise ValueError("MolmoAct2 inference_action_mode must be either 'continuous' or 'discrete'.")
    if requested_mode == "continuous" and training_mode == "discrete":
        raise ValueError("MolmoAct2 action_mode='discrete' checkpoint cannot run continuous inference.")
    if requested_mode == "discrete" and training_mode == "continuous":
        raise ValueError("MolmoAct2 action_mode='continuous' checkpoint cannot run discrete inference.")
    return requested_mode


def _strict_load_safetensors_weights(model: torch.nn.Module, checkpoint_location: str) -> None:
    index_path = os.path.join(checkpoint_location, SAFE_WEIGHTS_INDEX_NAME)
    single_file_path = os.path.join(checkpoint_location, SAFE_WEIGHTS_NAME)
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        loaded_keys = set(weight_map)
        model_keys = set(model.state_dict())
        missing_keys = sorted(model_keys - loaded_keys)
        unexpected_keys = sorted(loaded_keys - model_keys)
        if missing_keys or unexpected_keys:
            message = ["MolmoAct2 safetensors do not match the local model implementation."]
            if missing_keys:
                message.append(f"Missing keys: {missing_keys[:8]}")
            if unexpected_keys:
                message.append(f"Unexpected keys: {unexpected_keys[:8]}")
            raise RuntimeError(" ".join(message))
        for shard_file in sorted(set(weight_map.values())):
            state_dict = load_safetensors_file(os.path.join(checkpoint_location, shard_file), device="cpu")
            model.load_state_dict(state_dict, strict=False)
            del state_dict
        return
    if os.path.isfile(single_file_path):
        state_dict = load_safetensors_file(single_file_path, device="cpu")
        model.load_state_dict(state_dict, strict=True)
        return
    raise FileNotFoundError(
        f"MolmoAct2 checkpoint at {checkpoint_location} must contain {SAFE_WEIGHTS_NAME} "
        f"or {SAFE_WEIGHTS_INDEX_NAME}."
    )


def _sample_beta_timesteps(
    *,
    batch_size: int,
    device: torch.device,
    cutoff: float,
    time_offset: float,
    time_scale: float,
    alpha: float,
    beta: float,
) -> Tensor:
    if cutoff < time_offset:
        raise ValueError(f"flow-matching cutoff must be >= time_offset, got {cutoff} < {time_offset}")
    if time_scale <= 0:
        raise ValueError(f"flow-matching time_scale must be > 0, got {time_scale}")
    upper = min(cutoff, time_offset + time_scale)
    dist = Beta(torch.tensor(alpha, device=device), torch.tensor(beta, device=device))
    samples = dist.sample((batch_size,))
    scale = upper - time_offset
    if scale == 0:
        return torch.full((batch_size,), time_offset, device=device, dtype=samples.dtype)
    return time_offset + scale * samples


def _mask_discrete_action_spans(
    *,
    input_ids: Tensor,
    mask: Tensor,
    start_token_id: int | None,
    end_token_id: int | None,
) -> Tensor:
    if start_token_id is None or end_token_id is None:
        return mask
    mask = mask.clone()
    for batch_idx in range(input_ids.shape[0]):
        row = input_ids[batch_idx]
        starts = (row == int(start_token_id)).nonzero(as_tuple=False).flatten().tolist()
        ends = (row == int(end_token_id)).nonzero(as_tuple=False).flatten().tolist()
        end_ptr = 0
        for start in starts:
            while end_ptr < len(ends) and ends[end_ptr] < start:
                end_ptr += 1
            if end_ptr >= len(ends):
                mask[batch_idx, start:] = False
                break
            end = int(ends[end_ptr])
            mask[batch_idx, start : end + 1] = False
            end_ptr += 1
    return mask


def _drop_trivial_attention_mask(model_inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    attention_mask = model_inputs.get("attention_mask")
    if torch.is_tensor(attention_mask) and bool(attention_mask.to(dtype=torch.bool).all().item()):
        model_inputs = dict(model_inputs)
        model_inputs.pop("attention_mask", None)
    return model_inputs


def _expand_mask(mask: Tensor | None, num_flow_timesteps: int) -> Tensor | None:
    if mask is None:
        return None
    return (
        mask.unsqueeze(1)
        .expand(-1, num_flow_timesteps, *([-1] * (mask.ndim - 1)))
        .reshape(mask.shape[0] * num_flow_timesteps, *mask.shape[1:])
    )


def _action_dim_valid_mask(target: Tensor, action_dim_is_pad: Tensor | None) -> Tensor | None:
    if action_dim_is_pad is None:
        return None
    mask = ~action_dim_is_pad.to(device=target.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"action_dim_is_pad width {mask.shape[-1]} does not match target width {target.shape[-1]}."
        )
    if mask.shape[0] == 1 and target.shape[0] != 1:
        mask = mask.expand(target.shape[0], -1)
    if mask.shape[0] != target.shape[0]:
        raise ValueError(
            f"action_dim_is_pad batch {mask.shape[0]} does not match target batch {target.shape[0]}."
        )
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(1)
    return mask


def _mask_action_dim_tensor(tensor: Tensor, action_dim_is_pad: Tensor | None) -> Tensor:
    if action_dim_is_pad is None:
        return tensor
    valid_mask = _action_dim_valid_mask(tensor, action_dim_is_pad)
    if valid_mask is None:
        return tensor
    return tensor.masked_fill(~valid_mask, 0)


def _apply_action_dim_padding_mask(loss: Tensor, action_dim_is_pad: Tensor | None) -> Tensor:
    valid_mask = _action_dim_valid_mask(loss, action_dim_is_pad)
    if valid_mask is None:
        return loss
    valid = valid_mask.to(dtype=loss.dtype)
    denom = valid.sum(dim=-1).clamp_min(1.0)
    return (loss * valid).sum(dim=-1) / denom


def _apply_action_chunk_padding_mask(loss: Tensor, action_horizon_is_pad: Tensor | None) -> Tensor:
    if action_horizon_is_pad is None:
        return loss
    valid_action = (
        (~action_horizon_is_pad.to(device=loss.device, dtype=torch.bool)).unsqueeze(1).unsqueeze(-1)
    )
    return loss * valid_action


def _combine_rollout_seeds(first_seed: int, batch_size: int) -> int:
    seed = 0
    for idx in range(batch_size):
        seed = (seed + (idx + 1) * (first_seed + idx)) % (2**63 - 1)
    return seed


def _rollout_task_signature(batch: dict[str, Any]) -> tuple[Any, ...] | None:
    task = batch.get("task")
    if task is None:
        task = batch.get("observation.language")
    if task is None:
        return None
    if isinstance(task, str):
        return (task,)
    if isinstance(task, (list, tuple)):
        return tuple(str(item) for item in task)
    return (str(task),)


def _extract_discrete_token_bins(
    generated_ids: list[int],
    start_token_id: int,
    end_token_id: int,
    token_id_to_bin: dict[int, int],
) -> list[int]:
    start_idx = None
    end_idx = None
    for idx, token_id in enumerate(generated_ids):
        if token_id == start_token_id:
            start_idx = idx
            break
    if start_idx is not None:
        for idx in range(start_idx + 1, len(generated_ids)):
            if generated_ids[idx] == end_token_id:
                end_idx = idx
                break
    span_start = 0 if start_idx is None else start_idx + 1
    span_end = len(generated_ids) if end_idx is None else end_idx
    return [
        int(token_id_to_bin[token_id])
        for token_id in generated_ids[span_start:span_end]
        if token_id in token_id_to_bin
    ]


def _weighted_mean(values: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        numerator = values.sum()
        denominator = values.new_tensor(float(values.numel()))
    else:
        weights = weights.to(device=values.device, dtype=values.dtype)
        numerator = torch.dot(values, weights)
        denominator = weights.sum()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        denominator = denominator.detach().clone()
        torch.distributed.all_reduce(denominator, op=torch.distributed.ReduceOp.SUM)
        denominator = denominator / world_size
    return numerator / denominator.clamp_min(1.0)


def _weighted_per_example(
    values: Tensor,
    weights: Tensor | None,
    example_indices: Tensor,
    batch_size: int,
) -> Tensor:
    values = values.float()
    if weights is None:
        weights = torch.ones_like(values)
    else:
        weights = weights.to(device=values.device, dtype=values.dtype)
    loss_sum = torch.zeros(batch_size, device=values.device, dtype=torch.float32)
    weight_sum = torch.zeros(batch_size, device=values.device, dtype=torch.float32)
    loss_sum.scatter_add_(0, example_indices, values * weights)
    weight_sum.scatter_add_(0, example_indices, weights)
    global_weight_sum = weight_sum.sum().clamp_min(1.0)
    return loss_sum * float(batch_size) / global_weight_sum


class MolmoAct2Policy(PreTrainedPolicy):
    """为 LeRobot 封装内置 HF 模型的 MolmoAct2 策略。

    通过 ``config.action_mode`` 支持三种训练模式：``"continuous"``
    （仅 flow-matching）、``"discrete"``（仅自回归 token 预测）或
    ``"both"``（联合损失）。推理时，``config.inference_action_mode``
    选择由哪个头来生成动作。
    """

    config_class = MolmoAct2Config
    name = "molmoact2"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | os.PathLike[str],
        *,
        strict: bool = True,
        **kwargs: Any,
    ) -> MolmoAct2Policy:
        """加载 LeRobot 检查点，且不静默接受拓扑结构变化。"""
        return super().from_pretrained(
            pretrained_name_or_path,
            strict=strict,
            **kwargs,
        )

    def supports_rtc(self) -> bool:
        return self.config.inference_action_mode == "continuous"

    def __init__(
        self,
        config: MolmoAct2Config,
        *inputs,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        dataset_meta: Any | None = None,
        **kwargs,
    ):
        super().__init__(config, *inputs, **kwargs)
        _apply_norm_tag_metadata(self.config)
        self.config.validate_features()
        del inputs, kwargs, dataset_stats, dataset_meta
        self._checkpoint_action_mode = _saved_policy_action_mode(self.config)
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)
        self._rollout_action_generator: torch.Generator | None = None
        self._rollout_task_key: tuple[Any, ...] | None = None
        self._rollout_index_for_task = -1
        self.rtc_processor: RTCProcessor | None = None
        self.action_tokenizer: Any | None = None
        self._compile_applied = False
        self._compiled_module_names: tuple[str, ...] = ()
        self._load_hf_model()
        _validate_inference_action_mode(self.config, self._checkpoint_action_mode)
        if self.config.train_mode_vlm == "lora":
            self._apply_lora_adapters()
        self._apply_compile()
        self.init_rtc_processor()

    def _load_hf_model(self) -> None:
        require_package("transformers", extra="molmoact2")

        checkpoint_location = _resolve_checkpoint_location(
            self.config.checkpoint_path,
            revision=self.config.checkpoint_revision,
            force_download=bool(self.config.checkpoint_force_download),
        )
        storage_dtype = _torch_dtype(self.config.dtype)
        if HFMolmoAct2Config is None or MolmoAct2ForConditionalGeneration is None:
            raise RuntimeError("transformers is required to load MolmoAct2 checkpoints.")
        hf_config = HFMolmoAct2Config.from_pretrained(
            checkpoint_location,
            token=_hf_token(),
        )
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None and hasattr(text_config, "residual_dropout"):
            text_config.residual_dropout = float(self.config.llm_residual_dropout)
        self.model = MolmoAct2ForConditionalGeneration.from_pretrained(
            checkpoint_location,
            config=hf_config,
            dtype=storage_dtype,
            low_cpu_mem_usage=True,
            token=_hf_token(),
        )
        # 将 Hub 加载限制为本地代码加 safetensors，并校验本地实现与
        # 检查点的键空间完全一致。
        self._apply_bfloat16_parameter_policy()
        # 在 bfloat16 模式下，在这最后一次严格加载之前确立最终的目标
        # dtype 树。这样可以为 action expert 和其他目标为 fp32 的模块保留
        # 检查点中原始的 fp32 数值，而不是扩展已经被舍入过的 bf16 张量。
        _strict_load_safetensors_weights(self.model, checkpoint_location)
        hf_max_action_dim = int(getattr(self.model.config, "max_action_dim", -1))
        if hf_max_action_dim != int(self.config.expected_max_action_dim):
            raise ValueError(
                "MolmoAct2 checkpoint max_action_dim mismatch: "
                f"checkpoint={hf_max_action_dim}, expected={self.config.expected_max_action_dim}."
            )
        if hf_max_action_dim != 32:
            raise ValueError(
                f"MolmoAct2 released checkpoints must have max_action_dim=32, got {hf_max_action_dim}."
            )

        if not hasattr(self.model.config, "max_action_horizon"):
            raise ValueError("MolmoAct2 HF checkpoints must define `max_action_horizon`.")
        self._override_loaded_max_action_horizon(int(self.config.chunk_size))

        if not hasattr(self.model.config, "action_mode"):
            raise ValueError(
                "MolmoAct2 HF checkpoints must define `action_mode`. If this is a released "
                "MolmoAct2 checkpoint, refresh the local Hub cache with "
                "`policy.checkpoint_force_download=true` after the updated files are pushed."
            )
        checkpoint_action_mode = str(self.model.config.action_mode)
        _validate_checkpoint_action_mode(
            self.config,
            checkpoint_action_mode,
            has_action_expert=bool(getattr(self.model.config, "add_action_expert", False)),
        )

        if self.config.freeze_embedding:
            self._freeze_input_embeddings()
        if self.config.train_mode_vlm == "freeze":
            self._freeze_vlm_parameters()
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        self.train(self.training)

    def reset(self) -> None:
        """在 episode 之间清空动作队列和 rollout 生成器。"""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._rollout_action_generator = None

    def _set_inference_cuda_graph_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "model"):
            return
        hf_model = self._hf_model()
        enabled = bool(enabled and getattr(self.config, "enable_inference_cuda_graph", True))
        managers = [
            getattr(self._backbone(), "action_cuda_graph_manager", None),
            getattr(hf_model, "action_cuda_graph_manager", None),
            getattr(hf_model, "depth_decode_cuda_graph_manager", None),
        ]
        seen: set[int] = set()
        for manager in managers:
            if manager is None or id(manager) in seen:
                continue
            seen.add(id(manager))
            set_enabled = getattr(manager, "set_enabled", None)
            if callable(set_enabled):
                set_enabled(enabled)

    def init_rtc_processor(self) -> None:
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _action_expert(self) -> torch.nn.Module:
        return self._backbone()._require_action_expert()

    def _enable_gradient_checkpointing(self) -> None:
        enable_gradient_checkpointing = getattr(self._hf_model(), "gradient_checkpointing_enable", None)
        if callable(enable_gradient_checkpointing):
            try:
                enable_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                enable_gradient_checkpointing()
        else:
            transformer = getattr(self._backbone(), "transformer", None)
            if transformer is None:
                raise RuntimeError("gradient_checkpointing=true, but MolmoAct2 exposes no text transformer.")
            transformer.gradient_checkpointing = True

        transformer = getattr(self._backbone(), "transformer", None)
        if transformer is not None:
            transformer.gradient_checkpointing = True
        vision_backbone = getattr(self._backbone(), "vision_backbone", None)
        if vision_backbone is not None:
            vision_backbone.gradient_checkpointing = True

    def _freeze_vlm_parameters(self) -> None:
        trainable_params = 0
        for name, param in self.named_parameters():
            param.requires_grad = "action_expert" in name
            if param.requires_grad:
                trainable_params += param.numel()
        if trainable_params == 0:
            raise RuntimeError("train_mode_vlm='freeze', but no action_expert parameters were found.")

    def _unfreeze_action_expert_parameters(self) -> None:
        trainable_params = 0
        for name, param in self.named_parameters():
            if "action_expert" in name:
                param.requires_grad_(True)
                trainable_params += param.numel()
        if trainable_params == 0:
            raise RuntimeError("train_mode_vlm='lora', but no action_expert parameters were found.")

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self.config, "train_mode_vlm", "fft") == "freeze" and hasattr(self, "model"):
            self._hf_model().eval()
            self._action_expert().train(mode)
        self._set_inference_cuda_graph_enabled(not mode)
        return self

    def _freeze_input_embeddings(self) -> None:
        embedding_modules: list[torch.nn.Module] = []
        seen_module_ids: set[int] = set()
        hf_model = self._hf_model()
        for module in (hf_model, self._backbone()):
            get_input_embeddings = getattr(module, "get_input_embeddings", None)
            if not callable(get_input_embeddings):
                continue
            embeddings = get_input_embeddings()
            if embeddings is None or id(embeddings) in seen_module_ids:
                continue
            embedding_modules.append(embeddings)
            seen_module_ids.add(id(embeddings))

        if not embedding_modules:
            raise RuntimeError("freeze_embedding=true, but MolmoAct2 checkpoint exposes no input embeddings.")

        lm_head = getattr(hf_model, "lm_head", None)
        lm_head_params = {id(param) for param in lm_head.parameters()} if lm_head is not None else set()
        embedding_params = [param for embeddings in embedding_modules for param in embeddings.parameters()]
        if any(id(param) in lm_head_params for param in embedding_params):
            raise RuntimeError(
                "freeze_embedding=true would also freeze lm_head because input embeddings and lm_head "
                "share parameters in this checkpoint."
            )
        for embeddings in embedding_modules:
            base_embedding = getattr(embeddings, "embedding", None)
            if isinstance(base_embedding, torch.nn.Parameter):
                base_embedding.requires_grad = False
            elif isinstance(base_embedding, torch.nn.Module):
                for param in base_embedding.parameters():
                    param.requires_grad = False
            else:
                for param in embeddings.parameters():
                    param.requires_grad = False

    def _apply_bfloat16_parameter_policy(self) -> None:
        """应用固定的低显存 MolmoAct2 参数存储策略。

        大型文本和视觉矩阵保持 bf16。完整的 action expert、norms、RoPE
        状态、depth gates 和 LoRA 适配器使用 fp32 参数，因此在可训练时
        也使用 fp32 的 Adam 状态。符合条件的 action-expert 算子在
        autocast 下仍以 bf16 执行。
        """
        if self.config.dtype != "bfloat16":
            return

        self.model.to(dtype=torch.bfloat16)
        backbone = self._backbone()

        action_expert = getattr(backbone, "action_expert", None)
        if action_expert is not None:
            action_expert.to(dtype=torch.float32)

        depth_gate = getattr(backbone, "action_expert_depth_gate", None)
        if depth_gate is not None:
            depth_gate.to(dtype=torch.float32)

        fp32_module_types = tuple(
            module_type
            for module_type in (
                torch.nn.LayerNorm,
                ActionExpertRMSNorm,
                MolmoAct2RMSNorm,
            )
            if isinstance(module_type, type)
        )
        for module in self.model.modules():
            if isinstance(module, fp32_module_types):
                module.to(dtype=torch.float32)
            if isinstance(MolmoAct2RotaryEmbedding, type) and isinstance(module, MolmoAct2RotaryEmbedding):
                module.to(dtype=torch.float32)
                # ``original_inv_freq`` 是一个别名而非注册的 buffer，
                # 因此 Module.to() 不会自动更新它。
                module.original_inv_freq = module.inv_freq
                module._pos_sin_cache = torch.empty(0, device=module.inv_freq.device, dtype=torch.float32)
                module._pos_cos_cache = torch.empty(0, device=module.inv_freq.device, dtype=torch.float32)

    def get_optim_params(self) -> list[dict[str, Any]]:
        """返回带有按组件学习率的优化器参数组。"""
        vit_params: list[Tensor] = []
        connector_params: list[Tensor] = []
        action_expert_params: list[Tensor] = []
        vlm_params: list[Tensor] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "action_expert" in name:
                action_expert_params.append(param)
            # 原生 MolmoAct2 在 get_connector_parameters() 下显式管理
            # 检查点中额外的词表。在已发布的 ft_embedding="lm_head" FFT
            # 方案中它保持可训练，因此尽管它在结构上嵌套在文本嵌入之下，
            # 仍将其路由到 connector 学习率和独立裁剪组。
            elif ".wte.new_embedding" in name or any(
                part in name for part in ("image_pooling_2d", "image_projector")
            ):
                connector_params.append(param)
            elif any(part in name for part in ("vision", "image_encoder", "vit")):
                vit_params.append(param)
            elif any(part in name for part in ("multi_modal_projector", "connector", "mm_projector")):
                connector_params.append(param)
            else:
                vlm_params.append(param)

        vlm_lora = self.config.train_mode_vlm == "lora"
        vlm_lr = 5e-5 if vlm_lora else self.config.optimizer_lr
        vit_lr = 5e-5 if vlm_lora else self.config.optimizer_vit_lr
        connector_lr = 5e-5 if vlm_lora else self.config.optimizer_connector_lr

        groups: list[dict[str, Any]] = []
        if vlm_params:
            groups.append({"params": vlm_params, "lr": vlm_lr})
        if vit_params:
            groups.append({"params": vit_params, "lr": vit_lr})
        if connector_params:
            groups.append({"params": connector_params, "lr": connector_lr})
        if action_expert_params:
            groups.append({"params": action_expert_params, "lr": self.config.optimizer_action_expert_lr})
        return groups

    def _model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # 与 Pi0.5 保持一致：连续输入保持 fp32，然后在明确的模型/autocast
        # 边界处转入 bf16。
        return {
            key: value.to(dtype=torch.float32) if value.is_floating_point() else value
            for key, value in batch.items()
            if key in _MODEL_INPUT_KEYS and value is not None
        }

    def _autocast_context(self):
        compute_dtype = _torch_dtype(self.config.dtype)
        device_type = next(self.parameters()).device.type
        autocast_available = torch.amp.autocast_mode.is_autocast_available(device_type)
        if compute_dtype == torch.bfloat16:
            if not autocast_available:
                raise RuntimeError(
                    f"MolmoAct2 dtype='bfloat16' requires autocast support on device type {device_type!r}."
                )
            return torch.autocast(device_type=device_type, dtype=compute_dtype)
        if autocast_available:
            # 当请求 float32 时，外层的 eval/use_amp 上下文绝不能覆盖
            # 这个唯一的 Pi0.5 风格 dtype 开关。
            return torch.autocast(device_type=device_type, enabled=False)
        return nullcontext()

    def _output_action_dim(self, batch: dict[str, Tensor]) -> int:
        action_feature = self.config.output_features.get(ACTION)
        if action_feature is not None and action_feature.shape:
            action_dim = int(action_feature.shape[0])
            if action_dim > 0:
                return action_dim

        action_dim_is_pad = batch.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            valid_counts = (~action_dim_is_pad.to(dtype=torch.bool)).sum(dim=-1)
            if bool((valid_counts == valid_counts[0]).all()) and int(valid_counts[0]) > 0:
                return int(valid_counts[0])

        raise RuntimeError("MolmoAct2 inference requires a positive action dimension in output_features.")

    def _hf_model(self):
        base_model = getattr(self.model, "base_model", None)
        wrapped_model = getattr(base_model, "model", None) if base_model is not None else None
        return wrapped_model if wrapped_model is not None else self.model

    def _backbone(self):
        return self._hf_model().model

    def _iter_compile_targets(self) -> Iterator[tuple[str, torch.nn.Module, bool | None]]:
        """产出经过数值验证的 MolmoAct2 训练编译边界。

        外层策略/模型的前向刻意保持 eager：多模态打包、动作区间处理和
        指标计算包含依赖数据的 Python 控制流。编译这些包装层只会增加
        图断点，并不能改善以 transformer 为主的训练路径。在 FFT 和 LoRA
        两种模式下，VLM、视觉塔和 connector 也保持 eager。真实批次的
        同起点一致性测试发现，编译这些 BF16 路径会导致明显的激活、梯度
        和 Adam 更新偏差。Action-expert 块能通过前向、激活和梯度检查，
        并保留有用的 F 步 flow-matching 加速。
        """
        backbone = self._backbone()

        # 非重入激活检查点必须在前向和反向中重放同一个编译图。可变的
        # 编码器上下文维度会在每次块调用之前被显式标记，使 Dynamo 在
        # 第一次追踪时就将其视为符号化的。其余固定的批/动作维度保持静态，
        # 而不是在这里请求一个完全动态的图。
        sequence_dynamic = None
        action_expert = getattr(backbone, "action_expert", None)
        for index, block in enumerate(getattr(action_expert, "blocks", ())):
            yield f"action_expert.blocks.{index}", block, sequence_dynamic

    def _apply_compile(self) -> None:
        """在显式启用时安装官方风格的按块编译。"""
        if not self.config.compile_model or self._compile_applied:
            return
        if not hasattr(torch.nn.Module, "compile"):
            raise RuntimeError("MolmoAct2 compile_model=True requires torch.nn.Module.compile support.")

        if self.config.gradient_checkpointing:
            # 保持 Dynamo 缓存查找顺序在原始前向和激活检查点重计算之间
            # 稳定。这不会禁用图缓存或编译；它只禁用 LRU 命中重排序——
            # 这是 PyTorch 针对重复编译块在重放时保存不同张量签名问题
            # 推荐的解决办法。
            _set_dynamo_lru_cache(False)
            # DDP 编译器侧的图分割器只在原始 DDP 前向中生效，在检查点
            # 重计算中不生效。模型现有的块边界已经提供了天然的反向/
            # all-reduce 重叠，因此为保证图的一致性，保持禁用这个额外的
            # 分割器。
            _disable_dynamo_ddp_optimizer()

        torch.set_float32_matmul_precision("high")
        compile_options = {"emulate_precision_casts": True}
        compile_kwargs = {
            "backend": "inductor",
            "options": compile_options,
            "fullgraph": False,
        }
        compiled_names = []
        for name, module, dynamic in self._iter_compile_targets():
            module.compile(**compile_kwargs, dynamic=dynamic)
            compiled_names.append(name)

        if not compiled_names:
            raise RuntimeError("MolmoAct2 compile_model=True found no compatible model blocks.")
        self._compiled_module_names = tuple(compiled_names)
        self._compile_applied = True
        logger.info(
            "Enabled MolmoAct2 block-wise torch.compile for %d modules.",
            len(compiled_names),
        )

    def _override_loaded_max_action_horizon(self, action_horizon: int) -> None:
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {action_horizon}.")
        hf_model = self._hf_model()
        for cfg in (getattr(hf_model, "config", None), getattr(self._backbone(), "config", None)):
            if cfg is not None:
                cfg.max_action_horizon = int(action_horizon)

    def _generation_action_horizon(self) -> int:
        chunk_size = getattr(self.config, "chunk_size", None)
        if chunk_size is not None:
            return int(chunk_size)
        hf_model = self._hf_model()
        for cfg in (getattr(hf_model, "config", None), getattr(self._backbone(), "config", None)):
            if cfg is None:
                continue
            value = getattr(cfg, "max_action_horizon", None)
            if value is not None:
                return int(value)
        raise RuntimeError("MolmoAct2 could not resolve an action generation horizon.")

    def _encoder_attention_mask_for_action_expert(
        self,
        *,
        input_ids: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        # 已发布的 HF 基础模型是 BOTH 检查点，但当前官方训练器会显式地
        # 将其原生模型切换为连续动作格式。因此如果先调用 HF backbone 的
        # 辅助方法，即使本策略只训练连续模式，也会继承检查点中过时的
        # BOTH 模式 EOS/动作区间掩码。连续模式下直接构造普通的提示掩码；
        # 只有联合目标才应应用离散输出掩码。
        if getattr(self.config, "action_mode", None) != "both":
            if attention_mask is not None:
                return attention_mask.to(dtype=torch.bool).clone()
            if input_ids is not None:
                return input_ids != -1
            return None

        backbone = self._backbone()
        get_encoder_attention_mask = getattr(backbone, "_get_encoder_attention_mask", None)
        if callable(get_encoder_attention_mask):
            mask = get_encoder_attention_mask(input_ids, attention_mask)
        elif attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool)
        elif input_ids is not None:
            mask = input_ids != -1
        else:
            return None

        if input_ids is None or mask is None:
            return mask

        mask = mask.to(dtype=torch.bool).clone()
        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if eos_token_id is not None:
            mask &= input_ids != int(eos_token_id)
        return _mask_discrete_action_spans(
            input_ids=input_ids,
            mask=mask,
            start_token_id=getattr(self.model.config, "action_start_token_id", None),
            end_token_id=getattr(self.model.config, "action_end_token_id", None),
        )

    def _load_discrete_action_tokenizer(self) -> Any:
        if self.action_tokenizer is None:
            require_package("transformers", extra="molmoact2")
            require_package("scipy", extra="molmoact2")

            if UniversalActionProcessor is None:
                raise RuntimeError("transformers and scipy are required to load MolmoAct2 action tokenizer.")
            self.action_tokenizer = UniversalActionProcessor.from_pretrained_local(
                self.config.discrete_action_tokenizer,
            )
        return self.action_tokenizer

    def _resolve_inference_action_mode(self, requested_mode: str | None) -> str:
        return _resolve_inference_action_mode(self.config, requested_mode, self._checkpoint_action_mode)

    def _rollout_generator_for_inputs(
        self,
        batch: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Generator | None:
        if not bool(getattr(self.config, "per_episode_seed", False)):
            return None
        if self._rollout_action_generator is not None:
            return self._rollout_action_generator

        task_signature = _rollout_task_signature(batch)
        if task_signature != self._rollout_task_key:
            self._rollout_task_key = task_signature
            self._rollout_index_for_task = 0
        else:
            self._rollout_index_for_task += 1

        base_seed = int(getattr(self.config, "eval_seed", None) or 0)
        first_seed = base_seed + self._rollout_index_for_task * batch_size
        generator_device = (
            device if device.type == "cuda" and torch.cuda.is_available() else torch.device("cpu")
        )
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(_combine_rollout_seeds(first_seed, batch_size))
        self._rollout_action_generator = generator
        return generator

    def _prepare_flow_matching_tensors(
        self,
        *,
        actions: Tensor,
        action_dim_is_pad: Tensor | None,
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # 与原始 MolmoAct2 的 fp32+AMP 训练路径保持一致：归一化后的动作、
        # 采样的 flow 噪声、时间步和速度目标都保持 fp32。尽管其参数刻意
        # 以 fp32 存储，Autocast 仍会处理符合条件的 action-expert 矩阵
        # 乘法。
        actions = actions.to(dtype=torch.float32)
        batch_size = int(actions.shape[0])
        device = actions.device
        num_flow_timesteps = max(1, int(self.config.num_flow_timesteps))

        if timesteps is None:
            timesteps = (
                _sample_beta_timesteps(
                    batch_size=batch_size * num_flow_timesteps,
                    device=device,
                    cutoff=self.config.flow_matching_cutoff,
                    time_offset=self.config.flow_matching_time_offset,
                    time_scale=self.config.flow_matching_time_scale,
                    alpha=self.config.flow_matching_beta_alpha,
                    beta=self.config.flow_matching_beta_beta,
                )
                .to(dtype=actions.dtype)
                .view(batch_size, num_flow_timesteps)
            )
        else:
            expected_timesteps_shape = (batch_size, num_flow_timesteps)
            timesteps = timesteps.to(device=device, dtype=actions.dtype)
            if tuple(timesteps.shape) != expected_timesteps_shape:
                raise ValueError(
                    f"flow timesteps must have shape {expected_timesteps_shape}, got {tuple(timesteps.shape)}."
                )

        if self.config.mask_action_dim_padding:
            actions = _mask_action_dim_tensor(actions, action_dim_is_pad)

        expected_noise_shape = (batch_size, num_flow_timesteps, actions.shape[1], actions.shape[2])
        if noise is None:
            noise = torch.randn(*expected_noise_shape, device=device, dtype=actions.dtype)
        else:
            noise = noise.to(device=device, dtype=actions.dtype)
            if tuple(noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"flow noise must have shape {expected_noise_shape}, got {tuple(noise.shape)}."
                )
        if self.config.mask_action_dim_padding:
            noise = _mask_action_dim_tensor(noise, action_dim_is_pad)

        t_broadcast = timesteps.view(batch_size, num_flow_timesteps, 1, 1)
        actions_expanded = actions.unsqueeze(1).expand(-1, num_flow_timesteps, -1, -1)
        xt = (1.0 - t_broadcast) * noise + t_broadcast * actions_expanded
        target_velocity = actions_expanded - noise
        return actions, timesteps, xt, target_velocity

    def _prepare_joint_training_backbone_inputs(
        self,
        model_inputs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor | dict[str, Any], Tensor, Tensor]:
        backbone = self._backbone()
        input_ids = model_inputs.get("input_ids")
        inputs_embeds = model_inputs.get("inputs_embeds")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "MolmoAct2 joint flow training requires exactly one of input_ids or inputs_embeds."
            )

        images = None
        token_pooling = None
        merge_visual_inputs = getattr(backbone, "merge_visual_inputs", None)
        if callable(merge_visual_inputs):
            images, token_pooling = merge_visual_inputs(
                input_ids=input_ids,
                pixel_values=model_inputs.get("pixel_values"),
                image_token_pooling=model_inputs.get("image_token_pooling"),
                image_grids=model_inputs.get("image_grids"),
                image_num_crops=model_inputs.get("image_num_crops"),
                pixel_values_videos=model_inputs.get("pixel_values_videos"),
                video_token_pooling=model_inputs.get("video_token_pooling"),
                video_grids=model_inputs.get("video_grids"),
            )
        elif (
            model_inputs.get("pixel_values") is not None
            or model_inputs.get("pixel_values_videos") is not None
        ):
            raise RuntimeError("MolmoAct2 checkpoint does not expose merge_visual_inputs for joint training.")

        if images is not None and inputs_embeds is not None:
            raise ValueError("MolmoAct2 joint flow training cannot combine inputs_embeds with visual inputs.")
        if inputs_embeds is None:
            inputs_embeds, _image_features = backbone.build_input_embeddings(input_ids, images, token_pooling)
        # 外部嵌入绕过了内置文本模型的嵌入边界，因此在这里将它们
        # 归一化为当前 AMP 激活 dtype。
        device_type = inputs_embeds.device.type
        if torch.is_autocast_enabled(device_type):
            inputs_embeds = inputs_embeds.to(dtype=torch.get_autocast_dtype(device_type))

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        attention_mask = model_inputs.get("attention_mask")
        position_ids = model_inputs.get("position_ids")
        if position_ids is None:
            if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
                position_ids = _position_ids_from_attention_mask(attention_mask)
            else:
                position_ids = cache_position.unsqueeze(0)

        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            # 官方原生微调配置显式设置了 ``bi_directional_attn=None``。
            # 转换后的 HF processor 为了推理兼容性仍会输出图像的
            # ``token_type_ids``，但在这里转发它们会悄悄地把所有图像
            # token 变成双向，并改变 action expert 看到的所有 VLM KV
            # 状态。保持联合 flow 训练为因果模式，以匹配模型的预训练/
            # 微调方案；推理保持不变。
            causal_mask_mapping = backbone._build_native_attention_bias(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=None,
                past_key_values=None,
            )
        return inputs_embeds, causal_mask_mapping, position_ids, cache_position

    @staticmethod
    def _decoder_layer_kv_outputs(
        layer_outputs: tuple[Any, ...], *, output_attentions: bool
    ) -> tuple[Tensor, Tensor]:
        output_idx = 2 if output_attentions else 1
        return layer_outputs[output_idx], layer_outputs[output_idx + 1]

    @staticmethod
    def _action_time_conditioning(action_expert: torch.nn.Module, timesteps: Tensor) -> Tensor:
        time_conditioning = getattr(action_expert, "_time_conditioning", None)
        if callable(time_conditioning):
            return time_conditioning(timesteps)
        return action_expert.time_embed(timesteps)

    def _compute_flow_matching_loss_joint_per_layer(
        self,
        *,
        batch: dict[str, Tensor],
        model_inputs: dict[str, Tensor],
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, Tensor]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}. Expected 'mean' or 'none'.")
        backbone = self._backbone()
        transformer = getattr(backbone, "transformer", None)
        action_expert = backbone._require_action_expert()
        if transformer is None:
            raise RuntimeError("MolmoAct2 joint flow training requires a patchable text transformer.")
        if len(action_expert.blocks) != int(transformer.config.num_hidden_layers):
            raise RuntimeError(
                "MolmoAct2 joint flow training requires one action expert block per text transformer layer."
            )

        actions, timesteps, xt, target_velocity = self._prepare_flow_matching_tensors(
            actions=batch[ACTION],
            action_dim_is_pad=batch.get("action_dim_is_pad"),
            timesteps=timesteps,
            noise=noise,
        )
        num_flow_timesteps = max(1, int(self.config.num_flow_timesteps))
        batch_size = int(actions.shape[0])
        device = actions.device
        xt_flat = xt.reshape(batch_size * num_flow_timesteps, actions.shape[1], actions.shape[2])
        timesteps_flat = timesteps.reshape(batch_size * num_flow_timesteps)

        hidden_states, causal_mask_mapping, position_ids, cache_position = (
            self._prepare_joint_training_backbone_inputs(model_inputs)
        )
        if hidden_states.shape[0] != batch_size:
            raise ValueError(
                f"Backbone batch size {hidden_states.shape[0]} does not match action batch size {batch_size}."
            )

        encoder_attention_mask = self._encoder_attention_mask_for_action_expert(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        action_attention_mask = None
        if batch.get("action_horizon_is_pad") is not None:
            action_attention_mask = ~batch["action_horizon_is_pad"].to(device=device, dtype=torch.bool)

        # 与原始 AMP 实现保持一致：用动作激活的 dtype（而非 fp32 主权重
        # 的 dtype）来构造掩码和 RoPE 缓存。在 bf16 autocast 下，下面的
        # 线性层会产生 bf16 激活，即使 action_embed.weight 保持 fp32。
        action_parameter_dtype = action_expert.action_embed.weight.dtype
        conditioning = self._action_time_conditioning(action_expert, timesteps_flat)
        action_hidden = action_expert.action_embed(xt_flat.to(dtype=action_parameter_dtype))
        action_compute_dtype = action_hidden.dtype

        valid_action = None
        if action_attention_mask is not None:
            valid_action = action_attention_mask.to(device=device, dtype=action_compute_dtype).unsqueeze(-1)
            valid_action = _expand_mask(valid_action, num_flow_timesteps)

        rope_cache = None
        if len(action_expert.blocks) > 0 and action_expert.blocks[0].self_attn.rope is not None:
            rope_cache = action_expert.blocks[0].self_attn.rope.build_cache(
                seq_len=actions.shape[1],
                device=device,
                dtype=action_compute_dtype,
            )

        cross_mask = action_expert._build_cross_attention_mask(
            encoder_attention_mask,
            batch_size,
            action_compute_dtype,
        )
        cross_mask = _expand_mask(cross_mask, num_flow_timesteps)
        self_mask = action_expert._build_self_attention_mask(
            action_attention_mask,
            actions.shape[1],
            device,
            action_compute_dtype,
        )
        self_mask = _expand_mask(self_mask, num_flow_timesteps)

        if valid_action is not None:
            action_hidden = action_hidden * valid_action

        if transformer.config.rope_scaling_layers is not None:
            position_embeddings_mapping = {
                "default": transformer.rotary_embs["default"](hidden_states, position_ids),
                "scaling": transformer.rotary_embs["scaling"](hidden_states, position_ids),
            }
        else:
            position_embeddings = transformer.rotary_emb(hidden_states, position_ids)

        use_gradient_checkpointing = bool(
            getattr(self.config, "gradient_checkpointing", False)
            and self.training
            and torch.is_grad_enabled()
        )

        def run_layer(
            layer_idx: int, layer_hidden: Tensor, layer_action_hidden: Tensor
        ) -> tuple[Tensor, Tensor]:
            decoder_block = transformer.blocks[layer_idx]
            action_block = action_expert.blocks[layer_idx]
            if transformer.config.rope_scaling_layers is not None:
                position_embeddings_i = (
                    position_embeddings_mapping["scaling"]
                    if layer_idx in transformer.config.rope_scaling_layers
                    else position_embeddings_mapping["default"]
                )
            else:
                position_embeddings_i = position_embeddings

            decoder_kwargs = {
                "position_embeddings": position_embeddings_i,
                "attention_mask": causal_mask_mapping,
                "position_ids": position_ids,
                "past_key_values": None,
                "output_attentions": False,
                "use_cache": False,
                "cache_position": cache_position,
                "collect_layer_kv_states": True,
            }
            if use_gradient_checkpointing:
                # 外层检查点覆盖的是配对的文本层+动作层。只绕过
                # Transformers 的逐文本层检查点包装，避免文本块被
                # 重复计算两次。基类 Module 分发仍会遵循
                # Module.compile 和 hooks。
                layer_outputs = _call_module_without_gradient_checkpointing_layer(
                    decoder_block,
                    layer_hidden,
                    **decoder_kwargs,
                )
            else:
                layer_outputs = decoder_block(layer_hidden, **decoder_kwargs)
            next_hidden = layer_outputs[0]
            key_states, value_states = self._decoder_layer_kv_outputs(layer_outputs, output_attentions=False)
            key_states = backbone._cache_to_sequence(key_states)
            value_states = backbone._cache_to_sequence(value_states)
            if self.config.enable_knowledge_insulation:
                key_states = key_states.detach()
                value_states = value_states.detach()

            k_ctx = action_expert._project_kv_tensor(key_states, action_expert.context_k_proj)
            v_ctx = action_expert._project_kv_tensor(value_states, action_expert.context_v_proj)
            k_norm = action_block.cross_attn.k_norm
            if k_norm is not None:
                k_ctx = k_norm(k_ctx.transpose(1, 2)).transpose(1, 2)
            if num_flow_timesteps != 1:
                k_ctx = _expand_mask(k_ctx, num_flow_timesteps)
                v_ctx = _expand_mask(v_ctx, num_flow_timesteps)

            if self.config.compile_model and use_gradient_checkpointing:
                _mark_action_context_dynamic(k_ctx, v_ctx, cross_mask)

            next_action_hidden = action_block(
                layer_action_hidden,
                conditioning,
                cross_kv=(k_ctx, v_ctx),
                self_attn_mask=self_mask,
                attn_mask=cross_mask,
                is_causal=action_expert.config.causal_attn,
                modulation=None,
                rope_cache=rope_cache,
            )
            if valid_action is not None:
                next_action_hidden = next_action_hidden * valid_action
            return next_hidden, next_action_hidden

        for layer_idx in range(int(transformer.config.num_hidden_layers)):
            if use_gradient_checkpointing:
                hidden_states, action_hidden = torch.utils.checkpoint.checkpoint(
                    lambda layer_hidden, layer_action_hidden, idx=layer_idx: run_layer(
                        idx,
                        layer_hidden,
                        layer_action_hidden,
                    ),
                    hidden_states,
                    action_hidden,
                    use_reentrant=False,
                )
            else:
                hidden_states, action_hidden = run_layer(layer_idx, hidden_states, action_hidden)

        hidden_states = transformer.ln_f(hidden_states)
        pred_velocity = action_expert.final_layer(action_hidden, conditioning)
        if valid_action is not None:
            pred_velocity = pred_velocity * valid_action
        pred_velocity = pred_velocity.reshape(
            batch_size, num_flow_timesteps, actions.shape[1], actions.shape[2]
        ).to(dtype=target_velocity.dtype)

        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        loss = _apply_action_chunk_padding_mask(loss, batch.get("action_horizon_is_pad"))
        if self.config.mask_action_dim_padding:
            loss = _apply_action_dim_padding_mask(loss, batch.get("action_dim_is_pad"))
        loss = loss.reshape(batch_size, -1).mean(dim=1)
        if reduction == "mean":
            loss = loss.mean()
        return loss, hidden_states

    def _discrete_token_weights(self, valid_positions: Tensor) -> Tensor | None:
        mode = self.config.discrete_loss_token_weighting
        if mode in {"none", "token", "root_subsegments"}:
            return None
        if mode != "root_subsegments_root_tokens" and mode != "root_tokens":
            raise ValueError(f"Unsupported discrete_loss_token_weighting={mode!r}.")

        token_counts = valid_positions.sum(dim=1).to(dtype=torch.float32)
        example_weights = torch.zeros_like(token_counts)
        nonempty = token_counts > 0
        example_weights[nonempty] = 2.0 / torch.sqrt(token_counts[nonempty])
        return example_weights[:, None].expand_as(valid_positions)[valid_positions].to(dtype=torch.float32)

    def _discrete_loss_from_backbone_outputs(
        self,
        batch: dict[str, Tensor],
        outputs: Any,
        reduction: str = "mean",
    ) -> tuple[Tensor, Tensor | None]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}. Expected 'mean' or 'none'.")
        labels = batch.get("labels")
        if labels is None:
            raise RuntimeError("MolmoAct2 discrete training requires labels.")
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise RuntimeError("MolmoAct2 backbone did not return last_hidden_state.")

        ignore_index = -100
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
        valid_positions = shift_labels != ignore_index
        if not bool(valid_positions.any()):
            raise RuntimeError("MolmoAct2 discrete training labels contain no valid action tokens.")

        hidden_size = hidden_states.shape[-1]
        selected_hidden = hidden_states.reshape(-1, hidden_size)[valid_positions.reshape(-1)]
        selected_labels = shift_labels.reshape(-1)[valid_positions.reshape(-1)].to(
            device=hidden_states.device
        )
        logits = F.linear(selected_hidden, self.model.lm_head.weight).float()
        log_z = logits.logsumexp(dim=-1)
        target_logits = logits.gather(dim=-1, index=selected_labels[:, None]).squeeze(-1)
        token_ce_loss = log_z - target_logits
        token_weights = self._discrete_token_weights(valid_positions)
        if reduction == "none":
            example_indices = valid_positions.nonzero(as_tuple=False)[:, 0].to(device=hidden_states.device)
            ce_loss = _weighted_per_example(
                token_ce_loss,
                token_weights,
                example_indices,
                int(labels.shape[0]),
            )
        else:
            ce_loss = _weighted_mean(token_ce_loss, token_weights)
        if not self.config.softmax_auxiliary_loss:
            return ce_loss, None

        if reduction == "none":
            z_loss = self.config.softmax_auxiliary_loss_scale * _weighted_per_example(
                log_z.pow(2),
                token_weights,
                example_indices,
                int(labels.shape[0]),
            )
        else:
            z_loss = self.config.softmax_auxiliary_loss_scale * _weighted_mean(log_z.pow(2), token_weights)
        return ce_loss, z_loss

    def _action_token_id_to_bin(self) -> dict[int, int]:
        method = getattr(self.model, "_action_token_id_to_bin", None)
        if callable(method):
            return dict(method())
        start = getattr(self.model.config, "action_token_start_id", None)
        num_tokens = int(getattr(self.model.config, "num_action_tokens", 0) or 0)
        if start is None or num_tokens <= 0:
            return {}
        return {int(start) + idx: idx for idx in range(num_tokens)}

    def _require_discrete_eos_token_id(self) -> int:
        method = getattr(self.model, "_require_eos_token_id", None)
        if callable(method):
            return int(method())
        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if eos_token_id is None and getattr(self.model, "generation_config", None) is not None:
            eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if eos_token_id is None:
            raise RuntimeError("Discrete action generation requires eos_token_id in the checkpoint config.")
        return int(eos_token_id)

    def _discrete_generation_max_steps(self) -> int:
        if self.config.discrete_generation_max_steps is not None:
            return int(self.config.discrete_generation_max_steps)
        return max(1, self._generation_action_horizon() * 16)

    def _continue_discrete_generation_from_output(
        self,
        initial_output: Any,
        *,
        past_key_values: Any | None,
        attention_mask: Tensor | None,
        end_token_id: int,
        max_steps: int,
        attention_bias: Tensor | None = None,
    ) -> Tensor:
        consume_generation_tokens = getattr(self.model, "_consume_generation_tokens", None)
        ar_decode_step = getattr(self.model, "_run_ar_decode_step", None)
        if ar_decode_step is None:
            ar_decode_step = getattr(self.model, "_run_depth_decode_step", None)
        if attention_bias is None and not callable(consume_generation_tokens):
            raise RuntimeError("MolmoAct2 checkpoint does not expose discrete token generation helpers.")
        if attention_bias is not None and not callable(ar_decode_step):
            raise RuntimeError("MolmoAct2 checkpoint does not expose graph-backed AR decode helpers.")

        generated_tokens: list[Tensor] = []
        current_output = initial_output
        current_past_key_values = past_key_values
        current_attention_mask = attention_mask
        next_position_ids = _next_decode_position_ids_from_attention_mask(current_attention_mask)
        hit_end = False
        for _ in range(int(max_steps)):
            next_token = torch.argmax(current_output.logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token)
            if bool((next_token == int(end_token_id)).all()):
                hit_end = True
                break
            if attention_bias is None:
                current_output, current_attention_mask = consume_generation_tokens(
                    next_token,
                    past_key_values=current_past_key_values,
                    attention_mask=current_attention_mask,
                )
                current_past_key_values = current_output.past_key_values
            else:
                step_position_ids = next_position_ids
                last_hidden, current_past_key_values = ar_decode_step(
                    next_token,
                    past_key_values=current_past_key_values,
                    attention_bias=attention_bias,
                    position_ids=step_position_ids,
                )
                if next_position_ids is not None:
                    next_position_ids = next_position_ids + 1
                current_output = types.SimpleNamespace(
                    logits=self.model.lm_head(last_hidden),
                    past_key_values=current_past_key_values,
                )
        if not generated_tokens:
            raise RuntimeError("Discrete continuation generated no tokens.")
        if not hit_end:
            raise RuntimeError(
                f"Discrete continuation did not emit end token {int(end_token_id)} within {int(max_steps)} steps."
            )
        return torch.stack(generated_tokens, dim=1)

    def _make_discrete_ar_graph_decode_inputs(
        self,
        model_inputs: dict[str, Tensor],
        *,
        max_steps: int,
    ) -> tuple[Any | None, Tensor | None]:
        if not bool(getattr(self.config, "enable_inference_cuda_graph", False)):
            return None, None
        if self.training or self.model.training:
            return None, None
        ar_decode_step = getattr(self.model, "_run_ar_decode_step", None)
        if ar_decode_step is None:
            ar_decode_step = getattr(self.model, "_run_depth_decode_step", None)
        make_attention_bias = getattr(self.model, "_make_depth_decode_attention_bias", None)
        if not callable(ar_decode_step) or not callable(make_attention_bias):
            return None, None

        make_static_cache = getattr(self.model, "_make_ar_decode_static_cache", None)
        if callable(make_static_cache):
            static_cache = make_static_cache(model_inputs, max_steps=max_steps)
        else:
            graph_manager = getattr(self.model, "depth_decode_cuda_graph_manager", None)
            make_manager_static_cache = getattr(graph_manager, "make_static_cache", None)
            if not callable(make_manager_static_cache):
                return None, None
            prompt_len = int(model_inputs["input_ids"].shape[1])
            static_cache = make_manager_static_cache(max_cache_len=prompt_len + max(1, int(max_steps)))

        attention_bias = make_attention_bias(model_inputs, static_cache)
        return static_cache, attention_bias

    def _decode_discrete_action_chunk(self, generated_token_ids: Tensor, *, action_dim: int) -> Tensor:
        if (
            getattr(self.model.config, "action_start_token_id", None) is None
            or getattr(self.model.config, "action_end_token_id", None) is None
        ):
            raise RuntimeError("Discrete action generation requires <action_start>/<action_end> token IDs.")
        token_id_to_bin = self._action_token_id_to_bin()
        if not token_id_to_bin:
            raise RuntimeError(
                "Discrete action generation requires indexed action tokens in the checkpoint config."
            )

        action_tokenizer = self._load_discrete_action_tokenizer()
        if generated_token_ids.ndim == 1:
            generated_token_ids = generated_token_ids.unsqueeze(0)
        if generated_token_ids.ndim == 3:
            generated_token_ids = generated_token_ids[:, 0, :]
        if generated_token_ids.ndim != 2:
            raise ValueError(f"Unexpected generated token tensor shape {tuple(generated_token_ids.shape)}.")

        chunks: list[Tensor] = []
        for token_row in generated_token_ids:
            generated_ids = [int(token_id) for token_id in token_row.detach().cpu().tolist()]
            discrete_token_ids = _extract_discrete_token_bins(
                generated_ids,
                int(self.model.config.action_start_token_id),
                int(self.model.config.action_end_token_id),
                token_id_to_bin,
            )
            if not discrete_token_ids:
                raise RuntimeError(
                    "Model generated no decodable action tokens between <action_start>/<action_end>."
                )
            try:
                decoded = action_tokenizer.decode(
                    [discrete_token_ids],
                    time_horizon=self._generation_action_horizon(),
                    action_dim=int(action_dim),
                )
            except TypeError:
                decoded = action_tokenizer.decode([discrete_token_ids])
            action_chunk = np.asarray(decoded, dtype=np.float32)
            if action_chunk.ndim == 1:
                action_chunk = action_chunk[None, :]
            elif action_chunk.ndim == 3:
                if int(action_chunk.shape[0]) != 1:
                    action_chunk = action_chunk.reshape(action_chunk.shape[-2], action_chunk.shape[-1])
                else:
                    action_chunk = action_chunk[0]
            elif action_chunk.ndim > 3:
                action_chunk = action_chunk.reshape(action_chunk.shape[-2], action_chunk.shape[-1])
            if action_chunk.ndim != 2:
                raise RuntimeError(f"Decoded action chunk has unexpected shape {action_chunk.shape}.")
            chunks.append(torch.as_tensor(action_chunk, device=token_row.device, dtype=torch.float32))
        return torch.stack(chunks, dim=0)

    def _generate_discrete_actions_from_inputs(
        self,
        *,
        model_inputs: dict[str, Tensor],
        action_dim: int,
    ) -> Tensor:
        model_inputs = _drop_trivial_attention_mask(model_inputs)
        max_steps = self._discrete_generation_max_steps()
        static_cache, attention_bias = self._make_discrete_ar_graph_decode_inputs(
            model_inputs,
            max_steps=max_steps,
        )
        prefill_kwargs: dict[str, Any] = {}
        if static_cache is not None:
            prefill_kwargs["past_key_values"] = static_cache
        prefill_output = self.model(
            **model_inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            **prefill_kwargs,
        )
        generated_token_ids = self._continue_discrete_generation_from_output(
            prefill_output,
            past_key_values=prefill_output.past_key_values,
            attention_mask=model_inputs.get("attention_mask"),
            end_token_id=self._require_discrete_eos_token_id(),
            max_steps=max_steps,
            attention_bias=attention_bias,
        )
        return self._decode_discrete_action_chunk(generated_token_ids, action_dim=action_dim)

    def _generate_actions_from_inputs_with_rtc(
        self,
        *,
        model_inputs: dict[str, Tensor],
        action_dim_is_pad: Tensor | None,
        num_steps: int | None,
        generator: torch.Generator | None,
        inference_delay: int | None,
        prev_chunk_left_over: Tensor | None,
        execution_horizon: int | None,
    ) -> Tensor:
        backbone = self._backbone()
        action_expert = self._action_expert()
        outputs = backbone(
            **model_inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        encoder_kv_states = backbone._extract_kv_states(outputs.past_key_values)
        encoder_attention_mask = self._encoder_attention_mask_for_action_expert(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        depth_gate, depth_mask = backbone._depth_gate_from_condition(
            input_ids=model_inputs.get("input_ids"),
            encoder_attention_mask=encoder_attention_mask,
            layer_kv_states=encoder_kv_states,
        )
        encoder_kv_states = backbone._apply_depth_gate_to_layer_kv_states(
            encoder_kv_states,
            depth_mask,
            depth_gate,
        )

        steps = int(num_steps or backbone.config.flow_matching_num_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be >= 1, got {steps}.")
        source_tensor = encoder_kv_states[0][0]
        batch_size = int(source_tensor.shape[0])
        device = source_tensor.device
        trajectory = torch.randn(
            batch_size,
            self._generation_action_horizon(),
            int(backbone.config.max_action_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        if self.config.mask_action_dim_padding:
            trajectory = _mask_action_dim_tensor(trajectory, action_dim_is_pad)

        action_context = action_expert.prepare_context(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            state_embeddings=None,
            batch_size=batch_size,
            seq_len=trajectory.shape[1],
            device=device,
            dtype=trajectory.dtype,
        )
        flow_timesteps = [
            torch.full((batch_size,), idx / steps, device=device, dtype=trajectory.dtype)
            for idx in range(steps)
        ]
        modulation_cache = action_expert.get_or_prepare_modulation_cache(
            flow_timesteps,
            cache_key=(steps, batch_size, device, trajectory.dtype),
        )

        dt = 1.0 / steps
        mask_enabled = self.config.mask_action_dim_padding
        for idx, flow_timestep in enumerate(flow_timesteps):
            modulation = modulation_cache[idx]

            def denoise_step(input_trajectory: Tensor, step_modulation=modulation) -> Tensor:
                velocity = action_expert.forward_with_context(
                    input_trajectory,
                    step_modulation.conditioning,
                    context=action_context,
                    modulation=step_modulation,
                )
                if mask_enabled:
                    velocity = _mask_action_dim_tensor(velocity, action_dim_is_pad)
                return velocity

            if self._rtc_enabled():
                if self.rtc_processor is None:
                    raise RuntimeError("RTC is enabled but rtc_processor is not initialized.")

                def rtc_denoise_step(input_trajectory: Tensor) -> Tensor:
                    return -denoise_step(input_trajectory)

                rtc_time = 1.0 - float(flow_timestep[0].item())
                rtc_velocity = self.rtc_processor.denoise_step(
                    x_t=trajectory,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=int(inference_delay or 0),
                    time=rtc_time,
                    original_denoise_step_partial=rtc_denoise_step,
                    execution_horizon=execution_horizon,
                )
                velocity = -rtc_velocity
            else:
                velocity = denoise_step(trajectory)

            trajectory = trajectory + dt * velocity
            if mask_enabled:
                trajectory = _mask_action_dim_tensor(trajectory, action_dim_is_pad)
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=float(flow_timestep[0].item()), x_t=trajectory, v_t=velocity)

        return trajectory

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Any]]:
        """在 MolmoAct2 配置的 AMP 策略下计算训练损失。"""
        with self._autocast_context():
            return self._forward_impl(batch, reduction=reduction)

    def _forward_impl(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Any]]:
        """计算训练损失（flow-matching 损失和/或离散 token 损失）。"""
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}. Expected 'mean' or 'none'.")
        model_inputs = self._model_inputs(batch)
        losses: list[Tensor] = []
        metrics: dict[str, Any] = {}

        if self.config.action_mode == "discrete":
            outputs = self._backbone()(
                **model_inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            discrete_ce_loss, discrete_z_loss = self._discrete_loss_from_backbone_outputs(
                batch, outputs, reduction=reduction
            )
            discrete_loss = (
                discrete_ce_loss if discrete_z_loss is None else discrete_ce_loss + discrete_z_loss
            )
            losses.append(discrete_loss)
            metrics["discrete_ce_loss"] = discrete_ce_loss.detach().float().mean().item()
            if discrete_z_loss is not None:
                metrics["discrete_z_loss"] = discrete_z_loss.detach().float().mean().item()

        elif self.config.action_mode == "continuous":
            flow_loss, _ = self._compute_flow_matching_loss_joint_per_layer(
                batch=batch,
                model_inputs=model_inputs,
                reduction=reduction,
            )
            losses.append(flow_loss)
            metrics["action_flow_loss"] = flow_loss.detach().float().mean().item()

        else:
            flow_loss, hidden_states = self._compute_flow_matching_loss_joint_per_layer(
                batch=batch,
                model_inputs=model_inputs,
                reduction=reduction,
            )
            outputs = types.SimpleNamespace(last_hidden_state=hidden_states)
            discrete_ce_loss, discrete_z_loss = self._discrete_loss_from_backbone_outputs(
                batch, outputs, reduction=reduction
            )
            discrete_loss = (
                discrete_ce_loss if discrete_z_loss is None else discrete_ce_loss + discrete_z_loss
            )
            losses.append(discrete_loss)
            metrics["discrete_ce_loss"] = discrete_ce_loss.detach().float().mean().item()
            if discrete_z_loss is not None:
                metrics["discrete_z_loss"] = discrete_z_loss.detach().float().mean().item()
            losses.append(flow_loss)
            metrics["action_flow_loss"] = flow_loss.detach().float().mean().item()

        loss = torch.stack(losses).sum(dim=0)
        metrics["loss"] = loss.detach().float().mean().item()
        return loss, metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """通过连续 flow matching 或离散自回归解码生成动作块。"""
        if "action_mode" in kwargs:
            raise TypeError(
                "MolmoAct2 predict_action_chunk got unexpected keyword argument 'action_mode'; "
                "use 'inference_action_mode'."
            )
        model_inputs = self._model_inputs(batch)
        inference_action_mode = self._resolve_inference_action_mode(kwargs.get("inference_action_mode"))
        num_steps = kwargs.get("num_steps", getattr(self.config, "num_inference_steps", None))
        generator = kwargs.get("generator")
        device = next(self.parameters()).device
        batch_size = int(next(iter(model_inputs.values())).shape[0])
        if generator is None:
            generator = self._rollout_generator_for_inputs(
                batch,
                batch_size=batch_size,
                device=device,
            )
        action_dim = self._output_action_dim(batch)
        with self._autocast_context():
            if inference_action_mode == "discrete":
                if self._rtc_enabled():
                    raise ValueError("RTC is only supported for continuous MolmoAct2 inference.")
                actions = self._generate_discrete_actions_from_inputs(
                    model_inputs=model_inputs,
                    action_dim=action_dim,
                )
            elif self._rtc_enabled():
                actions = self._generate_actions_from_inputs_with_rtc(
                    model_inputs=model_inputs,
                    action_dim_is_pad=batch.get("action_dim_is_pad"),
                    num_steps=num_steps,
                    generator=generator,
                    inference_delay=kwargs.get("inference_delay"),
                    prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            else:
                generation_kwargs: dict[str, Tensor] = {}
                # LeRobot 检查点会在外层策略配置中记录其实际的训练动作
                # 模式，而其不可变的 HF 基础模型可能仍标记为 ``both``。
                # 对于已保存的 LeRobot 检查点，在推理时复用连续训练的
                # 掩码。原始 HF 检查点没有保存的外层模式，保留其原生的
                # BOTH 模式推理行为。
                if self._checkpoint_action_mode is not None:
                    generation_kwargs["encoder_attention_mask"] = (
                        self._encoder_attention_mask_for_action_expert(
                            input_ids=model_inputs.get("input_ids"),
                            attention_mask=model_inputs.get("attention_mask"),
                        )
                    )
                actions = self._backbone().generate_actions_from_inputs(
                    **model_inputs,
                    action_dim_is_pad=batch.get("action_dim_is_pad"),
                    action_horizon=self._generation_action_horizon(),
                    num_steps=num_steps,
                    generator=generator,
                    **generation_kwargs,
                )
        return actions[:, : self.config.n_action_steps, :action_dim].to(dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """从队列中取出一个动作步，队列为空时重新生成动作块。"""
        if self._rtc_enabled():
            raise AssertionError("RTC is not supported for select_action, use it with predict_action_chunk")
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def _get_default_peft_targets(self) -> dict[str, Any]:
        target_modules = self._lora_target_modules(
            prefix=r"model\.model",
            lm_head_path=r"model\.lm_head",
        )
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
            "r": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "bias": self.config.lora_bias,
            # 下面会在隔离的种子下应用官方的高斯初始化。先避免使用
            # PEFT 默认的 Kaiming 重置，否则会消耗全局训练的
            # RNG 流。
            "init_lora_weights": False,
        }

    def _get_inner_peft_targets(self) -> dict[str, Any]:
        target_modules = self._lora_target_modules(prefix="model", lm_head_path="lm_head")
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
            "r": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "bias": self.config.lora_bias,
            "init_lora_weights": False,
        }

    def _lora_target_modules(self, *, prefix: str, lm_head_path: str) -> str:
        vlm_linear_leaves = "w1|w2|w3|wq|wk|wv|wo|att_proj|attn_out|ff_proj|ff_out|patch_embedding"
        # 原生模型将其词表投影放在 ``transformer.ff_out``。HF 转换把
        # 这个权重移到了顶层的 ``lm_head``；显式包含它，使 LoRA 覆盖
        # 与官方 MolmoAct2 相同的线性模块。原生 MolmoAct2 还把图像
        # pooling 和投影放在 ``vision_backbone`` 内部。HF 转换把这些
        # connector 模块移到了 ``vision_backbone`` 旁边，因此也要显式
        # 包含它们转换后的父路径。
        return (
            rf"(?:{prefix}\.(transformer|vision_backbone|image_pooling_2d|image_projector)\.(?:.*\.)?"
            rf"({vlm_linear_leaves})|{lm_head_path})$"
        )

    def _build_inner_lora_config(self):
        require_package("peft", extra="molmoact2")

        return LoraConfig(**self._get_inner_peft_targets())

    @staticmethod
    def _official_lora_initialization_sort_key(module_name: str) -> tuple[int, int, int]:
        """返回 HF LoRA 层对应的原生 MolmoAct2 模块遍历顺序。

        官方训练分别将 PEFT 注入原生文本 transformer 和视觉骨干，然后
        通过遍历所得的原生模块树来初始化适配器。转换后的 HF 模型以
        不同的顺序注册了相同的 303 个线性目标。按原生结构顺序排序可以
        让种子 6198 为每个映射模块选出相同的高斯 LoRA-A 矩阵，而不仅仅
        是相同的分布。
        """

        text_match = re.search(
            r"(?:^|\.)transformer\.blocks\.(\d+)\.(?:self_attn|mlp)\."
            r"(attn_out|ff_out|att_proj|ff_proj)$",
            module_name,
        )
        if text_match is not None:
            native_leaf_order = {"attn_out": 0, "ff_out": 1, "att_proj": 2, "ff_proj": 3}
            return (0, int(text_match.group(1)), native_leaf_order[text_match.group(2)])

        if re.search(r"(?:^|\.)lm_head$", module_name) is not None:
            return (1, 0, 0)

        pooling_match = re.search(r"(?:^|\.)image_pooling_2d\.(wq|wk|wv|wo)$", module_name)
        if pooling_match is not None:
            return (2, 0, {"wq": 0, "wk": 1, "wv": 2, "wo": 3}[pooling_match.group(1)])

        projector_match = re.search(r"(?:^|\.)image_projector\.(w1|w2|w3)$", module_name)
        if projector_match is not None:
            return (3, 0, {"w1": 0, "w2": 1, "w3": 2}[projector_match.group(1)])

        if re.search(r"(?:^|\.)image_vit\.patch_embedding$", module_name) is not None:
            return (4, 0, 0)

        vit_match = re.search(
            r"(?:^|\.)image_vit\.transformer\.resblocks\.(\d+)\."
            r"(attention|feed_forward)\.(wq|wk|wv|wo|w1|w2)$",
            module_name,
        )
        if vit_match is not None:
            branch, leaf = vit_match.group(2), vit_match.group(3)
            if branch == "attention":
                leaf_order = {"wq": 0, "wk": 1, "wv": 2, "wo": 3}[leaf]
            else:
                leaf_order = {"w1": 4, "w2": 5}[leaf]
            return (5, int(vit_match.group(1)), leaf_order)

        raise RuntimeError(
            f"MolmoAct2 cannot map a LoRA target into the official initialization order: {module_name!r}."
        )

    def _apply_lora_adapters(self) -> None:
        require_package("peft", extra="molmoact2")

        peft_config = self._build_inner_lora_config()
        self._validate_peft_config(peft_config)

        for param in self.model.parameters():
            param.requires_grad_(False)
        # 为优化器稳定性将 LoRA 适配器保持为 fp32。这是刻意的
        # 显存/精度权衡，在高秩时尤其如此。
        self.model = get_peft_model(
            self.model,
            peft_config,
            autocast_adapter_dtype=True,
        )
        # 与官方训练器保持一致：适配器初始为恒等更新，A 从
        # N(0, 1/r) 采样，B 初始化为零。分叉 RNG 可以在不改变
        # 训练/数据 RNG 流的情况下，使每个 DDP rank 上的初始化
        # 完全相同。
        from peft.tuners.lora.layer import LoraLayer

        lora_layers = [
            (name, module) for name, module in self.model.named_modules() if isinstance(module, LoraLayer)
        ]
        lora_layers.sort(key=lambda item: self._official_lora_initialization_sort_key(item[0]))
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(6198)
            for _, module in lora_layers:
                for adapter_name in module.lora_A:
                    module.reset_lora_parameters(adapter_name, "gaussian")
        self._unfreeze_action_expert_parameters()
        self.train(self.training)

    def _validate_peft_config(self, peft_config) -> None:
        del peft_config
        if not self.config.checkpoint_path:
            raise ValueError("MolmoAct2 LoRA fine-tuning requires `policy.checkpoint_path`.")
