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
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from lerobot.utils.import_utils import _diffusers_available, _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer, UMT5EncoderModel
else:
    AutoTokenizer = None
    UMT5EncoderModel = None

if TYPE_CHECKING or _diffusers_available:
    from diffusers import AutoencoderKLWan
else:
    AutoencoderKLWan = None

from .adapters import WanVideoVAE38
from .video_dit import WanVideoDiT

logger = logging.getLogger(__name__)

# 自定义 MoT 视频 DiT 仍以分片的 `diffusion_pytorch_model*.safetensors` 形式
# 随原始（非 diffusers 的）Wan2.2 仓库发布；VAE 和 UMT5 文本编码器来自
# diffusers 转换版本。分词器是标准的 UMT5 分词器。
WAN_DIT_PATTERN = "diffusion_pytorch_model*.safetensors"
WAN_T5_TOKENIZER = "google/umt5-xxl"
WAN22_DIFFUSERS_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


class WanTextEncoder(torch.nn.Module):
    """基于 `transformers.UMT5EncoderModel` 的 FastWAM 文本编码器契约。

    暴露 `.dim`（隐藏层大小）和 `forward(ids, mask) -> [B, L, dim]`，
    与 `FastWAM.encode_prompt` 中的调用方式匹配。
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        *,
        pretrained: torch.nn.Module,
    ) -> None:
        super().__init__()
        # UMT5-XXL 是固定的预训练编码器——从不从头训练，因此必须始终提供一个
        # 真实的 `UMT5EncoderModel`（带权重，由 `load_pretrained_wan_text_encoder`
        # 从 diffusers 仓库加载）。不存在随机/离线构建。
        self.model = pretrained.to(device=device, dtype=dtype)
        self.dim = int(self.model.config.d_model)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids, attention_mask=mask.long()).last_hidden_state


class WanTokenizer:
    """UMT5 分词器封装，按 FastWAM 调用处所期望的方式返回
    `(input_ids, attention_mask)`。"""

    def __init__(self, name: str = WAN_T5_TOKENIZER, seq_len: int = 512) -> None:
        require_package("transformers", extra="fastwam")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.seq_len = int(seq_len)

    def __call__(
        self,
        sequence: str | Sequence[str],
        return_mask: bool = False,
        add_special_tokens: bool = True,
        **_: Any,
    ):
        if isinstance(sequence, str):
            sequence = [sequence]
        out = self.tokenizer(
            list(sequence),
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        if return_mask:
            return out.input_ids, out.attention_mask
        return out.input_ids


def build_wan_tokenizer(*, model_id: str = WAN_T5_TOKENIZER, tokenizer_max_len: int) -> WanTokenizer:
    return WanTokenizer(name=model_id, seq_len=int(tokenizer_max_len))


def load_pretrained_wan_vae(*, torch_dtype: torch.dtype, device: str) -> WanVideoVAE38:
    """从 diffusers 仓库加载真实的 Wan2.2 VAE 权重（离线创建基础模型）。"""
    require_package("diffusers", extra="fastwam")
    vae = AutoencoderKLWan.from_pretrained(WAN22_DIFFUSERS_MODEL_ID, subfolder="vae", torch_dtype=torch_dtype)
    return WanVideoVAE38(dtype=torch_dtype, device=device, pretrained=vae)


def load_pretrained_wan_text_encoder(
    *,
    model_id: str = WAN22_DIFFUSERS_MODEL_ID,
    subfolder: str | None = "text_encoder",
    torch_dtype: torch.dtype,
    device: str,
) -> WanTextEncoder:
    """加载 UMT5-XXL 编码器权重（默认为 Wan2.2 的 diffusers 仓库）。

    必须与分词器保持兼容（参见 `build_wan_tokenizer`）：编码器的嵌入表
    按分词器的词表进行索引。
    """
    require_package("transformers", extra="fastwam")
    encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder=subfolder, torch_dtype=torch_dtype)
    return WanTextEncoder(dtype=torch_dtype, device=device, pretrained=encoder)


def resolve_wan_dit_paths(
    model_id_or_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
) -> list[Path]:
    """从原始 Wan2.2 仓库或本地目录解析自定义 MoT DiT 的分片文件。"""
    path = Path(model_id_or_path).expanduser()
    if path.is_dir():
        return sorted(path.glob(WAN_DIT_PATTERN))

    snapshot_path = snapshot_download(
        repo_id=str(model_id_or_path),
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=[WAN_DIT_PATTERN],
    )
    return sorted(Path(snapshot_path).glob(WAN_DIT_PATTERN))


def load_wan_video_dit(
    paths: list[str | Path],
    *,
    dit_config: dict[str, Any],
    torch_dtype: torch.dtype,
    device: str,
) -> WanVideoDiT:
    model = WanVideoDiT(**dit_config)
    state_dict = _read_wan_dit_safetensors(paths)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device=device, dtype=torch_dtype)


def _read_wan_dit_safetensors(paths: list[str | Path]) -> dict[str, torch.Tensor]:
    state_dict = {}
    for path in paths:
        state_dict.update(load_file(str(path), device="cpu"))
    return state_dict
