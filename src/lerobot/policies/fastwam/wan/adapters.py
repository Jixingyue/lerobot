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

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan


class WanVideoVAE38(torch.nn.Module):
    """基于 `diffusers.AutoencoderKLWan`（Wan2.2-TI2V-5B）的 FastWAM VAE 契约。

    空间 16 倍 / 时间 4 倍压缩，48 个 latent 通道。diffusers 的
    `AutoencoderKLWan` 返回*原始* latents（不应用 `latents_mean`/
    `latents_std`），因此这里的 `encode`/`decode` 应用与 Wan 参考实现相同的
    标准化——`(latents - mean) / std`——为稳定性在 fp32 下完成。
    `encode` 使用确定性的后验 mode，与返回 latent 均值 `mu` 的原始 VAE 保持一致。
    """

    upsampling_factor = 16
    temporal_downsample_factor = 4
    z_dim = 48

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
        *,
        pretrained: AutoencoderKLWan,
    ) -> None:
        super().__init__()
        # Wan2.2 VAE 是固定的预训练模型——它从不从头训练，因此必须始终提供一个
        # 真实的 `AutoencoderKLWan`（带权重，由 `load_pretrained_wan_vae` 从
        # diffusers 仓库加载）。不存在随机/离线构建路径。
        self.vae = pretrained.to(device=device, dtype=dtype)

        # 从 VAE 自身的 config 中读取标准化统计量（diffusers 从 vae/config.json
        # 填充这些值）——单一事实来源，不做本地拷贝。diffusers 的 encode/decode
        # 返回*原始* latents，因此我们自己应用 (latent - mean) / std。
        # 非持久化：不放入 state_dict。
        self.register_buffer(
            "latents_mean",
            torch.tensor(self.vae.config.latents_mean).view(1, self.z_dim, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(self.vae.config.latents_std).view(1, self.z_dim, 1, 1, 1),
            persistent=False,
        )

    def _device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.vae.parameters())
        return param.device, param.dtype

    def encode(
        self,
        videos: list[torch.Tensor] | torch.Tensor,
        device: str | torch.device | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        del device, tile_size, tile_stride
        if tiled:
            raise NotImplementedError("Tiled Wan2.2 VAE encoding is not supported by the FastWAM adapter.")
        if isinstance(videos, (list, tuple)):
            videos = torch.stack(list(videos))
        dev, dtype = self._device_dtype()
        mu = self.vae.encode(videos.to(device=dev, dtype=dtype)).latent_dist.mode().float()
        mean = self.latents_mean.float().to(mu.device)
        std = self.latents_std.float().to(mu.device)
        return (mu - mean) / std

    def decode(
        self,
        hidden_states: list[torch.Tensor] | torch.Tensor,
        device: str | torch.device | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        del device, tile_size, tile_stride
        if tiled:
            raise NotImplementedError("Tiled Wan2.2 VAE decoding is not supported by the FastWAM adapter.")
        if isinstance(hidden_states, (list, tuple)):
            hidden_states = torch.stack(list(hidden_states))
        dev, dtype = self._device_dtype()
        z = hidden_states.float()
        z = z * self.latents_std.float().to(z.device) + self.latents_mean.float().to(z.device)
        out = self.vae.decode(z.to(device=dev, dtype=dtype)).sample
        return out.float().clamp_(-1.0, 1.0)
