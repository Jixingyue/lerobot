# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn


def sinusoidal_embedding_1d(dim, position):
    # 预处理
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}.")
    half = dim // 2
    position = position.type(torch.float64)

    # 计算
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}.")
    freqs = torch.outer(
        torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # 拆分 freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # 遍历样本
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # 预计算乘子
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # 应用旋转位置编码
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # 加入结果集合
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        参数：
            x(Tensor): 形状 [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        参数：
            x(Tensor): 形状 [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 各层
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    # 注意：FastWAM 从不运行上游 Wan 注意力的 forward。FastWAMAttentionBlock
    # 只复用上面定义的 q/k/v/o/norm 子模块，并通过 `fastwam_masked_attention`
    # （SDPA）计算注意力。原始的 flash-attention forward 已被移除，这也使得
    # 原先的 WanCrossAttention 子类被合并进本类（它只在 forward 上有所不同）：
    # 自注意力和交叉注意力现在共享同一个投影模块。


class WanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # 各层
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )

        # 调制
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    # 注意：上游 Wan 块的 forward（基于 flash-attention 的自注意力 + 交叉注意力 +
    # FFN）已被移除。FastWAM 将此块子类化为 FastWAMAttentionBlock，并重写 forward
    # 以使用带显式布尔掩码的 SDPA；这里只复用 __init__（norm/attention/ffn 子模块）。


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # 各层
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # 调制
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): 形状 [B, L1, C]
            e(Tensor): 形状 [B, L1, C]
        """
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class WanModel(nn.Module):
    r"""
    Wan 扩散骨干网络，同时支持文本生成视频和图像生成视频。
    """

    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        r"""
        初始化扩散模型骨干网络。

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                模型变体 - 't2v'（文本生成视频）或 'i2v'（图像生成视频）
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                视频嵌入的 3D 补丁维度 (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                文本嵌入的固定长度
            in_dim (`int`, *optional*, defaults to 16):
                输入视频通道数 (C_in)
            dim (`int`, *optional*, defaults to 2048):
                transformer 的隐藏维度
            ffn_dim (`int`, *optional*, defaults to 8192):
                前馈网络的中间维度
            freq_dim (`int`, *optional*, defaults to 256):
                正弦时间嵌入的维度
            text_dim (`int`, *optional*, defaults to 4096):
                文本嵌入的输入维度
            out_dim (`int`, *optional*, defaults to 16):
                输出视频通道数 (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                注意力头数量
            num_layers (`int`, *optional*, defaults to 32):
                transformer 块数量
            qk_norm (`bool`, *optional*, defaults to True):
                启用查询/键归一化
            cross_attn_norm (`bool`, *optional*, defaults to False):
                启用交叉注意力归一化
            eps (`float`, *optional*, defaults to 1e-6):
                归一化层的 epsilon 值
        """

        super().__init__()

        if model_type not in ["t2v", "i2v", "ti2v", "s2v"]:
            raise ValueError(f"model_type must be one of ['t2v', 'i2v', 'ti2v', 's2v'], got {model_type!r}.")
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # 嵌入层
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # 块
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps)
                for _ in range(num_layers)
            ]
        )

        # 头部
        self.head = Head(dim, out_dim, patch_size, eps)

        # 缓冲区（不要使用 register_buffer，否则 dtype 会在 to() 中被改变）
        if (dim % num_heads) != 0 or (dim // num_heads) % 2 != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads}) with an even head dim."
            )
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        # 初始化权重
        self.init_weights()

    # 注意：上游 Wan 扩散的 forward（基于 flash-attention）已被移除。
    # FastWAM 的 WanVideoDiT 将本模型子类化，使用 FastWAMAttentionBlock
    # 重建 `self.blocks`，并提供自己基于 SDPA 的 forward。只有构造函数
    # （嵌入层、块、头部、rope 缓冲区）和下面的辅助函数
    # （unpatchify / init_weights）被复用。WanModel 永远不会被直接运行。

    def unpatchify(self, x, grid_sizes):
        r"""
        从补丁嵌入重建视频张量。

        Args:
            x (List[Tensor]):
                补丁化特征列表，每个形状为 [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                补丁化之前的原始时空网格维度，
                    形状为 [B, 3]（3 个维度对应 F_patches、H_patches、W_patches）

        Returns:
            List[Tensor]:
                重建的视频张量，形状为 [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist(), strict=False):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size, strict=False)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        使用 Xavier 初始化初始化模型参数。
        """

        # 基础初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 初始化嵌入层
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # 初始化输出层
        nn.init.zeros_(self.head.head.weight)
