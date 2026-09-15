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
Wall-X：基于 Qwen2.5-VL 并使用 flow matching 的跨本体机器人控制。

[论文](https://github.com/x2-robot/wall-x)

安装 wall-x 的额外依赖：
```bash
pip install -e ".[wall_x]"
```

微调 wall-x 模型的示例：
```bash
lerobot-train \
--policy.type=wall_x \
--dataset.repo_id=your/dataset \
--batch_size=32 \
--steps=100000
```
"""

import logging
import math
from collections import deque
from os import PathLike
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from safetensors import SafetensorError
from safetensors.torch import load_file
from torch import Tensor
from torch.distributions import Beta
from torch.nn import CrossEntropyLoss

from lerobot.utils.constants import ACTION, MESSAGES_RENDERED
from lerobot.utils.import_utils import (
    _wallx_deps_available,
    require_package,
)
from lerobot.utils.language import require_single_text_output

from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_wall_x import WallXConfig
from .constant import WALL_X_GENERATION_PROMPT_IDS

if TYPE_CHECKING or _wallx_deps_available:
    from peft import LoraConfig, get_peft_model
    from torchdiffeq import odeint
    from transformers import AutoProcessor, BatchFeature
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
        Qwen2_5_VLForConditionalGeneration,
    )
    from transformers.utils import cached_file, is_torchdynamo_compiling

    from .qwen_model import (
        Qwen2_5_VLACausalLMOutputWithPast,
        Qwen2_5_VLConfig,
        Qwen2_5_VLMoEModel,
        configure_wall_x_vision_attention,
    )
else:
    LoraConfig = None
    get_peft_model = None
    odeint = None
    AutoProcessor = None
    BatchFeature = None
    Qwen2_5_VLForConditionalGeneration = None
    cached_file = None
    is_torchdynamo_compiling = None
    Qwen2_5_VLConfig = None
    Qwen2_5_VisionTransformerPretrainedModel = None
    Qwen2_5_VLACausalLMOutputWithPast = None
    Qwen2_5_VLMoEModel = None
    configure_wall_x_vision_attention = None


logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    """用于扩散时间步的正弦位置嵌入。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionHead(nn.Module):
    """
    基于 flow matching 的动作预测头。

    实现了 Beta 分布的噪声调度和时间嵌入，
    用于动作序列预测。
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.action_dim = sum(config.dof_config.values())
        self.propri_dim = sum(config.agent_pos_config.values())
        self.hidden_size = config.hidden_size

        # 用于噪声调度的 Beta 分布
        self.beta_alpha = 1.5
        self.beta_beta = 1.0
        self.s = 0.999

        # 正弦时间步嵌入
        self.time_embed = SinusoidalPosEmb(config.hidden_size)

        # 动作嵌入网络
        # 乘 2 是为了拼接 action + DOF mask
        self.w1 = nn.Linear(self.action_dim * 2, self.hidden_size, bias=False)
        self.w2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)  # 乘 2 是为了 action + time
        self.w3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        # 投影回动作空间
        self.action_proj_back = nn.Linear(self.hidden_size, self.action_dim, bias=False)

        # 本体感知投影
        self.propri_proj = nn.Linear(self.propri_dim * 2, self.hidden_size, bias=False)

    def sample_time(self, batch_size, device):
        """使用 Beta 分布采样时间步（为保证数值稳定性，始终使用 float32）。"""
        beta_dist = Beta(
            torch.tensor(self.beta_alpha, dtype=torch.float32, device=device),
            torch.tensor(self.beta_beta, dtype=torch.float32, device=device),
        )
        sample = beta_dist.sample([batch_size])
        time = (1 - sample) * self.s
        return time

    def forward(self, action_chunk, dof_mask=None):
        """
        训练时对动作序列进行带噪声注入的处理。

        Args:
            action_chunk: 动作序列 [batch, seq_len, action_dim]
            dof_mask: 自由度掩码 [batch, seq_len, action_dim]

        Returns:
            tuple: (action_embeddings, flow_target)
        """
        batch_size = action_chunk.shape[0]
        device = action_chunk.device
        weight_dtype = self.w1.weight.dtype

        # 在 autocast 之外采样时间（Beta 分布需要 float32）
        time = self.sample_time(batch_size, device)
        t = time.unsqueeze(-1).unsqueeze(-1)

        # 在 float32 下计算噪声和 flow
        noise = torch.randn_like(action_chunk, dtype=torch.float32)
        action_chunk_f32 = action_chunk.to(torch.float32)
        noisy_action = (1 - t) * noise + t * action_chunk_f32
        flow = action_chunk_f32 - noise

        # 投影带噪动作
        if dof_mask is not None:
            noisy_action = torch.cat([noisy_action, dof_mask.to(torch.float32)], dim=-1)

        # 转换为权重的 dtype 以供线性层使用
        noisy_action = noisy_action.to(dtype=weight_dtype)
        action_embed = self.w1(noisy_action)

        # 生成时间嵌入并进行组合
        time_embed = self.time_embed(time)
        time_embed = time_embed.unsqueeze(1).repeat(1, action_embed.shape[1], 1)
        time_embed = time_embed.to(dtype=weight_dtype)

        concat_embed = torch.cat([action_embed, time_embed], dim=-1)
        concat_embed = self.w2(concat_embed)
        embed = self.w3(self.act_fn(concat_embed))

        return embed, flow

    def step(self, timestep, noisy_action, dof_mask=None):
        """推理时的单个去噪步骤。"""
        weight_dtype = self.w1.weight.dtype

        if dof_mask is not None:
            noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)
        noisy_action = noisy_action.to(dtype=weight_dtype)

        time_embed = self.time_embed(timestep)
        action_embed = self.w1(noisy_action)

        time_embed = time_embed.unsqueeze(1).repeat(1, action_embed.shape[1], 1)
        time_embed = time_embed.to(device=noisy_action.device, dtype=weight_dtype)

        concat_embed = torch.cat([action_embed, time_embed], dim=-1)
        concat_embed = self.w2(concat_embed)
        embed = self.w3(self.act_fn(concat_embed))

        return embed

    def flow_loss(self, action_hidden_states, flow, dof_mask=None):
        """计算 flow matching 损失（为保证稳定性，所有计算都在 float32 下进行）。"""
        # 确保所有输入都是 float32
        action_hidden_states = action_hidden_states.to(torch.float32)
        flow = flow.to(torch.float32)

        action_pred = self.action_proj_back(action_hidden_states)
        loss = functional.mse_loss(action_pred, flow, reduction="none")

        if dof_mask is not None:
            dof_mask = dof_mask.reshape(-1, dof_mask.shape[-1]).to(torch.float32)
            loss = loss * dof_mask

        return loss

    def proprioception_proj(self, proprioception, dof_mask=None):
        """将本体感知数据投影到隐藏空间。"""
        # 确保设备和 dtype 正确对齐
        proprioception = proprioception.to(device=self.propri_proj.weight.device).to(
            dtype=self.propri_proj.weight.dtype
        )

        if dof_mask is not None:
            # 将本体感知与 DOF mask 拼接
            # TODO: 使用基于变量的维度检查以获得更好的灵活性
            proprioception = torch.cat([proprioception, dof_mask], dim=-1)

        proprioception = proprioception.to(device=self.propri_proj.weight.device).to(
            dtype=self.propri_proj.weight.dtype
        )
        return self.propri_proj(proprioception)


# 条件基类：当 transformers 不可用时，该类仍可被解析（继承自 nn.Module），
# 但无法实例化——WallXPolicy.__init__ 中的 require_package 会在此之前向用户
# 给出清晰的错误提示。
_Qwen2_5_VLForAction_Base = Qwen2_5_VLForConditionalGeneration if _wallx_deps_available else nn.Module


class Qwen2_5_VLMoEForAction(_Qwen2_5_VLForAction_Base):  # noqa: N801
    """
    用于动作处理的 Qwen2.5 视觉-语言混合专家（Mixture of Experts）模型。

    该模型在基础 Qwen2.5 VL 模型之上扩展了动作 token 处理能力，
    并可选地支持 LoRA 微调。
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer_with_MoE", "Qwen2_5_VLVisionBlock"]

    def init_weights(self):
        if getattr(self.model, "language_model", None) is not None:
            return
        super().init_weights()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        config=None,
        action_tokenizer_path=None,
        attn_implementation: str = "eager",
        vision_attn_implementation: str = "auto",
        cache_dir: str | PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        strict: bool = False,
        **kwargs: Any,
    ):
        """
        从预训练模型路径加载模型。

        Args:
            pretrained_model_path (str): 包含 model.safetensors 文件的模型目录路径
            config_path (str, optional): 配置文件路径；如果为 None，将在 pretrained_model_path 中查找 qwen25_config.json
            action_tokenizer_path (str, optional): 动作分词器路径；如果为 None，将从默认配置加载
            attn_implementation (str, optional): 注意力实现；如果为 None，将从默认配置加载
            vision_attn_implementation (str, optional): 视觉注意力后端。``auto`` 会在支持时使用 packed
                变长注意力，否则回退到 SDPA。
            **kwargs: 其他参数

        Returns:
            Qwen2_5_VLMoEForAction: 加载后的模型实例
        """
        Qwen2_5_VLMoEModel._require_eager_attention(attn_implementation)
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                strict=strict,
                **kwargs,
            )
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        processor = AutoProcessor.from_pretrained(
            pretrained_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_fast=True,
        )
        if action_tokenizer_path is not None:
            action_tokenizer = AutoProcessor.from_pretrained(action_tokenizer_path, trust_remote_code=True)
            processor.action_processor = action_tokenizer
        else:
            action_tokenizer = None

        # 将 pad_token_id 添加到 config
        config.pad_token_id = processor.tokenizer.pad_token_id
        config.text_config.pad_token_id = processor.tokenizer.pad_token_id

        # 使用配置和处理器初始化模型
        model = cls(
            config,
            processor=processor,
            action_tokenizer=action_tokenizer,
            vision_attn_implementation=vision_attn_implementation,
            **kwargs,
        )

        # 调整 token 嵌入大小以匹配处理器分词器的词表大小
        model.resize_token_embeddings(len(processor.tokenizer))

        logger.info("Loading Wall-X model from %s", pretrained_name_or_path)
        try:
            resolved_file = cached_file(
                pretrained_name_or_path,
                "model.safetensors",
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=kwargs.get("resume_download"),
                proxies=kwargs.get("proxies"),
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
            sd = load_file(resolved_file)
        except (OSError, SafetensorError) as error:
            raise OSError(
                f"Failed to load pretrained Wall-X weights from {pretrained_name_or_path!r}"
            ) from error
        logger.info("Loaded Wall-X state dict from model.safetensors")

        state_dict = {}
        # 过滤掉归一化器的统计参数
        del_keys = []
        for key in sd:
            if "action_preprocessor.normalizer" in key:
                del_keys.append(key)
        for key in del_keys:
            del sd[key]
        state_dict.update(sd)

        model.load_state_dict(state_dict, strict=False)

        return model

    def __init__(
        self,
        config,
        use_fast_tokenizer=False,
        processor=None,
        action_tokenizer=None,
        action_mapper=None,
        flow_loss_weight=1.0,
        vision_attn_implementation: str = "auto",
    ):
        """
        初始化用于动作处理的 Qwen2.5 VLMoE 模型。

        Args:
            config: 模型配置
            use_fast_tokenizer (bool): 是否使用快速分词器
            processor: 文本和图像处理器
            action_tokenizer: 动作专用分词器
            action_mapper: 动作映射工具
            flow_loss_weight (float): flow loss 计算的权重
        """
        Qwen2_5_VLMoEModel._require_eager_attention(config._attn_implementation)
        config._attn_implementation = "eager"
        # 文本的动作 token 岛需要使用 eager attention。视觉部分没有这种约束，
        # 因此保留其可移植的原生 SDPA 回退实现。
        config.vision_config._attn_implementation = "sdpa"
        super().__init__(config)

        # 初始化视觉 transformer 和语言模型组件
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        configure_wall_x_vision_attention(self.visual, vision_attn_implementation)
        self.model = Qwen2_5_VLMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化损失函数，不做 reduction，以便按通道计算损失
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.flow_loss_weight = flow_loss_weight
        self.use_fast_tokenizer = use_fast_tokenizer
        self.processor = processor
        self.action_tokenizer = action_tokenizer

        # 定义动作 token ID
        self.define_action_token_id()

        # rope deltas 的缓存
        self.rope_deltas = None

        # 初始化动作预处理器
        self.action_preprocessor = ActionHead(config)

        # 如果配置中有指定，则应用 LoRA
        if hasattr(config, "use_lora") and config.use_lora:
            self.add_lora(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
            )

        # 初始化权重并执行最终处理
        self.post_init()

    def to_bfloat16_for_selected_params(self):
        self.to(dtype=torch.bfloat16)

        params_to_keep_float32 = []

        for name, _param in self.named_parameters():
            if "input_layernorm" in name or "post_attention_layernorm" in name or "model.norm" in name:
                params_to_keep_float32.append(name)
            if "action_preprocessor" in name:
                params_to_keep_float32.append(name)

        for name, param in self.named_parameters():
            if name in params_to_keep_float32:
                param.data = param.data.to(torch.float32)

    def define_action_token_id(self):
        """
        根据分词器配置定义动作 token ID。

        为快速动作 token、本体感知 token 和通用动作 token 创建映射。
        """
        # 创建快速动作 token ID 列表
        fast_action_token_list = []
        if self.use_fast_tokenizer:
            for i in range(self.processor.tokenizer.init_kwargs["action_token_vocab_size"]):
                action_token_id = self.processor.tokenizer.convert_tokens_to_ids(f"<|action_token_{i}|>")
                fast_action_token_list.append(action_token_id)

        # 获取特殊动作 token ID
        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")
        propri_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|propri|>")

        # 保存动作 token ID 映射
        self.action_token_id_set = {
            "fast_action_token_list": fast_action_token_list,
            "propri_token_id": propri_token_id,
            "action_token_id": action_token_id,
        }

    def add_lora(self, r=8, lora_alpha=32, target_modules=None, lora_dropout=0.1):
        """
        为模型添加 LoRA（Low-Rank Adaptation，低秩适配）适配器。

        Args:
            r (int): 适配的秩
            lora_alpha (int): LoRA 缩放参数
            target_modules (list): 要应用 LoRA 的模块名称列表
            lora_dropout (float): LoRA 层的 dropout 概率
        """
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

        # 打印可训练参数的信息
        self.model.print_trainable_parameters()

    def get_input_embeddings(self):
        """获取输入嵌入层。"""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """设置输入嵌入层。"""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """获取输出嵌入层。"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """设置输出嵌入层。"""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """设置 decoder 模型。"""
        self.model = decoder

    def get_decoder(self):
        """获取 decoder 模型。"""
        return self.model

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        为视觉和文本 token 计算 3D RoPE（旋转位置嵌入，Rotary Position Embedding）索引。

        该方法计算的位置嵌入会考虑视觉 token（图像/视频）的时间、高度和宽度维度，
        同时为文本 token 保持标准的 1D 位置嵌入。

        对于视觉 token，3D 位置嵌入基于以下维度计算：
        - 时间维度：视频中的时间 patch
        - 高度维度：图像/视频帧中的垂直 patch
        - 宽度维度：图像/视频帧中的水平 patch

        对于文本 token，使用标准 1D 位置嵌入，并从最大视觉位置 ID 加 1 处继续编号。

        Args:
            input_ids (torch.LongTensor, optional): 形状为 (batch_size, sequence_length) 的输入 token ID
            image_grid_thw (torch.LongTensor, optional): 图像网格维度 (num_images, 3)，对应 [temporal, height, width]
            video_grid_thw (torch.LongTensor, optional): 视频网格维度 (num_videos, 3)，对应 [temporal, height, width]
            second_per_grid_ts (torch.Tensor, optional): 每个时间网格的时间间隔 (num_videos,)
            attention_mask (torch.Tensor, optional): 注意力掩码 (batch_size, sequence_length)

        Returns:
            tuple:
                - position_ids (torch.LongTensor): 形状为 (3, batch_size, sequence_length) 的 3D 位置 ID
                - mrope_position_deltas (torch.Tensor): 形状为 (batch_size, 1) 的 mRoPE 位置增量
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)

            # 初始化 3D 位置 ID 张量
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)

            # 处理批中的每条序列
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0

                # 查找视觉 token 并统计图像/视频数量
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                # 处理每个视觉 token（图像或视频）
                for _ in range(image_nums + video_nums):
                    # 查找下一个图像或视频 token
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1

                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1

                    # 判断当前处理的是图像 token 还是视频 token
                    if ed_image < ed_video:
                        # 处理图像 token
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        # 处理视频 token
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    # 计算空间合并之后的网格维度
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    # 为视觉 token 之前的文本 token 添加位置 ID
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # 计算视觉 token 的 3D 位置嵌入
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    # 计算带时间缩放的时间位置 ID
                    time_tensor = (
                        expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                    )
                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    # 计算空间位置 ID
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    )

                    # 添加视觉 token 的 3D 位置 ID
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                # 为剩余的文本 token 添加位置 ID
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # 拼接该序列的所有位置 ID
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            # 处理没有视觉 token 的情况 - 使用标准 1D 位置嵌入
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def train_step_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        moe_token_types: torch.LongTensor | None = None,  # MoE token 类型分配
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        action_chunk: torch.FloatTensor | None = None,  # 动作轨迹块
        proprioception: torch.FloatTensor | None = None,  # 关节位置/姿态数据
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        dof_mask: torch.FloatTensor | None = None,
        agent_pos_mask: torch.FloatTensor | None = None,
        **kwargs,
    ) -> tuple | Qwen2_5_VLACausalLMOutputWithPast:
        """
        使用包含视觉、文本和动作数据的多模态输入进行训练的前向传播。

        该方法处理训练期间完整的前向传播，处理各种输入模态，包括图像、视频、文本、
        本体感知数据和动作序列。它会同时为语言建模和使用 flow matching 的动作预测计算损失。

        Args:
            input_ids (torch.LongTensor, optional): 输入 token ID
            attention_mask (torch.Tensor, optional): 输入 token 的注意力掩码
            position_ids (torch.LongTensor, optional): token 的位置 ID
            past_key_values (List[torch.FloatTensor], optional): 用于生成的缓存键值对
            inputs_embeds (torch.FloatTensor, optional): 预计算的输入嵌入
            moe_token_types (torch.LongTensor, optional): 用于 MoE 路由的 token 类型分配
            labels (torch.LongTensor, optional): 用于损失计算的目标标签
            use_cache (bool, optional): 是否使用键值缓存
            output_attentions (bool, optional): 是否返回注意力权重
            output_hidden_states (bool, optional): 是否返回隐藏状态
            return_dict (bool, optional): 是否返回结构化输出
            pixel_values (torch.Tensor, optional): 图像像素值
            pixel_values_videos (torch.FloatTensor, optional): 视频像素值
            image_grid_thw (torch.LongTensor, optional): 图像网格维度（temporal、height、width）
            video_grid_thw (torch.LongTensor, optional): 视频网格维度（temporal、height、width）
            action_chunk (torch.FloatTensor, optional): 动作轨迹数据块
            proprioception (torch.FloatTensor, optional): 本体感知传感器数据（关节位置等）
            rope_deltas (torch.LongTensor, optional): RoPE 位置增量
            cache_position (torch.LongTensor, optional): 缓存位置索引
            second_per_grid_ts (torch.Tensor, optional): 每个时间网格的时间间隔
            dof_mask (torch.FloatTensor, optional): 动作 token 的自由度掩码
            agent_pos_mask (torch.FloatTensor, optional): 本体感知数据的智能体位置掩码
            **kwargs: 其他关键字参数

        Returns:
            Union[Tuple, Qwen2_5_VLACausalLMOutputWithPast]: 模型输出，包括损失、logits
                和辅助信息；当 return_dict=False 时返回 tuple
        """
        batch_size, seq_length = input_ids.shape

        # 如果未指定，则从模型配置中设置输出配置
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if rope_deltas is not None:
            self.rope_deltas = rope_deltas

        # 如果未提供 position_ids，则计算 RoPE 位置 ID
        # 注意：无法使用 4D 注意力掩码计算 rope deltas。TODO: 修复这一限制
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # 仅在 pre-fill 阶段、每次生成时计算一次 RoPE 索引
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # 使用之前计算好的 rope deltas 来获得正确的位置 ID
            else:
                delta = (
                    (cache_position[0] + self.rope_deltas).to(self.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=self.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # 否则 `deltas` 是一个 int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # 用多模态数据处理输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            # 处理图像嵌入
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # 处理视频嵌入
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).pooler_output
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]

                # 校验视频 token 与特征数量是否匹配
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # 处理本体感知数据（关节位置、姿态等）
            if proprioception is not None:
                mask = input_ids == self.action_token_id_set["propri_token_id"]
                proprioception_rows = mask.any(dim=-1)
                if proprioception_rows.any():
                    active_proprioception = proprioception[proprioception_rows].to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    active_agent_pos_mask = agent_pos_mask[proprioception_rows].to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    active_proprioception = self.action_preprocessor.proprioception_proj(
                        active_proprioception,
                        active_agent_pos_mask,
                    )
                    proprioception_mask = mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(
                        proprioception_mask.to(inputs_embeds.device),
                        active_proprioception.to(inputs_embeds.device, inputs_embeds.dtype),
                    )
            elif self.training:
                # 执行一次哑前向（dummy forward）以确保在 DDP 中注册梯度。
                # 这处理了某些进程有本体感知数据而另一些没有的情况；
                # 否则 DDP 会一直挂起，等待一个永远不会被计算出来的梯度。
                dummy_input = torch.randn(
                    2,
                    self.action_preprocessor.propri_dim * 2,
                    device=inputs_embeds.device,
                )
                dummy_forward = self.action_preprocessor.proprioception_proj(dummy_input)
                dummy_loss = sum(p.sum() for p in dummy_forward)
                inputs_embeds = inputs_embeds + 0 * dummy_loss

            # 处理动作块数据
            if action_chunk is not None:
                mask = input_ids == self.action_token_id_set["action_token_id"]
                action_rows = mask.any(dim=-1)
                if action_rows.any():
                    active_action_chunk = action_chunk[action_rows].to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    active_dof_mask = dof_mask[action_rows].to(inputs_embeds.device, inputs_embeds.dtype)
                    noisy_action_emb, flow = self.action_preprocessor(active_action_chunk, active_dof_mask)
                    action_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                    inputs_embeds = inputs_embeds.masked_scatter(
                        action_mask,
                        noisy_action_emb.to(inputs_embeds.device, inputs_embeds.dtype),
                    )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 通过主模型进行前向传播
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            moe_token_types=moe_token_types,  # 传入 token 类型以供 MoE 路由
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        logits = self.lm_head(hidden_states)

        # 初始化损失计算变量
        loss = None
        cross_entropy_loss, flow_loss = None, None
        channel_loss_dict = None
        channel_loss_count_dict = None

        # 如果提供了标签，则计算损失
        if labels is not None:
            loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)

            # 为语言建模计算标准交叉熵损失
            shift_logits = logits[..., :-1, :].contiguous().to(torch.float32)
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # 将标签移动到正确的设备以支持模型并行
            shift_labels = shift_labels.to(shift_logits.device)
            non_ignored_mask = shift_labels != -100
            _cross_entropy_loss = self.loss_fct(shift_logits, shift_labels)
            cross_entropy_loss = (
                _cross_entropy_loss[non_ignored_mask].mean()
                if non_ignored_mask.any()
                else torch.tensor(0.0, device=shift_logits.device, dtype=torch.float32)
            )

            # 如果交叉熵损失有效，则将其加入总损失
            if not torch.isnan(cross_entropy_loss):
                loss = loss + cross_entropy_loss.to(torch.float32)
            else:
                with torch.no_grad():
                    cross_entropy_loss.detach()

        if action_chunk is not None:
            action_mask = input_ids == self.action_token_id_set["action_token_id"]
            if action_mask.any():
                action_rows = action_mask.any(dim=-1)
                active_dof_mask = dof_mask[action_rows]
                action_hidden_states = hidden_states[action_mask].to(torch.float32)
                flow = flow.reshape(-1, flow.shape[-1]).to(torch.float32)
                _flow_loss = self.action_preprocessor.flow_loss(action_hidden_states, flow, active_dof_mask)
                if isinstance(_flow_loss, torch.Tensor):
                    flow_loss = _flow_loss.mean()
                if loss is not None:
                    loss = loss + self.flow_loss_weight * flow_loss.to(torch.float32)
                else:
                    loss = self.flow_loss_weight * flow_loss.to(torch.float32)
                _flow_loss = _flow_loss.view(
                    active_dof_mask.shape[0],
                    active_dof_mask.shape[1],
                    active_dof_mask.shape[2],
                )

        # 根据 return_dict 设置返回输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLACausalLMOutputWithPast(
            loss=loss,
            cross_entropy_loss=(cross_entropy_loss.clone() if cross_entropy_loss is not None else None),
            flow_loss=flow_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            channel_loss_dict=channel_loss_dict,
            channel_loss_count_dict=channel_loss_count_dict,
        )

    def predict_action(self, predict_mode: str, **kwargs):
        """
        使用指定的预测模式预测动作。

        Args:
            predict_mode (str): 预测模式，"fast" 或 "diffusion"
            **kwargs: 传递给 predict 方法的其他参数

        Returns:
            tuple: (predicted_action, ground_truth_action)，其中 ground_truth_action 可能为 None
        """
        assert predict_mode in ["fast", "diffusion"]

        output = self.predict(predict_mode=predict_mode, **kwargs)

        return output["predict_action"], output.get("gt_action", None)

    @torch.no_grad()
    def predict(
        self,
        predict_mode: str,
        pred_horizon: int | None = None,
        action_dim: int | None = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        moe_token_types: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        action_chunk: torch.FloatTensor | None = None,
        proprioception: torch.FloatTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        num_inference_timesteps: int | None = 10,
        dof_mask: torch.FloatTensor | None = None,
        agent_pos_mask: torch.FloatTensor | None = None,
        generation_prompt_ids: torch.LongTensor | None = None,
        re_generate: bool = False,
        **kwargs,
    ):
        """
        多模态预测方法，支持文本生成、快速动作预测以及基于扩散的动作预测。

        该方法处理三种预测模式：
        1. "text"：使用自回归解码的纯文本生成
        2. "fast"：使用离散动作 token 的快速动作预测
        3. "diffusion"：使用 diffusion/flow matching 的连续动作预测

        Args:
            predict_mode (str): 预测模式（"text"、"fast" 或 "diffusion"）
            pred_horizon (int, optional): 动作序列的预测时域
            action_dim (int, optional): 动作空间的维度
            input_ids (torch.LongTensor, optional): 输入 token ID
            attention_mask (torch.Tensor, optional): 输入 token 的注意力掩码
            position_ids (torch.LongTensor, optional): token 的位置 ID
            past_key_values (List[torch.FloatTensor], optional): 缓存的键值对
            inputs_embeds (torch.FloatTensor, optional): 预计算的输入嵌入
            moe_token_types (torch.LongTensor, optional): 用于 MoE 路由的 token 类型分配
            labels (torch.LongTensor, optional): 用于评估的目标标签
            use_cache (bool, optional): 是否使用键值缓存
            output_attentions (bool, optional): 是否返回注意力权重
            output_hidden_states (bool, optional): 是否返回隐藏状态
            return_dict (bool, optional): 是否返回结构化输出
            pixel_values (torch.Tensor, optional): 图像像素值
            pixel_values_videos (torch.FloatTensor, optional): 视频像素值
            image_grid_thw (torch.LongTensor, optional): 图像网格维度
            video_grid_thw (torch.LongTensor, optional): 视频网格维度
            action_chunk (torch.FloatTensor, optional): 真实动作序列
            proprioception (torch.FloatTensor, optional): 本体感知传感器数据
            rope_deltas (torch.LongTensor, optional): RoPE 位置增量
            cache_position (torch.LongTensor, optional): 缓存位置索引
            second_per_grid_ts (torch.Tensor, optional): 每个时间网格的时间间隔
            num_inference_timesteps (int, optional): 扩散推理步数
            dof_mask (torch.FloatTensor, optional): 自由度掩码
            agent_pos_mask (torch.FloatTensor, optional): 智能体位置掩码
            re_generate (bool, optional): 是否使用采样进行重新生成
            **kwargs: 其他关键字参数

        Returns:
            dict: 包含预测结果的字典，可能的键包括：
                - 'predict_action'：预测的动作序列
                - 'gt_action'：真实动作（如果有）
                - 'input_text'：输入文本（用于 text/fast 模式）
                - 'predict_output_text'：生成的文本（用于 text/fast 模式）
                - 'gt_output_text'：真实文本（用于 text/fast 模式）
        """
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        # text 和 fast 模式进行自回归生成时要求 batch size 为 1
        if predict_mode in ["text", "fast"]:
            assert batch_size == 1, "predict only support batch size 1 for ar generation"

        # 如果未指定，则从模型配置中设置输出配置
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 用多模态数据处理输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            # 处理图像嵌入
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]

                # 校验图像 token 与特征数量是否匹配
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # 处理视频嵌入
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).pooler_output
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]

                # 校验视频 token 与特征数量是否匹配
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # 处理本体感知数据
            if proprioception is not None:
                proprioception = proprioception.to(inputs_embeds.device).to(inputs_embeds.dtype)
                agent_pos_mask = agent_pos_mask.to(inputs_embeds.device).to(inputs_embeds.dtype)
                proprio_embed = self.action_preprocessor.proprioception_proj(
                    proprioception,
                    agent_pos_mask,
                )
                proprioception_mask = input_ids == self.action_token_id_set["propri_token_id"]
                proprio_embed = proprio_embed.to(torch.bfloat16)
                inputs_embeds[proprioception_mask] = proprio_embed.reshape(-1, inputs_embeds.shape[-1])

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 如果未提供 position_ids，则计算 RoPE 位置 ID
        # 注意：无法使用 4D 注意力掩码计算 rope deltas。TODO: 修复这一限制
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # 仅在 pre-fill 阶段、每次生成时计算一次 RoPE 索引
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # 使用之前计算好的 rope deltas 来获得正确的位置 ID
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # 否则 `deltas` 是一个 int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # 如果提供了动作块数据，则进行准备
        if action_chunk is not None:
            action_chunk = action_chunk.to(inputs_embeds.device).to(torch.float32)

        output = {}

        # 为 text 和 fast 模式切分输入序列（diffusion 模式不需要）
        if predict_mode == "text" or predict_mode == "fast":
            if generation_prompt_ids is None:
                raise ValueError(
                    "WALL-X fast/text prediction requires generation_prompt_ids from its input processor."
                )
            generation_prompt_ids = generation_prompt_ids.to(device=input_ids.device, dtype=input_ids.dtype)
            prompt_length = generation_prompt_ids.numel()
            if input_ids.shape[1] < prompt_length:
                matches = torch.empty(0, device=input_ids.device, dtype=torch.bool)
            else:
                matches = (
                    input_ids[0]
                    .unfold(dimension=0, size=prompt_length, step=1)
                    .eq(generation_prompt_ids)
                    .all(dim=-1)
                )

            if matches.any():
                split_pos = torch.nonzero(matches, as_tuple=True)[0][0].item()
                prompt_end = split_pos + prompt_length
                # 提取真实输出 token（包括换行符）
                gt_output_ids = input_ids[:, prompt_end:]
                # 从输入中移除输出部分，保留 prompt
                input_ids = input_ids[:, :prompt_end]
                inputs_embeds = inputs_embeds[:, :prompt_end, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :prompt_end]
                if labels is not None:
                    labels = labels[:, prompt_end:]
            else:
                raise ValueError(
                    "input_ids does not contain the generation prompt tokens <|im_start|>assistant"
                )

            # 解码输入文本以供输出
            input_text = self.processor.batch_decode(
                input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
            )
            output["input_text"] = input_text

        # 使用自回归生成处理 text 和 fast 预测模式
        if predict_mode == "text" or predict_mode == "fast":
            # 为生成初始化 MoE token 类型
            moe_token_types = torch.zeros_like(input_ids)
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "moe_token_types": moe_token_types,
                "image_grid_thw": image_grid_thw,
                "dof_mask": dof_mask,
                "agent_pos_mask": agent_pos_mask,
                "proprioception": proprioception,
            }

            # 生成输出 token
            predict_output_ids = self.generate(
                **batch,
                max_new_tokens=100,
                eos_token_id=[self.processor.tokenizer.eos_token_id],
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                temperature=(1.0 if not re_generate else 0.7),  # 重新生成时使用更高的温度
                do_sample=re_generate,  # 重新生成时启用采样
            )

            # 解码生成文本和真实文本
            gt_output_text = self.processor.batch_decode(
                gt_output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            predict_output_text = self.processor.batch_decode(
                predict_output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            output["gt_output_text"] = gt_output_text
            output["predict_output_text"] = predict_output_text

        # 在 fast 预测模式下将 token 转换为动作
        if predict_mode == "fast":
            action_id = []
            # 从生成的序列中提取动作 token
            for token_id_i in predict_output_ids[0]:
                if token_id_i.item() >= self.processor.tokenizer.init_kwargs["action_token_start_index"]:
                    action_id.append(
                        token_id_i.item() - self.processor.tokenizer.init_kwargs["action_token_start_index"]
                    )

            predict_action = self.processor.action_processor.decode(
                [action_id], time_horizon=pred_horizon, action_dim=action_dim
            )
            # 处理动作解码错误
            if np.sum(predict_action) == 0:
                print("Error in decoding action, predict_action is None")
                output["predict_action"] = None
            else:
                # 将离散 token 转换为连续动作
                predict_action = torch.tensor(predict_action, device=self.device)
                dof_mask = dof_mask.to(self.device).to(pixel_values.dtype)
                # 暂时移除了反归一化步骤
                predict_action = predict_action[:, :, dof_mask[0, 0, :].bool()]
                output["predict_action"] = predict_action

            # 如果有真实动作，则进行处理
            if action_chunk is not None:
                # 应用 DOF 掩码以获得真实动作
                # 暂时移除了反归一化步骤
                action_chunk = action_chunk[:, :, dof_mask[0, 0, :].bool()]
                output["gt_action"] = action_chunk
            else:
                output["gt_action"] = None

        # 处理基于扩散的动作预测
        if predict_mode == "diffusion":
            # 用随机噪声初始化
            noisy_action = torch.randn(
                size=(batch_size, pred_horizon, action_dim),
                dtype=torch.float32,
                device=inputs_embeds.device,
            )
            dof_mask = dof_mask.to(inputs_embeds.device).to(torch.float32)

            def step(timestep, noisy_action):
                """
                扩散过程的单个去噪步骤。

                Args:
                    timestep: 当前扩散时间步
                    noisy_action: 当前的带噪动作估计

                Returns:
                    torch.Tensor: 预测的干净动作
                """
                action_mask = input_ids == self.action_token_id_set["action_token_id"]
                assert action_mask.any(), "No action token found in input_ids"

                # 准备时间步以进行批处理
                timestep = timestep.unsqueeze(0).repeat(noisy_action.shape[0])
                action_embed = self.action_preprocessor.step(
                    timestep=timestep, noisy_action=noisy_action, dof_mask=dof_mask
                )
                action_embed = action_embed.reshape(-1, inputs_embeds.shape[-1])

                # 在赋值前确保 action_embed 具有正确的 dtype 和 device
                action_embed = action_embed.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

                # 创建嵌入的临时副本（clone 会保留 dtype）
                temp_inputs_embeds = inputs_embeds.clone()
                temp_inputs_embeds[action_mask] = action_embed

                # 通过 transformer 进行前向传播
                transformer_outputs = self.model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=temp_inputs_embeds,
                    moe_token_types=moe_token_types,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

                # 从隐藏状态中提取动作预测
                hidden_states = transformer_outputs.last_hidden_state
                action_mask = input_ids == self.action_token_id_set["action_token_id"]
                action_hidden_states = hidden_states[action_mask].to(torch.float32)
                pred = self.action_preprocessor.action_proj_back(action_hidden_states)
                return pred.reshape(batch_size, pred_horizon, action_dim)

            # 执行 ODE 积分以进行扩散采样
            times = torch.linspace(
                0,
                1,
                num_inference_timesteps + 1,
                device=inputs_embeds.device,
                dtype=torch.float32,
            )
            action_trajectory = odeint(step, noisy_action, times, method="euler")

            # 提取最终预测的动作
            # 暂时移除了反归一化步骤
            predict_action = action_trajectory[-1]
            output["predict_action"] = predict_action

            # 如果有真实动作，则进行处理
            # 暂时移除了反归一化步骤
            if action_chunk is not None:
                output["gt_action"] = action_chunk[:, :, dof_mask[0, 0, :].bool()]

        return output

    def forward(self, mode: str | None = None, predict_mode: str | None = "text", **kwargs):
        """
        面向不同执行模式的主前向传播分发器。

        该方法根据指定的模式将执行路由到相应的前向函数：
        - 无模式（None）：禁用梯度的训练步骤
        - 'predict'：预测/推理模式
        - 'train'：启用梯度的训练模式
        - 'validate'：禁用梯度的验证模式

        Args:
            mode (str, optional): 执行模式。如果为 None，默认为不带梯度的训练步骤
            predict_mode (str, optional): 'predict' 模式下的预测模式（"text"、"fast" 或 "diffusion"）
            **kwargs: 传递给所选前向函数的其他参数

        Returns:
            与所选模式相适应的模型输出

        Todo:
            - 在预测模式中增加对多模态数据类型进行区分的支持
        """
        if not mode:
            with torch.no_grad():
                return self.train_step_forward(**kwargs)
        elif mode == "predict":
            return self.predict(predict_mode=predict_mode, **kwargs)
        elif mode == "train":
            return self.train_step_forward(use_cache=False, **kwargs)
        elif mode == "validate":
            with torch.no_grad():
                return self.train_step_forward(use_cache=False, **kwargs)
        else:
            raise NotImplementedError("invalid key")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        moe_token_types=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        proprioception=None,
        dof_mask=None,
        agent_pos_mask=None,
        **kwargs,
    ):
        """
        为支持多模态的自回归生成准备输入。

        该方法处理生成所需的输入准备工作，包括根据缓存位置对输入进行正确切分、
        MoE token 类型管理以及多模态数据处理。
        在生成过程中，视觉输入仅在需要时才会被选择性地传入。

        Args:
            input_ids: 输入 token ID
            past_key_values: 之前生成步骤缓存的键值对
            attention_mask: 输入 token 的注意力掩码
            inputs_embeds: 预计算的输入嵌入
            moe_token_types: 用于 MoE 路由的 token 类型分配
            cache_position: 当前生成的缓存位置
            position_ids: token 的位置 ID
            use_cache: 是否使用键值缓存
            pixel_values: 图像像素值
            pixel_values_videos: 视频像素值
            image_grid_thw: 图像网格维度
            video_grid_thw: 视频网格维度
            second_per_grid_ts: 每个时间网格的时间间隔
            proprioception: 本体感知传感器数据
            dof_mask: 自由度掩码
            agent_pos_mask: 智能体位置掩码
            **kwargs: 其他参数

        Returns:
            dict: 为生成步骤准备好的模型输入

        Todo:
            - 使用各种输入配置对该函数进行充分测试

        Note:
            这是一个被重写的方法，用于处理多模态生成的特殊情况：
            - 通过 cache_position 切分 input_ids，只保留未处理的 token
            - 处理 input_embeds、生成方法以及 GPU 同步的特殊情况
            - 管理视觉输入以避免不必要的前向传播
        """
        if cache_position is None:
            past_length = 0
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_length = int(past_key_values.get_seq_length())
            input_length = input_ids.shape[1]
            end = input_length if input_length > past_length else past_length + input_length
            cache_position = torch.arange(
                past_length,
                end,
                dtype=torch.long,
                device=input_ids.device,
            )
            if cache_position.numel() == 0:
                cache_position = torch.arange(
                    input_length,
                    dtype=torch.long,
                    device=input_ids.device,
                )

        # 如果未提供 MoE token 类型，则进行初始化
        if moe_token_types is None:
            moe_token_types = torch.zeros_like(
                input_ids
            )  # FIXME: 处理改用 input_embeds 的情况
        else:
            # 确保 moe_token_types 的长度与 input_ids 匹配
            if moe_token_types.shape[1] < input_ids.shape[1]:
                # 计算所需的填充长度
                pad_length = input_ids.shape[1] - moe_token_types.shape[1]
                # 创建默认 token 类型 (0) 的填充张量
                pad_tensor = torch.zeros(
                    (moe_token_types.shape[0], pad_length),
                    dtype=moe_token_types.dtype,
                    device=moe_token_types.device,
                )
                # 将填充拼接到已有的 moe_token_types 上
                moe_token_types = torch.cat([moe_token_types, pad_tensor], dim=1)

        # 根据缓存状态和特殊情况处理输入切分
        if past_key_values is not None:
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # 异常情况 4：input_embeds 的情况
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
                moe_token_types = moe_token_types[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None or (  # 异常情况 1：提供了 input_embeds
                is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1]
            ):  # 异常情况 3：GPU 同步的边界情况
                input_ids = input_ids[:, -cache_position.shape[0] :]
                moe_token_types = moe_token_types[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # 默认情况（异常情况 2 为空操作）
                cache_pos = cache_position.clone()
                input_ids = input_ids[:, cache_pos]
                moe_token_types = moe_token_types[:, cache_pos]

        # 在续写步骤（非初始生成）中跳过视觉输入
        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # 确定本生成步骤使用 inputs_embeds 还是 input_ids
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        # 组装生成所需的全部模型输入
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "moe_token_types": moe_token_types,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
                "proprioception": proprioception,
                "dof_mask": dof_mask,
                "agent_pos_mask": agent_pos_mask,
            }
        )
        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取每个样本的图像和视频数量，以计算张量切分长度。

        这些参数直接根据 input_ids 计算，而不是经由 processor 传递，
        以避免接口修改带来的不可预测的影响。

        Args:
            input_ids (torch.LongTensor): 形状为 (batch_size, sequence_length) 的输入 token ID

        Returns:
            tuple:
                - image_nums (torch.LongTensor): 每个样本的图像数量
                - video_nums (torch.LongTensor): 每个样本的视频数量
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        # 查找视觉起始 token 及其紧随其后的 token
        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id

        # 统计视觉起始 token 之后的图像和视频数量
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        """
        为生成扩展输入，支持多模态张量。

        这是一个被重写的方法，支持扩展没有标准 batch 大小维度的张量，
        特别是与视觉相关的张量：
        - pixel_values.shape[0] = 所有图像样本的 sequence_lengths 之和
        - image_grid_thw.shape[0] = 所有样本的 num_images 之和
        - 视频张量采用类似的模式

        Args:
            expand_size (int): 扩展输入的倍数（用于 beam search 等）
            is_encoder_decoder (bool): 是否使用 encoder-decoder 架构
            input_ids (torch.LongTensor, optional): 输入 token ID
            **model_kwargs: 需要扩展的其他模型参数

        Returns:
            tuple: (expanded_input_ids, expanded_model_kwargs)
        """
        if expand_size == 1:
            return input_ids, model_kwargs

        # 定义需要特殊处理的、与视觉相关的张量对应的键
        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            """根据每个样本的图像/视频数量扩展与视觉相关的张量。"""
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                """按长度切分张量，并对每个样本进行重复。"""
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # 将图像按样本切分并计算序列长度
                    samples = torch.split(image_grid_thw, list(image_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # 根据每个样本的图像数量进行扩展
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    # 将视频按样本切分并计算序列长度
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    # 根据每个样本的视频数量进行扩展
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    # 处理列表类型的时间网格数据
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(
                            f"Expected value for key '{key}' to be a list, but got {type(dict_to_expand[key])} instead."
                        )
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = list(video_nums)
                    tensor = _repeat_interleave_samples(tensor, lengths=lengths, repeat_times=expand_size)
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            """使用 repeat_interleave 扩展标准张量。"""
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # 仅在 input_ids 可用于统计图像/视频数量时才扩展视觉输入。
        # 如果 input_ids 不可用，视觉输入就不会被使用，因此无需扩展。
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        # 使用标准的 repeat_interleave 扩展 input_ids
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        # 扩展所有其他模型参数
        model_kwargs = _expand_dict_for_generation(model_kwargs)

        # 处理 encoder-decoder 特有的扩展
        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


class WallXPolicy(PreTrainedPolicy):
    """
    用于跨本体机器人控制的 Wall-X 策略。

    将 Qwen2.5-VL 视觉-语言模型与动作预测相集成，
    使用 flow matching 处理连续动作空间。
    """

    config_class = WallXConfig
    name = "wall_x"

    def __init__(self, config: WallXConfig, **kwargs):
        require_package("transformers", extra="wallx")
        require_package("peft", extra="wallx")
        require_package("torchdiffeq", extra="wallx")
        require_package("qwen-vl-utils", extra="wallx", import_name="qwen_vl_utils")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # 初始化 wall-x 模型
        self.model = Qwen2_5_VLMoEForAction.from_pretrained(
            pretrained_name_or_path=config.pretrained_name_or_path,
            action_tokenizer_path=config.action_tokenizer_path,
            attn_implementation=config.attn_implementation,
            vision_attn_implementation=config.vision_attn_implementation,
        )
        self.model.to(config.device)
        self.model.to_bfloat16_for_selected_params()

        self.reset()

    def reset(self):
        """重置动作队列。"""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self):
        """获取用于优化的参数。"""
        return self.parameters()

    def _pretokenized_inputs(
        self,
        batch: dict[str, Any],
        *,
        compute_position_ids: bool = False,
        text_generation: bool = False,
    ) -> BatchFeature:
        names = (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
            "labels",
            "proprioception",
            "agent_pos_mask",
            "action_chunk",
            "dof_mask",
            "moe_token_types",
            "frame_index",
        )
        inputs = BatchFeature({name: batch[name] for name in names if name in batch})
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "moe_token_types"}
        missing = sorted(required - inputs.keys())
        if missing:
            raise ValueError(
                f"WALL-X requires tokenized inputs from its policy preprocessor; missing {missing}."
            )
        if text_generation:
            keep = required | {"video_grid_thw", "second_per_grid_ts"}
            inputs = BatchFeature({name: value for name, value in inputs.items() if name in keep})
        if compute_position_ids:
            position_ids, rope_deltas = self.model.get_rope_index(
                inputs.input_ids,
                inputs.get("image_grid_thw"),
                inputs.get("video_grid_thw"),
                inputs.get("second_per_grid_ts"),
                inputs.attention_mask,
            )
            inputs["position_ids"] = position_ids
            inputs["rope_deltas"] = rope_deltas
        return inputs

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        使用 Qwen2_5_VLMoEForAction 进行训练前向传播。

        Args:
            batch: 包含来自 preprocess_inputs() 的预处理输入的字典。
                   预期的键：input_ids、attention_mask、pixel_values、image_grid_thw、
                   proprioception、agent_pos_mask、action_chunk、dof_mask、moe_token_types
                   等。

        Returns:
            tuple: (loss, loss_dict)
        """
        recipe_supervision = MESSAGES_RENDERED in batch
        batch = self._pretokenized_inputs(batch, compute_position_ids=True)

        # 以 mode="train" 调用底层模型的 forward
        outputs = self.model(**batch, mode="train")

        flow_loss = outputs.flow_loss
        text_loss = outputs.cross_entropy_loss
        if recipe_supervision:
            loss = None
            if flow_loss is not None:
                loss = self.config.flow_loss_weight * flow_loss
            if text_loss is not None:
                weighted_text_loss = self.config.text_loss_weight * text_loss
                loss = weighted_text_loss if loss is None else loss + weighted_text_loss
            if loss is None:
                raise RuntimeError(
                    "WALL-OSS batch produced neither action nor text supervision. "
                    "Check the selected recipe and target annotations."
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("WALL-OSS produced a non-finite training loss.")
        else:
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("WALL-OSS action-only batch produced no training loss.")

        loss_dict = {"loss": loss.detach()}

        if outputs.flow_loss is not None:
            loss_dict["flow_loss"] = outputs.flow_loss.detach()
        if outputs.cross_entropy_loss is not None:
            loss_dict["cross_entropy_loss"] = outputs.cross_entropy_loss.detach()

        # 如果存在逐通道损失，则添加进去
        if outputs.channel_loss_dict is not None:
            for key, value in outputs.channel_loss_dict.items():
                if isinstance(value, torch.Tensor):
                    loss_dict[f"channel_{key}"] = value.detach()

        return loss, loss_dict

    def supports_text_generation(self) -> bool:
        return True

    @torch.no_grad()
    def generate_text(self, batch: dict[str, Tensor]) -> str:
        """根据契约渲染后的消息和当前观测解码出一条回复。"""
        self.eval()
        inputs = self._pretokenized_inputs(batch, text_generation=True)
        prompt_length = inputs.input_ids.shape[1]
        sampling = self.config.text_temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": 100,
            "min_new_tokens": 0,
            "do_sample": sampling,
            "eos_token_id": self.model.processor.tokenizer.eos_token_id,
            "pad_token_id": self.model.processor.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if sampling:
            generation_kwargs.update(
                temperature=self.config.text_temperature,
                top_p=self.config.text_top_p,
            )
        output_ids = self.model.generate(**inputs, **generation_kwargs)
        outputs = [
            value.strip()
            for value in self.model.processor.tokenizer.batch_decode(
                output_ids[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        ]
        return require_single_text_output(outputs, policy_name="WALL-X")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """为评估预测动作块。"""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        generation_prompt_ids = batch.get(WALL_X_GENERATION_PROMPT_IDS)
        batch = self._pretokenized_inputs(batch)

        if self.config.prediction_mode == "diffusion":
            output = self.model(
                **batch,
                action_dim=self.config.max_action_dim,
                pred_horizon=self.config.chunk_size,
                mode="predict",
                predict_mode="diffusion",
            )
        elif self.config.prediction_mode == "fast":
            if not isinstance(generation_prompt_ids, Tensor):
                raise ValueError(
                    "WALL-X fast prediction requires generation-prompt tokens from its input processor."
                )
            output = self.model(
                **batch,
                generation_prompt_ids=generation_prompt_ids,
                action_dim=self.config.output_features[ACTION].shape[0],
                pred_horizon=self.config.chunk_size,
                mode="predict",
                predict_mode="fast",
            )
        else:
            raise NotImplementedError(f"Prediction mode {self.config.prediction_mode} not implemented")

        # 从输出字典中提取动作张量
        actions = output["predict_action"]

        # 将动作去掉填充，恢复到实际的动作维度
        action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """选择单个动作用于环境执行。"""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # 使用动作队列
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()
