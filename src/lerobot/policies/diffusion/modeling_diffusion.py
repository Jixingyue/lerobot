#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
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
"""Diffusion Policy，出自论文 "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - 移除对 diffusers 中 DDPMScheduler 和学习率调度器的依赖。
"""

import math
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import _diffusers_available, require_package

if TYPE_CHECKING or _diffusers_available:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
else:
    DDIMScheduler = None
    DDPMScheduler = None

from ..pretrained import PreTrainedPolicy
from ..utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from .configuration_diffusion import DiffusionConfig


class DiffusionPolicy(PreTrainedPolicy):
    """
    Diffusion Policy，出自论文 "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    （论文：https://huggingface.co/papers/2303.04137，代码：https://github.com/real-stanford/diffusion_policy）。
    """

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig,
        **kwargs,
    ):
        """
        Args:
            config: 策略配置类实例，或为 None（此时使用配置类的默认实例化）。
            dataset_stats: 用于归一化的数据集统计量。如果这里没有传入，则期望在
                使用策略之前通过调用 `load_state_dict` 传入。
        """
        require_package("diffusers", extra="diffusion")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # 队列在策略推演（rollout）期间被填充，其中包含最近的 n 个观测和动作
        self._queues = None

        self.diffusion = DiffusionModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """清空观测和动作队列。应在 `env.reset()` 时调用"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """根据环境观测预测一个动作块。

        支持两种模式：
        - 在线模式（队列通过 select_action 填充）：从内部队列中堆叠观测。
        - 离线模式（队列为空，例如来自数据加载器的批次）：直接使用该批次。
        """
        queues_populated = any(len(q) > 0 for q in self._queues.values())
        if queues_populated:
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        else:
            batch = dict(batch)
            if self.config.image_features:
                for key in self.config.image_features:
                    if batch[key].ndim == 4:
                        batch[key] = batch[key].unsqueeze(1)
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        actions = self.diffusion.generate_actions(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """根据环境观测选择单个动作。

        该方法负责缓存观测历史以及由底层扩散模型生成的动作轨迹。其工作方式如下：
          - 缓存 `n_obs_steps` 步的观测（对于最初的几步，观测会被复制
            `n_obs_steps` 次以填满缓存）。
          - 扩散模型生成 `horizon` 步的动作。
          - 实际保留用于执行的动作有 `n_action_steps` 步，从当前步开始。
        示意如下：
            ----------------------------------------------------------------------------------------------
            （图例：o = n_obs_steps，h = horizon，a = n_action_steps）
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        注意，这意味着我们要求：`n_action_steps <= horizon - n_obs_steps + 1`。另外，
        "horizon" 可能不是描述该变量实际含义的最佳名称，因为这个周期实际上
        是从第一个观测开始测量的，而该观测（如果 `n_obs_steps` > 1）发生在过去。
        """
        # 注意：对于离线评估，批次中包含动作，因此我们需要将其弹出
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝，这样添加键不会修改原始批次
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # 注意：这一步必须在将图像堆叠到单个键之后进行。
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """让批次通过模型运行，并计算用于训练或验证的损失。"""
        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝，这样添加键不会修改原始批次
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.diffusion.compute_loss(batch)
        # 没有 output_dict，因此返回 None
        return loss, None


def _make_noise_scheduler(name: str, **kwargs: dict):
    """
    请求类型的噪声调度器实例的工厂函数。所有 kwargs 都会传递给调度器。
    """
    require_package("diffusers", extra="diffusion")

    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        # 构建观测编码器（取决于提供了哪些观测）。
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        if config.compile_model:
            # 编译 U-Net。对于扩散推理中常见的小批次重复循环，
            # 优先选择 "reduce-overhead"。
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    # ========= 推理 ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # 采样先验。
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # 预测模型输出。
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # 计算前一图像：x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """编码图像特征，并将它们与状态向量一起拼接。"""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        # 提取图像特征。
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # 在重排以将相机索引维度置于首位的同时，合并批次维和序列维。
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # 将批次维和序列维重新分离出来。相机索引维被吸收进
                # 特征维（相当于拼接各相机的特征）。
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                # 在传入共享编码器之前，合并批次维、序列维和"哪个相机"维。
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                # 将批次维和序列维重新分离出来。相机索引维被吸收进
                # 特征维（相当于拼接各相机的特征）。
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # 拼接特征，然后展平为 (B, global_cond_dim)。
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """
        该函数期望 `batch` 包含：
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                和/或
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # 编码图像特征，并将它们与状态向量一起拼接。
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # 运行采样
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        # 提取（从当前观测起）`n_action_steps` 步的动作。
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        该函数期望 `batch` 至少包含：
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                和/或
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # 输入校验。
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # 编码图像特征，并将它们与状态向量一起拼接。
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # 前向扩散。
        trajectory = batch[ACTION]
        # 采样要添加到轨迹的噪声。
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # 为批次中的每个样本采样一个随机加噪时间步。
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # 根据每个时间步的噪声幅度，向干净轨迹添加噪声。
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # 运行去噪网络（它可能对轨迹去噪，也可能尝试预测噪声）。
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # 计算损失。
        # 目标要么是原始轨迹，要么是噪声。
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # 在动作用副本填充的位置（数据集轨迹的边缘）掩蔽损失。
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            mask = in_episode_bound.unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            return (loss * mask).sum() / num_valid.clamp_min(1)

        return loss.mean()


class SpatialSoftmax(nn.Module):
    """
    Finn 等人在 "Deep Spatial Autoencoders for Visuomotor Learning" 中描述的
    Spatial Soft Argmax 操作（https://huggingface.co/papers/1509.06113）。
    是 robomimic 实现的最小化移植。

    从高层次看，它接收 2D 特征图（来自卷积网络/ViT），并返回每个通道
    激活值的"质心"，即供策略关注的图像空间中的关键点。

    示例：取尺寸为 (512x10x12) 的特征图。我们生成一个归一化坐标网格 (10x12x2)：
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    这通过对激活值 (512x120) 应用逐通道 softmax，并与坐标 (120x2) 做点积
    来实现，从而得到最大激活的期望点 (512x2)。

    上面的示例产生 512 个关键点（对应 512 个输入通道）。我们可以可选地
    提供 num_kp != None 来控制关键点数量。这通过先应用一个可学习的
    线性映射 (in_channels, H, W) -> (num_kp, H, W) 来实现。
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) 输入特征图形状。
            num_kp (int): 输出中的关键点数量。如果为 None，输出将与输入具有相同数量的通道。
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # 我们也可以直接使用 torch.linspace，但它的行为似乎与 numpy 略有不同，
        # 并会导致预训练模型的 pc_success 出现轻微下降。
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # 注册为 buffer，使其被移动到正确的设备上。
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) 输入特征图。
        Returns:
            (B, K, 2) 关键点的图像空间坐标。
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W]，其中 K 是关键点数量
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax 归一化
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2]，得到 x 和 y 维度上的空间坐标均值
        expected_xy = attention @ self.pos_grid
        # 重塑为 [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """将 RGB 图像编码为 1D 特征向量。

    包含先对图像进行归一化和裁剪的能力。
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        # 设置可选的预处理。
        if config.resize_shape is not None:
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.resize = None

        crop_shape = config.crop_shape
        if crop_shape is not None:
            self.do_crop = True
            # 评估时始终使用中心裁剪
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # 设置主干网络。
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # 注意：这里假设 layer4 特征图是 children()[-3]
        # TODO(alexander-soare): 使用更安全的替代方案。
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # 设置池化和最终层。
        # 使用一次试运行来获取特征图形状。
        # 虚拟形状与运行时预处理顺序一致：缩放 -> 裁剪。

        # 注意：配置类中有一个检查，确保所有图像具有相同的形状。
        images_shape = next(iter(config.image_features.values())).shape
        if config.crop_shape is not None:
            dummy_shape_h_w = config.crop_shape
        elif config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) 图像张量，像素值在 [0, 1] 范围内。
        Returns:
            (B, D) 图像特征。
        """
        # 预处理：如果配置了则缩放，如果配置了则裁剪。

        if self.resize is not None:
            x = self.resize(x)
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # 评估时始终使用中心裁剪。
                x = self.center_crop(x)
        # 提取主干特征。
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # 带非线性激活的最终线性层。
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: 需要替换其子模块的模块
        predicate: 以一个模块为参数，如果该模块需要被替换则必须返回 True。
        func: 以一个模块为参数，返回用于替换它的新模块。
    Returns:
        子模块已被替换的根模块。
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # 验证所有 BN 都已被替换
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """与 Attention is All You Need 中相同的 1D 正弦位置嵌入。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """带 FiLM 调制以实现条件化的 1D 卷积 UNet。

    注意：与原始扩散策略代码相比，这里移除了局部条件化。
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # 扩散时间步的编码器。
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # FiLM 条件化维度。
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # Unet 编码器中每个下采样块的输入通道/输出通道。对于解码器，
        # 我们只需将其反转。
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet 编码器。
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # 只要不是最后一个块就进行下采样。
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # 自编码器中间部分的处理。
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet 解码器。
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2，因为它还接收编码器的跳跃连接
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # 只要不是最后一个块就进行上采样。
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) 输入 Unet 的张量。
            timestep: (B,) 张量，值为 (timestep_we_are_denoising_from - 1)。
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) 扩散模型预测。
        """
        # 对于 1D 卷积，我们需要将特征维度放在第一位。
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # 如果存在全局条件特征，将其拼接到时间步嵌入上。
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        use_gc = self.config.gradient_checkpointing and self.training

        # 运行编码器，同时跟踪要传递给解码器的跳跃特征。
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            if use_gc:
                x = checkpoint(resnet, x, global_feature, use_reentrant=False)
                x = checkpoint(resnet2, x, global_feature, use_reentrant=False)
            else:
                x = resnet(x, global_feature)
                x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            if use_gc:
                x = checkpoint(mid_module, x, global_feature, use_reentrant=False)
            else:
                x = mid_module(x, global_feature)

        # 使用来自编码器的跳跃特征运行解码器。
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            if use_gc:
                x = checkpoint(resnet, x, global_feature, use_reentrant=False)
                x = checkpoint(resnet2, x, global_feature, use_reentrant=False)
            else:
                x = resnet(x, global_feature)
                x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """带 FiLM 调制以实现条件化的 ResNet 风格 1D 卷积块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # 设为 True 则同时用 FiLM 做缩放调制和偏置调制（默认为 False，
        # 表示 FiLM 只调制偏置）。
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM 调制（https://huggingface.co/papers/1709.07871）输出逐通道的偏置以及（可能的）缩放。
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # 用于使维度与残差匹配的最终卷积（如有必要）。
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # 获取条件嵌入。通过 unsqueeze 使其可广播到 `out`，得到 (B, out_channels, 1)。
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # 将嵌入视为缩放和偏置的列表。
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # 将嵌入视为偏置。
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out
