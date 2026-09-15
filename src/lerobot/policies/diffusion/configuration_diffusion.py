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
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("diffusion")
@dataclass
class DiffusionConfig(PreTrainedConfig):
    """DiffusionPolicy 的配置类。

    默认值针对使用 PushT 进行训练而配置，提供本体感知和单摄像头观测。

    你最可能需要修改的参数是那些依赖于环境/传感器的参数，
    即 `input_features` 和 `output_features`。

    关于输入和输出的说明：
        - "observation.state" 是必需的输入键。
        - 满足以下任一条件即可：
            - 至少有一个以 "observation.image" 开头的键作为输入。
              并且/或者
            - 键 "observation.environment_state" 作为输入。
        - 如果有多个以 "observation.image" 开头的键，它们将被视为多个摄像头视图。
          目前我们仅支持所有图像具有相同形状。
        - "action" 是必需的输出键。

    Args:
        n_obs_steps: 传递给策略的环境步数观测数量（取当前步及之前的若干步）。
        horizon: 扩散模型动作预测的大小，详见 `DiffusionPolicy.select_action`。
        n_action_steps: 单次调用策略时在环境中运行的动作步数。
            详见 `DiffusionPolicy.select_action`。
        input_features: 定义策略输入数据 PolicyFeature 的字典。键表示输入数据名称，
            值为 PolicyFeature，由 FeatureType 和 shape 属性组成。
        output_features: 定义策略输出数据 PolicyFeature 的字典。键表示输出数据名称，
            值为 PolicyFeature，由 FeatureType 和 shape 属性组成。
        normalization_mapping: 从 FeatureType 的字符串值（例如 "STATE"、"VISUAL"）
            映射到对应 NormalizationMode（例如 NormalizationMode.MIN_MAX）的字典。
        vision_backbone: 用于编码图像的 torchvision resnet 骨干网络名称。
        resize_shape: 作为视觉骨干网络预处理步骤，将图像调整为 (H, W) 形状。
            若为 None，则不进行调整大小，使用原始图像分辨率。
        crop_ratio: (0, 1] 区间内的比率，用于从 resize_shape 推导裁剪尺寸
            （crop_h = int(resize_shape[0] * crop_ratio)，宽度同理）。
            设为 1.0 可禁用裁剪。仅在 resize_shape 不为 None 时生效。
        crop_shape: 将图像裁剪为 (H, W) 形状。当设置了 resize_shape 且 crop_ratio < 1.0 时，
            会自动计算该值。也可以直接设置，用于使用仅裁剪（不调整大小）的旧配置。
            若为 None 且无推导适用，则不进行裁剪。
        crop_is_random: 训练时裁剪是否随机（评估模式下始终为中心裁剪）。
        pretrained_backbone_weights: 用于初始化骨干网络的 torchvision 预训练权重。
            `None` 表示不使用预训练权重。
        use_group_norm: 是否在骨干网络中用组归一化替换批归一化。
            组大小设为约 16（精确地说，feature_dim // 16）。
        spatial_softmax_num_keypoints: SpatialSoftmax 的关键点数量。
        use_separate_rgb_encoder_per_camera: 是否为每个摄像头视图使用单独的 RGB 编码器。
        down_dims: 扩散建模 Unet 中每个时间下采样阶段的特征维度。
            你可以提供任意数量的维度，从而控制下采样的程度。
        kernel_size: 扩散建模 Unet 的卷积核大小。
        n_groups: Unet 卷积块中组归一化使用的组数。
        diffusion_step_embed_dim: Unet 通过一个小型非线性网络以扩散时间步为条件。
            这是该网络的输出维度，即嵌入维度。
        use_film_scale_modulation: Unet 条件化使用 FiLM（https://huggingface.co/papers/1709.07871）。
            默认使用偏置调制，该参数表示是否也使用缩放调制。
        gradient_checkpointing: 训练时是否对 Unet 残差块进行检查点保存。
            这会以反向传播时重新计算这些块为代价，减少激活内存。
        noise_scheduler_type: 使用的噪声调度器名称。支持的选项：["DDPM", "DDIM"]。
        num_train_timesteps: 前向扩散调度的扩散步数。
        beta_schedule: 扩散 beta 调度名称，参照 Hugging Face diffusers 的 DDPMScheduler。
        beta_start: 第一个前向扩散步的 beta 值。
        beta_end: 最后一个前向扩散步的 beta 值。
        prediction_type: 扩散建模 Unet 做出的预测类型。从 "epsilon" 或 "sample" 中选择。
            从潜变量建模的角度来看，两者结果等价，但 "epsilon" 在许多深度神经网络
            设置中表现更好。
        clip_sample: 推理时每个去噪步是否将样本裁剪到 [-`clip_sample_range`, +`clip_sample_range`]。
            警告：你需要确保动作空间已归一化以适应此范围。
        clip_sample_range: 如上所述的裁剪范围幅度。
        num_inference_steps: 推理时使用的反向扩散步数（步长均匀分布）。
            若未提供，默认为与 `num_train_timesteps` 相同。
        do_mask_loss_for_padding: 当存在复制填充的动作时是否对损失进行掩码。
            详见 `LeRobotDataset` 和 `load_previous_and_future_frames`。注意，此参数默认为 False，
            因为原始 Diffusion Policy 实现也是如此。
    """

    # 输入/输出结构。
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # 原始实现不对最后 7 帧采样，
    # 这样可以避免过度填充并提升训练效果。
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # 架构/建模。
    # 视觉骨干网络。
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True
    # Unet。
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    gradient_checkpointing: bool = False
    # 噪声调度器。
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # 推理
    num_inference_steps: int | None = None

    # 优化
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # 损失计算
    do_mask_loss_for_padding: bool = False

    # 训练预设
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """输入验证（非详尽）。"""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                # 当 crop_ratio == 1.0 时，在 resize+ratio 路径下显式禁用裁剪。
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # 检查 horizon 大小与 U-Net 下采样是否兼容。
        # U-Net 每阶段下采样 2 倍。
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # 检查所有输入图像是否具有相同形状。
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
