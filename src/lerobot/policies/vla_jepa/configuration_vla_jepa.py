# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)


@PreTrainedConfig.register_subclass("vla_jepa")
@dataclass
class VLAJEPAConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    chunk_size: int = 7
    n_action_steps: int = 7

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    qwen_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    jepa_encoder_name: str = "facebook/vjepa2-vitl-fpc64-256"
    freeze_qwen: bool = False
    enable_world_model: bool = True
    # 启用跨本体（cross-embodiment）迁移：当在动作或状态维度不同的机器人上微调预训练模型时，
    # 输入/输出投影层必须从头重新初始化，而网络的其余部分保留其预训练权重。
    # 这里列出允许出现形状不匹配的键前缀；其他任何不匹配都会报错。
    # 例如 ["model.action_model.action_encoder", "model.action_model.state_encoder"]
    reinit_modules: list[str] | None = None

    tokenizer_padding_side: str = "left"
    prompt_template: str = "Your task is {instruction}. Infer the temporal dynamics from frames {actions} and produce the corresponding policy actions {e_actions}."
    special_action_token: str = "<|action_{}|>"
    embodied_action_token: str = "<|embodied_action|>"

    action_dim: int = 7
    state_dim: int = 8

    # 相对动作：在预处理时将绝对动作转换为相对动作（action -= state），
    # 并在后处理时逆转该转换。需要 `state_dim`（OBS_STATE）。
    use_relative_actions: bool = False
    # 保持绝对（不转换为相对）的关节名称。空列表表示所有维度均为相对。
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # 在运行时由 make_policy 根据数据集元数据填充（用于构建排除掩码）。
    action_feature_names: list[str] | None = None

    num_action_tokens_per_timestep: int = 8
    num_embodied_action_tokens_per_instruction: int = 32
    num_inference_timesteps: int = 4

    action_hidden_size: int = 1024
    action_model_type: str = "DiT-B"
    action_num_layers: int = 16
    action_num_heads: int | None = None
    action_attention_head_dim: int | None = None
    action_dropout: float = 0.2
    action_num_timestep_buckets: int = 1000
    action_noise_beta_alpha: float = 1.5
    action_noise_beta_beta: float = 1.0
    action_noise_s: float = 0.999
    # 动作头所学位置嵌入表的大小。保持为 1024 以与已发布的检查点一致；
    # 只有当 `chunk_size` 接近该值时才应调大它。
    action_max_seq_len: int = 1024
    # 未使用。予以保留是因为已发布的检查点会序列化该字段，而 draccus 会拒绝
    # dataclass 中不再声明的 config.json 键。
    num_target_vision_tokens: int = 32

    # 每个样本加载的视频帧总数
    num_video_frames: int = 8
    predictor_depth: int = 12
    predictor_num_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    predictor_dropout: float = 0.0
    world_model_loss_weight: float = 0.1
    # JEPA 编码器的时间 tubelet 大小（例如 vjepa2-vitl-fpc64-256 为 2）。启用世界模型时，
    # 以编码器自身的 `config.tubelet_size` 为准，该值仅用于下方的 `num_video_frames`
    # 健全性检查。
    jepa_tubelet_size: int = 2
    # 世界模型预测器所针对的相机视角（多余的视角会被裁剪，缺失的视角用第一个视角填充）。
    # 该值会固化到检查点形状中。`None` 时回退到 `jepa_tubelet_size`，
    # 这也是已发布检查点所编码的值。
    world_model_num_views: int | None = None
    repeated_diffusion_steps: int = 8  # 每个批次项独立的噪声采样次数（CogACT 风格）
    # 若为 True，则以因果方式编码世界模型上下文，而不是从存在信息泄漏的共享前向过程中切片（#4153）。
    causal_world_model_context: bool = False

    resize_images_to: tuple[int, int] | None = None
    # 来自 starVLA LIBERO 评估循环的夹爪后处理。默认关闭：它只对 LIBERO 的动作约定
    # 正确，并且当夹爪的物理范围不近似为 [0, 1] 时会将夹爪固定为常量。原因请参见文档。
    binarize_gripper_action: bool = False
    pre_snap_gripper_action: bool = False
    clip_normalized_actions: bool = True
    # 夹爪在动作向量中的索引。建议保留默认值，改为设置 `gripper_joint_names`，
    # 由其根据数据集元数据解析索引。
    gripper_dim: int = 6
    gripper_threshold: float = 0.5
    # 标识夹爪的动作维度名称。当这些名称与 `action_feature_names` 匹配时，
    # 解析出的索引优先于 `gripper_dim`。
    gripper_joint_names: list[str] = field(default_factory=lambda: ["gripper"])
    torch_dtype: str = "bfloat16"

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.freeze_qwen and self.enable_world_model:
            # 冻结 qwen 主干后没有梯度流入，世界模型训练便失去意义
            self.enable_world_model = False
        if self.freeze_qwen:
            logger.warning(
                "freeze_qwen=True: action-head conditioning is read from %s positions at the last "
                "decoder layer. These learned readouts stay fixed from the source checkpoint and "
                "cannot adapt to a new embodiment while the Qwen backbone is frozen, so conditioning "
                "quality may degrade under domain shift.",
                self.embodied_action_token,
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError("`n_action_steps` must be <= `chunk_size`.")
        if self.num_video_frames < 2 * self.jepa_tubelet_size:
            raise ValueError(
                f"`video_horizon` ({self.num_video_frames}) must be >= 2 * `jepa_tubelet_size` "
                f"({self.jepa_tubelet_size}) to have at least one context and one GT temporal position."
            )

    @property
    def num_world_model_views(self) -> int:
        """世界模型预测器所针对的相机视角（参见 `world_model_num_views`）。"""
        return self.world_model_num_views or self.jepa_tubelet_size

    @property
    def resolved_gripper_dim(self) -> int:
        """夹爪索引，尽可能从 `action_feature_names` 解析得到。

        当数据集元数据不可用时（例如重建已保存的处理器流水线时没有附带数据集），
        回退到原始的 `gripper_dim`。
        """
        if not self.action_feature_names or not self.gripper_joint_names:
            return self.gripper_dim
        wanted = [name.lower() for name in self.gripper_joint_names if name]
        for index, name in enumerate(self.action_feature_names):
            lowered = str(name).lower()
            if any(token == lowered or token in lowered for token in wanted):
                return index
        return self.gripper_dim

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("VLAJEPA requires at least one visual input feature.")
        if self.action_feature is None:
            raise ValueError("VLAJEPA requires an action output feature.")
        self.action_dim = self.action_feature.shape[0]
        if self.robot_state_feature is not None:
            self.state_dim = self.robot_state_feature.shape[0]
        # 当索引超出范围时，夹爪相关步骤会静默地不做任何操作，这看起来像是“二值化已执行”，
        # 但实际什么都没发生。因此在构建时直接显式报错。
        if self.pre_snap_gripper_action or self.binarize_gripper_action:
            gripper_dim = self.resolved_gripper_dim
            if gripper_dim >= self.action_dim:
                raise ValueError(
                    f"`gripper_dim` ({gripper_dim}) is out of range for a {self.action_dim}-dim "
                    f"action. Set `gripper_dim`/`gripper_joint_names` to the real gripper index, "
                    f"or disable `pre_snap_gripper_action`/`binarize_gripper_action`."
                )

    def set_dataset_feature_metadata(self, dataset_features: dict[str, Any]) -> None:
        """根据实际使用的数据集推导动作/状态维度以及各维度名称。

        `input_features` 保留的是*预训练*的特征键（rename_map 需要它们），否则
        `validate_features` 会从预训练配置中读到过时的维度。该方法由
        `make_policy` 在构建模型和处理器流水线之前调用。同时会把
        `observation.state` 写入 `input_features`，以便对其进行归一化。
        """
        if OBS_STATE in dataset_features:
            shape = tuple(dataset_features[OBS_STATE]["shape"])
            self.state_dim = shape[0]
            self.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=shape)
        if ACTION in dataset_features:
            self.action_dim = dataset_features[ACTION]["shape"][0]
            names = dataset_features[ACTION].get("names")
            if names:
                self.action_feature_names = list(names)

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        # 只有世界模型会消费索引 0 之后的帧，因此若不启用世界模型，请求完整窗口
        # 会导致每个样本的每个相机都解码 `num_video_frames` 帧，随后又将其丢弃。
        if not self.enable_world_model:
            return [0]
        # 当动作块能容纳于视频窗口内时，与原始仓库的 `range(video_horizon)` 一致。
        # 对于更长的动作块，则让帧在整个块上跨步分布，而不是聚集在开头，
        # 以使世界模型能够看到整个时域上的动态。
        if self.num_video_frames >= self.chunk_size:
            return list(range(self.num_video_frames))
        stride = (self.chunk_size - 1) // (self.num_video_frames - 1)
        return [i * stride for i in range(self.num_video_frames)]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
