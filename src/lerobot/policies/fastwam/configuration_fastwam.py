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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
    PreTrainedConfig,
)
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_STATE

WAN22_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN22_DIFFUSERS_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FASTWAM_BASE_MODEL_ID = "lerobot/fastwam_base"
WAN_T5_TOKENIZER_ID = "google/umt5-xxl"


_FASTWAM_VIDEO_BASE_COMPAT_KEYS = (
    "patch_size",
    "in_dim",
    "hidden_dim",
    "ffn_dim",
    "freq_dim",
    "text_dim",
    "out_dim",
    "num_heads",
    "attn_head_dim",
    "num_layers",
)

_FASTWAM_ACTION_BASE_COMPAT_KEYS = (
    "hidden_dim",
    "ffn_dim",
    "num_heads",
    "attn_head_dim",
    "num_layers",
    "text_dim",
    "freq_dim",
)


def default_video_dit_config(action_dim: int) -> dict[str, Any]:
    return {
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1.0e-6,
        "seperated_timestep": True,
        "use_gradient_checkpointing": False,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": action_dim,
        "action_group_causal_mask_mode": "group_diagonal",
        "fp32_attention": True,
    }


def default_action_dit_config(action_dim: int) -> dict[str, Any]:
    return {
        "action_dim": action_dim,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1.0e-6,
        "use_gradient_checkpointing": False,
        "fp32_attention": True,
    }


def _coerce_enum(enum_cls: type, value: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (TypeError, ValueError) as exc:
        member = getattr(enum_cls, str(value), None)
        if member is None:
            raise ValueError(f"Cannot coerce {value!r} into {enum_cls.__name__}.") from exc
        return member


def _coerce_policy_features(features: dict[str, Any] | None) -> dict[str, PolicyFeature] | None:
    if features is None:
        return None
    coerced = {}
    for name, feature in features.items():
        if isinstance(feature, PolicyFeature):
            coerced[name] = feature
            continue
        coerced[name] = PolicyFeature(
            type=_coerce_enum(FeatureType, feature["type"]),
            shape=tuple(feature["shape"]),
        )
    return coerced


def _is_local_model_id(value: str) -> bool:
    path = Path(value).expanduser()
    return path.is_absolute() or value.startswith(("./", "../", "~")) or path.exists()


def _validate_wan_model_id(value: str, field_name: str) -> str:
    if value == WAN22_MODEL_ID or _is_local_model_id(value):
        return value
    raise ValueError(f"`{field_name}` must be `{WAN22_MODEL_ID}` or an explicit local path, got `{value}`.")


def is_fastwam_base_compatible_config(config: FastWAMConfig) -> bool:
    """返回 `fastwam_base` 的部分权重是否可以初始化此配置。"""

    default_video_config = default_video_dit_config(config.action_dim)
    default_action_config = default_action_dit_config(config.action_dim)
    return all(
        config.video_dit_config.get(key) == default_video_config.get(key)
        for key in _FASTWAM_VIDEO_BASE_COMPAT_KEYS
    ) and all(
        config.action_dit_config.get(key) == default_action_config.get(key)
        for key in _FASTWAM_ACTION_BASE_COMPAT_KEYS
    )


@PreTrainedConfig.register_subclass("fastwam")
@dataclass
class FastWAMConfig(PreTrainedConfig):
    """FastWAM LeRobot 策略的配置。

    Args:
        action_dim (int): 每个时间步的标量动作通道数。
        proprio_dim (int | None): 用作额外文本上下文 token 的本体感知通道数。
            `None` 表示禁用本体感知条件输入。
        action_horizon (int): 一次策略调用预测的动作数量。
        num_video_frames (int): 原始视频采样窗口（以数据集帧数计）。模型在按
            `action_video_freq_ratio` 降采样后，实际操作 `model_video_frames` 帧。
        action_video_freq_ratio (int): 动作以视频帧率的该倍数进行采样。每隔
            `action_video_freq_ratio` 个原始帧取一个视频帧，因此模型看到
            `(num_video_frames - 1) // ratio + 1` 帧，覆盖与 `action_horizon`
            个动作相同的时间窗口（每个视频帧对应 ratio 个动作）。
        image_size (tuple[int, int]): 拼接后的图像尺寸，格式为 `(height, width)`。
        context_len (int): 文本嵌入的最大 token 长度。
        video_dit_config (dict[str, Any] | None): Wan 视频专家的配置。
        action_dit_config (dict[str, Any] | None): 动作专家的配置。
        use_gradient_checkpointing (bool): 在两个 DiT 专家中启用激活检查点
            （以计算换内存；会传递到 DiT 配置中）。
        compile_action_infer (bool): 使用 ``torch.compile`` 以 reduce-overhead 模式
            编译带缓存的动作去噪路径。首次推理调用会产生编译预热开销；之后相同
            形状的调用会复用计算图。
        freeze_video_expert (bool): 冻结约 5B 的 Wan 视频专家（`model.video_expert`），
            只训练动作专家 + 本体感知编码器。可大幅减少 AdamW 优化器的内存占用；
            视频专家保留其预训练权重。（如果启用，还应设置 `loss.lambda_video=0`
            以跳过此时已无梯度的视频损失计算。）
    """

    n_obs_steps: int = 1
    action_dim: int = 7
    proprio_dim: int | None = 8
    action_horizon: int = 32
    n_action_steps: int = 32
    num_video_frames: int = 33
    action_video_freq_ratio: int = 4
    image_size: tuple[int, int] = (224, 448)
    context_len: int = 128
    model_id: str = WAN22_MODEL_ID
    tokenizer_model_id: str = WAN_T5_TOKENIZER_ID
    text_encoder_model_id: str = WAN22_DIFFUSERS_MODEL_ID
    base_model_id: str | None = FASTWAM_BASE_MODEL_ID
    tokenizer_max_len: int = 128
    load_text_encoder: bool = True
    mot_checkpoint_mixed_attn: bool = False
    torch_dtype: str = "bfloat16"
    prompt_template: str = (
        "A video recorded from a robot's point of view executing the following instruction: {task}"
    )
    num_inference_steps: int = 10
    inference_seed: int | None = 42
    rand_device: str = "cpu"
    text_cfg_scale: float = 1.0
    negative_prompt: str = ""
    sigma_shift: float | None = None
    tiled: bool = False
    fp32_attention: bool = True
    compile_action_infer: bool = False
    use_gradient_checkpointing: bool = False
    freeze_video_expert: bool = False
    toggle_action_dimensions: list[int] = field(default_factory=list)
    video_scheduler: dict[str, float | int] = field(
        default_factory=lambda: {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}
    )
    action_scheduler: dict[str, float | int] = field(
        default_factory=lambda: {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}
    )
    loss: dict[str, float] = field(default_factory=lambda: {"lambda_video": 1.0, "lambda_action": 1.0})
    video_dit_config: dict[str, Any] | None = None
    action_dit_config: dict[str, Any] | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    input_features: dict[str, PolicyFeature] | None = None
    output_features: dict[str, PolicyFeature] | None = None
    optimizer_lr: float = 1.0e-4
    optimizer_weight_decay: float = 1.0e-2

    def __post_init__(self) -> None:
        super().__post_init__()
        self.image_size = tuple(self.image_size)
        self.model_id = _validate_wan_model_id(self.model_id, "model_id")
        self.input_features = _coerce_policy_features(self.input_features)
        self.output_features = _coerce_policy_features(self.output_features)
        self.toggle_action_dimensions = [int(dim) for dim in self.toggle_action_dimensions]
        self.video_dit_config = self.video_dit_config or default_video_dit_config(self.action_dim)
        self.action_dit_config = self.action_dit_config or default_action_dit_config(self.action_dim)
        self.video_dit_config["fp32_attention"] = bool(self.fp32_attention)
        self.action_dit_config["fp32_attention"] = bool(self.fp32_attention)
        self.video_dit_config["use_gradient_checkpointing"] = bool(self.use_gradient_checkpointing)
        self.action_dit_config["use_gradient_checkpointing"] = bool(self.use_gradient_checkpointing)
        if self.input_features is None:
            height, width = self.image_size
            self.input_features = {
                "observation.images.image": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, height, width),
                )
            }
            if self.proprio_dim is not None:
                self.input_features[OBS_STATE] = PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(self.proprio_dim,),
                )
        if self.output_features is None:
            self.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))}
        self.validate_features()
        if self.pretrained_path or self.use_peft or not self.base_model_id:
            return
        if not is_fastwam_base_compatible_config(self):
            return
        self.pretrained_path = Path(self.base_model_id)
        self._auto_pretrained_path = True

    def _save_pretrained(self, save_directory: Path) -> None:
        if not getattr(self, "_auto_pretrained_path", False):
            super()._save_pretrained(save_directory)
            return

        pretrained_path = self.pretrained_path
        self.pretrained_path = None
        try:
            super()._save_pretrained(save_directory)
        finally:
            self.pretrained_path = pretrained_path

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def set_dataset_feature_metadata(self, dataset_features: dict[str, Any]) -> None:
        """根据数据集真实的相机键重建视觉输入特征。

        FastWAM 的 `__post_init__` 会安装一个合成的单图像默认值
        （`observation.images.image`，宽度为完整的 `image_size`）。对于具有一个或多个
        独立命名相机的数据集（例如 `observation.images.top`、`observation.images.wrist`），
        此钩子——在数据集元数据已知后由 `make_policy` 调用——会用实际的相机键替换
        该默认值，每个相机按策略原生的单相机分辨率声明
        （`image_size[0]` x `image_size[1] // num_cameras`）。`make_fastwam_pre_post_processors`
        中配套的 resize 步骤会把原始帧 resize 到匹配的尺寸，因此支持异构的源分辨率
        （例如 480x640）。
        """
        image_keys = sorted(
            key
            for key, feature in dataset_features.items()
            if key.startswith("observation.images.") and feature.get("dtype") in ("video", "image")
        )
        if not image_keys:
            return
        height, total_width = self.image_size
        per_cam_width = total_width // len(image_keys)
        new_inputs: dict[str, PolicyFeature] = {
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, per_cam_width))
            for key in image_keys
        }
        if self.proprio_dim is not None and OBS_STATE in dataset_features:
            new_inputs[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.proprio_dim,))
        self.input_features = new_inputs
        self.validate_features()

    def validate_features(self) -> None:
        if self.action_dim <= 0:
            raise ValueError(f"`action_dim` must be positive, got {self.action_dim}.")
        if self.action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {self.action_horizon}.")
        if self.n_action_steps > self.action_horizon:
            raise ValueError("`n_action_steps` cannot exceed `action_horizon`.")
        if self.action_video_freq_ratio <= 0:
            raise ValueError(
                f"`action_video_freq_ratio` must be positive, got {self.action_video_freq_ratio}."
            )
        # 视频帧按 action_video_freq_ratio 降采样；得到的模型帧数必须满足
        # T % 4 == 1 才能进行 VAE 时间维 tokenization（与原始 FastWAM 数据集的
        # 断言保持一致）。
        if (self.num_video_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                f"`num_video_frames - 1` ({self.num_video_frames - 1}) must be divisible by "
                f"`action_video_freq_ratio` ({self.action_video_freq_ratio})."
            )
        if ((self.num_video_frames - 1) // self.action_video_freq_ratio) % 4 != 0:
            raise ValueError(
                f"Subsampled video transitions ({(self.num_video_frames - 1) // self.action_video_freq_ratio}) "
                "must be divisible by 4 for VAE tokenization (i.e. model_video_frames % 4 == 1)."
            )
        if self.action_horizon % ((self.num_video_frames - 1) // self.action_video_freq_ratio) != 0:
            raise ValueError(
                f"`action_horizon` ({self.action_horizon}) must be divisible by the number of "
                f"video transitions ({(self.num_video_frames - 1) // self.action_video_freq_ratio})."
            )
        if not self.image_features:
            raise ValueError("FastWAM requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("FastWAM requires `action` in output_features.")
        action_shape = tuple(self.action_feature.shape)
        if action_shape != (self.action_dim,):
            raise ValueError(
                f"FastWAM action feature shape must be ({self.action_dim},), got {action_shape}."
            )
        if self.proprio_dim is not None:
            state_feature = self.robot_state_feature
            if state_feature is None:
                raise ValueError("FastWAM requires `observation.state` when `proprio_dim` is set.")
            state_shape = tuple(state_feature.shape)
            if state_shape != (self.proprio_dim,):
                raise ValueError(
                    f"FastWAM state feature shape must be ({self.proprio_dim},), got {state_shape}."
                )
        height, width = self.image_size
        image_width_sum = 0
        for name, feature in self.image_features.items():
            shape = tuple(feature.shape)
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"FastWAM image feature `{name}` must have shape (3, H, W), got {shape}.")
            if shape[1] != height:
                raise ValueError(f"FastWAM image feature `{name}` height must be {height}, got {shape[1]}.")
            image_width_sum += shape[2]
        if image_width_sum != width:
            raise ValueError(f"FastWAM image feature widths must sum to {width}, got {image_width_sum}.")

    @property
    def model_video_frames(self) -> int:
        """模型实际操作的视频帧数，即原始 `num_video_frames` 窗口按
        `action_video_freq_ratio` 降采样后的结果（例如 33 -> 9）。"""
        return (self.num_video_frames - 1) // self.action_video_freq_ratio + 1

    @property
    def observation_delta_indices(self) -> list[int]:
        # 加载模型受监督的视频帧：按 action_video_freq_ratio 降采样的未来窗口
        # （例如 [0, 4, 8, ..., 32] -> 9 帧）。因此相邻视频帧之间间隔
        # `action_video_freq_ratio` 个动作，而动作按完整速率加载
        # （`action_delta_indices` = range(action_horizon)）。如果返回 None 则只加载
        # 当前帧，会使视频目标变成静态重复（退化的监督信号）。
        return list(range(0, self.num_video_frames, self.action_video_freq_ratio))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
