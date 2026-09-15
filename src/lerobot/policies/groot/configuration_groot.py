#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, DiffuserSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from .utils import read_json

logger = logging.getLogger(__name__)

GROOT_N1_7 = "n1.7"
# 旧版 GR00T N1.5 标识符。N1.5 不是受支持的 model_version（它被有意排除在
# _GROOT_MODEL_VERSION_ALIASES 之外，因此 normalize_groot_model_version
# 仍会拒绝它）。保留它只是为了让 infer_groot_model_version 能够识别
# N1.5 的基础路径/检查点，从而让 N1.7 的配置/加载器拒绝这种不匹配。
GROOT_N1_5 = "n1.5"
# 当检测到 N1.5 检查点、配置或处理器流水线时，附加到每个抛出错误上的
# 规范指引信息。请保持该消息与 docs/source/groot.mdx 同步。
GROOT_N1_5_REMOVAL_GUIDANCE = (
    "GR00T N1.5 support was removed from LeRobot. "
    "To keep using an N1.5 checkpoint, pin the last release that supports it: "
    "`pip install 'lerobot==0.5.1'`. To use the current release, migrate to GR00T N1.7 "
    "(model_version='n1.7', base model nvidia/GR00T-N1.7-3B)."
)
GROOT_N1_7_BASE_MODEL = "nvidia/GR00T-N1.7-3B"
GROOT_N1_7_BACKBONE_MODEL = "nvidia/Cosmos-Reason2-2B"
# GR00T N1.7 的默认训练分辨率。当 processor_config 缺少尺寸信息时作为回退值。
# 通过强制缩放来防止全分辨率下的 patch 化不匹配。与 groot_n1_7.py 中的
# GR00T_N1_7_DEFAULTS 保持一致。
N1_7_DEFAULT_IMAGE_TARGET_SIZE = (256, 256)
N1_7_DEFAULT_IMAGE_CROP_SIZE = (230, 230)
GROOT_ACTION_DECODE_TRANSFORM_LIBERO = "libero"
# 哨兵值，表示"用户未选择动作解码变换"：__post_init__ 会将其解析为
# 对应具身（embodiment）的默认值（'libero_sim' 对应 'libero'，其他为 None）。
# 它与显式的 'none'（解析为 None）不同，这样显式禁用的选择才能在
# draccus 的保存/加载往返中保留下来。
GROOT_ACTION_DECODE_TRANSFORM_AUTO = "auto"

_GROOT_MODEL_VERSION_ALIASES = {
    "n1.7": GROOT_N1_7,
    "n1_7": GROOT_N1_7,
    "n1d7": GROOT_N1_7,
    "n17": GROOT_N1_7,
    "1.7": GROOT_N1_7,
}

# 旧版 N1.5 的各种写法，保留它们只是为了检测并以
# GROOT_N1_5_REMOVAL_GUIDANCE 拒绝（见上方的 GROOT_N1_5）。
# 切勿将这些映射到受支持的版本。
_GROOT_N1_5_VERSION_ALIASES = {"n1.5", "n1_5", "n1d5", "n15", "1.5"}

_GROOT_ACTION_DECODE_TRANSFORM_ALIASES = {
    GROOT_ACTION_DECODE_TRANSFORM_AUTO: GROOT_ACTION_DECODE_TRANSFORM_AUTO,
    "none": None,
    "": None,
    GROOT_ACTION_DECODE_TRANSFORM_LIBERO: GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
}


def normalize_groot_model_version(model_version: str) -> str:
    normalized = _GROOT_MODEL_VERSION_ALIASES.get(model_version.lower())
    if normalized is None:
        supported = GROOT_N1_7
        message = f"Unsupported GR00T model_version '{model_version}'. Supported versions: {supported}."
        if model_version.lower() in _GROOT_N1_5_VERSION_ALIASES:
            message = f"{message} {GROOT_N1_5_REMOVAL_GUIDANCE}"
        raise ValueError(message)
    return normalized


def normalize_groot_action_decode_transform(transform: str | None) -> str | None:
    if transform is None:
        return None
    normalized = _GROOT_ACTION_DECODE_TRANSFORM_ALIASES.get(transform.lower())
    if normalized is None and transform.lower() not in _GROOT_ACTION_DECODE_TRANSFORM_ALIASES:
        supported = ", ".join(
            sorted(key for key, value in _GROOT_ACTION_DECODE_TRANSFORM_ALIASES.items() if value is not None)
        )
        raise ValueError(
            f"Unsupported GR00T N1.7 action decode transform '{transform}'. "
            f"Supported transforms: none, {supported}."
        )
    return normalized


def infer_groot_model_version(model_path: str | None) -> str | None:
    if not model_path:
        return None
    model_path_lower = model_path.lower()
    if "gr00t-n1.7" in model_path_lower or "gr00t_n1.7" in model_path_lower:
        return GROOT_N1_7
    # 检测旧版 N1.5 路径，以便 N1.7 的配置/加载器拒绝这种不匹配。
    # N1.5 不受支持，但这里仍必须识别它，以便明确报错，
    # 而不是悄悄地把 N1.5 检查点当作 N1.7 处理。
    if "gr00t-n1.5" in model_path_lower or "gr00t_n1.5" in model_path_lower:
        return GROOT_N1_5
    config_version = _infer_groot_model_version_from_local_config(model_path)
    if config_version is not None:
        return config_version
    return None


def is_raw_groot_n1_7_checkpoint(model_path: str | Path | None) -> bool:
    if model_path is None:
        return False

    path = Path(model_path).expanduser()
    if path.is_dir():
        config_path = path / "config.json"
    elif path.name == "config.json":
        config_path = path
    else:
        return False

    config = read_json(config_path)
    return "type" not in config and _infer_groot_model_version_from_config(config) == GROOT_N1_7


def infer_groot_n1_7_embodiment_tag(model_path: str | Path | None) -> str | None:
    if model_path is None:
        return None

    processor_config_path = Path(model_path).expanduser() / "processor_config.json"
    processor_config = read_json(processor_config_path)

    modality_configs = processor_config.get("processor_kwargs", {}).get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None
    if "libero_sim" in modality_configs:
        return "libero_sim"
    if len(modality_configs) == 1:
        return next(iter(modality_configs))
    return None


def infer_groot_n1_7_action_horizon(
    model_path: str | Path | None, embodiment_tag: str | None = None
) -> int | None:
    if model_path is None:
        return None

    processor_config_path = Path(model_path).expanduser() / "processor_config.json"
    processor_config = read_json(processor_config_path)

    processor_kwargs = processor_config.get("processor_kwargs", {})
    if not isinstance(processor_kwargs, dict):
        return None
    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None

    if embodiment_tag is None:
        embodiment_tag = infer_groot_n1_7_embodiment_tag(model_path)
    if embodiment_tag is None:
        return None

    embodiment_config = modality_configs.get(embodiment_tag, {})
    if not isinstance(embodiment_config, dict):
        return None
    action_config = embodiment_config.get("action", {})
    if not isinstance(action_config, dict):
        return None
    delta_indices = action_config.get("delta_indices", [])
    if not isinstance(delta_indices, list):
        return None
    return len(delta_indices) or None


def infer_groot_n1_7_action_execution_horizon(
    model_path: str | Path | None, embodiment_tag: str | None = None
) -> int | None:
    action_horizon = infer_groot_n1_7_action_horizon(model_path, embodiment_tag)
    if action_horizon is None:
        return None

    if embodiment_tag is None:
        embodiment_tag = infer_groot_n1_7_embodiment_tag(model_path)
    if embodiment_tag == "libero_sim":
        # NVIDIA 的 N1.7 LIBERO rollout 封装会在解码出的 16 个动作执行 8 个后
        # 重新规划。保持该执行节奏可以避免使用过时的开环动作块。
        return min(action_horizon, 8)
    return action_horizon


def _infer_groot_model_version_from_local_config(model_path: str) -> str | None:
    path = Path(model_path).expanduser()
    if path.is_dir():
        config_path = path / "config.json"
    elif path.name == "config.json":
        config_path = path
    else:
        return None

    return _infer_groot_model_version_from_config(read_json(config_path))


def _infer_groot_model_version_from_config(config: dict) -> str | None:
    model_version = config.get("model_version")
    if isinstance(model_version, str):
        if model_version.lower() in _GROOT_N1_5_VERSION_ALIASES:
            return GROOT_N1_5
        try:
            return normalize_groot_model_version(model_version)
        except ValueError:
            return None

    candidates = [config.get("model_type"), *(config.get("architectures") or [])]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        normalized = candidate.lower().replace("-", "_")
        if normalized in {"gr00tn1d7", "gr00t_n1d7", "gr00t_n1_7"}:
            return GROOT_N1_7
        if normalized in {"gr00t_n1_5", "gr00tn1_5", "gr00t_n15", "gr00t_n1d5", "gr00tn1d5"}:
            return GROOT_N1_5
    if config.get("model_name") == GROOT_N1_7_BACKBONE_MODEL:
        return GROOT_N1_7
    # Eagle VLM 主干是 N1.7 之前的 GR00T 检查点特有的（N1.7 使用 Cosmos/Qwen3-VL）。
    backbone_cfg = config.get("backbone_cfg")
    if isinstance(backbone_cfg, dict) and "eagle_path" in backbone_cfg:
        return GROOT_N1_5
    return None


@PreTrainedConfig.register_subclass("groot")
@dataclass
class GrootConfig(PreTrainedConfig):
    """Groot 策略封装的配置。"""

    # 基本策略设置
    n_obs_steps: int = 1
    chunk_size: int = 40
    n_action_steps: int = 40

    # 维度设置（必须与预训练 GR00T 模型的期望一致）
    # 最大状态维度。较短的状态会被零填充。
    max_state_dim: int = 132

    # 最大动作维度。较短的动作会被零填充。
    max_action_dim: int = 132

    # GR00T 在其处理器步骤内部对状态/动作进行归一化（按具身使用
    # q01/q99 百分位数的 min/max 归一化），而 Qwen3-VL 主干的图像处理器
    # 负责图像归一化。因此该策略不使用 LeRobot 的
    # NormalizerProcessorStep/UnnormalizerProcessorStep，所以此映射对每个特征
    # 都有意设为 IDENTITY，且 make_groot_pre_post_processors 不会参考它。
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Groot 特有的模型参数

    # 基础 GR00T N1.7 模型的路径或 HuggingFace 模型 ID，将加载其主干权重和
    # 检查点附属文件（statistics.json、processor_config.json 等）。这是模型的
    # *来源*，有意与继承的 `pretrained_path` 区分开：
    # `pretrained_path`（`--policy.path`）指向已保存的 LeRobot 检查点目录，其
    # `config.json` 带有 `type` 字段；而原始的 NVIDIA GR00T 检查点没有该字段，
    # 因此只能通过 `base_model_path`（`--policy.base_model_path`）加载。
    # 未设置时默认为 GROOT_N1_7_BASE_MODEL（在 __post_init__ 中解析）。
    base_model_path: str | None = None

    # 可选的具名动作变换，在原始 N1.7 检查点解码之后、env.step() 之前应用。
    # 'auto'（默认）解析为具身默认值（'libero_sim' 对应 'libero'，其他不做变换）。
    # 传入 'none' 可显式禁用该变换，对 'libero_sim' 同样有效。
    action_decode_transform: str | None = GROOT_ACTION_DECODE_TRANSFORM_AUTO

    # 训练时使用的具身标签（例如 'new_embodiment'、'gr1'）
    embodiment_tag: str = "new_embodiment"

    # 微调控制参数

    # 是否微调 llm 主干
    tune_llm: bool = False

    # 是否微调视觉塔
    tune_visual: bool = False

    # 是否微调投影器
    tune_projector: bool = True

    # 是否微调扩散模型
    tune_diffusion_model: bool = True

    # 是否微调动作头中的 VL LayerNorm + VL 自注意力投影器。
    tune_vlln: bool = True

    # 要微调的 LLM 主干顶层数量（0 = 不微调）。允许只调整最后的语言层，
    # 而不解冻整个主干；与 `tune_llm` 相互独立，后者会调整整个 LLM。
    tune_top_llm_layers: int = 0

    # 推理时参数：用于解码动作块的 flow-matching 去噪步数。
    # 在推理延迟与动作质量之间权衡。
    # None 表示保留检查点中的值（GR00T N1.7 默认：4）。
    num_inference_timesteps: int | None = None

    # 推理时参数：实时分块（RTC）重叠混合的渐变率，在 RTC 引擎提供
    # 上一块前缀时使用。值越大，对重叠前缀的混合越激进。
    # None 表示保留检查点中的值（GR00T N1.7 默认：6.0）。
    rtc_ramp_rate: float | None = None

    # 推理时参数：是否为 Qwen3-VL 主干请求 flash-attention-2 核。
    # flash-attn 是可选的、由用户自行管理的优化；当它不存在时（默认情况），
    # 主干会透明地回退到数值等价的 SDPA。
    # 只有在安装了与你的 torch/CUDA 环境匹配的 flash-attn 构建后才设为 True。
    use_flash_attention: bool = False

    # 启用 GR00T 风格的状态相对动作块（动作块以当前观测状态为参照表达）。
    use_relative_actions: bool = False

    # relative_exclude_joints 指定保持绝对值的动作维度；匹配方式为
    # 针对数据集动作特征名的子串匹配且大小写不敏感。默认为空时，所有维度
    # 都被视为相对值（包括夹爪）——例如设为 ["gripper"] 可让夹爪保持绝对值，
    # 与 Isaac-GR00T 的单臂 + 绝对夹爪约定一致。
    relative_exclude_joints: list[str] = field(default_factory=list)

    # 训练参数
    optimizer_lr: float = 1e-4
    # Isaac-GR00T N1.7 微调使用 AdamW，betas 为 (0.9, 0.999)。
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    use_bf16: bool = True
    # 原生 N1.7 微调方案将模型参数保持为 FP32，并在 BF16 autocast 下计算。
    model_params_fp32: bool = True

    # TODO(Steven)：在未来的版本中移除这些已弃用的字段。
    # 已弃用的 Isaac-GR00T runner / GR00T N1.5 字段，以及（从未接线的）LoRA 字段——
    # 除 `tokenizer_assets_repo` 的 N1.5 绊线检测和 __post_init__ 中 `image_size` 的
    # 旧值重映射外，LeRobot 的 N1.7 实现均不使用它们。保留它们只是为了让早期
    # lerobot 版本保存的 config.json（尤其是 GR00T N1.5 检查点）仍能被 draccus 解析
    # ——它会拒绝未知字段——随后以清晰的 N1.5 移除提示拒绝，而不是抛出
    # 难以理解的 draccus 解码错误。
    image_size: tuple[int, int] = (256, 256)  # 图像尺寸调整由主干的图像处理器处理。
    tokenizer_assets_repo: str | None = None
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_full_model: bool = False
    video_backend: str = "decord"
    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True
    dataset_paths: list[str] | None = None
    output_dir: str = "./tmp/gr00t"
    save_steps: int = 1000
    max_steps: int = 10000
    batch_size: int = 32
    dataloader_num_workers: int = 8
    report_to: str = "wandb"
    resume: bool = False

    def __post_init__(self):
        if self.tokenizer_assets_repo is not None:
            raise ValueError(
                "Config sets 'tokenizer_assets_repo', which only existed for GR00T N1.5; this looks "
                f"like a legacy GR00T N1.5 checkpoint or config. {GROOT_N1_5_REMOVAL_GUIDANCE}"
            )

        self.action_decode_transform = normalize_groot_action_decode_transform(self.action_decode_transform)
        if self.base_model_path is None:
            self.base_model_path = GROOT_N1_7_BASE_MODEL

        # N1.7 LIBERO 检查点输出的夹爪动作范围为 [0, 1]，但 LIBERO
        # 仿真器期望的是 OpenVLA/[-1, 1] 的符号约定。NVIDIA 的 rollout
        # 封装会应用这一转换；这里照搬该做法，使在 'libero_sim' 具身上的
        # 评估能够正确抓取，而不是得到 0% 的成功率。
        # 这与针对动作执行范围已经做的具身特定处理一致
        # （见 infer_groot_n1_7_action_execution_horizon）。
        # 只有 'auto' 哨兵值会解析为具身默认值；显式的 'none'
        # （上面已归一化为 None）会保持该变换处于禁用状态。
        if self.action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_AUTO:
            self.action_decode_transform = (
                GROOT_ACTION_DECODE_TRANSFORM_LIBERO if self.embodiment_tag == "libero_sim" else None
            )

        # GR00T N1.5 时代的默认值（例如来自旧命令或过期配置的
        # --policy.chunk_size=50）会被迁移为 N1.7 检查点所期望的值，并给出警告。
        # dataclass 的默认值已经是 N1.7 的值，因此普通的
        # GrootConfig() 永远不会触发此逻辑。
        legacy_default_remaps = (
            ("max_state_dim", 64, 132),
            ("max_action_dim", 32, 132),
            ("chunk_size", 50, 40),
            ("n_action_steps", 50, 40),
            ("image_size", (224, 224), (256, 256)),
        )
        for field_name, legacy_value, n1_7_value in legacy_default_remaps:
            current_value = getattr(self, field_name)
            if isinstance(legacy_value, tuple):
                current_value = tuple(current_value)
            if current_value == legacy_value:
                logger.warning(
                    "GrootConfig.%s=%s matches a legacy GR00T N1.5-era default; remapping it to %s, "
                    "the value expected by GR00T N1.7 checkpoints. Set a different value explicitly "
                    "if this is not what you want.",
                    field_name,
                    legacy_value,
                    n1_7_value,
                )
                setattr(self, field_name, n1_7_value)

        inferred_version = infer_groot_model_version(self.base_model_path)
        if inferred_version is not None and inferred_version != GROOT_N1_7:
            message = (
                f"GR00T model_version '{GROOT_N1_7}' does not match base_model_path "
                f"'{self.base_model_path}', which looks like '{inferred_version}'."
            )
            if inferred_version == GROOT_N1_5:
                message = f"{message} {GROOT_N1_5_REMOVAL_GUIDANCE}"
            raise ValueError(message)

        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )

    def validate_features(self) -> None:
        """校验并设置 Groot 的输入/输出特征。"""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "Groot policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature
        else:
            state_shape = self.input_features[OBS_STATE].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature
        else:
            action_shape = self.output_features[ACTION].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    f"Either reduce action dimension or increase max_action_dim in config."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        """返回优化器配置。"""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=1.0,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        """返回调度器配置。

        Isaac-GR00T 使用 HF Trainer 的 cosine 调度，并基于实际训练更新次数
        进行约 5% 的预热；DiffuserSchedulerConfig 封装了相同的
        diffusers/transformers `get_scheduler("cosine")` 实现，并在运行时
        从外层的 --steps 值推导 num_training_steps。
        """
        return DiffuserSchedulerConfig(
            name="cosine",
            num_warmup_steps=math.ceil(self.max_steps * self.warmup_ratio),
        )

    @property
    def observation_delta_indices(self) -> None:
        """返回增量观测的索引（Groot 为 None）。"""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """返回增量动作的索引。"""
        model_action_horizon = (
            infer_groot_n1_7_action_horizon(self.base_model_path, self.embodiment_tag) or 40
        )
        return list(range(min(self.chunk_size, model_action_horizon)))

    @property
    def drop_n_last_frames(self) -> int:
        """排除无法提供完整 N1.7 动作块的回合尾部。"""
        return max(0, len(self.action_delta_indices) - 1)

    @property
    def reward_delta_indices(self) -> None:
        """返回增量奖励的索引（Groot 为 None）。"""
        return None
