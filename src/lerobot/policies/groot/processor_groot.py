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
import random
from copy import copy, deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as tv_functional
from einops import rearrange
from torchvision.transforms import InterpolationMode

from lerobot.utils.import_utils import _datasets_available, _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import (
        AutoTokenizer,
        ProcessorMixin,
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )
else:
    AutoTokenizer = None
    ProcessorMixin = object
    Qwen2VLImageProcessor = None
    Qwen3VLProcessor = None
    Qwen3VLVideoProcessor = None

if TYPE_CHECKING or _datasets_available:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = None

from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    to_relative_actions,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.device_utils import get_safe_torch_device

from .configuration_groot import (
    GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
    GROOT_N1_5_REMOVAL_GUIDANCE,
    GROOT_N1_7_BACKBONE_MODEL,
    N1_7_DEFAULT_IMAGE_CROP_SIZE,
    N1_7_DEFAULT_IMAGE_TARGET_SIZE,
    GrootConfig,
    is_raw_groot_n1_7_checkpoint,
)
from .utils import (
    as_int_pair,
    as_optional_float,
    as_optional_int,
    config_value,
    flatten_n1_7_modality_stats,
    has_modality_stats,
    infer_n1_7_batch_size_and_device,
    prepare_n1_7_language_batch,
    read_json,
    relative_eef_to_absolute,
    stat_dim_from_entry,
)

# 原生 GR00T N1.7 动作时域：检查点训练时预测的是 40 步的动作分块，
# 因此处理器侧的时域上限为该值。
N1_7_NATIVE_ACTION_HORIZON = 40

N1_7_EMBODIMENT_MAPPING = {
    "oxe_droid_relative_eef_relative_joint": 24,
    "xdof_relative_eef_relative_joint": 27,
    "xdof_relative_eef_relative_joint_subtask": 27,
    "real_g1_relative_eef_relative_joints": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "real_r1_pro_sharpa_relative_eef_human": 26,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 26,
    "real_r1_pro_sharpa_relative_eef_mecka": 26,
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "new_embodiment": 10,
}


@dataclass
class _GrootN17CheckpointProcessorAssets:
    """从原始 Isaac-GR00T N1.7 检查点加载的处理器元数据。

    公开的 N1.7 检查点会把预处理和动作解码的选项与模型权重存放在一起。
    将这些值集中保存可以避免回退到 LeRobot 的默认值——那些默认值对旧版
    GR00T 变体有效，但会改变 N1.7 的输入或解码后的动作。
    """

    stats: dict[str, dict[str, Any]]
    raw_stats: dict[str, Any]
    modality_config: dict[str, Any]
    embodiment_mapping: dict[str, int]
    formalize_language: bool
    valid_action_horizon: int | None
    max_action_horizon: int | None
    video_horizon: int | None
    use_percentiles: bool
    use_relative_action: bool
    state_dropout_prob: float
    clip_outliers: bool
    video_modality_keys: list[str] | None
    image_crop_size: list[int] | None
    image_target_size: list[int] | None
    shortest_image_edge: int | None
    crop_fraction: float | None
    use_albumentations: bool
    letter_box_transform: bool


@dataclass(frozen=True)
class _GrootN17ActionGroup:
    key: str
    indices: list[int]
    relative: bool


def _load_n1_7_checkpoint_processor_assets(config: GrootConfig) -> _GrootN17CheckpointProcessorAssets | None:
    """从检查点的附属 JSON 文件加载 N1.7 处理器设置。

    对于非原始 N1.7 检查点返回 ``None``，以便通用 GR00T 流水线继续使用
    调用方提供的数据集统计量和配置值。
    """

    if not is_raw_groot_n1_7_checkpoint(config.base_model_path):
        return None

    checkpoint_path = Path(config.base_model_path).expanduser()
    processor_config = read_json(checkpoint_path / "processor_config.json")
    processor_kwargs = processor_config.get("processor_kwargs", {})
    if not isinstance(processor_kwargs, dict):
        processor_kwargs = {}

    all_stats = read_json(checkpoint_path / "statistics.json")
    raw_stats = all_stats.get(config.embodiment_tag)
    if not isinstance(raw_stats, dict):
        raw_stats = {}

    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        modality_configs = {}
    modality_config = modality_configs.get(config.embodiment_tag)
    if not isinstance(modality_config, dict):
        modality_config = {}

    use_relative_action = bool(processor_kwargs.get("use_relative_action", False))
    state_dropout_prob = as_optional_float(processor_kwargs.get("state_dropout_prob"))
    if state_dropout_prob is None:
        state_dropout_prob = 0.0
    stats = _load_n1_7_checkpoint_stats(
        checkpoint_path,
        processor_kwargs,
        config.embodiment_tag,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_relative_action=use_relative_action,
    )
    embodiment_mapping = _load_n1_7_embodiment_mapping(checkpoint_path) or dict(N1_7_EMBODIMENT_MAPPING)
    formalize_language = processor_kwargs.get("formalize_language", True)
    if not isinstance(formalize_language, bool):
        formalize_language = True
    clip_outliers = processor_kwargs.get("clip_outliers", True)
    if not isinstance(clip_outliers, bool):
        clip_outliers = True
    use_albumentations = processor_kwargs.get("use_albumentations", False)
    if not isinstance(use_albumentations, bool):
        use_albumentations = False
    letter_box_transform = processor_kwargs.get("letter_box_transform", False)
    if not isinstance(letter_box_transform, bool):
        letter_box_transform = False

    valid_action_horizon = _load_n1_7_checkpoint_action_horizon(processor_kwargs, config.embodiment_tag)
    video_horizon = _load_n1_7_checkpoint_video_horizon(processor_kwargs, config.embodiment_tag)
    video_modality_keys = _load_n1_7_checkpoint_video_modality_keys(processor_kwargs, config.embodiment_tag)
    max_action_horizon = processor_kwargs.get("max_action_horizon")
    if not isinstance(max_action_horizon, int):
        max_action_horizon = None

    return _GrootN17CheckpointProcessorAssets(
        stats=stats,
        raw_stats=raw_stats,
        modality_config=modality_config,
        embodiment_mapping=embodiment_mapping,
        formalize_language=formalize_language,
        valid_action_horizon=valid_action_horizon,
        max_action_horizon=max_action_horizon,
        video_horizon=video_horizon,
        use_percentiles=bool(processor_kwargs.get("use_percentiles", False)),
        use_relative_action=use_relative_action,
        state_dropout_prob=state_dropout_prob,
        clip_outliers=clip_outliers,
        video_modality_keys=video_modality_keys,
        image_crop_size=as_int_pair(processor_kwargs.get("image_crop_size")),
        image_target_size=as_int_pair(processor_kwargs.get("image_target_size")),
        shortest_image_edge=as_optional_int(processor_kwargs.get("shortest_image_edge")),
        crop_fraction=as_optional_float(processor_kwargs.get("crop_fraction")),
        use_albumentations=use_albumentations,
        letter_box_transform=letter_box_transform,
    )


def _load_n1_7_embodiment_mapping(checkpoint_path: Path) -> dict[str, int] | None:
    mapping = read_json(checkpoint_path / "embodiment_id.json")
    if not mapping:
        return None
    parsed: dict[str, int] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        try:
            parsed[key] = int(value)
        except (TypeError, ValueError):
            continue
    return parsed or None


def _load_n1_7_checkpoint_stats(
    checkpoint_path: Path,
    processor_kwargs: dict[str, Any],
    embodiment_tag: str,
    *,
    raw_stats: dict[str, Any] | None = None,
    modality_config: dict[str, Any] | None = None,
    use_relative_action: bool = False,
) -> dict[str, dict[str, Any]]:
    """将检查点的模态分组统计量转换为 LeRobot 的扁平张量统计量。

    Isaac-GR00T 的统计量按 EEF 位姿、关节等语义分组作为键。LeRobot 的
    归一化器作用于单个向量，因此本函数在展平每一项所选统计量的同时，
    保留检查点中的分组顺序。
    """

    if raw_stats is None:
        all_stats = read_json(checkpoint_path / "statistics.json")
        raw_stats = all_stats.get(embodiment_tag)
    if not isinstance(raw_stats, dict):
        return {}

    if modality_config is None:
        modality_configs = processor_kwargs.get("modality_configs", {})
        if not isinstance(modality_configs, dict):
            return {}
        modality_config = modality_configs.get(embodiment_tag)
    if not isinstance(modality_config, dict):
        return {}

    use_percentiles = processor_kwargs.get("use_percentiles", False)
    return {
        OBS_STATE: flatten_n1_7_modality_stats(
            embodiment_stats=raw_stats,
            embodiment_config=modality_config,
            modality="state",
            use_percentiles=bool(use_percentiles),
            use_relative_action=use_relative_action,
        ),
        ACTION: flatten_n1_7_modality_stats(
            embodiment_stats=raw_stats,
            embodiment_config=modality_config,
            modality="action",
            use_percentiles=bool(use_percentiles),
            use_relative_action=use_relative_action,
        ),
    }


def _load_n1_7_checkpoint_action_horizon(
    processor_kwargs: dict[str, Any],
    embodiment_tag: str,
) -> int | None:
    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
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


def _load_n1_7_checkpoint_video_horizon(
    processor_kwargs: dict[str, Any],
    embodiment_tag: str,
) -> int | None:
    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None
    embodiment_config = modality_configs.get(embodiment_tag, {})
    if not isinstance(embodiment_config, dict):
        return None
    video_config = embodiment_config.get("video", {})
    if not isinstance(video_config, dict):
        return None
    delta_indices = video_config.get("delta_indices", [])
    if not isinstance(delta_indices, list):
        return None
    return len(delta_indices) or None


def _load_n1_7_checkpoint_video_modality_keys(
    processor_kwargs: dict[str, Any],
    embodiment_tag: str,
) -> list[str] | None:
    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None
    embodiment_config = modality_configs.get(embodiment_tag, {})
    if not isinstance(embodiment_config, dict):
        return None
    video_config = embodiment_config.get("video", {})
    if not isinstance(video_config, dict):
        return None
    modality_keys = video_config.get("modality_keys", [])
    if not isinstance(modality_keys, list):
        return None
    keys = [key for key in modality_keys if isinstance(key, str)]
    return keys or None


# GR00T 在自己的处理器步骤内部完成动作的归一化和表示，因此它刻意没有标准的
# NormalizerProcessorStep/UnnormalizerProcessorStep，也没有通用的相对/绝对动作步骤。
# ``lerobot-train`` 仍可能发出这些通用覆写键；对于 GR00T 流水线而言，它们匹配不到
# 任何步骤是合理的，因此提前将其丢弃，同时不掩盖其他无关的拼写错误键。
_GROOT_ABSENT_STANDARD_OVERRIDE_KEYS = frozenset(
    {
        "absolute_actions_processor",
        "normalizer_processor",
        "relative_actions_processor",
        "unnormalizer_processor",
    }
)


def _drop_groot_absent_standard_overrides(overrides: dict[str, Any] | None) -> dict[str, Any] | None:
    """移除 GR00T 流水线中没有对应步骤的标准覆写键。"""

    if not overrides:
        return overrides

    filtered: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in _GROOT_ABSENT_STANDARD_OVERRIDE_KEYS:
            logging.debug(
                "Ignoring override key '%s': GR00T normalizes inside its own processor steps and has "
                "no matching step (see GrootConfig.normalization_mapping).",
                key,
            )
            continue
        filtered[key] = value
    return filtered


def _apply_groot_step_overrides(
    pipeline: PolicyProcessorPipeline,
    overrides: dict[str, Any] | None,
) -> None:
    """将 ``from_pretrained`` 风格的步骤覆写应用到新建的流水线上。

    原始 N1.7 检查点是从零构建处理器，而不是反序列化得到，因此调用方的覆写
    必须应用到已构造好的步骤上。覆写键匹配步骤的注册表名，或者为方便起见
    匹配其类名（``PolicyProcessorPipeline.from_pretrained`` 仅按注册表名匹配
    已注册的步骤——建议优先使用注册表名，这样在检查点转换并从序列化流水线
    重新加载后覆写仍能生效）。匹配不到的键或字段会直接报错，而不是被静默
    丢弃（GR00T 没有对应步骤的标准归一化键会事先由
    ``_drop_groot_absent_standard_overrides`` 移除）。
    """

    if not overrides:
        return

    def _step_keys(step: ProcessorStep) -> set[str]:
        keys = {type(step).__name__}
        registry_name = getattr(type(step), "_registry_name", None)
        if registry_name:
            keys.add(registry_name)
        return keys

    for override_key, step_overrides in overrides.items():
        matched_steps = [step for step in pipeline.steps if override_key in _step_keys(step)]
        if not matched_steps:
            available = [
                getattr(type(step), "_registry_name", None) or type(step).__name__ for step in pipeline.steps
            ]
            raise KeyError(
                f"Override key '{override_key}' does not match any step of the GR00T processor pipeline "
                f"built for this raw N1.7 checkpoint. Available step keys: {available}."
            )
        for step in matched_steps:
            if not is_dataclass(step):
                raise TypeError(
                    f"Cannot apply overrides to step '{override_key}': it is not a dataclass step."
                )
            init_field_names = {f.name for f in fields(step) if f.init}
            for field_name, value in dict(step_overrides).items():
                if field_name not in init_field_names:
                    raise TypeError(
                        f"Override field '{field_name}' is not a config field of step '{override_key}'. "
                        f"Available fields: {sorted(init_field_names)}."
                    )
                setattr(step, field_name, value)
            # 重新派生那些由被覆写配置计算出来的属性（例如
            # DeviceProcessorStep 会在 __post_init__ 中解析其 torch.device）。
            post_init = getattr(step, "__post_init__", None)
            if callable(post_init):
                post_init()


def _set_groot_preprocessor_training(
    preprocessor: PolicyProcessorPipeline,
    *,
    training: bool,
) -> None:
    """设置 GR00T 随机性处理器步骤的仅运行时模式。

    任何暴露了 ``training`` 字段的 dataclass 步骤都会参与，因此处理器步骤
    可以自行启用仅训练时的行为（dropout、数据增强），而无需本辅助函数逐一列举。
    """
    for step in preprocessor.steps:
        if is_dataclass(step) and any(f.name == "training" for f in fields(step)):
            step.training = training


def make_groot_pre_post_processors_from_pretrained(
    config: GrootConfig,
    pretrained_path: str,
    *,
    revision: str | None = None,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """为原始 N1.7 检查点或序列化的 LeRobot 流水线加载 Groot 处理器。"""

    # 丢弃 lerobot-train 无条件发出的标准 normalizer/unnormalizer 覆写键：
    # GR00T 没有这类步骤，它们会导致原始检查点和序列化覆写两条路径都报错。
    # 此操作必须在下面任一分支之前执行。
    preprocessor_overrides = _drop_groot_absent_standard_overrides(preprocessor_overrides)
    postprocessor_overrides = _drop_groot_absent_standard_overrides(postprocessor_overrides)

    if is_raw_groot_n1_7_checkpoint(pretrained_path):
        processor_cfg = copy(config)
        processor_cfg.base_model_path = str(pretrained_path)
        preprocessor, postprocessor = make_groot_pre_post_processors(
            config=processor_cfg,
            dataset_stats=dataset_stats,
            dataset_meta=dataset_meta,
        )
        # 原始检查点没有可供加载覆写的序列化流水线，
        # 因此将调用方的覆写（例如来自 lerobot-eval 或策略服务器的
        # device 和 rename_map）应用到新建的步骤上。
        _apply_groot_step_overrides(preprocessor, preprocessor_overrides)
        _apply_groot_step_overrides(postprocessor, postprocessor_overrides)
        _apply_groot_action_decode_transform(postprocessor, config.action_decode_transform)
        return preprocessor, postprocessor

    preprocessor, postprocessor = _load_groot_processor_pipelines(
        pretrained_path,
        revision=revision,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
        preprocessor_config_filename=preprocessor_config_filename,
        postprocessor_config_filename=postprocessor_config_filename,
    )
    _reconnect_groot_relative_absolute_steps(preprocessor, postprocessor)
    _reconnect_groot_n1_7_pack_decode_steps(preprocessor, postprocessor)
    _apply_groot_action_decode_transform(postprocessor, config.action_decode_transform)
    _set_groot_preprocessor_training(preprocessor, training=dataset_meta is not None)
    return preprocessor, postprocessor


def _load_groot_processor_pipelines(
    pretrained_path: str,
    *,
    revision: str | None,
    preprocessor_overrides: dict[str, Any],
    postprocessor_overrides: dict[str, Any],
    preprocessor_config_filename: str,
    postprocessor_config_filename: str,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    # 在反序列化之前注册 GR00T N1.5 的拒绝存根，这样当已保存的 N1.5 流水线
    # 引用它们的注册表名时，会以规范的移除指引报错。
    _register_removed_n1_5_step_stubs()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=preprocessor_config_filename,
        revision=revision,
        overrides=preprocessor_overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=postprocessor_config_filename,
        revision=revision,
        overrides=postprocessor_overrides,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def _reconnect_groot_relative_absolute_steps(
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> None:
    relative_step = next(
        (step for step in preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)),
        None,
    )
    if relative_step is None:
        return

    for step in postprocessor.steps:
        if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
            step.relative_step = relative_step


def _reconnect_groot_n1_7_pack_decode_steps(
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> None:
    """将反序列化后的 N1.7 动作解码步骤重新关联到其打包步骤。

    打包步骤保存着逐实例的原始状态缓存，相对动作解码要从中读取参考状态；
    而这个关联本身不会被序列化。
    """

    pack_step = next(
        (step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep)),
        None,
    )
    if pack_step is None:
        return

    for step in postprocessor.steps:
        if isinstance(step, GrootN17ActionDecodeStep) and step.pack_step is None:
            step.pack_step = pack_step


def _apply_groot_action_decode_transform(
    postprocessor: PolicyProcessorPipeline,
    action_decode_transform: str | None,
) -> None:
    use_libero_transform = action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO

    for step in postprocessor.steps:
        if isinstance(step, GrootN17ActionDecodeStep):
            step.action_decode_transform = action_decode_transform
        elif isinstance(step, GrootActionUnpackUnnormalizeStep):
            step.libero_gripper_action = use_libero_transform
            if use_libero_transform:
                step.libero_gripper_binarize = True


def _resolve_feature_names_from_dataset_meta(dataset_meta: Any | None, feature_key: str) -> list[str] | None:
    features = getattr(dataset_meta, "features", {}) or {}
    feature = features.get(feature_key) if isinstance(features, dict) else None
    names = feature.get("names") if isinstance(feature, dict) else getattr(feature, "names", None)
    return list(names) if names is not None else None


def _resolve_action_feature_names_from_dataset_meta(dataset_meta: Any | None) -> list[str] | None:
    return _resolve_feature_names_from_dataset_meta(dataset_meta, ACTION)


def _resolve_visual_modality_keys_from_dataset_meta(dataset_meta: Any | None) -> list[str] | None:
    features = getattr(dataset_meta, "features", {}) or {}
    if not isinstance(features, dict):
        return None

    keys: list[str] = []
    for key, value in features.items():
        dtype = value.get("dtype") if isinstance(value, dict) else getattr(value, "dtype", None)
        feature_type = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
        is_visual = dtype in {"image", "video"} or str(feature_type).upper().endswith("VISUAL")
        if not is_visual or not isinstance(key, str) or not key.startswith(f"{OBS_IMAGES}."):
            continue
        keys.append(key.removeprefix(f"{OBS_IMAGES}."))
    return keys or None


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    item = getattr(value, "item", None)
    if callable(item):
        return int(item())
    return int(value)


def _to_float_tensor(value: Any, *, key: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"Cannot compute relative action statistics: sample is missing '{key}'.")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _state_reference_batch(state: torch.Tensor) -> torch.Tensor:
    if state.ndim == 1:
        return state.unsqueeze(0)
    if state.ndim == 2:
        return state
    if state.ndim > 2:
        return state.reshape(-1, state.shape[-1])[-1:].contiguous()
    raise ValueError(f"observation.state must have at least 1 dimension, got shape {tuple(state.shape)}.")


def _action_training_batch(action: torch.Tensor, state_batch: torch.Tensor) -> torch.Tensor:
    if action.ndim == 1:
        return action.unsqueeze(0)
    if action.ndim == 2:
        if state_batch.shape[0] == action.shape[0] and state_batch.shape[0] > 1:
            return action
        return action.unsqueeze(0)
    if action.ndim == 3:
        return action
    raise ValueError(f"action must be (D,), (T, D), (B, D), or (B, T, D), got {tuple(action.shape)}.")


def _relative_action_chunks_by_horizon(
    relative_action: torch.Tensor, pad_mask: Any | None
) -> list[list[np.ndarray]]:
    if relative_action.ndim == 2:
        relative_action = relative_action.unsqueeze(0)
    if relative_action.ndim != 3:
        raise ValueError(
            "Cannot compute horizon-preserving relative action statistics from "
            f"shape {tuple(relative_action.shape)}."
        )

    batch_size, horizon, _action_dim = relative_action.shape
    keep = torch.ones(batch_size, horizon, dtype=torch.bool)
    if pad_mask is not None:
        mask = torch.as_tensor(pad_mask, dtype=torch.bool).cpu()
        if mask.ndim == 1 and batch_size == 1 and mask.numel() == horizon:
            keep[0, :] = not bool(mask.any())
        elif mask.ndim == 2 and tuple(mask.shape) == (batch_size, horizon):
            complete_chunks = ~mask.any(dim=1)
            keep = complete_chunks[:, None].expand(batch_size, horizon).clone()

    chunks: list[list[np.ndarray]] = [[] for _ in range(horizon)]
    relative_np = relative_action.detach().cpu().numpy()
    for batch_idx in range(batch_size):
        for horizon_idx in range(horizon):
            if keep[batch_idx, horizon_idx]:
                chunks[horizon_idx].append(relative_np[batch_idx, horizon_idx])
    return chunks


def _compute_horizon_relative_action_stats(
    chunks_by_horizon: list[list[np.ndarray]],
) -> dict[str, np.ndarray]:
    if not chunks_by_horizon or not any(chunks_by_horizon):
        raise ValueError("Cannot compute relative action statistics without unpadded action vectors.")

    stats: dict[str, list[np.ndarray]] = {key: [] for key in ("min", "max", "mean", "std", "q01", "q99")}
    counts: list[int] = []
    for horizon_idx, vectors in enumerate(chunks_by_horizon):
        if len(vectors) < 2:
            raise ValueError(
                "Cannot compute horizon-preserving relative action statistics from fewer than 2 "
                f"unpadded vectors at action timestep {horizon_idx}."
            )
        values = np.stack(vectors, axis=0).astype(np.float32)
        stats["min"].append(np.min(values, axis=0))
        stats["max"].append(np.max(values, axis=0))
        stats["mean"].append(np.mean(values, axis=0))
        stats["std"].append(np.std(values, axis=0))
        stats["q01"].append(np.quantile(values, 0.01, axis=0).astype(np.float32))
        stats["q99"].append(np.quantile(values, 0.99, axis=0).astype(np.float32))
        counts.append(len(vectors))

    computed = {key: np.stack(values, axis=0) for key, values in stats.items()}
    computed["count"] = np.asarray(counts, dtype=np.int64)
    return computed


def _iter_action_state_training_samples(dataset: Any):
    ensure_reader = getattr(dataset, "_ensure_reader", None)
    # 只有默认的 parquet reader 暴露了 hf_dataset；其他 reader
    # （例如 lance）会落到下面通用的逐条循环。
    if callable(ensure_reader) and hasattr(reader := ensure_reader(), "hf_dataset"):
        if reader.hf_dataset is None:
            reader.load_and_activate()
        delta_indices = getattr(reader, "delta_indices", None)
        for idx in range(len(dataset)):
            item = reader.hf_dataset[idx]
            action = item.get(ACTION)
            state = item.get(OBS_STATE)
            pad_mask = None
            if delta_indices is not None and ACTION in delta_indices:
                ep_idx = _as_int(item["episode_index"])
                abs_idx = _as_int(item["index"])
                query_indices, padding = reader._get_query_indices(abs_idx, ep_idx)
                action = reader._query_hf_dataset({ACTION: query_indices[ACTION]})[ACTION]
                pad_mask = padding.get(f"{ACTION}_is_pad")
            yield action, state, pad_mask
        return

    for idx in range(len(dataset)):
        item = dataset[idx]
        yield item.get(ACTION), item.get(OBS_STATE), item.get(f"{ACTION}_is_pad")


def _make_relative_action_training_stats(
    dataset: Any,
    *,
    exclude_joints: list[str] | None,
    action_names: list[str] | None,
    preserve_action_horizon: bool = True,
) -> dict[str, dict[str, Any]]:
    try:
        dataset_len = len(dataset)
    except TypeError as exc:
        raise ValueError(
            "Cannot compute relative action statistics for a dataset without a finite length. "
            "Disable streaming or provide precomputed relative action statistics."
        ) from exc

    if dataset_len == 0:
        raise ValueError("Cannot compute relative action statistics for an empty dataset.")

    relative_step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=list(exclude_joints or []),
        action_names=action_names,
    )
    stats = deepcopy(getattr(getattr(dataset, "meta", None), "stats", {}) or {})
    chunks_by_horizon: list[list[np.ndarray]] | None = None
    num_vectors = 0

    for action_value, state_value, pad_mask in _iter_action_state_training_samples(dataset):
        action = _to_float_tensor(action_value, key=ACTION)
        state = _to_float_tensor(state_value, key=OBS_STATE)
        state_batch = _state_reference_batch(state)
        action_batch = _action_training_batch(action, state_batch)
        if action_batch.shape[0] != state_batch.shape[0]:
            if state_batch.shape[0] == 1:
                state_batch = state_batch.expand(action_batch.shape[0], -1)
            else:
                raise ValueError(
                    "Cannot compute relative action statistics: action and state batch sizes differ "
                    f"({action_batch.shape[0]} vs {state_batch.shape[0]})."
                )

        relative_action = to_relative_actions(
            action_batch,
            state_batch,
            relative_step._build_mask(action_batch.shape[-1]),
        )
        if not preserve_action_horizon:
            relative_action = relative_action.reshape(-1, relative_action.shape[-1]).unsqueeze(0)
            pad_mask = None
        sample_chunks = _relative_action_chunks_by_horizon(relative_action, pad_mask)
        if chunks_by_horizon is None:
            chunks_by_horizon = [[] for _ in range(len(sample_chunks))]
        if len(sample_chunks) != len(chunks_by_horizon):
            raise ValueError(
                "Cannot compute horizon-preserving relative action statistics from samples with "
                f"different action horizons ({len(sample_chunks)} vs {len(chunks_by_horizon)})."
            )
        for horizon_idx, vectors in enumerate(sample_chunks):
            chunks_by_horizon[horizon_idx].extend(vectors)
            num_vectors += len(vectors)

    if num_vectors < 2:
        raise ValueError(
            "Cannot compute relative action statistics from fewer than 2 unpadded action vectors."
        )

    stats[ACTION] = _compute_horizon_relative_action_stats(chunks_by_horizon or [])
    return stats


def _relative_stats_action_horizon(action_stats: dict[str, Any]) -> int | None:
    """若存在保留时域的相对动作统计量，则返回其分块时域长度。"""
    for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
        value = action_stats.get(stat_name)
        if value is None:
            continue
        tensor = torch.as_tensor(value)
        return tensor.shape[0] if tensor.ndim >= 2 else None
    return None


def _stats_preserve_action_horizon(stats: dict[str, dict[str, Any]] | None) -> bool:
    if not stats or ACTION not in stats:
        return False
    action_stats = stats.get(ACTION) or {}
    for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
        value = action_stats.get(stat_name)
        if value is None:
            continue
        return torch.as_tensor(value).ndim >= 2
    return False


def _make_relative_action_training_stats_from_dataset_meta(
    config: GrootConfig, dataset_meta: Any | None
) -> dict[str, dict[str, Any]] | None:
    repo_id = getattr(dataset_meta, "repo_id", None)
    root = getattr(dataset_meta, "root", None)
    fps = getattr(dataset_meta, "fps", None)
    if dataset_meta is None or repo_id is None or root is None or fps is None:
        return None

    require_package("datasets", extra="groot")

    # 相对统计量是在 N1.7 原生时域上按分块时间步逐一计算的，因此即使
    # config.chunk_size 实际执行的步数更少，用于统计的数据集也必须产出
    # 原生长度的动作窗口。
    delta_timestamps = {ACTION: [index / fps for index in range(N1_7_NATIVE_ACTION_HORIZON)]}
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        revision=getattr(dataset_meta, "revision", None),
        download_videos=False,
        return_uint8=True,
    )
    return _make_relative_action_training_stats(
        dataset,
        exclude_joints=list(config.relative_exclude_joints or []),
        action_names=_resolve_action_feature_names_from_dataset_meta(dataset_meta),
        preserve_action_horizon=True,
    )


def _slice_stats_entry(stats: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    if not indices:
        return {}

    max_index = max(indices)
    sliced: dict[str, Any] = {}
    for stat_name, value in stats.items():
        if stat_name == "count":
            sliced[stat_name] = torch.as_tensor(value).flatten().tolist()
            continue
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim >= 2:
            if tensor.shape[-1] <= max_index:
                continue
            sliced[stat_name] = tensor[..., indices].tolist()
        else:
            tensor = tensor.flatten()
            if tensor.numel() <= max_index:
                continue
            sliced[stat_name] = [float(tensor[index].item()) for index in indices]

    if "min" in sliced and "max" in sliced:
        min_arr = np.asarray(sliced["min"], dtype=np.float32)
        max_arr = np.asarray(sliced["max"], dtype=np.float32)
        if "mean" not in sliced:
            sliced["mean"] = ((min_arr + max_arr) * 0.5).tolist()
        if "std" not in sliced:
            sliced["std"] = (np.abs(max_arr - min_arr) * 0.5).tolist()
    return sliced


def _feature_group_key(name: str) -> str:
    base = name.removesuffix(".pos").split(".")[-1]
    return base.replace(" ", "_") or "action"


def _infer_n1_7_action_groups(
    action_names: list[str],
    *,
    action_dim: int,
    exclude_joints: list[str],
) -> list[_GrootN17ActionGroup]:
    if not action_names or action_dim <= 0:
        return []

    names = list(action_names[:action_dim])
    exclude_tokens = [str(token).lower() for token in exclude_joints if token]
    groups: list[_GrootN17ActionGroup] = []
    current_indices: list[int] = []

    def flush_relative_group() -> None:
        if not current_indices:
            return
        key = (
            "single_arm"
            if not any(group.key == "single_arm" for group in groups)
            else f"single_arm_{len(groups)}"
        )
        groups.append(_GrootN17ActionGroup(key=key, indices=list(current_indices), relative=True))
        current_indices.clear()

    for index, name in enumerate(names):
        lowered = str(name).lower()
        is_excluded = any(token == lowered or token in lowered for token in exclude_tokens)
        if is_excluded:
            flush_relative_group()
            groups.append(
                _GrootN17ActionGroup(key=_feature_group_key(str(name)), indices=[index], relative=False)
            )
        else:
            current_indices.append(index)

    flush_relative_group()
    return groups


def _group_stats_by_action_groups(
    stats: dict[str, Any], groups: list[_GrootN17ActionGroup]
) -> dict[str, dict[str, list[float]]]:
    return {group.key: _slice_stats_entry(stats, group.indices) for group in groups}


def _grouped_stats_support_percentiles(
    raw_stats: dict[str, Any],
    modality_config: dict[str, Any],
    *,
    use_relative_action: bool,
) -> bool:
    state_keys = modality_config.get("state", {}).get("modality_keys", [])
    for key in state_keys:
        stats = raw_stats.get("state", {}).get(key, {})
        if "q01" not in stats or "q99" not in stats:
            return False

    action_cfg = modality_config.get("action", {})
    action_keys = action_cfg.get("modality_keys", [])
    action_configs = action_cfg.get("action_configs", [])
    for idx, key in enumerate(action_keys):
        cfg = action_configs[idx] if idx < len(action_configs) else {}
        is_relative = (
            use_relative_action and isinstance(cfg, dict) and config_value(cfg.get("rep")) == "relative"
        )
        if is_relative:
            continue
        stats = raw_stats.get("action", {}).get(key, {})
        if "q01" not in stats or "q99" not in stats:
            return False
    return True


def _build_n1_7_relative_action_processor_assets(
    config: GrootConfig,
    dataset_stats: dict[str, dict[str, Any]] | None,
    dataset_meta: Any | None,
    *,
    base_assets: _GrootN17CheckpointProcessorAssets | None = None,
) -> _GrootN17CheckpointProcessorAssets | None:
    if not config.use_relative_actions or not dataset_stats:
        return None

    try:
        action_dim = int(config.output_features[ACTION].shape[0])
    except Exception:
        return None

    action_names = _resolve_action_feature_names_from_dataset_meta(dataset_meta)
    if not action_names:
        return None

    groups = _infer_n1_7_action_groups(
        action_names,
        action_dim=action_dim,
        exclude_joints=list(config.relative_exclude_joints or []),
    )
    if not groups or not any(group.relative for group in groups):
        return None

    meta_stats = getattr(dataset_meta, "stats", None) or {}
    state_stats = (meta_stats.get(OBS_STATE) if isinstance(meta_stats, dict) else None) or dataset_stats.get(
        OBS_STATE, {}
    )
    absolute_action_stats = (
        meta_stats.get(ACTION) if isinstance(meta_stats, dict) else None
    ) or dataset_stats.get(ACTION, {})
    relative_action_stats = dataset_stats.get(ACTION, {})
    if not state_stats or not absolute_action_stats or not relative_action_stats:
        return None

    raw_stats: dict[str, Any] = {
        "state": _group_stats_by_action_groups(state_stats, groups),
        "action": _group_stats_by_action_groups(absolute_action_stats, groups),
        "relative_action": {
            group.key: _slice_stats_entry(relative_action_stats, group.indices)
            for group in groups
            if group.relative
        },
    }

    action_configs = [
        {
            "rep": "RELATIVE" if group.relative else "ABSOLUTE",
            "type": "NON_EEF",
            "format": "DEFAULT",
            "state_key": None,
        }
        for group in groups
    ]
    # 保留时域的相对统计量是在数据集样本的原生分块长度上按分块时间步计算的，
    # 因此即使 config.chunk_size 要求执行更少的步数，处理器时域也由它决定。
    action_horizon = _relative_stats_action_horizon(relative_action_stats) or min(
        config.chunk_size, N1_7_NATIVE_ACTION_HORIZON
    )
    modality_config: dict[str, Any] = {
        "state": {"modality_keys": [group.key for group in groups]},
        "action": {
            "modality_keys": [group.key for group in groups],
            "action_configs": action_configs,
            "delta_indices": list(range(action_horizon)),
        },
    }
    video_modality_keys = (
        base_assets.video_modality_keys if base_assets is not None else None
    ) or _resolve_visual_modality_keys_from_dataset_meta(dataset_meta)
    if video_modality_keys:
        modality_config["video"] = {
            "modality_keys": list(video_modality_keys),
            "delta_indices": [0],
        }

    if config.chunk_size > action_horizon:
        logging.warning(
            "GrootConfig.chunk_size=%d exceeds the relative-action stats horizon %d; clamping the "
            "valid action horizon to %d. The GR00T N1.7 action head decodes at most the horizon "
            "baked into the relative-action statistics.",
            config.chunk_size,
            action_horizon,
            action_horizon,
        )

    use_percentiles = _grouped_stats_support_percentiles(raw_stats, modality_config, use_relative_action=True)
    flat_stats = {
        OBS_STATE: flatten_n1_7_modality_stats(
            embodiment_stats=raw_stats,
            embodiment_config=modality_config,
            modality="state",
            use_percentiles=use_percentiles,
            use_relative_action=True,
        ),
        ACTION: flatten_n1_7_modality_stats(
            embodiment_stats=raw_stats,
            embodiment_config=modality_config,
            modality="action",
            use_percentiles=use_percentiles,
            use_relative_action=True,
        ),
    }

    return _GrootN17CheckpointProcessorAssets(
        stats=flat_stats,
        raw_stats=raw_stats,
        modality_config=modality_config,
        embodiment_mapping=base_assets.embodiment_mapping
        if base_assets is not None
        else dict(N1_7_EMBODIMENT_MAPPING),
        formalize_language=base_assets.formalize_language if base_assets is not None else True,
        valid_action_horizon=min(config.chunk_size, action_horizon),
        max_action_horizon=action_horizon,
        video_horizon=base_assets.video_horizon if base_assets is not None else None,
        use_percentiles=use_percentiles,
        use_relative_action=True,
        state_dropout_prob=base_assets.state_dropout_prob if base_assets is not None else 0.0,
        clip_outliers=base_assets.clip_outliers if base_assets is not None else True,
        video_modality_keys=video_modality_keys,
        image_crop_size=base_assets.image_crop_size if base_assets is not None else None,
        image_target_size=base_assets.image_target_size if base_assets is not None else None,
        shortest_image_edge=base_assets.shortest_image_edge if base_assets is not None else None,
        crop_fraction=base_assets.crop_fraction if base_assets is not None else None,
        use_albumentations=base_assets.use_albumentations if base_assets is not None else False,
        letter_box_transform=base_assets.letter_box_transform if base_assets is not None else False,
    )


def make_groot_pre_post_processors(
    config: GrootConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """为 Groot 策略创建预处理器和后处理器。

    这会创建一条处理流水线，将 LeRobot 数据格式转换为 Isaac-GR00T 模型
    所期望的格式：

    预处理步骤：
    1. 可选的键重命名（针对数据集的键映射）
    2. 为未批量化的数据添加批量维度
    3. 打包 video/state/action/language/embodiment，并在填充前可选地应用 min-max 归一化
    4. 使用 GR00T N1.7 VLM 主干（Qwen3-VL）将视频+语言编码为中间 VLM 内容
    5. 将 VLM 内容整理（collate）为批量化的主干输入张量
    6. 将张量移动到设备（GPU）

    注意：我们可选地在填充之前利用数据集提供的统计量对 STATE 和 ACTION
    应用 min-max 归一化，将值映射到 [-1, 1]。这与 SO100 风格的预处理一致，
    并保持尺度与 GR00T 一致。

    Args:
        config: Groot 配置，包含 data_config、embodiment_tag 等。
        dataset_stats: 可选的逐键 min/max 统计量，用于填充前的归一化。

    Returns:
        （preprocessor, postprocessor）流水线组成的元组
    """

    dataset_meta = dataset_meta or getattr(config, "_runtime_dataset_meta", None)
    checkpoint_assets = _load_n1_7_checkpoint_processor_assets(config)
    checkpoint_stats = checkpoint_assets.stats if checkpoint_assets is not None else None
    checkpoint_has_stats = has_modality_stats(checkpoint_stats)
    if config.use_relative_actions and not checkpoint_has_stats:
        relative_dataset_stats = dataset_stats
        if not _stats_preserve_action_horizon(relative_dataset_stats):
            relative_dataset_stats = _make_relative_action_training_stats_from_dataset_meta(
                config, dataset_meta
            )
        relative_assets = _build_n1_7_relative_action_processor_assets(
            config,
            relative_dataset_stats,
            dataset_meta,
            base_assets=checkpoint_assets,
        )
        if relative_assets is None:
            raise ValueError(
                "GR00T relative-action training requires horizon-preserving relative action statistics. "
                "Pass dataset_meta with a local LeRobot dataset root, or pass precomputed relative dataset_stats."
            )
        checkpoint_assets = relative_assets
        checkpoint_stats = checkpoint_assets.stats
        checkpoint_has_stats = has_modality_stats(checkpoint_stats)

    action_horizon = (
        checkpoint_assets.max_action_horizon
        if checkpoint_assets is not None and checkpoint_assets.max_action_horizon is not None
        else min(config.chunk_size, N1_7_NATIVE_ACTION_HORIZON)
    )
    valid_action_horizon = (
        checkpoint_assets.valid_action_horizon
        if checkpoint_assets is not None and checkpoint_assets.valid_action_horizon is not None
        else action_horizon
    )
    padded_stats = checkpoint_stats if checkpoint_has_stats else (dataset_stats or {})
    embodiment_mapping = (
        checkpoint_assets.embodiment_mapping
        if checkpoint_assets is not None
        else dict(N1_7_EMBODIMENT_MAPPING)
    )
    formalize_language = checkpoint_assets.formalize_language if checkpoint_assets is not None else True
    clip_outliers = checkpoint_assets.clip_outliers if checkpoint_assets is not None else True
    video_modality_keys = checkpoint_assets.video_modality_keys if checkpoint_assets is not None else None
    try:
        env_action_dim = int(config.output_features[ACTION].shape[0])
    except Exception:
        env_action_dim = 0
    pack_step = GrootN17PackInputsStep(
        state_horizon=1,
        action_horizon=action_horizon,
        valid_action_horizon=valid_action_horizon,
        video_horizon=checkpoint_assets.video_horizon if checkpoint_assets is not None else None,
        max_state_dim=config.max_state_dim,
        max_action_dim=config.max_action_dim,
        language_key="task",
        formalize_language=formalize_language,
        embodiment_tag=config.embodiment_tag,
        embodiment_mapping=embodiment_mapping,
        normalize_min_max=True,
        training=dataset_meta is not None,
        state_dropout_prob=(checkpoint_assets.state_dropout_prob if checkpoint_assets is not None else 0.0),
        stats=padded_stats,
        clip_outliers=clip_outliers,
        video_modality_keys=video_modality_keys,
        raw_stats=checkpoint_assets.raw_stats if checkpoint_assets is not None else None,
        use_percentiles=checkpoint_assets.use_percentiles if checkpoint_assets is not None else False,
        modality_config=checkpoint_assets.modality_config if checkpoint_assets is not None else None,
    )

    # 确定图像预处理的几何尺寸。当检查点的 processor_config 提供了
    # image_target_size 时以它为准；否则回退到 N1.7 主干训练时所用的几何尺寸。
    # 如果没有这一回退，对于没有 processor_config 图像尺寸的原始基础检查点
    # （例如用新本体微调 nvidia/GR00T-N1.7-3B，此时 checkpoint_assets 为 None），
    # 就会对全分辨率相机帧做 patchify，导致 VLM token 数量膨胀，并向模型送入
    # 其从未训练过的分辨率。
    if checkpoint_assets is not None and checkpoint_assets.image_target_size is not None:
        image_target_size = checkpoint_assets.image_target_size
        image_crop_size = checkpoint_assets.image_crop_size
        shortest_image_edge = checkpoint_assets.shortest_image_edge
        crop_fraction = checkpoint_assets.crop_fraction
    else:
        image_target_size = list(N1_7_DEFAULT_IMAGE_TARGET_SIZE)
        image_crop_size = list(N1_7_DEFAULT_IMAGE_CROP_SIZE)
        shortest_image_edge = None
        crop_fraction = None
    use_albumentations = checkpoint_assets.use_albumentations if checkpoint_assets is not None else False
    letter_box_transform = checkpoint_assets.letter_box_transform if checkpoint_assets is not None else False

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        pack_step,
        GrootN17VLMEncodeStep(
            model_name=GROOT_N1_7_BACKBONE_MODEL,
            image_crop_size=image_crop_size,
            image_target_size=image_target_size,
            shortest_image_edge=shortest_image_edge,
            crop_fraction=crop_fraction,
            use_albumentations=use_albumentations,
            letter_box_transform=letter_box_transform,
            training=dataset_meta is not None,
            device=config.device,
        ),
        DeviceProcessorStep(device=config.device),
    ]
    uses_native_relative_actions = bool(
        checkpoint_assets is not None and checkpoint_assets.use_relative_action
    )
    relative_step: RelativeActionsProcessorStep | None = None
    if config.use_relative_actions and not uses_native_relative_actions:
        logging.warning(
            "GR00T relative actions are using the generic RelativeActionsProcessorStep fallback because "
            "the checkpoint already carries non-relative statistics. Relative deltas will be normalized "
            "with absolute action stats rather than Isaac-GR00T's per-horizon relative stats. For "
            "OSS-faithful relative normalization, build from a checkpoint without baked-in stats (or "
            "pass dataset_meta) so native relative stats are computed."
        )
        relative_step = RelativeActionsProcessorStep(
            enabled=True,
            exclude_joints=list(config.relative_exclude_joints or []),
            action_names=_resolve_action_feature_names_from_dataset_meta(dataset_meta),
        )
        input_steps.insert(2, relative_step)

    if checkpoint_assets is not None and not checkpoint_has_stats and not has_modality_stats(padded_stats):
        raise ValueError(
            f"GR00T N1.7 checkpoint '{config.base_model_path}' has no statistics for embodiment tag "
            f"'{config.embodiment_tag}', and no dataset stats were provided to fall back to, so "
            "actions cannot be normalized or decoded. Pass dataset_stats, or set "
            "config.embodiment_tag to an embodiment present in the checkpoint's statistics.json."
        )
    if checkpoint_assets is None or not checkpoint_has_stats:
        # 当检查点附属文件中没有所配置本体标签对应的统计量时
        # （例如使用默认的 'new_embodiment' 标签微调原始基础检查点），
        # 上面的打包步骤用数据集统计量做了归一化；解码步骤必须用相同的
        # 统计量做逆变换，而不能使用检查点解码器——后者的空统计量会
        # 静默返回仍处于归一化状态的 [-1, 1] 动作。
        action_decode_step: ProcessorStep = GrootActionUnpackUnnormalizeStep(
            env_action_dim=env_action_dim,
            stats=padded_stats,
            normalize_min_max=True,
            clip_normalized_action=True,
            libero_gripper_action=config.action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
        )
    else:
        action_decode_step = GrootN17ActionDecodeStep(
            env_action_dim=env_action_dim,
            raw_stats=checkpoint_assets.raw_stats,
            modality_config=checkpoint_assets.modality_config,
            use_percentiles=checkpoint_assets.use_percentiles,
            use_relative_action=checkpoint_assets.use_relative_action,
            pack_step=pack_step,
            action_decode_transform=config.action_decode_transform,
        )

    output_steps: list[ProcessorStep] = [action_decode_step]
    if relative_step is not None:
        output_steps.append(AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step))
    output_steps.append(DeviceProcessorStep(device="cpu"))

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


# GR00T 专属的处理器步骤


def _to_uint8_np_bthwc(img_t: torch.Tensor) -> np.ndarray:
    # img_t: (B, C, H, W) 或 (B, T, C, H, W)，取值在 [0,1] 的 float 或 uint8
    if img_t.dtype.is_floating_point:
        img_t = (img_t.clamp(0, 1) * 255.0).to(torch.uint8)
    if img_t.dim() == 4:
        return rearrange(img_t.cpu().numpy(), "b c h w -> b 1 h w c")
    if img_t.dim() == 5:
        return rearrange(img_t.cpu().numpy(), "b t c h w -> b t h w c")
    raise ValueError(f"Expected image tensor shape (B, C, H, W) or (B, T, C, H, W), got {tuple(img_t.shape)}")


def _align_video_horizon(video: np.ndarray, horizon: int | None) -> np.ndarray:
    """通过截断帧或在左侧填充帧，使视频时域与检查点要求的一致。"""

    if horizon is None or horizon <= 0:
        return video
    current = video.shape[1]
    if current == horizon:
        return video
    if current > horizon:
        return video[:, -horizon:]
    pad = np.repeat(video[:, :1], horizon - current, axis=1)
    return np.concatenate([pad, video], axis=1)


def _build_n1_7_processor(model_name: str = GROOT_N1_7_BACKBONE_MODEL) -> ProcessorMixin:
    require_package("transformers", extra="groot")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    image_processor = Qwen2VLImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    video_processor = Qwen3VLVideoProcessor.from_pretrained(model_name, trust_remote_code=True)
    proc = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )
    proc.tokenizer.padding_side = "left"
    return proc


def _transform_n1_7_image_for_vlm_albumentations(
    image: np.ndarray,
    *,
    image_crop_size: list[int] | None,
    image_target_size: list[int] | None,
    shortest_image_edge: int | None,
    crop_fraction: float | None,
    letter_box_transform: bool = False,
    crop_position: tuple[float, float] | None = None,
) -> np.ndarray:
    """复刻 Isaac-GR00T albumentations 预处理的 cv2/INTER_AREA 评估变换。

    仅用于以 ``use_albumentations=True`` 保存的检查点。cv2 只能在
    CPU/numpy 上运行，因此该路径无法在 GPU 上执行；默认的（非
    albumentations）几何处理由 :func:`_transform_n1_7_image_for_vlm_torch`
    在设备上完成。这里的 cv2/INTER_AREA 缩放和向下取整的中心裁剪有意与
    那条 torch 路径不同，并且必须与上游参考实现逐位一致。该热点路径
    接收并返回 numpy 数组，以避免逐帧的 PIL 来回转换。

    ``crop_position`` 选择 ``crop_fraction`` 窗口的位置：``None`` 保持
    确定性的中心裁剪（评估时的约定），而 [0, 1] 内的 ``(y, x)`` 比例值
    则为 Isaac 训练时随机裁剪放置窗口（(0.5, 0.5) 即中心）。训练时每个
    样本采样一个位置，并在所有相机视角间复用。
    """
    if image_target_size is None:
        return image

    target_h, target_w = image_target_size

    image_np = np.asarray(image)
    if image_np.ndim == 2:
        image_np = np.repeat(image_np[:, :, None], 3, axis=2)
    elif image_np.ndim == 3 and image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]

    if not image_np.flags.c_contiguous:
        image_np = np.ascontiguousarray(image_np)

    if letter_box_transform:
        height, width = image_np.shape[:2]
        if height != width:
            square_edge = max(height, width)
            pad_h = square_edge - height
            pad_w = square_edge - width
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            image_np = cv2.copyMakeBorder(image_np, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    resize_edge = shortest_image_edge or target_h

    def resize_shortest_edge(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        shortest_edge = min(height, width)
        if shortest_edge == resize_edge:
            return frame
        scale = resize_edge / float(shortest_edge)
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))
        return cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    image_np = resize_shortest_edge(image_np)

    if crop_fraction is None and image_crop_size is not None:
        crop_fraction = image_crop_size[0] / float(target_h)
    if crop_fraction is not None and 0.0 < crop_fraction < 1.0:
        height, width = image_np.shape[:2]
        crop_h = max(1, int(height * crop_fraction))
        crop_w = max(1, int(width * crop_fraction))
        if crop_position is None:
            top = max(0, (height - crop_h) // 2)
            left = max(0, (width - crop_w) // 2)
        else:
            pos_y, pos_x = crop_position
            top = int(round((height - crop_h) * min(max(pos_y, 0.0), 1.0)))
            left = int(round((width - crop_w) * min(max(pos_x, 0.0), 1.0)))
        image_np = image_np[top : top + crop_h, left : left + crop_w]

    return resize_shortest_edge(image_np)


def _transform_n1_7_image_for_vlm_torch(
    image: torch.Tensor,
    *,
    image_crop_size: list[int] | None,
    image_target_size: list[int] | None,
    shortest_image_edge: int | None,
    crop_fraction: float | None,
    letter_box_transform: bool = False,
) -> torch.Tensor:
    """默认的（非 albumentations）N1.7 图像变换。

    可选地先填充为正方形，然后缩放到 ``shortest_image_edge``，按
    ``crop_fraction`` 做中心裁剪，最后缩放到 ``image_target_size``。

    该函数处理 ``(C, H, W)`` uint8 张量，并将结果保留在输入张量所在的
    设备上，因此当张量在 GPU 上时缩放/裁剪也在 GPU 上运行。带抗锯齿的
    双三次插值与 PIL 的 ``Image.Resampling.BICUBIC`` 非常接近（最坏输入下
    逐像素误差小于 ``2/255``）。``use_albumentations`` 的 cv2/INTER_AREA
    路径没有对应的 torch 实现，仍保留在
    :func:`_transform_n1_7_image_for_vlm_albumentations` 中。
    """
    if image_target_size is None:
        return image

    target_h, target_w = image_target_size
    _, height, width = image.shape

    if letter_box_transform:
        square_edge = max(height, width)
        if height != width:
            left = (square_edge - width) // 2
            top = (square_edge - height) // 2
            image = tv_functional.pad(
                image, [left, top, square_edge - width - left, square_edge - height - top], fill=0
            )

    resize_edge = shortest_image_edge or target_h
    image = tv_functional.resize(
        image, [resize_edge, resize_edge], interpolation=InterpolationMode.BICUBIC, antialias=True
    )

    if crop_fraction is None and image_crop_size is not None:
        crop_fraction = image_crop_size[0] / float(target_h)
    if crop_fraction is not None and 0.0 < crop_fraction < 1.0:
        # 与 PIL 辅助函数的中心裁剪严格保持一致：裁剪尺寸用 round()，
        # 但偏移量用 floor()（torchvision.center_crop 会对偏移量四舍五入，
        # 当 (edge - crop) 为奇数时会使裁剪区域偏移 1px）。
        crop_h = max(1, int(round(image.shape[-2] * crop_fraction)))
        crop_w = max(1, int(round(image.shape[-1] * crop_fraction)))
        top = max(0, (image.shape[-2] - crop_h) // 2)
        left = max(0, (image.shape[-1] - crop_w) // 2)
        image = image[..., top : top + crop_h, left : left + crop_w]

    if tuple(image.shape[-2:]) != (target_h, target_w):
        image = tv_functional.resize(
            image, [target_h, target_w], interpolation=InterpolationMode.BICUBIC, antialias=True
        )
    return image


@dataclass
@ProcessorStepRegistry.register(name="groot_n1_7_pack_inputs_v1")
class GrootN17PackInputsStep(ProcessorStep):
    """将 LeRobot transition 打包为 N1.7 所期望的原始张量布局。

    在 Qwen3-VL 处理器接触样本之前，本步骤保留检查点的相机顺序、视频时域、
    语言格式、归一化统计量、动作掩码语义以及本体 id 映射。
    """

    state_horizon: int = 1
    action_horizon: int = N1_7_NATIVE_ACTION_HORIZON
    valid_action_horizon: int = N1_7_NATIVE_ACTION_HORIZON
    video_horizon: int | None = None
    max_state_dim: int = 132
    max_action_dim: int = 132
    language_key: str = "task"
    formalize_language: bool = True
    embodiment_tag: str = "new_embodiment"
    embodiment_mapping: dict[str, int] = field(default_factory=lambda: dict(N1_7_EMBODIMENT_MAPPING))
    normalize_min_max: bool = True
    training: bool = False
    state_dropout_prob: float = 0.0
    stats: dict[str, dict[str, Any]] | None = None
    clip_outliers: bool = True
    use_percentiles: bool = False
    video_modality_keys: list[str] | None = None
    raw_stats: dict[str, Any] | None = None
    modality_config: dict[str, Any] | None = None
    _last_raw_state: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _warned_image_keys: bool = field(default=False, init=False, repr=False)

    def _ordered_image_keys(self, obs: dict[str, Any]) -> list[str]:
        available = {key for key in obs if key.startswith(OBS_IMAGES)}
        if not available and OBS_IMAGE in obs:
            return [OBS_IMAGE]
        if not self.video_modality_keys:
            return sorted(available)

        ordered: list[str] = []
        unmatched: list[str] = []
        for modality_key in self.video_modality_keys:
            candidates = [f"{OBS_IMAGES}.{modality_key}"]
            # 针对使用通用相机名转换的数据集的别名（例如 LIBERO 转换后
            # 腕部相机为 `observation.images.image2`），这样原始的 N1.7
            # LIBERO 检查点可以直接匹配这些数据集。
            if modality_key == "wrist_image":
                candidates.append(f"{OBS_IMAGES}.image2")

            match = next((candidate for candidate in candidates if candidate in available), None)
            if match is None:
                unmatched.append(modality_key)
            else:
                ordered.append(match)

        if not ordered:
            if not self._warned_image_keys:
                self._warned_image_keys = True
                logging.warning(
                    "None of the GR00T N1.7 checkpoint video modality keys %s match a camera among %s; "
                    "falling back to feeding all cameras in alphabetical order, which is unlikely to be "
                    "the layout the checkpoint was trained with. Rename the dataset cameras (e.g. via "
                    "--rename_map) to match the checkpoint keys.",
                    self.video_modality_keys,
                    sorted(available),
                )
            return sorted(available)
        unused = sorted(available - set(ordered))
        if (unmatched or unused) and not self._warned_image_keys:
            self._warned_image_keys = True
            if unmatched:
                logging.warning(
                    "GR00T N1.7 checkpoint video modality keys %s have no matching camera among %s; "
                    "the model will receive %d view(s) instead of the %d it was trained with. Rename "
                    "the dataset cameras (e.g. via --rename_map) to match the checkpoint keys %s.",
                    unmatched,
                    sorted(available),
                    len(ordered),
                    len(self.video_modality_keys),
                    self.video_modality_keys,
                )
            if unused:
                logging.warning(
                    "Dropping camera(s) %s: the GR00T N1.7 checkpoint only consumes the video modality "
                    "keys %s, which matched %s.",
                    unused,
                    self.video_modality_keys,
                    ordered,
                )
        return ordered

    def _state_groups_from_tensor(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.modality_config is None or self.raw_stats is None:
            return {}
        state_config = self.modality_config.get("state", {})
        if not isinstance(state_config, dict):
            return {}
        state_keys = state_config.get("modality_keys", [])
        if not isinstance(state_keys, list):
            return {}

        grouped: dict[str, torch.Tensor] = {}
        start_idx = 0
        for key in state_keys:
            if not isinstance(key, str):
                continue
            key_stats = self.raw_stats.get("state", {}).get(key, {})
            dim = stat_dim_from_entry(key_stats) if isinstance(key_stats, dict) else 0
            if dim <= 0:
                continue
            grouped[key] = state[:, start_idx : start_idx + dim]
            start_idx += dim
        return grouped

    def _convert_relative_action_groups_for_training(
        self, action: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        if self.modality_config is None or self.raw_stats is None:
            return action

        action_config = self.modality_config.get("action", {})
        if not isinstance(action_config, dict):
            return action
        action_keys = action_config.get("modality_keys", [])
        action_configs = action_config.get("action_configs", [])
        if not isinstance(action_keys, list) or not isinstance(action_configs, list):
            return action

        state_groups = self._state_groups_from_tensor(state)
        if not state_groups:
            return action

        converted = action
        start_idx = 0
        cloned = False
        for idx, key in enumerate(action_keys):
            if not isinstance(key, str):
                continue
            key_stats = self.raw_stats.get("action", {}).get(key, {})
            dim = stat_dim_from_entry(key_stats) if isinstance(key_stats, dict) else 0
            if dim <= 0:
                continue
            end_idx = start_idx + dim
            if end_idx > action.shape[-1]:
                break

            cfg = (
                action_configs[idx]
                if idx < len(action_configs) and isinstance(action_configs[idx], dict)
                else {}
            )
            if config_value(cfg.get("rep")) == "relative":
                action_type = config_value(cfg.get("type"))
                if action_type != "non_eef":
                    raise ValueError(f"Unsupported relative N1.7 action config for '{key}': {cfg}")
                state_key = cfg.get("state_key") or key
                reference = state_groups.get(state_key)
                if reference is None:
                    raise KeyError(f"Missing raw state group '{state_key}' for relative N1.7 action '{key}'")
                if reference.shape[-1] != dim:
                    raise ValueError(
                        f"Relative N1.7 action group '{key}' has dim {dim}, but state group "
                        f"'{state_key}' has dim {reference.shape[-1]}."
                    )
                if not cloned:
                    converted = action.clone()
                    cloned = True
                converted[..., start_idx:end_idx] -= reference[:, None, :]

            start_idx = end_idx

        return converted

    def _normalize_action_groups_for_training(self, action: torch.Tensor) -> torch.Tensor | None:
        if self.modality_config is None or self.raw_stats is None:
            return None

        action_config = self.modality_config.get("action", {})
        if not isinstance(action_config, dict):
            return None
        action_keys = action_config.get("modality_keys", [])
        action_configs = action_config.get("action_configs", [])
        if not isinstance(action_keys, list) or not isinstance(action_configs, list):
            return None

        normalized_groups: list[torch.Tensor] = []
        start_idx = 0
        for idx, key in enumerate(action_keys):
            if not isinstance(key, str):
                continue
            cfg = (
                action_configs[idx]
                if idx < len(action_configs) and isinstance(action_configs[idx], dict)
                else {}
            )
            is_relative = config_value(cfg.get("rep")) == "relative"
            stats_modality = "relative_action" if is_relative else "action"
            key_stats = self.raw_stats.get(stats_modality, {}).get(key, {})
            dim = stat_dim_from_entry(key_stats) if isinstance(key_stats, dict) else 0
            if dim <= 0:
                continue
            end_idx = start_idx + dim
            if end_idx > action.shape[-1]:
                return None

            min_v, max_v = _n1_7_decode_stats_for_action(
                self.raw_stats,
                key,
                cfg,
                use_relative_action=True,
                use_percentiles=self.use_percentiles,
            )
            group = action[..., start_idx:end_idx]
            min_t = torch.as_tensor(min_v, dtype=group.dtype, device=group.device)
            max_t = torch.as_tensor(max_v, dtype=group.dtype, device=group.device)
            if min_t.ndim == 1:
                min_t = min_t.view(1, 1, -1)
                max_t = max_t.view(1, 1, -1)
            elif min_t.ndim == 2:
                if group.shape[1] > min_t.shape[0]:
                    return None
                min_t = min_t[: group.shape[1]].unsqueeze(0)
                max_t = max_t[: group.shape[1]].unsqueeze(0)
            else:
                return None

            denom = max_t - min_t
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            normalized = torch.where(mask, 2 * (group - min_t) / safe_denom - 1, torch.zeros_like(group))
            if self.clip_outliers:
                normalized = normalized.clamp(-1.0, 1.0)
            normalized_groups.append(normalized)
            start_idx = end_idx

        if not normalized_groups or start_idx != action.shape[-1]:
            return None
        return torch.cat(normalized_groups, dim=-1)

    def _uses_relative_action_groups(self) -> bool:
        """当动作模态声明了至少一个相对分组时返回 True。

        相对分组使用按分块时间步排列的（二维）``relative_action`` 统计量进行
        归一化，扁平的 ``_min_max_norm`` 回退逻辑无法正确处理这一点，因此
        当相对配置的分组归一化失败时，必须明确报错，而不是静默地对每个
        时间步做出错误的缩放。
        """
        if not isinstance(self.modality_config, dict):
            return False
        action_config = self.modality_config.get("action", {})
        if not isinstance(action_config, dict):
            return False
        action_configs = action_config.get("action_configs", [])
        if not isinstance(action_configs, list):
            return False
        return any(
            isinstance(cfg, dict) and config_value(cfg.get("rep")) == "relative" for cfg in action_configs
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        raw_state_for_action: torch.Tensor | None = None

        def _align_vec(vec: Any, target_dim: int, *, default: float) -> torch.Tensor:
            t = torch.as_tensor(vec)
            t = t.flatten().to(
                dtype=torch.float32,
                device=next(
                    (v.device for v in obs.values() if isinstance(v, torch.Tensor)), torch.device("cpu")
                ),
            )
            d = int(t.shape[-1]) if t.numel() > 0 else 0
            if d == target_dim:
                return t
            if d < target_dim:
                pad = torch.full((target_dim - d,), default, dtype=t.dtype, device=t.device)
                return torch.cat([t, pad], dim=0)
            return t[:target_dim]

        def _min_max_norm(x: torch.Tensor, key: str) -> torch.Tensor:
            if not self.normalize_min_max or self.stats is None or key not in self.stats:
                return x
            stats_k = self.stats[key]
            last_dim = x.shape[-1]
            min_v = _align_vec(stats_k.get("min", torch.zeros(last_dim)), last_dim, default=0.0)
            max_v = _align_vec(stats_k.get("max", torch.ones(last_dim)), last_dim, default=1.0)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            mapped = 2 * (x - min_v) / safe_denom - 1
            normalized = torch.where(mask, mapped, torch.zeros_like(mapped))
            if self.clip_outliers:
                normalized = normalized.clamp(-1.0, 1.0)
            return normalized

        def _cache_raw_state(state: torch.Tensor) -> None:
            if self.modality_config is None or self.raw_stats is None:
                return
            state_config = self.modality_config.get("state", {})
            if not isinstance(state_config, dict):
                return
            state_keys = state_config.get("modality_keys", [])
            if not isinstance(state_keys, list):
                return

            raw_state = state.detach().cpu().float().numpy()
            start_idx = 0
            grouped: dict[str, np.ndarray] = {}
            for key in state_keys:
                if not isinstance(key, str):
                    continue
                key_stats = self.raw_stats.get("state", {}).get(key, {})
                dim = len(key_stats.get("mean") or key_stats.get("min") or key_stats.get("q01") or [])
                if dim <= 0:
                    continue
                grouped[key] = raw_state[:, start_idx : start_idx + dim]
                start_idx += dim
            if grouped:
                self._last_raw_state = grouped

        img_keys = self._ordered_image_keys(obs)
        if img_keys:
            cams = [_align_video_horizon(_to_uint8_np_bthwc(obs[k]), self.video_horizon) for k in img_keys]
            video = np.stack(cams, axis=2)  # (B, T, V, H, W, C)
            obs["video"] = video
            image_keys_to_remove = [key for key in obs if key.startswith(OBS_IMAGES)]
            if OBS_IMAGE in obs:
                image_keys_to_remove.append(OBS_IMAGE)
            for k in image_keys_to_remove:
                obs.pop(k, None)

        bsz, _device = infer_n1_7_batch_size_and_device(obs, transition.get(TransitionKey.ACTION))
        comp["language"] = prepare_n1_7_language_batch(
            comp.get(self.language_key),
            bsz,
            formalize_language=self.formalize_language,
        )

        if OBS_STATE in obs:
            state = obs[OBS_STATE]
            if state.dim() != 2:
                raise ValueError(f"state must be (B, D), got {tuple(state.shape)}")
            bsz, dim = state.shape
            if dim > self.max_state_dim:
                raise ValueError(f"State dimension {dim} exceeds max_state_dim {self.max_state_dim}.")
            _cache_raw_state(state)
            raw_state_for_action = state
            if self.normalize_min_max:
                state = _min_max_norm(state, OBS_STATE)
            state = state.unsqueeze(1)
            if dim < self.max_state_dim:
                pad = torch.zeros(bsz, 1, self.max_state_dim - dim, dtype=state.dtype, device=state.device)
                state = torch.cat([state, pad], dim=2)
            if self.training and torch.is_grad_enabled() and self.state_dropout_prob > 0:
                drop_state = torch.tensor(
                    [random.random() < self.state_dropout_prob for _ in range(bsz)],
                    dtype=torch.bool,
                    device=state.device,
                ).view(bsz, 1, 1)
                state = state.masked_fill(drop_state, 0)
            obs["state"] = state

        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            if action.dim() == 2:
                action = action.unsqueeze(1)
            elif action.dim() == 3:
                pass
            else:
                raise ValueError(f"action must be (B, D) or (B, T, D), got {tuple(action.shape)}")

            bsz, horizon, dim = action.shape
            if horizon > self.action_horizon:
                raise ValueError(f"Action horizon {horizon} exceeds action_horizon {self.action_horizon}.")
            if dim > self.max_action_dim:
                raise ValueError(f"Action dimension {dim} exceeds max_action_dim {self.max_action_dim}.")
            if raw_state_for_action is not None:
                action = self._convert_relative_action_groups_for_training(action, raw_state_for_action)
            if self.normalize_min_max:
                normalized_action = self._normalize_action_groups_for_training(action)
                if normalized_action is not None:
                    action = normalized_action
                elif self._uses_relative_action_groups():
                    raise ValueError(
                        "GrootN17PackInputsStep could not apply native grouped normalization to a "
                        "relative-action chunk: the action layout or horizon does not match the "
                        f"checkpoint relative_action stats (action shape {tuple(action.shape)}). The flat "
                        "min/max fallback cannot honor per-chunk-timestep relative stats, so refusing to "
                        "silently wrongly normalize. Recompute the relative action stats so their horizon and "
                        "dimensions match the action chunk."
                    )
                else:
                    flat = _min_max_norm(action.reshape(bsz * horizon, dim), ACTION)
                    action = flat.view(bsz, horizon, dim)
            valid_dim = min(dim, self.max_action_dim)
            valid_horizon = min(horizon, self.valid_action_horizon, self.action_horizon)
            if dim < self.max_action_dim:
                pad = torch.zeros(
                    bsz, horizon, self.max_action_dim - dim, dtype=action.dtype, device=action.device
                )
                action = torch.cat([action, pad], dim=2)
            if horizon < self.action_horizon:
                pad = torch.zeros(
                    bsz,
                    self.action_horizon - horizon,
                    self.max_action_dim,
                    dtype=action.dtype,
                    device=action.device,
                )
                action = torch.cat([action, pad], dim=1)
                horizon = self.action_horizon
            horizon_valid = torch.zeros(bsz, horizon, dtype=torch.bool, device=action.device)
            horizon_valid[:, :valid_horizon] = True
            action_is_pad = comp.get(f"{ACTION}_is_pad")
            if action_is_pad is None:
                action_is_pad = comp.get("action_horizon_is_pad")
            if action_is_pad is not None:
                action_pad = torch.as_tensor(action_is_pad, dtype=torch.bool, device=action.device)
                if action_pad.ndim == 1:
                    if bsz == 1 and action_pad.numel() == horizon:
                        action_pad = action_pad.unsqueeze(0)
                    elif horizon == 1 and action_pad.numel() == bsz:
                        action_pad = action_pad.view(bsz, 1)
                if action_pad.ndim != 2 or action_pad.shape[0] != bsz:
                    raise ValueError(
                        "action_is_pad must have shape (B, T) matching the action batch; "
                        f"got {tuple(action_pad.shape)} for action {tuple(action.shape)}."
                    )
                pad_horizon = min(horizon, action_pad.shape[1])
                horizon_valid[:, :pad_horizon] &= ~action_pad[:, :pad_horizon]

            if valid_horizon < horizon or action_is_pad is not None:
                action = action.clone()
                action[:, valid_horizon:, :] = 0
                action = action * horizon_valid.unsqueeze(-1).to(dtype=action.dtype)
            action_mask = torch.zeros(
                bsz, horizon, self.max_action_dim, dtype=torch.float32, device=action.device
            )
            action_mask[:, :, :valid_dim] = horizon_valid.unsqueeze(-1).to(dtype=action_mask.dtype)
            transition[TransitionKey.ACTION] = action
            comp["action_mask"] = action_mask

        emb_id = self.embodiment_mapping.get(self.embodiment_tag, 0)
        bsz, device = infer_n1_7_batch_size_and_device(obs, transition.get(TransitionKey.ACTION))
        if "action_mask" not in comp:
            action_mask = torch.zeros(bsz, self.action_horizon, dtype=torch.float32, device=device)
            valid_horizon = min(self.valid_action_horizon, self.action_horizon)
            action_mask[:, :valid_horizon] = 1.0
            comp["action_mask"] = action_mask
        comp["embodiment_id"] = torch.full((bsz,), emb_id, dtype=torch.int32, device=device)

        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "state_horizon": self.state_horizon,
            "action_horizon": self.action_horizon,
            "valid_action_horizon": self.valid_action_horizon,
            "video_horizon": self.video_horizon,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "language_key": self.language_key,
            "formalize_language": self.formalize_language,
            "embodiment_tag": self.embodiment_tag,
            "embodiment_mapping": self.embodiment_mapping,
            "normalize_min_max": self.normalize_min_max,
            "state_dropout_prob": self.state_dropout_prob,
            "clip_outliers": self.clip_outliers,
            "use_percentiles": self.use_percentiles,
            "video_modality_keys": self.video_modality_keys,
            "raw_stats": self.raw_stats,
            "modality_config": self.modality_config,
        }

    def get_cached_raw_state(self) -> dict[str, np.ndarray] | None:
        """返回最近一次未经归一化的状态，按检查点模态键拆分。"""

        return self._last_raw_state

    def state_dict(self) -> dict[str, torch.Tensor]:
        if not self.stats:
            return {}

        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                flat[f"{key}.{stat_name}"] = torch.as_tensor(value).cpu()
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if not state:
            return
        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                reconstructed.setdefault(key, {})[stat_name] = tensor
        if reconstructed:
            self.stats = reconstructed


@dataclass
@ProcessorStepRegistry.register(name="groot_n1_7_vlm_encode_v1")
class GrootN17VLMEncodeStep(ProcessorStep):
    """使用 Qwen3-VL 处理器对 N1.7 打包好的视频-语言 prompt 进行分词。

    打包后的视频形状为 ``(B, T, V, H, W, C)``。每一帧/每个视角都会成为
    同一条聊天消息中的一个图像项，从而使生成的图像 token 与 Isaac-GR00T
    使用的时序 VLM 打包方式一致。

    图像以 ``(C, H, W)`` uint8 张量的形式交给基于 torchvision 的 Qwen3-VL
    处理器（没有逐帧的 PIL 来回转换），并且当 ``device`` 解析为 CUDA 设备时，
    缩放/重缩放/归一化/patchify 都在该设备上运行。这使得输出在 CPU 上
    保持逐位一致，同时把主要的预处理开销从 GPU 的关键路径上移走。
    """

    model_name: str = GROOT_N1_7_BACKBONE_MODEL
    image_crop_size: list[int] | None = None
    image_target_size: list[int] | None = None
    shortest_image_edge: int | None = None
    crop_fraction: float | None = None
    use_albumentations: bool = False
    letter_box_transform: bool = False
    # 仅运行时的训练/评估模式：True 启用 Isaac 训练时的随机裁剪
    # （每个样本一个窗口，并在各视角间复用）；False 保持确定性的中心裁剪。
    # 该字段不会被序列化——重新加载的流水线默认为评估模式，只有在使用
    # dataset_meta 构建处理器时才会重新启用。
    training: bool = False
    device: str | None = None
    _proc: ProcessorMixin | None = field(default=None, init=False, repr=False)

    @property
    def proc(self) -> ProcessorMixin:
        if self._proc is None:
            self._proc = _build_n1_7_processor(self.model_name)
        return self._proc

    def _target_device(self) -> torch.device | None:
        # albumentations 路径只能使用 cv2/numpy，因此无法在 GPU 上运行。
        if self.device is None or self.use_albumentations:
            return None
        try:
            return get_safe_torch_device(self.device)
        except (AssertionError, RuntimeError):
            # 训练时序列化的设备（例如 "cuda"）在处理器被重新加载到其他地方时
            # （例如仅用 CPU 评估）可能不可用，而本步骤不在标准的设备覆写
            # 集合中。此时回退到逐位一致的 CPU 路径，而不是直接崩溃。
            return None

    def _build_sample_images(
        self, video: Any, batch_size: int, target_device: torch.device | None
    ) -> list[list[Any]]:
        """按批量中的每个样本返回其有序的 ``(timestep, view)`` 帧。

        ``use_albumentations`` 保留旧版逐帧的 cv2/INTER_AREA 变换；
        否则帧为 ``(C, H, W)`` uint8 张量（设置了 ``target_device`` 时
        会移动到该设备），供基于 torchvision 的 Qwen 处理器使用。
        """
        if self.use_albumentations:
            video_np = np.asarray(video)
            train_crop = self.training and torch.is_grad_enabled()
            sample_images: list[list[Any]] = []
            for batch_idx in range(batch_size):
                # Isaac-GR00T 每个样本只采样一个裁剪窗口，并在该样本的
                # 每个 (timestep, view) 帧上复用，从而保持跨视角几何一致。
                # 评估时保持中心裁剪。
                crop_position = (random.random(), random.random()) if train_crop else None
                sample_images.append(
                    [
                        _transform_n1_7_image_for_vlm_albumentations(
                            video_np[batch_idx, timestep, view_idx],
                            image_crop_size=self.image_crop_size,
                            image_target_size=self.image_target_size,
                            shortest_image_edge=self.shortest_image_edge,
                            crop_fraction=self.crop_fraction,
                            letter_box_transform=self.letter_box_transform,
                            crop_position=crop_position,
                        )
                        for timestep in range(video_np.shape[1])
                        for view_idx in range(video_np.shape[2])
                    ]
                )
            return sample_images

        video_t = video if torch.is_tensor(video) else torch.from_numpy(np.ascontiguousarray(video))
        # (B, T, V, H, W, C) uint8 -> (B, T, V, C, H, W)
        video_t = video_t.permute(0, 1, 2, 5, 3, 4).contiguous()
        if target_device is not None and video_t.device != target_device:
            video_t = video_t.to(target_device, non_blocking=(target_device.type == "cuda"))

        frames_per_sample: list[list[Any]] = []
        for batch_idx in range(batch_size):
            sample = video_t[batch_idx]  # (T, V, C, H, W)
            frames_per_sample.append(
                [
                    _transform_n1_7_image_for_vlm_torch(
                        sample[timestep, view_idx],
                        image_crop_size=self.image_crop_size,
                        image_target_size=self.image_target_size,
                        shortest_image_edge=self.shortest_image_edge,
                        crop_fraction=self.crop_fraction,
                        letter_box_transform=self.letter_box_transform,
                    )
                    for timestep in range(sample.shape[0])
                    for view_idx in range(sample.shape[1])
                ]
            )
        return frames_per_sample

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        video = obs.get("video")
        if video is None:
            return transition

        batch_size = int(video.shape[0])
        languages = prepare_n1_7_language_batch(
            comp.get("language"),
            batch_size,
            formalize_language=False,
        )

        target_device = self._target_device()
        sample_images = self._build_sample_images(video, batch_size, target_device)

        texts: list[str] = []
        images: list[Any] = []
        for batch_idx in range(batch_size):
            frames = sample_images[batch_idx]
            conversation = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in frames],
                        {"type": "text", "text": languages[batch_idx]},
                    ],
                }
            ]
            texts.append(
                self.proc.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            images.extend(frames)

        proc_kwargs: dict[str, Any] = {
            "text": texts,
            "images": images,
            "return_tensors": "pt",
            "padding": True,
        }
        if target_device is not None:
            proc_kwargs["device"] = str(target_device)
        encoded = self.proc(**proc_kwargs)
        for key, value in encoded.items():
            comp[key] = value
        obs.pop("video", None)
        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "image_crop_size": self.image_crop_size,
            "image_target_size": self.image_target_size,
            "shortest_image_edge": self.shortest_image_edge,
            "crop_fraction": self.crop_fraction,
            "use_albumentations": self.use_albumentations,
            "letter_box_transform": self.letter_box_transform,
            "device": self.device,
        }


def _n1_7_decode_stats_for_action(
    raw_stats: dict[str, Any],
    key: str,
    action_config: dict[str, Any],
    *,
    use_relative_action: bool,
    use_percentiles: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """选取解码一个检查点动作分组所需的 min/max 数组。"""

    is_relative = use_relative_action and config_value(action_config.get("rep")) == "relative"
    modality = "relative_action" if is_relative else "action"
    stats = raw_stats.get(modality, {}).get(key, {})
    if not isinstance(stats, dict):
        raise KeyError(f"Missing N1.7 statistics for {modality}.{key}")
    min_name = "min" if is_relative else ("q01" if use_percentiles else "min")
    max_name = "max" if is_relative else ("q99" if use_percentiles else "max")
    if min_name not in stats or max_name not in stats:
        raise KeyError(f"Missing '{min_name}'/'{max_name}' statistics for {modality}.{key}")
    return np.asarray(stats[min_name], dtype=np.float32), np.asarray(stats[max_name], dtype=np.float32)


def _unnormalize_min_max(action: np.ndarray, min_v: np.ndarray, max_v: np.ndarray) -> np.ndarray:
    return (np.clip(action, -1.0, 1.0) + 1.0) * 0.5 * (max_v - min_v) + min_v


def _n1_7_decode_valid_horizon(action_config: dict[str, Any], action_np: np.ndarray) -> int | None:
    if action_np.ndim != 3:
        return None
    delta_indices = action_config.get("delta_indices", [])
    if not isinstance(delta_indices, list) or not delta_indices:
        return None
    return max(1, min(action_np.shape[1], len(delta_indices)))


def _n1_7_action_group_slice(
    action_keys: list[Any], decoded_groups: dict[str, np.ndarray], target_key: str
) -> slice:
    start_idx = 0
    for key in action_keys:
        if not isinstance(key, str) or key not in decoded_groups:
            continue
        dim = decoded_groups[key].shape[-1]
        end_idx = start_idx + dim
        if key == target_key:
            return slice(start_idx, end_idx)
        start_idx = end_idx

    raise KeyError(f"Missing N1.7 action group '{target_key}' required by action decode transform.")


def _apply_n1_7_action_decode_transform(
    decoded: np.ndarray,
    *,
    transform: str | None,
    action_keys: list[Any],
    decoded_groups: dict[str, np.ndarray],
) -> np.ndarray:
    if transform is None:
        return decoded

    if transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO:
        gripper_slice = _n1_7_action_group_slice(action_keys, decoded_groups, "gripper")
        if gripper_slice.stop is None or gripper_slice.stop > decoded.shape[-1]:
            raise ValueError(
                "N1.7 LIBERO action decode transform requested, but the decoded gripper action "
                "is outside the sliced environment action."
            )
        if gripper_slice.stop - gripper_slice.start != 1:
            raise ValueError("N1.7 LIBERO action decode transform expects a scalar gripper action.")

        transformed = decoded.copy()
        gripper = transformed[..., gripper_slice]
        transformed[..., gripper_slice] = -np.sign(2.0 * gripper - 1.0)
        return transformed

    raise ValueError(f"Unsupported N1.7 action decode transform '{transform}'.")


@dataclass
@ProcessorStepRegistry.register(name="groot_n1_7_action_decode_v1")
class GrootN17ActionDecodeStep(ProcessorStep):
    """将完整的 132 维 N1.7 模型动作解码回环境动作。

    N1.7 预测的是按检查点顺序排列的动作分组。本步骤用检查点统计量对每个
    分组做反归一化，利用打包时缓存的原始状态将相对分组转换为绝对值，按
    检查点顺序拼接各分组，最后切片到环境动作维度。

    相对动作解码从已关联的 ``pack_step``（在 ``from_pretrained`` 之后由
    ``_reconnect_groot_n1_7_pack_decode_steps`` 重新关联）读取参考状态，
    即最近一次预处理调用所见到的状态。因此，在预测后立即解码整个分块的
    引擎（RTC、异步策略服务器）使用的是预测时的状态，与 Isaac-GR00T 一致。
    而同步的逐步骤队列路径则针对最新观测解码每个弹出的 (B, D) 动作：
    参考状态可能比分块预测时所用的观测更新，并且逐时间步的相对统计量会
    被当作弹出动作处于分块第 0 步来应用。要修正这一点，需要在后处理过程中
    让每个入队动作都携带参考状态和分块索引。
    """

    env_action_dim: int = 0
    raw_stats: dict[str, Any] | None = None
    modality_config: dict[str, Any] | None = None
    use_percentiles: bool = False
    use_relative_action: bool = False
    action_decode_transform: str | None = None
    pack_step: GrootN17PackInputsStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition
        if self.raw_stats is None or self.modality_config is None:
            return transition

        action_config = self.modality_config.get("action", {})
        if not isinstance(action_config, dict):
            return transition
        action_keys = action_config.get("modality_keys", [])
        action_configs = action_config.get("action_configs", [])
        if not isinstance(action_keys, list) or not isinstance(action_configs, list):
            return transition

        action_np = action.detach().cpu().float().numpy()
        if self.use_relative_action and action_np.ndim != 3:
            raise NotImplementedError(
                "GrootN17ActionDecodeStep cannot decode native relative actions one step at a time. "
                "Decode the full action chunk returned by predict_action_chunk while the matching "
                "GrootN17PackInputsStep state is still cached, then queue the decoded absolute actions."
            )
        # 同步动作队列将弹出的动作为 (B, D) 做后处理；这里把它们当作
        # 单步 (B, 1, D) 分块来解码，并在最后把时域维挤压回去，
        # 以便两种形态共用下面的分块解码逻辑。
        squeeze_horizon = action_np.ndim == 2
        if squeeze_horizon:
            action_np = action_np[:, None, :]
        valid_horizon = _n1_7_decode_valid_horizon(action_config, action_np)
        if valid_horizon is not None:
            action_np = action_np[:, :valid_horizon]
        decoded_groups: dict[str, np.ndarray] = {}
        start_idx = 0
        for idx, key in enumerate(action_keys):
            if not isinstance(key, str):
                continue
            stats_entry = self.raw_stats.get("action", {}).get(key, {})
            if not isinstance(stats_entry, dict):
                continue
            dim = stat_dim_from_entry(stats_entry)
            if dim <= 0:
                continue
            cfg = (
                action_configs[idx]
                if idx < len(action_configs) and isinstance(action_configs[idx], dict)
                else {}
            )
            normalized = action_np[..., start_idx : start_idx + dim]
            min_v, max_v = _n1_7_decode_stats_for_action(
                self.raw_stats,
                key,
                cfg,
                use_relative_action=self.use_relative_action,
                use_percentiles=self.use_percentiles,
            )
            # 逐时间步统计量的每个分块步骤对应一行；将其与解码时域对齐
            # （分块总是从第 0 步开始，而弹出的 (B, D) 动作按第 0 步解码）。
            if min_v.ndim == 2 and normalized.shape[1] <= min_v.shape[0]:
                min_v = min_v[: normalized.shape[1]]
                max_v = max_v[: normalized.shape[1]]
            decoded_groups[key] = _unnormalize_min_max(normalized, min_v, max_v)
            start_idx += dim

        if self.use_relative_action:
            raw_state = self.pack_step.get_cached_raw_state() if self.pack_step is not None else None
            if raw_state is None:
                raise RuntimeError(
                    "GrootN17ActionDecodeStep requires the raw state cached by its connected "
                    "GrootN17PackInputsStep to convert relative N1.7 actions back to absolute actions. "
                    "Build both pipelines through make_groot_pre_post_processors (or load them together "
                    "via make_groot_pre_post_processors_from_pretrained) and run the preprocessor on an "
                    "observation before decoding actions."
                )
            for idx, key in enumerate(action_keys):
                if not isinstance(key, str) or key not in decoded_groups or idx >= len(action_configs):
                    continue
                cfg = action_configs[idx]
                if not isinstance(cfg, dict) or config_value(cfg.get("rep")) != "relative":
                    continue
                state_key = cfg.get("state_key") or key
                if state_key not in raw_state:
                    raise KeyError(f"Missing cached raw state '{state_key}' for relative N1.7 action '{key}'")
                reference = raw_state[state_key]
                action_type = config_value(cfg.get("type"))
                action_format = config_value(cfg.get("format"))
                if action_type == "non_eef":
                    decoded_groups[key] = decoded_groups[key] + reference[:, None, :]
                elif action_type == "eef" and action_format == "xyz+rot6d":
                    decoded_groups[key] = relative_eef_to_absolute(decoded_groups[key], reference)
                else:
                    raise ValueError(f"Unsupported relative N1.7 action config for '{key}': {cfg}")

        if not decoded_groups:
            return transition

        decoded = np.concatenate(
            [decoded_groups[key] for key in action_keys if isinstance(key, str) and key in decoded_groups],
            axis=-1,
        )
        if self.env_action_dim and decoded.shape[-1] > self.env_action_dim:
            decoded = decoded[..., : self.env_action_dim]
        decoded = _apply_n1_7_action_decode_transform(
            decoded,
            transform=self.action_decode_transform,
            action_keys=action_keys,
            decoded_groups=decoded_groups,
        )
        if squeeze_horizon:
            decoded = decoded[:, 0]
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = torch.as_tensor(
            decoded, dtype=action.dtype, device=action.device
        )
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "env_action_dim": self.env_action_dim,
            "raw_stats": self.raw_stats,
            "modality_config": self.modality_config,
            "use_percentiles": self.use_percentiles,
            "use_relative_action": self.use_relative_action,
            "action_decode_transform": self.action_decode_transform,
        }


# v2：与 N1.5 时代的 v1 步骤不同，本步骤不再把 (B, T, D) 动作分块
# 坍缩到最后一个时间步，因此旧的序列化 v1 流水线绝不能被静默加载到
# 本步骤中（v1 在下方以带移除指引的存根形式占位）。
@dataclass
@ProcessorStepRegistry.register(name="groot_action_unpack_unnormalize_v2")
class GrootActionUnpackUnnormalizeStep(ProcessorStep):
    env_action_dim: int = 0
    # 若预处理器中使用了 min-max 归一化，则在此应用其逆变换
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None
    clip_normalized_action: bool = False
    libero_gripper_action: bool = False
    libero_gripper_binarize: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # 预期模型输出位于 TransitionKey.ACTION 中，形状为 (B, T, D_model)
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition

        # 切片到环境维度，同时保留可选的动作时域。
        # 同步 rollout 对所选动作按 (B, D) 后处理；RTC 对分块按 (B, T, D)
        # 后处理，与 Isaac-GR00T 的 decode_action 约定一致。
        if self.env_action_dim and action.shape[-1] >= self.env_action_dim:
            action = action[..., : self.env_action_dim]

        # 镜像 _min_max_norm 的 min-max 反归一化：
        # 正向：y = 2 * (x - min) / denom - 1，当 denom==0 时 y=0
        # 逆向：x = (y+1)/2 * denom + min，当 denom==0 时 x = min
        if self.normalize_min_max and self.stats is not None:
            if self.clip_normalized_action:
                action = action.clamp(-1.0, 1.0)
            stats_k = self.stats.get(ACTION, {})
            d = action.shape[-1]
            min_v = torch.as_tensor(
                stats_k.get("min", torch.zeros(d)), dtype=action.dtype, device=action.device
            )
            max_v = torch.as_tensor(
                stats_k.get("max", torch.ones(d)), dtype=action.dtype, device=action.device
            )
            if min_v.numel() != d:
                min_v = torch.nn.functional.pad(min_v.flatten()[:d], (0, max(0, d - min_v.numel())))
                min_v = min_v.to(action.device, dtype=action.dtype)
            if max_v.numel() != d:
                max_v = torch.nn.functional.pad(max_v.flatten()[:d], (0, max(0, d - max_v.numel())))
                max_v = max_v.to(action.device, dtype=action.dtype)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            inv = (action + 1.0) * 0.5 * safe_denom + min_v
            action = torch.where(mask, inv, min_v)

        if self.libero_gripper_action and action.shape[-1] >= 7:
            gripper = action[..., -1]
            if self.libero_gripper_binarize:
                gripper = -torch.sign(2.0 * gripper - 1.0)
            else:
                gripper = -(2.0 * gripper - 1.0)
            action = action.clone()
            action[..., -1] = gripper

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        """
        返回处理器配置的可序列化字典。

        不包含 'stats'，因为统计量通过 state_dict() 单独保存。
        """
        return {
            "env_action_dim": self.env_action_dim,
            "normalize_min_max": self.normalize_min_max,
            "clip_normalized_action": self.clip_normalized_action,
            "libero_gripper_action": self.libero_gripper_action,
            "libero_gripper_binarize": self.libero_gripper_binarize,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        以扁平 state 字典的形式返回归一化统计量。

        这使得统计量可以保存到 safetensors 文件，与 normalizer_processor 类似。
        """
        if not self.stats:
            return {}

        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                tensor = torch.as_tensor(value).cpu()
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """
        从扁平 state 字典加载归一化统计量。

        这使得在 from_pretrained 期间可以从 safetensors 文件加载统计量。
        """
        if not state:
            return

        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                if key not in reconstructed:
                    reconstructed[key] = {}
                reconstructed[key][stat_name] = tensor

        if reconstructed:
            self.stats = reconstructed


# 只有 GR00T N1.5 处理器流水线才会序列化的注册表名。已保存的 N1.5 检查点
# 会在其处理器 JSON 中引用这些名字，因此反序列化时必须以规范的 N1.5 移除
# 指引报错，而不是抛出难以理解的注册表 KeyError（对于
# ``groot_action_unpack_unnormalize_v1``，还要避免静默加载动作分块语义
# 已发生变化的 v2 步骤）。
_REMOVED_N1_5_STEP_NAMES = (
    "groot_pack_inputs_v3",
    "groot_eagle_encode_v3",
    "groot_eagle_collate_v3",
    "groot_action_unpack_unnormalize_v1",
)


def _register_removed_n1_5_step_stub(registry_name: str) -> None:
    """为已移除的 GR00T N1.5 处理器步骤名注册一个拒绝型存根。

    该操作是幂等的：``ProcessorStepRegistry.register`` 在遇到重复名字时会
    抛错，因此已注册的名字会被跳过。这样调用方在每次加载处理器时都可以
    重新执行，而无需“只运行一次”的守卫。
    """
    if registry_name in ProcessorStepRegistry.list():
        return

    @ProcessorStepRegistry.register(name=registry_name)
    class _RemovedGrootN15ProcessorStep(ProcessorStep):
        def __init__(self, **_kwargs: Any) -> None:
            raise ValueError(
                f"Processor step '{registry_name}' belongs to a GR00T N1.5 processor pipeline. "
                f"{GROOT_N1_5_REMOVAL_GUIDANCE}"
            )

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            raise NotImplementedError

        def transform_features(self, features):
            raise NotImplementedError


def _register_removed_n1_5_step_stubs() -> None:
    """惰性注册 GR00T N1.5 移除存根。

    将注册时机从导入时推迟，使导入本模块不产生全局副作用；该函数恰好在
    GR00T 处理器流水线反序列化之前调用（这是已保存的 N1.5 流水线唯一可能
    引用这些注册表名的时机）。通过 :func:`_register_removed_n1_5_step_stub`
    保证幂等。
    """
    for registry_name in _REMOVED_N1_5_STEP_NAMES:
        _register_removed_n1_5_step_stub(registry_name)
