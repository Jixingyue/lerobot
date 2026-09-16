#!/usr/bin/env python

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

import importlib
import inspect
import logging
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

import torch

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.envs import EnvConfig, env_to_policy_features
from lerobot.lerobot_types import PolicyAction
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.import_utils import _peft_available, require_package

from .evo1.configuration_evo1 import Evo1Config
from .groot.configuration_groot import GrootConfig
from .molmoact2.configuration_molmoact2 import MolmoAct2Config
from .pretrained import PreTrainedPolicy
from .utils import validate_visual_features_consistency

if TYPE_CHECKING or _peft_available:
    from peft import PeftConfig, PeftModel
else:
    PeftConfig = None
    PeftModel = None


def _reconnect_relative_absolute_steps(
    preprocessor: PolicyProcessorPipeline, postprocessor: PolicyProcessorPipeline
) -> None:
    """在反序列化后，将 AbsoluteActionsProcessorStep.relative_step 连接到 RelativeActionsProcessorStep。

    策略从磁盘加载后，预处理器和后处理器会根据各自的配置独立重建。
    AbsoluteActionsProcessorStep 需要一个对 RelativeActionsProcessorStep 的实时引用，
    以便在推理时读取缓存的状态。该引用不可序列化，因此在加载后我们在这里重新建立它。
    """
    relative_step = next((s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)), None)
    if relative_step is None:
        return
    for step in postprocessor.steps:
        if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
            step.relative_step = relative_step


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """
    根据注册的名称获取策略类。

    解析过程基于约定：查找以 ``name`` 注册的 draccus 配置类，将其
    ``configuration_*`` 模块路径改写为 ``modeling_*``，然后从中导入
    ``<X>Policy`` 类。建模模块仅在调用时导入，从而保持沉重的可选依赖
    处于延迟加载状态。这对内置策略和第三方 lerobot 插件（任何通过
    ``@PreTrainedConfig.register_subclass`` 注册的内容）都适用。

    Args:
        name: 策略的注册名称（例如 "act"、"diffusion"、"pi0"）。
    Returns:
        与给定名称对应的策略类。

    Raises:
        ValueError: 如果策略名称未注册。
        ImportError: 如果策略的可选依赖未安装。
    """
    return _get_policy_cls_from_policy_name(name=name)


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """
    根据策略类型实例化策略配置对象。

    该工厂函数通过将字符串标识符映射到对应的配置类，
    简化了策略配置对象的创建。

    Args:
        policy_type: 策略的注册类型（任何通过
                     ``@PreTrainedConfig.register_subclass`` 注册的名称，例如 "act"、"diffusion"、"pi0"）。
        **kwargs: 传递给配置类构造函数的关键字参数。

    Returns:
        `PreTrainedConfig` 子类的实例。

    Raises:
        ValueError: 如果 `policy_type` 无法识别。
    """
    try:
        config_cls = PreTrainedConfig.get_choice_class(policy_type)
    except Exception as e:
        raise ValueError(f"Policy type '{policy_type}' is not available.") from e
    return config_cls(**kwargs)


class ProcessorConfigKwargs(TypedDict, total=False):
    """
    定义处理器配置关键字参数的 TypedDict。

    它为传递给 `make_pre_post_processors` 的可选参数提供类型提示，
    提高代码清晰度并支持静态分析。

    Attributes:
        preprocessor_config_filename: 预处理器配置的文件名。
        postprocessor_config_filename: 后处理器配置的文件名。
        preprocessor_overrides: 预处理器配置的覆盖字典。
        postprocessor_overrides: 后处理器配置的覆盖字典。
        dataset_stats: 用于归一化的数据集统计信息。
    """

    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None
    dataset_meta: Any | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    pretrained_revision: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为给定策略创建或加载预处理器和后处理器流水线。

    该函数充当工厂。它既可以从预训练路径加载现有的处理器流水线，
    也可以根据策略配置从头创建新的流水线。每种策略类型都有其处理器
    专用的工厂函数（例如 `make_tdmpc_pre_post_processors`）。

    Args:
        policy_cfg: 要为其创建处理器的策略配置。
        pretrained_path: 可选路径，用于从中加载预训练的处理器流水线。
            如果提供，则从该路径加载流水线。
        **kwargs: 处理器配置的关键字参数，如
            `ProcessorConfigKwargs` 中所定义。

    Returns:
        包含输入（预处理器）和输出（后处理器）流水线的元组。

    Raises:
        ValueError: 如果给定策略配置类型不存在处理器工厂。
    """
    if pretrained_path:
        if isinstance(policy_cfg, GrootConfig):
            from .groot.processor_groot import make_groot_pre_post_processors_from_pretrained

            return make_groot_pre_post_processors_from_pretrained(
                config=policy_cfg,
                pretrained_path=pretrained_path,
                revision=pretrained_revision,
                dataset_stats=kwargs.get("dataset_stats"),
                dataset_meta=kwargs.get("dataset_meta"),
                preprocessor_overrides=kwargs.get("preprocessor_overrides"),
                postprocessor_overrides=kwargs.get("postprocessor_overrides"),
                preprocessor_config_filename=kwargs.get(
                    "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                ),
                postprocessor_config_filename=kwargs.get(
                    "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                ),
            )

        if isinstance(policy_cfg, MolmoAct2Config):
            from .molmoact2.processor_molmoact2 import (
                make_molmoact2_pre_post_processors_from_pretrained,
            )

            return make_molmoact2_pre_post_processors_from_pretrained(
                config=policy_cfg,
                pretrained_path=pretrained_path,
                revision=pretrained_revision,
                preprocessor_overrides=kwargs.get("preprocessor_overrides"),
                postprocessor_overrides=kwargs.get("postprocessor_overrides"),
                preprocessor_config_filename=kwargs.get(
                    "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                ),
                postprocessor_config_filename=kwargs.get(
                    "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                ),
            )

        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
            ),
            overrides=kwargs.get("preprocessor_overrides", {}),
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
            revision=pretrained_revision,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
            ),
            overrides=kwargs.get("postprocessor_overrides", {}),
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
            revision=pretrained_revision,
        )
        _reconnect_relative_absolute_steps(preprocessor, postprocessor)
        if isinstance(policy_cfg, Evo1Config):
            from .evo1.processor_evo1 import reconcile_evo1_processors

            preprocessor, postprocessor = reconcile_evo1_processors(
                policy_cfg,
                preprocessor,
                postprocessor,
            )
        return preprocessor, postprocessor

    # 根据策略配置创建新的处理器，通过命名约定解析各策略专用的工厂函数
    # （延迟导入使可选依赖保持可选）。
    return _make_processors_from_policy_config(
        config=policy_cfg,
        dataset_stats=kwargs.get("dataset_stats"),
        dataset_meta=kwargs.get("dataset_meta"),
    )


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    rename_map: dict[str, str] | None = None,
    defer_weight_load: bool = False,
) -> PreTrainedPolicy:
    """
    实例化策略模型。

    该工厂函数处理创建策略的逻辑，这需要确定输入和输出特征的形状。
    这些形状可以从 `LeRobotDatasetMetadata` 对象或 `EnvConfig` 对象推导。
    该函数既可以从头初始化新策略，也可以加载预训练策略。

    Args:
        cfg (PreTrainedConfig): 要创建的策略的配置。如果设置了
            `cfg.pretrained_path`，策略将从该路径加载权重。
        ds_meta (LeRobotDatasetMetadata | None): 用于推断特征形状和
            类型的数据集元数据。同时为归一化层提供统计信息。
        env_cfg (EnvConfig | None): 用于推断特征形状和类型的环境配置。
            必须提供 `ds_meta` 或 `env_cfg` 之一。
        rename_map (dict[str, str] | None): 可选的映射，用于将数据集或环境的特征键
            重命名以匹配策略期望的特征名称（例如 `"left"` → `"camera1"`）。
        defer_weight_load (bool): 构建与 `from_pretrained` 完全相同的策略——相同的
            配置解析、相同的来自统计信息的缓冲区、相同的设备放置和 eval 模式——
            但跳过 safetensors 权重加载。用于从 DCP 检查点恢复训练时，其分片权重
            会在 `accelerator.prepare()` 之后流式载入（分布式检查点引擎会覆盖随机初始化）。

    Returns:
        PreTrainedPolicy: 已实例化并放置到设备上的策略模型。

    Raises:
        ValueError: 如果同时提供了 `ds_meta` 和 `env_cfg`，或两者都未提供。
        NotImplementedError: 如果尝试使用不支持的策略-后端组合
            （例如 VQBeT 搭配 'mps'）。
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    # 注意：目前，如果你尝试在 mps 后端上运行 vqbet，会得到这个错误。
    # TODO(aliberts, rcadene): 在策略中实现 check_backend_compatibility？
    # NotImplementedError: 算子 'aten::unique_dim' 目前尚未在 MPS 设备上实现。如果你希望
    # 在该功能的原型阶段优先添加此算子，请在
    # https://github.com/pytorch/pytorch/issues/77764 上留言。作为临时修复，你可以设置环境
    # 变量 `PYTORCH_ENABLE_MPS_FALLBACK=1`，让 CPU 作为该算子的回退。警告：这会比在 MPS
    # 上原生运行更慢。
    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        if env_cfg is None:
            raise ValueError("env_cfg cannot be None when ds_meta is not provided")
        features = env_to_policy_features(env_cfg)

    if rename_map:
        features = {rename_map.get(key, key): feature for key, feature in features.items()}

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    if not cfg.input_features:
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}

    # 存储动作特征名称以支持 relative_exclude_joints
    if ds_meta is not None and hasattr(cfg, "action_feature_names"):
        raw_action_feature = next(
            (
                feature
                for raw_key, feature in ds_meta.features.items()
                if (rename_map or {}).get(raw_key, raw_key) == ACTION
            ),
            None,
        )
        action_names = raw_action_feature.get("names") if raw_action_feature is not None else None
        if action_names is not None:
            # 分组元数据将维度名称存储在值中，而不是组键中。
            if isinstance(action_names, dict) and all(
                isinstance(group, (list, tuple)) for group in action_names.values()
            ):
                action_names = [name for group in action_names.values() for name in group]
            cfg.action_feature_names = list(action_names)
    if ds_meta is not None:
        set_dataset_feature_metadata = getattr(cfg, "set_dataset_feature_metadata", None)
        if callable(set_dataset_feature_metadata):
            set_dataset_feature_metadata(ds_meta.features)
        cfg._runtime_dataset_meta = ds_meta

    kwargs["config"] = cfg

    # 如果可用，将 dataset_stats 传递给策略（某些策略如 SARM 需要）
    if ds_meta is not None and hasattr(ds_meta, "stats"):
        kwargs["dataset_stats"] = ds_meta.stats

    if ds_meta is not None:
        kwargs["dataset_meta"] = ds_meta

    if not cfg.pretrained_path and cfg.use_peft:
        raise ValueError(
            "Instantiating a policy with `use_peft=True` without a checkpoint is not supported since that requires "
            "the PEFT config parameters to be set. For training with PEFT, see `lerobot_train.py` on how to do that."
        )

    if cfg.pretrained_path and not cfg.use_peft:
        if defer_weight_load:
            # 与 from_pretrained 相同的构建路径（配置已由调用者从检查点解析；
            # dataset_stats/dataset_meta 关键字参数相同），只是去掉了权重加载——
            # 通过构建方式保证一致性。
            policy = policy_cls(**kwargs)
            policy.eval()
        else:
            # 加载预训练策略，并在需要时覆盖配置（例如，如果存在
            # 我们想要调整的推理时超参数）。
            kwargs["pretrained_name_or_path"] = cfg.pretrained_path
            kwargs["revision"] = cfg.pretrained_revision
            policy = policy_cls.from_pretrained(**kwargs)
    elif cfg.pretrained_path and cfg.use_peft:
        # 在策略之上加载预训练的 PEFT 模型。预训练路径指向适配器的文件夹/仓库，
        # 而适配器的配置中包含基础策略的路径。所以我们需要先获取适配器配置，
        # 然后加载正确的策略，再应用 PEFT。
        require_package("peft", extra="peft")

        logging.info("Loading policy's PEFT adapter.")

        peft_pretrained_path = str(cfg.pretrained_path)
        peft_config = PeftConfig.from_pretrained(
            peft_pretrained_path,
            revision=cfg.pretrained_revision,
        )

        kwargs["pretrained_name_or_path"] = peft_config.base_model_name_or_path
        if not kwargs["pretrained_name_or_path"]:
            # 这意味着要么存在 bug，要么是我们使用 PEFT 从头训练了策略。
            # 更可能的情况是存在 bug，因此我们抛出错误。
            raise ValueError(
                "No pretrained model name found in adapter config. Can't instantiate the pre-trained policy on which "
                "the adapter was trained."
            )

        kwargs["revision"] = peft_config.revision
        policy = policy_cls.from_pretrained(**kwargs)
        policy = PeftModel.from_pretrained(
            policy,
            peft_pretrained_path,
            config=peft_config,
            revision=cfg.pretrained_revision,
            is_trainable=True,
        )

    else:
        # 创建一个全新的策略。
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    if not rename_map:
        validate_visual_features_consistency(cfg, features)
        # TODO: (jadechoghari) - 添加 check_state(cfg, features) 和 check_action(cfg, features)

    return policy


def _get_policy_cls_from_policy_name(name: str) -> type[PreTrainedPolicy]:
    """使用动态导入，根据策略的注册名称获取策略类。

    对内置策略和第三方 lerobot 插件同样适用：通过 draccus ChoiceRegistry
    解析以 ``name`` 注册的配置类，并按命名约定从同级的 ``modeling_*``
    模块导入策略类。

    Args:
        name: 策略的名称。
    Returns:
        与给定名称对应的策略类。
    """
    if name not in PreTrainedConfig.get_known_choices():
        raise ValueError(
            f"Unknown policy name '{name}'. Available policies: {PreTrainedConfig.get_known_choices()}"
        )

    config_cls = PreTrainedConfig.get_choice_class(name)
    config_cls_name = config_cls.__name__

    model_name = config_cls_name.removesuffix("Config")  # 例如 DiffusionConfig -> Diffusion
    if model_name == config_cls_name:
        raise ValueError(
            f"The config class name '{config_cls_name}' does not follow the expected naming convention."
            f"Make sure it ends with 'Config'!"
        )
    cls_name = model_name + "Policy"  # 例如 DiffusionConfig -> DiffusionPolicy
    module_path = config_cls.__module__.replace(
        "configuration_", "modeling_"
    )  # 例如 configuration_diffusion -> modeling_diffusion

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if e.name == module_path:
            # 该策略类型不存在 modeling_* 模块本身。而现有模块内部缺失的
            # 可选依赖会原样向上传播，使其可操作的安装提示保持可见。
            raise ValueError(f"Policy class for '{name}' is not implemented.") from e
        raise
    policy_cls = getattr(module, cls_name, None)
    if policy_cls is None:
        raise ValueError(
            f"Policy class '{cls_name}' not found in '{module_path}'. "
            f"Policies must expose '<Name>Policy' in the sibling 'modeling_*' module by naming convention."
        )
    return policy_cls


def _make_processors_from_policy_config(
    config: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[Any, Any]:
    """使用动态导入，根据策略配置创建预处理器和后处理器。

    按命名约定从策略的 ``processor_*`` 模块解析
    ``make_{type}_pre_post_processors``。对内置策略和第三方 lerobot 插件均适用。

    Args:
        config: 策略配置对象。
        dataset_stats: 用于归一化的数据集统计信息。
        dataset_meta: 数据集元数据，仅转发给声明了 ``dataset_meta``
            参数的工厂（例如 groot、molmoact2）。
    Returns:
        包含输入（预处理器）和输出（后处理器）流水线的元组。
    """

    policy_type = config.type
    function_name = f"make_{policy_type}_pre_post_processors"
    module_path = config.__class__.__module__.replace(
        "configuration_", "processor_"
    )  # 例如 configuration_diffusion -> processor_diffusion
    logging.debug(
        f"Instantiating pre/post processors using function '{function_name}' from module '{module_path}'"
    )
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if e.name == module_path:
            # 该策略类型不存在 processor_* 模块本身。而现有模块内部缺失的
            # 可选依赖会原样向上传播，使其可操作的安装提示保持可见。
            raise ValueError(f"Processor for policy type '{policy_type}' is not implemented.") from e
        raise
    function = getattr(module, function_name, None)
    if function is None:
        raise ValueError(f"Processor for policy type '{policy_type}' is not implemented.")
    call_kwargs: dict[str, Any] = {"dataset_stats": dataset_stats}
    if "dataset_meta" in inspect.signature(function).parameters:
        call_kwargs["dataset_meta"] = dataset_meta
    return function(config, **call_kwargs)
