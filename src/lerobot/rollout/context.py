# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Rollout 上下文：在策略分发之前一次性创建的共享状态。

按主题分组为五个子上下文——:class:`RuntimeContext`、
:class:`HardwareContext`、:class:`PolicyContext`、:class:`ProcessorContext`
和 :class:`DatasetContext`——并组装成 :class:`RolloutContext`。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from threading import Event
from typing import TYPE_CHECKING

import torch

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
    rename_stats,
)
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.import_utils import _peft_available, require_package

from .configs import RolloutConfig
from .inference import (
    InferenceEngine,
    RTCInferenceConfig,
    SyncInferenceConfig,
    create_inference_engine,
)
from .inference.rtc import supports_rtc_inference
from .robot_wrapper import ThreadSafeRobot

if TYPE_CHECKING or _peft_available:
    from peft import PeftConfig, PeftModel
else:
    PeftConfig = None
    PeftModel = None

logger = logging.getLogger(__name__)


def _wrap_predict_action_chunk_with_torch_compile(
    policy: PreTrainedPolicy,
    *,
    backend: str,
    mode: str,
) -> bool:
    """安装 JIT 包装器，并报告其是否配置成功。

    ``torch.compile`` 在首次调用时才进行惰性编译，因此这里的成功
    并不能保证后端编译在预热期间一定会成功。
    """
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is not available in this PyTorch build")
        return False

    try:
        policy.predict_action_chunk = torch.compile(
            policy.predict_action_chunk,
            backend=backend,
            mode=mode,
        )
    except Exception as exc:
        logger.warning("Failed to configure torch.compile: %s", exc)
        return False

    logger.info("torch.compile configured for predict_action_chunk")
    return True


def _validate_trained_rtc_rollout_config(policy_config, inference_config: RTCInferenceConfig) -> None:
    """当 rollout 无法保留所有已训练的 RTC 前缀时快速失败。"""
    rtc = inference_config.rtc
    if not rtc.enabled or rtc.mode != "trained":
        return
    if policy_config.type != "pi05":
        raise ValueError(
            "--inference.rtc.mode=trained currently requires a PI05 checkpoint; "
            f"got policy type {policy_config.type!r}."
        )

    training_max_delay = int(getattr(policy_config, "rtc_training_max_delay", 0))
    if training_max_delay <= 0:
        raise ValueError(
            "--inference.rtc.mode=trained requires a checkpoint trained with "
            "--policy.rtc_training_max_delay > 0."
        )
    if rtc.execution_horizon < training_max_delay:
        raise ValueError(
            f"--inference.rtc.execution_horizon ({rtc.execution_horizon}) must be at least the "
            f"checkpoint's rtc_training_max_delay ({training_max_delay})."
        )
    if inference_config.queue_threshold < training_max_delay:
        raise ValueError(
            f"--inference.queue_threshold ({inference_config.queue_threshold}) must be at least the "
            f"checkpoint's rtc_training_max_delay ({training_max_delay})."
        )

    # RTC 要求 d <= s <= H - d（arXiv 2506.07339）：执行范围超过 H - d 将会
    # 提交下一个块无法重新规划的动作，因此重叠部分永远无法闭合。
    chunk_size = int(getattr(policy_config, "chunk_size", 0))
    if chunk_size and rtc.execution_horizon > chunk_size - training_max_delay:
        raise ValueError(
            f"--inference.rtc.execution_horizon ({rtc.execution_horizon}) must be at most "
            f"chunk_size - rtc_training_max_delay ({chunk_size} - {training_max_delay} = "
            f"{chunk_size - training_max_delay})."
        )


def _resolve_action_key_order(
    policy_action_names: list[str] | None, dataset_action_names: list[str]
) -> list[str]:
    """选择动作名称的顺序，用于将策略张量输出映射到机器人动作字典。"""
    if not policy_action_names:
        return dataset_action_names
    policy_action_names = list(policy_action_names)
    if len(policy_action_names) != len(dataset_action_names):
        logger.warning(
            "policy.action_feature_names length (%d) != dataset action dim (%d); using dataset order",
            len(policy_action_names),
            len(dataset_action_names),
        )
        return dataset_action_names
    if set(dataset_action_names) != set(policy_action_names):
        logger.warning("policy.action_feature_names keys don't match dataset; using dataset order")
        return dataset_action_names
    return policy_action_names


def _align_state_feature_order(
    observation_features_hw: dict[str, type | tuple], policy_action_names: list[str] | None
) -> dict[str, type | tuple]:
    """对标量状态特征排序，使其与检查点的关节顺序一致。"""
    if not policy_action_names:
        return observation_features_hw

    scalar_names = [
        name for name, feature in observation_features_hw.items() if not isinstance(feature, tuple)
    ]
    if set(scalar_names) != set(policy_action_names) or scalar_names == policy_action_names:
        return observation_features_hw

    reordered = {name: observation_features_hw[name] for name in policy_action_names}
    reordered.update(
        {name: feature for name, feature in observation_features_hw.items() if name not in reordered}
    )
    logger.warning(
        "Robot state order %s differs from checkpoint joint order %s; reordering state",
        scalar_names,
        policy_action_names,
    )
    return reordered


# ---------------------------------------------------------------------------
# 子上下文
# ---------------------------------------------------------------------------


@dataclass
class RuntimeContext:
    """与所有策略共享的运行时参数。"""

    cfg: RolloutConfig
    shutdown_event: Event
    # 控制循环的 ``CycleTimer`` 向何处发送节奏摘要；None
    # 表示继续使用 ``logger.info``。声明了 ``supports_interactive`` 的策略
    # 必须将其转发给在 ``run()`` 中构建的计时器，因为会话期间
    # 会静音所有低于 ERROR 级别的日志。
    cadence_report: Callable[[str], None] | None = None


@dataclass
class HardwareContext:
    """已连接的硬件。

    原始机器人在需要时可通过 ``robot_wrapper.inner`` 获取
    （例如用于断开连接）；其他情况下策略应通过
    线程安全的包装器来访问。

    ``initial_position`` 保存机器人连接时的关节位置。
    策略在关闭前用它将机器人恢复到安全姿态。
    """

    robot_wrapper: ThreadSafeRobot
    teleop: Teleoperator | None
    initial_position: dict | None = None


@dataclass
class PolicyContext:
    """已加载的策略及其推理引擎。"""

    policy: PreTrainedPolicy
    preprocessor: PolicyProcessorPipeline
    postprocessor: PolicyProcessorPipeline
    inference: InferenceEngine


@dataclass
class ProcessorContext:
    """机器人侧流水线（在策略之外运行）。"""

    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation]


@dataclass
class DatasetContext:
    """数据集与特征记录。"""

    dataset: LeRobotDataset | None
    dataset_features: dict = field(default_factory=dict)
    hw_features: dict = field(default_factory=dict)
    ordered_action_keys: list[str] = field(default_factory=list)


@dataclass
class RolloutContext:
    """传递给每个 rollout 策略的子上下文集合。

    由 :func:`build_rollout_context` 在策略分发之前一次性构建。
    """

    runtime: RuntimeContext
    hardware: HardwareContext
    policy: PolicyContext
    processors: ProcessorContext
    data: DatasetContext


# ---------------------------------------------------------------------------
#  构建# ---------------------------------------------------------------------------


def _load_pretrained_policy(policy_config: PreTrainedConfig) -> PreTrainedPolicy:
    """加载策略权重，使 adapter 和基础模型的版本保持相互独立。"""
    pretrained_revision = policy_config.pretrained_revision
    policy_class = get_policy_class(policy_config.type)

    if not policy_config.use_peft:
        return policy_class.from_pretrained(
            policy_config.pretrained_path,
            config=policy_config,
            revision=pretrained_revision,
        )

    require_package("peft", extra="peft")

    peft_path = policy_config.pretrained_path
    peft_config = PeftConfig.from_pretrained(peft_path, revision=pretrained_revision)
    policy = policy_class.from_pretrained(
        pretrained_name_or_path=peft_config.base_model_name_or_path,
        config=policy_config,
        revision=peft_config.revision,
    )
    return PeftModel.from_pretrained(
        policy,
        peft_path,
        config=peft_config,
        revision=pretrained_revision,
    )


def build_rollout_context(
    cfg: RolloutConfig,
    shutdown_event: Event,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> RolloutContext:
    """装配策略、处理器、硬件、数据集和推理引擎。

    顺序是策略优先、硬件最后，这样无效的 ``--policy.path``
    可以在不触碰机器人的情况下快速失败。在任何策略访问之前，
    缺少策略配置会抛出 ``ValueError``。
    """
    is_rtc = isinstance(cfg.inference, RTCInferenceConfig)

    # --- 1. 策略（繁重的 I/O，但尚未涉及硬件） -------------------
    policy_config = cfg.policy
    if policy_config is None:
        raise ValueError("--policy.path is required for rollout")
    logger.info("Loading policy from '%s'...", policy_config.pretrained_path)
    # 策略构造函数和自定义处理器也必须使用解析后的 rollout 设备。
    policy_config.device = cfg.device

    if is_rtc:
        _validate_trained_rtc_rollout_config(policy_config, cfg.inference)

    if hasattr(policy_config, "compile_model"):
        policy_config.compile_model = cfg.use_torch_compile

    if policy_config.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy = _load_pretrained_policy(policy_config)

    if is_rtc:
        if not supports_rtc_inference(policy):
            raise ValueError(
                f"RTC inference is not supported by policy type '{policy_config.type}': "
                "the policy must implement RTC semantics and predict_action_chunk must accept "
                "inference_delay and prev_chunk_left_over. Use '--inference.type=sync' instead."
            )
        policy.config.rtc_config = cfg.inference.rtc
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()

    policy = policy.to(cfg.device)
    policy.eval()
    logger.info("Policy loaded: type=%s, device=%s", policy_config.type, cfg.device)

    torch_compile_active = cfg.use_torch_compile
    if cfg.use_torch_compile and policy.type not in ("pi0", "pi05"):
        torch_compile_active = _wrap_predict_action_chunk_with_torch_compile(
            policy,
            backend=cfg.torch_compile_backend,
            mode=cfg.torch_compile_mode,
        )

    if cfg.use_torch_compile and not torch_compile_active:
        # RolloutConfig.__post_init__ 会重新加载策略配置，因此在向下游
        # 传递生效状态时避免使用 dataclasses.replace。
        cfg = copy(cfg)
        cfg.use_torch_compile = False

    # --- 2. 机器人侧处理器（用户提供或使用默认值） --------
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    # --- 3. 硬件（副作用最重，延后处理） -----------------
    logger.info("Connecting robot (%s)...", cfg.robot.type if cfg.robot else "?")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info("Robot connected: %s", robot.name)

    # 保存初始关节位置，以便在关闭时恢复到安全姿态。
    initial_obs = robot.get_observation()
    initial_position = {k: v for k, v in initial_obs.items() if k.endswith(".pos")}
    logger.info("Captured initial robot position (%d keys)", len(initial_position))

    robot_wrapper = ThreadSafeRobot(robot)

    teleop = None
    if cfg.teleop is not None:
        logger.info("Connecting teleoperator (%s)...", cfg.teleop.type if cfg.teleop else "?")
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()
        logger.info("Teleoperator connected")

    # TODO(Steven): 一旦 Teleoperator 的电机控制方法标准化
    # （``enable_torque`` / ``disable_torque`` / ``write_goal_positions``），
    # 就在这里根据这些方法的存在性来约束 DAgger 策略，并快速失败给出
    # 有帮助的提示信息，而不是依赖操作者手动预对齐主手。
    # 参见 :func:`DAggerStrategy._apply_transition` 中对应的
    # 已禁用调用点。
    # if isinstance(cfg.strategy, DAggerStrategyConfig) and teleop is not None:
    #     required_teleop_methods = ("enable_torque", "disable_torque", "write_goal_positions")
    #     missing = [m for m in required_teleop_methods if not callable(getattr(teleop, m, None))]
    #     if missing:
    #         teleop.disconnect()
    #         raise ValueError(
    #             f"DAgger strategy requires a teleoperator with motor control methods "
    #             f"{required_teleop_methods}. '{type(teleop).__name__}' is missing: {missing}"
    #         )

    # --- 4. 特征与动作键的对齐 ---------------------
    # TODO(Steven): 只有 ``.pos`` 关节特征会作为状态和动作目标路由给策略；
    # 速度和力矩通道（如果存在）保留在原始观测中，
    # 但会被排除在面向策略的张量之外。
    all_obs_features = robot.observation_features
    # ``observation_features`` 的值要么是元组（相机形状），要么是
    # ``float`` 类型本身，用作标量电机特征的哨兵值——
    # 参见 ``Robot.observation_features`` 上的 ``dict[str, type | tuple]`` 注解。
    # 保留相机（元组）以及关节位置（.pos）和底盘速度（.vel）两种
    # 标量状态特征。LeKiwi 的 observation.state 是 9 维（6 个手臂 .pos +
    # x/y/theta.vel），策略也是在这 9 维上训练/归一化的；旧的仅 .pos
    # 过滤会把 6 维状态送入 9 维归一化器 → RuntimeError（size 6 vs 9）。
    # 纯手臂机器人没有 .vel 状态键，因此对它们来说这是无操作。
    observation_features_hw = {
        k: v
        for k, v in all_obs_features.items()
        if isinstance(v, tuple) or (v is float and k.endswith((".pos", ".vel")))
    }
    policy_action_names = getattr(policy_config, "action_feature_names", None)
    observation_features_hw = _align_state_feature_order(
        observation_features_hw,
        list(policy_action_names) if policy_action_names else None,
    )
    # 同时保留关节位置（.pos）和底盘速度（.vel）动作特征，
    # 以便移动操作机器人也能控制底盘（例如 LeKiwi：6 个手臂 .pos +
    # x/y/theta.vel = 9 维动作）。纯手臂机器人没有 .vel 键，
    # 因此对它们来说这是无操作。如果没有 .vel 键，底盘速度会被
    # 悄悄从 dataset_features[ACTION]/ordered_action_keys 中丢弃，底盘将永远不动。
    action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith((".pos", ".vel"))}

    # 动作侧始终是必需的：同步推理从 ``dataset_features[ACTION]``
    # 读取动作名称，以便将策略张量映射回机器人动作。
    action_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=action_features_hw),
        use_videos=cfg.dataset.video if cfg.dataset else True,
    )
    # 由于 build_dataset_frame 的需要，观测侧的聚合也是必需的
    observation_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=observation_features_hw),
        use_videos=cfg.dataset.video if cfg.dataset else True,
    )
    dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)
    hw_features = hw_to_dataset_features(observation_features_hw, "observation")
    raw_action_keys = list(action_features_hw.keys())
    ordered_action_keys = _resolve_action_key_order(
        list(policy_action_names) if policy_action_names else None,
        raw_action_keys,
    )

    # 在未启用 rename_map 时校验视觉特征
    rename_map = cfg.rename_map
    if not rename_map:
        expected_visuals = {
            k for k, v in policy_config.input_features.items() if v.type == FeatureType.VISUAL
        }
        provided_visuals = {
            f"observation.images.{k}" for k, v in robot.observation_features.items() if isinstance(v, tuple)
        }
        policy_subset = expected_visuals.issubset(provided_visuals)
        hw_subset = provided_visuals.issubset(expected_visuals)
        if not (policy_subset or hw_subset):
            raise ValueError(
                f"Visual feature mismatch between policy and robot hardware.\n"
                f"Policy expects: {expected_visuals}\n"
                f"Robot provides: {provided_visuals}\n"
                f"Use --rename_map to map camera names, e.g. "
                f"""--rename_map='{{"observation.images.top": "observation.images.cam0"}}'"""
            )

    # --- 5. 数据集 -------------
    dataset = None
    if cfg.dataset is not None:
        logger.info("Setting up dataset (repo_id=%s)...", cfg.dataset.repo_id)
        # 策略拥有的列会在 resume/create 分支之上与机器人/策略特征合并，
        # 因此 ``ctx.data.dataset_features`` 在两条路径上描述的是同一套模式。
        dataset_features.update(cfg.strategy.extra_dataset_features())
        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
            )
        else:
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if not repo_name.startswith("rollout_"):
                raise ValueError(
                    "Dataset names for rollout must start with 'rollout_'. "
                    "Use --dataset.repo_id=<user>/rollout_<name> for policy deployment datasets."
                )
            cfg.dataset.stamp_repo_id()
            target_video_mb = getattr(cfg.strategy, "target_video_file_size_mb", None)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                video_files_size_in_mb=target_video_mb,
            )

    if dataset is not None:
        logger.info("Dataset ready: %s (%d existing episodes)", dataset.repo_id, dataset.num_episodes)

    # --- 6. 策略前/后处理器（如有需要则使用数据集统计量） ---
    dataset_stats = None
    if dataset is not None:
        dataset_stats = rename_stats(
            dataset.meta.stats,
            cfg.rename_map,
        )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=policy_config.pretrained_path,
        pretrained_revision=policy_config.pretrained_revision,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    relative_action_step = next(
        (
            step
            for step in getattr(preprocessor, "steps", ())
            if isinstance(step, RelativeActionsProcessorStep) and step.enabled
        ),
        None,
    )
    if isinstance(cfg.inference, SyncInferenceConfig) and relative_action_step is not None:
        raise NotImplementedError(
            "SyncInferenceEngine does not support policies with relative actions for now."
            "Use --inference.type=rtc or remove relative action processor steps from the policy pipeline."
        )

    # --- 7. 推理策略（需要策略 + 前/后处理器 + 硬件） --
    logger.info(
        "Creating inference engine (type=%s)...",
        cfg.inference.type if hasattr(cfg.inference, "type") else "sync",
    )
    task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
    inference_strategy = create_inference_engine(
        cfg.inference,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot_wrapper,
        hw_features=hw_features,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=task_str,
        fps=cfg.fps,
        device=cfg.device,
        use_torch_compile=torch_compile_active,
        compile_warmup_inferences=cfg.compile_warmup_inferences,
        shutdown_event=shutdown_event,
    )

    # --- 8. 组装 ---------------------------------------------------
    logger.info("Rollout context assembled successfully")
    return RolloutContext(
        runtime=RuntimeContext(cfg=cfg, shutdown_event=shutdown_event),
        hardware=HardwareContext(
            robot_wrapper=robot_wrapper, teleop=teleop, initial_position=initial_position
        ),
        policy=PolicyContext(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            inference=inference_strategy,
        ),
        processors=ProcessorContext(
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        ),
        data=DatasetContext(
            dataset=dataset,
            dataset_features=dataset_features,
            hw_features=hw_features,
            ordered_action_keys=ordered_action_keys,
        ),
    )
