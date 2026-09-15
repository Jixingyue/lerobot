# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
分布式 HILSerl 机器人策略训练的 learner 服务器运行入口。

本脚本实现了分布式 HILSerl 架构中的 learner 组件。
它负责初始化策略网络、维护回放缓冲区，并根据从 actor 服务器接收到的
转移（transitions）更新策略。

使用示例：

- 启动 learner 服务器进行训练：
```bash
python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**注意**：请在启动 actor 服务器之前先启动 learner 服务器。learner 会开启一个 gRPC 服务器
与各个 actor 通信。

**注意**：如果在配置中将 wandb.enable 设为 true，可通过 Weights & Biases
监控训练进度。

**工作流程**：
1. 创建包含正确策略、数据集和环境设置的训练配置
2. 使用该配置启动此 learner 服务器
3. 使用相同配置启动 actor 服务器
4. 通过 wandb 仪表盘监控训练进度

有关完整 HILSerl 训练工作流程的更多细节，请参见：
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _grpc_available, require_package

if TYPE_CHECKING or _grpc_available:
    import grpc

    from lerobot.transport import services_pb2_grpc
else:
    grpc = None
    services_pb2_grpc = None

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file as load_safetensors
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from lerobot.cameras import opencv  # noqa: F401
from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    load_training_metadata,
    save_checkpoint,
    should_save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset, make_dataset
from lerobot.optim import load_optimizer_state
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.utils.constants import (
    ACTION,
    ALGORITHM_DIR,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.process import ProcessSignalHandler, ensure_multiprocessing_start_method
from lerobot.utils.random_utils import load_rng_state, set_seed
from lerobot.utils.utils import (
    format_big_number,
    init_logging,
)

from .algorithms.base import RLAlgorithm
from .algorithms.factory import make_algorithm
from .buffer import ReplayBuffer
from .data_sources import OnlineOfflineMixer
from .learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService
from .train_rl import TrainRLServerPipelineConfig
from .trainer import RLTrainer


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    # 如果缺少可选的 ``hilserl`` 额外依赖，尽快以友好的错误信息失败。
    require_package("grpcio", extra="hilserl", import_name="grpc")
    if not use_threads(cfg):
        ensure_multiprocessing_start_method(cfg.policy.concurrency.multiprocessing_context)

    # 使用配置中的 job_name
    train(
        cfg,
        job_name=cfg.job_name,
    )

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    """
    初始化并运行训练流程的主训练函数。

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置
        job_name (str | None, optional): 用于日志记录的作业名称。默认为 None。
    """

    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # 创建 logs 目录以确保其存在
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # 使用显式指定的日志文件初始化日志记录
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    # 如果启用了 WandB，则设置 WandB 日志记录
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.common.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # 处理恢复训练（resume）逻辑
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
    )


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: Any,  # Event
) -> None:
    """
    启动训练所需的 learner 线程。

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置
        wandb_logger (WandBLogger | None): 指标日志记录器
        shutdown_event: 用于通知关闭的事件（Event）
    """
    # 创建多进程队列
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    communication_process = concurrency_entity(
        target=start_learner,
        args=(
            parameters_queue,
            transition_queue,
            interaction_message_queue,
            shutdown_event,
            cfg,
        ),
        daemon=True,
    )
    communication_process.start()

    try:
        add_actor_information_and_train(
            cfg=cfg,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
            transition_queue=transition_queue,
            interaction_message_queue=interaction_message_queue,
            parameters_queue=parameters_queue,
        )
        logging.info("[LEARNER] Training process stopped")
    except Exception:
        logging.exception("[LEARNER] Unhandled exception in training loop")
        shutdown_event.set()
    finally:
        logging.info("[LEARNER] Closing queues")
        transition_queue.close()
        interaction_message_queue.close()
        parameters_queue.close()

        communication_process.join()
        logging.info("[LEARNER] Communication process joined")

        transition_queue.cancel_join_thread()
        interaction_message_queue.cancel_join_thread()
        parameters_queue.cancel_join_thread()

        logging.info("[LEARNER] Cleanup complete")


# 核心算法函数


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: Any,  # Event
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
):
    """
    在在线强化学习设置中，负责将数据从 actor 传输到 learner、管理训练更新，
    并记录训练进度。

    本函数持续执行以下操作：
    - 将转移（transitions）从 actor 传输到回放缓冲区。
    - 记录接收到的交互消息。
    - 确保仅当回放缓冲区中积累了足够数量的转移后才开始训练。
    - 将训练更新委托给 ``RLAlgorithm``。
    - 定期将更新后的权重推送给各个 actor。
    - 记录训练统计信息，包括损失值和优化频率。

    注意：本函数并不遵循单一职责原则，未来应当拆分为多个函数。
    之所以这样做，是因为 Python 的 GIL。多线程时速度极慢，性能会下降
    200 倍。因此我们需要用单个线程来完成所有工作。

    Args:
        cfg (TrainRLServerPipelineConfig): 包含超参数的配置对象。
        wandb_logger (WandBLogger | None): 用于跟踪训练进度的日志记录器。
        shutdown_event (Event): 用于通知关闭的事件。
        transition_queue (Queue): 用于从 actor 接收转移的队列。
        interaction_message_queue (Queue): 用于从 actor 接收交互消息的队列。
        parameters_queue (Queue): 用于向 actor 发送策略参数的队列。
    """
    # 在开头提取所有配置变量，这可以带来 7% 的速度提升
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    online_step_before_learning = cfg.policy.online_step_before_learning
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps

    # 为多进程初始化日志记录
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    logging.info("Initializing policy")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    assert isinstance(policy, nn.Module)

    policy.train()

    algorithm = make_algorithm(cfg=cfg.algorithm, policy=policy)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=cfg.policy.dataset_stats,
    )

    # 将初始策略权重推送给各个 actor
    push_actor_policy_to_queue(parameters_queue=parameters_queue, algorithm=algorithm)
    last_time_policy_pushed = time.time()

    log_training_info(cfg=cfg, policy=policy)

    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    batch_size = cfg.batch_size
    offline_replay_buffer = None

    if cfg.dataset is not None:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device,
        )

    # DataMixer：仅在线数据，或在线/离线各 50% 混合
    data_mixer = OnlineOfflineMixer(
        online_buffer=replay_buffer,
        offline_buffer=offline_replay_buffer,
        online_ratio=cfg.online_ratio,
    )
    # RLTrainer 拥有迭代器和预处理器，并负责创建优化器。
    trainer = RLTrainer(
        algorithm=algorithm,
        data_mixer=data_mixer,
        batch_size=batch_size,
        preprocessor=preprocessor,
    )

    # 如果是恢复训练，则需要加载训练状态
    optimizers = algorithm.get_optimizers()
    resume_optimization_step, resume_interaction_step = load_training_state(
        cfg=cfg, optimizers=optimizers, algorithm=algorithm, device=device
    )

    logging.info("Starting learner thread")
    interaction_message = None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    algorithm.optimization_step = optimization_step
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id

    # 注意：这是 LEARNER 的主循环
    while True:
        # 如果收到关闭请求，则退出训练循环
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        # 将 actor 服务器发送来的所有可用转移处理到回放缓冲区中
        process_transitions(
            transition_queue=transition_queue,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            dataset_repo_id=dataset_repo_id,
            shutdown_event=shutdown_event,
        )

        # 处理 actor 服务器发送来的所有可用交互消息
        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
        )

        # 等待回放缓冲区积累足够多的样本后再开始训练
        if len(replay_buffer) < online_step_before_learning:
            continue

        time_for_one_optimization_step = time.time()

        # 执行一个训练步（trainer 拥有 data_mixer 迭代器；algorithm 拥有 UTD 循环）
        stats = trainer.training_step()

        # 必要时将策略推送给各个 actor
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            push_actor_policy_to_queue(parameters_queue=parameters_queue, algorithm=algorithm)
            last_time_policy_pushed = time.time()

        training_infos = stats.to_log_dict()

        # 按指定间隔记录训练指标
        optimization_step = algorithm.optimization_step
        if optimization_step % log_freq == 0:
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step

            # 记录训练指标
            if wandb_logger:
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

        # 计算并记录优化频率
        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

        logging.info(f"[LEARNER] Optimization frequency loop [Hz]: {frequency_for_one_optimization_step}")

        # 记录优化频率
        if wandb_logger:
            wandb_logger.log_dict(
                {
                    "Optimization frequency loop [Hz]": frequency_for_one_optimization_step,
                    "Optimization step": optimization_step,
                },
                mode="train",
                custom_step_key="Optimization step",
            )

        if optimization_step % log_freq == 0:
            logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

        # 按指定间隔保存检查点
        if saving_checkpoint and should_save_checkpoint(optimization_step, save_freq, online_steps):
            save_training_checkpoint(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                interaction_message=interaction_message,
                policy=policy,
                optimizers=optimizers,
                replay_buffer=replay_buffer,
                algorithm=algorithm,
                offline_replay_buffer=offline_replay_buffer,
                dataset_repo_id=dataset_repo_id,
                fps=fps,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )


def start_learner(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: Any,  # Event
    cfg: TrainRLServerPipelineConfig,
):
    """
    启动训练用的 learner 服务器。
    它将从 actor 服务器接收转移和交互消息，
    并向 actor 服务器发送策略参数。

    Args:
        parameters_queue: 用于向 actor 发送策略参数的队列
        transition_queue: 用于从 actor 接收转移的队列
        interaction_message_queue: 用于从 actor 接收交互消息的队列
        shutdown_event: 用于通知关闭的事件
        cfg: 训练配置
    """
    if not use_threads(cfg):
        # 创建进程专属的日志文件
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_process_{os.getpid()}.log")

        # 使用显式指定的日志文件初始化日志记录
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Learner server process logging initialized")

        # 设置进程处理器以处理关闭信号
        # 但使用来自主进程的关闭事件
        # 为多进程（MP）返回
        # TODO: 检查这是否有用
        _ = ProcessSignalHandler(False, display_pid=True)

    service = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    services_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    host = cfg.policy.actor_learner_config.learner_host
    port = cfg.policy.actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")


def save_training_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict | None,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer,
    algorithm: RLAlgorithm | None = None,
    offline_replay_buffer: ReplayBuffer | None = None,
    dataset_repo_id: str | None = None,
    fps: int = 30,
    preprocessor=None,
    postprocessor=None,
) -> None:
    """
    保存训练检查点及相关数据。

    本函数执行以下步骤：
    1. 创建以当前优化步命名的检查点目录
    2. 保存策略模型、配置和优化器状态
    3. 保存当前交互步，以便恢复训练
    4. 更新 "last" 检查点符号链接，使其指向此检查点
    5. 将回放缓冲区保存为数据集，供后续使用
    6. 如果存在离线回放缓冲区，将其保存为单独的数据集

    Args:
        cfg: 训练配置
        optimization_step: 当前优化步
        online_steps: 在线步的总数
        interaction_message: 包含交互信息的字典
        policy: 要保存的策略模型
        optimizers: 优化器字典
        replay_buffer: 要保存为数据集的回放缓冲区
        offline_replay_buffer: 可选的、要保存的离线回放缓冲区
        dataset_repo_id: 数据集的仓库 ID
        fps: 数据集的每秒帧数
        preprocessor: 可选的、要保存的预处理器流水线
        postprocessor: 可选的、要保存的后处理器流水线
    """
    logging.info(f"Checkpoint policy after step {optimization_step}")
    _num_digits = max(6, len(str(online_steps)))
    interaction_step = interaction_message["Interaction step"] if interaction_message is not None else 0

    # 创建检查点目录
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, online_steps, optimization_step)

    # 保存策略产物（pretrained_model/）以及 Trainer 脚手架（training_state/）。
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizers,
        scheduler=None,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    # 算法自有的张量存放在其独立的组件子文件夹中，
    # 这样它们可以单独被 `push_to_hub`，也不会让推理产物变得臃肿。
    if algorithm is not None:
        algorithm.save_pretrained(checkpoint_dir / ALGORITHM_DIR)

    # 在 training_step.json 中补充 RL 特有的 interaction_step 计数器，
    # 以便两者都能从单个文件恢复。
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    write_json(
        {"step": optimization_step, "interaction_step": interaction_step},
        training_state_dir / TRAINING_STEP,
    )

    # 更新 "last" 符号链接
    update_last_checkpoint(checkpoint_dir)

    # TODO：暂时把回放缓冲区保存在这里，以后部署到机器人上时移除
    # 我们希望通过键盘输入来控制这一行为
    dataset_dir = os.path.join(cfg.output_dir, "dataset")
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    # 保存数据集
    # 注意：处理配置中未指定数据集 repo id 的情况，
    # 例如没有演示数据的 RL 训练
    repo_id_buffer_save = cfg.env.task if dataset_repo_id is None else dataset_repo_id
    replay_buffer.to_lerobot_dataset(repo_id=repo_id_buffer_save, fps=fps, root=dataset_dir)

    if offline_replay_buffer is not None:
        dataset_offline_dir = os.path.join(cfg.output_dir, "dataset_offline")
        if os.path.exists(dataset_offline_dir) and os.path.isdir(dataset_offline_dir):
            shutil.rmtree(dataset_offline_dir)

        offline_replay_buffer.to_lerobot_dataset(
            cfg.dataset.repo_id,
            fps=fps,
            root=dataset_offline_dir,
        )

    logging.info("Resume training")


# 训练设置相关函数


def handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """
    处理训练的恢复（resume）逻辑。

    当 resume 为 True 时：
    - 验证检查点是否存在
    - 加载检查点配置
    - 记录恢复详情
    - 返回检查点配置

    当 resume 为 False 时：
    - 检查输出目录是否已存在（以防止意外覆盖）
    - 返回原始配置

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置

    Returns:
        TrainRLServerPipelineConfig: 更新后的配置

    Raises:
        RuntimeError: resume 为 True 但未找到检查点，或 resume 为 False 但目录已存在时抛出
    """
    out_dir = cfg.output_dir

    # 情形 1：不恢复训练，但需要检查目录是否存在以防止覆盖
    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg

    # 情形 2：恢复训练
    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    # 记录已找到有效检查点并正在恢复
    logging.info(
        colored(
            "Valid checkpoint found: resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    # 使用 Draccus 加载配置
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, "train_config.json")
    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)

    # 确保返回的配置中设置了 resume 标志
    checkpoint_cfg.resume = True
    return checkpoint_cfg


def load_training_state(
    cfg: TrainRLServerPipelineConfig,
    optimizers: Optimizer | dict[str, Optimizer],
    algorithm: RLAlgorithm | None = None,
    device: str | torch.device = "cpu",
):
    """
    从最近的检查点加载训练状态（优化器、随机数生成器、训练步 + 交互步，
    以及算法自有的张量）。

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置；`cfg.resume` 控制是否加载，
            `cfg.output_dir` 用于定位最近的检查点。
        optimizers (Optimizer | dict[str, Optimizer]): 要将状态加载到其中的优化器。
        algorithm (RLAlgorithm | None, optional): 需要恢复状态字典的算法。
            若要实现与主流程完全等价的恢复，则必须提供；策略本身通过
            `make_policy` 单独恢复。默认为 None。
        device (str | torch.device, optional): 加载后的算法张量所放置的设备。
            默认为 "cpu"。

    Returns:
        tuple[int | None, int | None]: `(optimization_step, interaction_step)`；
        当未恢复训练或加载训练状态失败时返回 `(None, None)`。
    """
    if not cfg.resume:
        return None, None

    # 构造最近检查点目录的路径
    checkpoint_dir = Path(cfg.output_dir) / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK

    logging.info(f"Loading training state from {checkpoint_dir}")

    try:
        # 从标准的 `training_state/` 文件夹恢复优化器 + RNG + 训练步
        training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
        load_rng_state(training_state_dir)
        step = load_training_metadata(training_state_dir)["step"]
        optimizers = load_optimizer_state(optimizers, training_state_dir)

        # 恢复算法自有的张量
        if algorithm is not None:
            algo_dir = checkpoint_dir / ALGORITHM_DIR
            if algo_dir.is_dir():
                tensors = load_safetensors(str(algo_dir / SAFETENSORS_SINGLE_FILE))
                algorithm.load_state_dict(tensors, device=device)
                logging.info(f"Loaded algorithm state from {algo_dir}")
            else:
                logging.warning(
                    f"No algorithm state found at {algo_dir}; "
                    "will keep their freshly-initialised values. Adam moments restored from the "
                    "old optimizer state may not match these reset parameters."
                )

        # 从补充后的 training_step.json 中读取 interaction_step
        training_step_path = checkpoint_dir / TRAINING_STATE_DIR / TRAINING_STEP
        interaction_step = int(load_json(training_step_path).get("interaction_step", 0))

        logging.info(f"Resuming from step {step}, interaction step {interaction_step}")
        return step, interaction_step

    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        return None, None


def log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module) -> None:
    """
    记录有关训练流程的信息。

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置
        policy (nn.Module): 策略模型
    """
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(
    cfg: TrainRLServerPipelineConfig, device: str, storage_device: str
) -> ReplayBuffer:
    """
    初始化回放缓冲区：要么创建空缓冲区，要么在恢复训练时从数据集加载。

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置
        device (str): 存储张量所用的设备
        storage_device (str): 用于存储优化的设备

    Returns:
        ReplayBuffer: 初始化后的回放缓冲区
    """
    if not cfg.resume:
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    logging.info("Resume training load the online dataset")
    dataset_path = os.path.join(cfg.output_dir, "dataset")

    # 注意：在 RL 中，有可能不存在数据集。
    repo_id = None
    if cfg.dataset is not None:
        repo_id = cfg.dataset.repo_id
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
    )
    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.policy.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        optimize_memory=True,
    )


def initialize_offline_replay_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
) -> ReplayBuffer:
    """
    从数据集初始化离线回放缓冲区。

    Args:
        cfg (TrainRLServerPipelineConfig): 训练配置
        device (str): 存储张量所用的设备
        storage_device (str): 用于存储优化的设备

    Returns:
        ReplayBuffer: 初始化后的离线回放缓冲区
    """
    if not cfg.resume:
        logging.info("make_dataset offline buffer")
        offline_dataset = make_dataset(cfg)
    else:
        logging.info("load offline dataset")
        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=dataset_offline_path,
        )

    logging.info("Convert to a offline replay buffer")
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
    )
    return offline_replay_buffer


# 工具/辅助函数


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.learner == "threads"


def check_nan_in_transition(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    raise_error: bool = False,
) -> bool:
    """
    检查转移数据中是否存在 NaN 值。

    Args:
        observations: 观测张量组成的字典
        actions: 动作张量
        next_state: 下一状态张量组成的字典
        raise_error: 如果为 True，检测到 NaN 时抛出 ValueError

    Returns:
        bool: 检测到 NaN 值时返回 True，否则返回 False
    """
    nan_detected = False

    # 检查观测
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logging.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")

    # 检查下一状态
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logging.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")

    # 检查动作
    if torch.isnan(actions).any():
        logging.error("actions contains NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")

    return nan_detected


def push_actor_policy_to_queue(parameters_queue: Queue, algorithm: RLAlgorithm) -> None:
    logging.debug("[LEARNER] Pushing actor policy to the queue")

    # 创建一个字典来容纳所有的 state dict
    state_dicts = algorithm.get_weights()
    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


def process_interaction_message(
    message, interaction_step_shift: int, wandb_logger: WandBLogger | None = None
):
    """以一致的方式处理单条交互消息。"""
    message = bytes_to_python_object(message)
    # 对交互步进行偏移，以与检查点中保存的状态保持一致
    message["Interaction step"] += interaction_step_shift

    # 如果日志记录器可用，则进行记录
    if wandb_logger:
        wandb_logger.log_dict(d=message, mode="train", custom_step_key="Interaction step")

    return message


def process_transitions(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    dataset_repo_id: str | None,
    shutdown_event: Any,  # Event
):
    """处理队列中所有可用的转移。

    Args:
        transition_queue: 用于从 actor 接收转移的队列
        replay_buffer: 用于添加转移的回放缓冲区
        offline_replay_buffer: 用于添加转移的离线回放缓冲区
        dataset_repo_id: 数据集的仓库 ID
        shutdown_event: 用于通知关闭的事件
    """
    while not transition_queue.empty() and not shutdown_event.is_set():
        transition_list = transition_queue.get()
        transition_list = bytes_to_transitions(buffer=transition_list)

        for transition in transition_list:
            # 跳过含有 NaN 值的转移
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition[ACTION],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            replay_buffer.add(**transition)

            # 如果这是一次人工干预（intervention），则添加到离线缓冲区
            if dataset_repo_id is not None and transition.get("complementary_info", {}).get(
                TeleopEvents.IS_INTERVENTION.value
            ):
                offline_replay_buffer.add(**transition)


def process_interaction_messages(
    interaction_message_queue: Queue,
    interaction_step_shift: int,
    wandb_logger: WandBLogger | None,
    shutdown_event: Any,  # Event
) -> dict | None:
    """处理队列中所有可用的交互消息。

    Args:
        interaction_message_queue: 用于接收交互消息的队列
        interaction_step_shift: 交互步的偏移量
        wandb_logger: 用于跟踪进度的日志记录器
        shutdown_event: 用于通知关闭的事件

    Returns:
        dict | None: 处理的最后一条交互消息；如果没有处理任何消息则返回 None
    """
    last_message = None
    while not interaction_message_queue.empty() and not shutdown_event.is_set():
        message = interaction_message_queue.get()
        last_message = process_interaction_message(
            message=message,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
        )

    return last_message


if __name__ == "__main__":
    train_cli()
    logging.info("[LEARNER] main finished")
