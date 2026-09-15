#!/usr/bin/env python

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
"""
用于分布式 HILSerl 机器人策略训练的 Actor 服务器运行脚本。

本脚本实现了分布式 HILSerl 架构中的 actor 组件。
它在机器人环境中执行策略、收集经验，
并将 transition 发送给 learner 服务器以更新策略。

使用示例：

- 启动一个用于带有人工介入（human-in-the-loop）的真实机器人训练的 actor 服务器：
```bash
python -m lerobot.rl.actor --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**注意**：actor 服务器需要一个正在运行的 learner 服务器以供连接。请确保在
启动 actor 之前先启动 learner 服务器。

**注意**：人工介入是 HILSerl 训练的关键。在训练期间按下手柄右上角的
扳机按钮即可接管机器人控制。初期应频繁介入，随后随着策略
改进逐步减少介入次数。

**工作流程**：
1. 使用 `lerobot-find-joint-limits` 确定机器人工作空间的边界
2. 使用 `gym_manipulator.py` 的录制模式记录演示
3. 使用 `crop_dataset_roi.py` 处理数据集并确定相机裁剪区域
4. 使用训练配置启动 learner 服务器
5. 使用相同的配置启动本 actor 服务器
6. 通过人工介入引导策略学习

有关完整 HILSerl 训练工作流程的更多详情，参见：
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import time
from collections.abc import Generator
from functools import lru_cache
from queue import Empty
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _grpc_available, require_package

if TYPE_CHECKING or _grpc_available:
    import grpc

    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import (
        bytes_to_state_dict,
        grpc_channel_options,
        python_object_to_bytes,
        receive_bytes_in_chunks,
        send_bytes_in_chunks,
        transitions_to_bytes,
    )
else:
    grpc = None
    services_pb2 = None
    services_pb2_grpc = None
    bytes_to_state_dict = None
    grpc_channel_options = None
    python_object_to_bytes = None
    receive_bytes_in_chunks = None
    send_bytes_in_chunks = None
    transitions_to_bytes = None

import torch
from torch import nn
from torch.multiprocessing import Queue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import TransitionKey
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.process import ProcessSignalHandler, ensure_multiprocessing_start_method
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import (
    Transition,
    move_transition_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    init_logging,
)

from .algorithms.base import RLAlgorithm
from .algorithms.factory import make_algorithm
from .gym_manipulator import (
    make_processors,
    make_robot_env,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from .queue import get_last_item_from_queue
from .train_rl import TrainRLServerPipelineConfig

# 主入口


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    # 如果缺少可选的 ``hilserl`` extra，快速失败并给出友好的错误信息。
    require_package("grpcio", extra="hilserl", import_name="grpc")
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        ensure_multiprocessing_start_method(cfg.policy.concurrency.multiprocessing_context)
        display_pid = True

    # 创建日志目录以确保其存在
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    # 使用显式的日志文件初始化日志记录
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host,
        port=cfg.policy.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        # 如果使用多线程，我们可以复用该通道
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    try:
        act_with_policy(
            cfg=cfg,
            shutdown_event=shutdown_event,
            parameters_queue=parameters_queue,
            transitions_queue=transitions_queue,
            interactions_queue=interactions_queue,
        )
        logging.info("[ACTOR] Policy loop finished")
    except Exception:
        logging.exception("[ACTOR] Unhandled exception in act_with_policy")
        shutdown_event.set()
    finally:
        logging.info("[ACTOR] Closing queues")
        transitions_queue.close()
        interactions_queue.close()
        parameters_queue.close()

        transitions_process.join()
        logging.info("[ACTOR] Transitions process joined")
        interactions_process.join()
        logging.info("[ACTOR] Interactions process joined")
        receive_policy_process.join()
        logging.info("[ACTOR] Receive policy process joined")

        transitions_queue.cancel_join_thread()
        interactions_queue.cancel_join_thread()
        parameters_queue.cancel_join_thread()

        logging.info("[ACTOR] Cleanup complete")


# 核心算法函数


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: Any,  # Event
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
):
    """
    在环境中执行策略交互。

    该函数在环境中展开（roll out）策略，收集交互数据并将其推入队列，以流式发送给 learner。
    一旦某个 episode 完成，就从队列中取出从 learner 接收到的更新后的网络参数并加载到网络中。

    Args:
        cfg: 交互过程的配置项。
        shutdown_event: 用于检查进程是否应关闭的事件。
        parameters_queue: 用于从 learner 接收更新后网络参数的队列。
        transitions_queue: 用于向 learner 发送 transition 的队列。
        interactions_queue: 用于向 learner 发送交互信息的队列。
    """
    # 为多进程初始化日志记录
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")

    logging.info("make_env online")

    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### 在 actor 和 learner 两个进程中分别实例化策略
    ### 为了避免通过端口发送策略对象，我们在双方各自创建策略实例，
    ### learner 每 n 步发送一次更新后的参数，以更新 actor 的参数
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy = policy.to(device).eval()
    assert isinstance(policy, nn.Module)

    # 构建算法
    algorithm = make_algorithm(cfg=cfg.algorithm, policy=policy)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=cfg.policy.dataset_stats,
    )

    transition = reset_and_build_transition(online_env, env_processor, action_processor)

    # 注意：目前我们仅处理单环境的情况
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    episode_intervention = False
    # 添加用于计算介入率的计数器
    episode_intervention_steps = 0
    episode_total_steps = 0

    policy_timer = TimerManager("Policy inference", log=False)

    for interaction_step in range(cfg.policy.online_steps):
        start_time = time.perf_counter()
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down act_with_policy")
            return

        observation = {
            k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
        }

        # 对策略推理计时，并检查是否满足 FPS 要求
        with policy_timer:
            normalized_observation = preprocessor.process_observation(observation)
            action = policy.select_action(batch=normalized_observation)
            # 仅对连续部分进行反归一化。
            if cfg.policy.num_discrete_actions is not None:
                continuous_action = postprocessor.process_action(action[..., :-1])
                discrete_action = action[..., -1:].to(
                    device=continuous_action.device, dtype=continuous_action.dtype
                )
                action = torch.cat([continuous_action, discrete_action], dim=-1)
            else:
                action = postprocessor.process_action(action)
        policy_fps = policy_timer.fps_last

        log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)

        # 使用新的 step 函数
        new_transition = step_env_and_process_transition(
            env=online_env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        # 从处理后的 transition 中提取各项值
        next_observation = {
            k: v
            for k, v in new_transition[TransitionKey.OBSERVATION].items()
            if k in cfg.policy.input_features
        }

        # Teleop action 是实际在环境中执行的动作
        # 它要么来自遥操作设备，要么来自策略
        executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]

        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)

        sum_reward_episode += float(reward)
        episode_total_steps += 1

        # 从 transition 信息中检查是否发生了介入
        intervention_info = new_transition[TransitionKey.INFO]
        is_intervention = bool(intervention_info.get(TeleopEvents.IS_INTERVENTION, False))
        if is_intervention:
            episode_intervention = True
            episode_intervention_steps += 1

        complementary_info = {
            "discrete_penalty": torch.tensor(
                [new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)]
            ),
            TeleopEvents.IS_INTERVENTION.value: is_intervention,
        }
        # 为 learner 创建 transition（转换为旧格式）
        list_transition_to_send_to_learner.append(
            Transition(
                state=observation,
                action=executed_action,
                reward=reward,
                next_state=next_observation,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
        )

        # 更新 transition 以供下一次迭代使用
        transition = new_transition

        if done or truncated:
            logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")

            update_policy_parameters(algorithm=algorithm, parameters_queue=parameters_queue, device=device)

            if len(list_transition_to_send_to_learner) > 0:
                push_transitions_to_transport_queue(
                    transitions=list_transition_to_send_to_learner,
                    transitions_queue=transitions_queue,
                )
                list_transition_to_send_to_learner = []

            stats = get_frequency_stats(policy_timer)
            policy_timer.reset()

            # 计算介入率
            intervention_rate = 0.0
            if episode_total_steps > 0:
                intervention_rate = episode_intervention_steps / episode_total_steps

            # 将 episode 奖励发送给 learner
            interactions_queue.put(
                python_object_to_bytes(
                    {
                        "Episodic reward": sum_reward_episode,
                        "Interaction step": interaction_step,
                        "Episode intervention": int(episode_intervention),
                        "Intervention rate": intervention_rate,
                        **stats,
                    }
                )
            )

            # 重置介入计数器和环境
            sum_reward_episode = 0.0
            episode_intervention = False
            episode_intervention_steps = 0
            episode_total_steps = 0

            transition = reset_and_build_transition(online_env, env_processor, action_processor)

        if cfg.env.fps is not None:
            dt_time = time.perf_counter() - start_time
            precise_sleep(max(1 / cfg.env.fps - dt_time, 0.0))


#  通信函数 - 汇总所有 gRPC/消息传递函数


def establish_learner_connection(
    stub: "services_pb2_grpc.LearnerServiceStub",
    shutdown_event: Any,  # Event
    attempts: int = 30,
) -> bool:
    """与 learner 建立连接。

    Args:
        stub (services_pb2_grpc.LearnerServiceStub): 用于连接的 stub。
        shutdown_event (Event): 用于检查是否应建立连接的事件。
        attempts (int): 建立连接的尝试次数。
    Returns:
        bool: 如果连接已建立则返回 True，否则返回 False。
    """
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # 强制尝试连接并检查状态
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False


@lru_cache(maxsize=1)
def learner_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> "tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]":
    """返回 learner 服务的客户端。

    GRPC 使用 HTTP/2，这是一种二进制协议，可在单个连接上多路复用请求。
    因此我们只需创建一个客户端并复用它。

    Returns:
        tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]: stub 和通道。
    """

    channel = grpc.insecure_channel(
        f"{host}:{port}",
        grpc_channel_options(),
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel


def receive_policy(
    cfg: TrainRLServerPipelineConfig,
    parameters_queue: Queue,
    shutdown_event: Any,  # Event
    learner_client: "services_pb2_grpc.LearnerServiceStub | None" = None,
    grpc_channel: "grpc.Channel | None" = None,
) -> None:
    """从 learner 接收参数。

    Args:
        cfg (TrainRLServerPipelineConfig): actor 的配置。
        parameters_queue (Queue): 用于接收参数的队列。
        shutdown_event (Event): 用于检查进程是否应关闭的事件。
        learner_client (services_pb2_grpc.LearnerServiceStub | None): 可选的预创建 stub。
        grpc_channel (grpc.Channel | None): 可选的预创建通道。
    """
    logging.info("[ACTOR] Start receiving parameters from the Learner")
    if not use_threads(cfg):
        # 创建进程专属的日志文件
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_policy_{os.getpid()}.log")

        # 使用显式的日志文件初始化日志记录
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor receive policy process logging initialized")

        # 设置进程处理器以处理关闭信号
        # 但使用主进程的 shutdown 事件
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        iterator = learner_client.StreamParameters(services_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )

    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")


def send_transitions(
    cfg: TrainRLServerPipelineConfig,
    transitions_queue: Queue,
    shutdown_event: Any,  # Event
    learner_client: "services_pb2_grpc.LearnerServiceStub | None" = None,
    grpc_channel: "grpc.Channel | None" = None,
) -> None:
    """向 learner 发送 transition。

    该函数持续从队列中取出消息并进行处理：

    - Transition 数据：
        - 收集一批 transition（观测、动作、奖励、下一观测）。
        - 将 transition 移动到 CPU 并使用 PyTorch 序列化。
        - 将序列化后的数据包装在 `services_pb2.Transition` 消息中并发送给 learner。

    Args:
        cfg (TrainRLServerPipelineConfig): actor 的配置。
        transitions_queue (Queue): 用于接收 transition 的队列。
        shutdown_event (Event): 用于检查进程是否应关闭的事件。
        learner_client (services_pb2_grpc.LearnerServiceStub | None): 可选的预创建 stub。
        grpc_channel (grpc.Channel | None): 可选的预创建通道。
    """

    if not use_threads(cfg):
        # 创建进程专属的日志文件
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_transitions_{os.getpid()}.log")

        # 使用显式的日志文件初始化日志记录
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor transitions process logging initialized")

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendTransitions(
            transitions_stream(
                shutdown_event, transitions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming transitions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Transitions process stopped")


def send_interactions(
    cfg: TrainRLServerPipelineConfig,
    interactions_queue: Queue,
    shutdown_event: Any,  # Event
    learner_client: "services_pb2_grpc.LearnerServiceStub | None" = None,
    grpc_channel: "grpc.Channel | None" = None,
) -> None:
    """向 learner 发送交互信息。

    该函数持续从队列中取出消息并进行处理：

    - 交互消息：
        - 包含有关 episode 奖励和策略计时的有用统计信息。
        - 消息使用 `pickle` 序列化并发送给 learner。

    Args:
        cfg (TrainRLServerPipelineConfig): actor 的配置。
        interactions_queue (Queue): 用于接收交互信息的队列。
        shutdown_event (Event): 用于检查进程是否应关闭的事件。
        learner_client (services_pb2_grpc.LearnerServiceStub | None): 可选的预创建 stub。
        grpc_channel (grpc.Channel | None): 可选的预创建通道。
    """

    if not use_threads(cfg):
        # 创建进程专属的日志文件
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_interactions_{os.getpid()}.log")

        # 使用显式的日志文件初始化日志记录
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor interactions process logging initialized")

        # 设置进程处理器以处理关闭信号
        # 但使用主进程的 shutdown 事件
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendInteractions(
            interactions_stream(
                shutdown_event, interactions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming interactions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Interactions process stopped")


def transitions_stream(
    shutdown_event: Any,  # Event
    transitions_queue: Queue,
    timeout: float,
) -> "Generator[Any, None, services_pb2.Empty]":
    while not shutdown_event.is_set():
        try:
            message = transitions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Transition queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message, services_pb2.Transition, log_prefix="[ACTOR] Send transitions"
        )

    return services_pb2.Empty()


def interactions_stream(
    shutdown_event: Any,  # Event
    interactions_queue: Queue,
    timeout: float,
) -> "Generator[Any, None, services_pb2.Empty]":
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Interaction queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message,
            services_pb2.InteractionMessage,
            log_prefix="[ACTOR] Send interactions",
        )

    return services_pb2.Empty()


#  策略函数


def update_policy_parameters(algorithm: RLAlgorithm, parameters_queue: Queue, device):
    """取出 learner 推送的最新权重并加载到 ``algorithm.policy`` 中。"""
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        logging.info("[ACTOR] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # TODO: 检查编码器参数同步可能存在的问题：
        # 1. 当 shared_encoder=True 时，我们从 actor 的 state_dict 加载的是过期的编码器参数，
        #    而不是来自 critic（单独优化的）更新后的编码器参数
        # 2. 当 freeze_vision_encoder=True 时，发送/加载被冻结的参数会浪费带宽
        # 3. 需要为 actor 和 discrete_critic 正确处理编码器参数
        # 可能的修复方案：
        # - 当 shared_encoder=True 时，发送 critic 的编码器状态
        # - 当 freeze_vision_encoder=True 时，完全跳过编码器参数
        # - 确保 discrete_critic 获得正确的编码器状态（目前使用的是 encoder_critic）
        algorithm.load_weights(state_dicts, device=device)


#  工具函数


def push_transitions_to_transport_queue(transitions: list, transitions_queue):
    """以较小的分块向 learner 发送 transition，以避免网络问题。

    Args:
        transitions: 要发送的 transition 列表
        message_queue: 用于向 learner 发送消息的队列
        chunk_size: 每个发送分块的大小
    """
    transition_to_send_to_learner = []
    for transition in transitions:
        tr = move_transition_to_device(transition=transition, device="cpu")
        for key, value in tr["state"].items():
            if torch.isnan(value).any():
                logging.warning(f"Found NaN values in transition {key}")

        transition_to_send_to_learner.append(tr)

    transitions_queue.put(transitions_to_bytes(transition_to_send_to_learner))


def get_frequency_stats(timer: TimerManager) -> dict[str, float]:
    """获取策略的频率统计信息。

    Args:
        timer (TimerManager): 包含已收集指标的计时器。

    Returns:
        dict[str, float]: 策略的频率统计信息。
    """
    stats = {}
    if timer.count > 1:
        avg_fps = timer.fps_avg
        p90_fps = timer.fps_percentile(90)
        logging.debug(f"[ACTOR] Average policy frame rate: {avg_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {p90_fps}")
        stats = {
            "Policy frequency [Hz]": avg_fps,
            "Policy frequency 90th-p [Hz]": p90_fps,
        }
    return stats


def log_policy_frequency_issue(policy_fps: float, cfg: TrainRLServerPipelineConfig, interaction_step: int):
    if policy_fps < cfg.env.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.env.fps} at step {interaction_step}"
        )


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.actor == "threads"


if __name__ == "__main__":
    actor_cli()
