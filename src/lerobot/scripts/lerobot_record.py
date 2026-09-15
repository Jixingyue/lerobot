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

"""
通过遥操作（teleoperation）录制数据集。这是一个纯粹的数据采集
工具——不包含任何策略推理。如需部署训练好的策略，请改用
``lerobot-rollout``。

需要：pip install 'lerobot[core_scripts]'  （包含 dataset、hardware 和 viz 附加依赖）

示例：

```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --display_data=true
```

如果要将数据流式传输到 Foxglove 而不是 Rerun，请添加 ``--display_mode=foxglove``（然后将
Foxglove 应用连接到 ``ws://127.0.0.1:8765``；可通过 ``--display_port=<port>`` 覆盖端口）。

使用双臂 so100 录制的示例：
```shell
lerobot-record \\
  --robot.type=bi_so_follower \\
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \\
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \\
  --robot.id=bimanual_follower \\
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \\
  --teleop.type=bi_so_leader \\
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \\
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \\
  --teleop.id=bimanual_leader \\
  --display_data=true \\
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \\
  --dataset.num_episodes=25 \\
  --dataset.single_task="Grab and handover the red cube to the other arm" \\
  --dataset.streaming_encoding=true \\
  --dataset.encoder_threads=2
```

使用自定义视频编码参数录制的示例：
```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --dataset.rgb_encoder.vcodec=h264 \\
    --dataset.rgb_encoder.preset=fast \\
    --dataset.rgb_encoder.extra_options={"tune": "film", "profile:v": "high", "bf": 2} \\
    --display_data=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # 用于控制机器人的遥操作器（必填）
    teleop: TeleoperatorConfig | None = None
    # 在屏幕上显示所有相机画面
    display_data: bool = False
    # display_data 为 True 时使用的可视化后端："rerun" 或 "foxglove"。
    display_mode: str = "rerun"
    # 对于 "rerun"：要发送到的远程服务器 IP。对于 "foxglove"：WebSocket
    # 服务器绑定的网卡接口（127.0.0.1 表示仅本地，0.0.0.0 表示所有接口）。
    display_ip: str | None = None
    # 对于 "rerun"：远程服务器的端口。对于 "foxglove"：WebSocket 服务器绑定的端口。
    display_port: int | None = None
    # 是否显示压缩（JPEG）图像而不是原始帧
    display_compressed_images: bool = False
    # 使用语音合成播报事件。
    play_sounds: bool = True
    # 在已有数据集上继续录制。
    resume: bool = False

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError(
                "A teleoperator is required for recording. "
                "Use --teleop.type=... to specify one. "
                "For policy-based deployment, use lerobot-rollout instead."
            )


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     [ Teleoperator ]
     |
     |  [teleop.get_action] -> raw_action
     |          |
     |          V
     | [teleop_action_processor]
     |          |
     '---> processed_teleop_action
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # 在遥操作器之后运行
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # 在机器人之前运行
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # 在机器人之后运行
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_mode: str = "rerun",
    display_compressed_images: bool = False,
    timer: CycleTimer | None = None,
):
    """以 *fps* 的频率由遥操作器驱动机器人，并可选择录制每一帧。

    *timer* 让运行多个阶段的调用方能够在所有阶段之间共用同一个
    :class:`~lerobot.utils.cycle_timer.CycleTimer`——:func:`record` 每次调用
    录制一个 episode，两次调用之间还有一个不录制的重置阶段——从而使节拍
    统计覆盖整个会话，并按 episode 报告。如果不提供，每次调用都会获得一个
    私有的计时器：节奏和慢循环警告完全相同，只是没有运行结束时的汇总，
    因为单个阶段不存在可供汇总的完整运行。
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    if timer is None:
        timer = CycleTimer(fps, records_data=dataset is not None)

    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        # 在 `tick()` 之前检查：本次迭代不是一个控制节拍，因此不应
        # 将其计入节拍计时。
        if events["exit_early"]:
            events["exit_early"] = False
            break

        timer.tick()

        with timer.section("observe"):
            # 获取机器人观测
            obs = robot.get_observation()

        with timer.section("process_obs"):
            # 对原始机器人观测应用流水线，默认为 IdentityProcessor
            obs_processed = robot_observation_processor(obs)

            if dataset is not None:
                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        with timer.section("teleop"):
            # 从遥操作器获取动作
            if isinstance(teleop, Teleoperator):
                act = teleop.get_action()
                if robot.name == "unitree_g1":
                    teleop.send_feedback(obs)

                # 对原始遥操作动作应用流水线，默认为 IdentityProcessor
                act_processed_teleop = teleop_action_processor((act, obs))
                action_values = act_processed_teleop
                robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

            elif isinstance(teleop, list):
                arm_action = teleop_arm.get_action()
                arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
                keyboard_action = teleop_keyboard.get_action()
                base_action = robot._from_keyboard_to_base_action(keyboard_action)
                act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
                act_processed_teleop = teleop_action_processor((act, obs))
                action_values = act_processed_teleop
                robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
            else:
                robot_action_to_send = None
                no_action_count += 1
                if no_action_count == 1 or no_action_count % 10 == 0:
                    logging.warning(
                        "No teleoperator provided, skipping action generation. "
                        "This is likely to happen when resetting the environment without a teleop device. "
                        "The robot won't be at its rest position at the start of the next episode."
                    )

        # 没有需要发送的内容，也没有需要录制的内容，但该阶段仍然需要保持节拍，
        # 也仍然需要结束：过去直接 `continue` 跳过循环体剩余部分的做法，会在
        # `control_time_s` 始终不推进时以满 CPU 速度空转。
        if robot_action_to_send is None:
            timer.wait()
            timestamp = time.perf_counter() - start_episode_t
            continue

        with timer.section("send"):
            # 向机器人发送动作
            # 动作最终可能会通过 `max_relative_target` 被裁剪，
            # 因此实际发送的动作会被保存到数据集中。action = postprocessor.process(action)
            # TODO(steven, pepijn, adil)：我们应当使用一个流水线步骤来裁剪动作，
            # 从而使发送的动作就是我们输入给机器人的动作。
            _sent_action = robot.send_action(robot_action_to_send)

        # 写入数据集
        if dataset is not None:
            with timer.section("record"):
                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": single_task}
                dataset.add_frame(frame)

        if display_data:
            with timer.section("telemetry"):
                log_visualization_data(
                    display_mode,
                    observation=obs_processed,
                    action=action_values,
                    compress_images=display_compressed_images,
                )

        timer.wait()

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_visualization(
            cfg.display_mode, session_name="recording", ip=cfg.display_ip, port=cfg.display_port
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # 当调用方未提供处理器时，回退到恒等（identity）流水线。
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn)：将来这应当来自遥操作器或策略
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    # 整个会话共用一个计时器，这样它的统计描述的是整个录制过程，
    # 而不是其中某一个 episode 的片段。下面的重置阶段特意使用各自
    # 私有的计时器运行：它们不写入任何帧，因此把它们的节拍折算进来
    # 会稀释所有回答“我是否按 `fps` 录制？”的数值。
    timer = CycleTimer(cfg.dataset.fps)

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # 拒绝 eval_ 前缀——策略评估请使用 lerobot-rollout
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        # 先连接遥操作器，再连接机器人，这样在遥操作器初始化期间机器人不会
        # 一直处于空闲状态（并可能触发固件看门狗）。与 lerobot_teleoperate.py 一致。
        if teleop is not None:
            teleop.connect()
        robot.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.rgb_encoder.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                episode_index = dataset.num_episodes
                log_say(f"Recording episode {episode_index}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_mode=cfg.display_mode,
                    display_compressed_images=display_compressed_images,
                    timer=timer,
                )

                # 在不录制的情况下运行几秒钟，留出时间手动重置环境
                # 对最后一个待录制的 episode 跳过重置
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)

                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_mode=cfg.display_mode,
                        display_compressed_images=display_compressed_images,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    timer.log_episode_summary("discarded episode")
                    timer.restart()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                # 在此关闭刚刚保存的 episode 的统计窗口。摘要会在下一个
                # episode 的第一个节拍时发出，因此重置阶段、`save_episode`
                # 以及其间的语音提示都会被排除在节拍统计之外，而不会被算到
                # 它们恰好相邻的某个 episode 头上。随后 `restart()` 会把
                # 第一个节拍也豁免掉，因为那时相机已经空闲了数秒。
                timer.log_episode_summary(f"episode {episode_index}")
                timer.restart()
    finally:
        # 首先（并且放在 `finally` 中）：大多数录制会话都是通过 ^C 结束的，
        # 而汇总在视频编码和 hub 上传把它刷走之前最有用。
        timer.log_run_summary()

        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if listener is not None:
            listener.stop()

        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
