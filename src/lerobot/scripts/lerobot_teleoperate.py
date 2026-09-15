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
通过遥操作控制机器人的简单脚本。

需要安装：pip install 'lerobot[hardware]'

示例：

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

若要将数据流式传输到 Foxglove 而不是 Rerun，请添加 ``--display_mode=foxglove``
（然后将 Foxglove 应用连接到 ``ws://127.0.0.1:8765``；可用 ``--display_port=<port>`` 覆盖端口）：

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true \
    --display_mode=foxglove
```

双臂 so100 的遥操作示例：

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
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
    gamepad,
    homunculus,
    keyboard,
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
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: 如果有更多机器人需要多个遥操作器（比如 lekiwi），最好在 teleop.py 和 record.py 中通过 List[Teleoperator] 支持这一点
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # 限制最大每秒帧数。
    fps: int = 60
    teleop_time_s: float | None = None
    # 在屏幕上显示所有相机画面
    display_data: bool = False
    # 当 display_data 为 True 时使用的可视化后端："rerun" 或 "foxglove"。
    display_mode: str = "rerun"
    # 对于 "rerun"：要发送到的远程服务器的 IP。对于 "foxglove"：WebSocket 服务器
    # 绑定的接口（127.0.0.1 仅限本地，0.0.0.0 为所有接口）。
    display_ip: str | None = None
    # 对于 "rerun"：远程服务器的端口。对于 "foxglove"：WebSocket 服务器绑定的端口。
    display_port: int | None = None
    # 是否显示压缩后的（JPEG）图像而不是原始帧
    display_compressed_images: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    display_mode: str = "rerun",
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    """
    该函数持续从遥操作设备读取动作，通过可选的流水线处理它们，将其发送给机器人，
    并可选地显示机器人的状态。该循环以指定频率运行，
    直到达到设定时长或被手动中断为止。

    参数：
        teleop: 提供控制动作的遥操作器设备实例。
        robot: 被控制的机器人实例。
        fps: 控制循环的目标频率，单位为每秒帧数。
        display_data: 若为 True，则获取机器人观测并在控制台和可视化后端中显示。
        display_mode: 当 display_data 为 True 时使用的可视化后端（"rerun" 或 "foxglove"）。
        display_compressed_images: 若为 True，则在将图像发送到后端显示前先进行压缩。
        duration: 遥操作循环的最大时长（秒）。若为 None，循环将无限运行。
        teleop_action_processor: 用于处理遥操作器原始动作的可选流水线。
        robot_action_processor: 在动作发送给机器人之前处理动作的可选流水线。
        robot_observation_processor: 用于处理机器人原始观测的可选流水线。
    """

    display_len = max(len(key) for key in robot.action_features)
    # 遥操作不写入数据集，因此错过截止时间只会损失控制平滑度。
    # 下方的实时读数是瞬时频率；当循环跟不上时，计时器会添加警告，
    # 并给出时间耗在何处的汇总。
    timer = CycleTimer(fps, records_data=False)
    start = time.perf_counter()
    try:
        while True:
            timer.tick()
            loop_start = time.perf_counter()  # 用于下方的实时读数

            with timer.section("observe"):
                # 获取机器人观测
                # 目前除了用于可视化外其实并不需要
                # 由于默认是恒等处理器，
                # teleop_action_processor 可以接受 None 作为观测
                obs = robot.get_observation()

                if robot.name == "unitree_g1":
                    teleop.send_feedback(obs)

            with timer.section("teleop"):
                # 获取遥操作动作
                raw_action = teleop.get_action()

                # 通过流水线处理遥操作动作
                teleop_action = teleop_action_processor((raw_action, obs))

                # 通过流水线处理发送给机器人的动作
                robot_action_to_send = robot_action_processor((teleop_action, obs))

            with timer.section("send"):
                # 将处理后的动作发送给机器人（robot_action_processor.to_output 应返回 RobotAction）
                _ = robot.send_action(robot_action_to_send)

            if display_data:
                with timer.section("telemetry"):
                    # 通过流水线处理机器人观测
                    obs_transition = robot_observation_processor(obs)

                    log_visualization_data(
                        display_mode,
                        observation=obs_transition,
                        action=teleop_action,
                        compress_images=display_compressed_images,
                    )

                    print("\n" + "-" * (display_len + 10))
                    print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                    # 显示已发送的最终机器人动作
                    for motor, value in robot_action_to_send.items():
                        print(f"{motor:<{display_len}} | {value:>7.2f}")
                    move_cursor_up(len(robot_action_to_send) + 3)

            timer.wait()
            loop_s = time.perf_counter() - loop_start
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

            if duration is not None and time.perf_counter() - start >= duration:
                return
    finally:
        # 放在 `finally` 中，这样 ^C（遥操作会话通常的结束方式）也能输出报告。
        timer.log_run_summary()


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_visualization(
            cfg.display_mode, session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            display_mode=cfg.display_mode,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
