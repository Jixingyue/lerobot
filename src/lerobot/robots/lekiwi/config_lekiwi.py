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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig

from ..config import RobotConfig


def lekiwi_cameras_config() -> dict[str, CameraConfig]:
    return {
        "front": OpenCVCameraConfig(
            index_or_path="/dev/video0",
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
            rotation=Cv2Rotation.ROTATE_180,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video2",
            fps=30,
            width=480,
            height=640,
            fourcc="MJPG",
            rotation=Cv2Rotation.ROTATE_90,
        ),
    }


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiConfig(RobotConfig):
    port: str = "/dev/ttyACM0"  # 连接总线的端口

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` 出于安全目的限制相对位置目标向量的大小。
    # 将其设为正标量可让所有电机使用相同的值，或者设为将电机名称
    # 映射到该电机 max_relative_target 值的字典。
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    # 为向后兼容以前的策略/数据集，请设为 `True`
    use_degrees: bool = True

    # 电机 `sync_read` 失败时的额外重试次数。Feetech 总线偶尔会
    # 返回损坏的状态包（"Incorrect status packet!"），尤其是在多个关节同时
    # 移动时，这会导致控制循环中止。重试是立即执行的（无休眠）且仅在
    # 失败时发生，因此稳态读取开销不变。
    num_read_retries: int = 2


@dataclass
class LeKiwiHostConfig:
    # 网络配置
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # 应用程序运行时长
    connection_time_s: int = 30

    # 看门狗：如果超过 0.5 秒未收到命令，则停止机器人。
    watchdog_timeout_ms: int = 500

    # 如果机器人出现抖动，请降低频率，并在命令行中用 `top` 监控 CPU 负载
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("lekiwi_client")
@dataclass
class LeKiwiClientConfig(RobotConfig):
    # 网络配置
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # 移动
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # 速度控制
            "speed_up": "r",
            "speed_down": "f",
            # 退出遥操作
            "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
