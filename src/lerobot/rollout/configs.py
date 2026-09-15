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

"""rollout 部署引擎的配置数据类。"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import ClassVar

import draccus

from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available

from .inference import InferenceEngineConfig, SyncInferenceConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 策略配置（通过 draccus ChoiceRegistry 进行多态分发）
# ---------------------------------------------------------------------------


@dataclass
class RolloutStrategyConfig(draccus.ChoiceRegistry, abc.ABC):
    """rollout 策略配置的抽象基类。

    在 CLI 上使用 ``--strategy.type=<name>`` 来选择策略。注册表是
    开放的：第三方包可以注册自己的策略并通过同一个标志来驱动它——
    参见 ``docs/source/inference.mdx`` 中的 "Bring your own strategy"。

    下面的 ClassVar 和钩子声明了引擎代表策略所安排的事项，
    这样策略之外的任何代码都无需知道其具体类型。
    """

    # 该策略是否遵守 ``--interactive=true`` 所要求的可重启 run() 契约
    # （参见 ``RolloutStrategy``）。
    supports_interactive: ClassVar[bool] = False
    # "none"：拒绝任何 --dataset.* 标志。"optional"：当给出 --dataset.* 标志时
    # 创建数据集（``ctx.data.dataset`` 可能为 None）。"required"：
    # --dataset.repo_id 是必填项。
    dataset_mode: ClassVar[str] = "none"
    # --teleop.type 是否为必填（人在回路中的策略）。
    requires_teleop: ClassVar[bool] = False

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    def requires_streaming_encoding(self) -> bool:
        """是否必须强制开启 ``--dataset.streaming_encoding``。

        当帧是在定时控制循环内部写入时返回 True，因为阻塞式编码
        会破坏节奏。用方法而不是 ClassVar，是为了让结果可以
        依赖策略自身的字段（参见 DAgger）。
        """
        return False

    def extra_dataset_features(self) -> dict[str, dict]:
        """策略拥有的数据集列，会合并到机器人/策略特征中。

        每个记录的帧都必须包含这里声明的所有键。
        """
        return {}


@RolloutStrategyConfig.register_subclass("base")
@dataclass
class BaseStrategyConfig(RolloutStrategyConfig):
    """不记录数据的自主 rollout。"""

    supports_interactive: ClassVar[bool] = True


@RolloutStrategyConfig.register_subclass("sentry")
@dataclass
class SentryStrategyConfig(RolloutStrategyConfig):
    """持续自主 rollout，始终开启记录。

    回合时长由相机分辨率、FPS 和 ``target_video_file_size_mb``
    推导得出，使每个保存的回合产生的视频文件恰好超过目标大小。
    这样回合边界与数据集的视频文件分块对齐，每次 ``push_to_hub``
    调用上传的都是完整的视频文件，而不是重新上传一个尚未越过
    分块边界的不断增大的文件。
    """

    supports_interactive: ClassVar[bool] = True
    dataset_mode: ClassVar[str] = "required"

    upload_every_n_episodes: int = 5
    # 用于回合轮换的目标视频文件大小（MB）。当估计的视频时长
    # 将超过此限制时保存回合。
    # 设为 None 时默认为 DEFAULT_VIDEO_FILE_SIZE_IN_MB。
    target_video_file_size_mb: int | None = None

    def requires_streaming_encoding(self) -> bool:
        return True


@RolloutStrategyConfig.register_subclass("highlight")
@dataclass
class HighlightStrategyConfig(RolloutStrategyConfig):
    """通过环形缓冲区实现按需记录的自主 rollout。

    一个内存受限的环形缓冲区持续捕获遥测数据。当用户
    按下保存键时，缓冲区内容会被刷入数据集，并持续进行
    实时记录，直到再次按下该键为止。
    """

    dataset_mode: ClassVar[str] = "required"

    ring_buffer_seconds: float = 10.0
    ring_buffer_max_memory_mb: int = 1024
    save_key: str = "s"
    push_key: str = "h"

    def requires_streaming_encoding(self) -> bool:
        return True


@dataclass
class DAggerKeyboardConfig:
    """DAgger 控制的键盘按键绑定。

    按键以单个字符（例如 ``"c"``、``"h"``）或
    特殊键名（``"space"``）指定。
    """

    pause_resume: str = "space"
    correction: str = "tab"
    upload: str = "enter"


@dataclass
class DAggerPedalConfig:
    """DAgger 控制的脚踏板配置。

    踏板代码是 evdev 键码字符串（例如 ``"KEY_A"``）。
    """

    device_path: str = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
    pause_resume: str = "KEY_A"
    correction: str = "KEY_B"
    upload: str = "KEY_C"


@RolloutStrategyConfig.register_subclass("episodic")
@dataclass
class EpisodicStrategyConfig(RolloutStrategyConfig):
    """面向回合的记录方式，行为与 ``lerobot-record`` 一致。

    记录 ``dataset.num_episodes`` 个回合，每个回合最长 ``dataset.episode_time_s`` 秒。
    每个回合结束后，运行 ``dataset.reset_time_s`` 秒的重置时间。

    键盘控制：
        右方向键  — 提前结束当前回合或重置阶段
        左方向键  — 丢弃当前回合并重新记录
        Escape    — 停止记录会话

    在回合之间：
    - 如果没有遥操作主手，机器人保持在启动时捕获的初始关节位置。
    - 否则，机器人平滑移动到遥操作主手的位置。
    """

    dataset_mode: ClassVar[str] = "required"

    # 仅当未指定遥操作主手时适用。
    # 为 True（默认）时，将机器人移回启动时捕获的关节位置。
    # 否则，让机器人保持在当前位置。
    reset_to_initial_position: bool = True

    # 是否开启主手 -> 从手的平滑交接行为。
    # 为 False 时，回退到从手 -> 主手的交接方式。
    # 注意：主手 -> 从手的交接仅在主手具备 `send_feedback` 能力时受支持。
    smooth_leader_to_follower_handover: bool = True

    # 是否在重置阶段开始时开启平滑交接行为：
    # 主手被驱动到从手位置（带驱动的遥操作设备，
    # 参见 `smooth_leader_to_follower_handover`），或者从手被
    # 滑动到遥操作姿态（无驱动的遥操作设备）。对于离合器式
    # 遥操作设备（例如 VR 控制器）应禁用，这类设备在接入时会以
    # 当前机器人姿态为基准重新对齐：那里的交接本来就是连续的，
    # 阻塞式插值只会延迟重置阶段的开始。
    smooth_handover: bool = True


@RolloutStrategyConfig.register_subclass("dagger")
@dataclass
class DAggerStrategyConfig(RolloutStrategyConfig):
    """人在回路的数据采集（DAgger / RaC）。

    在自主策略执行和人工干预之间交替。
    干预帧会被标记为 ``intervention=True``。

    输入通过键盘或脚踏板控制，由 ``input_device`` 选择。
    每种设备都提供三个动作：

    1. **pause_resume** — 切换策略执行的开/关。
    2. **correction** — 切换人工纠正记录的开/关。
    3. **upload** — 按需将数据集推送到 hub（仅纠正模式）。

    当 ``record_autonomous=False``（默认）时，只记录人工纠正窗口——
    每次纠正成为独立的回合。设为 ``True`` 则同时记录自主帧和纠正帧，
    并采用基于大小的回合轮换（与 Sentry 相同）和后台上传。
    纠正进行中时 ``push_to_hub`` 会被阻塞。
    """

    # TODO(Steven): DAgger 本不应要求数据集（用户可能只想 rollout+干预
    # 而不记录），但目前为了简化实现仍要求提供数据集。
    dataset_mode: ClassVar[str] = "required"
    requires_teleop: ClassVar[bool] = True

    # 要收集的纠正回合数（仅纠正模式）。
    # 为 None 时，回退到 ``--dataset.num_episodes``。
    num_episodes: int | None = None
    record_autonomous: bool = False
    upload_every_n_episodes: int = 5
    # 用于回合轮换的目标视频文件大小（MB）（仅限 record_autonomous
    # 模式）。为 None 时默认为 DEFAULT_VIDEO_FILE_SIZE_IN_MB。
    target_video_file_size_mb: int | None = None
    # 是否在阶段切换时开启平滑交接行为：
    # 暂停时主手被驱动到从手位置（具备 `send_feedback` 能力的
    # 遥操作设备），纠正开始时从手被滑动到遥操作姿态
    # （无驱动的遥操作设备）。对于离合器式遥操作设备
    # （例如 VR 控制器）应禁用，这类设备在接入时会以当前机器人
    # 姿态为基准重新对齐：那里的交接本来就是连续的，
    # 阻塞式插值只会延迟纠正的开始。
    smooth_handover: bool = True
    input_device: str = "keyboard"
    keyboard: DAggerKeyboardConfig = field(default_factory=DAggerKeyboardConfig)
    pedal: DAggerPedalConfig = field(default_factory=DAggerPedalConfig)

    def __post_init__(self):
        if self.input_device not in ("keyboard", "pedal"):
            raise ValueError(f"DAgger input_device must be 'keyboard' or 'pedal', got '{self.input_device}'")

    def requires_streaming_encoding(self) -> bool:
        # 仅当自主阶段也被记录时才需要；纠正在阶段之间保存。
        return self.record_autonomous

    def extra_dataset_features(self) -> dict[str, dict]:
        return {"intervention": {"dtype": "bool", "shape": (1,), "names": None}}


# ---------------------------------------------------------------------------
# 顶层 rollout 配置
# ---------------------------------------------------------------------------


@dataclass
class RolloutConfig:
    """``lerobot-rollout`` CLI 的顶层配置。

    组合了硬件、策略、运行时等设置。
    ``__post_init__`` 方法执行快速失败校验，
    尽早拒绝无效的标志组合。
    """

    # 硬件
    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None

    # 策略（通过 __post_init__ 从 --policy.path 加载）
    policy: PreTrainedConfig | None = None

    # 策略（多态：--strategy.type=base|sentry|highlight|dagger|episodic，
    # 或第三方包注册的任何名称）
    strategy: RolloutStrategyConfig = field(default_factory=BaseStrategyConfig)

    # 推理后端（多态：--inference.type=sync|rtc）
    inference: InferenceEngineConfig = field(default_factory=SyncInferenceConfig)

    # 数据集（根据策略的 ``dataset_mode`` 为必填、可选或拒绝）
    dataset: DatasetRecordConfig | None = None

    # 运行时
    fps: float = 30.0
    # 运行时长（秒）；0 = 无限（24/7 模式）。在交互模式下，
    # 它限制每个 /start 片段，而不是整个会话。
    duration: float = 0.0
    # 通过 stdin 以聊天式命令（/start、/subtask、/vqa、/autosteer、
    # /reset、/stop）控制 rollout，同时硬件和策略保持热状态。
    # 在 /start 之前机器人不会移动，会话期间低于 ERROR 级别的
    # 日志会被静音。
    interactive: bool = False
    # /autosteer：两次 "what is the next subtask?" 查询之间的机器人运动
    # 秒数，从子任务生效时刻开始计算。值越小重新规划越早，
    # 但循环中用于生成文本而非执行动作的时间占比越高。
    autosteer_interval_s: float = 10.0
    # 每个策略动作发送的机器人指令数。值大于 1 时会在相邻策略动作
    # 之间线性插值以获得更平滑的运动：指令以 ``fps × multiplier`` Hz
    # 发送给机器人，而策略推理和数据集记录仍保持 ``fps`` Hz。
    interpolation_multiplier: int = 1
    device: str | None = None
    task: str = ""
    display_data: bool = False
    # display_data 为 True 时使用的可视化后端："rerun" 或 "foxglove"。
    display_mode: str = "rerun"
    # 对于 "rerun"：要发送到的远程服务器 IP。对于 "foxglove"：WebSocket
    # 服务器绑定的接口（127.0.0.1 仅限本地，0.0.0.0 为所有接口）。
    display_ip: str | None = None
    # 对于 "rerun"：远程服务器的端口。对于 "foxglove"：WebSocket 服务器绑定的端口。
    display_port: int | None = None
    # 是否显示压缩（JPEG）图像而不是原始帧
    display_compressed_images: bool = False
    # 使用语音合成朗读事件
    play_sounds: bool = True
    resume: bool = False
    # 用于将机器人/数据集观测键映射到策略键的重命名映射
    rename_map: dict[str, str] = field(default_factory=dict)

    # 硬件关闭清理
    # 为 True（默认）时，在断开连接前平滑插值将机器人移回
    # 启动时捕获的关节位置。设为 False 则在关闭时
    # 让机器人保持在最终到达的姿态。
    return_to_initial_position: bool = True

    # Torch compile
    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    compile_warmup_inferences: int = 2

    def __post_init__(self):
        """校验配置不变量，并从 ``--policy.path`` 加载策略配置。"""
        if self.interpolation_multiplier < 1:
            raise ValueError(f"interpolation_multiplier must be >= 1, got {self.interpolation_multiplier}")

        # --- 策略能力 ---
        # 只读取策略的声明，绝不检查其具体类型，这样
        # 第三方策略与内置策略的校验方式完全一致。
        strategy = self.strategy
        if strategy.requires_teleop and self.teleop is None:
            raise ValueError(f"{strategy.type} strategy requires --teleop.type to be set")
        if strategy.dataset_mode == "required" and self.dataset is None:
            raise ValueError(f"{strategy.type} strategy requires --dataset.repo_id to be set")
        if strategy.dataset_mode == "none" and self.dataset is not None:
            raise ValueError(
                f"{strategy.type} strategy does not record data: drop the --dataset.* flags "
                "or pick a recording strategy."
            )
        if self.dataset is not None and not self.dataset.repo_id:
            raise ValueError("--dataset.repo_id must be set when passing --dataset.* flags")

        # 交互模式对每个片段调用一次 strategy.run()，因此只有声明了
        # ``supports_interactive`` 的策略才能由它驱动。
        if self.interactive and not self.strategy.supports_interactive:
            supported = " or ".join(
                sorted(
                    name
                    for name, choice_cls in RolloutStrategyConfig.get_known_choices().items()
                    if choice_cls.supports_interactive
                )
            )
            raise ValueError(
                f"--interactive=true supports --strategy.type={supported} (got '{self.strategy.type}')."
            )

        if self.autosteer_interval_s < 0:
            raise ValueError(f"--autosteer_interval_s must be >= 0 (got {self.autosteer_interval_s}).")

        # 在定时控制循环内部写帧的策略承受不起阻塞式编码：
        # 代表它强制开启流式编码。
        if (
            self.dataset is not None
            and strategy.requires_streaming_encoding()
            and not self.dataset.streaming_encoding
        ):
            logger.warning("%s strategy forces streaming_encoding=True", strategy.type)
            self.dataset.streaming_encoding = True

        # --- 策略加载 ---
        if self.robot is None:
            raise ValueError("--robot.type is required for rollout")

        policy_path = parser.get_path_arg("policy")
        if policy_path:
            yaml_overrides = parser.get_yaml_overrides("policy")
            cli_overrides = parser.get_cli_overrides("policy") or []
            policy_overrides = yaml_overrides + cli_overrides
            pretrained_revision = parser.parse_arg("pretrained_revision", cli_overrides)
            if pretrained_revision is None:
                pretrained_revision = parser.parse_arg("pretrained_revision", yaml_overrides)
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path,
                revision=pretrained_revision,
                cli_overrides=policy_overrides,
            )
            self.policy.pretrained_path = policy_path
        if self.policy is None:
            raise ValueError("--policy.path is required for rollout")

        # --- 任务解析 ---
        # 当传入任何 --dataset.* 标志时，draccus 会创建一个 single_task="" 的 DatasetRecordConfig。
        # 如果用户通过顶层 --task 标志设置了任务，则将其传播下去，
        # 使所有下游使用者（推理引擎、数据集帧构建器）都能看到它。
        if self.dataset is not None and not self.dataset.single_task and self.task:
            logger.info("Propagating top-level task '%s' to dataset config", self.task)
            self.dataset.single_task = self.task
        elif self.dataset is not None and self.dataset.single_task and not self.task:
            logger.info("Propagating dataset single_task '%s' to top-level task", self.dataset.single_task)
            self.task = self.dataset.single_task

        # --- 设备解析 ---
        # 未显式设置时从策略配置解析设备，使所有组件
        # （policy.to、预处理器、推理引擎）使用相同的
        # 设备字符串，而不是不一致的回退值。
        if self.device is None or not is_torch_device_available(self.device):
            resolved = self.policy.device
            if resolved:
                self.device = resolved
                logger.info("Resolved device from policy config: %s", self.device)
            else:
                self.device = auto_select_torch_device().type
                logger.info("No policy config to resolve device from; auto-selected device: %s", self.device)

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]
