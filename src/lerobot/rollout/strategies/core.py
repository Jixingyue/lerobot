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

"""Rollout 策略的抽象基类（ABC）以及共享的动作派发辅助函数。"""

from __future__ import annotations

import abc
import contextlib
import logging
from typing import TYPE_CHECKING

from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_visualization_data

from ..inference import InferenceEngine

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig
    from ..context import HardwareContext, ProcessorContext, RolloutContext, RuntimeContext

logger = logging.getLogger(__name__)


class RolloutStrategy(abc.ABC):
    """Rollout 执行策略的抽象基类。

    每个具体策略都实现一个自包含的控制循环，并带有各自的录制/交互语义。
    各策略之间是互斥的——每个会话只运行一个策略。这里也是第三方策略的
    扩展点：在一个已注册的 :class:`RolloutStrategyConfig` 旁边编写其子类，
    ``lerobot-rollout --strategy.type=<name>`` 便可以在不修改 LeRobot
    的情况下驱动它（参见 ``docs/source/inference.mdx`` 中的
    "Bring your own strategy"）。

    生命周期：先调用一次 ``setup()``，然后是 ``run()``，最后调用一次
    ``teardown()``。配置中声明 ``supports_interactive = True`` 的策略还可
    通过 ``--interactive=true`` 驱动，该模式会在每个启动/停止片段调用一次
    ``run()``。此类策略必须保证 ``run()`` 可重复启动：

    - 绝不要在 ``run()`` 中终结（finalize）数据集——那属于 ``teardown()``
      的职责；片段结束时至多保存一个不完整的尾部回合；
    - 需要跨片段保留的状态应存放在实例上，而不是 ``run()`` 的局部变量中
      （``CycleTimer`` 有意采取相反的做法，见 ``run()``）；
    - 绝不要绑定键盘/终端监听器——stdin 属于命令行提示符；
    - 每个节拍末尾调用一次 ``engine.pump_query(obs_processed)``，见
      ``run()``。

    一次性策略（``supports_interactive = False``，即默认值）则可以在
    ``run()`` 退出时自由地进行终结，例如通过 ``VideoEncodingManager``。
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config
        self._engine: InferenceEngine | None = None
        self._interpolator: ActionInterpolator | None = None
        self._warmup_flushed: bool = False
        self._cached_obs_processed: dict | None = None

    def _init_engine(self, ctx: RolloutContext) -> None:
        """挂载推理引擎和动作插值器，然后启动后端。

        根据配置中的 ``interpolation_multiplier`` 创建一个
        :class:`ActionInterpolator`，并启动推理引擎。
        请在 ``setup()`` 中调用本方法，以便各策略共享完全相同的
        初始化过程，而无需重复编写代码。
        """
        self._interpolator = ActionInterpolator(multiplier=ctx.runtime.cfg.interpolation_multiplier)
        self._engine = ctx.policy.inference
        logger.info("Starting inference engine...")
        self.reset_control_state()
        self._engine.start()
        self._warmup_flushed = False
        logger.info("Inference engine started")

    def reset_control_state(self) -> None:
        """清除回合作用域的控制状态，使暂停的会话能够干净地重启。

        重置推理引擎（策略隐藏状态、动作队列）、动作插值器以及缓存的已处理
        观测；节奏（pacing）状态保持不变。``RolloutController`` 会在每个
        运行片段之前于其服务线程上调用本方法。仅可在控制循环未运行时调用：
        这些重置不会与正在运行的循环同步。如果调用方在循环运行时重置控制
        状态——或者某个策略把自己的计时器提升到了实例上——还必须调用
        ``timer.restart()``。
        """
        if self._engine is not None:
            self._engine.reset()
        if self._interpolator is not None:
            self._interpolator.reset()
        self._cached_obs_processed = None

    def _process_observation_and_notify(self, processors: ProcessorContext, obs_raw: dict) -> dict:
        """运行观测处理器并通知引擎——该操作被节流到策略节拍上。

        调用方有责任在每次循环迭代时调用 ``robot.get_observation()``，从而使
        ``obs_raw`` 对动作后处理器保持新鲜。本辅助函数只对相对昂贵的部分
        ——处理器流水线和 ``engine.notify_observation``——进行门控，使其在
        插值器发出需要新动作的信号时才执行（每
        ``interpolation_multiplier`` 个节拍一次）。在被插值的节拍上，
        会复用缓存的 ``obs_processed``。

        当 ``interpolation_multiplier == 1`` 时，这与未节流的路径等价：
        ``needs_new_action()`` 每个节拍都为 True。

        每当调用 ``interpolator.reset()``（预热完成、DAgger 阶段切换回
        AUTONOMOUS）时，缓存会被隐式失效，因为 reset 会让
        ``needs_new_action()`` 在下一次调用时返回 True。
        """
        if self._cached_obs_processed is None or self._interpolator.needs_new_action():
            obs_processed = processors.robot_observation_processor(obs_raw)
            self._engine.notify_observation(obs_processed)
            self._cached_obs_processed = obs_processed
        return self._cached_obs_processed

    def _handle_warmup(self, use_torch_compile: bool, timer: CycleTimer) -> bool:
        """处理 torch.compile 预热阶段。

        如果调用方应当 ``continue``（仍在预热中）则返回 ``True``。
        预热节拍通过 *timer* 控制节奏，使循环的节拍保持锚定。在预热
        结束后的第一次迭代中，引擎和插值器会被重置，以丢弃陈旧的
        预热状态。
        """
        engine = self._engine
        interpolator = self._interpolator
        if not use_torch_compile:
            return False
        if not engine.ready:
            timer.wait()
            return True
        if not self._warmup_flushed:
            logger.info("Warmup complete — flushing stale state and resuming engine")
            engine.reset()
            interpolator.reset()
            timer.restart()
            self._warmup_flushed = True
            engine.resume()
        return False

    def _teardown_hardware(self, hw: HardwareContext, return_to_initial_position: bool = True) -> None:
        """停止推理引擎，可选地让机器人回到初始位置，然后断开硬件连接。"""
        if self._engine is not None:
            logger.info("Stopping inference engine...")
            self._engine.stop()
        robot = hw.robot_wrapper.inner
        if robot.is_connected:
            if return_to_initial_position and hw.initial_position:
                logger.info("Returning robot to initial position before shutdown...")
                self.return_to_initial_position(hw)
            elif not return_to_initial_position:
                logger.info(
                    "Skipping return-to-initial-position (disabled by config); leaving robot in final pose."
                )
            logger.info("Disconnecting robot...")
            robot.disconnect()
        teleop = hw.teleop
        if teleop is not None and teleop.is_connected:
            logger.info("Disconnecting teleoperator...")
            teleop.disconnect()

    @staticmethod
    def return_to_initial_position(hw: HardwareContext, duration_s: float = 3.0, fps: int = 50) -> bool:
        """平滑地将机器人插值回其初始位置。

        插值完成时返回 ``True``；中途失败时返回 ``False``——此时机器人
        处于任意位姿，因此在返回 ``False`` 时，调用方不得报告重置已完成。
        """
        robot = hw.robot_wrapper
        target = hw.initial_position
        try:
            current_obs = robot.get_observation()
            current_pos = {k: v for k, v in current_obs.items() if k in target}
            steps = max(int(duration_s * fps), 1)
            for step in range(1, steps + 1):
                t = step / steps
                interp = {}
                for k in current_pos:
                    interp[k] = current_pos[k] * (1 - t) + target[k] * t
                robot.send_action(interp)
                precise_sleep(1 / fps)
        except Exception as e:
            logger.warning("Could not return to initial position: %s", e)
            return False
        return True

    @staticmethod
    def _log_telemetry(
        obs_processed: dict | None,
        action_dict: dict | None,
        runtime_ctx: RuntimeContext,
    ) -> None:
        """如果启用了 display_data，则将观测/动作遥测数据记录到可视化后端。"""
        cfg = runtime_ctx.cfg
        if not cfg.display_data:
            return
        log_visualization_data(
            cfg.display_mode,
            observation=obs_processed,
            action=action_dict,
            compress_images=cfg.display_compressed_images,
        )

    def setup(self, ctx: RolloutContext) -> None:
        """策略专属的初始化（键盘监听器、缓冲区等）。

        默认实现只挂载并启动推理引擎；覆写本方法时必须首先调用
        ``self._init_engine(ctx)``（或 ``super().setup(ctx)``）。
        """
        self._init_engine(ctx)

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """主 rollout 循环。在收到关闭请求或达到持续时长时返回。

        实现方必须在进入循环之前调用 ``engine.resume()``（异步后端启动时
        处于暂停状态，而交互式控制器会在每个片段结束时再次暂停），并在每个
        节拍末尾调用 ``engine.pump_query(obs_processed)``——文本查询通道
        只有通过它才能向前推进，而且耗时数秒的生成绝不能位于动作路径中。

        每次调用 ``run()`` 都会构建自己的 ``CycleTimer``，并在其
        ``finally`` 中通过 ``timer.log_run_summary()`` 上报：全新计时器的
        启动豁免正好可以吸收 ``reset_control_state()`` 在每次 ``/start``
        时重新预置（re-prime）的插值器，而且每个片段都会得到各自的节拍
        报告。
        """

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """清理工作：终结数据集、停止线程、断开硬件连接。"""


# ---------------------------------------------------------------------------
# 共享辅助函数
# ---------------------------------------------------------------------------


def safe_push_to_hub(dataset, tags=None, private=False) -> bool:
    """将数据集推送到 hub；如果尚未保存任何回合则跳过。

    如果尝试了推送则返回 ``True``，如果跳过则返回 ``False``。
    """
    if dataset.num_episodes == 0:
        logger.warning("No episodes saved — skipping push to hub")
        return False
    dataset.push_to_hub(tags=tags, private=private)
    return True


def estimate_max_episode_seconds(
    dataset_features: dict,
    fps: float,
    target_size_mb: float = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
) -> float:
    """保守地估计多少秒的视频会超过 *target_size_mb*。

    每个相机会产生各自的视频文件，因此回合时长由**最慢**的、填满
    ``target_size_mb`` 的相机决定——即每帧像素数最少（码率最低）的
    那个相机。

    这里刻意使用**偏低**的每像素比特数估计，使计算出的时长*长于*
    实际情况。当计时器触发时，实际视频文件必定已经超过目标大小，
    这使得回合边界与数据集的视频文件分块对齐——每次
    ``push_to_hub`` 上传的都是完整文件，而不是重复上传一个仍在
    增长的文件。

    该估计有意忽略了编解码器相关的设置（CRF、preset）：我们只需要
    码率的粗略下界，而不是精确预测。

    当不存在视频特征时，回退为 300 秒（5 分钟）。
    """
    # 对于机器人画面的 CRF-30 流式视频，0.1 bits-per-pixel 是一个*偏低*
    # 的估计（实际通常为 0.1 – 0.3 bpp）。低估码率会高估时长 → 保存时
    # 回合会*大于* target_size_mb，这正是我们想要的。
    conservative_bpp = 0.1

    # 收集每个相机的像素数——每个相机都有各自的视频文件。
    camera_pixels = []
    for feat in dataset_features.values():
        if feat.get("dtype") == "video":
            shape = feat.get("shape", ())

            # (H, W, C)——bits-per-pixel 是按空间像素计的指标，
            # 因此计数时要排除通道维度。
            if len(shape) == 3:
                pixels = shape[0] * shape[1]
                camera_pixels.append(pixels)
            else:
                raise ValueError(f"Unexpected video feature shape: {shape}")

    if not camera_pixels:
        return 300.0

    # 使用最小的相机：它产生的码率最低，因此达到目标所需的时间最长
    # ——这是保守的选择。
    min_pixels = min(camera_pixels)
    bits_per_frame = min_pixels * conservative_bpp
    bytes_per_second = (bits_per_frame * fps) / 8

    # 以防万一，防止除以零
    if bytes_per_second <= 0:
        return 300.0

    return (target_size_mb * 1024 * 1024) / bytes_per_second


# ---------------------------------------------------------------------------
# 共享的动作派发辅助函数
# ---------------------------------------------------------------------------


def send_next_action(
    obs_processed: dict,
    obs_raw: dict,
    ctx: RolloutContext,
    interpolator: ActionInterpolator,
    timer: CycleTimer | None = None,
) -> dict | None:
    """向机器人派发下一个动作。

    从推理引擎取出下一个动作张量，送入插值器，并将插值后的动作通过
    ``robot_action_processor`` 发送给机器人。对同步和异步后端的工作
    方式完全相同——rollout 策略无需做任何分支判断。

    当提供 *timer* 时，从引擎取动作和向机器人发送动作会分别作为其节拍
    摘要中的 ``infer`` 和 ``send`` 步骤计时，没有动作可发送的节拍也会
    在其中计数。注意，在异步后端上 ``infer`` 只是一次队列拉取——推理在
    线程外运行，因此其延迟表现为"饥饿节拍"（starved ticks），而不是
    循环体耗时。

    返回已发送的动作字典；如果没有就绪的动作（例如异步队列为空、
    插值器尚未预置），则返回 ``None``。
    """
    engine = ctx.policy.inference
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys
    # ``nullcontext`` 接受（并忽略）区段名称，因此在未传入 timer 时，
    # 它可以原样替代 ``timer.section``。
    section = timer.section if timer is not None else contextlib.nullcontext

    if interpolator.needs_new_action():
        with section("infer"):
            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
            action_tensor = engine.get_action(obs_frame)
        if action_tensor is not None:
            interpolator.add(action_tensor.cpu())

    interp = interpolator.get()
    if interp is None:
        if timer is not None:
            timer.note_starved_tick()
        return None

    if len(interp) != len(ordered_keys):
        raise ValueError(f"Interpolated tensor length ({len(interp)}) != action keys ({len(ordered_keys)})")
    action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
    with section("send"):
        processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
        ctx.hardware.robot_wrapper.send_action(processed)
    return action_dict
