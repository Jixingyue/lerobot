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

"""支持可插拔 rollout 策略的策略部署引擎。

``lerobot-rollout`` 是在真实机器人上运行已训练策略的
唯一 CLI。

策略
----------
    --strategy.type=base       自主 rollout，不录制
    --strategy.type=sentry     持续录制并自动上传
    --strategy.type=highlight  环形缓冲区 + 按键保存
    --strategy.type=dagger     人在回路（DAgger / RaC）
    --strategy.type=episodic   面向 episode 的录制，带重置阶段
    --strategy.type=<name>     来自已安装的 ``lerobot_strategy_*``
                               包的任意策略（参见文档中的 "Bring your own strategy"）

推理后端
------------------
    --inference.type=sync      每个控制周期调用一次策略（默认）
    --inference.type=rtc       面向慢速 VLA 模型的实时分块（Real-Time Chunking）

使用示例
--------------
::

    # Base 模式 — 使用同步推理进行快速评估
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --task="pick up cube" --duration=30

    # 交互式会话：在输入 /start 之前机器人保持空闲，之后可通过
    # stdin 中的 /subtask、/vqa、/autosteer、/reset 和 /stop 控制运行。
    # 使用 --strategy.type=sentry 时还会录制，并为帧标注其任务。
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --task="pick up cube" \\
        --interactive=true

    # Base 模式 — 为慢速 VLA（Pi0、Pi0.5、SmolVLA）使用 RTC 推理
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/pi0_base \\
        --inference.type=rtc \\
        --inference.rtc.execution_horizon=10 \\
        --inference.rtc.max_guidance_weight=10.0 \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
        --task="pick up cube" --duration=60

    # Sentry 模式 — 持续录制并周期性上传
    lerobot-rollout \\
        --strategy.type=sentry \\
        --strategy.upload_every_n_episodes=5 \\
        --policy.path=lerobot/pi0_base \\
        --inference.type=rtc \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --dataset.repo_id=user/rollout_sentry_data \\
        --dataset.single_task="patrol" --duration=3600

    # Highlight 模式 — 环形缓冲区，按 's' 保存，按 'h' 推送
    lerobot-rollout \\
        --strategy.type=highlight \\
        --strategy.ring_buffer_seconds=30 \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --dataset.repo_id=user/rollout_highlight_data \\
        --dataset.single_task="pick up cube"

    # DAgger 模式 — 仅人在回路修正
    lerobot-rollout \\
        --strategy.type=dagger \\
        --strategy.num_episodes=20 \\
        --policy.path=outputs/pretrain/checkpoints/last/pretrained_model \\
        --robot.type=bi_openarm_follower \\
        --teleop.type=openarm_mini \\
        --dataset.repo_id=user/rollout_hil_data \\
        --dataset.single_task="Fold the T-shirt"

    # DAgger 模式 — 使用 RTC 推理持续录制
    lerobot-rollout \\
        --strategy.type=dagger \\
        --strategy.record_autonomous=true \\
        --strategy.num_episodes=50 \\
        --inference.type=rtc \\
        --inference.rtc.execution_horizon=10 \\
        --policy.path=user/my_pi0_policy \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --teleop.type=so101_leader \\
        --teleop.port=/dev/ttyACM1 \\
        --dataset.repo_id=user/rollout_dagger_rtc_data \\
        --dataset.single_task="Grasp the block"

    # 使用 Rerun 可视化和 torch.compile
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --task="pick up cube" --duration=60 \\
        --display_data=true \\
        --use_torch_compile=true

    # Episodic 模式 — 面向 episode 的录制，带重置阶段
    lerobot-rollout \\
        --strategy.type=episodic \\
        --policy.path=user/my_policy \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --teleop.type=so100_leader \\
        --teleop.port=/dev/ttyACM1 \\
        --dataset.repo_id=user/rollout_episodic_data \\
        --dataset.num_episodes=20 \\
        --dataset.single_task="Grab the cube"

    # 恢复之前的 sentry 录制会话
    lerobot-rollout \\
        --strategy.type=sentry \\
        --policy.path=user/my_policy \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --dataset.repo_id=user/rollout_sentry_data \\
        --dataset.single_task="patrol" \\
        --resume=true

    # 使用自定义视频编码参数进行 rollout
    lerobot-rollout \\
        --strategy.type=base \\
        --policy.path=lerobot/act_koch_real \\
        --robot.type=koch_follower \\
        --robot.port=/dev/ttyACM0 \\
        --task="pick up cube" --duration=60 \\
        --display_data=true \\
        --dataset.rgb_encoder.vcodec=h264 \\
        --dataset.rgb_encoder.preset=fast \\
        --dataset.rgb_encoder.extra_options={"tune": "film", "profile:v": "high", "bf": 2}

    # 流式传输到 Foxglove 而不是 Rerun：
    # 添加 --display_mode=foxglove，然后将 Foxglove 应用连接到 ws://127.0.0.1:8765。
"""

import logging

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    lekiwi,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.rollout import (
    InteractiveSession,
    LinkedEvent,
    RolloutConfig,
    build_rollout_context,
    create_strategy,
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
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

logger = logging.getLogger(__name__)


@parser.wrap()
def rollout(cfg: RolloutConfig):
    """策略部署的主入口点。"""
    init_logging()

    if cfg.display_data:
        logger.info(
            "Initializing %s visualization (ip=%s, port=%s)",
            cfg.display_mode,
            cfg.display_ip,
            cfg.display_port,
        )
        init_visualization(cfg.display_mode, session_name="rollout", ip=cfg.display_ip, port=cfg.display_port)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event
    if cfg.interactive:
        # /reset 和 /stop 通过本地标志结束控制循环；进程信号仍然
        # 通过父事件传播。
        shutdown_event = LinkedEvent(shutdown_event)

    logger.info("Building rollout context...")
    ctx = build_rollout_context(cfg, shutdown_event)

    strategy = create_strategy(cfg.strategy)
    logger.info("Rollout strategy: %s", cfg.strategy.type)
    logger.info(
        "Robot: %s | FPS: %.0f | Duration: %s",
        cfg.robot.type if cfg.robot else "?",
        cfg.fps,
        f"{cfg.duration}s" if cfg.duration > 0 else "infinite",
    )

    try:
        strategy.setup(ctx)
        if cfg.interactive:
            logger.info("Rollout setup complete — starting interactive session (robot idle until /start)")
            InteractiveSession(strategy, ctx).run()
        else:
            logger.info("Rollout setup complete, starting rollout...")
            strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        strategy.teardown(ctx)
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

    logger.info("Rollout finished")


def main():
    """``lerobot-rollout`` 的 CLI 入口。"""
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
