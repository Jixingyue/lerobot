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
在机器人上回放数据集中某个 episode 的动作。

需要安装：pip install 'lerobot[core_scripts]'  （包含 dataset + hardware + viz 附加依赖）

示例：

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=<USER>/record-test \
    --dataset.episode=0
```

双臂 so100 的回放示例：
```shell
lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.processor import (
    make_default_robot_action_processor,
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
    lekiwi,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # 数据集标识。按约定应为 '{hf_username}/{dataset_name}' 格式（例如 `lerobot/test`）。
    repo_id: str
    # 要回放的 episode。
    episode: int
    # 数据集存储的根目录（例如 'dataset/path'）。若为 None，默认为 $HF_LEROBOT_HOME/repo_id。
    root: str | Path | None = None
    # 限制每秒帧数。默认使用策略的 fps。
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # 使用语音合成朗读事件。
    play_sounds: bool = True


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    actions = dataset.select_columns(ACTION)

    robot.connect()

    # 回放必须达到数据集自身的帧率，否则轨迹会以错误的速度回放。
    # 它不写入任何内容，因此错过截止时间只是一个控制稳定性问题。
    timer = CycleTimer(dataset.fps, records_data=False)

    try:
        log_say("Replaying episode", cfg.play_sounds, blocking=True)
        for idx in range(dataset.num_frames):
            timer.tick()

            with timer.section("read_frame"):
                action_array = actions[idx][ACTION]
                action = {}
                for i, name in enumerate(dataset.features[ACTION]["names"]):
                    action[name] = action_array[i]

            with timer.section("observe"):
                robot_obs = robot.get_observation()

            with timer.section("send"):
                processed_action = robot_action_processor((action, robot_obs))
                _ = robot.send_action(processed_action)

            timer.wait()
    finally:
        timer.log_run_summary()
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
