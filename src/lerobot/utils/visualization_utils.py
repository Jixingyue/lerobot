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

"""与后端无关的可视化分发。

在运行时通过显示模式字符串（例如 ``--display_mode`` CLI 标志）选择可视化后端，
调用方无需针对后端做分支判断。具体实现位于
:mod:`lerobot.utils.rerun_visualization` 和 :mod:`lerobot.utils.foxglove_visualization`；
导入本模块不会导入 ``rerun`` 或 ``foxglove``（每个后端都在
``require_package`` 守卫之后惰性导入其 SDK）。
"""

from lerobot.lerobot_types import RobotAction, RobotObservation

from .foxglove_visualization import init_foxglove, log_foxglove_data, shutdown_foxglove
from .rerun_visualization import init_rerun, log_rerun_data, shutdown_rerun

# 可在运行时通过显示模式字符串（例如 --display_mode 标志）选择的可视化后端。
VISUALIZATION_MODES = ("rerun", "foxglove")


def init_visualization(
    display_mode: str,
    *,
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
) -> None:
    """初始化由 ``display_mode`` 选定的可视化后端。

    对于 ``"rerun"``，``ip``/``port`` 指向可选的远程 Rerun 服务器。对于 ``"foxglove"``，
    ``ip`` 是 WebSocket 服务器绑定的网络接口（仅本地用 ``127.0.0.1``，
    所有接口用 ``0.0.0.0``），``port`` 是其端口。
    """

    if display_mode == "rerun":
        init_rerun(session_name=session_name, ip=ip, port=port)
    elif display_mode == "foxglove":
        init_foxglove(host=ip or "127.0.0.1", port=port)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def log_visualization_data(
    display_mode: str,
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """将观测/动作数据记录到由 ``display_mode`` 选定的后端。"""

    if display_mode == "rerun":
        log_rerun_data(observation=observation, action=action, compress_images=compress_images)
    elif display_mode == "foxglove":
        log_foxglove_data(observation=observation, action=action, compress_images=compress_images)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def shutdown_visualization(display_mode: str) -> None:
    """关闭由 ``display_mode`` 选定的后端。"""

    if display_mode == "rerun":
        shutdown_rerun()
    elif display_mode == "foxglove":
        shutdown_foxglove()
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")
