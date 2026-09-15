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

"""Rerun 可视化后端。

将控制循环实时流式传输到 Rerun 查看器（:func:`log_rerun_data`）。调用方通常通过
:mod:`lerobot.utils.visualization_utils` 中的分发在运行时选择后端，
而不是直接从本模块导入。需要 ``viz`` extra（``pip install 'lerobot[viz]'``）。
"""

import numbers
import os

import numpy as np

from lerobot.configs import DEPTH_MILLIMETER_UNIT, infer_depth_unit
from lerobot.lerobot_types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
from .import_utils import require_package


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def init_rerun(
    session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None
) -> None:
    """
    初始化 Rerun SDK，用于可视化控制循环。

    参数：
        session_name：Rerun 会话名称。
        ip：连接 Rerun 服务器的可选 IP。
        port：连接 Rerun 服务器的可选端口。
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    log_rerun_data.blueprint = None  # 为新会话重置 blueprint 缓存

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)


def shutdown_rerun() -> None:
    """优雅地关闭 Rerun SDK。"""

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    rr.rerun_shutdown()


def _build_blueprint(observation_paths: set[str], action_paths: set[str], image_paths: set[str]):
    """构建 Rerun blueprint，将相机图像、观测和动作标量分别布局到不同的视图中。

    相机图像、观测和动作标量以网格形式排列。
    """

    # 安全且零开销：`log_rerun_data` 已经执行过 `require_package` 守卫并导入了 rerun。
    import rerun.blueprint as rrb

    views = [rrb.Spatial2DView(origin=path, name=path) for path in sorted(image_paths)]

    if observation_paths:
        views.append(rrb.TimeSeriesView(name="observation", contents=sorted(observation_paths)))
    if action_paths:
        views.append(rrb.TimeSeriesView(name="action", contents=sorted(action_paths)))

    return rrb.Blueprint(rrb.Grid(*views))


def _ensure_blueprint(observation_paths: set[str], action_paths: set[str], image_paths: set[str]) -> None:
    """在首次收到观测和动作数据时构建并发送 blueprint（只执行一次）。"""
    if getattr(log_rerun_data, "blueprint", None) is not None:
        return

    if not (observation_paths or action_paths or image_paths):
        return

    # 安全且零开销：`log_rerun_data` 已经执行过 `require_package` 守卫并导入了 rerun。
    import rerun as rr

    blueprint = _build_blueprint(observation_paths, action_paths, image_paths)
    log_rerun_data.blueprint = blueprint
    rr.send_blueprint(blueprint)


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    将观测和动作数据记录到 Rerun，用于实时可视化。

    该函数遍历提供的观测和动作字典，并将其内容发送到 Rerun 查看器。
    它会针对不同的数据类型做相应处理：
    - 标量值（浮点数、整数）记录为 `rr.Scalars`。
    - 形似图像的 3D NumPy 数组（例如通道数为 1、3 或 4 且位于第一维）会从
      CHW 转置为 HWC 格式，（可选）压缩为 JPEG，并记录为 `rr.Image` 或 `rr.EncodedImage`。
    - 1D NumPy 数组在同一个实体路径下作为单个 `rr.Scalars` 批次记录，
      使所有维度共享同一个视图，而不是每个元素拆分到一个视图。
    - 多维**动作**数组会被展平，并作为单个 `rr.Scalars` 批次记录。

    若键名尚未包含 "observation." 或 "action." 前缀，会自动加上相应命名空间。

    首次调用时会构建并发送 blueprint，使观测和动作标量获得各自独立的
    时间序列视图，每张图像获得自己的空间视图。

    参数：
        observation：包含待记录观测数据的可选字典。
        action：包含待记录动作数据的可选字典。
        compress_images：是否在记录前压缩图像，以 CPU 和画质为代价节省带宽和内存。
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    observation_paths: set[str] = set()
    action_paths: set[str] = set()
    image_paths: set[str] = set()

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
                observation_paths.add(key)
            elif isinstance(v, np.ndarray):
                arr = v
                # 需要时将 CHW -> HWC 转换
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    rr.log(key, rr.Scalars(arr.astype(float)))
                    observation_paths.add(key)
                else:
                    if arr.shape[-1] == 1:
                        # 在录制时，深度单位从帧类型推断。
                        depth_unit = infer_depth_unit(arr.dtype)
                        img_entity = rr.DepthImage(
                            arr,
                            meter=1000.0 if depth_unit == DEPTH_MILLIMETER_UNIT else 1.0,
                            colormap=rr.components.Colormap.Viridis,
                        )
                    else:
                        img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=True)
                    image_paths.add(key)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
                action_paths.add(key)
            elif isinstance(v, np.ndarray):
                # 将任意（包括高维）数组展平为单个批处理的 Scalars
                rr.log(key, rr.Scalars(v.reshape(-1).astype(float)))
                action_paths.add(key)

    _ensure_blueprint(observation_paths, action_paths, image_paths)
