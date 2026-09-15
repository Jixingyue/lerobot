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

"""Foxglove 可视化后端。

通过 Foxglove WebSocket 服务器提供实时控制循环流式传输
（:func:`log_foxglove_data`）和可寻址的数据集回放
（:func:`serve_foxglove_dataset_playback`）。调用方通常在运行时通过
:mod:`lerobot.utils.visualization_utils` 中的分派来选择
后端，而不是直接从这里导入。需要 ``viz`` 附加依赖
（``pip install 'lerobot[viz]'``）。
"""

import logging
import numbers
import time

import cv2
import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation

from .constants import (
    ACTION,
    ACTION_PREFIX,
    DONE,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
    OBS_STR,
    REWARD,
    SUCCESS,
    TRUNCATED,
)
from .import_utils import require_package

# 所有标量话题共享的静态模式。每条消息携带一个由 ``{label, value}``
# 键值对组成的扁平列表，而不是每个特征一个字段，因此无论机器人报告哪些
# 观测/动作特征，同一模式都适用。``label`` 字段名正是 Foxglove 用来
# 自动命名各条序列的依据，因此单个过滤路径即可绘制每个特征，例如
# ``/observation/state.scalars[:]``。
_SCALARS_SCHEMA = {
    "type": "object",
    "title": "lerobot.Scalars",
    "properties": {
        "scalars": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "number"},
                },
            },
        }
    },
}


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def init_foxglove(host: str = "127.0.0.1", port: int | None = 8765) -> None:
    """
    启动一个 Foxglove WebSocket 服务器以可视化控制循环。

    在 Foxglove 应用中通过 ``ws://<host>:<port>`` 连接。当已有
    服务器在运行时，多次调用此函数为空操作。

    参数:
        host: WebSocket 服务器绑定的主机接口。
        port: WebSocket 服务器绑定的端口（默认 8765）。
    """

    require_package("foxglove-sdk", extra="viz", import_name="foxglove")
    import foxglove

    # 实时流式传输状态作为 ``log_foxglove_data`` 的属性存在：
    # ``.server`` 是共享的 WebSocket 服务器，
    # ``.channels`` 为每个话题缓存一个 Foxglove 频道
    if getattr(log_foxglove_data, "server", None) is not None:
        return
    log_foxglove_data.server = foxglove.start_server(host=host, port=port or 8765)
    log_foxglove_data.channels = {}


def shutdown_foxglove() -> None:
    """停止 Foxglove WebSocket 服务器并清空已缓存的频道。"""

    server = getattr(log_foxglove_data, "server", None)
    if server is not None:
        server.stop()
    log_foxglove_data.server = None
    log_foxglove_data.channels = {}


def _foxglove_safe_name(name: str) -> str:
    """将 ``.`` 替换为 ``_``，使特征名称成为单个 Foxglove 话题路径段。

    Foxglove 把 ``.`` 视为路径分隔符，因此像 ``observation.images.front``
    这样未经处理的名称会被拆分成嵌套的路径段，而不是命名一个话题。
    """

    return name.replace(".", "_")


def _foxglove_topic(key: str, *, is_image: bool = False) -> str:
    """为特征 ``key`` 构造 Foxglove 话题。

    相机特征映射为每个来源独立的图像话题（``/observation/images/<name>``）；标量特征
    则按来源共享一个聚合话题：观测为 ``/observation/state``，动作为
    ``/action/state``。
    """

    if is_image:
        name = str(key)
        for prefix in (f"{OBS_IMAGES}.", OBS_PREFIX):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        return f"/{OBS_STR}/images/{_foxglove_safe_name(name)}"
    source = ACTION if (str(key).startswith(ACTION_PREFIX) or str(key) == ACTION) else OBS_STR
    return f"/{source}/state"


def _log_foxglove_scalars(
    topic: str, values: dict[str, float], *, channels: dict | None = None, log_time: int | None = None
) -> None:
    """使用静态 :data:`_SCALARS_SCHEMA` 在带类型的 JSON 频道上记录标量。

    ``values`` 是一个从特征名称到值的有序映射；它会以 ``scalars`` 数组的形式
    发出，数组元素为 ``{label, value}`` 对象。插入顺序会被保留，因此各条序列
    在不同消息之间保持稳定。

    ``channels`` 是要复用的逐话题频道缓存（默认为
    :func:`log_foxglove_data` 上的实时流缓存；数据集回放会传入自己的本地缓存以保持自包含）。
    ``log_time`` 是消息时间（纳秒）；为 ``None`` 时使用服务器的接收时间。
    """

    if not values:
        return

    import foxglove

    if channels is None:
        channels = log_foxglove_data.channels
    channel = channels.get(topic)
    if channel is None:
        channel = channels[topic] = foxglove.Channel(topic, schema=_SCALARS_SCHEMA, message_encoding="json")
    msg = {"scalars": [{"label": label, "value": value} for label, value in values.items()]}
    if log_time is None:
        channel.log(msg)
    else:
        channel.log(msg, log_time=log_time)


def _labeled_scalars(name: str, values, labels: list[str] | None = None) -> dict[str, float]:
    """将一维序列展开为 ``{label: value}`` 条目，并使用一致的回退命名。"""

    flat = [float(v) for v in values]
    if labels is None or len(labels) != len(flat):
        labels = [f"{name}_{i}" for i in range(len(flat))]
    return dict(zip(labels, flat, strict=True))


def _log_foxglove_image(
    topic: str,
    frame_id: str,
    arr: np.ndarray,
    *,
    compress_images: bool,
    channels: dict | None = None,
    log_time: int | None = None,
    depth_range: tuple[float, float] | None = None,
    raw_depth_values: bool = False,
) -> None:
    """在缓存的逐话题频道上记录一幅图像。

    编码方式根据通道数和 dtype 选择：单通道 ``float`` 或 ``uint16``
    帧为深度图（``32FC1``/``16UC1``），单通道 ``uint8`` 为 ``mono8``，3 通道为 ``rgb8``
    （float 输入假定在 [0, 1] 范围内，转换为 uint8），4 通道为 ``rgba8``；其他通道数
    会被跳过并发出警告。设置 ``compress_images`` 时，``rgb8`` 改为使用 JPEG 编码。

    参数:
        topic: 要记录到的 Foxglove 话题。
        frame_id: 标记在消息上的帧 id。
        arr: HWC 或 CHW 格式的图像（CHW 会被转置为 HWC），任意 dtype。
        compress_images: 对 ``rgb8`` 帧进行 JPEG 编码；其他编码忽略此参数。
        channels: 要复用的逐话题频道缓存（参见 :func:`_log_foxglove_scalars`）。
        log_time: 消息时间（纳秒），同时写入头部时间戳；为 ``None``
            时使用服务器的接收时间。
        depth_range: 深度帧自身输入单位下的 ``(lo, hi)`` 裁剪边界。深度帧
            （``32FC1``/``16UC1``）会被重新缩放到 Foxglove 对相应编码的默认显示
            最大值（``1.0`` / ``10000``），以便以合理的对比度显示；``depth_range``
            设置源范围，否则使用帧自身的最小/最大值。``mono8``/``rgb8``/``rgba8`` 忽略此参数。
        raw_depth_values: 为 True 时，深度值不做重新缩放，按原样记录。
    """

    from foxglove.channels import CompressedImageChannel, RawImageChannel
    from foxglove.messages import CompressedImage, RawImage, Timestamp

    if channels is None:
        channels = log_foxglove_data.channels
    time_ns = time.time_ns() if log_time is None else log_time
    timestamp = Timestamp(sec=time_ns // 1_000_000_000, nsec=time_ns % 1_000_000_000)
    log_kwargs = {} if log_time is None else {"log_time": log_time}

    # 必要时将 CHW -> HWC（与 log_rerun_data 一致）。
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    height, width = arr.shape[0], arr.shape[1]
    n_channels = 1 if arr.ndim == 2 else arr.shape[2]

    if n_channels == 1 and arr.dtype != np.uint8:
        # 深度图：根据 dtype 推断编码。
        encoding, target_dtype, value_max = (
            ("32FC1", np.float32, 1.0)
            if np.issubdtype(arr.dtype, np.floating)
            else ("16UC1", np.uint16, 10000.0)
        )
        if not raw_depth_values:
            # 按照给定的 depth_range 重新缩放到该编码的显示最大值。
            lo, hi = depth_range if depth_range is not None else (float(arr.min()), float(arr.max()))
            arr = arr.clip(lo, hi).astype(np.float32)
            arr = (arr - lo) / ((hi - lo) if hi > lo else 1.0) * value_max
        arr = np.ascontiguousarray(arr, dtype=target_dtype)
    else:
        if n_channels == 3 and np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0).clip(0, 255)
        arr = np.ascontiguousarray(arr, dtype=np.uint8)

        if compress_images and n_channels == 3:
            buf_src = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", buf_src)
            channel = channels.get(topic)
            if channel is None:
                channel = channels[topic] = CompressedImageChannel(topic=topic)
            channel.log(
                CompressedImage(timestamp=timestamp, frame_id=frame_id, data=buf.tobytes(), format="jpeg"),
                **log_kwargs,
            )
            return

        encoding = {1: "mono8", 3: "rgb8", 4: "rgba8"}.get(n_channels)
        if encoding is None:
            logging.warning(
                "Foxglove: skipping image on topic '%s' with unsupported shape %s (%d channels); "
                "expected 1 (mono8/16UC1/32FC1), 3 (rgb8), or 4 (rgba8) channels.",
                topic,
                tuple(arr.shape),
                n_channels,
            )
            return

    channel = channels.get(topic)
    if channel is None:
        channel = channels[topic] = RawImageChannel(topic=topic)
    channel.log(
        RawImage(
            timestamp=timestamp,
            frame_id=frame_id,
            width=width,
            height=height,
            encoding=encoding,
            step=width * n_channels * arr.itemsize,
            data=arr.tobytes(),
        ),
        **log_kwargs,
    )


def log_foxglove_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    将观测和动作数据记录到 Foxglove WebSocket 服务器以进行实时可视化。

    与 ``log_rerun_data`` 对应，但通过 :func:`init_foxglove`
    启动的服务器发出 Foxglove 消息。数据映射如下：
    - 标量（以及一维数组的元素）按来源累积，并使用静态的
      ``lerobot.Scalars`` 模式以带类型的 JSON 消息记录到
      ``/observation/state`` 和 ``/action/state`` 话题：即一个由
      ``{label, value}`` 对象组成的 ``scalars`` 数组（参见
      :data:`_SCALARS_SCHEMA`）。``label`` 字段让 Foxglove 能够自动命名各条序列，因此
      ``/observation/state.scalars[:].value`` 可以一次绘制所有特征。
    - 形似图像的三维 NumPy 数组会在必要时从 CHW 转置为 HWC，并记录到
      每个来源各自的话题（例如 ``/observation/images/front``）上，形式为
      ``RawImage``（当 ``compress_images`` 为 True 时为 JPEG
      ``CompressedImage``）。

    参数:
        observation: 可选的字典，包含要记录的观测数据。
        action: 可选的字典，包含要记录的动作数据。
        compress_images: 是否在记录前对图像进行 JPEG 压缩，以带宽
            换取 CPU 和画质。
    """

    require_package("foxglove-sdk", extra="viz", import_name="foxglove")

    if getattr(log_foxglove_data, "server", None) is None:
        raise RuntimeError("init_foxglove() must be called before log_foxglove_data().")

    now = time.time_ns()

    if observation:
        obs_scalars: dict[str, float] = {}
        for k, v in observation.items():
            if v is None:
                continue
            key = k[len(OBS_PREFIX) :] if str(k).startswith(OBS_PREFIX) else str(k)
            if _is_scalar(v):
                obs_scalars[key] = float(v)
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    obs_scalars.update(_labeled_scalars(key, v))
                else:
                    _log_foxglove_image(
                        _foxglove_topic(k, is_image=True),
                        key,
                        v,
                        compress_images=compress_images,
                        log_time=now,
                    )
        _log_foxglove_scalars(_foxglove_topic(OBS_STATE), obs_scalars, log_time=now)

    if action:
        action_scalars: dict[str, float] = {}
        for k, v in action.items():
            if v is None:
                continue
            key = k[len(ACTION_PREFIX) :] if str(k).startswith(ACTION_PREFIX) else str(k)
            if _is_scalar(v):
                action_scalars[key] = float(v)
            elif isinstance(v, np.ndarray):
                action_scalars.update(_labeled_scalars(key, v.flatten()))
        _log_foxglove_scalars(_foxglove_topic(ACTION), action_scalars, log_time=now)


# ── 通过 Foxglove WebSocket 服务器进行数据集回放 ─────────────────────
# LeRobotDataset 在磁盘上支持随机访问，因此我们不采用发后即忘的前向流，
# 而是发布一条可寻址的时间线，并根据用户在 Foxglove 应用中
# 拖动/播放到的时间按需提供帧。这依赖 SDK 的 PlaybackControl 能力。


def _feature_dim_names(feature: dict | None) -> list[str] | None:
    """尽力为一维特征返回各维度的序列标签，无法确定时返回 ``None`` 以回退到索引。

    LeRobot 记录特征 ``names`` 的方式并不一致：可能是扁平列表
    （``["x", "y"]``）、类别映射
    （``{"motors": ["motor_0", "motor_1"]}``），或名称到索引的映射
    （``{"delta_x": 0, "delta_y": 1}``）。每种形式都会被处理，但只有当标签数量
    与特征的一维形状匹配时才返回标签，这样格式错误/不匹配的 ``names`` 不会
    悄无声息地给序列贴错标签。
    """

    if not feature:
        return None
    shape = feature.get("shape")
    dim = shape[0] if shape and len(shape) == 1 else None
    names = feature.get("names")
    labels: list[str] | None = None
    if isinstance(names, dict):
        values = list(names.values())
        if values and all(isinstance(v, (list, tuple)) for v in values):
            labels = [str(n) for group in values for n in group]
        elif values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            labels = [name for name, _ in sorted(names.items(), key=lambda kv: kv[1])]
    elif isinstance(names, (list, tuple)):
        labels = [str(n) for n in names]
    if labels is not None and dim is not None and len(labels) == dim:
        return labels
    return None


def _frame_to_scalars(sample: dict, key: str, labels: list[str] | None = None) -> dict[str, float]:
    """将一帧中的向量/标量特征 ``key`` 展平为 ``{label: value}`` 条目。

    ``labels`` 为每个维度提供一个名称（来自数据集的特征元数据）；当缺失或
    长度不对时，各维度回退到 ``{name}_{i}``（特征的短名称），与
    实时流保持一致，使序列名称相互吻合。标量特征变为单个条目。缺失或为
    ``None`` 的特征产生空映射。
    """

    v = sample.get(key)
    if v is None:
        return {}
    arr = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    if key.startswith(OBS_PREFIX):
        name = key[len(OBS_PREFIX) :]
    elif key.startswith(ACTION_PREFIX):
        name = key[len(ACTION_PREFIX) :]
    else:
        name = key
    if arr.ndim == 0:
        return {name: float(arr)}
    return _labeled_scalars(name, arr.flatten(), labels)


def _playback_times_ns(dataset) -> list[int]:
    """逐帧时间戳（纳秒），无需解码视频即可读取。"""
    if hasattr(dataset, "hf_dataset"):
        return [int(round(float(t) * 1e9)) for t in dataset.hf_dataset["timestamp"]]
    # 没有 hf_dataset 的存储格式（例如 lance）：时间戳落在 fps 网格上。
    return [int(round(i * 1e9 / dataset.fps)) for i in range(len(dataset))]


def serve_foxglove_dataset_playback(
    dataset,
    episode_index: int,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    compress_images: bool = False,
    autoplay: bool = True,
) -> None:
    """将单个数据集 episode 作为可寻址、可拖动浏览的时间线提供给 Foxglove。

    启动一个 Foxglove WebSocket 服务器，在该
    episode 的时间范围内发布 ``PlaybackControl`` 能力。Foxglove 应用驱动播放/暂停/寻址/倍速；
    一个后台线程和一个
    ``ServerListener`` 按需从磁盘上的 ``dataset`` 读取帧，并以其数据集时间戳
    记录它们，因此用户可以在 episode 中的任意位置拖动浏览。阻塞运行，直到被中断。

    参数:
        dataset: 为待可视化的单个 episode 加载的 ``LeRobotDataset``。
        episode_index: 正在可视化的 episode 索引（仅用于会话名称）。
        host: WebSocket 服务器绑定的主机接口。
        port: WebSocket 服务器绑定的端口。
        compress_images: 是否在记录前对相机帧进行 JPEG 压缩。
        autoplay: 为 True 时，一旦客户端连接就自动开始播放，而不是
            等待用户在 Foxglove 应用中按下播放。
    """

    require_package("foxglove-sdk", extra="viz", import_name="foxglove")
    import bisect
    import threading

    import foxglove
    from foxglove.websocket import (
        Capability,
        PlaybackCommand,
        PlaybackControlRequest,
        PlaybackState,
        PlaybackStatus,
        ServerListener,
    )

    times_ns = _playback_times_ns(dataset)
    n_frames = len(times_ns)
    if n_frames == 0:
        raise ValueError("Cannot visualize an empty episode.")
    first_ns, last_ns = times_ns[0], times_ns[-1]
    camera_keys = list(dataset.meta.camera_keys)
    # 全数据集范围的 q01/q99 深度边界（回退到 min/max），用于将深度归一化到 [0, 1]。
    depth_ranges: dict[str, tuple[float, float]] = {}
    for key in dataset.meta.depth_keys:
        stats = (dataset.meta.stats or {}).get(key)
        if not stats:
            continue
        lo = stats["q01"] if "q01" in stats else stats["min"]
        hi = stats["q99"] if "q99" in stats else stats["max"]
        depth_ranges[key] = (float(np.asarray(lo).item()), float(np.asarray(hi).item()))
    # 来自数据集元数据的各维度序列标签（例如关节名称），只计算一次。
    scalar_labels = {
        OBS_STATE: _feature_dim_names(dataset.meta.features.get(OBS_STATE)),
        ACTION: _feature_dim_names(dataset.meta.features.get(ACTION)),
    }
    # 本地频道缓存，使回放服务器自包含，不触碰实时流缓存。
    channels: dict = {}

    def emit_frame(i: int) -> None:
        """记录帧 ``i`` 的每个频道，并以其数据集时间戳标记。"""
        sample = dataset[i]
        log_time = times_ns[i]
        for key in camera_keys:
            arr = sample.get(key)
            if arr is None:
                continue
            arr = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
            _log_foxglove_image(
                _foxglove_topic(key, is_image=True),
                key,
                arr,
                compress_images=compress_images,
                channels=channels,
                log_time=log_time,
                depth_range=depth_ranges.get(key),
                raw_depth_values=True,
            )
        _log_foxglove_scalars(
            _foxglove_topic(OBS_STATE),
            _frame_to_scalars(sample, OBS_STATE, scalar_labels[OBS_STATE]),
            channels=channels,
            log_time=log_time,
        )
        _log_foxglove_scalars(
            _foxglove_topic(ACTION),
            _frame_to_scalars(sample, ACTION, scalar_labels[ACTION]),
            channels=channels,
            log_time=log_time,
        )
        episode_scalars = {}
        for feat, label in (
            (DONE, "done"),
            (TRUNCATED, "truncated"),
            (REWARD, "reward"),
            (SUCCESS, "success"),
        ):
            v = sample.get(feat)
            if v is not None:
                episode_scalars[label] = float(v)
        _log_foxglove_scalars("/episode/state", episode_scalars, channels=channels, log_time=log_time)

    lock = threading.Lock()
    stop_event = threading.Event()
    # 共享的回放状态，由 ``lock`` 保护。``seek_idx`` 是由监听器设置、
    # 由回放循环处理的一次性请求；回放循环是*唯一*发出帧的线程（因此
    # 对磁盘数据集 / 视频解码器的并发随机访问永远不会重叠）。
    state = {
        "status": PlaybackStatus.Paused,
        "cursor": first_ns,
        "speed": 1.0,
        "last_idx": -1,
        "seek_idx": None,
    }

    def index_at(t_ns: int) -> int:
        return max(0, min(n_frames - 1, bisect.bisect_right(times_ns, t_ns) - 1))

    # 一次性门闩，使自动播放只在第一个客户端订阅时触发。
    autoplay_started = threading.Event()

    class _PlaybackListener(ServerListener):
        def on_subscribe(self, client, channel):
            # 当客户端真正连接（订阅）后自动开始播放。使用
            # 订阅钩子而不是一开始就以 Playing 状态启动，意味着时间线不会
            # 在有人观看之前前进。只触发一次；之后用户仍可暂停/寻址。
            if not autoplay:
                return
            with lock:
                if autoplay_started.is_set() or state["status"] != PlaybackStatus.Paused:
                    return
                autoplay_started.set()
                state["status"] = PlaybackStatus.Playing
                cursor, speed = state["cursor"], state["speed"]
            server.broadcast_playback_state(PlaybackState(PlaybackStatus.Playing, cursor, speed, False, ""))

        def on_playback_control_request(self, req: PlaybackControlRequest):
            # 这里只修改状态；所有帧的发出都由回放循环执行。
            with lock:
                did_seek = False
                if req.seek_time is not None:
                    cursor = max(first_ns, min(last_ns, req.seek_time))
                    state["cursor"] = cursor
                    state["last_idx"] = state["seek_idx"] = index_at(cursor)
                    did_seek = True
                if req.playback_speed and req.playback_speed > 0:
                    state["speed"] = req.playback_speed
                if req.playback_command == PlaybackCommand.Play:
                    # 从末尾处再次播放时，从头开始重播。
                    if state["cursor"] >= last_ns:
                        state["cursor"] = first_ns
                        state["last_idx"] = state["seek_idx"] = 0
                        did_seek = True
                    state["status"] = PlaybackStatus.Playing
                elif req.playback_command == PlaybackCommand.Pause:
                    state["status"] = PlaybackStatus.Paused
                status, cursor, speed = state["status"], state["cursor"], state["speed"]
                request_id = req.request_id or ""
            return PlaybackState(status, cursor, speed, did_seek, request_id)

    server = foxglove.start_server(
        name=f"{dataset.repo_id}/episode_{episode_index}",
        host=host,
        port=port,
        capabilities=[Capability.PlaybackControl, Capability.Time],
        server_listener=_PlaybackListener(),
        playback_time_range=(first_ns, last_ns),
    )

    def playback_loop() -> None:
        # 限制游标在单个 tick 内最多前进多少。否则缓慢的帧解码（或任何卡顿）
        # 会使 ``dt`` 变得很大，产生一大批巨量的追赶帧；对其钳制后，在解码器
        # 缓慢时回放会落后于挂钟时间，但每个 tick 发出的帧范围是有界的。
        max_tick_dt_s = 0.25
        prev = time.monotonic()
        while not stop_event.is_set():
            time.sleep(1.0 / 60.0)
            ended = False
            speed = 1.0
            with lock:
                now = time.monotonic()
                dt = min(now - prev, max_tick_dt_s)
                prev = now
                # 排队的寻址请求总会被处理，即使处于暂停状态，这样拖动浏览才能更新画面。
                work = []
                seek_idx = state["seek_idx"]
                if seek_idx is not None:
                    state["seek_idx"] = None
                    work.append(seek_idx)
                if state["status"] == PlaybackStatus.Playing:
                    cursor = state["cursor"] + int(dt * 1e9 * state["speed"])
                    start_idx = state["last_idx"] + 1
                    if cursor >= last_ns:
                        cursor, target, ended = last_ns, n_frames - 1, True
                    else:
                        target = index_at(cursor)
                    state["cursor"] = cursor
                    work.extend(range(start_idx, target + 1))
                    # 播放时游标只会增长（寻址会在监听器中重置 last_idx），因此
                    # 这里 target >= last_idx；直接赋值是正确的，也比 max() 更清晰。
                    state["last_idx"] = target
                    if ended:
                        state["status"] = PlaybackStatus.Ended
                if not work:
                    continue
                cursor, speed = state["cursor"], state["speed"]
            # 在锁之外发出帧；这是唯一调用 emit_frame 的线程。在帧与帧之间
            # 重新检查 stop_event，使关闭过程即使在批次中途也能保持响应。
            for i in work:
                if stop_event.is_set():
                    break
                emit_frame(i)
            server.broadcast_time(cursor)
            if ended:
                server.broadcast_playback_state(PlaybackState(PlaybackStatus.Ended, cursor, speed, False, ""))

    # 发出第一帧，以便各频道被发布（在循环启动之前完成，从而使帧的发出
    # 保持单线程）。晚连接的客户端在寻址/播放后会重新收到帧。
    emit_frame(0)
    with lock:
        state["last_idx"] = 0
    server.broadcast_time(first_ns)
    server.broadcast_playback_state(PlaybackState(PlaybackStatus.Paused, first_ns, 1.0, True, ""))

    thread = threading.Thread(target=playback_loop, name="foxglove-playback", daemon=True)
    thread.start()

    print(f"Foxglove server running. Connect the Foxglove app to ws://{host}:{port}")
    print("Use the playback controls in Foxglove to play/pause and scrub the episode. Ctrl-C to exit.")
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Ctrl-C received. Exiting.")
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
        server.stop()
        channels.clear()
