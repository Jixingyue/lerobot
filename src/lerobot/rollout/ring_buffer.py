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

"""用于 Highlight Reel rollout 策略的内存受限环形缓冲区。"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch


class RolloutRingBuffer:
    """用于观测/动作帧的固定容量环形缓冲区。

    在内存中保存最近 *N* 秒的遥测数据，同时受时间
    （``max_frames``）和内存（``max_memory_bytes``）限制。
    当任一限制达到时，最旧的帧会被逐出。

    .. note::
       此类是**单线程**的。``append``/``drain``/``clear``
       必须全部从同一线程（rollout 主循环）调用。
       后台线程的并发访问会破坏 ``_current_bytes`` 的
       记账统计。

    Parameters
    ----------
    max_seconds:
        缓冲遥测数据的最大时长。
    max_memory_mb:
        硬性内存上限（MiB）。当估计的总大小超过
        此值时帧会被逐出。
    fps:
        每秒帧数——用于将 ``max_seconds`` 换算为
        帧数。
    """

    def __init__(self, max_seconds: float = 30.0, max_memory_mb: int = 2048, fps: float = 30.0) -> None:
        self._max_frames = int(max_seconds * fps)
        self._max_bytes = int(max_memory_mb * 1024 * 1024)
        self._buffer: deque[dict] = deque(maxlen=self._max_frames)
        self._current_bytes: int = 0

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def append(self, frame: dict) -> None:
        """将 *frame* 加入缓冲区，若已满则逐出最旧的帧。"""
        frame_bytes = _estimate_frame_bytes(frame)

        # 逐出最旧的帧，直到低于内存上限
        while self._current_bytes + frame_bytes > self._max_bytes and self._buffer:
            evicted = self._buffer.popleft()
            self._current_bytes -= _estimate_frame_bytes(evicted)

        self._buffer.append(frame)
        self._current_bytes += frame_bytes

    def drain(self) -> list[dict]:
        """返回所有缓冲的帧并清空缓冲区。"""
        frames = list(self._buffer)
        self._buffer.clear()
        self._current_bytes = 0
        return frames

    def clear(self) -> None:
        """丢弃所有缓冲的帧。"""
        self._buffer.clear()
        self._current_bytes = 0

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def estimated_bytes(self) -> int:
        """所有缓冲帧的估计总字节数。"""
        return self._current_bytes


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------


def _estimate_frame_bytes(frame: dict) -> int:
    """对单个帧字典的字节数粗略估算。"""
    total = 0
    for v in frame.values():
        if isinstance(v, torch.Tensor):
            # ``torch.Tensor`` 没有 ``nbytes``；显式计算它，
            # 以便即使帧中保存未转换的张量也能遵守内存上限。
            total += v.nelement() * v.element_size()
        elif isinstance(v, np.ndarray) or hasattr(v, "nbytes"):
            total += v.nbytes
        elif isinstance(v, (int, float)):
            total += 8
        elif isinstance(v, (str, bytes)):
            total += len(v)
    return max(total, 1)  # 避免零大小的帧
