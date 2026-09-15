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

"""实时分块（Real-Time Chunking，RTC）的延迟跟踪工具。"""

from collections import deque

import numpy as np


class LatencyTracker:
    """跟踪最近的延迟，并提供最大值/百分位数查询。

    Args:
        maxlen (int | None): 可选的滑动窗口大小。若提供，则只保留最近的
            ``maxlen`` 个延迟样本；若为 ``None``，则保留全部。
    """

    def __init__(self, maxlen: int = 100):
        self._values = deque(maxlen=maxlen)
        self.reset()

    def reset(self) -> None:
        """清除所有已记录的延迟。"""
        self._values.clear()
        self.max_latency = 0.0

    def add(self, latency: float) -> None:
        """添加一个延迟样本（单位：秒）。"""
        # 确保为数值且非负
        val = float(latency)

        if val < 0:
            return
        self._values.append(val)
        self.max_latency = max(self.max_latency, val)

    def __len__(self) -> int:
        return len(self._values)

    def max(self) -> float | None:
        """返回最大延迟；若为空则返回 None。"""
        return self.max_latency

    def percentile(self, q: float) -> float | None:
        """返回已记录延迟的 q 分位数（q 取值 [0,1]）；若为空则返回 None。"""
        if not self._values:
            return 0.0
        q = float(q)
        if q <= 0.0:
            return min(self._values)
        if q >= 1.0:
            return self.max_latency
        vals = np.array(list(self._values), dtype=np.float32)
        return float(np.quantile(vals, q))

    def p95(self) -> float | None:
        """返回第 95 百分位延迟；若为空则返回 None。"""
        return self.percentile(0.95)
