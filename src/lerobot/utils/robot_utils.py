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

import platform
import time


def precise_sleep(seconds: float, spin_threshold: float = 0.010, sleep_margin: float = 0.005):
    """
    以更高的 CPU 消耗为代价，等待 `seconds` 秒，精度高于单独使用 time.sleep。

    参数：
      - seconds：等待时长
      - spin_threshold：若剩余时间 <= spin_threshold -> 自旋；否则睡眠（秒）。默认 10ms
      - sleep_margin：睡眠时在截止时间前预留这段时间，避免睡过头。默认 5ms

    说明：
        默认参数的选择是为了在常见的 30 FPS 使用场景下，优先保证计时精度而非 CPU 占用。
    """
    if seconds <= 0:
        return
    if spin_threshold < 0:
        raise ValueError(f"spin_threshold must be >= 0, got {spin_threshold}")
    if sleep_margin < 0:
        raise ValueError(f"sleep_margin must be >= 0, got {sleep_margin}")

    system = platform.system()
    # 在 macOS 和 Windows 上，调度器 / 睡眠粒度可能导致
    # 短时间睡眠不准确。与其在整个时长内烧 CPU，
    # 不如大部分时间睡眠，只在最后几毫秒自旋，
    # 从而以低得多的 CPU 占用获得良好的精度。
    if system in ("Darwin", "Windows"):
        end_time = time.perf_counter() + seconds
        while True:
            remaining = end_time - time.perf_counter()
            if remaining <= 0:
                break
            # 如果剩余时间超过几毫秒，就睡掉大部分剩余时间，
            # 并为最后的自旋留出一小段余量。
            if remaining > spin_threshold:
                # 睡眠，但通过预留一小段余量避免睡过截止时间。
                time.sleep(max(remaining - sleep_margin, 0))
            else:
                # 最后短暂自旋，以在不长时间睡眠的情况下命中精确计时。
                pass
    else:
        # 在 Linux 上，time.sleep 对大多数用途已足够精确
        time.sleep(seconds)
