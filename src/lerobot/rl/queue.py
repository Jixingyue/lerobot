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

import platform
from contextlib import suppress
from queue import Empty
from typing import Any

from torch.multiprocessing import Queue


def get_last_item_from_queue(queue: Queue, block=True, timeout: float = 0.1) -> Any:
    if block:
        try:
            item = queue.get(timeout=timeout)
        except Empty:
            return None
    else:
        item = None

    # 清空队列，只保留最新的参数
    if platform.system() == "Darwin":
        # 在 Mac 上，由于 `qsize` 的实现不可靠，应避免使用它。
        # Python 源码中 `qsize` 的代码上有这样一条注释：
        # 由于 sem_getvalue() 存在缺陷，在 Mac OSX 上会抛出 NotImplementedError
        try:
            while True:
                item = queue.get_nowait()
        except Empty:
            pass

        return item

    # 关于使用 qsize 的细节，见 https://github.com/huggingface/lerobot/issues/1523
    while queue.qsize() > 0:
        with suppress(Empty):
            item = queue.get_nowait()

    return item
