# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import abc

from lerobot.lerobot_types import BatchType

from ..buffer import ReplayBuffer, concatenate_batch_transitions


class DataMixer(abc.ABC):
    """所有数据混合策略的抽象接口。"""

    @abc.abstractmethod
    def sample(self, batch_size: int) -> BatchType:
        """抽取一个包含 ``batch_size`` 个转移的批次。"""
        raise NotImplementedError

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """产出批次的无限迭代器。"""
        while True:
            yield self.sample(batch_size)


class OnlineOfflineMixer(DataMixer):
    """混合来自在线和离线回放缓冲区的转移。"""

    def __init__(
        self,
        online_buffer: ReplayBuffer,
        offline_buffer: ReplayBuffer | None = None,
        online_ratio: float = 1.0,
    ):
        if not 0.0 <= online_ratio <= 1.0:
            raise ValueError(f"online_ratio must be in [0, 1], got {online_ratio}")
        self.online_buffer = online_buffer
        self.offline_buffer = offline_buffer
        self.online_ratio = online_ratio

    def sample(self, batch_size: int) -> BatchType:
        if self.offline_buffer is None:
            return self.online_buffer.sample(batch_size)

        n_online = max(1, int(batch_size * self.online_ratio))
        n_offline = batch_size - n_online

        online_batch = self.online_buffer.sample(n_online)
        offline_batch = self.offline_buffer.sample(n_offline)
        return concatenate_batch_transitions(online_batch, offline_batch)

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """通过组合缓冲区的异步迭代器来产出批次。"""

        n_online = max(1, int(batch_size * self.online_ratio))

        online_iter = self.online_buffer.get_iterator(
            batch_size=n_online,
            async_prefetch=async_prefetch,
            queue_size=queue_size,
        )

        if self.offline_buffer is None:
            yield from online_iter
            return

        n_offline = batch_size - n_online
        offline_iter = self.offline_buffer.get_iterator(
            batch_size=n_offline,
            async_prefetch=async_prefetch,
            queue_size=queue_size,
        )

        while True:
            yield concatenate_batch_transitions(next(online_iter), next(offline_iter))
