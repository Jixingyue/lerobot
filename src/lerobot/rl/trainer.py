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

from collections.abc import Iterator
from typing import Any

from lerobot.lerobot_types import BatchType

from .algorithms.base import RLAlgorithm
from .algorithms.configs import TrainingStats
from .data_sources.data_mixer import DataMixer


class RLTrainer:
    """统一的训练步编排器。

    持有算法、一个 DataMixer 以及一个可选的预处理器。
    """

    def __init__(
        self,
        algorithm: RLAlgorithm,
        data_mixer: DataMixer,
        batch_size: int,
        *,
        preprocessor: Any | None = None,
    ):
        self.algorithm = algorithm
        self.data_mixer = data_mixer
        self.batch_size = batch_size
        self._preprocessor = preprocessor

        self._iterator: Iterator[BatchType] | None = None

        self.algorithm.make_optimizers_and_scheduler()

    def _build_data_iterator(self) -> Iterator[BatchType]:
        """创建一个全新的、由算法配置的迭代器（可选地经过预处理）。"""
        raw = self.algorithm.configure_data_iterator(
            data_mixer=self.data_mixer,
            batch_size=self.batch_size,
        )
        if self._preprocessor is not None:
            return _PreprocessedIterator(raw, self._preprocessor)
        return raw

    def reset_data_iterator(self) -> None:
        """丢弃当前迭代器，使其在下一个训练步时被惰性重建。"""
        self._iterator = None

    def set_data_mixer(self, data_mixer: DataMixer, *, reset: bool = True) -> None:
        """替换当前使用的数据混合器，并可选择重置迭代器。"""
        self.data_mixer = data_mixer
        if reset:
            self.reset_data_iterator()

    def training_step(self) -> TrainingStats:
        """执行一个训练步（与具体算法无关）。"""
        if self._iterator is None:
            self._iterator = self._build_data_iterator()
        return self.algorithm.update(self._iterator)


def preprocess_rl_batch(preprocessor: Any, batch: BatchType) -> BatchType:
    """仅对 RL 观测应用策略预处理。"""
    observations = batch["state"]
    next_observations = batch["next_state"]
    batch["state"] = preprocessor.process_observation(observations)
    batch["next_state"] = preprocessor.process_observation(next_observations)

    return batch


class _PreprocessedIterator:
    """对每个采样出的 RL 批次进行预处理的迭代器包装器。"""

    __slots__ = ("_raw", "_preprocessor")

    def __init__(self, raw_iterator: Iterator[BatchType], preprocessor: Any) -> None:
        self._raw = raw_iterator
        self._preprocessor = preprocessor

    def __iter__(self) -> _PreprocessedIterator:
        return self

    def __next__(self) -> BatchType:
        batch = next(self._raw)
        return preprocess_rl_batch(self._preprocessor, batch)
