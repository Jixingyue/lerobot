#!/usr/bin/env python

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
import logging
import math
from collections.abc import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    """针对 episode 帧的采样器，仅存储每个 episode 的边界。

    逻辑位置会即时映射为帧索引（构造时内存开销为 O(num_episodes)），
    而不是将每个帧索引物化为一个 Python 列表。

    每个 epoch 使用以 `(seed, epoch)` 为种子的 `torch.randperm` 进行打乱，因此数据顺序
    是 `(seed, epoch)` 的纯函数：它可以在每个 rank 上复现而无需同步全局 RNG
    （无需在分布式 rank 之间同步 `generator`），并且 `state_dict` /
    `load_state_dict` 通过重新生成该 epoch 的排列并从保存的偏移量继续，
    从而以样本精确的方式恢复运行。每次调用 `__iter__` 都会推进 epoch。
    在恢复的 epoch 中，`__len__` 仍报告完整长度。

    Epoch 推进：`__iter__` 会立即推进 epoch，而 `set_epoch` / `load_state_dict`
    则显式设置它。在同一次运行中，调用方应只依赖这两种机制之一，
    而不是同时使用两者：手动推进 epoch *并且* 让 `__iter__` 在相同的
    迭代上自动推进，会导致跳过或重复 epoch。训练循环纯粹通过 `__iter__`
    （经由 `cycle`）来驱动它；`set_epoch` / `load_state_dict` 仅用于在迭代
    开始之前（重新）定位（例如恢复时或测试中）。
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ):
        """
        Args:
            dataset_from_indices: 数据集中每个 episode 的起始索引。
            dataset_to_indices: 数据集中每个 episode 的结束索引。
            episode_indices_to_use: 要使用的 episode 索引；None 表示全部。
            drop_n_first_frames: 从每个 episode 开头丢弃的帧数。
            drop_n_last_frames: 从每个 episode 结尾丢弃的帧数。
            shuffle: 是否打乱索引。
            seed: 用于推导排列的种子（与 epoch 一起）。
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                f"dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts
        for episode_idx in np.flatnonzero(used & (lengths <= 0)):
            logger.warning(
                "Episode %d has %d frames but drop_n_first_frames=%d and "
                "drop_n_last_frames=%d removes all frames. Skipping.",
                episode_idx,
                to_indices[episode_idx] - from_indices[episode_idx],
                drop_n_first_frames,
                drop_n_last_frames,
            )
        used &= lengths > 0
        if not used.any():
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self._starts = starts[used]
        self._cum_lengths = np.cumsum(lengths[used])
        self._num_frames = int(self._cum_lengths[-1])
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._start_index = 0
        self._absolute_to_relative = absolute_to_relative_idx

    @property
    def indices(self) -> list[int]:
        """按未打乱顺序物化的帧索引；O(num_frames)，仅用于内省。"""
        return [self._frame_index(k) for k in range(self._num_frames)]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        # 从 (seed, epoch) 推导每个 epoch 的种子，使排列成为两者的纯函数，
        # 从而在不触碰全局 RNG 的情况下在每个 rank 上完全一致地复现。
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        absolute_idx = int(self._starts[episode]) + position_in_episode
        if self._absolute_to_relative is not None:
            return self._absolute_to_relative[absolute_idx]
        return absolute_idx

    def __iter__(self) -> Iterator[int]:
        # 立即推进 epoch 状态，而不是等到生成器首次被消费时。
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        if self.shuffle:
            order = torch.randperm(self._num_frames, generator=self._epoch_generator(epoch))
            for k in range(start, self._num_frames):
                yield self._frame_index(int(order[k]))
        else:
            for k in range(start, self._num_frames):
                yield self._frame_index(k)

    def __len__(self) -> int:
        return self._num_frames


def compute_sampler_state(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict:
    """将优化步数映射为 `EpisodeAwareSampler` 状态，以实现样本精确的恢复。

    在 accelerate 的批次分片下，一个 step 会消耗 `batch_size * num_processes` 个采样器
    位置，并且每个 rank 每个 epoch 会看到 `ceil(ceil(num_frames / batch_size) / num_processes)` 个批次
    （包含 `even_batches` 填充）。可以证明起始索引始终低于
    `num_frames`；这里的 `min` 是防御性的。

    假设（只有满足这些条件时恢复才是样本精确的）：
        - `num_processes` 和 `batch_size` 与写入检查点的那次运行一致。两者都会影响
          一个 step 消耗的位置数量，因此任一者发生变化都会导致 epoch/偏移量错误。
          调用方传入检查点中的 `num_processes` 和 `batch_size`，并在不匹配时发出警告。
        - accelerate 使用 `even_batches=True`（其默认值）。`ceil(... / num_processes)` 这一项
          对应的就是该填充；若 `even_batches=False`，每个 epoch 的批次数会不同，
          边界就会出错。
    """
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_frames)
    return {"epoch": epoch, "start_index": start_index}
