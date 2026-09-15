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
from collections.abc import Callable
from pathlib import Path

import datasets
import torch
import torch.utils

from lerobot.utils.constants import HF_LEROBOT_HOME

from .compute_stats import aggregate_stats
from .feature_utils import get_hf_features_from_features
from .lerobot_dataset import LeRobotDataset
from .video_utils import VideoFrame

logger = logging.getLogger(__name__)


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """由多个底层 `LeRobotDataset` 组成的数据集。

    底层的多个 `LeRobotDataset` 实际上是被拼接在一起的，该类沿用了 `LeRobotDataset` 的大部分 API
    结构。
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        *,
        token: str | bool | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        # 构造底层数据集时传入除 `transform` 和 `delta_timestamps` 之外的所有参数，
        # 这两项由本类处理。
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
                token=token,
            )
            for repo_id in repo_ids
        ]

        # 禁用并非所有数据集共有的数据键。注意：我们可能会在该类的后续版本中放宽这一
        # 限制。目前，至少为了能使用 PyTorch 默认的 DataLoader collate 函数，
        # 这样做是必要的。
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logger.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                    "other datasets."
                )
                self.disabled_features.update(extra_keys)

        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): 对于包含多个不同取值范围的机器人的数据集，
        # 我们不应执行这种聚合。相反，应该为每个机器人
        # 单独做一次归一化。
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])
        self.set_image_transforms(image_transforms)

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """替换本数据集及其子数据集的 transform。"""
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms
        for dataset in getattr(self, "_datasets", []):
            dataset.set_image_transforms(self.image_transforms)

    def clear_image_transforms(self) -> None:
        """移除本数据集及其子数据集的 transform。"""
        self.set_image_transforms(None)

    @property
    def repo_id_to_index(self):
        """返回从数据集 repo_id 到本类自动创建的数据集索引的映射。

        该索引会作为一个数据键并入 `__getitem__` 返回的字典中。
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def fps(self) -> int:
        """数据采集时使用的每秒帧数。

        注意：目前这依赖于 __init__ 中的检查，以确保所有子数据集具有相同的信息。
        """
        return self._datasets[0].meta.info.fps

    @property
    def video(self) -> bool:
        """如果该数据集从 mp4 文件加载视频帧，则返回 True。

        如果只从 png 文件加载图像，则返回 False。

        注意：目前这依赖于 __init__ 中的检查，以确保所有子数据集具有相同的信息。
        """
        return len(self._datasets[0].meta.video_keys) > 0

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update(
                {
                    k: v
                    for k, v in get_hf_features_from_features(dataset.features).items()
                    if k not in self.disabled_features
                }
            )
        return features

    @property
    def camera_keys(self) -> list[str]:
        """用于访问相机图像和视频流的键。"""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image | VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """需要解码为图像才能访问的视频帧的键。

        注意：如果数据集只包含图像，则为空；
        如果数据集只包含视频，则等于 `self.cameras`；
        在图像/视频混合数据集的情况下，甚至可以是 `self.cameras` 的子集。
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """样本/帧的数量。"""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """episode 的数量。"""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """当加载帧的时间戳与请求帧不够接近时，用于丢弃这些帧的容差（以秒为单位）。
        仅在提供了 `delta_timestamps` 或从 mp4 文件加载视频帧时使用。
        """
        # 1e-4 用于考虑可能出现的数值误差
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # 根据索引确定要从哪个数据集获取元素。
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
