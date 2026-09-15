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
from pprint import pformat

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_IMAGE, OBS_PREFIX, OBS_STATE, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .storage import DEFAULT_STORAGE_FORMAT, load_dataset_metadata
from .streaming_dataset import StreamingLeRobotDataset
from .utils import resolve_episode_indices


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig,
    ds_meta: LeRobotDatasetMetadata,
    rename_map: dict[str, str] | None = None,
) -> dict[str, list] | None:
    """通过读取配置中的 'delta_indices' 属性来解析 delta_timestamps。

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): 从中读取 delta_indices 的配置。
            ``PreTrainedConfig`` 和具体的 ``RewardModelConfig`` 子类都暴露
            下面使用的 ``{observation,action,reward}_delta_indices`` 属性。
        ds_meta (LeRobotDatasetMetadata): 用于构建 delta_timestamps 的数据集，
            提供其特征和 fps。

    Returns:
        dict[str, list] | None: delta_timestamps 字典，例如：
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            如果结果字典为空，则返回 `None`。
    """
    # 只有选择启用特定模态历史信息的策略（目前是带 MEM 的 Pi05）
    # 才会定义这些；其他所有策略都回退到共享的观测索引。
    explicit_image_indices = getattr(cfg, "image_observation_delta_indices", None)
    image_indices = (
        explicit_image_indices if explicit_image_indices is not None else cfg.observation_delta_indices
    )
    explicit_state_indices = getattr(cfg, "state_observation_delta_indices", None)
    state_indices = (
        explicit_state_indices if explicit_state_indices is not None else cfg.observation_delta_indices
    )

    delta_timestamps = {}
    matched_image_keys = []
    for key in ds_meta.features:
        policy_key = (rename_map or {}).get(key, key)
        if policy_key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if policy_key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        # `OBS_IMAGE` 同时匹配 `observation.image` 和 `observation.images.<cam>`
        # 两种约定；如果只匹配 `OBS_IMAGES`，会悄无声息地让使用单数键的
        # 数据集完全没有图像历史。
        if policy_key.startswith(OBS_IMAGE):
            indices = image_indices
            matched_image_keys.append(key)
        elif policy_key == OBS_STATE:
            indices = state_indices
        else:
            indices = cfg.observation_delta_indices if policy_key.startswith(OBS_PREFIX) else None
        if indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in indices]

    # 如果策略请求的图像历史没有任何数据集键能够提供，训练会在
    # 单帧上进行而不会报任何错误，因此这里选择直接报错而不是静默降级。
    if explicit_image_indices is not None and len(explicit_image_indices) > 1 and not matched_image_keys:
        raise ValueError(
            f"{type(cfg).__name__} requests {len(explicit_image_indices)} history frames per camera, but no "
            f"dataset feature maps to an image key. Dataset features: {sorted(ds_meta.features)}. "
            "Image keys must be named `observation.image*` after applying `--rename_map`."
        )

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """处理创建数据集之前设置 delta timestamps 和图像变换的逻辑。

    Args:
        cfg (TrainPipelineConfig): 包含 DatasetConfig 和 PreTrainedConfig 的 TrainPipelineConfig 配置。

    Raises:
        NotImplementedError: MultiLeRobotDataset 当前处于停用状态。

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        # 存储感知的加载器：与 LeRobotDatasetMetadata(...) 相同，另外支持
        # 根路径为对象存储 URI（例如 ``hf://``）的数据集。
        ds_meta = load_dataset_metadata(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            revision=cfg.dataset.revision,
            repo_type=cfg.dataset.repo_type,
        )
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta, cfg.rename_map)
        episodes = resolve_episode_indices(
            cfg.dataset.episodes, ds_meta.total_episodes, cfg.dataset.exclude_episodes
        )
        if cfg.dataset.streaming and ds_meta.storage_format != DEFAULT_STORAGE_FORMAT:
            raise ValueError(
                f"dataset.streaming=True is not supported for storage_format="
                f"{ds_meta.storage_format!r}: StreamingLeRobotDataset only reads the default "
                f"{DEFAULT_STORAGE_FORMAT!r} layout. Note that some formats (e.g. 'lance') "
                "support remote map-style access without streaming mode."
            )
        if not cfg.dataset.streaming:
            if cfg.dataset.repo_type == "bucket" and ds_meta.storage_format == DEFAULT_STORAGE_FORMAT:
                raise ValueError(
                    f"repo_type='bucket' is streaming-only for the default {DEFAULT_STORAGE_FORMAT!r} "
                    "storage format: set dataset.streaming=true to train from an HF Storage Bucket."
                )
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                tolerance_s=cfg.tolerance_s,
                repo_type=cfg.dataset.repo_type,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
                repo_type=cfg.dataset.repo_type,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): 为多数据集添加正式支持
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            if key in dataset.meta.depth_keys:
                continue  # 将深度键排除在 ImageNet 统计量之外
            dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def make_train_eval_datasets(
    cfg: TrainPipelineConfig,
) -> tuple[LeRobotDataset | MultiLeRobotDataset, LeRobotDataset | None]:
    """根据 eval_split 划分 episode，创建训练数据集和可选的评估数据集。

    每个任务的最后 ceil(n_episodes * eval_split) 个 episode 被留出用于评估。
    如果 eval_split == 0.0，返回 (full_dataset, None)。
    """
    full_dataset = make_dataset(cfg)

    if cfg.dataset.eval_split == 0.0:
        return full_dataset, None

    base_episodes = (
        full_dataset.episodes if full_dataset.episodes is not None else list(range(full_dataset.num_episodes))
    )

    episode_tasks = full_dataset.meta.episodes["tasks"]
    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx in base_episodes:
        task_key = episode_tasks[ep_idx][0] if episode_tasks[ep_idx] else ""
        task_to_episodes.setdefault(task_key, []).append(ep_idx)

    train_episodes, eval_episodes = [], []
    for eps in task_to_episodes.values():
        n_eval = math.ceil(len(eps) * cfg.dataset.eval_split)
        train_episodes.extend(eps[: len(eps) - n_eval])
        eval_episodes.extend(eps[len(eps) - n_eval :])

    if not train_episodes:
        raise ValueError(
            f"eval_split={cfg.dataset.eval_split} leaves 0 training episodes from {len(base_episodes)} total."
        )

    logging.info(
        f"Train/eval split: {len(train_episodes)} train, {len(eval_episodes)} eval "
        f"(eval_split={cfg.dataset.eval_split}, {len(task_to_episodes)} tasks)"
    )

    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, full_dataset.meta, cfg.rename_map)

    train_image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    train_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=train_image_transforms,
        depth_output_unit=cfg.dataset.depth_output_unit,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        repo_type=cfg.dataset.repo_type,
    )

    eval_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        depth_output_unit=cfg.dataset.depth_output_unit,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        repo_type=cfg.dataset.repo_type,
    )

    if cfg.dataset.use_imagenet_stats:
        for ds in (train_dataset, eval_dataset):
            for key in ds.meta.camera_keys:
                if key in ds.meta.depth_keys:
                    continue
                ds.meta.stats.setdefault(key, {})
                for stats_type, stats in IMAGENET_STATS.items():
                    ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return train_dataset, eval_dataset
