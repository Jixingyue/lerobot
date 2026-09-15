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
""" 可视化给定配置下图像变换的效果。

本脚本会生成变换后图像的示例，其输出与 LeRobot 数据集的输出一致。
此外，每个单独的变换都可以单独可视化，也可以可视化组合变换的示例

示例：
```bash
lerobot-imgtransform-viz \
  --repo_id=lerobot/pusht \
  --episodes='[0]' \
  --image_transforms.enable=True
```
"""

import logging
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import draccus
from torchvision.transforms import ToPILImage

from lerobot.configs import DatasetConfig
from lerobot.datasets import LeRobotDataset
from lerobot.transforms import (
    ImageTransforms,
    ImageTransformsConfig,
    make_transform_from_config,
)

OUTPUT_DIR = Path("outputs/image_transforms")
to_pil = ToPILImage()


def save_all_transforms(cfg: ImageTransformsConfig, original_frame, output_dir, n_examples):
    output_dir_all = output_dir / "all"
    output_dir_all.mkdir(parents=True, exist_ok=True)

    tfs = ImageTransforms(cfg)
    for i in range(1, n_examples + 1):
        transformed_frame = tfs(original_frame)
        to_pil(transformed_frame).save(output_dir_all / f"{i}.png", quality=100)

    print("Combined transforms examples saved to:")
    print(f"    {output_dir_all}")


def save_each_transform(cfg: ImageTransformsConfig, original_frame, output_dir, n_examples):
    if not cfg.enable:
        logging.warning(
            "No single transforms will be saved, because `image_transforms.enable=False`. To enable, set `enable` to True in `ImageTransformsConfig` or in the command line with `--image_transforms.enable=True`."
        )
        return

    print("Individual transforms examples saved to:")
    for tf_name, tf_cfg in cfg.tfs.items():
        # 使用 min_max 范围内的随机值应用若干次变换
        output_dir_single = output_dir / tf_name
        output_dir_single.mkdir(parents=True, exist_ok=True)

        tf = make_transform_from_config(tf_cfg)
        for i in range(1, n_examples + 1):
            transformed_frame = tf(original_frame)
            to_pil(transformed_frame).save(output_dir_single / f"{i}.png", quality=100)

        # 应用最小值、最大值、平均值变换
        tf_cfg_kwgs_min = deepcopy(tf_cfg.kwargs)
        tf_cfg_kwgs_max = deepcopy(tf_cfg.kwargs)
        tf_cfg_kwgs_avg = deepcopy(tf_cfg.kwargs)

        for key, (min_, max_) in tf_cfg.kwargs.items():
            avg = (min_ + max_) / 2
            tf_cfg_kwgs_min[key] = [min_, min_]
            tf_cfg_kwgs_max[key] = [max_, max_]
            tf_cfg_kwgs_avg[key] = [avg, avg]

        tf_min = make_transform_from_config(replace(tf_cfg, **{"kwargs": tf_cfg_kwgs_min}))
        tf_max = make_transform_from_config(replace(tf_cfg, **{"kwargs": tf_cfg_kwgs_max}))
        tf_avg = make_transform_from_config(replace(tf_cfg, **{"kwargs": tf_cfg_kwgs_avg}))

        tf_frame_min = tf_min(original_frame)
        tf_frame_max = tf_max(original_frame)
        tf_frame_avg = tf_avg(original_frame)

        to_pil(tf_frame_min).save(output_dir_single / "min.png", quality=100)
        to_pil(tf_frame_max).save(output_dir_single / "max.png", quality=100)
        to_pil(tf_frame_avg).save(output_dir_single / "mean.png", quality=100)

        print(f"    {output_dir_single}")


@draccus.wrap()
def visualize_image_transforms(cfg: DatasetConfig, output_dir: Path = OUTPUT_DIR, n_examples: int = 5):
    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        episodes=cfg.episodes,
        revision=cfg.revision,
        video_backend=cfg.video_backend,
    )

    output_dir = output_dir / cfg.repo_id.split("/")[-1]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取第 1 个 episode 的第 1 个相机的第 1 帧
    original_frame = dataset[0][dataset.meta.camera_keys[0]]
    to_pil(original_frame).save(output_dir / "original_frame.png", quality=100)
    print("\nOriginal frame saved to:")
    print(f"    {output_dir / 'original_frame.png'}.")

    save_all_transforms(cfg.image_transforms, original_frame, output_dir, n_examples)
    save_each_transform(cfg.image_transforms, original_frame, output_dir, n_examples)


def main():
    visualize_image_transforms()


if __name__ == "__main__":
    main()
