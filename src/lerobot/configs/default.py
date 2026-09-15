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
from dataclasses import dataclass, field

from lerobot.transforms import ImageTransformsConfig
from lerobot.utils.import_utils import get_safe_default_video_backend

from .video import DEFAULT_DEPTH_UNIT, DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    # 这里可以提供一个数据集列表。`train.py` 会创建并拼接所有数据集。注意：只保留
    # 各数据集之间共有的数据键。每个数据集都会得到一个额外的变换，将
    # "dataset_index" 插入到返回的样本中。索引映射按照数据集提供的顺序进行。
    repo_id: str
    # Hub 仓库类型："dataset"（默认）或 "bucket"（表示 HF 存储桶，
    # hf://buckets/）。存储桶是否需要 streaming=true 取决于数据集的存储格式，
    # 因此在数据集工厂加载时进行检查。
    repo_type: str = "dataset"
    # 具体本地数据集树（例如 'dataset/path'）的根目录。如果为 None，本地数据集
    # 会在 $HF_LEROBOT_HOME/repo_id 下查找，Hub 下载则使用 $HF_LEROBOT_HOME/hub 下的
    # revision 安全缓存。
    root: str | None = None
    episodes: list[int] | None = None
    # 要剔除的剧集索引（例如损坏或异构的剧集）。在 `episodes` 之上应用。
    exclude_episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_video_backend)
    # 当为 True 时，RGB 视频帧以 uint8 张量（0-255）返回，而不是 float32（0.0-1.0）。
    # 这会减少内存占用并加快 DataLoader 的 IPC。训练流水线会处理转换。
    return_uint8: bool = False
    # 加载时深度图反量化到的物理单位："mm"（毫米）或 "m"（米）。
    # 对没有深度相机的数据集无效。
    depth_output_unit: str = DEFAULT_DEPTH_UNIT
    streaming: bool = False
    # 每个任务留出用于离线评估的剧集比例（0.0 = 禁用）。
    eval_split: float = 0.0

    def __post_init__(self) -> None:
        if self.repo_type not in ("dataset", "bucket"):
            raise ValueError(f"repo_type must be 'dataset' or 'bucket', got {self.repo_type!r}")
        if self.eval_split != 0.0 and self.streaming:
            raise ValueError(
                "eval_split requires map-style datasets and is not supported with dataset.streaming=true."
            )
        if self.depth_output_unit not in (DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT):
            raise ValueError(
                f"depth_output_unit must be '{DEPTH_METER_UNIT}' or '{DEPTH_MILLIMETER_UNIT}', got {self.depth_output_unit!r}"
            )
        if not (0.0 <= self.eval_split < 1.0):
            raise ValueError(f"eval_split must be in [0.0, 1.0), got {self.eval_split}")
        if self.episodes is not None:
            if any(ep < 0 for ep in self.episodes):
                raise ValueError(
                    f"Episode indices must be non-negative, got: {[ep for ep in self.episodes if ep < 0]}"
                )
            if len(self.episodes) != len(set(self.episodes)):
                duplicates = sorted({ep for ep in self.episodes if self.episodes.count(ep) > 1})
                raise ValueError(f"Episode indices contain duplicates: {duplicates}")
        if self.exclude_episodes is not None:
            negative_episodes = [episode for episode in self.exclude_episodes if episode < 0]
            if negative_episodes:
                logger.warning(
                    "Ignoring negative exclude_episodes entries: %s",
                    negative_episodes,
                )
                self.exclude_episodes = [episode for episode in self.exclude_episodes if episode >= 0]


@dataclass
class WandBConfig:
    enable: bool = False
    # 设置为 true 时，即使 training.save_checkpoint=True 也不保存 artifact
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    resume: str | None = None  # 允许的值：'allow'、'must'、'never' 或 'auto'。
    mode: str | None = None  # 允许的值：'online'、'offline'、'disabled'。默认为 'online'
    console: str = "wrap"
    console_multipart: bool = False
    console_chunk_max_seconds: int = 0
    add_tags: bool = True  # 如果为 True，将配置作为标签保存到 WandB 运行中。


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` 指定 gym.vector.VectorEnv 中使用的环境数量。
    # 设置为 0 时会根据可用 CPU 核心数和 n_episodes 自动调整。
    batch_size: int = 0
    # `use_async_envs` 指定是否使用异步环境（多进程）。
    # 默认为 True；当 batch_size=1 时自动降级为 SyncVectorEnv。
    use_async_envs: bool = True
    # 是否将评估 rollout 记录为磁盘上的 LeRobot 数据集。
    recording: bool = False
    # 如果设置，将记录的评估数据集推送到此仓库 id 下的 Hub（每个任务一个仓库，
    # 以任务和环境索引作为后缀）。需要 recording=true。
    recording_repo_id: str | None = None
    # 推送的记录仓库是否应为私有。
    recording_private: bool = False

    def __post_init__(self) -> None:
        if self.recording_repo_id is not None and not self.recording:
            raise ValueError("eval.recording_repo_id requires eval.recording=true.")
        if self.batch_size == 0:
            self.batch_size = self._auto_batch_size()
        if self.batch_size > self.n_episodes:
            self.batch_size = self.n_episodes

    def _auto_batch_size(self) -> int:
        """根据 CPU 核心数选择 batch_size，并以 n_episodes 为上限。"""
        import math
        import os

        cpu_cores = os.cpu_count() or 4
        # 每个异步环境 worker 大约需要 1 个核心；为主进程 + 推理留出余量。
        by_cpu = max(1, math.floor(cpu_cores * 0.7))
        return min(by_cpu, self.n_episodes, 64)


@dataclass
class EMAConfig:
    """策略权重的指数移动平均（EMA）。

    扩散类策略的标准做法（Chi et al. 2023, "Diffusion Policy", section V.D）：
    参考实现在所有配置中都启用它并评估 EMA 权重。这里默认关闭，
    因为它会在内存中保留参数的第二份完整副本。

    衰减遵循 diffusers 的 `EMAModel` 中的预热调度：
    `decay_t = 1 - (1 + t / inv_gamma) ** -power`，并限制在 `[min_decay, max_decay]` 范围内。
    默认值与参考实现一致。或者，也可以设置 `decay` 在每一步使用恒定衰减，
    如 openpi 在 pi0/pi05 中所用（`ema_decay=0.99`）。
    """

    enable: bool = False
    # 恒定衰减系数（openpi 风格，例如 pi0/pi05 的 0.99）。设置后，
    # 下面的预热调度将被绕过，影子权重在每一步都使用此衰减。
    decay: float | None = None
    # 影子权重保持为实时权重硬拷贝的优化器步数。
    update_after_step: int = 0
    # 预热调度参数（参见类的 docstring）。
    inv_gamma: float = 1.0
    power: float = 0.75
    min_decay: float = 0.0
    max_decay: float = 0.9999
    # 在周期性环境评估期间评估 EMA 权重（而不是实时权重）。
    # 离线评估损失（--eval_steps）始终使用实时权重：它在每个 rank 上运行，
    # 而 EMA 影子只存在于主进程中。
    use_for_eval: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_decay <= self.max_decay <= 1.0):
            raise ValueError(
                "Expected 0 <= ema.min_decay <= ema.max_decay <= 1, got "
                f"min_decay={self.min_decay} and max_decay={self.max_decay}."
            )
        if self.inv_gamma <= 0:
            raise ValueError(f"ema.inv_gamma must be positive, got {self.inv_gamma}.")
        if self.power <= 0:
            raise ValueError(f"ema.power must be positive, got {self.power}.")
        if self.update_after_step < 0:
            raise ValueError(f"ema.update_after_step must be >= 0, got {self.update_after_step}.")
        if self.decay is not None:
            if not 0.0 <= self.decay <= 1.0:
                raise ValueError(f"ema.decay must be in [0, 1], got {self.decay}.")
            # 保持字面量与上面字段默认值同步。
            if self.min_decay != 0.0 or self.max_decay != 0.9999:
                raise ValueError(
                    "ema.decay (constant decay) and ema.min_decay/ema.max_decay (schedule clamp) are "
                    "mutually exclusive: set one or the other."
                )


@dataclass
class PeftConfig:
    # PEFT 提供了许多微调方法，其中层适配器是最常见的，目前也是最有效的方法，
    # 因此我们将在这个高层配置接口中专注于这些方法。

    # 可以是一个字符串（模块名后缀或 'all-linear'）、一个模块名后缀列表，
    # 或一个描述要用所配置的 PEFT 方法适配的模块名的正则表达式。
    # 一些策略对此有默认值，这样你就不必*必须*选择要适配哪些层，
    # 但根据你的情况，手动设置可能仍然值得。
    target_modules: list[str] | str | None = None

    # 要完全微调并与适配器权重一起存储的模块名称/后缀。对于不属于
    # 预训练模型的层（例如动作状态投影层）很有用。根据策略不同，
    # 默认值为预训练策略中新创建的层。如果你在微调一个已经训练过的策略，
    # 可能想将其设置为 `[]`。对应 PEFT 的 `modules_to_save`。
    full_training_modules: list[str] | None = None

    # 要应用于策略的 PEFT（适配器）方法。需要是有效的 PEFT 类型。
    method_type: str = "LORA"

    # 适配器初始化方法。默认值请查阅具体的 PEFT 适配器文档。
    init_type: str | None = None

    # 我们预期所有 PEFT 适配器都在某种程度上做秩分解，因此此参数指定
    # 适配器使用的秩。一般来说，秩越高意味着可训练参数越多，越接近完全微调。
    r: int = 16

    # LoRA 缩放的 Alpha 参数（scaling = lora_alpha / r）。
    # 一般来说，alpha 越高意味着适配信号越强。
    # 如果为 None，PEFT 库默认为 alpha=8，这可能会抑制高秩适配器的效果。
    # 常用的值是 r（alpha == rank）或 2*r。
    lora_alpha: int | None = None


@dataclass
class JobConfig:
    # 训练运行的位置。None（省略）或 "local" 表示在本机运行。
    # 其他任何值都是 HF Jobs 规格，会将运行提交到 HF Jobs。
    # 使用 `hf jobs hardware` 命令列出可用的规格和价格。
    target: str | None = None
    # 远程作业的运行时镜像（本地运行时忽略）。
    image: str = "huggingface/lerobot-gpu:latest"
    # 远程作业的最大运行时长，以 HF Jobs 时长字符串表示（例如 "2h"）。
    # 默认为 "2d"：我们改为传递一个显式的、宽裕的上限。设置更小的值
    # 可以快速失败，设置更大的值适合长时间运行。
    timeout: str | None = "2d"
    # 提交后即退出，而不是在前台流式输出作业日志。
    detach: bool = False
    # 附加到 HF 作业以及本次运行推送到 Hub 的任何数据集的额外标签。
    # 始终会添加 "lerobot" 标签；例如 --job.tags '["lelab"]' 可以添加更多。
    tags: list[str] = field(default_factory=list)

    # 同一个谓词的两个入口：staticmethod 直接测试来自 argv 的原始 target 字符串
    # （在任何 JobConfig 存在之前，用于尽早决定分派），而 property 则是
    # 为已经持有配置实例的代码提供的易用访问器。
    @staticmethod
    def is_remote_target(target: str | None) -> bool:
        """当 `target` 指定的是 HF Jobs 规格而非本地运行时为 True。"""
        return target not in (None, "local")

    @property
    def is_remote(self) -> bool:
        """当训练应该在 HF Jobs 上运行而非本机时为 True。"""
        return self.is_remote_target(self.target)
