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
"""训练一个策略。

需要：pip install 'lerobot[training]'  （包含 dataset、accelerate 和 wandb 附加依赖）

分布式运行请使用 torchrun 启动；所有并行/加速选项都位于配置中
（`--parallelism.*`、`--accelerator.*`），因此仅凭
train_config.json 即可复现一次运行：

```bash
torchrun --nproc-per-node=8 $(which lerobot-train) \
    --dataset.repo_id=... --policy.type=act \
    --parallelism.dp_shard=8 --accelerator.mixed_precision=bf16
```
"""

import dataclasses
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_metadata,
    publish_trained_model,
    push_checkpoint_to_hub,
    resume_after_prepare,
    resume_before_prepare,
    save_checkpoint,
    should_save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import JobConfig, parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.distributed import (
    ParallelDims,
    finalize_sharded_policy,
    is_main_process,
    make_accelerator,
    set_fsdp_wrap_modules,
)
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.jobs import submit_to_hf
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.policies.factory import ProcessorConfigKwargs
from lerobot.processor.rename_processor import rename_batch_keys, rename_stats
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR
from lerobot.utils.import_utils import _peft_available, register_third_party_plugins, require_package
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

if TYPE_CHECKING or _peft_available:
    from peft import PeftModel
else:
    PeftModel = None

from .lerobot_eval import eval_policy_all

EMA_STATE_FILENAME = "ema_state.pt"


@contextmanager
def _ema_weights(ema: Any, policy: PreTrainedPolicy) -> Iterator[None]:
    """临时将 EMA 影子权重换入 `policy`，退出时恢复实时权重。"""
    params = list(policy.parameters())
    ema.store(params)
    ema.copy_to(params)
    try:
        yield
    finally:
        ema.restore(params)


@contextmanager
def _make_eval_envs(cfg: TrainPipelineConfig) -> Iterator[dict[str, dict[int, Any]]]:
    """为一次运行创建评估环境，并确保始终释放它们。"""
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
    )
    try:
        yield envs
    finally:
        close_envs(envs)


def _preprocess_dataset_batch(
    batch: dict[str, Any],
    camera_keys: list[str],
    rename_map: dict[str, str],
    preprocessor: Any,
) -> Any:
    """以完全相同的方式为训练和留出评估准备原始数据集批次。"""
    for cam_key in camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
    batch = rename_batch_keys(batch, rename_map)
    return preprocessor(batch)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
) -> tuple[MetricsTracker, dict | None]:
    """
    执行单个训练步以更新策略权重。

    此函数执行前向和反向传播、裁剪梯度，并步进优化器和学习率调度器。
    Accelerator 会自动处理混合精度训练，并且——在梯度累积下——
    会在非末尾微批次上抑制梯度同步，并对损失进行缩放。

    参数:
        train_metrics (MetricsTracker)：用于记录训练统计信息的 MetricsTracker 实例。
        policy (PreTrainedPolicy)：待训练的策略模型（即 `accelerator.prepare` 返回的模型）。
        batch (Any)：一个训练数据批次。
        optimizer (Optimizer)：用于更新策略参数的优化器。
        grad_clip_norm (float)：梯度裁剪的最大范数（<= 0 时不裁剪）。
        accelerator (Accelerator)：用于分布式训练和混合精度的 Accelerator 实例。
        lr_scheduler (LRScheduler | None，可选)：可选的学习率调度器，每个微批次
            步进一次。默认为 None。
        lock (Lock | None，可选)：可选的锁，用于线程安全的优化器更新。
            默认为 None。
        sample_weighter (SampleWeighter | None，可选)：可选的 SampleWeighter 实例，
            用于按样本加权损失。默认为 None。

    返回:
        tuple[MetricsTracker, dict | None]：更新后的 MetricsTracker（包含本步的新统计
        信息），以及策略前向传播输出的字典（用于日志记录）。
    """
    start_time = time.perf_counter()
    policy.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 如果提供了加权器，则计算样本权重
    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    # 在梯度累积下，此上下文会在非末尾微批次上抑制梯度同步（FSDP2：
    # set_requires_gradient_sync）并对损失做除法；
    # 当 gradient_accumulation_steps == 1 时，它是一个透明的空操作。
    with accelerator.accumulate(policy):
        # 让 accelerator 处理混合精度
        with accelerator.autocast():
            # 使用 `policy(...)`，绝不要用 `policy.forward(...)`：FSDP2 通过
            # nn.Module 的 forward 钩子来 all-gather 参数，而钩子只会经由 __call__ 触发。
            if sample_weights is not None:
                # 使用逐样本损失进行加权训练
                # 注意：支持样本加权的策略必须实现 forward(batch, reduction="none")
                per_sample_loss, output_dict = policy(batch, reduction="none")

                # 加权损失：每个样本的贡献按其权重缩放。
                # 我们除以权重之和（而不是批次大小），这样当某些权重为零时，
                # 其余样本会按比例贡献更多，从而保持梯度尺度。
                # 权重会被预先归一化为总和等于 batch_size，以获得稳定的训练动态。
                epsilon = 1e-6
                loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)

                # 记录加权统计信息
                if output_dict is None:
                    output_dict = {}
                for key, value in weight_stats.items():
                    output_dict[f"sample_weight_{key}"] = value
            else:
                loss, output_dict = policy(batch)

            # TODO(rcadene)：policy.unnormalize_outputs(out_dict)

        # 使用 accelerator 的 backward 方法
        accelerator.backward(loss)

        # 梯度只有在同步微批次上才是完整的；裁剪不完整的梯度毫无意义。
        # 始终传入完整的参数列表：accelerate 的 FSDP2 路径要求其与 prepare 后
        # 模型的参数精确匹配，才能得到全局正确的范数。
        grad_norm = None
        if accelerator.sync_gradients and grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        # 优化器步进（在梯度累积下，非末尾微批次上为空操作）
        with lock if lock is not None else nullcontext():
            optimizer.step()
        optimizer.zero_grad()

        # 每个批次（而不是每个 epoch）步进一次 PyTorch 调度器
        if lr_scheduler is not None:
            lr_scheduler.step()

    # 如果策略有 update 方法，则更新内部缓冲区。这些缓冲区跟踪的是优化器更新
    # （EMA、目标网络），而不是微批次：在梯度累积下以同步步为门控。
    if accelerator.sync_gradients and has_method(
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"
    ):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    if grad_norm is not None:
        train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    if torch.cuda.is_available():
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    # 聚合策略的标量输出，以便在日志窗口内记录并在各 rank 间归约。
    if output_dict:
        train_metrics.update_metrics(output_dict)
    return train_metrics, output_dict


def make_dataloaders(
    cfg: TrainPipelineConfig,
    dataset,
    eval_dataset,
    step: int,
    parallel_dims: ParallelDims,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
    """构建训练（以及可选的评估）dataloader，包括采样器的恢复偏移量。

    采样器偏移量是从 `step` *派生*出来的（`resume_before_prepare` 只加载步数 + RNG）：
    每个循环步在 `dp_world_size` 个相互独立的数据并行工作进程上各消耗
    `batch_size` 个样本——不含梯度累积因子，因为 `step` 计数的是微批次。

    参数:
        cfg (TrainPipelineConfig)：训练配置（批次大小、工作进程数、流式加载、恢复、种子）。
        dataset (LeRobotDataset | MultiLeRobotDataset)：训练数据集。
        eval_dataset (LeRobotDataset | None)：可选的留出分片；提供时会构建一个评估
            dataloader（当 `cfg.max_eval_samples > 0` 时按任务子采样）。
        step (int)：采样器恢复所对应的循环步（全新运行为 0）。
        parallel_dims (ParallelDims)：解析后的并行拓扑；提供设备类型以及
            恢复偏移量回退使用的数据并行世界大小。

    返回:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]：训练
        dataloader 和评估 dataloader（不存在评估分片时为 None）。
    """
    active_cfg = cfg.trainable_config
    if not cfg.dataset.streaming:
        # 所有非流式（map 风格）数据集都使用 EpisodeAwareSampler。
        # 顺序是 (seed, epoch) 的纯函数，因此每个 rank 都会独立产生
        # 相同的排列。随后 accelerate 通过 BatchSamplerShard 将其不相交地
        # 分片到各个数据并行 rank 上，无需借助 `generator` 属性来同步 RNG，
        # 并且恢复可以做到样本级精确。
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
            shuffle=True,
            seed=cfg.seed if cfg.seed is not None else 0,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        if cfg.resume and step > 0:
            # 恢复偏移量取决于产生 `step` 时所用的 (dp_world_size, batch_size)，
            # 因此使用检查点中记录的值（对于未存储这些值的旧检查点，
            # 回退到当前值）。
            metadata = load_training_metadata(cfg.checkpoint_path / TRAINING_STATE_DIR)
            saved_dp_world = metadata["dp_world_size"]
            saved_batch_size = metadata["batch_size"]
            ckpt_dp_world = saved_dp_world or parallel_dims.dp_world_size
            ckpt_batch_size = saved_batch_size or cfg.batch_size
            if is_main_process() and saved_dp_world not in (None, parallel_dims.dp_world_size):
                logging.warning(
                    f"Resuming with dp_world_size={parallel_dims.dp_world_size} but the "
                    f"checkpoint was written with dp_world_size={saved_dp_world}. The data order "
                    "resumes at the right epoch/offset, but per-rank sample-exactness requires "
                    "the same data-parallel world size."
                )
            if is_main_process() and saved_batch_size not in (None, cfg.batch_size):
                logging.warning(
                    f"Resuming with batch_size={cfg.batch_size} but the checkpoint was written "
                    f"with batch_size={saved_batch_size}. The data order resumes at the right "
                    "epoch/offset, but per-rank sample-exactness requires the same batch size."
                )
            sampler_state = compute_sampler_state(step, len(sampler), ckpt_batch_size, ckpt_dp_world)
            sampler.load_state_dict(sampler_state)
            if is_main_process():
                logging.info(
                    f"Resuming data order at epoch {sampler_state['epoch']}, "
                    f"sample {sampler_state['start_index']}"
                )
    else:
        shuffle = True
        sampler = None

    device_type = parallel_dims.device_type
    # 仅当数据集确实声明了语言列时，才换用感知语言的 collate；
    # 否则保持使用 PyTorch 的默认 collate，以免影响
    # 不含语言的训练运行。
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device_type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        multiprocessing_context=cfg.dataloader_multiprocessing_context if cfg.num_workers > 0 else None,
    )

    # 如果存在留出分片，则构建评估 dataloader
    eval_dataloader = None
    if eval_dataset is not None:
        eval_ds = eval_dataset
        if cfg.max_eval_samples > 0 and hasattr(eval_dataset, "hf_dataset"):
            task_arr = eval_dataset.hf_dataset.data.column("task_index").to_numpy()
            unique_tasks = sorted(set(task_arr.tolist()))
            per_task = max(1, cfg.max_eval_samples // len(unique_tasks))
            selected: list[int] = []
            for t in unique_tasks:
                frames = (task_arr == t).nonzero()[0][:per_task]
                selected.extend(frames.tolist())
            eval_ds = torch.utils.data.Subset(eval_dataset, selected)

        eval_collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device_type == "cuda",
            drop_last=False,
            collate_fn=eval_collate_fn,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            multiprocessing_context=cfg.dataloader_multiprocessing_context if cfg.num_workers > 0 else None,
        )
    return dataloader, eval_dataloader


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """
    训练策略的主函数。

    此函数统筹整个训练流水线，包括：
    - 设置日志、随机种子和分布式引擎。
    - 创建数据集、评估环境（如适用）、策略和优化器。
    - 处理从检查点恢复（围绕 `accelerator.prepare` 分两阶段进行）。
    - 运行主训练循环，包括获取数据批次并调用 `update_policy`。
    - 定期记录指标、保存模型检查点并评估策略。
    - 如果已配置，将训练好的模型发布到 Hugging Face Hub。

    参数:
        cfg (TrainPipelineConfig)：包含所有训练配置的 `TrainPipelineConfig`
            对象，由 `parser.wrap()` 从 CLI 解析。使用 `--resume` 时，它是检查点
            的 `train_config.json` 中记录的配置；当 `cfg.job.is_remote` 时，运行
            会被派发到 HF Jobs，而不是在本地执行。
    """
    if cfg.job.is_remote:
        return submit_to_hf(cfg)

    require_package("accelerate", extra="training")

    cfg.validate()  # 所有快速失败检查都在此处触发，先于任何分布式初始化

    # --- 引擎与拓扑 --------------------------------------------------------------------
    # 该工厂是唯一的 accelerate 配置点：它会防范环境变量
    # 干扰，根据启动的 world 解析声明的并行度，并
    # 基于配置镜像构建 Accelerator。
    accelerator = make_accelerator(cfg)
    parallel_dims = ParallelDims.from_config(
        cfg.parallelism, accelerator.num_processes, accelerator.device.type
    )
    init_logging(accelerator=accelerator)

    if is_main_process():
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process():
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process():
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- 数据（主进程下载一次；其他 peer 读取已填充的缓存）----------------
    if is_main_process():
        logging.info("Creating dataset")
        dataset, eval_dataset = make_train_eval_datasets(cfg)
    accelerator.wait_for_everyone()
    if not is_main_process():
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    # --- 策略（权重来源由恢复规则决定）-------------------------------------
    # 恢复时，cfg 是从检查点的 train_config.json 解析而来的，因此
    # cfg.checkpoint_format 就是记录下来的值：携带 DCP 的格式会在此跳过
    # safetensors 加载，而在 prepare 之后流式载入分片权重
    # （resume_after_prepare）。
    defer_weight_load = cfg.resume and cfg.checkpoint_format.wants_dcp
    if cfg.is_reward_model_training:
        if is_main_process():
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process():
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
            defer_weight_load=defer_weight_load,
        )

    peft_model = None
    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        require_package("peft", extra="peft")

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)
        peft_model = policy

    accelerator.wait_for_everyone()

    # --- 处理器（覆盖项只构建一次，作为一个带类型的映射）-------------------------------
    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path
    if not cfg.resume and getattr(active_cfg, "recipe", None) is not None:
        if processor_pretrained_path is not None and is_main_process():
            logging.warning(
                "Language recipe fine-tuning rebuilds processors from the active configuration; "
                "saved processors from %s will not be loaded.",
                processor_pretrained_path,
            )
        # 语言微调必须使用当前生效的 recipe，而不是已保存的处理器 recipe。
        processor_pretrained_path = None

    processor_kwargs = ProcessorConfigKwargs()
    processor_dataset_stats = rename_stats(dataset.meta.stats, cfg.rename_map)
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = processor_dataset_stats
    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta
    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
        # 恢复时，以检查点中保存的处理器统计信息为准：它们可能已被
        # 策略调整过（例如 EVO1 会将 state/action 统计填充到 max_state_dim），
        # 强行灌入原始数据集统计会导致归一化崩溃（#4006）。
        # 这与上面的 `dataset_stats` 参数一致，该参数在恢复时同样会被跳过。
        if not cfg.resume:
            preprocessor_overrides["normalizer_processor"]["stats"] = processor_dataset_stats
            postprocessor_overrides["unnormalizer_processor"]["stats"] = processor_dataset_stats
        if getattr(active_cfg, "use_relative_actions", False):
            preprocessor_overrides["relative_actions_processor"] = {
                "enabled": True,
                "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
                "action_names": getattr(active_cfg, "action_feature_names", None),
            }
            postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides
        processor_kwargs["postprocessor_overrides"] = postprocessor_overrides

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
            **processor_kwargs,
        )

    # 在 prepare 之前、针对未分片的参数创建——accelerate 的 FSDP2 路径要求
    # 模型和优化器放在同一个 prepare() 调用中，并由它自己重新绑定参数组。
    if is_main_process():
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # --- 恢复阶段 1 + dataloader ----------------------------------------------------------
    step = 0  # 循环步数（= 每个数据并行工作进程消耗的微批次数）
    if cfg.resume:
        step = resume_before_prepare(cfg)  # 仅恢复步数 + RNG；分片状态在 prepare 之后加载

    dataloader, eval_dataloader = make_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims)

    # --- prepare 与恢复阶段 2 ---------------------------------------------------------------
    # FSDP 包装单元的类名在 prepare 之前即时解析：优先使用用户覆盖，
    # 否则使用策略的 _fsdp_wrap_modules 声明——绝不静默接受仅包装根节点。
    set_fsdp_wrap_modules(accelerator, accelerator.unwrap_model(policy) if peft_model else policy)
    accelerator.wait_for_everyone()
    if eval_dataloader is not None:
        policy, optimizer, dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler, eval_dataloader
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    finalize_sharded_policy(policy, parallel_dims)
    if cfg.resume:
        resume_after_prepare(cfg, accelerator, policy, optimizer, lr_scheduler)

    # --- 辅助组件（在核心装配之后，遵循构造顺序约定）-------------
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process():
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    # --- 信息横幅（仅主进程；numel() 读取的是元数据——在 DTensor 上它是
    # 全局形状，因此即使分片后总数也是正确的）---------------------------------------------
    # 一个循环步在每个 dp 工作进程上消耗一个微批次；优化器每次更新看到
    # `samples_per_step x gradient_accumulation_steps` 个样本。
    samples_per_step = cfg.batch_size * parallel_dims.dp_world_size
    effective_batch_size = samples_per_step * cfg.accelerator.gradient_accumulation.steps
    if is_main_process():
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(
            f"Effective batch size: {cfg.batch_size} x {parallel_dims.dp_world_size} dp workers "
            f"x {cfg.accelerator.gradient_accumulation.steps} grad accum = {effective_batch_size} "
            f"(topology: dp_replicate={parallel_dims.dp_replicate}, dp_shard={parallel_dims.dp_shard})"
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    dl_iter = cycle(dataloader)
    policy.train()

    # 策略权重的 EMA 影子副本（Chi 等人 2023，Diffusion Policy，第 V.D 节）。该影子
    # 副本仅存在于主进程上，这在 DDP 下是安全的，因为每次梯度同步后每个 rank
    # 都持有相同的权重。diffusers 采用懒导入，因此基础训练路径
    # 不依赖它。
    ema = None
    if cfg.ema.enable:
        if parallel_dims.is_sharded:
            raise NotImplementedError(
                "--ema.enable=true is not supported with sharded training (FSDP2/HSDP/CP): the "
                "parameters are sharded across ranks. Use a replicated (DDP) or single-GPU run."
            )
        if cfg.peft is not None:
            raise NotImplementedError("--ema.enable=true is not supported together with PEFT adapters.")
        require_package("diffusers", extra="diffusion")
        if is_main_process():
            from diffusers.training_utils import EMAModel  # noqa: PLC0415

            # 恒定的 --ema.decay 通过调度的钳制来表达：当
            # min_decay == max_decay 时，预热曲线在每一步都被钉在该值上。
            min_decay = cfg.ema.min_decay if cfg.ema.decay is None else cfg.ema.decay
            max_decay = cfg.ema.max_decay if cfg.ema.decay is None else cfg.ema.decay
            ema = EMAModel(
                accelerator.unwrap_model(policy).parameters(),
                decay=max_decay,
                min_decay=min_decay,
                update_after_step=cfg.ema.update_after_step,
                use_ema_warmup=True,
                inv_gamma=cfg.ema.inv_gamma,
                power=cfg.ema.power,
            )
            ema.to(device)
            if cfg.ema.decay is not None:
                logging.info(
                    "EMA enabled: decay=%g (constant), update_after_step=%d, use_for_eval=%s",
                    cfg.ema.decay,
                    cfg.ema.update_after_step,
                    cfg.ema.use_for_eval,
                )
            else:
                logging.info(
                    "EMA enabled: max_decay=%g, inv_gamma=%g, power=%g, update_after_step=%d, use_for_eval=%s",
                    cfg.ema.max_decay,
                    cfg.ema.inv_gamma,
                    cfg.ema.power,
                    cfg.ema.update_after_step,
                    cfg.ema.use_for_eval,
                )
            if cfg.checkpoint_path is not None:
                ema_path = cfg.checkpoint_path / TRAINING_STATE_DIR / EMA_STATE_FILENAME
                if ema_path.exists():
                    ema.load_state_dict(torch.load(ema_path, map_location=device, weights_only=True))
                    logging.info("Resumed EMA shadow from %s", ema_path)
                else:
                    logging.warning(
                        "Resuming with --ema.enable=true but %s is missing; "
                        "restarting the shadow from the current weights.",
                        ema_path,
                    )

    train_metrics = {
        # 每个 rank 的 loss 只反映全局批次的一个分片；取均值即可还原出
        # 数据并行组实际优化的 loss。grad_norm 和 lr 在每个 rank 上
        # 已经相同（梯度同步之后 / 确定性调度器），因此对它们做归约
        # 只会是一次空操作的集合通信。
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        # 对瓶颈类计时报告最慢的 rank，这样多 GPU 运行能暴露出
        # 真正的落后者，而不是只呈现 rank 0 的视角。
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        "preprocessing_s": AverageMeter("prep_s", ":.3f", reduction="max"),
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "step_s": AverageMeter("step_s", ":.3f", reduction="max"),
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if torch.cuda.is_available():
        # 取 max()，因为可用余量取决于最坏情况下的 rank。
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        dp_world_size=parallel_dims.dp_world_size,
    )

    if is_main_process():
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        step_start = time.perf_counter()
        batch = next(dl_iter)
        preprocessing_start = time.perf_counter()
        train_tracker.dataloading_s = preprocessing_start - step_start
        batch = _preprocess_dataset_batch(batch, dataset.meta.camera_keys, cfg.rename_map, preprocessor)
        train_tracker.preprocessing_s = time.perf_counter() - preprocessing_start

        train_tracker, _ = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
        )
        train_tracker.step_s = time.perf_counter() - step_start

        # 将实时权重的一个优化器步更新拉入 EMA 影子副本（仅主进程）。
        # 影子副本跟踪的是优化器更新，而不是微批次：在梯度累积下
        # 以同步步为门控。
        if ema is not None and accelerator.sync_gradients:
            ema.step(accelerator.unwrap_model(policy).parameters())

        # 注意：评估和检查点发生在第 `step` 次训练更新完成*之后*，因此
        # 我们在此处递增 `step`。
        step += 1
        if is_main_process():
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = should_save_checkpoint(step, cfg.save_freq, cfg.steps)
        is_env_eval_step = cfg.env_eval_freq > 0 and step % cfg.env_eval_freq == 0
        is_eval_step = cfg.eval_steps > 0 and eval_dataloader is not None and step % cfg.eval_steps == 0

        if is_log_step:
            # 集合归约必须在每个 rank 上运行，且位于下面的主进程门控之前。
            train_tracker.reduce_across_ranks()
            if is_main_process():
                if train_tracker.step_s.avg > 0:
                    train_tracker.samples_per_s = samples_per_step / train_tracker.step_s.avg
                logging.info(train_tracker)
                if wandb_logger:
                    # 策略的各个子损失（latent_loss、action_loss……）已由
                    # update_policy 聚合到跟踪器中，因此 to_dict() 已经携带了它们
                    # 加窗且跨 rank 归约后的平均值——无需逐步透传 output_dict。
                    wandb_log_dict = train_tracker.to_dict()
                    # 如果启用了样本加权，则记录其统计信息
                    if sample_weighter is not None:
                        weighter_stats = sample_weighter.get_stats()
                        wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                    if ema is not None and ema.cur_decay_value is not None:
                        wandb_log_dict["ema/decay"] = ema.cur_decay_value
                        wandb_log_dict["ema/step"] = ema.optimization_step
                    wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if is_eval_step:
            policy.eval()
            eval_loss_sum = 0.0
            n_eval_batches = 0
            with torch.no_grad(), accelerator.autocast():
                for eval_batch in eval_dataloader:
                    eval_batch = _preprocess_dataset_batch(
                        eval_batch, dataset.meta.camera_keys, cfg.rename_map, preprocessor
                    )
                    loss, _ = policy(eval_batch)  # 使用 __call__，这样 FSDP2 的 forward 钩子才会运行
                    eval_loss_sum += loss.item()
                    n_eval_batches += 1
            eval_loss = eval_loss_sum / max(n_eval_batches, 1)
            eval_loss = torch.tensor(eval_loss, device=device)
            eval_loss = accelerator.reduce(eval_loss, reduction="mean").item()
            policy.train()

            if is_main_process():
                logging.info(f"step {step}: eval_loss={eval_loss:.4f}")
                if wandb_logger:
                    wandb_logger.log_dict({"eval_loss": eval_loss}, step=step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            # 集合操作：每个 rank 都参与（gather / DCP 分片写入）；仅限 rank 0 的
            # 文件写入在 save_checkpoint 内部进行门控——调用处没有 rank 分支。
            if is_main_process():
                logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                accelerator=accelerator,
            )
            if is_main_process():
                if ema is not None:
                    # 保存影子副本以实现精确恢复，同时保存一份可直接加载的 EMA
                    # 权重副本（lerobot-eval --policy.path=<checkpoint>/pretrained_model_ema）。
                    torch.save(ema.state_dict(), checkpoint_dir / TRAINING_STATE_DIR / EMA_STATE_FILENAME)
                    unwrapped_policy = accelerator.unwrap_model(policy)
                    ema_dir = checkpoint_dir / f"{PRETRAINED_MODEL_DIR}_ema"
                    with _ema_weights(ema, unwrapped_policy):
                        unwrapped_policy.save_pretrained(ema_dir)
                        cfg.save_pretrained(ema_dir)
                        preprocessor.save_pretrained(ema_dir)
                        postprocessor.save_pretrained(ema_dir)
                update_last_checkpoint(checkpoint_dir)
                if cfg.save_checkpoint_to_hub:
                    push_checkpoint_to_hub(
                        checkpoint_dir,
                        cfg.policy.repo_id,
                        private=cfg.policy.private,
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
            accelerator.wait_for_everyone()

        if cfg.env and is_env_eval_step:
            if is_main_process():
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                eval_policy_model = accelerator.unwrap_model(policy)
                # 启用时评估 EMA 权重：换入操作只发生在主
                # 进程上（其他 rank 在下面的屏障处等待），之后会被
                # 精确撤销，因此实时权重在各 rank 间保持同步。
                use_ema_for_eval = ema is not None and cfg.ema.use_for_eval
                if use_ema_for_eval:
                    logging.info("Evaluating the EMA weights")
                weights_cm = _ema_weights(ema, eval_policy_model) if use_ema_for_eval else nullcontext()
                with weights_cm, _make_eval_envs(cfg) as eval_env, torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env（字典结构保持英文记号）
                        policy=eval_policy_model,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # 总体指标（与具体套件无关）
                aggregated = eval_info["overall"]

                # 可选：按套件记录日志
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meter/跟踪器
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    dp_world_size=parallel_dims.dp_world_size,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process():
        progbar.close()
        logging.info("End of training")

    # --- 发布（集合安全：所有 rank；模型提交时会 gather 分片权重）---------
    if getattr(active_cfg, "push_to_hub", False):
        unwrapped = accelerator.unwrap_model(policy)
        model_to_publish = unwrapped.get_base_model() if peft_model is not None else unwrapped
        publish_trained_model(
            cfg,
            model_to_publish,
            preprocessor,
            postprocessor,
            dataset.meta,
            peft_model=unwrapped if peft_model is not None else None,
        )

        # 上面的推送发送的是实时权重；当启用 EMA 时，被评估的权重
        # 是影子副本，因此也要把它们推送到同级的 `<repo_id>-ema` 仓库下。
        # 影子副本仅存在于主进程上，因此就构造而言这是仅限 rank 0 的操作。
        # 非致命：即使此处失败，实时模型也已经上传。
        if ema is not None:
            ema_repo_id = f"{active_cfg.repo_id}-ema"
            orig_repo_id = unwrapped.config.repo_id
            try:
                unwrapped.config.repo_id = ema_repo_id
                with _ema_weights(ema, unwrapped):
                    unwrapped.push_model_to_hub(cfg, dataset_meta=dataset.meta)
                preprocessor.push_to_hub(ema_repo_id)
                postprocessor.push_to_hub(ema_repo_id)
                logging.info("Pushed EMA weights to %s", ema_repo_id)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to push EMA weights to %s: %s", ema_repo_id, exc)
            finally:
                unwrapped.config.repo_id = orig_repo_id

    # 妥善清理分布式进程组
    accelerator.wait_for_everyone()
    accelerator.end_training()


def _remote_target_in_argv() -> bool:
    """在 draccus 解析之前，从原始 CLI 中检测远程 HF Jobs 运行请求。

    返回:
        bool：当 CLI 请求远程 HF Jobs 运行（`--job.target=<非本地目标>`）时为 True。
    """
    target = None
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--job.target" and i + 1 < len(args):
            target = args[i + 1]
        elif tok.startswith("--job.target="):
            target = tok.split("=", 1)[1]
    return JobConfig.is_remote_target(target)


def main():
    register_third_party_plugins()
    if _remote_target_in_argv():
        # 策略设备是在远程 pod 上解析的，而不是在这里，因此静音
        # PreTrainedConfig 在解析配置时发出的客户端
        # “Device '...' is not available” 警告（它会在 train() 远程派发之前触发）。
        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    train()


if __name__ == "__main__":
    main()
