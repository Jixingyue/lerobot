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
"""训练输出持久化：检查点、两阶段恢复和 hub 发布。

秩纪律：这里每个可能包含集合通信的函数
都有相应文档说明，并且必须在所有秩上运行；仅秩 0 的文件写入
位于每个连续区域的一个分组 ``is_main_process()`` 门控之下，放置在所有
集合通信之后。叶子保存/加载助手本身不携带秩门控——例外是
``PreTrainedPolicy._save_pretrained``，其门控是内部的，因为它的集合通信聚合
和写入位于同一方法中。
"""

import logging
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from huggingface_hub import HfApi, ModelCard, ModelCardData, snapshot_download
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.__version__ import __version__
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.distributed.checkpoint import (
    is_sharded_module,
    load_sharded_model,
    load_sharded_optimizer,
    save_sharded_model,
    save_sharded_optimizer,
)
from lerobot.distributed.utils import is_main_process
from lerobot.optim import (
    load_optimizer_state,
    load_scheduler_state,
    save_optimizer_state,
    save_scheduler_state,
)
from lerobot.policies import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.hub import find_latest_hub_checkpoint
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.random_utils import load_rng_state, save_rng_state

if TYPE_CHECKING:
    from accelerate import Accelerator

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.rewards.pretrained import PreTrainedRewardModel


def get_step_identifier(step: int, total_steps: int) -> str:
    """将步数格式化为用于检查点目录名称的零填充标识符。

    参数:
        step (int): 要格式化的训练步数。
        total_steps (int): 训练总步数；设置填充宽度
            （最少 6 位数字）。

    返回:
        str: 零填充的步数标识符，例如 `"005000"`。
    """
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """返回与步数对应的检查点子目录。

    参数:
        output_dir (Path): 训练运行的输出目录。
        total_steps (int): 训练总步数；设置标识符填充。
        step (int): 检查点的训练步数。

    返回:
        Path: 检查点步数目录，`output_dir/checkpoints/<step-identifier>`。
    """
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def should_save_checkpoint(step: int, save_freq: int, total_steps: int) -> bool:
    """是否应在 ``step`` 处保存检查点。

    每 ``save_freq`` 步保存一次检查点，并且在最后一步之后始终保存。
    非正的 ``save_freq`` 会禁用周期性保存（只写入最终检查点），
    这与 ``log_freq``/``eval_freq`` 处理非正值的方式一致，
    并避免了 ``step % 0`` 引发的 ``ZeroDivisionError``。
    """
    return (save_freq > 0 and step % save_freq == 0) or step == total_steps


def update_last_checkpoint(checkpoint_dir: Path) -> None:
    """将检查点目录中的 `last` 符号链接指向给定的检查点。

    任何已存在的 `last` 符号链接都会被替换。链接目标是相对于检查点
    目录的，因此当运行目录被移动时目录树仍然有效。

    参数:
        checkpoint_dir (Path): `last` 链接应指向的检查点步数目录。
    """
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


# ---------------------------------------------------------------------------------------------
# training_step.json
# ---------------------------------------------------------------------------------------------


def save_training_metadata(step: int, save_dir: Path, cfg: TrainPipelineConfig) -> None:
    """记录步数计数器以及恢复时推断拓扑变化所需的一切。

    `step` 统计循环迭代次数（= 微批次），因此
    采样器恢复偏移量为 `step x batch_size x dp_world_size`，不含梯度累积因子。
    记录 `grad_accum_steps` 和并行度快照，以便在优化器更新节奏
    或分片拓扑发生变化时恢复能精确警告。

    参数:
        step (int): 要记录的训练步数（微批次计数器）。
        save_dir (Path): 写入 `training_step.json` 的 `training_state/` 目录。
        cfg (TrainPipelineConfig): 训练配置，其批量大小、梯度累积
            和并行度设置与步数一起被快照。
    """
    state: dict[str, Any] = {
        "step": step,
        "dp_world_size": cfg.parallelism.dp_world_size,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.accelerator.gradient_accumulation.steps,
        "parallelism": {
            "dp_replicate": cfg.parallelism.dp_replicate,
            "dp_shard": cfg.parallelism.dp_shard,
            "ring_degree": cfg.parallelism.context_parallel.ring_degree,
            "ulysses_degree": cfg.parallelism.context_parallel.ulysses_degree,
        },
    }
    write_json(state, save_dir / TRAINING_STEP)


def load_training_metadata(training_state_dir: Path) -> dict[str, Any]:
    """一次性读取 `save_training_metadata` 记录的所有内容。

    每个键始终存在：检查点之前不存在的字段返回 None，因此调用方
    读取 `metadata["batch_size"]` 时，拼写错误会得到 KeyError 而不是静默的 None。

    参数:
        training_state_dir (Path): 检查点的 `training_state/` 目录。

    返回:
        dict[str, Any]: `step` 加上与其一起记录的 `dp_world_size`、`batch_size`、
            `grad_accum_steps` 和 `parallelism` 快照（未记录处为 None）。
    """
    state = load_json(training_state_dir / TRAINING_STEP)
    return {
        "step": int(state["step"]),
        "dp_world_size": state.get("dp_world_size", state.get("num_processes")),
        "batch_size": state.get("batch_size"),
        "grad_accum_steps": state.get("grad_accum_steps"),
        "parallelism": state.get("parallelism"),
    }


# ---------------------------------------------------------------------------------------------
# 检查点保存
# ---------------------------------------------------------------------------------------------


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
    accelerator: "Accelerator | None" = None,
) -> None:
    """此函数创建以下目录结构：

    005000/  #  检查点处的训练步数
    ├── pretrained_model/
    │   ├── config.json  # 策略配置
    │   ├── model.safetensors  # 策略权重（checkpoint_format ∈ {safetensors, safetensors_dcp}，或任何非分片运行）
    │   ├── pytorch_model_fsdp_0/  # DCP 模型分片（checkpoint_format ∈ {dcp, safetensors_dcp}）
    │   ├── train_config.json  # 训练配置
    │   ├── policy_preprocessor.json  # 预处理器配置（如果提供了预处理器）
    │   ├── policy_preprocessor_step_*.safetensors  # 有状态预处理器步骤的状态
    │   ├── policy_postprocessor.json  # 后处理器配置（如果提供了后处理器）
    │   └── policy_postprocessor_step_*.safetensors  # 有状态后处理器步骤的状态
    └── training_state/
        ├── optimizer_param_groups.json  # 优化器参数组（非分片运行）
        ├── optimizer_state.safetensors  # 优化器状态（非分片运行）
        ├── optimizer_0/  # DCP 优化器分片（分片运行）
        ├── rng_state.safetensors  # 随机数状态
        ├── scheduler_state.json  # 调度器状态（如果提供了调度器）
        └── training_step.json  # 训练步数 + dp_world_size/batch_size/grad_accum + 拓扑

    集合通信：必须在每个秩上调用。仅秩 0 的写入在内部门控，因此
    调用点不需要秩分支。

    参数:
        checkpoint_dir (Path): 要写入的检查点步数目录（例如 `.../checkpoints/005000`）。
        step (int): 该检查点处的训练步数。
        cfg (TrainPipelineConfig): 本次运行使用的训练配置。
        policy (PreTrainedPolicy): 要保存的策略。
        optimizer (Optimizer): 要保存其状态的优化器。
        scheduler (LRScheduler | None, optional): 要保存其状态的调度器。默认为 None。
        preprocessor (PolicyProcessorPipeline | None, optional): 要保存的预处理器/流水线。
            默认为 None。
        postprocessor (PolicyProcessorPipeline | None, optional): 要保存的后处理器/流水线。
            默认为 None。
        accelerator (Accelerator | None, optional): 策略准备时使用的加速器；
            用于解包模型，在分片运行中是必需的，它拥有 DCP 保存
            通道。默认为 None（普通单进程保存）。
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    fmt = cfg.checkpoint_format
    policy_to_save = accelerator.unwrap_model(policy) if accelerator is not None else policy
    sharded = is_sharded_module(policy_to_save)

    # -- 模型工件：两个支持集合通信的调用 ----------------------------------
    if cfg.peft is not None:
        # PeftModel.save_pretrained 是没有内部门控的外部 API，
        # 适配器是复制的（PEFT x 分片在验证时被拒绝）：主秩写入。
        if is_main_process():
            policy_to_save.save_pretrained(pretrained_dir)
    elif fmt.wants_safetensors or not sharded:
        # 分片时是集合通信（完整聚合）；在所有多秩布局中写入仅发生在主进程
        # （门控位于 _save_pretrained 内部，紧邻其集合通信聚合）。
        policy_to_save.save_pretrained(pretrained_dir)
    if fmt.wants_dcp and sharded:
        save_sharded_model(accelerator, policy_to_save, pretrained_dir)

    # -- 侧车配置：整个连续的仅秩 0 区域的单一门控 ----------------
    if is_main_process():
        if fmt.wants_dcp and not fmt.wants_safetensors:
            # save_pretrained 未运行：保持仅 DCP 检查点的自描述性。
            policy_to_save.config.save_pretrained(pretrained_dir)
        cfg.save_pretrained(pretrained_dir)
        if cfg.peft is not None:
            # PEFT 的 save_pretrained 只写入适配器权重 + 配置；重新加载
            # 基础模型所需的策略配置被显式写入。
            policy_to_save.config.save_pretrained(pretrained_dir)
        if preprocessor is not None:
            preprocessor.save_pretrained(pretrained_dir)
        if postprocessor is not None:
            postprocessor.save_pretrained(pretrained_dir)

    save_training_state(
        checkpoint_dir, step, cfg, optimizer, scheduler, accelerator, sharded=sharded, model=policy_to_save
    )
    if accelerator is not None:
        accelerator.wait_for_everyone()


def save_training_state(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    optimizer: Optimizer | dict[str, Optimizer] | None = None,
    scheduler: LRScheduler | None = None,
    accelerator: "Accelerator | None" = None,
    *,
    sharded: bool = False,
    model: PreTrainedPolicy | None = None,
) -> None:
    """写入 training_state/。分片下是集合通信：在每个秩上调用。

    参数:
        checkpoint_dir (Path): 检查点步数目录；`training_state/` 在其中创建。
        step (int): 该检查点处的训练步数。
        cfg (TrainPipelineConfig): 本次运行使用的训练配置（其拓扑和
            累积设置记录在 `training_step.json` 中）。
        optimizer (Optimizer | dict[str, Optimizer] | None, optional): 要保存
            其状态的优化器。默认为 None。
        scheduler (LRScheduler | None, optional): 要保存其状态的调度器。
            默认为 None。
        accelerator (Accelerator | None, optional): 当 `sharded` 为 True 时必需——
            它拥有 DCP 优化器保存通道。默认为 None。
        sharded (bool): 模型的分片状态，在 `save_checkpoint` 中计算一次并
            传递到这里，使两处不会不一致。默认为 False。
        model (PreTrainedPolicy | None, optional): 仅分片优化器通道需要：
            torch 的优化器 DCP API 与模型耦合（状态字典以模型 FQN 为键），
            因此 accelerate 的 `save_fsdp_optimizer` 需要分片模块
            与优化器一起。默认为 None。
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    # 所有秩：在 DCP 优化器集合通信写入之前目录必须存在
    # （exist_ok 使并发 mkdir 在共享文件系统上无竞争）。
    save_dir.mkdir(parents=True, exist_ok=True)

    if optimizer is not None and sharded:
        if accelerator is None or model is None:
            raise ValueError("Saving a sharded optimizer state requires the accelerator and model.")
        # 集合通信——所有秩将其 DCP 分片写入 optimizer_0/。
        save_sharded_optimizer(accelerator, optimizer, model, save_dir)

    if is_main_process():  # 整个仅秩 0 区域的单一分组门控
        save_training_metadata(step, save_dir, cfg)
        save_rng_state(save_dir)
        if scheduler is not None:
            save_scheduler_state(scheduler, save_dir)
        if optimizer is not None and not sharded:
            save_optimizer_state(optimizer, save_dir)


# ---------------------------------------------------------------------------------------------
# 两阶段恢复
# ---------------------------------------------------------------------------------------------


def resume_before_prepare(cfg: TrainPipelineConfig) -> int:
    """阶段 1——在 `accelerator.prepare()` 之前：恢复 RNG 并返回步数计数器。

    仅纯加载器。采样器恢复偏移量在数据加载器工厂内部从返回的步数
    *推导*得出，绑定到分片对象的一切（模型 DCP 分片、优化器、
    调度器）在 `resume_after_prepare` 中加载。

    参数:
        cfg (TrainPipelineConfig): 恢复的训练配置；`cfg.checkpoint_path` 定位
            要从中恢复的检查点。

    返回:
        int: 检查点中记录的训练步数（微批次计数器）。

    引发:
        NotADirectoryError: 如果检查点没有 `training_state/` 目录。
        ValueError: 如果恢复的拓扑相对于检查点中记录的拓扑
            跨越了分片/非分片边界。
    """
    training_state_dir = cfg.checkpoint_path / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)
    metadata = load_training_metadata(training_state_dir)
    _guard_resume_changes(cfg, metadata)
    load_rng_state(training_state_dir)
    return metadata["step"]


def _guard_resume_changes(cfg: TrainPipelineConfig, metadata: dict[str, Any]) -> None:
    """将恢复运行的设置与检查点中记录的设置进行对照检查。

    两个层级，均由检查点记录的并行度快照驱动：

    - **硬错误**：当恢复在任一方向上跨越分片/非分片边界时：
      检查点的训练状态工件仅支持在同类拓扑上恢复
      （重新分片可跨大小工作，但不能跨类别）。没有记录快照的
      检查点跳过此检查。
    - **一条警告**：列出所有其他有差异的已记录设置——这些更改是
      合法的（DCP 会跨拓扑重新分片权重和优化器状态，采样器偏移量
      会自适应），但更改 ``grad_accum_steps`` 会改变优化器更新节奏，因此
      恢复时会精确说明差异所在。采样器精确性警告
      （``dp_world_size``/``batch_size``）与数据加载器工厂中的采样器计算放在一起。

    参数:
        cfg (TrainPipelineConfig): 恢复的训练配置，与检查点中记录的
            设置进行比较。
        metadata (dict[str, Any]): 检查点记录的训练元数据，由
            `load_training_metadata` 返回。

    引发:
        ValueError: 如果检查点记录的是分片拓扑而恢复运行是
            非分片的，或反之。
    """
    snapshot = metadata["parallelism"]

    if snapshot is not None:
        recorded_sharded = (
            snapshot.get("dp_shard", 1) != 1
            or snapshot.get("ring_degree", 1) * snapshot.get("ulysses_degree", 1) > 1
        )
        if recorded_sharded != cfg.parallelism.is_sharded:
            raise ValueError(
                f"Cannot resume: the checkpoint was written with a "
                f"{'sharded' if recorded_sharded else 'non-sharded'} topology "
                f"(dp_replicate={snapshot.get('dp_replicate')}, dp_shard={snapshot.get('dp_shard')}) "
                f"but this run is {'sharded' if cfg.parallelism.is_sharded else 'non-sharded'} "
                f"(dp_replicate={cfg.parallelism.dp_replicate}, dp_shard={cfg.parallelism.dp_shard})."
            )

    recorded = {
        "grad_accum_steps": (
            metadata["grad_accum_steps"],
            cfg.accelerator.gradient_accumulation.steps,
        ),
    }
    if snapshot is not None:
        recorded.update(
            {
                "dp_replicate": (snapshot.get("dp_replicate"), cfg.parallelism.dp_replicate),
                "dp_shard": (snapshot.get("dp_shard"), cfg.parallelism.dp_shard),
                "ring_degree": (
                    snapshot.get("ring_degree"),
                    cfg.parallelism.context_parallel.ring_degree,
                ),
                "ulysses_degree": (
                    snapshot.get("ulysses_degree"),
                    cfg.parallelism.context_parallel.ulysses_degree,
                ),
            }
        )
    changed = [f"{key}: {was} -> {now}" for key, (was, now) in recorded.items() if was not in (None, now)]
    if changed and is_main_process():
        logging.warning(
            "Resuming with settings that differ from the checkpoint: " + "; ".join(changed) + ". "
            "Topology changes reshard safely via DCP; a changed grad_accum_steps shifts the "
            "optimizer-update cadence (the step counter keeps counting micro-batches)."
        )


def resume_after_prepare(
    cfg: TrainPipelineConfig,
    accelerator: "Accelerator",
    policy: PreTrainedPolicy,
    optimizer: Optimizer | dict[str, Optimizer],
    scheduler: LRScheduler | None,
) -> None:
    """阶段 2——在 `accelerator.prepare()` 之后：模型 (DCP) -> 优化器 -> 调度器。

    分片下是集合通信：在每个秩上调用。模型权重来源遵循
    检查点自身记录的 `checkpoint_format`（恢复时，`cfg` 是从检查点的
    train_config.json 解析的）：含 DCP 的格式在此处将分片加载到已准备的
    模型中（其构造跳过了 safetensors 加载）；safetensors 格式已在分片之前
    由 `from_pretrained` 加载——此处没有模型步骤。

    参数:
        cfg (TrainPipelineConfig): 恢复的训练配置；`cfg.checkpoint_path` 定位
            检查点，`cfg.checkpoint_format` 选择模型权重来源。
        accelerator (Accelerator): 策略准备时使用的加速器；它解包
            模型并拥有 DCP 加载通道。
        policy (PreTrainedPolicy): 要加载权重的已准备（可能已分片）策略。
        optimizer (Optimizer | dict[str, Optimizer]): 要恢复的已准备优化器。
        scheduler (LRScheduler | None): 要恢复的调度器，如果运行没有则为 None。

    引发:
        FileNotFoundError: 如果检查点格式声明了 DCP 模型分片但分片
            目录缺失（例如在上传前被修剪）。
    """
    checkpoint_dir = cfg.checkpoint_path
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    unwrapped = accelerator.unwrap_model(policy)
    sharded = is_sharded_module(unwrapped)

    if cfg.checkpoint_format.wants_dcp:
        from accelerate.utils.constants import FSDP_MODEL_NAME

        dcp_dir = pretrained_dir / f"{FSDP_MODEL_NAME}_0"
        if not dcp_dir.is_dir():
            raise FileNotFoundError(
                f"checkpoint_format={cfg.checkpoint_format.value} declares DCP model shards, "
                f"but {dcp_dir} is missing. If the shards were pruned, convert what remains "
                "with `lerobot-convert-dcp` or resume from a safetensors checkpoint."
            )
        load_sharded_model(accelerator, unwrapped, pretrained_dir)

    if sharded:
        # 需要已准备的优化器：FSDP2 的 prepare 将参数组重新绑定到 DTensor，
        # 但从不迁移 optimizer.state——DCP 在此处重新分片（可跨拓扑变更工作）。
        load_sharded_optimizer(accelerator, optimizer, unwrapped, training_state_dir)
    else:
        load_optimizer_state(optimizer, training_state_dir)

    if scheduler is not None:
        load_scheduler_state(scheduler, training_state_dir)


# ---------------------------------------------------------------------------------------------
# Hub：检查点推送（恢复工件）和发布（分发工件）
# ---------------------------------------------------------------------------------------------


def push_checkpoint_to_hub(
    checkpoint_dir: Path,
    repo_id: str,
    *,
    private: bool | None = None,
) -> None:
    """将已保存的检查点目录上传到 Hub 的 checkpoints/<name>/ 下。

    启用 save_checkpoint_to_hub 时，每个保存步骤调用一次，因此
    超时或崩溃的运行仍会在 Hub 上留下可恢复的检查点。
    模型仓库以幂等方式创建，提交会以检查点步数打标签，
    这样检查点可以用 --policy.pretrained_revision=<step>
    而不是 commit sha 来恢复。

    目录按原样上传——包括 DCP 格式下的 DCP 分片：此目录树
    是为*恢复*而非分发而存在的，`resolve_resume_checkpoint` 会
    对称地将其下载回来。

    参数:
        checkpoint_dir (Path): 要上传的本地检查点步数目录。
        repo_id (str): 要推送的 Hub 模型仓库（缺失时幂等创建）。
        private (bool | None): 新创建的仓库是否应为私有。默认为
            None（除非组织默认为私有，否则为公开）。
    """
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=f"checkpoints/{checkpoint_dir.name}",
        commit_message=f"checkpoint {checkpoint_dir.name}",
    )
    api.create_tag(
        repo_id=repo_id,
        tag=checkpoint_dir.name,
        revision=commit.oid,
        repo_type="model",
        exist_ok=True,
    )


def resolve_resume_checkpoint(repo_id: str, output_dir: Path) -> Path:
    """将 Hub 训练仓库的最新检查点下载到本地运行目录。

    `push_checkpoint_to_hub` 的对称操作：给定一个保存有
    `checkpoints/<step>/{pretrained_model,training_state}` 子树的模型仓库，
    将编号最大的步数下载到 `output_dir/checkpoints/<step>/`，重建本地
    `last` 符号链接，并返回该本地检查点目录。用于在没有原始本地
    运行目录的机器（或 HF Jobs pod）上从 Hub 恢复训练。

    参数:
        repo_id (str): 保存有 `checkpoints/<step>/` 子树的 Hub 模型仓库。
        output_dir (Path): 下载检查点的本地运行目录。

    返回:
        Path: 本地检查点步数目录，`output_dir/checkpoints/<step>`。

    引发:
        FileNotFoundError: 如果仓库在 `checkpoints/` 下没有检查点。
    """
    latest = find_latest_hub_checkpoint(repo_id)
    if latest is None:
        raise FileNotFoundError(
            f"No checkpoint found in '{repo_id}' under '{CHECKPOINTS_DIR}/'. "
            "Was the run trained with --save_checkpoint_to_hub?"
        )
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=f"{latest}/*",
        local_dir=str(output_dir),
    )
    checkpoint_dir = output_dir / latest
    update_last_checkpoint(checkpoint_dir)
    return checkpoint_dir


def publish_trained_model(
    cfg: TrainPipelineConfig,
    model: "PreTrainedPolicy | PreTrainedRewardModel",
    preprocessor: PolicyProcessorPipeline | None,
    postprocessor: PolicyProcessorPipeline | None,
    dataset_meta: "LeRobotDatasetMetadata | None",
    *,
    peft_model: Any | None = None,
) -> None:
    """将完整的训练捆绑包作为可分发的模型仓库发布。

    集合通信安全：在所有秩上调用——模型提交通过 `save_pretrained`
    聚合分片权重；上传仅在主进程上进行（在
    `HubMixin.push_to_hub` 内部和此处门控）。提交按顺序为：(1) 模型
    （PEFT 时跳过——适配器替代完整权重），(2) 预处理器，(3) 后处理器，
    (4) 捆绑侧车：README.md 模型卡片 + train_config.json（PEFT 情况下
    还有适配器权重和被包装策略的配置）。每次提交都上传一个新组装的
    目录，因此发布的仓库只携带可分发的工件。

    参数:
        cfg (TrainPipelineConfig): 训练配置；保存为 `train_config.json` 并用于
            渲染模型卡片。
        model (PreTrainedPolicy | PreTrainedRewardModel): 要发布的已训练模型；
            其配置提供目标仓库 id、可见性、许可证和标签。
        preprocessor (PolicyProcessorPipeline | None): 要与模型一起发布的
            预处理器流水线（如果有）。
        postprocessor (PolicyProcessorPipeline | None): 要与模型一起发布的
            后处理器流水线（如果有）。
        dataset_meta (LeRobotDatasetMetadata | None): 用于模型卡片的数据集
            元数据（如果可用）。
        peft_model (Any | None): 训练适配器时的 PEFT 包装器；其适配器权重
            在发布的仓库中替代完整模型权重。默认为 None。

    引发:
        ValueError: 如果模型配置没有仓库 id（`--policy.repo_id`）。
    """
    model_cfg = model.config
    repo_id = model_cfg.repo_id
    if not repo_id:
        raise ValueError("Publishing requires a repo id (--policy.repo_id).")
    ignore = ["*.tmp", "*.log"]

    if peft_model is None:
        # 调用是在拥有每个方法的确切对象上进行的（从不通过 PEFT 的
        # 属性转发），因此下面的 peft 分支永远不会触及此路径。
        model.push_to_hub(repo_id, private=model_cfg.private, ignore_patterns=ignore)
    if preprocessor is not None:
        preprocessor.push_to_hub(repo_id, private=model_cfg.private)
    if postprocessor is not None:
        postprocessor.push_to_hub(repo_id, private=model_cfg.private)

    if is_main_process():
        api = HfApi()
        repo_id = api.create_repo(repo_id=repo_id, private=model_cfg.private, exist_ok=True).repo_id
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id
            saved_path.mkdir(parents=True, exist_ok=True)
            if peft_model is not None:
                peft_model.save_pretrained(saved_path)  # 适配器权重 + 适配器配置
                model.config.save_pretrained(saved_path)  # PEFT 无法写入策略配置
            card = generate_model_card(model_cfg, cfg=cfg, dataset_meta=dataset_meta)
            card.save(str(saved_path / "README.md"))
            cfg.save_pretrained(saved_path)  # train_config.json
            commit_info = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message="Upload model card and train config",
                allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
                ignore_patterns=ignore,
            )
        # 约定：lerobot.jobs.hf.submit_to_hf 监视这条确切的 "Model pushed to <url>"
        # 日志行，以便提前结束远程运行。保持措辞和 URL 格式同步。
        logging.info(f"Model pushed to {commit_info.repo_url.url}")

    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------------------------
# 模型卡片
# ---------------------------------------------------------------------------------------------

_BASE_MODEL_MAPPING = {
    "smolvla": "lerobot/smolvla_base",
    "pi0": "lerobot/pi0_base",
    "pi05": "lerobot/pi05_base",
    "pi0_fast": "lerobot/pi0fast-base",
    "xvla": "lerobot/xvla-base",
}


def build_card_context(
    cfg: TrainPipelineConfig | None,
    dataset_meta: "LeRobotDatasetMetadata | None",
    input_features: dict | None,
    output_features: dict | None,
) -> dict:
    """为模型卡片模板收集可选数据。

    仅返回纯值（不含 Markdown）——
    ``lerobot/templates/lerobot_modelcard_template.md`` 中的模板决定如何以及是否
    显示每一项。一切都是尽力而为：任何不可用的内容都留空/None，
    模板只是跳过该部分，因此这永远不会破坏 Hub 推送。

    参数:
        cfg (TrainPipelineConfig | None): 提供训练部分的训练配置
            （如果可用）。
        dataset_meta (LeRobotDatasetMetadata | None): 提供数据集、
            机器人类型和摄像头部分的数据集元数据（如果可用）。
        input_features (dict | None): 策略的输入特征声明（如果有）。
        output_features (dict | None): 策略的输出特征声明（如果有）。

    返回:
        dict: 包含 `training`、`input_features`、`output_features`、
            `dataset`、`robot_type` 和 `cameras` 条目的模板上下文；
            不可用的部分保持为空/None。
    """
    context = {
        "training": None,
        "input_features": input_features or {},
        "output_features": output_features or {},
        "dataset": None,
        "robot_type": None,
        "cameras": [],
    }

    if cfg is not None:
        optimizer = getattr(cfg, "optimizer", None)
        context["training"] = {
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
            "optimizer": getattr(optimizer, "type", None) if optimizer else None,
            "lr": getattr(optimizer, "lr", None) if optimizer else None,
            "lerobot_version": __version__,
        }

    if dataset_meta is not None:
        context["dataset"] = {
            "repo_id": dataset_meta.repo_id,
            "episodes": dataset_meta.total_episodes,
            "frames": dataset_meta.total_frames,
            "fps": dataset_meta.fps,
            "tasks": [str(task) for task in dataset_meta.tasks.index],
        }
        context["robot_type"] = dataset_meta.robot_type
        context["cameras"] = [key.split(".")[-1] for key in dataset_meta.camera_keys]

    return context


def generate_model_card(
    model_cfg: PreTrainedConfig | RewardModelConfig,
    cfg: TrainPipelineConfig | None = None,
    dataset_meta: "LeRobotDatasetMetadata | None" = None,
) -> ModelCard:
    """为已训练的策略或奖励模型渲染 LeRobot 模型卡片。

    故意做成自由函数：每个模板变量都来自参数——模型配置、
    训练配置和数据集元数据——没有一个来自活动模型，因此卡片
    也可以仅从检查点的 `config.json` 渲染（参见 `lerobot-convert-dcp`）。
    配置类型选择模板：奖励模型获得奖励模型卡片，策略获得
    带有训练/数据集部分的策略卡片。

    参数:
        model_cfg (PreTrainedConfig | RewardModelConfig): 提供类型、
            许可证、标签、仓库 id 以及——对策略而言——特征声明的模型配置。
        cfg (TrainPipelineConfig | None, optional): 用于训练和数据集
            卡片部分的训练配置。默认为 None。
        dataset_meta (LeRobotDatasetMetadata | None, optional): 用于
            数据集卡片部分的数据集元数据。默认为 None。

    返回:
        ModelCard: 已渲染并验证的 LeRobot 模型卡片。
    """
    model_type = model_cfg.type
    base_model = _BASE_MODEL_MAPPING.get(model_type)

    if isinstance(model_cfg, RewardModelConfig):
        tags = {"robotics", "lerobot", "reward-model", model_type}
        template_card = (
            files("lerobot.templates")
            .joinpath("lerobot_rewardmodel_modelcard_template.md")
            .read_text("utf-8")
        )
        context: dict[str, Any] = {}  # 奖励模板仅从 card_data 渲染
    else:
        tags = {"robotics", "lerobot", model_type}
        template_card = (
            files("lerobot.templates").joinpath("lerobot_modelcard_template.md").read_text("utf-8")
        )
        context = build_card_context(cfg, dataset_meta, model_cfg.input_features, model_cfg.output_features)
        # 模板用它来预填充命令和 "Fine-tuned from" 行。
        context["policy_repo_id"] = model_cfg.repo_id
        context["base_model"] = base_model

    card_data = ModelCardData(
        license=model_cfg.license or "apache-2.0",
        library_name="lerobot",
        pipeline_tag="robotics",
        tags=list(tags.union(model_cfg.tags or [])),
        model_name=model_type,
        datasets=cfg.dataset.repo_id if cfg is not None else None,
        base_model=base_model,
    )
    card = ModelCard.from_template(card_data, template_str=template_card, **context)
    card.validate()
    return card
