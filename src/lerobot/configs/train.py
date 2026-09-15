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
import builtins
import datetime as dt
import json
import multiprocessing
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.configs.accelerator import AcceleratorConfig, ActivationCheckpointingMode
from lerobot.configs.parallelism import ParallelismConfig
from lerobot.optim import LRSchedulerConfig, OptimizerConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.hub import HubMixin, find_latest_hub_checkpoint
from lerobot.utils.sample_weighting import SampleWeightingConfig

from . import parser
from .default import DatasetConfig, EMAConfig, EvalConfig, JobConfig, PeftConfig, WandBConfig
from .policies import PreTrainedConfig
from .rewards import RewardModelConfig

TRAIN_CONFIG_NAME = "train_config.json"


class CheckpointFormat(str, Enum):
    """训练检查点内部的模型产物格式。

    仅选择*模型*产物；training_state 的布局与格式无关（在分片运行下
    优化器通道始终为 DCP，否则为 safetensors+json）。

    - SAFETENSORS（默认）：完整的 `model.safetensors` —— 兼容性最高，
      分片情况下每次保存需做一次聚合。
    - DCP：仅分片的 `pytorch_model_fsdp_0/*.distcp` —— 保存/恢复最快；
      分发前请用 `lerobot-convert-dcp` 转换。
    - SAFETENSORS_AND_DCP：两种产物独立写入。
    """

    SAFETENSORS = "safetensors"
    DCP = "dcp"
    SAFETENSORS_AND_DCP = "safetensors_dcp"

    @property
    def wants_safetensors(self) -> bool:
        """当应写入完整的 `model.safetensors` 产物时为 True。"""
        return self in (CheckpointFormat.SAFETENSORS, CheckpointFormat.SAFETENSORS_AND_DCP)

    @property
    def wants_dcp(self) -> bool:
        """当应写入分片的 DCP 模型分片（`pytorch_model_fsdp_0/`）时为 True。"""
        return self in (CheckpointFormat.DCP, CheckpointFormat.SAFETENSORS_AND_DCP)


def _migrate_legacy_rabc_fields(config: dict[str, Any]) -> dict[str, Any] | None:
    """返回旧版 RA-BC 字段迁移后的配置内容；无需迁移时返回 None。"""
    legacy_fields = (
        "use_rabc",
        "rabc_progress_path",
        "rabc_kappa",
        "rabc_epsilon",
        "rabc_head_mode",
    )
    if not any(key in config for key in legacy_fields):
        return None

    migrated_config = dict(config)
    use_rabc = bool(migrated_config.pop("use_rabc", False))
    rabc_progress_path = migrated_config.pop("rabc_progress_path", None)
    rabc_kappa = migrated_config.pop("rabc_kappa", None)
    rabc_epsilon = migrated_config.pop("rabc_epsilon", None)
    rabc_head_mode = migrated_config.pop("rabc_head_mode", None)

    # 新配置可能已经显式定义了 sample_weighting。在这种情况下，
    # 旧版字段在从配置内容中剥离后将被忽略。
    if migrated_config.get("sample_weighting") is None and use_rabc:
        sample_weighting: dict[str, Any] = {"type": "rabc"}
        if rabc_progress_path is not None:
            sample_weighting["progress_path"] = rabc_progress_path
        if rabc_kappa is not None:
            sample_weighting["kappa"] = rabc_kappa
        if rabc_epsilon is not None:
            sample_weighting["epsilon"] = rabc_epsilon
        if rabc_head_mode is not None:
            sample_weighting["head_mode"] = rabc_head_mode
        migrated_config["sample_weighting"] = sample_weighting

    return migrated_config


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    reward_model: RewardModelConfig | None = None
    # 将 `dir` 设置为你希望保存所有运行输出的位置。如果你用相同的 `dir` 值
    # 运行另一个训练会话，其内容将被覆盖，除非你将 `resume` 设置为 true。
    output_dir: Path | None = None
    job_name: str | None = None
    # 将 `resume` 设置为 true 以恢复之前的运行。传入指向本地检查点的
    # train_config.json 或保存有 `checkpoints/<step>/` 子树的 Hub 仓库 id 的
    # `--config_path`（会下载最新的检查点并从中恢复）。注意，恢复时的默认行为
    # 是使用检查点中的配置，而不管恢复时训练命令提供了什么配置
    # （CLI `--*` 标志仍会覆盖）。
    resume: bool = False
    # `seed` 用于训练（例如：模型初始化、数据集打乱），
    # 也用于评估环境。
    seed: int | None = 1000
    # 设置为 True 以使用确定性的 cuDNN 算法来保证可复现性。
    # 这会禁用 cudnn.benchmark，并可能使训练速度降低约 10-20%。
    cudnn_deterministic: bool = False
    # 数据加载器的 worker 数量。
    num_workers: int = 4
    batch_size: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    # DataLoader worker 的启动方式。在使用非 fork 安全库（PyAV / torchcodec /
    # ffmpeg）时，"spawn" 比 "fork" 更安全，但由于 worker 需要重新导入模块
    # 而不是继承父进程状态，因此每次运行都会增加一些 worker 启动时间。
    # 在合适的情况下可以用 `--dataloader_multiprocessing_context=fork` 覆盖，
    # 或将其设置为 `null` 以使用 Python 的平台默认值。
    dataloader_multiprocessing_context: str | None = "spawn"
    steps: int = 100_000
    # 每 N 步在仿真环境中运行策略以测量奖励/成功率（0 = 禁用）。
    env_eval_freq: int = 20_000
    log_freq: int = 200
    # 每 N 步在留出的剧集上计算评估损失（0 = 禁用）。需要 eval_split > 0。
    eval_steps: int = 0
    # 评估样本总数上限，在各任务间均匀分配（0 = 使用全部留出数据）。
    max_eval_samples: int = 0
    tolerance_s: float = 1e-4
    save_checkpoint: bool = True
    # 检查点每 `save_freq` 次训练迭代保存一次，并在最后一个训练步骤后保存。
    # 非正值会禁用周期性保存，只保留最终检查点。
    save_freq: int = 20_000
    # 检查点内部的模型产物格式；非默认值需要分片运行。
    checkpoint_format: CheckpointFormat = CheckpointFormat.SAFETENSORS
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    # 进程拓扑：dp_replicate / dp_shard（HSDP）以及上下文并行度的占位符。
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    # 交给 Accelerator 的执行时配置：混合精度、梯度累积、
    # FSDP/DDP 调优参数、compile 和激活检查点的占位符。
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # 训练期间维护策略权重的 EMA 影子（参见 EMAConfig）。
    ema: EMAConfig = field(default_factory=EMAConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig | None = None

    # 训练运行的位置（本地默认，或某种 HF Jobs 规格）。参见 JobConfig。
    job: JobConfig = field(default_factory=JobConfig)
    # 在写入时将每个保存的检查点推送到 Hub（policy.repo_id），而不仅仅是
    # 最终模型（便于在运行中途监控进度）。可选；无论如何最终模型都会被推送。
    # 在本地和远程的行为相同。
    save_checkpoint_to_hub: bool = False

    # 样本加权配置（例如用于 RA-BC 训练）
    sample_weighting: SampleWeightingConfig | None = None

    # 观测值的重命名映射，用于覆盖 image 和 state 的键名
    rename_map: dict[str, str] = field(default_factory=dict)
    checkpoint_path: Path | None = field(init=False, default=None)

    @property
    def is_reward_model_training(self) -> bool:
        """当配置针对的是奖励模型而非策略时为 True。"""
        return self.reward_model is not None

    @property
    def trainable_config(self) -> PreTrainedConfig | RewardModelConfig:
        """返回当前生效的配置（policy 或 reward_model）。"""
        if self.is_reward_model_training:
            return self.reward_model  # type: ignore[return-value]
        return self.policy  # type: ignore[return-value]

    def _resolve_pretrained_from_cli(self) -> None:
        """将 CLI 上传入的预训练来源解析为已加载的配置。

        预训练路径（`--policy.path`、`--reward_model.path`）和
        `--config_path` 只能通过重新读取 CLI 参数来恢复：在 `validate()` 运行时
        draccus 已经消费了它们，因此它们不会反映在 `self` 上。
        恰好只有一个来源生效，按优先级顺序为：
        奖励模型路径、策略路径，然后是 resume。
        """
        reward_model_path = parser.get_path_arg("reward_model")
        policy_path = parser.get_path_arg("policy")

        if reward_model_path:
            cli_overrides = parser.get_cli_overrides("reward_model")
            self.reward_model = RewardModelConfig.from_pretrained(
                reward_model_path, cli_overrides=cli_overrides
            )
            self.reward_model.pretrained_path = str(Path(reward_model_path))
        elif policy_path:
            overrides = parser.get_yaml_overrides("policy") + (parser.get_cli_overrides("policy") or [])
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=overrides)
            self.policy.pretrained_path = Path(policy_path)
        elif self.resume:
            self._resolve_resume_checkpoint()

    def _resolve_resume_checkpoint(self) -> None:
        """将可训练配置指向 `--config_path` 指定的检查点。

        `config_path` 可以是本地路径（指向检查点的 train_config.json 或其
        pretrained_model/ 目录），也可以是 Hub 仓库 id。对于 Hub 仓库，最新的检查点
        会被下载到一个全新的本地运行目录并从那里恢复。当分派到 HF Job
        （`job.is_remote`）时跳过下载：pod 在本地执行恢复时会自行下载，
        而 `submit_to_hf` 会为远程命令解析源仓库。
        """
        config_path = parser.parse_arg("config_path")
        if not config_path:
            raise ValueError(
                f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
            )

        if Path(config_path).resolve().exists():
            # `config_path` 可能指向检查点的 train_config.json 或其
            # pretrained_model/ 目录（两者都在上文有说明）—— 将任一种情况
            # 解析为 pretrained_model/ 目录。
            config_path_obj = Path(config_path)
            policy_dir = config_path_obj.parent if config_path_obj.is_file() else config_path_obj
            self.checkpoint_path = policy_dir.parent
        elif self.job.is_remote:
            return
        else:
            from lerobot.common.train_utils import resolve_resume_checkpoint

            # `self.output_dir` 是从检查点的配置中加载的，指向原始运行的
            # （现已不存在的）本地目录。改为恢复到一个全新的本地目录，
            # 除非用户显式传入了 --output_dir。
            cli_output_dir = parser.parse_arg("output_dir")
            if cli_output_dir:
                self.output_dir = Path(cli_output_dir)
            else:
                now = dt.datetime.now()
                self.output_dir = Path("outputs/train") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_resume"
            self.checkpoint_path = resolve_resume_checkpoint(config_path, self.output_dir)
            policy_dir = self.checkpoint_path / PRETRAINED_MODEL_DIR

        if self.policy is not None:
            self.policy.pretrained_path = policy_dir
        if self.reward_model is not None:
            self.reward_model.pretrained_path = str(policy_dir)

    def validate(self) -> None:
        available_contexts = multiprocessing.get_all_start_methods()
        if (
            self.dataloader_multiprocessing_context is not None
            and self.dataloader_multiprocessing_context not in available_contexts
        ):
            raise ValueError(
                "`dataloader_multiprocessing_context` must be None or one of "
                f"{available_contexts} on this platform, got "
                f"{self.dataloader_multiprocessing_context!r}."
            )

        self._resolve_pretrained_from_cli()

        if self.policy is None and self.reward_model is None:
            raise ValueError(
                "Neither policy nor reward_model is configured. "
                "Please specify one with `--policy.path` or `--reward_model.path`."
            )

        active_cfg = self.trainable_config
        if self.rename_map and active_cfg.pretrained_path is None:
            raise ValueError(
                "`rename_map` requires a pretrained policy checkpoint. "
                "Fresh initialization derives feature names from the current dataset, so no rename is applied."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{active_cfg.type}"
            else:
                self.job_name = f"{self.env.type}_{active_cfg.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = active_cfg.get_optimizer_preset()
            self.scheduler = active_cfg.get_scheduler_preset()

        if self.eval_steps > 0 and self.dataset.eval_split == 0.0:
            raise ValueError("eval_steps > 0 requires dataset.eval_split > 0.0 to hold out eval data.")

        # 远程运行的 repo_id 是在 submit_to_hf 中自动生成的（策略可能在这里
        # 才从 --policy.path 解析出来），因此不要提前对它们强制要求该参数。
        if (
            hasattr(active_cfg, "push_to_hub")
            and active_cfg.push_to_hub
            and not active_cfg.repo_id
            and not self.job.is_remote
        ):
            raise ValueError("'repo_id' argument missing. Please specify it to push the model to the hub.")

        if self.save_checkpoint_to_hub and not (self.policy is not None and self.policy.repo_id):
            raise ValueError("save_checkpoint_to_hub requires --policy.repo_id.")

        self._validate_distributed()

    def _validate_distributed(self) -> None:
        """针对分布式训练范围的快速失败检查。

        Raises:
            ValueError: 当配置请求了已验证范围之外的任何内容时抛出：上下文
                并行或 CFG 并行（保留的占位符）、compile 或激活检查点占位符、
                在非分片运行下使用 DCP 检查点格式，或者在分片训练下使用
                fp16 混合精度、PEFT、奖励模型训练、训练中环境评估，
                或多优化器配置。
        """
        if self.parallelism.cp_size > 1:
            raise ValueError(
                "Context parallelism is not implemented yet: --parallelism.context_parallel.* "
                "degrees must be 1 (reserved for the CP engine round)."
            )
        if self.parallelism.cfg_parallel != 1:
            raise ValueError(
                "CFG parallelism is inference-only and must be 1 for training "
                "(cfg_parallel is reserved for the serving round)."
            )
        if self.accelerator.compile.enabled:
            raise ValueError("--accelerator.compile is a placeholder and not wired yet.")
        if self.accelerator.activation_checkpointing.mode is not ActivationCheckpointingMode.NONE:
            raise ValueError("--accelerator.activation_checkpointing is a placeholder and not wired yet.")
        if self.checkpoint_format is not CheckpointFormat.SAFETENSORS and not self.parallelism.is_sharded:
            raise ValueError(
                f"checkpoint_format={self.checkpoint_format.value} requires a sharded run "
                "(--parallelism.dp_shard != 1); non-sharded checkpoints are always safetensors."
            )
        if self.parallelism.is_sharded:
            if self.accelerator.mixed_precision == "fp16":
                raise ValueError(
                    "fp16 is not supported under sharded training (GradScaler over DTensor "
                    "gradients is unverified); use bf16 or full precision."
                )
            if self.peft is not None:
                raise ValueError("PEFT is not supported under sharded training yet.")
            if self.is_reward_model_training:
                raise ValueError(
                    "Reward-model training is not supported under sharded training yet "
                    "(reward models declare no FSDP wrap units and have no sharded save path)."
                )
            if self.env is not None and self.env_eval_freq > 0:
                raise ValueError(
                    "In-training environment evaluation is not supported under sharded training "
                    "(a rank-0-only rollout of a sharded model deadlocks on collectives); set "
                    "--env_eval_freq=0 and evaluate with lerobot-eval on saved checkpoints."
                )
            if self.optimizer is not None and self.optimizer.builds_multiple_optimizers:
                raise ValueError("Multi-optimizer configs are not supported under sharded training.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """用于 draccus 预训练路径加载的键。"""
        return ["policy", "reward_model"]

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]  # 因为第三方库 draccus 使用 Any 作为返回类型

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            dl_kwargs = {
                "repo_id": model_id,
                "revision": revision,
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "resume_download": resume_download,
                "token": token,
                "local_files_only": local_files_only,
            }
            try:
                config_file = hf_hub_download(filename=TRAIN_CONFIG_NAME, **dl_kwargs)
            except HfHubHTTPError as e:
                # 根目录没有 train_config.json：这是一个来自中断运行的周期性检查点仓库。
                # 回退到最新检查点的配置，这样就可以直接用 `--config_path=<repo>`
                # 从仓库恢复运行。
                latest = find_latest_hub_checkpoint(model_id, token=token, revision=revision)
                if latest is None:
                    raise FileNotFoundError(
                        f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                    ) from e
                config_file = hf_hub_download(
                    filename=f"{latest}/{PRETRAINED_MODEL_DIR}/{TRAIN_CONFIG_NAME}", **dl_kwargs
                )

        cli_args = kwargs.pop("cli_args", [])
        # 旧版 RA-BC 迁移仅适用于框架保存的检查点（始终为 JSON）。
        # 手写的 YAML/TOML 配置应使用当前的 sample_weighting 结构。
        if config_file is not None and config_file.endswith(".json"):
            with open(config_file) as f:
                config = json.load(f)
            migrated_config = _migrate_legacy_rabc_fields(config)
            if migrated_config is not None:
                with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
                    json.dump(migrated_config, f)
                    config_file = f.name

        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)
