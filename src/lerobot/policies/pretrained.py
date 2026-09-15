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
from __future__ import annotations

import abc
import builtins
import dataclasses
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, TypeVar, Unpack

from huggingface_hub import hf_hub_download, save_torch_state_dict
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import resolve_safetensors_device
from lerobot.utils.hub import HubMixin
from lerobot.utils.import_utils import _peft_available, require_package

from .utils import log_model_loading_keys

if TYPE_CHECKING or _peft_available:
    from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftType, get_peft_model
else:
    PEFT_TYPE_TO_CONFIG_MAPPING = None
    PeftType = None
    get_peft_model = None

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

T = TypeVar("T", bound="PreTrainedPolicy")

# 固定为远大于任何策略总大小的值，使 save_torch_state_dict 始终恰好输出一个
# `model.safetensors`（无分片、无索引）——这是一个常量，而不是计算出的字节数。
_SINGLE_FILE_SHARD_SIZE = "1TB"


class ActionSelectKwargs(TypedDict, total=False):
    noise: Tensor | None


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    策略模型的基类。
    """

    config_class: None
    name: None

    # --- 声明式并行/加速接口 ----------------------------------------
    # 构成 FSDP2 包装单元（以及接线完成后的激活检查点单元）的模块类名。
    # 在 `accelerator.prepare()` 之前由 `lerobot.distributed.set_fsdp_wrap_modules`
    # 解析到 accelerate 插件上；任何地方都没有包装来源的分片训练会响亮地失败，
    # 而不是静默地只包装根模块。
    _fsdp_wrap_modules: ClassVar[list[str] | None] = None
    # 在分片策略上被调用时，必须触发 FSDP2 unshard/reshard 钩子的
    # 非 `forward` 入口点（在 prepare 之后通过 `torch.distributed.fsdp
    # .register_fsdp_forward_method` 注册）；未注册就调用它们会在
    # Tensor/DTensor 混合时崩溃。
    _fsdp_forward_methods: ClassVar[tuple[str, ...]] = ("select_action", "predict_action_chunk")
    # （未来）激活检查点接线的能力门控。
    supports_gradient_checkpointing: ClassVar[bool] = False
    # 声明式上下文并行计划（diffusers `ContextParallelModelPlan` 语义：
    # 模块 FQN -> 序列拆分/聚合规范）。预留给 CP 引擎轮次。
    _cp_plan: ClassVar[dict[str, Any] | None] = None
    # `drop_queued_actions` 要清除的属性名：`populate_queues` 风格的 `_queues` 字典
    # 和裸的 `_action_queue` deque。使用其他名称的分块策略必须扩展这个
    # ClassVar（或重写该方法），否则丢弃队列会静默地什么都不做。
    _action_queue_attrs: ClassVar[tuple[str, ...]] = ("_queues", "_action_queue")

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")
        # rollout 栈通过 supports_text_generation() 来控制文本查询，因此没有它
        # 的 generate_text() 重写是不可达的。通过 MRO 比较，所以从符合要求的
        # 父类继承的重写也算数。
        if (
            cls.generate_text is not PreTrainedPolicy.generate_text
            and cls.supports_text_generation is PreTrainedPolicy.supports_text_generation
        ):
            raise TypeError(
                f"{cls.__name__} provides generate_text() but not supports_text_generation(). "
                "Override supports_text_generation() too (returning True, or a "
                "checkpoint-conditional value), otherwise the text head is never used."
            )

    def _save_pretrained(self, save_directory: Path) -> None:
        """将该策略的参数（和配置）序列化到 `save_directory`。

        分片在内部处理：在 FSDP2 下，完整的 state dict 通过 COLLECTIVE 收集，
        因此当策略是分片状态时，此方法（通过 `save_pretrained`）必须在
        每个 rank 上调用——只在 rank 0 上调用会死锁。在所有布局
        （单机、DDP、分片）下，文件写入仅发生在主进程上。

        Args:
            save_directory (Path): 策略配置（`config.json`）和
                safetensors 权重文件的目标目录。
        """
        # 延迟导入：持久化层仅在保存时才引入 lerobot.distributed。
        from lerobot.distributed.checkpoint import full_model_state_dict, is_sharded_module
        from lerobot.distributed.utils import is_main_process

        model_to_save = self.module if hasattr(self, "module") else self
        if is_sharded_module(model_to_save):
            logging.info("Gathering the full state dict from all ranks (sharded policy).")
        state_dict = full_model_state_dict(model_to_save)  # 分片时为集合操作；非主进程上为 {}
        if not state_dict or not is_main_process():
            # 分片：收集结果只在主 rank 上具象化（空检查）。
            # 非分片多 rank（DDP）：每个 rank 都持有完整字典——显式的 rank
            # 门控防止 N 个 rank 争抢同一批文件。单进程：不会走到这里。
            return
        self.config._save_pretrained(save_directory)
        save_torch_state_dict(state_dict, str(save_directory), max_shard_size=_SINGLE_FILE_SHARD_SIZE)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        默认情况下，策略使用 `policy.eval()` 被设置为评估模式（dropout 模块被
        停用）。要训练它，你应该先用 `policy.train()` 将其设回训练模式。
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        missing_keys, unexpected_keys = load_model_as_safetensor(
            model, model_file, strict=strict, device=resolve_safetensors_device(map_location)
        )
        log_model_loading_keys(missing_keys, unexpected_keys)
        return model

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        返回要传递给优化器的策略特定参数字典。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """每当环境被重置时调用。

        执行诸如清除缓存之类的操作。
        """
        raise NotImplementedError

    def drop_queued_actions(self) -> None:
        """丢弃由先前 ``select_action`` 调用预计算的动作。

        强制下一次 ``select_action`` 执行全新的前向传播，使回合中途的条件
        变化（例如新指令）立即生效，而不是等队列排空后才生效。
        与 :meth:`reset` 不同，回合的其余状态会被保留。请在调用
        ``select_action`` 的线程中调用它。清除 :attr:`_action_queue_attrs` 中
        命名的队列；不维护动作队列的策略继承到的是空操作。
        """
        for attr in self._action_queue_attrs:
            queue = getattr(self, attr, None)
            if isinstance(queue, dict):
                # populate_queues 风格的字典：只清除 ACTION，不清除观测历史。
                if ACTION in queue:
                    queue[ACTION].clear()
            elif queue is not None:
                queue.clear()

    def supports_rtc(self) -> bool:
        """该策略是否实现了实时分块（Real-Time Chunking）推理语义。"""
        return False

    def supports_text_generation(self) -> bool:
        """该策略是否实现了 :meth:`generate_text`（两者需一起重写）。"""
        return False

    def generate_text(self, batch: dict[str, Any]) -> str:
        """在预处理后的观测批次上运行策略的文本头。

        请求作为补充数据附带在 ``batch`` 上（:data:`~lerobot.utils.constants.QUERY_KIND`
        / ``QUERY_TEXT``）；``next_subtask`` 回复会直接送入 ``set_task``，因此它必须
        恰好是一个子任务，而不是计划或编号列表。返回生成的文本，且不得
        修改产生动作的状态（队列、观测历史）。
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no text head — it cannot answer questions or plan subtasks."
        )

    # TODO(aliberts, rcadene): 拆分为 'forward' 和 'compute_loss'？
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: 损失以及可能的其他信息。除了作为
                Tensor 的损失之外，所有其他项都应该是便于日志记录的原生 Python 类型。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """针对给定观测返回动作块（用于动作分块策略），可能以批处理模式。

        使用动作分块的子类应在 `select_action` 中使用此方法来构建
        缓存起来供选择使用的动作块。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """返回一个要在环境中执行的动作（可能以批处理模式）。

        当模型使用观测历史或输出动作序列时，此方法负责处理缓存。
        """
        raise NotImplementedError

    def push_model_to_hub(
        self,
        cfg: TrainPipelineConfig,
        peft_model=None,
        state_dict: dict[str, Tensor] | None = None,
        dataset_meta: LeRobotDatasetMetadata | None = None,
    ) -> None:
        """将该策略发布到 Hub。

        已弃用：请改用 :func:`lerobot.common.train_utils.publish_trained_model`，
        它会在发布模型的同时发布预/后处理器。

        Args:
            cfg (TrainPipelineConfig): 训练配置；保存为 `train_config.json`
                并用于渲染模型卡片。
            peft_model: 训练适配器时的 PEFT 包装器，其权重会在发布的仓库中
                取代完整模型权重。默认为 None。
            state_dict (dict[str, Tensor] | None): 被忽略；当策略为分片状态时，
                权重现在在内部收集。默认为 None。
            dataset_meta (LeRobotDatasetMetadata | None): 模型卡片用的数据集元数据
                （如果可用）。默认为 None。
        """
        from lerobot.common.train_utils import publish_trained_model

        warnings.warn(
            "PreTrainedPolicy.push_model_to_hub is deprecated and will be removed in a future "
            "version. Use lerobot.common.train_utils.publish_trained_model(cfg, model, "
            "preprocessor, postprocessor, dataset_meta) instead.",
            FutureWarning,
            stacklevel=2,
        )
        if state_dict is not None:
            warnings.warn(
                "The `state_dict` argument is ignored: sharded weights are gathered internally "
                "when the policy is saved.",
                FutureWarning,
                stacklevel=2,
            )
        publish_trained_model(cfg, self, None, None, dataset_meta, peft_model=peft_model)

    def wrap_with_peft(
        self,
        peft_config=None,
        peft_cli_overrides: dict | None = None,
    ) -> PreTrainedPolicy:
        """
        用 PEFT 适配器包装该策略以进行参数高效微调。

        该方法是 PEFT 集成的唯一入口。子类应重写
        `_get_default_peft_targets()` 以提供默认目标模块，并重写
        `_validate_peft_config()` 进行策略特定的验证。

        Args:
            peft_config: 可选的 PEFT 适配器配置（例如 LoraConfig）。
                如果提供，则直接使用（并应用 CLI 覆盖）。
            peft_cli_overrides: 可选的 CLI 覆盖字典（method_type、target_modules、r 等）。
                这些会与策略默认值合并以构建最终配置。
        """
        require_package("peft", extra="peft")

        # 如果用户提供了完整配置，直接使用（并应用覆盖）
        if peft_config is not None:
            final_config = peft_config
            if peft_cli_overrides:
                final_config = self._apply_peft_cli_overrides(final_config, peft_cli_overrides)
        else:
            # 根据默认值 + CLI 覆盖构建配置
            final_config = self._build_peft_config(peft_cli_overrides or {})

        # 验证配置
        self._validate_peft_config(final_config)

        # 冻结基础参数，只训练适配器参数
        for p in self.parameters():
            p.requires_grad_(False)

        # 存储预训练路径以作为 PEFT 的 base_model_name_or_path
        if self.config.pretrained_path:
            self.name_or_path = str(self.config.pretrained_path)

        # 用 PEFT 包装
        peft_model = get_peft_model(self, final_config)

        # 将配置标记为使用 PEFT，以便后续正确加载
        peft_model.config.use_peft = True

        logging.info(f"Wrapped {self.name} with PEFT ({type(final_config).__name__})")
        return peft_model

    def _get_default_peft_targets(self) -> dict[str, any] | None:
        """
        返回该策略的默认 PEFT 目标模块。

        在子类中重写以提供策略特定的默认值。这些默认值
        与 PEFT 方法无关——它们只指定要作用于哪些模块。

        """
        return None

    def _validate_peft_config(self, peft_config) -> None:
        """
        验证该策略的 PEFT 配置。

        在子类中重写以添加策略特定的验证或警告。
        默认实现检查 pretrained_path 是否存在。

        Args:
            peft_config: 要验证的 PEFT 配置。

        Raises:
            ValueError: 如果配置无效。
        """
        if not self.config.pretrained_path:
            raise ValueError(
                "Training from scratch using PEFT is unlikely to yield good results. "
                "Supply a `policy.pretrained_path` to fine-tune an existing model."
            )

    def _preprocess_peft_cli_overrides(self, cli_overrides: dict, peft_method_type) -> dict:
        """
        预处理 CLI 覆盖：重命名键并处理方法特定的 init_type。

        Args:
            cli_overrides: CLI 选项字典（会被复制，不会被修改）。
            peft_method_type: PEFT 方法的 PeftType 枚举值。

        Returns:
            预处理后的字典，键已重命名，init_type 已映射到方法特定的键。
        """
        require_package("peft", extra="peft")

        cli_overrides = cli_overrides.copy()

        # 处理 full_training_modules -> modules_to_save 的重命名
        if "full_training_modules" in cli_overrides:
            cli_overrides["modules_to_save"] = cli_overrides.pop("full_training_modules")

        # 移除 method_type，因为它在别处处理
        cli_overrides.pop("method_type", None)

        # 根据 PEFT 方法特殊处理 init_type
        init_type = cli_overrides.pop("init_type", None)
        if init_type is not None:
            if peft_method_type == PeftType.LORA:
                cli_overrides["init_lora_weights"] = init_type
            elif peft_method_type == PeftType.MISS:
                cli_overrides["init_weights"] = init_type
            else:
                raise ValueError(f"Init type '{init_type}' unknown for PEFT method {peft_method_type}.")

        return cli_overrides

    def _build_peft_config(self, cli_overrides: dict):
        """根据策略默认值和 CLI 覆盖构建 PEFT 配置。"""
        require_package("peft", extra="peft")

        # 确定 PEFT 方法类型（默认为 LORA）
        method_type_str = cli_overrides.get("method_type") or "lora"
        peft_method_type = PeftType[method_type_str.upper()]
        peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]

        # 预处理 CLI 覆盖
        cli_overrides = self._preprocess_peft_cli_overrides(cli_overrides, peft_method_type)

        # 从策略默认值开始，应用 CLI 覆盖
        config_dict = dict(self._get_default_peft_targets() or {})
        for key, value in cli_overrides.items():
            if value is not None:
                config_dict[key] = value

        # 确保有 target_modules
        if not config_dict.get("target_modules"):
            raise ValueError(
                f"Policy '{self.name}' does not define default target_modules. "
                "Please pass --peft.target_modules explicitly."
            )

        return peft_config_cls(**config_dict)

    def _apply_peft_cli_overrides(self, peft_config, cli_overrides: dict):
        """将 CLI 覆盖应用到现有的 PEFT 配置。"""
        require_package("peft", extra="peft")

        # 从现有配置或 CLI 覆盖获取方法类型
        method_type_str = cli_overrides.get("method_type")
        if method_type_str:
            peft_method_type = PeftType[method_type_str.upper()]
            peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]
        else:
            peft_method_type = PeftType(peft_config.peft_type)
            peft_config_cls = type(peft_config)

        # 预处理 CLI 覆盖
        cli_overrides = self._preprocess_peft_cli_overrides(cli_overrides, peft_method_type)

        # 从现有配置开始，应用 CLI 覆盖
        config_dict = {k: v for k, v in dataclasses.asdict(peft_config).items() if not k.startswith("_")}
        for key, value in cli_overrides.items():
            if value is not None:
                config_dict[key] = value

        return peft_config_cls(**config_dict)
