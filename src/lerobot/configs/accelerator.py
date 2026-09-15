#!/usr/bin/env python

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
"""执行时配置：交给 `Accelerator`（或由其应用）的所有内容。

每个子配置都镜像了对应 accelerate 对象的简单类型子集，
并在运行时构建它（就像 ``OptimizerConfig.build()`` 构建 ``torch.optim.Optimizer`` 一样），
因此整棵树可以通过 CLI 和 ``train_config.json`` 往返，解析配置时
永远不会导入 accelerate。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from lerobot.configs.parallelism import ParallelismConfig

if TYPE_CHECKING:
    from accelerate import Accelerator
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        FullyShardedDataParallelPlugin,
        GradientAccumulationPlugin,
    )


@dataclass
class FSDPConfig:
    """镜像 LeRobot 支持的 `FullyShardedDataParallelPlugin` 子集（仅 FSDP2）。

    恰好只应用一种 wrap 策略：`wrap_modules`（构成 FSDP 单元的模块*类名*——
    之后也是激活检查点单元）或 `min_num_params`（基于大小）。
    当两者都为 None 时，使用策略自身的 `_fsdp_wrap_modules` 声明；
    如果完全不存在 wrap 来源，运行会大声失败，而不是静默地只 wrap 根模块。
    """

    reshard_after_forward: bool = True
    wrap_modules: list[str] | None = None
    min_num_params: int | None = None
    cpu_offload: bool = False
    # 与模块 FQN 匹配的正则表达式，用于将其参数排除在分片之外。
    ignored_modules: str | None = None

    def __post_init__(self) -> None:
        """验证 wrap 策略字段。

        Raises:
            ValueError: 如果同时设置了 ``wrap_modules`` 和 ``min_num_params``
                （它们是互斥的 wrap 策略），或 ``min_num_params`` < 1。
        """
        if self.wrap_modules is not None and self.min_num_params is not None:
            raise ValueError(
                "fsdp.wrap_modules and fsdp.min_num_params are mutually exclusive wrap policies."
            )
        if self.min_num_params is not None and self.min_num_params < 1:
            raise ValueError(f"fsdp.min_num_params must be >= 1, got {self.min_num_params}.")

    def build_plugin(self) -> "FullyShardedDataParallelPlugin":
        """为 `Accelerator(fsdp_plugin=...)` 构建 FSDP2 插件。

        Returns:
            FullyShardedDataParallelPlugin: 携带镜像的 wrap 策略、重新分片、
                CPU offload 和 ignored-modules 设置的 FSDP2（`fsdp_version=2`）插件。
        """
        from accelerate.utils import FullyShardedDataParallelPlugin

        use_size_policy = self.min_num_params is not None
        return FullyShardedDataParallelPlugin(
            fsdp_version=2,
            reshard_after_forward=self.reshard_after_forward,
            auto_wrap_policy="size_based_wrap" if use_size_policy else "transformer_based_wrap",
            # 这里合理地仍可能为 None：策略声明的默认值会在 `accelerator.prepare()`
            # 之前应用（参见 lerobot.distributed.factory.set_fsdp_wrap_modules）。
            transformer_cls_names_to_wrap=list(self.wrap_modules) if self.wrap_modules else None,
            min_num_params=self.min_num_params,
            cpu_offload=self.cpu_offload,
            ignored_modules=self.ignored_modules,
            # state_dict_type 保持 FSDP2 默认值（SHARDED_STATE_DICT），从不切换：
            # 完整聚合通过 torch 的 state-dict API 进行，不会查询该插件。
            # activation_checkpointing 保持 False：AC 由 LeRobot 拥有。
        )


@dataclass
class DDPConfig:
    """镜像 LeRobot 公开的 `DistributedDataParallelKwargs` 子集。"""

    # 当前脚本内的默认值，为具有条件计算的模型保留。
    find_unused_parameters: bool = True
    gradient_as_bucket_view: bool = False
    static_graph: bool = False

    def build_kwargs_handler(self) -> "DistributedDataParallelKwargs":
        """为 `Accelerator(kwargs_handlers=[...])` 构建 DDP kwargs 处理器。

        Returns:
            DistributedDataParallelKwargs: 携带镜像的 DDP 字段的处理器，
                当 accelerate 用 `DistributedDataParallel` 包装模型时应用。
        """
        from accelerate.utils import DistributedDataParallelKwargs

        return DistributedDataParallelKwargs(
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
            static_graph=self.static_graph,
        )


@dataclass
class GradientAccumulationConfig:
    """镜像 LeRobot 支持的 `GradientAccumulationPlugin` 子集。

    只有步数是一个可调参数。``sync_with_dataloader`` 由
    :meth:`build_plugin` 固定为 False：训练循环会循环使用有限的 dataloader，
    因此 accelerate 默认的在每个 dataloader 结束时同步，会在每个数据集 epoch
    边界而不是每 ``steps`` 个微批次强制进行优化器步进。
    """

    steps: int = 1

    def __post_init__(self) -> None:
        """验证累积步数。

        Raises:
            ValueError: 如果 ``steps`` < 1。
        """
        if self.steps < 1:
            raise ValueError(f"gradient_accumulation.steps must be >= 1, got {self.steps}.")

    def build_plugin(self) -> "GradientAccumulationPlugin":
        """为 `Accelerator(gradient_accumulation_plugin=...)` 构建插件。

        这是一个命名的插件参数，而不是 `kwargs_handlers` 条目：accelerate 通过
        其专用的构造函数参数消费此对象 —— `KwargsHandler` 基类只是借给它
        `to_kwargs()`，因此是消费点而不是继承决定了它的角色。

        Returns:
            GradientAccumulationPlugin: 携带镜像的步数，并固定
                ``sync_with_dataloader=False``（参见类的 docstring）。
        """
        from accelerate.utils import GradientAccumulationPlugin

        return GradientAccumulationPlugin(num_steps=self.steps, sync_with_dataloader=False)


@dataclass
class CompileConfig:
    """torch.compile 参数 —— 已配置的占位符：接线将在后续轮次落地。

    它将遵循的初始化顺序约定已经固定：compile 在 CP dispatch 安装和
    激活检查点之后、`fully_shard` 之前应用，按区域（每个 wrap 单元）进行——
    这是唯一经过 FSDP2 验证的组合。
    """

    enabled: bool = False
    backend: str = "inductor"
    mode: str | None = None
    regional: bool = True


class ActivationCheckpointingMode(str, Enum):
    NONE = "none"
    FULL = "full"


@dataclass
class ActivationCheckpointingConfig:
    """激活检查点参数 —— 已配置的占位符：接线将在后续轮次落地。

    AC 单元将与 FSDP wrap 单元一致（一个声明驱动两者），在
    torch.compile 和 `fully_shard` 之前应用（与 CompileConfig 相同的顺序约定）。
    """

    mode: ActivationCheckpointingMode = ActivationCheckpointingMode.NONE


@dataclass
class AcceleratorConfig:
    """构建 `Accelerator` —— `parallelism` 拓扑的运行时对应物。

    `mixed_precision` 为 DDP/单 GPU 运行选择 accelerate 原生的 AMP，
    为分片运行选择 FSDP2 的 `MixedPrecisionPolicy`（由 accelerate 推导）。
    分片运行仅支持 "no" 和 "bf16"；fp16 的 GradScaler-over-DTensor 路径
    未经验证，会在配置验证时快速失败。
    """

    mixed_precision: str = "no"
    gradient_accumulation: GradientAccumulationConfig = field(default_factory=GradientAccumulationConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    activation_checkpointing: ActivationCheckpointingConfig = field(
        default_factory=ActivationCheckpointingConfig
    )

    def __post_init__(self) -> None:
        """验证面向 accelerate 的标量字段。

        Raises:
            ValueError: 如果 ``mixed_precision`` 不是 ``"no"``、``"fp16"``、``"bf16"`` 之一。
        """
        if self.mixed_precision not in ("no", "fp16", "bf16"):
            raise ValueError(
                f"mixed_precision must be one of 'no', 'fp16', 'bf16', got {self.mixed_precision!r}."
            )

    def build(self, parallelism: ParallelismConfig, *, cpu: bool = False) -> "Accelerator":
        """将镜像的字段转换为就绪的 `Accelerator`（每个进程调用一次）。

        `parallelism` 必须已经针对世界大小完成解析。降级矩阵在这里编码，
        且仅在此处：分片 -> FSDP2（通过 accelerate 的 `ParallelismConfig` mesh
        实现 +HSDP），仅复制 -> DDP kwargs，单进程 -> 普通模式。

        Args:
            parallelism (ParallelismConfig): 已解析的进程拓扑；选择配置哪条
                accelerate 路径（FSDP2 mesh、DDP kwargs 处理器或普通模式）。
            cpu (bool): 即使 CUDA 可用也强制使用 CPU 执行。默认为 False。

        Returns:
            Accelerator: 此进程配置好的 accelerate 入口。
        """
        from accelerate import Accelerator

        kwargs: dict = {
            # LeRobot 在每个训练步手动步进一次调度器；accelerate 不得
            # 按 num_processes 重新缩放调度器的步进。
            "step_scheduler_with_optimizer": False,
            "gradient_accumulation_plugin": self.gradient_accumulation.build_plugin(),
            "mixed_precision": self.mixed_precision,
            "cpu": cpu,
        }
        if parallelism.is_sharded:
            kwargs["fsdp_plugin"] = self.fsdp.build_plugin()
            kwargs["parallelism_config"] = _accelerate_parallelism_config(parallelism)
        elif parallelism.is_replicated_only:
            kwargs["kwargs_handlers"] = [self.ddp.build_kwargs_handler()]
        return Accelerator(**kwargs)


def _accelerate_parallelism_config(parallelism: ParallelismConfig) -> object:
    """LeRobot 拓扑 -> accelerate 的 `ParallelismConfig`。

    CP 被如实声明（`cp_size = ring x ulysses`），这样 accelerate 会构建规范的
    mesh，将 CP 并入 FSDP 分片组（`dp_shard_cp`），并在 CP 组内复制批次。
    ring/ulysses 的子结构保持为 `lerobot.distributed.ParallelDims` 的私有内容。

    Args:
        parallelism (ParallelismConfig): 要转换的已解析 LeRobot 拓扑。

    Returns:
        object: 镜像 `dp_replicate`、`dp_shard` 和折叠后的 `cp_size` 的
            accelerate `ParallelismConfig`（标注为 `object`，这样导入本模块
            永远不会导入 accelerate）。
    """
    from accelerate.parallelism_config import ParallelismConfig as AccelerateParallelismConfig

    return AccelerateParallelismConfig(
        dp_replicate_size=parallelism.dp_replicate,
        dp_shard_size=parallelism.dp_shard,
        cp_size=parallelism.cp_size,
    )
