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
"""分片感知的检查点原语。

两个产物通道，各有不同的归属：

- **可分发**的 ``model.safetensors``：由 ``PreTrainedPolicy.save_pretrained``
  通过 :func:`full_model_state_dict` 生成 —— 当模型被分片时，这是一次集合式的全量 gather；
- **恢复**通道（分片运行）：通过 accelerate 的 ``save/load_fsdp_model`` 和
  ``save/load_fsdp_optimizer`` 写入/读取的 torch DCP 目录（``pytorch_model_fsdp_0/`` 和
  ``optimizer_0/``，名称从 accelerate 常量导入），加载时可在拓扑变化间重新分片。

所有涉及分片状态的函数都是集合操作，必须在所有 rank 上运行。
"""

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from accelerate import Accelerator


def is_sharded_module(module: nn.Module) -> bool:
    """当 `fully_shard` 拥有此模块的参数时为 True（FSDP2 的原地类替换）。

    Args:
        module (nn.Module): 要检查的模块（会通过 `_orig_mod` 看穿 torch.compile 包装器）。

    Returns:
        bool: 当模块（或其编译后的 `_orig_mod`）是 `FSDPModule` 时为 True。
    """
    from torch.distributed.fsdp import FSDPModule

    if isinstance(module, FSDPModule):
        return True
    # torch.compile 会包装分片模块；与 accelerate 的 `_orig_mod` 检查保持一致。
    orig_mod = getattr(module, "_orig_mod", None)
    return orig_mod is not None and isinstance(orig_mod, FSDPModule)


def full_model_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """无论参数如何布局，都返回模块的完整（未分片）state dict。

    分片模块通过 torch 的 DCP state-dict API 进行 gather：这是一个必须在每个 rank 上
    运行的集合操作；当 ``cpu_offload=True`` 时，完整字典只在主 rank 上实体化，
    其他每个 rank 收到字面量 ``{}``（已运行时验证 —— 仅在 rank 0 上调用会死锁）。
    普通模块在每个 rank 上返回 ``module.state_dict()``。

    Args:
        module (nn.Module): 要读取 state dict 的（可能已分片的）模块。

    Returns:
        dict[str, torch.Tensor]: 完整的 state dict —— 当模块已分片时仅在主 rank 上
            （其他位置为 ``{}``），否则在每个 rank 上。
    """
    if not is_sharded_module(module):
        return module.state_dict()

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(module, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def _fsdp_plugin(accelerator: "Accelerator") -> object:
    """accelerator 的 FSDP 插件，下方所有 DCP 保存/加载辅助函数都需要它。

    Args:
        accelerator (Accelerator): prepare 了分片模型的 accelerator。

    Returns:
        object: 由 `accelerator.state` 持有的 FSDP 插件。

    Raises:
        RuntimeError: 如果 accelerator 未配置 FSDP 插件。
    """
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if plugin is None:
        raise RuntimeError("Sharded checkpointing requires an FSDP-prepared Accelerator.")
    return plugin


def save_sharded_model(accelerator: "Accelerator", model: nn.Module, output_dir: Path) -> None:
    """写入 DCP 模型分片（`pytorch_model_fsdp_0/`）。集合操作：在所有 rank 上调用。

    Args:
        accelerator (Accelerator): prepare 了分片模型的 accelerator。
        model (nn.Module): 要保存的已 prepare（分片）模型。
        output_dir (Path): 在其中创建分片子目录的目录。
    """
    from accelerate.utils import save_fsdp_model

    # accelerate 1.14 的 DCP 辅助函数会对路径做字符串包含检查：
    # 始终传入 str，绝不传 Path。
    save_fsdp_model(_fsdp_plugin(accelerator), accelerator, model, str(output_dir))


def load_sharded_model(accelerator: "Accelerator", model: nn.Module, input_dir: Path) -> None:
    """将 DCP 模型分片加载到已 prepare（分片）的模型中。集合操作：在所有 rank 上调用。

    Args:
        accelerator (Accelerator): prepare 了分片模型的 accelerator。
        model (nn.Module): 要加载进去的已 prepare（分片）模型。
        input_dir (Path): 包含 `pytorch_model_fsdp_0/` 分片子目录的目录。
    """
    from accelerate.utils import load_fsdp_model
    from accelerate.utils.constants import FSDP_MODEL_NAME

    # 传入精确的分片目录：accelerate 的加载会用子串检查来解析路径
    # （路径中含 "pytorch_model_fsdp" -> 直接使用），这对恰好包含该标记的
    # 运行路径会误判；精确目录使该检查具有确定性。
    load_fsdp_model(_fsdp_plugin(accelerator), accelerator, model, str(input_dir / f"{FSDP_MODEL_NAME}_0"))


def save_sharded_optimizer(
    accelerator: "Accelerator", optimizer: torch.optim.Optimizer, model: nn.Module, output_dir: Path
) -> None:
    """写入 DCP 优化器分片（`optimizer_0/`）。集合操作：在所有 rank 上调用。

    Args:
        accelerator (Accelerator): prepare 了模型和优化器的 accelerator。
        optimizer (torch.optim.Optimizer): 要保存其状态的已 prepare 优化器。
        model (nn.Module): 优化器状态所键入的已 prepare（分片）模型。
        output_dir (Path): 在其中创建分片子目录的目录。
    """
    from accelerate.utils import save_fsdp_optimizer

    save_fsdp_optimizer(_fsdp_plugin(accelerator), accelerator, optimizer, model, str(output_dir))


def load_sharded_optimizer(
    accelerator: "Accelerator", optimizer: torch.optim.Optimizer, model: nn.Module, input_dir: Path
) -> None:
    """将 DCP 优化器分片加载到已 prepare 的优化器中。集合操作：在所有 rank 上调用。

    必须在 ``accelerator.prepare()`` 之后运行：FSDP2 的 prepare 会将优化器的参数组
    重新绑定到分片 DTensor，但从不迁移 ``optimizer.state`` —— 重新分片加载是
    恢复它的唯一正确方式。

    Args:
        accelerator (Accelerator): prepare 了模型和优化器的 accelerator。
        optimizer (torch.optim.Optimizer): 要恢复状态进去的已 prepare 优化器。
        model (nn.Module): 优化器状态所键入的已 prepare（分片）模型。
        input_dir (Path): 包含 `optimizer_0/` 分片子目录的目录。
    """
    from accelerate.utils import load_fsdp_optimizer
    from accelerate.utils.constants import OPTIMIZER_NAME

    # 与 load_sharded_model 相同的原因，使用精确的分片目录：accelerate 的子串
    # 检查（路径中含 "optimizer"）会误读例如 --job_name=optimizer_sweep 的运行路径。
    load_fsdp_optimizer(
        _fsdp_plugin(accelerator), accelerator, optimizer, model, str(input_dir / f"{OPTIMIZER_NAME}_0")
    )


def dcp_to_safetensors(dcp_dir: Path, output_dir: Path, *, delete_dcp: bool = False) -> Path:
    """将 DCP 分片目录合并为单个 `model.safetensors`（离线，单进程）。

    这是 `accelerate.utils.merge_fsdp_weights` 的薄封装，后者无需进程组即可加载分片，
    直接写入 safetensors，并且 —— 在被要求时 —— 删除已合并的分片目录本身，
    仅在主进程上且仅在合并成功后执行。

    Args:
        dcp_dir (Path): 要合并的 DCP 分片目录（例如 `.../pytorch_model_fsdp_0`）。
        output_dir (Path): 合并后的 `model.safetensors` 写入的目录。
        delete_dcp (bool): 是否在分片目录合并后将其删除。
            默认为 False。

    Returns:
        Path: 写入的 `model.safetensors` 文件的路径。
    """
    from accelerate.utils import merge_fsdp_weights

    merge_fsdp_weights(
        str(dcp_dir), str(output_dir), safe_serialization=True, remove_checkpoint_dir=delete_dcp
    )
    return output_dir / "model.safetensors"
