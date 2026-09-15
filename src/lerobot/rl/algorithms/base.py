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

from __future__ import annotations

import abc
import builtins
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from torch.optim import Optimizer

from lerobot.lerobot_types import BatchType
from lerobot.utils.hub import HubMixin

from .configs import RLAlgorithmConfig, TrainingStats

if TYPE_CHECKING:
    from torch import nn

    from ..data_sources.data_mixer import DataMixer

T = TypeVar("T", bound="RLAlgorithm")


class RLAlgorithm(HubMixin, abc.ABC):
    """所有 RL 算法的基类。"""

    config_class: type[RLAlgorithmConfig]
    name: str
    config: RLAlgorithmConfig

    @abc.abstractmethod
    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """一次完整的训练步骤。

        算法按需多次调用 ``next(batch_iterator)``
        （例如 SAC 调用 ``utd_ratio`` 次）以获取新的批次。
        该迭代器由训练器（trainer）持有；算法只是从中
        消费数据。
        """
        raise NotImplementedError

    def configure_data_iterator(
        self,
        data_mixer: DataMixer,
        batch_size: int,
        *,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ) -> Iterator[BatchType]:
        """创建该算法所需的数据迭代器。

        默认实现使用标准的 ``data_mixer.get_iterator()``。
        需要特殊采样的算法应重写此方法。
        """
        return data_mixer.get_iterator(
            batch_size=batch_size,
            async_prefetch=async_prefetch,
            queue_size=queue_size,
        )

    @abc.abstractmethod
    def make_optimizers_and_scheduler(self) -> dict[str, Optimizer]:
        """构建并返回训练期间使用的优化器。

        在构造完成后于 learner 端调用一次。
        """
        raise NotImplementedError

    def get_optimizers(self) -> dict[str, Optimizer]:
        """返回优化器，用于检查点保存/外部调度。"""
        return {}

    @property
    def optimization_step(self) -> int:
        """当前 learner 的优化步数。

        这是检查点保存/恢复稳定契约的一部分。算法可以
        使用此默认存储，也可以重写以实现自定义行为。
        """
        return getattr(self, "_optimization_step", 0)

    @optimization_step.setter
    def optimization_step(self, value: int) -> None:
        self._optimization_step = int(value)

    def get_weights(self) -> dict[str, Any]:
        """要推送给 actor 的策略 state-dict。"""
        return {}

    @abc.abstractmethod
    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        """加载从 learner 接收到的策略 state-dict。"""
        raise NotImplementedError

    @abc.abstractmethod
    def state_dict(self) -> dict[str, torch.Tensor]:
        """算法持有的可训练张量。

        必须为算法持有的、不属于策略的所有内容返回一个扁平的
        张量映射（例如 critic 集成、目标网络、温度参数）。
        没有仅训练用张量的算法应显式返回空字典。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
    ) -> None:
        """原地加载算法持有的张量。

        实现必须保持优化器所引用的任何 ``nn.Parameter`` 的身份不变
        （例如 SAC 的 ``log_alpha``），即使用 ``.copy_()``
        而不是重新绑定属性。
        """
        raise NotImplementedError

    def _save_pretrained(self, save_directory: Path) -> None:
        """将算法的张量和配置持久化到 ``save_directory``。

        写入 ``model.safetensors``（通过 :meth:`state_dict` 得到的算法张量）
        和 ``config.json``（通过 :meth:`RLAlgorithmConfig.save_pretrained`）。
        """
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_safetensors(tensors, str(save_directory / SAFETENSORS_SINGLE_FILE))
        self.config._save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        policy: nn.Module,
        config: RLAlgorithmConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        device: str | torch.device = "cpu",
        **algo_kwargs: Any,
    ) -> T:
        """构建算法并从 ``pretrained_name_or_path`` 加载其权重。"""
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            )
        if hasattr(config, "policy_config"):
            config.policy_config = policy.config

        instance = cls(policy=policy, config=config, **algo_kwargs)

        model_id = str(pretrained_name_or_path)
        if os.path.isdir(model_id):
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
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
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        tensors = load_safetensors(model_file)
        instance.load_state_dict(tensors, device=device)
        return instance
