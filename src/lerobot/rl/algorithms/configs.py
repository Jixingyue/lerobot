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
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.utils.hub import HubMixin

T = TypeVar("T", bound="RLAlgorithmConfig")

logger = logging.getLogger(__name__)


@dataclass
class TrainingStats:
    """由 ``algorithm.update()`` 返回，用于日志记录和检查点保存。"""

    losses: dict[str, float] = field(default_factory=dict)
    grad_norms: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, float]:
        """将所有统计信息扁平化为单个字典，用于日志记录。"""

        d: dict[str, float] = {}
        for name, val in self.losses.items():
            d[name] = val
        for name, val in self.grad_norms.items():
            d[f"{name}_grad_norm"] = val
        for name, val in self.extra.items():
            d[name] = val
        return d


@dataclass
class RLAlgorithmConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    """算法配置的注册表。"""

    @property
    def type(self) -> str:
        """此算法配置的注册名称（例如 ``"sac"``）。"""
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @classmethod
    @abc.abstractmethod
    def from_policy_config(cls, policy_cfg: Any) -> RLAlgorithmConfig:
        """从策略配置构建算法配置。

        每个注册的配置子类都必须重写此方法。
        """
        raise NotImplementedError(f"{cls.__name__} must implement from_policy_config()")

    def _save_pretrained(self, save_directory: Path) -> None:
        """将此配置序列化为 ``save_directory`` 内的 ``config.json``。"""
        # 按基类进行编码，以便 draccus 包含 choice "type" 键，
        # `from_pretrained` 需要该键来解析具体的子类。
        with open(save_directory / CONFIG_NAME, "w") as f:
            json.dump(draccus.encode(self, RLAlgorithmConfig), f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **algo_kwargs: Any,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.error(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
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
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        with draccus.config_type("json"):
            instance = draccus.parse(RLAlgorithmConfig, config_file, args=[])

        if cls is not RLAlgorithmConfig and not isinstance(instance, cls):
            raise TypeError(
                f"Config at {model_id} has type '{instance.type}' but was loaded via "
                f"{cls.__name__}; use the matching subclass or RLAlgorithmConfig.from_pretrained()."
            )

        for key, value in algo_kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance
