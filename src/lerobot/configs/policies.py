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
import abc
import builtins
import json
import os
import tempfile
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.optim import LRSchedulerConfig, OptimizerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from lerobot.utils.hub import HubMixin

from .types import FeatureType, PolicyFeature

T = TypeVar("T", bound="PreTrainedConfig")
logger = getLogger(__name__)


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):  # type: ignore[misc,name-defined] #TODO: draccus 问题
    """
    策略模型的基础配置类。

    Args:
        n_obs_steps: 要传递给策略的观测所覆盖的环境步数（包含当前步以及
            向前回溯的若干步）。
        input_features: 定义策略输入数据的 PolicyFeature 的字典。键表示
            输入数据的名称，值是 PolicyFeature，由 FeatureType 和 shape 属性组成。
        output_features: 定义策略输出数据的 PolicyFeature 的字典。键表示
            输出数据的名称，值是 PolicyFeature，由 FeatureType 和 shape 属性组成。
        normalization_mapping: 将 FeatureType 的字符串值（例如 "STATE"、"VISUAL"）映射到
            对应的 NormalizationMode（例如 NormalizationMode.MIN_MAX）的字典
    """

    n_obs_steps: int = 1

    # 可以将 `input_features` 设置为 None/null，以便从数据集中推断这些值。
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    device: str | None = None  # 例如 "cuda"、"cuda:0"、"cpu" 或 "mps"
    # `use_amp` 决定训练和评估时是否使用自动混合精度（AMP）。使用 AMP 时，
    # 会启用自动梯度缩放。
    use_amp: bool = False

    # 策略训练时是否使用了 PEFT。
    use_peft: bool = False

    push_to_hub: bool = True  # type: ignore[assignment] # TODO: 使用不同的名字以避免覆盖
    repo_id: str | None = None

    # 上传到 Hugging Face hub 上的私有仓库。
    private: bool | None = None
    # 为 hub 上的策略添加标签。
    tags: list[str] | None = None
    # 为 hub 上的策略添加标签。
    license: str | None = None
    # 托管在 Hub 上的模型的仓库 ID，或包含使用 `Policy.save_pretrained` 保存的
    # 权重的目录路径。如果未提供，策略将从头初始化。
    pretrained_path: Path | None = None
    # 可选的 Hub revision（commit hash、分支或标签），用于固定预训练模型的版本。
    pretrained_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # 必要时自动停用 AMP
        if self.use_amp and not is_amp_available(self.device):
            logger.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None:  # type: ignore[type-arg] #TODO: 暂无实现
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: 暂无实现
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: 暂无实现
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        # 针对基类进行编码，这样 draccus 会包含 choice 的 "type" 键，
        # `from_pretrained` 需要它来解析具体的子类。
        with open(save_directory / CONFIG_NAME, "w") as f:
            json.dump(draccus.encode(self, PreTrainedConfig), f, indent=4)

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
        **policy_kwargs: Any,
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

        with open(config_file) as f:
            config = json.load(f)

        # 从序列化的 "type" 标签解析出具体的配置子类，然后直接针对该类
        # 解析配置（带 CLI 覆盖项）。"type" 键会被剥离，因为 draccus
        # 只在解析注册表基类时才会消费它。
        policy_type = config.pop("type", None)
        if policy_type is None:
            raise ValueError(f"Missing 'type' field in {CONFIG_NAME} of {model_id}")
        try:
            config_cls = cls.get_choice_class(policy_type)
        except Exception as e:
            raise ValueError(
                f"Policy type '{policy_type}' (from {CONFIG_NAME} of {model_id}) is not registered. "
                f"Available policy types: {cls.get_known_choices()}"
            ) from e

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            config_file = f.name

        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        with draccus.config_type("json"):
            return draccus.parse(config_cls, config_file, args=cli_overrides)
