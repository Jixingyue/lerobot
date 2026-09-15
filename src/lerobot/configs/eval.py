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

import datetime as dt
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

from lerobot import envs, policies  # noqa: F401

from . import parser
from .default import EvalConfig
from .policies import PreTrainedConfig

logger = getLogger(__name__)


@dataclass
class EvalPipelineConfig:
    # 托管在 Hub 上的模型的仓库 ID，或包含使用 `Policy.save_pretrained` 保存的
    # 权重的目录路径。如果未提供，策略将从头初始化（便于调试）。
    # 此参数与 `--config` 互斥。
    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    # 观测值的重命名映射，用于覆盖 image 和 state 的键名
    rename_map: dict[str, str] = field(default_factory=dict)
    # 显式同意执行来自 Hub 的远程代码（hub 环境所必需）。
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        # HACK: 这里再次解析 CLI 参数，以获取预训练路径（如果有的话）。
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            yaml_overrides = parser.get_yaml_overrides("policy")
            cli_overrides = parser.get_cli_overrides("policy") or []
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=yaml_overrides + cli_overrides
            )
            self.policy.pretrained_path = Path(policy_path)

        else:
            logger.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type if self.policy is not None else 'scratch'}"
            else:
                self.job_name = (
                    f"{self.env.type}_{self.policy.type if self.policy is not None else 'scratch'}"
                )

            logger.warning(f"No job name provided, using '{self.job_name}' as job name.")

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval") / eval_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """这使得解析器可以使用 `--policy.path=local/dir` 从策略加载配置"""
        return ["policy"]
