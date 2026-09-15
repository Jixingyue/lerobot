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

# 注意：gymnasium 目前是核心依赖，但未来可能会迁移为可选 extra。
# 当该迁移发生时，请取消下方守卫代码的注释，
# 并将 extra 名称更新为届时包含 gymnasium 的那个。
# from lerobot.utils.import_utils import require_package
# require_package("gymnasium", extra="<update_extra>", import_name="gymnasium")

from .configs import AlohaEnv, EnvConfig, HILSerlRobotEnvConfig, HubEnvConfig, PushtEnv
from .factory import make_env, make_env_config, make_env_pre_post_processors
from .utils import check_env_attributes_and_types, close_envs, env_to_policy_features, preprocess_observation

__all__ = [
    "AlohaEnv",
    "EnvConfig",
    "HILSerlRobotEnvConfig",
    "HubEnvConfig",
    "PushtEnv",
    "check_env_attributes_and_types",
    "close_envs",
    "env_to_policy_features",
    "make_env",
    "make_env_config",
    "make_env_pre_post_processors",
    "preprocess_observation",
]
