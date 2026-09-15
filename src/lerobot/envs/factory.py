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
from __future__ import annotations

from typing import Any

import gymnasium as gym

from .configs import EnvConfig, HubEnvConfig
from .utils import _call_make_env, _download_hub_file, _import_hub_module, _normalize_hub_result


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    try:
        cls = EnvConfig.get_choice_class(env_type)
    except KeyError as err:
        raise ValueError(
            f"Environment type '{env_type}' is not registered. "
            f"Available: {list(EnvConfig.get_known_choices().keys())}"
        ) from err
    return cls(**kwargs)


def make_env_pre_post_processors(
    env_cfg: EnvConfig,
    policy_cfg: Any,
) -> tuple[Any, Any]:
    """
    为环境观测创建预处理器和后处理器流水线。

    返回一个 (preprocessor, postprocessor) 元组。默认情况下委托给
    ``env_cfg.get_env_processors()``。XVLAConfig 的策略特定重写
    保留在这里，因为它依赖的是*策略*配置，而不是环境配置。
    """
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig

    if isinstance(policy_cfg, XVLAConfig):
        from lerobot.policies.xvla.processor_xvla import make_xvla_libero_pre_post_processors

        return make_xvla_libero_pre_post_processors()

    return env_cfg.get_env_processors()


def make_env(
    cfg: EnvConfig | str,
    n_envs: int = 1,
    use_async_envs: bool = False,
    hub_cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """根据配置或 Hub 引用创建 gym 向量化环境。

    Args:
        cfg (EnvConfig | str): 可以是一个描述要在本地构建的环境的 `EnvConfig` 对象，
            也可以是一个 Hugging Face Hub 仓库标识符（例如 `"username/repo"`）。在后一种情况下，
            该仓库必须包含一个 Python 文件（通常是 `env.py`）。
        n_envs (int, optional): 要返回的并行环境数量。默认为 1。
        use_async_envs (bool, optional): 返回 AsyncVectorEnv 还是 SyncVectorEnv。默认为
            False。
        hub_cache_dir (str | None): 已下载 hub 文件的可选缓存路径。
        trust_remote_code (bool): **明确同意**执行来自 Hub 的远程代码。
            默认 False —— 必须设为 True 才能导入/执行 hub 的 `env.py`。
    Raises:
        ValueError: 当 n_envs < 1 时
        ModuleNotFoundError: 当所请求的 env 包未安装时

    Returns:
        dict[str, dict[int, gym.vector.VectorEnv]]:
            从套件名称到按索引排列的向量化环境的映射。
            - 对于多任务基准（例如 LIBERO）：每个套件一个条目，每个 task_id 一个向量化环境。
            - 对于单任务环境：单个套件条目（cfg.type），task_id=0。

    """
    # 如果用户传入的是 hub id 字符串（例如 "username/repo"、"username/repo@main:env.py"）
    # 简化处理：仅支持 hub 提供的 `make_env`
    # TODO: (jadechoghari): 弃用字符串 API 并移除此检查
    if isinstance(cfg, str):
        hub_path: str | None = cfg
    elif isinstance(cfg, HubEnvConfig):
        hub_path = cfg.hub_path
    else:
        hub_path = None

    # 如果设置了 hub_path，则下载并调用 hub 提供的 `make_env`
    if hub_path:
        # 当 trust_remote_code 为 False 时，_download_hub_file 会抛出同样的 RuntimeError
        repo_id, file_path, local_file, revision = _download_hub_file(
            hub_path, trust_remote_code, hub_cache_dir
        )

        # 导入并呈现清晰的导入错误
        module = _import_hub_module(local_file, repo_id)

        # 调用 hub 提供的 make_env
        env_cfg = None if isinstance(cfg, str) else cfg
        raw_result = _call_make_env(module, n_envs=n_envs, use_async_envs=use_async_envs, cfg=env_cfg)

        # 将返回值规范化为 {suite: {task_id: vec_env}}
        return _normalize_hub_result(raw_result)

    # 此时 cfg 必须是 EnvConfig（而不是字符串），否则 hub_path 早已被设置
    if isinstance(cfg, str):
        raise TypeError("cfg should be an EnvConfig at this point")

    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")

    return cfg.create_envs(n_envs=n_envs, use_async_envs=use_async_envs)
