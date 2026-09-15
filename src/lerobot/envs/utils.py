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
import importlib.util
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import singledispatch
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torch import Tensor

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.utils import get_channel_first_image_shape

from .configs import EnvConfig


def parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """将 ``camera_name`` 规范化为非空的字符串列表。

    接受逗号分隔的字符串（``"cam_a,cam_b"``）或字符串序列
    （元组/列表）。会去除空白；空条目会被丢弃。
    对于不支持的输入类型抛出 ``TypeError``，
    当规范化后的列表为空时抛出 ``ValueError``。
    """
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list | tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def _convert_nested_dict(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _convert_nested_dict(v)
        elif isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v)
        else:
            result[k] = v
    return result


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # TODO(jadechoghari, imstevenpmwork): 重构此处以使用环境提供的 features（不要硬编码）
    """将环境观测转换为 LeRobot 格式的观测。
    Args:
        observation: 来自 Gym 向量化环境的观测批次字典。
    Returns:
        观测批次字典，其键已重命名为 LeRobot 格式，值为张量。
    """
    # 映射为策略所期望的输入
    return_observations = {}
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            imgs = {f"{OBS_IMAGES}.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {OBS_IMAGE: observations["pixels"]}

        for imgkey, img in imgs.items():
            # TODO(aliberts, rcadene): 使用 transforms.ToTensor()？
            img_tensor = torch.from_numpy(img)

            # 在非向量化环境中预处理观测时，我们需要添加一个批次维度。
            # human-in-the-loop RL 就是这种情况，那里只有一个环境。
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
            # 合理性检查：图像是通道在最后（channel last）
            _, h, w, c = img_tensor.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img_tensor.shape=}"

            # 合理性检查：图像是 uint8
            assert img_tensor.dtype == torch.uint8, f"expect torch.uint8, but instead {img_tensor.dtype=}"

            # 转换为通道在前（channel first）、float32 类型、范围 [0,1]
            img_tensor = einops.rearrange(img_tensor, "b h w c -> b c h w").contiguous()
            img_tensor = img_tensor.type(torch.float32)
            img_tensor /= 255

            return_observations[imgkey] = img_tensor

    if "environment_state" in observations:
        env_state = torch.from_numpy(observations["environment_state"]).float()
        if env_state.dim() == 1:
            env_state = env_state.unsqueeze(0)

        return_observations[OBS_ENV_STATE] = env_state

    if "agent_pos" in observations:
        agent_pos = torch.from_numpy(observations["agent_pos"]).float()
        if agent_pos.dim() == 1:
            agent_pos = agent_pos.unsqueeze(0)
        return_observations[OBS_STATE] = agent_pos

    if "robot_state" in observations:
        return_observations[f"{OBS_STR}.robot_state"] = _convert_nested_dict(observations["robot_state"])

    # 处理 IsaacLab Arena 格式：观测包含 'policy' 和 'camera_obs' 键
    if "policy" in observations:
        return_observations[f"{OBS_STR}.policy"] = observations["policy"]

    if "camera_obs" in observations:
        return_observations[f"{OBS_STR}.camera_obs"] = observations["camera_obs"]

    # 透传上面尚未处理的任何剩余 ndarray/tensor 键，
    # 这样环境插件就可以通过 get_env_processors() 暴露额外的观测键。
    _handled = {"pixels", "environment_state", "agent_pos", "robot_state", "policy", "camera_obs"}
    for key, value in observations.items():
        if key in _handled:
            continue
        target = f"{OBS_STR}.{key}"
        if target in return_observations:
            continue
        if isinstance(value, np.ndarray):
            val = torch.from_numpy(value).float()
            if val.dim() == 1:
                val = val.unsqueeze(0)
            return_observations[target] = val
        elif isinstance(value, Tensor):
            val = value.float()
            if val.dim() == 1:
                val = val.unsqueeze(0)
            return_observations[target] = val

    return return_observations


def env_to_policy_features(env_cfg: EnvConfig) -> dict[str, PolicyFeature]:
    # TODO(jadechoghari, imstevenpmwork): 移除这种键的硬编码，直接使用嵌套键原样传递
    # （还需要重构 preprocess_observation，并将归一化从策略中外置出来）
    policy_features = {}
    for key, ft in env_cfg.features.items():
        if ft.type is FeatureType.VISUAL:
            if len(ft.shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={ft.shape})")

            shape = get_channel_first_image_shape(ft.shape)
            feature = PolicyFeature(type=ft.type, shape=shape)
        else:
            feature = ft

        policy_key = env_cfg.features_map[key]
        policy_features[policy_key] = feature

    return policy_features


def _sub_env_has_attr(env: gym.vector.VectorEnv, attr: str) -> bool:
    try:
        env.get_attr(attr)
        return True
    except (AttributeError, Exception):
        return False


# 由 `rollout()` 在 `reset(options=...)` 中传入，用于标记新 rollout 的开始。
# FreezeAfterEpisodeEnd 仅在收到此选项时解冻，因此 Gymnasium 无参数的自动重置
# 不会被误认为真正的新 episode。
NEW_ROLLOUT_OPTION = "lerobot_new_rollout"


class FreezeAfterEpisodeEnd(gym.Wrapper):
    """一旦某个子环境的 episode 结束，就停止执行模拟器工作。

    `rollout()` 在 `done` 被锁存的情况下运行 `while not np.all(done)`，因此提前
    终止的子环境仍会持续被 step —— 包括物理模拟和离屏渲染 ——
    直到批次中最慢的子环境结束。批次会运行
    `max(episode_lengths)` 次迭代，去完成只需要
    `mean(episode_lengths)` 的工作。

    本包装器缓存终止时的 transition，并在后续任何 `step()` 或自动重置时
    重放它，因此已完成的子环境不再产生任何开销。rollout 本来就会忽略
    这些 transition。

    冻结状态特意在 Gymnasium 的自动重置后依然保持。在
    `AutoresetMode.NEXT_STEP` 下，向量化环境会在下一步重置已终止的子环境，
    并让它跑完整个额外的 episode —— 由于 `done` 保持锁存，
    rollout 会丢弃这个 episode。吸收掉那次重置是节省开销的大头。

    只有携带 `NEW_ROLLOUT_OPTION` 的显式重置才能解冻，因此该信号是
    显式的而非推断出来的：Gymnasium 的自动重置调用 `reset()` 时不带
    参数，但传入 `seeds=None` 的调用者也是如此，若混淆两者会导致
    某个环境在整个 rollout 期间一直处于冻结状态。

    `AutoresetMode.DISABLED` 在这里不是替代方案 —— Gymnasium 断言在该模式下
    已终止的环境绝不会被 step，因此这个包装器永远不会被触发。
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._frozen: tuple | None = None

    def reset(self, *, seed=None, options=None):
        if self._frozen is not None and not (options or {}).get(NEW_ROLLOUT_OPTION):
            # Gymnasium 对 rollout 已经完成的子环境进行自动重置。
            # 重放终止时的观测，而不是重建模拟。
            obs, _, _, _, info = self._frozen
            return obs, info
        self._frozen = None
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        if self._frozen is not None:
            return self._frozen
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            # 重放时将奖励置零，这样即使调用者对填充的尾部累加奖励，
            # 冻结的子环境也不会虚增回报。
            self._frozen = (obs, 0.0, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    @property
    def is_frozen(self) -> bool:
        return self._frozen is not None


def freeze_after_episode_end(env_fn: Callable[[], gym.Env]) -> Callable[[], gym.Env]:
    """包装一个环境工厂，使构建出的环境在其 episode 结束后冻结。"""

    def _fn() -> gym.Env:
        return FreezeAfterEpisodeEnd(env_fn())

    return _fn


class _LazyAsyncVectorEnv:
    """将 AsyncVectorEnv 的创建推迟到首次使用时。

    预先创建所有任务的 AsyncVectorEnv 会派生 N_tasks × n_envs 个 worker
    进程，它们全都会立即分配 EGL/GPU 资源。由于任务是
    顺序评估的，同一时间只需要一个任务的 worker 存活。
    本包装器保存工厂函数，并在首次 reset()/step()/call() 时创建真正的
    AsyncVectorEnv，使进程数峰值保持为 n_envs。
    """

    def __init__(
        self,
        env_fns: list[Callable],
        observation_space=None,
        action_space=None,
        metadata=None,
    ):
        self._env_fns = env_fns
        self._env: gym.vector.AsyncVectorEnv | None = None
        self.num_envs = len(env_fns)
        if observation_space is not None and action_space is not None and metadata is not None:
            self.observation_space = observation_space
            self.action_space = action_space
            self.metadata = metadata
        else:
            tmp = env_fns[0]()
            self.observation_space = tmp.observation_space
            self.action_space = tmp.action_space
            self.metadata = tmp.metadata
            tmp.close()
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    def _ensure(self) -> None:
        if self._env is None:
            self._env = gym.vector.AsyncVectorEnv(
                [freeze_after_episode_end(fn) for fn in self._env_fns],
                context="forkserver",
                shared_memory=True,
                autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
            )

    @property
    def unwrapped(self):
        return self

    def reset(self, **kwargs):
        self._ensure()
        return self._env.reset(**kwargs)

    def step(self, actions):
        self._ensure()
        return self._env.step(actions)

    def call(self, name, *args, **kwargs):
        self._ensure()
        return self._env.call(name, *args, **kwargs)

    def get_attr(self, name):
        self._ensure()
        return self._env.get_attr(name)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def check_env_attributes_and_types(env: gym.vector.VectorEnv) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("once", UserWarning)

        if not (_sub_env_has_attr(env, "task_description") and _sub_env_has_attr(env, "task")):
            warnings.warn(
                "The environment does not have 'task_description' and 'task'. Some policies require these features.",
                UserWarning,
                stacklevel=2,
            )


def _close_single_env(env: Any) -> None:
    try:
        env.close()
    except Exception as exc:
        print(f"Exception while closing env {env}: {exc}")


@singledispatch
def close_envs(obj: Any) -> None:
    """默认：当类型无法识别时抛出异常。"""
    raise NotImplementedError(f"close_envs not implemented for type {type(obj).__name__}")


@close_envs.register
def _(env: Mapping) -> None:
    for v in env.values():
        if isinstance(v, Mapping):
            close_envs(v)
        elif hasattr(v, "close"):
            _close_single_env(v)


@close_envs.register
def _(envs: Sequence) -> None:
    if isinstance(envs, (str | bytes)):
        return
    for v in envs:
        if isinstance(v, Mapping) or isinstance(v, Sequence) and not isinstance(v, (str | bytes)):
            close_envs(v)
        elif hasattr(v, "close"):
            _close_single_env(v)


@close_envs.register
def _(env: gym.Env) -> None:
    _close_single_env(env)


# 辅助函数：安全地将 python 文件作为模块加载
def _load_module_from_path(path: str, module_name: str | None = None):
    module_name = module_name or f"hub_env_{os.path.basename(path).replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Could not load module spec for {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


# 辅助函数：解析 hub 字符串（支持 "user/repo"、"user/repo@rev"，可选路径）
# 示例：
#   "user/repo" -> 将在仓库根目录查找 env.py
#   "user/repo@main:envs/my_env.py" -> 显式指定 revision 和路径
def _parse_hub_url(hub_uri: str):
    # 非常小的解析器：[repo_id][@revision][:path]
    # repo_id 是必需的（user/repo 或 org/repo）
    revision = None
    file_path = "env.py"
    if "@" in hub_uri:
        repo_and_rev, *rest = hub_uri.split(":", 1)
        repo_id, rev = repo_and_rev.split("@", 1)
        revision = rev
        if rest:
            file_path = rest[0]
    else:
        repo_id, *rest = hub_uri.split(":", 1)
        if rest:
            file_path = rest[0]
    return repo_id, revision, file_path


def _download_hub_file(
    cfg_str: str,
    trust_remote_code: bool,
    hub_cache_dir: str | None,
) -> tuple[str, str, str, str]:
    """
    解析 `cfg_str`（hub URL），强制执行 `trust_remote_code` 检查，并返回
    (repo_id, file_path, local_file, revision)。
    """
    if not trust_remote_code:
        raise RuntimeError(
            f"Refusing to execute remote code from the Hub for '{cfg_str}'. "
            "Executing hub env modules runs arbitrary Python code from third-party repositories. "
            "If you trust this repo and understand the risks, call `make_env(..., trust_remote_code=True)` "
            "and prefer pinning to a specific revision: 'user/repo@<commit-hash>:env.py'."
        )

    repo_id, revision, file_path = _parse_hub_url(cfg_str)

    try:
        local_file = hf_hub_download(
            repo_id=repo_id, filename=file_path, revision=revision, cache_dir=hub_cache_dir
        )
    except Exception as e:
        # 回退到 snapshot 下载
        snapshot_dir = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=hub_cache_dir)
        local_file = os.path.join(snapshot_dir, file_path)
        if not os.path.exists(local_file):
            raise FileNotFoundError(
                f"Could not find {file_path} in repository {repo_id}@{revision or 'main'}"
            ) from e

    return repo_id, file_path, local_file, revision


def _import_hub_module(local_file: str, repo_id: str) -> Any:
    """
    将已下载的文件作为模块导入，并呈现有用的导入错误信息。
    """
    module_name = f"hub_env_{repo_id.replace('/', '_')}"
    try:
        module = _load_module_from_path(local_file, module_name=module_name)
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise ModuleNotFoundError(
            f"Hub env '{repo_id}:{os.path.basename(local_file)}' failed to import because the dependency "
            f"'{missing}' is not installed locally.\n\n"
        ) from e
    except ImportError as e:
        raise ImportError(
            f"Failed to load hub env module '{repo_id}:{os.path.basename(local_file)}'. Import error: {e}\n\n"
        ) from e
    return module


def _call_make_env(module: Any, n_envs: int, use_async_envs: bool, cfg: EnvConfig | None) -> Any:
    """
    确保模块暴露了 make_env 并调用它。
    """
    if not hasattr(module, "make_env"):
        raise AttributeError(
            f"The hub module {getattr(module, '__name__', 'hub_module')} must expose `make_env(n_envs=int, use_async_envs=bool)`."
        )
    entry_fn = module.make_env
    # 仅当 cfg 不为 None 时才传入（即提供的是 EnvConfig 而非字符串形式的 hub ID 时）
    if cfg is not None:
        return entry_fn(n_envs=n_envs, use_async_envs=use_async_envs, cfg=cfg)
    else:
        return entry_fn(n_envs=n_envs, use_async_envs=use_async_envs)


def _normalize_hub_result(result: Any) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """
    将 hub `make_env` 可能的返回类型规范化为如下映射：
      { suite_name: { task_id: vector_env } }
    接受：
      - dict（假定已经正确）
      - gym.vector.VectorEnv
      - gym.Env（会被包装为 SyncVectorEnv）
    """
    if isinstance(result, dict):
        return result

    # VectorEnv：如果可用则使用其 spec.id
    if isinstance(result, gym.vector.VectorEnv):
        suite_name = getattr(result, "spec", None) and getattr(result.spec, "id", None) or "hub_env"
        return {suite_name: {0: result}}

    # 单个 Env：包装为 SyncVectorEnv
    if isinstance(result, gym.Env):
        vec = gym.vector.SyncVectorEnv([lambda: result])
        suite_name = getattr(result, "spec", None) and getattr(result.spec, "id", None) or "hub_env"
        return {suite_name: {0: vec}}

    raise ValueError(
        "Hub `make_env` must return either a mapping {suite: {task_id: vec_env}}, "
        "a gym.vector.VectorEnv, or a single gym.Env."
    )
