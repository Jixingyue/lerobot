#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.lerobot_types import RobotObservation

from .utils import _LazyAsyncVectorEnv, parse_camera_names

logger = logging.getLogger(__name__)

# LeRobot 包装器所使用的扁平动作/状态向量的维度。
# 它们对应 RoboCasa365 中的 PandaOmron 机器人。
OBS_STATE_DIM = 16  # base_pos(3) + base_quat(4) + ee_pos_rel(3) + ee_quat_rel(4) + gripper_qpos(2)
ACTION_DIM = 12  # base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# 默认的 PandaOmron 相机。我们直接将这些原始名称以
# `observation.images.<name>` 的形式暴露出来，使 LeRobot 数据集/策略的键
# 与 RoboCasa 的原生约定一致（不做隐式重命名）。
DEFAULT_CAMERAS = [
    "robot0_agentview_left",
    "robot0_eye_in_hand",
    "robot0_agentview_right",
]

# 用于采样的物体网格注册表。RoboCasa 的上游默认值为
# ("objaverse", "lightwheel")，但 objaverse 包非常大（约 30GB），
# 大多数用户 —— 包括我们的 CI 镜像 —— 只下载 lightwheel 包
# （`download_kitchen_assets` 中的 `--type objs_lw`）。当某个被采样的物体
# 类别在所有注册表中都没有候选项时，robocasa 会崩溃并抛出
# `ValueError: Probabilities contain NaN`（概率归一化时出现 0/0 除法）。
# 将范围限定到磁盘上实际存在的注册表，
# 可以避免 NaN，并与资源下载所提供的内容保持一致。
DEFAULT_OBJ_REGISTRIES: tuple[str, ...] = ("lightwheel",)

# 可作为 `--env.task` 接受的任务组快捷方式。当用户传入这些
# 名称之一时，我们会将其展开为上游的 RoboCasa 任务列表，并自动设置
# 数据集划分。单个任务名（可选地以逗号分隔）仍然
# 优先；这仅在组名完全匹配时才会触发。
_TASK_GROUP_SPLITS = {
    "atomic_seen": "target",
    "composite_seen": "target",
    "composite_unseen": "target",
    "pretrain50": "pretrain",
    "pretrain100": "pretrain",
    "pretrain200": "pretrain",
    "pretrain300": "pretrain",
}


def _resolve_tasks(task: str) -> tuple[list[str], str | None]:
    """将 `--env.task` 的值解析为 (task_names, split_override)。

    如果 `task` 是已知的任务组名（例如 `atomic_seen`、`pretrain100`），
    则通过 `robocasa.utils.dataset_registry.{TARGET,PRETRAINING}_TASKS`
    将其展开，并返回对应的划分。否则将 `task` 视为单个任务或
    以逗号分隔的列表，并保持划分不变（None）。
    """
    key = task.strip()
    if key in _TASK_GROUP_SPLITS:
        from robocasa.utils.dataset_registry import PRETRAINING_TASKS, TARGET_TASKS

        combined = {**TARGET_TASKS, **PRETRAINING_TASKS}
        if key not in combined:
            raise ValueError(
                f"Task group '{key}' is not available in this version of robocasa. "
                f"Known groups: {sorted(combined.keys())}."
            )
        return list(combined[key]), _TASK_GROUP_SPLITS[key]

    names = [t.strip() for t in task.split(",") if t.strip()]
    if not names:
        raise ValueError("`task` must contain at least one RoboCasa task name.")
    return names, None


def _get_task_horizon(task: str) -> int:
    """返回 RoboCasa 为某个任务注册的 rollout 时长。"""
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    try:
        return int(get_task_horizon(task))
    except ValueError as exc:
        raise ValueError(
            f"No RoboCasa horizon is registered for task '{task}'. "
            "Set `--env.episode_length=<steps>` explicitly."
        ) from exc


def convert_action(flat_action: np.ndarray) -> dict[str, Any]:
    """将扁平的 (12,) 动作向量拆分为 RoboCasa 动作字典。

    布局：base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1)
    """
    return {
        "action.base_motion": flat_action[0:4],
        "action.control_mode": flat_action[4:5],
        "action.end_effector_position": flat_action[5:8],
        "action.end_effector_rotation": flat_action[8:11],
        "action.gripper_close": flat_action[11:12],
    }


class RoboCasaEnv(gym.Env):
    """用于 RoboCasa365 厨房环境的 LeRobot gym.Env 包装器。

    包装 robocasa 包中的 RoboCasaGymEnv，并将其基于字典的
    观测和动作转换为 LeRobot 所期望的扁平数组。
    原始的 RoboCasa 相机名称会以 `pixels/<cam>` 的形式原样保留。
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str,
        camera_name: str | Sequence[str] = ",".join(DEFAULT_CAMERAS),
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 512,
        visualization_height: int = 512,
        split: str | None = None,
        episode_length: int | None = None,
        obj_registries: Sequence[str] = DEFAULT_OBJ_REGISTRIES,
        episode_index: int = 0,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.split = split
        self.obj_registries = tuple(obj_registries)
        # 每个 worker 的索引（0..n_envs-1），用于将用户提供的种子
        # 分散到各个工厂，使每个子环境探索不同的布局，
        # 即使向 `reset()` 传入的是相同的种子。
        self.episode_index = int(episode_index)

        self.camera_name = parse_camera_names(camera_name)

        self._max_episode_steps = episode_length if episode_length is not None else _get_task_horizon(task)

        # 延迟创建 —— 在 worker 子进程内首次 reset() 时创建，
        # 以避免在 fork() 后继承过期的 GPU/EGL 上下文。
        self._env: Any = None
        self.task_description = ""

        images = {
            cam: spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
            for cam in self.camera_name
        }

        if self.obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": spaces.Dict(images)})
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "agent_pos": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(OBS_STATE_DIM,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported obs_type '{self.obs_type}'. Use 'pixels' or 'pixels_agent_pos'.")

        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

    def _ensure_env(self) -> None:
        """在首次使用时创建底层的 RoboCasaGymEnv。

        在 fork() 之后于 worker 子进程内调用，这样每个 worker 都能获得
        自己干净的渲染上下文，而不是从父进程继承一个过期的上下文
        （那会在 AsyncVectorEnv 中导致崩溃）。
        """
        if self._env is not None:
            return
        from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

        # RoboCasaGymEnv 默认 split="test"，而 create_env 会拒绝该值
        # （只有 None/"all"/"pretrain"/"target" 是有效的）。始终传入
        # 有效的值，以免命中那个默认值。额外的 kwargs 会
        # 通过 create_env/robosuite.make 转发给底层的厨房环境。
        self._env = RoboCasaGymEnv(
            env_name=self.task,
            camera_widths=self.observation_width,
            camera_heights=self.observation_height,
            split=self.split if self.split is not None else "all",
            obj_registries=self.obj_registries,
        )

        ep_meta = self._env.env.get_ep_meta()
        self.task_description = ep_meta.get("lang", self.task)

    def _format_raw_obs(self, raw_obs: dict) -> RobotObservation:
        """将 RoboCasaGymEnv 的观测字典转换为 LeRobot 格式。"""
        # RoboCasaGymEnv 在 "video.<cam>" 键下输出相机帧。
        images = {cam: raw_obs[f"video.{cam}"] for cam in self.camera_name if f"video.{cam}" in raw_obs}

        if self.obs_type == "pixels":
            return {"pixels": images}

        # `state.*` 键来自包装器内部的 PandaOmronKeyConverter。
        agent_pos = np.concatenate(
            [
                raw_obs.get("state.base_position", np.zeros(3)),
                raw_obs.get("state.base_rotation", np.zeros(4)),
                raw_obs.get("state.end_effector_position_relative", np.zeros(3)),
                raw_obs.get("state.end_effector_rotation_relative", np.zeros(4)),
                raw_obs.get("state.gripper_qpos", np.zeros(2)),
            ],
            axis=-1,
        ).astype(np.float32)

        return {"pixels": images, "agent_pos": agent_pos}

    def render(self) -> np.ndarray:
        self._ensure_env()
        assert self._env is not None
        return self._env.render()

    def reset(self, seed=None, **kwargs):
        self._ensure_env()
        assert self._env is not None
        super().reset(seed=seed)
        # 将种子分散到各个 worker，使 n_envs 个工厂不会全都
        # 生成相同的场景。当用户显式提供种子时，我们按
        # episode_index 偏移；没有种子时则回退到 episode_index，
        # 这样每个 worker 仍然是彼此不同的，而不是继承相同的
        # 全局 RNG 状态。
        worker_seed = seed + self.episode_index if seed is not None else self.episode_index
        raw_obs, info = self._env.reset(seed=worker_seed)

        ep_meta = self._env.env.get_ep_meta()
        self.task_description = ep_meta.get("lang", self.task)

        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        assert self._env is not None
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )

        action_dict = convert_action(action)
        raw_obs, reward, done, truncated, info = self._env.step(action_dict)

        is_success = bool(info.get("success", False))
        terminated = done or is_success
        info.update({"task": self.task, "done": done, "is_success": is_success})

        observation = self._format_raw_obs(raw_obs)
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "done": bool(done),
                "is_success": bool(is_success),
            }
            self.reset()

        return observation, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()


def _make_env_fns(
    *,
    task: str,
    n_envs: int,
    camera_names: list[str],
    obs_type: str,
    render_mode: str,
    observation_width: int,
    observation_height: int,
    visualization_width: int,
    visualization_height: int,
    split: str | None,
    episode_length: int | None,
    obj_registries: Sequence[str],
) -> list[Callable[[], RoboCasaEnv]]:
    """为单个任务构建 n_envs 个工厂可调用对象。

    每个工厂携带一个不同的 ``episode_index``（``0..n_envs-1``），
    这样 ``RoboCasaEnv.reset()`` 就能从用户提供的种子派生出
    每个 worker 的种子序列。
    """

    def _make_env(episode_index: int) -> RoboCasaEnv:
        return RoboCasaEnv(
            task=task,
            camera_name=camera_names,
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
            split=split,
            episode_length=episode_length,
            obj_registries=obj_registries,
            episode_index=episode_index,
        )

    return [partial(_make_env, i) for i in range(n_envs)]


def create_robocasa_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = ",".join(DEFAULT_CAMERAS),
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    episode_length: int | None = None,
    obj_registries: Sequence[str] = DEFAULT_OBJ_REGISTRIES,
) -> dict[str, dict[int, Any]]:
    """创建返回形状一致的向量化 RoboCasa365 环境。

    Returns:
        dict[task_name][task_id] -> vec_env（env_cls([...])，恰好包含 n_envs 个工厂）

    `task` 可以是：
      - 单个任务名（例如 `CloseFridge`）
      - 以逗号分隔的任务名列表（例如 `CloseFridge,PickPlaceCoffee`）
      - 基准测试组快捷方式（`atomic_seen`、`composite_seen`、
        `composite_unseen`、`pretrain50`、`pretrain100`、`pretrain200`、
        `pretrain300`），会自动展开为上游任务列表并自动设置数据集
        `split`（"target" 或 "pretrain"）。
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    obs_type = gym_kwargs.pop("obs_type", "pixels_agent_pos")
    render_mode = gym_kwargs.pop("render_mode", "rgb_array")
    observation_width = gym_kwargs.pop("observation_width", 256)
    observation_height = gym_kwargs.pop("observation_height", 256)
    visualization_width = gym_kwargs.pop("visualization_width", 512)
    visualization_height = gym_kwargs.pop("visualization_height", 512)
    split = gym_kwargs.pop("split", None)

    camera_names = parse_camera_names(camera_name)
    task_names, group_split = _resolve_tasks(str(task))
    if group_split is not None and split is None:
        split = group_split

    logger.info(
        "Creating RoboCasa envs | tasks=%s | split=%s | n_envs(per task)=%d",
        task_names,
        split,
        n_envs,
    )

    is_async = env_cls is gym.vector.AsyncVectorEnv

    cached_obs_space: spaces.Space | None = None
    cached_act_space: spaces.Space | None = None
    cached_metadata: dict[str, Any] | None = None
    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for task_name in task_names:
        fns = _make_env_fns(
            task=task_name,
            n_envs=n_envs,
            camera_names=camera_names,
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
            split=split,
            episode_length=episode_length,
            obj_registries=obj_registries,
        )

        if is_async:
            lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
            if cached_obs_space is None:
                cached_obs_space = lazy.observation_space
                cached_act_space = lazy.action_space
                cached_metadata = lazy.metadata
            out[task_name][0] = lazy
        else:
            out[task_name][0] = env_cls(fns)
        logger.info("Built vec env | task=%s | n_envs=%d", task_name, n_envs)

    return {name: dict(task_map) for name, task_map in out.items()}
