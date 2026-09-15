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

import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.lerobot_types import RobotObservation

from .utils import _LazyAsyncVectorEnv, parse_camera_names


def _get_suite(name: str) -> benchmark.Benchmark:
    """按名称实例化 LIBERO 套件，并带有清晰的校验。"""
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    """校验/规范化 task id。若为 None → 选择所有任务。"""
    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids


# LIBERO-plus 的扰动变体将扰动信息编码在文件名中，
# 但磁盘上只存在基础的 `.pruned_init` —— 去掉后缀以匹配
# LIBERO-plus 自己的 suite.get_task_init_states()（我们在这里重新实现它，
# 以便为 PyTorch 2.6+ 的 numpy pickle 传入 weights_only=False）。
_LIBERO_PERTURBATION_SUFFIX_RE = re.compile(r"_(?:language|view|light)_[^.]*|_(?:table|tb)_\d+")


def get_task_init_states(task_suite: Any, i: int, is_libero_plus: bool = False) -> np.ndarray:
    task = task_suite.tasks[i]
    filename = Path(task.init_states_file)
    root = Path(get_libero_path("init_states"))

    if not is_libero_plus:
        init_states_path = root / task.problem_folder / filename.name
        return torch.load(init_states_path, weights_only=False)  # nosec B614

    # LIBERO-plus：`_add_` / `_level` 变体将额外物体布局以一维数组的形式
    # 存储在 libero_newobj/ 下，必须将其 reshape 为 (1, -1)。
    if "_add_" in filename.name or "_level" in filename.name:
        init_states_path = root / "libero_newobj" / task.problem_folder / filename.name
        init_states = torch.load(init_states_path, weights_only=False)  # nosec B614
        return init_states.reshape(1, -1)

    # LIBERO-plus 的扰动变体将扰动信息编码在文件名中，
    # 但磁盘上只存在基础的 `.pruned_init` —— 去掉后缀以匹配。
    stripped = _LIBERO_PERTURBATION_SUFFIX_RE.sub("", filename.stem) + filename.suffix
    init_states_path = root / task.problem_folder / stripped
    return torch.load(init_states_path, weights_only=False)  # nosec B614


def get_libero_dummy_action():
    """获取虚拟/空操作动作，用于在机器人不做任何动作时推进模拟。"""
    return [0, 0, 0, 0, 0, 0, -1]


ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,  # 最长的训练 demo 有 193 步
    "libero_object": 280,  # 最长的训练 demo 有 254 步
    "libero_goal": 300,  # 最长的训练 demo 有 270 步
    "libero_10": 520,  # 最长的训练 demo 有 505 步
    "libero_90": 400,  # 最长的训练 demo 有 373 步
}


class LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        episode_length: int | None = None,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = True,
        episode_index: int = 0,
        n_envs: int = 1,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        control_freq: int = 20,
        control_mode: str = "relative",
        is_libero_plus: bool = False,
        hard_reset: bool = True,
    ):
        super().__init__()
        if control_freq <= 0:
            raise ValueError(f"control_freq must be positive, got {control_freq}")
        if not hard_reset and not init_states:
            raise ValueError("hard_reset=False requires init_states=True")
        self.task_id = task_id
        self.is_libero_plus = is_libero_plus
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = parse_camera_names(
            camera_name
        )  # agentview_image（主视角）或 robot0_eye_in_hand_image（腕部）

        # 将原始相机名称映射为 "image1" 和 "image2"。
        # 随后预处理步骤 `preprocess_observation` 会按照 LeRobot 约定
        # 为它们加上 `.images.*` 前缀（例如 `observation.images.image`、`observation.images.image2`）。
        # 这确保无论原始相机如何命名，策略都能一致地接收到
        # 期望格式的观测。
        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",
            }
        self.camera_name_mapping = camera_name_mapping
        self.num_steps_wait = num_steps_wait
        self.control_freq = control_freq
        self.hard_reset = hard_reset
        self.episode_index = episode_index
        self.episode_length = episode_length
        # 加载一次并保留
        self._init_states = (
            get_task_init_states(task_suite, self.task_id, is_libero_plus=self.is_libero_plus)
            if self.init_states
            else None
        )
        self._reset_stride = n_envs  # 执行重置时，将 `_reset_stride` 累加到 `init_state_id` 上。

        self.init_state_id = self.episode_index  # 将每个子环境绑定到一个固定的初始状态

        # 在不分配 GPU 资源的情况下提取任务元数据（在 fork 之前是安全的）。
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        self._task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        self._env: OffScreenRenderEnv | None = (
            None  # 延迟创建 —— 在 worker 子进程内首次 reset() 时创建
        )

        default_steps = 500
        self._max_episode_steps = (
            TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)
            if self.episode_length is None
            else self.episode_length
        )
        self.control_mode = control_mode
        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )

        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "robot_state": spaces.Dict(
                        {
                            "eef": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                                    "quat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                                    ),
                                    "mat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64
                                    ),
                                }
                            ),
                            "gripper": spaces.Dict(
                                {
                                    "qpos": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                    "qvel": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                }
                            ),
                            "joints": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                    "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                }
                            ),
                        }
                    ),
                }
            )

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def _ensure_env(self) -> None:
        """在首次使用时创建底层的 OffScreenRenderEnv。

        在 fork() 之后于 worker 子进程内调用，这样每个 worker 都能获得
        自己干净的 EGL 上下文，而不是从父进程继承一个过期的上下文
        （那会在 AsyncVectorEnv 中导致 EGL_BAD_CONTEXT 崩溃）。
        """
        if self._env is not None:
            return
        env = OffScreenRenderEnv(
            bddl_file_name=self._task_bddl_file,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            control_freq=self.control_freq,
            # 软重置会跳过 LIBERO 的模型和渲染器重建。它需要显式启用，
            # 因为稳定步（settle steps）可能使其观测与硬重置不同。
            hard_reset=self.hard_reset,
        )
        env.reset()
        self._env = env

    def render(self):
        self._ensure_env()
        raw_obs = self._env.env._get_observations()
        pixels = self._format_raw_obs(raw_obs)["pixels"]
        image = next(iter(pixels.values()))
        image = image[::-1, ::-1]  # 为可视化同时翻转 H 和 W
        return image

    def _format_raw_obs(self, raw_obs: RobotObservation) -> RobotObservation:
        assert self._env is not None, "_format_raw_obs called before _ensure_env()"
        images = {}
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            images[self.camera_name_mapping[camera_name]] = image

        eef_pos = raw_obs.get("robot0_eef_pos")
        eef_quat = raw_obs.get("robot0_eef_quat")

        # 来自控制器的旋转矩阵
        eef_mat = self._env.robots[0].controller.ee_ori_mat if eef_pos is not None else None
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        gripper_qvel = raw_obs.get("robot0_gripper_qvel")
        joint_pos = raw_obs.get("robot0_joint_pos")
        joint_vel = raw_obs.get("robot0_joint_vel")
        obs = {
            "pixels": images,
            "robot_state": {
                "eef": {
                    "pos": eef_pos,  # (3,)
                    "quat": eef_quat,  # (4,)
                    "mat": eef_mat,  # (3, 3)
                },
                "gripper": {
                    "qpos": gripper_qpos,  # (2,)
                    "qvel": gripper_qvel,  # (2,)
                },
                "joints": {
                    "pos": joint_pos,  # (7,)
                    "vel": joint_vel,  # (7,)
                },
            },
        }
        if self.obs_type == "pixels":
            return {"pixels": images.copy()}

        if self.obs_type == "pixels_agent_pos":
            # 校验必需字段是否存在
            if eef_pos is None or eef_quat is None or gripper_qpos is None:
                raise ValueError(
                    f"Missing required robot state fields in raw observation. "
                    f"Got eef_pos={eef_pos is not None}, eef_quat={eef_quat is not None}, "
                    f"gripper_qpos={gripper_qpos is not None}"
                )
            return obs

        raise NotImplementedError(
            f"The observation type '{self.obs_type}' is not supported in LiberoEnv. "
            "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
        )

    def reset(self, seed=None, **kwargs):
        self._ensure_env()
        super().reset(seed=seed)
        self._env.seed(seed)
        raw_obs = self._env.reset()
        if self.init_states and self._init_states is not None:
            raw_obs = self._env.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
            self.init_state_id += self._reset_stride  # 重置时更改 init_state_id

        # 重置后，物体可能不稳定（轻微漂浮、相互穿插等）。
        # 用空操作动作让模拟器运行几帧，使一切稳定下来。
        # 增大该值可以提高各次重置之间的确定性和可复现性。
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())

        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
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
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": done,
                "is_success": is_success,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        # 原样返回终止时的观测。终止后的重置由调用方负责；
        # 下面创建的向量化环境使用 NEXT_STEP 自动重置。因此在这里重置
        # 会导致重置两次，并跳过一个初始状态。
        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            finally:
                # LIBERO 在 close 时会删除其内部环境，因此该包装器
                # 必须在下次重置之前重新创建。
                self._env = None


def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
    control_mode: str,
    camera_name_mapping: dict[str, str] | None = None,
    is_libero_plus: bool = False,
) -> list[Callable[[], LiberoEnv]]:
    """为单个 (suite, task_id) 构建 n_envs 个工厂可调用对象。"""

    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        local_kwargs = dict(kwargs)
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_length=episode_length,
            episode_index=episode_index,
            n_envs=n_envs,
            control_mode=control_mode,
            camera_name_mapping=camera_name_mapping,
            is_libero_plus=is_libero_plus,
            **local_kwargs,
        )

    fns: list[Callable[[], LiberoEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


# ---- Main API ----------------------------------------------------------------


def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    control_mode: str = "relative",
    episode_length: int | None = None,
    camera_name_mapping: dict[str, str] | None = None,
    is_libero_plus: bool = False,
) -> dict[str, dict[int, Any]]:
    """
    创建具有统一返回结构的向量化 LIBERO 环境。

    Returns:
        dict[suite_name][task_id] -> vec_env（env_cls([...])，恰好包含 n_envs 个工厂）
    Notes:
        - n_envs 是*每个任务*的 rollout 数量（episode_index = 0..n_envs-1）。
        - `task` 可以是单个套件，也可以是以逗号分隔的套件列表。
        - 可以在 `gym_kwargs` 中传入 `task_ids`（list[int]）来限定每个套件的任务。
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # 可选：限定到特定任务

    camera_names = parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    print(
        f"Creating LIBERO envs | suites={suite_names} | n_envs(per task)={n_envs} | init_states={init_states}"
    )
    if task_ids_filter is not None:
        print(f"Restricting to task_ids={task_ids_filter}")

    is_async = env_cls is gym.vector.AsyncVectorEnv
    is_sync = env_cls is gym.vector.SyncVectorEnv

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        total = len(suite.tasks)
        selected = _select_task_ids(total, task_ids_filter)
        if not selected:
            raise ValueError(f"No tasks selected for suite '{suite_name}' (available: {total}).")

        # 同一套件中的所有任务共享相同的观测/动作空间。
        # 探测一次并复用，避免为每个任务创建临时环境。
        cached_obs_space: spaces.Space | None = None
        cached_act_space: spaces.Space | None = None
        cached_metadata: dict[str, Any] | None = None

        for tid in selected:
            fns = _make_env_fns(
                suite=suite,
                episode_length=episode_length,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
                control_mode=control_mode,
                camera_name_mapping=camera_name_mapping,
                is_libero_plus=is_libero_plus,
            )
            if is_async:
                lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
                if cached_obs_space is None:
                    cached_obs_space = lazy.observation_space
                    cached_act_space = lazy.action_space
                    cached_metadata = lazy.metadata
                out[suite_name][tid] = lazy
            elif is_sync:
                out[suite_name][tid] = gym.vector.SyncVectorEnv(
                    fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
                )
            else:
                out[suite_name][tid] = env_cls(fns)
            print(f"Built vec env | suite={suite_name} | task_id={tid} | n_envs={n_envs}")

    return {suite: dict(task_map) for suite, task_map in out.items()}
