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
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import metaworld
import metaworld.policies as policies
import numpy as np
from gymnasium import spaces

from lerobot.lerobot_types import RobotObservation

from .utils import _LazyAsyncVectorEnv

# ---- 从外部 JSON 文件加载配置数据 ----
CONFIG_PATH = Path(__file__).parent / "metaworld_config.json"
try:
    with open(CONFIG_PATH) as f:
        data = json.load(f)
except FileNotFoundError as err:
    raise FileNotFoundError(
        "Could not find 'metaworld_config.json'. "
        "Please ensure the configuration file is in the same directory as the script."
    ) from err
except json.JSONDecodeError as err:
    raise ValueError(
        "Failed to decode 'metaworld_config.json'. Please ensure it is a valid JSON file."
    ) from err

# ---- 处理已加载的数据 ----

# 提取顶层字典并进行类型检查
task_descriptions_obj = data.get("TASK_DESCRIPTIONS")
if not isinstance(task_descriptions_obj, dict):
    raise TypeError("Expected TASK_DESCRIPTIONS to be a dict[str, str]")
TASK_DESCRIPTIONS: dict[str, str] = task_descriptions_obj

task_name_to_id_obj = data.get("TASK_NAME_TO_ID")
if not isinstance(task_name_to_id_obj, dict):
    raise TypeError("Expected TASK_NAME_TO_ID to be a dict[str, int]")
TASK_NAME_TO_ID: dict[str, int] = task_name_to_id_obj

# difficulty -> tasks 映射
difficulty_to_tasks = data.get("DIFFICULTY_TO_TASKS")
if not isinstance(difficulty_to_tasks, dict):
    raise TypeError("Expected 'DIFFICULTY_TO_TASKS' to be a dict[str, list[str]]")
DIFFICULTY_TO_TASKS: dict[str, list[str]] = difficulty_to_tasks

# 将策略字符串转换为实际的策略类
task_policy_mapping = data.get("TASK_POLICY_MAPPING")
if not isinstance(task_policy_mapping, dict):
    raise TypeError("Expected 'TASK_POLICY_MAPPING' to be a dict[str, str]")
TASK_POLICY_MAPPING: dict[str, Any] = {
    task_name: getattr(policies, policy_class_name)
    for task_name, policy_class_name in task_policy_mapping.items()
}
ACTION_DIM = 4
OBS_DIM = 4


class MetaworldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task,
        camera_name="corner2",
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=480,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task.replace("metaworld-", "")
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.camera_name = camera_name

        self._env_name = self.task  # 上面已去掉 "metaworld-" 前缀
        self._env = None  # 延迟创建 —— 在 worker 子进程内首次 reset() 时创建
        self._max_episode_steps = 500  # MT1 环境的 max_path_length 始终为 500
        self.task_description = TASK_DESCRIPTIONS[self.task]

        self.expert_policy = TASK_POLICY_MAPPING[self.task]()

        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(OBS_DIM,),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(ACTION_DIM,), dtype=np.float32)

    def _ensure_env(self) -> None:
        """在首次使用时创建底层的 MetaWorld 环境。

        在 fork() 之后于 worker 子进程内调用，这样每个 worker 都能获得
        自己干净的渲染上下文，而不是从父进程继承一个过期的上下文
        （那会在 AsyncVectorEnv 中导致崩溃）。
        """
        if self._env is not None:
            return
        mt1 = metaworld.MT1(self._env_name, seed=42)
        env = mt1.train_classes[self._env_name](render_mode="rgb_array", camera_name=self.camera_name)
        env.set_task(mt1.train_tasks[0])
        if self.camera_name == "corner2":
            env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        env.reset()
        env._freeze_rand_vec = False  # 否则没有随机化
        env.seeded_rand_vec = True  # 使用带种子的 RNG，使 reset(seed=X) 能控制物体位置
        self._env = env

    def render(self) -> np.ndarray:
        """
        渲染当前环境帧。

        Returns:
            np.ndarray: 从环境渲染出的 RGB 图像。
        """
        self._ensure_env()
        image = self._env.render()
        if self.camera_name == "corner2":
            # 来自该相机的图像是翻转的 —— 进行纠正
            image = np.flip(image, (0, 1))
        return image

    def _format_raw_obs(self, raw_obs: np.ndarray) -> RobotObservation:
        image = None
        if self._env is not None:
            image = self._env.render()
            if self.camera_name == "corner2":
                # 注意：MetaWorld 环境中的 "corner2" 相机输出的图像在两个轴上都是反转的。
                image = np.flip(image, (0, 1))
        agent_pos = raw_obs[:4]
        if self.obs_type == "state":
            raise NotImplementedError(
                "'state' obs_type not implemented for MetaWorld. Use pixel modes instead."
            )

        elif self.obs_type in ("pixels", "pixels_agent_pos"):
            assert image is not None, (
                "Expected `image` to be rendered before constructing pixel-based observations. "
                "This likely means `env.render()` returned None or the environment was not provided."
            )

            if self.obs_type == "pixels":
                obs = {"pixels": image.copy()}

            else:  # pixels_agent_pos
                obs = {
                    "pixels": image.copy(),
                    "agent_pos": agent_pos,
                }
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")
        return obs

    def reset(
        self,
        seed: int | None = None,
        **kwargs,
    ) -> tuple[RobotObservation, dict[str, Any]]:
        """
        将环境重置为初始状态。

        Args:
            seed (Optional[int]): 用于环境初始化的随机种子。

        Returns:
            observation (RobotObservation): 格式化后的初始观测。
            info (Dict[str, Any]): 关于重置状态的附加信息。
        """
        self._ensure_env()
        super().reset(seed=seed)

        if seed is not None:
            self._env.seed(seed)
        raw_obs, info = self._env.reset(seed=seed)

        observation = self._format_raw_obs(raw_obs)

        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        """
        执行一步环境交互。

        Args:
            action (np.ndarray): 要执行的动作，必须是形状为 (action_dim,) 的一维数组。

        Returns:
            observation (RobotObservation): 该步之后格式化后的观测。
            reward (float): 该步的标量奖励。
            terminated (bool): episode 是否成功终止。
            truncated (bool): episode 是否因时间限制而被截断。
            info (Dict[str, Any]): 附加的环境信息。
        """
        self._ensure_env()
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_obs, reward, done, truncated, info = self._env.step(action)

        # 判断任务是否成功
        is_success = bool(info.get("success", 0))
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "done": done,
                "is_success": is_success,
            }
        )

        # 将原始观测格式化为期望的结构
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


# ---- Main API ----------------------------------------------------------------


def create_metaworld_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """
    创建具有统一返回结构的向量化 Meta-World 环境。

    Returns:
        dict[task_group][task_id] -> vec_env（env_cls([...])，恰好包含 n_envs 个工厂）
    Notes:
        - n_envs 是*每个任务*的 rollout 数量（episode_index = 0..n_envs-1）。
        - `task` 可以是单个难度组（例如 "easy"、"medium"、"hard"），也可以是以逗号分隔的列表。
        - 如果任务名不在 DIFFICULTY_TO_TASKS 中，则将其视为单个自定义任务。
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_groups = [t.strip() for t in task.split(",") if t.strip()]
    if not task_groups:
        raise ValueError("`task` must contain at least one Meta-World task or difficulty group.")

    print(f"Creating Meta-World envs | task_groups={task_groups} | n_envs(per task)={n_envs}")

    is_async = env_cls is gym.vector.AsyncVectorEnv
    cached_obs_space = None
    cached_act_space = None
    cached_metadata = None
    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for group in task_groups:
        # 如果不在难度预设中，则将其视为单个自定义任务
        tasks = DIFFICULTY_TO_TASKS.get(group, [group])

        for tid, task_name in enumerate(tasks):
            print(f"Building vec env | group={group} | task_id={tid} | task={task_name}")

            # 构建 n_envs 个工厂
            fns = [(lambda tn=task_name: MetaworldEnv(task=tn, **gym_kwargs)) for _ in range(n_envs)]

            if is_async:
                lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
                if cached_obs_space is None:
                    cached_obs_space = lazy.observation_space
                    cached_act_space = lazy.action_space
                    cached_metadata = lazy.metadata
                out[group][tid] = lazy
            else:
                out[group][tid] = env_cls(fns)

    # 为保持一致，返回普通 dict
    return {group: dict(task_map) for group, task_map in out.items()}
