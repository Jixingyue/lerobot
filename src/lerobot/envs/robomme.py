"""用于 LeRobot 评估的 RoboMME 环境包装器。

将 RoboMME 的 ``BenchmarkEnvBuilder`` 包装为与 Gymnasium 兼容的
``VectorEnv``，适用于 ``lerobot_eval``。

RoboMME 任务：
  Counting（计数）:    BinFill, PickXtimes, SwingXtimes, StopCube
  Permanence（持久性）:  VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap
  Reference（参照）:   PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder
  Imitation（模仿）:   MoveCube, InsertPeg, PatternLock, RouteStick

数据集：lerobot/robomme（LeRobot v3.0，1,600 个 episode）
安装：见 docker/Dockerfile.benchmark.robomme（仅 Linux——mani-skill 与 numpy 的版本固定冲突）
基准测试：https://github.com/RoboMME/robomme_benchmark
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .utils import _LazyAsyncVectorEnv

ROBOMME_TASKS = [
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
]


class RoboMMEGymEnv(gym.Env):
    """对单个 RoboMME episode 环境的轻量 Gymnasium 包装。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        task: str = "PickXtimes",
        action_space_type: str = "joint_angle",
        dataset: str = "test",
        episode_idx: int = 0,
        max_steps: int = 300,
        front_camera_name: str = "camera1",
        wrist_camera_name: str = "camera2",
    ):
        super().__init__()
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        self._task = task
        self.task = task
        self.task_description = task
        self._action_space_type = action_space_type
        self._dataset = dataset
        self._episode_idx = episode_idx
        self._max_steps = max_steps
        self._max_episode_steps = max_steps
        self._front_camera_name = front_camera_name
        self._wrist_camera_name = wrist_camera_name

        self._builder = BenchmarkEnvBuilder(
            env_id=task,
            dataset=dataset,
            action_space=action_space_type,
            gui_render=False,
            max_steps=max_steps,
        )
        self._env = None
        self._last_raw_obs: dict | None = None

        action_dim = 8 if action_space_type == "joint_angle" else 7
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        # `pixels` 必须是嵌套的 Dict，这样 envs/utils.py 中的
        # `preprocess_observation()` 才能识别它，并将每个相机映射到
        # `observation.images.<cam>`。扁平布局（`pixels/image`、
        # `pixels/wrist_image`）会静默地从批次中丢弃所有图像。
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        front_camera_name: spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8),
                        wrist_camera_name: spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8),
                    }
                ),
                "agent_pos": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # 当 n_episodes > n_envs 时，一个包装器可能会被多次 reset。
        # 在替换之前先关闭之前的 SAPIEN 环境；否则每次
        # reset 都会保留 Vulkan 文件描述符和 fence 分配。
        self.close()
        self._env = self._builder.make_env_for_episode(
            episode_idx=self._episode_idx,
            max_steps=self._max_steps,
        )
        obs, info = self._env.reset()
        self._last_raw_obs = obs
        task_goal = info.get("task_goal")
        if isinstance(task_goal, list | tuple):
            task_goal = task_goal[0] if task_goal else ""
        self.task_description = str(task_goal or self._task)
        return self._convert_obs(obs), self._convert_info(info)

    def close(self):
        """立即释放底层的 ManiSkill/SAPIEN 环境。"""
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
        self._last_raw_obs = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._last_raw_obs = obs

        terminated_bool = bool(terminated.item()) if hasattr(terminated, "item") else bool(terminated)
        truncated_bool = bool(truncated.item()) if hasattr(truncated, "item") else bool(truncated)

        status = info.get("status", "ongoing")
        is_success = status == "success"
        conv_info = self._convert_info(info)
        conv_info["is_success"] = is_success

        return self._convert_obs(obs), float(reward), terminated_bool, truncated_bool, conv_info

    def render(self) -> np.ndarray | None:
        """返回上一次观测中的前置相机图像，用于视频录制。"""
        if self._last_raw_obs is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        front = self._last_raw_obs.get("front_rgb_list")
        if front is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        frame = front[-1] if isinstance(front, list) else front
        return np.asarray(frame, dtype=np.uint8)

    def _convert_obs(self, obs: dict) -> dict:
        front_rgb = (
            obs["front_rgb_list"][-1] if isinstance(obs["front_rgb_list"], list) else obs["front_rgb_list"]
        )
        wrist_rgb = (
            obs["wrist_rgb_list"][-1] if isinstance(obs["wrist_rgb_list"], list) else obs["wrist_rgb_list"]
        )
        joint_state = (
            obs["joint_state_list"][-1]
            if isinstance(obs["joint_state_list"], list)
            else obs["joint_state_list"]
        )
        gripper_state = (
            obs["gripper_state_list"][-1]
            if isinstance(obs["gripper_state_list"], list)
            else obs["gripper_state_list"]
        )

        front_rgb = np.asarray(front_rgb, dtype=np.uint8)
        wrist_rgb = np.asarray(wrist_rgb, dtype=np.uint8)
        joint = np.asarray(joint_state, dtype=np.float32).flatten()[:7]
        gripper = np.asarray(gripper_state, dtype=np.float32).flatten()[:1]
        state = np.concatenate([joint, gripper])

        front_camera_name = getattr(self, "_front_camera_name", "camera1")
        wrist_camera_name = getattr(self, "_wrist_camera_name", "camera2")
        return {
            "pixels": {front_camera_name: front_rgb, wrist_camera_name: wrist_rgb},
            "agent_pos": state,
        }

    def _convert_info(self, info: dict) -> dict:
        return {
            "status": info.get("status", "ongoing"),
            "task_goal": info.get("task_goal", ""),
        }


def _make_env_fns(
    *,
    task: str,
    n_envs: int,
    action_space_type: str,
    dataset: str,
    episode_length: int,
    task_id: int,
    front_camera_name: str,
    wrist_camera_name: str,
) -> list[Callable[[], RoboMMEGymEnv]]:
    """为一个 RoboMME 任务 id 构建 n_envs 个工厂可调用对象。"""

    def _make_one(episode_index: int) -> RoboMMEGymEnv:
        return RoboMMEGymEnv(
            task=task,
            action_space_type=action_space_type,
            dataset=dataset,
            episode_idx=episode_index,
            max_steps=episode_length,
            front_camera_name=front_camera_name,
            wrist_camera_name=wrist_camera_name,
        )

    return [partial(_make_one, task_id + i) for i in range(n_envs)]


def create_robomme_envs(
    task: str,
    n_envs: int = 1,
    action_space_type: str = "joint_angle",
    dataset: str = "test",
    episode_length: int = 300,
    task_ids: list[int] | None = None,
    front_camera_name: str = "camera1",
    wrist_camera_name: str = "camera2",
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """创建用于评估的向量化 RoboMME 环境。

    `task` 可以是单个 RoboMME 任务名（例如 "PickXtimes"），也可以是
    逗号分隔的列表（例如 "PickXtimes,BinFill,StopCube"）。每个任务
    在返回的映射中对应独立的测试套件。

    返回 {suite_name: {task_id: VectorEnv}}，与 lerobot 期望的格式一致。
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of env factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    if task_ids is None:
        task_ids = [0]

    task_names = [t.strip() for t in task.split(",") if t.strip()]
    is_async = env_cls is gym.vector.AsyncVectorEnv
    cached_obs_space: spaces.Space | None = None
    cached_act_space: spaces.Space | None = None
    cached_metadata: dict[str, Any] | None = None
    out: dict[str, dict[int, gym.vector.VectorEnv]] = {}
    for task_name in task_names:
        envs_by_task: dict[int, gym.vector.VectorEnv] = {}
        for task_id in task_ids:
            fns = _make_env_fns(
                task=task_name,
                n_envs=n_envs,
                action_space_type=action_space_type,
                dataset=dataset,
                episode_length=episode_length,
                task_id=task_id,
                front_camera_name=front_camera_name,
                wrist_camera_name=wrist_camera_name,
            )
            if is_async:
                lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
                if cached_obs_space is None:
                    cached_obs_space = lazy.observation_space
                    cached_act_space = lazy.action_space
                    cached_metadata = lazy.metadata
                envs_by_task[task_id] = lazy
            else:
                envs_by_task[task_id] = env_cls(fns)
        out[task_name] = envs_by_task
    return out
