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
"""用于 LeRobot 的 VLABench 环境包装器。

VLABench 是一个面向语言条件机器人操作、具备长程推理能力的
大规模基准测试，基于 MuJoCo/dm_control 构建。

- 论文：https://arxiv.org/abs/2412.18194
- GitHub：https://github.com/OpenMOSS/VLABench
- 网站：https://vlabench.github.io
"""

from __future__ import annotations

import contextlib
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from lerobot.lerobot_types import RobotObservation

from .utils import _LazyAsyncVectorEnv

logger = logging.getLogger(__name__)

ACTION_DIM = 7  # 位置(3) + 欧拉角(3) + 夹爪(1)
ACTION_LOW = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

# 每种任务类型的默认最大 episode 步数
DEFAULT_MAX_EPISODE_STEPS = 500

# VLABench 任务套件
PRIMITIVE_TASKS = [
    "select_fruit",
    "select_toy",
    "select_chemistry_tube",
    "add_condiment",
    "select_book",
    "select_painting",
    "select_drink",
    "insert_flower",
    "select_billiards",
    "select_ingredient",
    "select_mahjong",
    "select_poker",
    # Physical（物理）系列
    "density_qa",
    "friction_qa",
    "magnetism_qa",
    "reflection_qa",
    "simple_cuestick_usage",
    "simple_seesaw_usage",
    "sound_speed_qa",
    "thermal_expansion_qa",
    "weight_qa",
]

COMPOSITE_TASKS = [
    "cluster_billiards",
    "cluster_book",
    "cluster_drink",
    "cluster_toy",
    "cook_dishes",
    "cool_drink",
    "find_unseen_object",
    "get_coffee",
    "hammer_nail",
    "heat_food",
    "make_juice",
    "play_mahjong",
    "play_math_game",
    "play_poker",
    "play_snooker",
    "rearrange_book",
    "rearrange_chemistry_tube",
    "set_dining_table",
    "set_study_table",
    "store_food",
    "take_chemistry_experiment",
    "use_seesaw_complex",
]

SUITE_TASKS: dict[str, list[str]] = {
    "primitive": PRIMITIVE_TASKS,
    "composite": COMPOSITE_TASKS,
}


class VLABenchEnv(gym.Env):
    """VLABench 环境的 Gymnasium 包装器。

    将基于 dm_control 的 VLABench 模拟器包装在标准 gym.Env 接口之后。
    支持多个相机（前置、第二视角、腕部）和末端执行器控制。
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        task: str = "select_fruit",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        render_resolution: tuple[int, int] = (480, 480),
        robot: str = "franka",
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        action_mode: str = "eef",
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.render_resolution = render_resolution
        self.robot = robot
        self._max_episode_steps = max_episode_steps
        self.action_mode = action_mode

        # 延迟创建——在 worker 子进程内第一次 reset() 时创建，以避免
        # AsyncVectorEnv 派生 worker 时继承失效的 GPU/EGL 上下文。
        # 我们绝不缓存 `env.physics`：dm_control 将其暴露为 weakref
        # 代理，在 reset（重建仿真）后会失效，因此我们始终
        # 在调用点通过 `self._env.physics` 重新获取。
        self._env = None
        self.task_description = ""  # 在第一次 reset 时填充
        # 机器人基座链接在世界坐标系下的 XYZ，缓存起来。VLABench 数据集
        # 记录的 `observation.state` 位置和 `actions` 位置都处于
        # 机器人基座坐标系（见 VLABench/scripts/convert_to_lerobot.py，
        # 其中从 ee_pos 中减去了 `robot_frame_pos`）。机器人在每个
        # 任务中以固定偏移安装，因此每次环境构建后缓存一次是安全的。
        self._robot_base_xyz: np.ndarray | None = None

        h, w = self.render_resolution

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in VLABenchEnv. "
                "Please use 'pixels' or 'pixels_agent_pos'."
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                            "second_image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                            "wrist_image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                        }
                    ),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                            "second_image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                            "wrist_image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
                        }
                    ),
                    "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                }
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)

    # 当 MuJoCo 在 VLABench 的 20 步 reset
    # 热身期间抛出 `PhysicsError`（例如 mjWARN_BADQACC）时，
    # 重建底层环境的最大尝试次数。某些随机的任务/布局采样会落入
    # 不稳定的初始构型；重新采样布局几乎总能
    # 得到稳定的构型。少数上游任务（尤其是
    # `select_mahjong`）的布局采样器发散频率较高，
    # 需要远多于 5 次的重试，因此我们选取一个较宽松的上限。
    _ENSURE_ENV_MAX_ATTEMPTS = 20

    def _ensure_env(self) -> None:
        """首次使用时创建底层 VLABench 环境。

        在 fork() 之后的 worker 子进程内调用，因此每个 worker 都有
        自己干净的渲染上下文，而不是从父进程继承失效的
        上下文（继承会导致 AsyncVectorEnv 崩溃）。

        在遇到 `PhysicsError` 时重试：VLABench 的
        `LM4ManipDMEnv.reset()` 会在切换重力/流体的同时执行 20 次
        热身 `step()` 调用，以使场景稳定下来；对于某些随机布局，
        MuJoCo 的积分器会发散并抛出 `mjWARN_BADQACC`。
        重新采样布局几乎总能得到稳定的布局，因此我们会重试若干次
        再放弃。每次尝试之间，我们会用操作系统熵重新播种 NumPy 的
        全局 RNG，使上游任务采样器探索全新的初始状态——否则，当
        采样器在当前 RNG 状态下是确定性的时候，重试可能会
        重现同一个发散构型。
        """
        if self._env is not None:
            return

        import VLABench.robots  # noqa: F401  # type: ignore[import-untyped]
        import VLABench.tasks  # noqa: F401  # type: ignore[import-untyped]
        from dm_control.rl.control import PhysicsError  # type: ignore[import-untyped]
        from VLABench.envs import load_env  # type: ignore[import-untyped]

        h, w = self.render_resolution
        last_exc: PhysicsError | None = None
        for attempt in range(1, self._ENSURE_ENV_MAX_ATTEMPTS + 1):
            try:
                env = load_env(task=self.task, robot=self.robot, render_resolution=(h, w))
                self._env = env
                break
            except PhysicsError as exc:
                last_exc = exc
                logger.warning(
                    "PhysicsError on attempt %d/%d while building task '%s': %s. Retrying with fresh layout…",
                    attempt,
                    self._ENSURE_ENV_MAX_ATTEMPTS,
                    self.task,
                    exc,
                )
                np.random.seed(None)
        if self._env is None:
            assert last_exc is not None
            raise RuntimeError(
                f"VLABench task '{self.task}' failed to produce a stable "
                f"initial layout after {self._ENSURE_ENV_MAX_ATTEMPTS} "
                f"attempts. This task's upstream sampler diverges too "
                f"often for the configured robot; consider removing it "
                f"from the eval set. Last physics error: {last_exc}"
            ) from last_exc

        # 从 dm_control 任务中提取任务描述
        task_obj = self._env.task
        if hasattr(task_obj, "task_description"):
            self.task_description = task_obj.task_description
        elif hasattr(task_obj, "language_instruction"):
            self.task_description = task_obj.language_instruction
        else:
            self.task_description = self.task

        # 缓存机器人基座的世界坐标位置，使 `_build_ctrl_from_action` 和
        # `_get_obs` 可以在机器人坐标系（数据集）和
        # 世界坐标系（dm_control）之间转换，而无需每次都访问物理引擎。
        try:
            self._robot_base_xyz = np.asarray(self._env.get_robot_frame_position(), dtype=np.float64).reshape(
                3
            )
        except Exception:
            # 回退到 VLABench 默认的 Franka 基座位置。
            self._robot_base_xyz = np.array([0.0, -0.4, 0.78], dtype=np.float64)

    def _get_obs(self) -> dict:
        """从环境中获取当前观测。"""
        assert self._env is not None

        obs = self._env.get_observation()
        h, w = self.render_resolution

        def _to_hwc3(arr: np.ndarray) -> np.ndarray:
            """将任意相机数组强制转换为声明的 (h, w, 3) uint8 形状。"""
            a = np.asarray(arr)
            # 如果存在前导的单一批量维度，则丢弃。
            while a.ndim > 3 and a.shape[0] == 1:
                a = a[0]
            if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
                # CHW → HWC
                a = np.transpose(a, (1, 2, 0))
            if a.ndim == 2:
                a = np.stack([a] * 3, axis=-1)
            if a.ndim != 3:
                return np.zeros((h, w, 3), dtype=np.uint8)
            # 强制为 3 通道。
            if a.shape[-1] == 1:
                a = np.repeat(a, 3, axis=-1)
            elif a.shape[-1] == 4:
                a = a[..., :3]
            elif a.shape[-1] != 3:
                return np.zeros((h, w, 3), dtype=np.uint8)
            if a.shape[:2] != (h, w):
                a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
            return a.astype(np.uint8)

        # 提取相机图像——VLABench 返回 (n_cameras, C, H, W) 或单独的数组
        raw_frames: list[np.ndarray] = []
        if "rgb" in obs:
            rgb = obs["rgb"]
            if isinstance(rgb, np.ndarray):
                if rgb.ndim == 4:
                    raw_frames = [rgb[i] for i in range(rgb.shape[0])]
                elif rgb.ndim == 3:
                    raw_frames = [rgb]

        image_keys = ["image", "second_image", "wrist_image"]
        images: dict[str, np.ndarray] = {}
        for i, key in enumerate(image_keys):
            if i < len(raw_frames):
                images[key] = _to_hwc3(raw_frames[i])
            else:
                images[key] = np.zeros((h, w, 3), dtype=np.uint8)

        # 将 VLABench 原始的 ee_state `[pos_world(3), quat_wxyz(4), open(1)]`
        # 转换为数据集的 observation.state 布局 `[pos_robot(3), euler_xyz(3),
        # gripper(1)]`。见 VLABench/scripts/convert_to_lerobot.py——位置
        # 以机器人基座坐标系存储，朝向以 scipy 外旋
        # 'xyz' 欧拉角存储。
        raw = np.asarray(obs.get("ee_state", np.zeros(8)), dtype=np.float64).ravel()
        pos_world = raw[:3] if raw.size >= 3 else np.zeros(3, dtype=np.float64)
        quat_wxyz = raw[3:7] if raw.size >= 7 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        gripper = float(raw[7]) if raw.size >= 8 else 0.0

        base = self._robot_base_xyz if self._robot_base_xyz is not None else np.zeros(3, dtype=np.float64)
        pos_robot = pos_world - base
        euler_xyz = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_euler(
            "xyz", degrees=False
        )

        ee_state = np.concatenate([pos_robot, euler_xyz, [gripper]]).astype(np.float64)

        if self.obs_type == "pixels":
            return {"pixels": images}
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": images,
                "agent_pos": ee_state.astype(np.float64),
            }
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")

    # ---- 动作适配（EEF → 关节控制）--------------------------------
    #
    # HF vlabench 数据集记录的是 7 维动作
    # `[x, y, z（机器人坐标系）, rx, ry, rz（scipy 外旋 xyz）, gripper]`，
    # 与 VLABench 自己的评估流水线（evaluator.base）完全一致：
    #   pos, euler, g = policy(...)
    #   quat = euler_to_quaternion(*euler)      # extrinsic xyz -> wxyz
    #   _, qpos = robot.get_qpos_from_ee_pos(physics, pos=pos + base, quat=quat)
    #   env.step(np.concatenate([qpos, [g, g]]))
    #
    # VLABench 的 dm_control 任务会直接写入 `data.ctrl[:] = action`——对于
    # Franka 来说是 9 个条目（7 个手臂关节 + 2 个夹爪手指）。我们复刻
    # 上述转换，使策略的 EEF 命令能够真正驱动机器人。

    _FRANKA_FINGER_OPEN = 0.04  # 夹爪完全张开时的 qpos

    def _build_ctrl_from_action(self, action: np.ndarray, ctrl_dim: int) -> np.ndarray:
        """将 7 维 EEF 动作转换为大小为 `ctrl_dim` 的关节命令向量。

        对于默认的 Franka（ctrl_dim=9）：7 个手臂关节 qpos（通过 IK）+
        2 个夹爪手指 qpos（根据夹爪标量决定张开/闭合）。
        如果动作本身已是关节空间（形状与 ctrl_dim 匹配），则
        直接透传。
        """
        if action.shape[0] == ctrl_dim:
            return action.astype(np.float64, copy=False)

        if action.shape[0] != 7:
            # 未知布局——回退为零填充，以免仿真崩溃。
            padded: np.ndarray = np.zeros(ctrl_dim, dtype=np.float64)
            padded[: min(action.shape[0], ctrl_dim)] = action[:ctrl_dim]
            return padded

        from dm_control.utils.inverse_kinematics import qpos_from_site_pose

        # 动作位置处于机器人基座坐标系（见 convert_to_lerobot.py）；
        # dm_control 的 IK 需要世界坐标系下的目标。
        base = self._robot_base_xyz if self._robot_base_xyz is not None else np.zeros(3, dtype=np.float64)
        pos_world = np.asarray(action[:3], dtype=np.float64) + base
        rx, ry, rz = float(action[3]), float(action[4]), float(action[5])
        gripper = float(np.clip(action[6], 0.0, 1.0))

        # 数据集的欧拉角是 scipy 外旋 'xyz'（与 VLABench 的
        # `euler_to_quaternion` 相同）。scipy 输出 `[x, y, z, w]`；
        # dm_control 的 IK 和 MuJoCo 使用 `[w, x, y, z]`，因此需要重排。
        qxyzw = Rotation.from_euler("xyz", [rx, ry, rz], degrees=False).as_quat()
        quat = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]], dtype=np.float64)

        assert self._env is not None
        robot = self._env.task.robot
        site_name = robot.end_effector_site.full_identifier

        # inplace=False，使 IK 不会在步进中途修改物理状态——我们只
        # 需要解出的 qpos。获取一个新的 physics 句柄——缓存它可能会
        # 在 reset 后得到失效的 weakref。
        ik_result = qpos_from_site_pose(
            self._env.physics,
            site_name=site_name,
            target_pos=pos_world,
            target_quat=quat,
            inplace=False,
            max_steps=100,
        )
        n_dof = robot.n_dof  # Franka 为 7
        arm_qpos = ik_result.qpos[:n_dof]

        # 数据集夹爪约定：1 = 张开（手指 qpos = 0.04），
        # 0 = 闭合（手指 qpos = 0.0）。见 VLABench/scripts/convert_to_lerobot.py，
        # 其中 `trajectory[i][-1] > 0.03` 被编码为 `1`。
        finger_qpos = gripper * self._FRANKA_FINGER_OPEN

        ctrl = np.zeros(ctrl_dim, dtype=np.float64)
        ctrl[:n_dof] = arm_qpos
        # 剩余条目为夹爪手指（Franka 通常为 2 个）。
        ctrl[n_dof:] = finger_qpos
        return ctrl

    def reset(self, seed=None, **kwargs) -> tuple[RobotObservation, dict[str, Any]]:
        self._ensure_env()
        assert self._env is not None
        super().reset(seed=seed)

        if seed is not None:
            self._seed_inner_env(int(self.np_random.integers(0, 2**31 - 1)))

        self._env.reset()

        observation = self._get_obs()
        info = {"is_success": False}
        return observation, info

    def _seed_inner_env(self, seed: int) -> None:
        """将 `seed` 传播给内部 dm_control 环境。`Environment.reset()`
        不接受种子，因此我们直接为任务和环境的
        `RandomState` 重新播种。尽力而为：当某个 VLABench
        版本上不存在预期属性时会静默跳过。
        """
        for owner_attr, rng_attr in (("task", "random"), (None, "_random_state")):
            owner = getattr(self._env, owner_attr) if owner_attr else self._env
            rng = getattr(owner, rng_attr, None)
            rng_seed = getattr(rng, "seed", None)
            if callable(rng_seed):
                rng_seed(seed)

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        from dm_control.rl.control import PhysicsError  # type: ignore[import-untyped]

        self._ensure_env()
        assert self._env is not None

        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )

        if self.action_mode not in ("eef", "joint", "delta_eef"):
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        # 始终重新获取 physics——dm_control 返回的 weakref 代理可能
        # 在 reset 后失效。
        physics = self._env.physics
        ctrl_dim = int(physics.data.ctrl.shape[0])
        ctrl = self._build_ctrl_from_action(action, ctrl_dim)
        try:
            timestep = self._env.step(ctrl)
        except PhysicsError as exc:
            # 物理积分器发散（例如 mjWARN_BADQACC）。将其视为
            # 优雅的失败终止，而非硬性崩溃——
            # 多任务评估的其余部分仍应继续运行。
            logger.warning(
                "PhysicsError during step on task '%s': %s. Terminating episode.",
                self.task,
                exc,
            )
            observation = self._get_obs()
            info = {"task": self.task, "is_success": False, "physics_error": True}
            # 丢弃失效的环境，使下一次 reset() 能干净地重建它。
            with contextlib.suppress(Exception):
                self._env.close()
            self._env = None
            return observation, 0.0, True, False, info

        # 从 dm_control 的 timestep 中提取奖励
        reward = float(timestep.reward) if timestep.reward is not None else 0.0

        # 通过任务的终止条件检查成功
        is_success = False
        if hasattr(self._env, "task") and hasattr(self._env.task, "should_terminate_episode"):
            is_success = bool(self._env.task.should_terminate_episode(self._env.physics))

        terminated = is_success
        truncated = False
        info = {
            "task": self.task,
            "is_success": is_success,
        }

        observation = self._get_obs()

        if terminated:
            self.reset()

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        self._ensure_env()
        obs = self._get_obs()
        return obs["pixels"]["image"]

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


# ---- 主 API ----------------------------------------------------------------


def create_vlabench_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """
    创建向量化 VLABench 环境，返回形状保持一致。

    Returns:
        dict[suite_name][task_id] -> vec_env（env_cls([...])，恰好包含 n_envs 个工厂）

    Notes:
        - n_envs 是*每个任务*的 rollout 数量。
        - `task` 可以是套件名（"primitive"、"composite"）、逗号分隔的
          套件名列表，或单独的任务名（例如 "select_fruit,heat_food"）。
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_groups = [t.strip() for t in task.split(",") if t.strip()]
    if not task_groups:
        raise ValueError("`task` must contain at least one VLABench task or suite name.")

    logger.info(
        "Creating VLABench envs | task_groups=%s | n_envs(per task)=%d",
        task_groups,
        n_envs,
    )

    is_async = env_cls is gym.vector.AsyncVectorEnv
    cached_obs_space = None
    cached_act_space = None
    cached_metadata = None
    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for group in task_groups:
        # 检查它是否为套件名，否则视为单独的任务
        tasks = SUITE_TASKS.get(group, [group])

        for tid, task_name in enumerate(tasks):
            logger.info(
                "Building vec env | group=%s | task_id=%d | task=%s",
                group,
                tid,
                task_name,
            )

            fns = [(lambda tn=task_name: VLABenchEnv(task=tn, **gym_kwargs)) for _ in range(n_envs)]

            if is_async:
                lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
                if cached_obs_space is None:
                    cached_obs_space = lazy.observation_space
                    cached_act_space = lazy.action_space
                    cached_metadata = lazy.metadata
                out[group][tid] = lazy
            else:
                out[group][tid] = env_cls(fns)

    return {group: dict(task_map) for group, task_map in out.items()}
