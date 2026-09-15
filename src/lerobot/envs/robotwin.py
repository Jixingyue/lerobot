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

import importlib
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from lerobot.lerobot_types import RobotObservation
from lerobot.utils.import_utils import _scipy_available

from .utils import _LazyAsyncVectorEnv

# scipy 仅用于末端位姿合成（``--env.action_mode=ee``）；对其加以保护，
# 以便在未安装 scipy 时也能导入本模块（以及模拟 RoboTwin 运行时的基础环境单元测试）。
if _scipy_available:
    from scipy.spatial.transform import Rotation
else:
    Rotation = None

logger = logging.getLogger(__name__)

# RoboTwin 2.0 使用的相机名称。包装器在查找
# get_obs() 输出中的键时会追加 "_rgb"（例如 "head_camera" → "head_camera_rgb"）。
ROBOTWIN_CAMERA_NAMES: tuple[str, ...] = (
    "head_camera",
    "left_camera",
    "right_camera",
)

ACTION_DIM = 14  # 7 自由度 × 2 条手臂（关节空间控制模式）
# 末端位姿控制模式：每条手臂 [x, y, z, qx, qy, qz, qw, gripper] = 8，双臂 = 16。
# 供世界模型策略（例如 LingBot-VA）使用，这类策略预测末端位姿增量，并通过 CuRobo IK 执行。
EEF_ACTION_DIM = 16
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
DEFAULT_EPISODE_LENGTH = 1200
OFFICIAL_INSTRUCTION_ENV = "LEROBOT_ROBOTWIN_OFFICIAL_INSTRUCTION"
OFFICIAL_INSTRUCTION_TYPE_ENV = "LEROBOT_ROBOTWIN_INSTRUCTION_TYPE"
OFFICIAL_INSTRUCTION_MAX_ENV = "LEROBOT_ROBOTWIN_INSTRUCTION_MAX"


def _compose_eef_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    """将单臂预测的增量位姿合成到初始位姿上。

    ``new_pose`` / ``init_pose`` 是 8 维向量
    ``[x, y, z, qx, qy, qz, qw, gripper]``。平移部分
    相加，旋转部分进行合成（``init_R * new_R``），夹爪取值来自
    预测结果。与上游 LingBot-VA RoboTwin 客户端中的
    ``add_eef_pose`` 保持一致。
    """
    new_r = Rotation.from_quat(new_pose[3:7])
    init_r = Rotation.from_quat(init_pose[3:7])
    out_rot = (init_r * new_r).as_quat()
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def _add_init_eef_pose(delta_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    """将双臂（16 维）预测的增量位姿合成到初始末端位姿上，并对四元数归一化。"""
    left = _compose_eef_pose(delta_pose[:8], init_pose[:8])
    right = _compose_eef_pose(delta_pose[8:], init_pose[8:])
    out = np.concatenate([left, right])
    # 按上游客户端的做法对两个四元数（索引 3:7 和 11:15）进行归一化。
    out[3:7] = out[3:7] / (np.linalg.norm(out[3:7]) + 1e-8)
    out[11:15] = out[11:15] / (np.linalg.norm(out[11:15]) + 1e-8)
    return out


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _arm_for_block(block: Any) -> str:
    return "left" if float(block.get_pose().p[0]) < 0 else "right"


def _robotwin_blocks_episode_info(task_name: str, env: Any) -> dict[str, str] | None:
    """推断 RoboTwin 官方指令生成器用于积木排序任务的 episode-info 字典。"""
    if task_name == "blocks_ranking_rgb":
        return {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": _arm_for_block(env.block1),
            "{b}": _arm_for_block(env.block2),
            "{c}": _arm_for_block(env.block3),
        }
    if task_name == "blocks_ranking_size":
        return {
            "{A}": "large block",
            "{B}": "medium block",
            "{C}": "small block",
            "{a}": _arm_for_block(env.block1),
            "{b}": _arm_for_block(env.block2),
            "{c}": _arm_for_block(env.block3),
        }
    return None


def _generate_robotwin_official_instruction(task_name: str, env: Any) -> str:
    """使用 RoboTwin 官方任务模板生成语言，与其评估客户端保持一致。"""
    fallback = task_name.replace("_", " ")
    episode_info = _robotwin_blocks_episode_info(task_name, env)
    if episode_info is None:
        logger.warning(
            "Official RoboTwin instruction is not implemented for task=%s; using %r.", task_name, fallback
        )
        return fallback

    try:
        # 属于 robotwin 模拟器仓库的一部分，由运行 robotwin 的 docker 镜像拉取
        # 见 https://github.com/RoboTwin-Platform/RoboTwin/tree/main/description
        # 用于生成官方指令
        from description.utils.generate_episode_instructions import generate_episode_descriptions
    except Exception:
        logger.warning(
            "Failed to import RoboTwin official instruction generator; using %r.", fallback, exc_info=True
        )
        return fallback

    instruction_type = os.environ.get(OFFICIAL_INSTRUCTION_TYPE_ENV, "seen")
    try:
        max_descriptions = int(os.environ.get(OFFICIAL_INSTRUCTION_MAX_ENV, "1000000"))
    except ValueError:
        max_descriptions = 1000000

    results = generate_episode_descriptions(task_name, [episode_info], max_descriptions=max_descriptions)
    if not results:
        logger.warning(
            "RoboTwin generated no official instructions for task=%s; using %r.", task_name, fallback
        )
        return fallback

    options = results[0].get(instruction_type) or results[0].get("seen") or results[0].get("unseen")
    if not options:
        logger.warning(
            "RoboTwin generated no %s official instructions for task=%s; using %r.",
            instruction_type,
            task_name,
            fallback,
        )
        return fallback

    return str(np.random.choice(options))


# D435 尺寸来自 task_config/_camera_config.yml（demo_clean.yml 选用的配置）。
DEFAULT_CAMERA_H = 240
DEFAULT_CAMERA_W = 320

# 任务列表来自 RoboTwin 2.0 的 `envs/` 目录——与上游完全一致
# （截至 main 分支为 50 个任务；更早的版本为 60 个，划分方式不同）。
# 请使用以下命令保持同步：
#   gh api /repos/RoboTwin-Platform/RoboTwin/contents/envs --paginate \
#     | jq -r '.[].name' | grep -E '\.py$' | grep -v '^_' | sed 's/\.py$//'
ROBOTWIN_TASKS: tuple[str, ...] = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)


_ROBOTWIN_SETUP_CACHE: dict[str, dict[str, Any]] = {}


def _load_robotwin_setup_kwargs(task_name: str) -> dict[str, Any]:
    """构建 RoboTwin 的 setup_demo 所需的 kwargs 字典。

    复刻 RoboTwin 的 ``script/eval_policy.py`` 所做的配置加载：
    读取 ``task_config/demo_clean.yml``，从
    ``_embodiment_config.yml`` 解析具身文件，加载机器人自己的
    ``config.yml``，并从 ``_camera_config.yml`` 读取相机尺寸。

    默认使用 ``aloha-agilex`` 单机器人双臂（这是 beat_block_hammer
    和大多数冒烟测试任务唯一使用的具身）。
    """
    if task_name in _ROBOTWIN_SETUP_CACHE:
        return dict(_ROBOTWIN_SETUP_CACHE[task_name])

    import os

    import yaml  # type: ignore[import-untyped]
    from envs import CONFIGS_PATH  # type: ignore[import-not-found]

    task_config = "demo_clean"
    with open(os.path.join(CONFIGS_PATH, f"{task_config}.yml"), encoding="utf-8") as f:
        args = yaml.safe_load(f)

    # 解析具身——demo_clean.yml 使用 [aloha-agilex]（双臂单机器人）
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)
    embodiment = args.get("embodiment", ["aloha-agilex"])
    if len(embodiment) == 1:
        robot_file = embodiment_types[embodiment[0]]["file_path"]
        args["left_robot_file"] = robot_file
        args["right_robot_file"] = robot_file
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiment_types[embodiment[0]]["file_path"]
        args["right_robot_file"] = embodiment_types[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"embodiment must have 1 or 3 items, got {len(embodiment)}")

    with open(os.path.join(args["left_robot_file"], "config.yml"), encoding="utf-8") as f:
        args["left_embodiment_config"] = yaml.safe_load(f)
    with open(os.path.join(args["right_robot_file"], "config.yml"), encoding="utf-8") as f:
        args["right_embodiment_config"] = yaml.safe_load(f)

    # 相机尺寸
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)
    head_cam = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_cam]["h"]
    args["head_camera_w"] = camera_config[head_cam]["w"]

    # 无头模式覆盖
    args["render_freq"] = 0
    args["task_name"] = task_name
    args["task_config"] = task_config

    _ROBOTWIN_SETUP_CACHE[task_name] = args
    return dict(args)


def _load_robotwin_task(task_name: str) -> type:
    """动态导入并返回一个 RoboTwin 2.0 任务类。

    RoboTwin 任务位于仓库根目录下的 ``envs/<task_name>.py``，
    安装后应位于 ``sys.path`` 上。
    """
    try:
        module = importlib.import_module(f"envs.{task_name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not import RoboTwin task '{task_name}'. "
            "Ensure RoboTwin 2.0 is installed and its 'envs/' directory is on PYTHONPATH. "
            "See the RoboTwin installation guide: https://robotwin-platform.github.io/doc/usage/robotwin-install.html"
        ) from e
    task_cls = getattr(module, task_name, None)
    if task_cls is None:
        raise AttributeError(f"Task class '{task_name}' not found in envs/{task_name}.py")
    return task_cls


class RoboTwinEnv(gym.Env):
    """围绕单个 RoboTwin 2.0 任务的 Gymnasium 包装器。

    RoboTwin 使用基于 SAPIEN 的自定义 API（``setup_demo`` / ``get_obs`` /
    ``take_action`` / ``check_success``），而非标准 gym 接口。
    本类将该 API 桥接到 Gymnasium，使 ``lerobot-eval`` 能够像驱动
    LIBERO 或 Meta-World 一样驱动 RoboTwin。

    底层 SAPIEN 环境在*工作进程内*第一次调用 ``reset()`` 时
    惰性创建。这是兼容
    ``gym.vector.AsyncVectorEnv`` 的必要条件：SAPIEN 分配的
    EGL/GPU 上下文不能从父进程 fork。

    观测
    ------------
    ``pixels`` 字典使用原始 RoboTwin 相机名作为键（例如
    ``"head_camera"``、``"left_camera"``）。``envs/utils.py`` 中的
    ``preprocess_observation`` 随后会将它们转换为
    ``observation.images.<cam>``。

    动作
    -------
    处于 ``[-1, 1]`` 的 14 维 float32 数组（关节空间，每条手臂 7 自由度）。

    自动求导
    --------
    ``setup_demo`` 和 ``take_action`` 会驱动 CuRobo 的 Newton 轨迹
    优化器，其内部会调用 ``cost.backward()``。lerobot_eval 使用
    ``torch.no_grad()`` 包裹 rollout，因此这两个调用点会重新启用梯度。
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        task_name: str,
        episode_index: int = 0,
        n_envs: int = 1,
        camera_names: Sequence[str] = ROBOTWIN_CAMERA_NAMES,
        observation_height: int | None = None,
        observation_width: int | None = None,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        render_mode: str = "rgb_array",
        action_mode: str = "joint",
    ):
        super().__init__()
        self.task_name = task_name
        self.task = task_name  # 供 utils.py 中的 add_envs_task() 使用
        self.task_description = task_name.replace("_", " ")
        self.episode_index = episode_index
        self._reset_stride = n_envs
        # "joint"：通过 take_action(action) 执行 14 维关节空间动作。"ee"：16 维末端位姿
        # 增量（叠加到 episode 的初始末端位姿上），通过 take_action(.., "ee") + IK 执行。
        if action_mode not in ("joint", "ee"):
            raise ValueError(f"action_mode must be 'joint' or 'ee'; got {action_mode!r}")
        self.action_mode = action_mode
        self._action_dim = EEF_ACTION_DIM if action_mode == "ee" else ACTION_DIM
        self._init_eef_pose: np.ndarray | None = None
        self.camera_names = list(camera_names)
        # 默认为 D435 尺寸（task_config/demo_clean.yml 内置的相机类型）。
        # 基于 YAML 的查找推迟到 reset()，以免构造时
        # 导入 RoboTwin 的 `envs` 模块——快速测试无需安装 RoboTwin。
        self.observation_height = observation_height or DEFAULT_CAMERA_H
        self.observation_width = observation_width or DEFAULT_CAMERA_W
        self.episode_length = episode_length
        self._max_episode_steps = episode_length  # lerobot_eval.rollout 会读取该值
        self.render_mode = render_mode

        self._env: Any | None = None  # 延迟创建——在 worker 内第一次 reset() 时创建
        self._step_count: int = 0
        self._black_frame: np.ndarray = np.zeros(
            (self.observation_height, self.observation_width, 3), dtype=np.uint8
        )

        image_spaces = {
            cam: spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
            for cam in self.camera_names
        }
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(image_spaces),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(ACTION_DIM,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(self._action_dim,), dtype=np.float32
        )

    def _ensure_env(self) -> None:
        """首次使用时创建 SAPIEN 环境。

        在 fork() 之后的 worker 子进程内调用，因此每个 worker 都有
        自己的 EGL/GPU 上下文，而不是从父进程继承失效的
        上下文（继承会导致 AsyncVectorEnv 崩溃）。
        """
        if self._env is not None:
            return
        task_cls = _load_robotwin_task(self.task_name)
        self._env = task_cls()

    def _get_obs(self) -> RobotObservation:
        assert self._env is not None, "_get_obs called before _ensure_env()"
        raw = self._env.get_obs()
        cameras_raw = raw.get("observation", {})

        images: dict[str, np.ndarray] = {}
        for cam in self.camera_names:
            cam_data = cameras_raw.get(cam)
            img = cam_data.get("rgb") if cam_data else None
            if img is None:
                images[cam] = self._black_frame
                continue
            img = np.asarray(img, dtype=np.uint8)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] != 3:
                img = img[..., :3]
            images[cam] = img

        ja = raw.get("joint_action") or {}
        vec = ja.get("vector")
        if vec is not None:
            arr = np.asarray(vec, dtype=np.float32).ravel()
            joint_state = (
                arr[:ACTION_DIM] if arr.size >= ACTION_DIM else np.zeros(ACTION_DIM, dtype=np.float32)
            )
        else:
            joint_state = np.zeros(ACTION_DIM, dtype=np.float32)

        return {"pixels": images, "agent_pos": joint_state}

    def _read_eef_pose(self) -> np.ndarray:
        """读取当前 16 维双臂末端位姿 [左臂(xyz+quat)+grip，右臂(xyz+quat)+grip]。"""
        assert self._env is not None, "_read_eef_pose called before _ensure_env()"
        ep = self._env.get_obs()["endpose"]
        pose = (
            list(ep["left_endpose"])
            + [ep["left_gripper"]]
            + list(ep["right_endpose"])
            + [ep["right_gripper"]]
        )
        return np.asarray(pose, dtype=np.float64)

    def reset(self, seed: int | None = None, **kwargs) -> tuple[RobotObservation, dict]:
        self._ensure_env()
        super().reset(seed=seed)
        assert self._env is not None  # set by _ensure_env() above

        actual_seed = self.episode_index if seed is None else seed
        setup_kwargs = _load_robotwin_setup_kwargs(self.task_name)
        setup_kwargs.update(seed=actual_seed, is_test=True)
        with torch.enable_grad():
            self._env.setup_demo(**setup_kwargs)
        self.episode_index += self._reset_stride
        self._step_count = 0

        use_official_instruction = self.task_name in {"blocks_ranking_rgb", "blocks_ranking_size"}
        if _env_flag(OFFICIAL_INSTRUCTION_ENV, default=use_official_instruction):
            self.task_description = _generate_robotwin_official_instruction(self.task_name, self._env)
            if hasattr(self._env, "set_instruction"):
                self._env.set_instruction(instruction=self.task_description)
            logger.info("RoboTwin official instruction | task=%s | %s", self.task_name, self.task_description)
        else:
            self.task_description = self.task_name.replace("_", " ")

        # 在 ee 模式下，策略预测相对于初始末端位姿的位姿增量。
        if self.action_mode == "ee":
            self._init_eef_pose = self._read_eef_pose()

        obs = self._get_obs()
        return obs, {"is_success": False, "task": self.task_name}

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        assert self._env is not None, "step() called before reset()"
        if action.ndim != 1 or action.shape[0] != self._action_dim:
            raise ValueError(f"Expected 1-D action of shape ({self._action_dim},), got {action.shape}")

        with torch.enable_grad():
            if self.action_mode == "ee":
                ee_action = _add_init_eef_pose(np.asarray(action, dtype=np.float64), self._init_eef_pose)
                self._env.take_action(ee_action, action_type="ee")
            elif hasattr(self._env, "take_action"):
                self._env.take_action(action)
            else:
                self._env.step(action)

        self._step_count += 1

        is_success = bool(getattr(self._env, "eval_success", False))
        if not is_success and hasattr(self._env, "check_success"):
            is_success = bool(self._env.check_success())

        obs = self._get_obs()
        reward = float(is_success)
        terminated = is_success
        truncated = self._step_count >= self.episode_length

        info: dict[str, Any] = {
            "task": self.task_name,
            "is_success": is_success,
            "step": self._step_count,
        }
        if terminated or truncated:
            info["final_info"] = {
                "task": self.task_name,
                "is_success": is_success,
            }
            self.reset()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        self._ensure_env()
        obs = self._get_obs()
        # 渲染时优先使用头部相机；不可用时回退到第一个可用相机。
        if "head_camera" in obs["pixels"]:
            return obs["pixels"]["head_camera"]
        return next(iter(obs["pixels"].values()))

    def close(self) -> None:
        if self._env is not None:
            if hasattr(self._env, "close_env"):
                import contextlib

                with contextlib.suppress(TypeError):
                    self._env.close_env()
            self._env = None


# ---- 多任务工厂 --------------------------------------------------------


def _make_env_fns(
    *,
    task_name: str,
    n_envs: int,
    camera_names: list[str],
    observation_height: int,
    observation_width: int,
    episode_length: int,
    action_mode: str = "joint",
) -> list[Callable[[], RoboTwinEnv]]:
    """返回单个任务的 n_envs 个工厂可调用对象。"""

    def _make_one(episode_index: int) -> RoboTwinEnv:
        return RoboTwinEnv(
            task_name=task_name,
            episode_index=episode_index,
            n_envs=n_envs,
            camera_names=camera_names,
            observation_height=observation_height,
            observation_width=observation_width,
            episode_length=episode_length,
            action_mode=action_mode,
        )

    return [partial(_make_one, i) for i in range(n_envs)]


def create_robotwin_envs(
    task: str,
    n_envs: int,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    camera_names: Sequence[str] = ROBOTWIN_CAMERA_NAMES,
    observation_height: int = DEFAULT_CAMERA_H,
    observation_width: int = DEFAULT_CAMERA_W,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    action_mode: str = "joint",
) -> dict[str, dict[int, Any]]:
    """创建向量化的 RoboTwin 2.0 环境。

    Returns:
        ``dict[task_name][0] -> VectorEnv``——每个任务一个条目，每个
        条目包装 ``n_envs`` 个并行 rollout。

    Args:
        task: 逗号分隔的任务名列表（例如 ``"beat_block_hammer"``
            或 ``"beat_block_hammer,click_bell"``）。
        n_envs: 每个任务的并行 rollout 数量。
        env_cls: 向量环境构造函数（例如 ``gym.vector.AsyncVectorEnv``）。
        camera_names: 要包含在观测中的相机。
        observation_height: 所有相机的像素高度。
        observation_width: 所有相机的像素宽度。
        episode_length: 截断前的最大步数。
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be callable (e.g. gym.vector.AsyncVectorEnv).")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    task_names = [t.strip() for t in str(task).split(",") if t.strip()]
    if not task_names:
        raise ValueError("`task` must contain at least one RoboTwin task name.")

    unknown = [t for t in task_names if t not in ROBOTWIN_TASKS]
    if unknown:
        raise ValueError(f"Unknown RoboTwin tasks: {unknown}. Available tasks: {sorted(ROBOTWIN_TASKS)}")

    logger.info(
        "Creating RoboTwin envs | tasks=%s | n_envs(per task)=%d",
        task_names,
        n_envs,
    )

    is_async = env_cls is gym.vector.AsyncVectorEnv
    cached_obs_space: spaces.Space | None = None
    cached_act_space: spaces.Space | None = None
    cached_metadata: dict[str, Any] | None = None

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for task_name in task_names:
        fns = _make_env_fns(
            task_name=task_name,
            n_envs=n_envs,
            camera_names=list(camera_names),
            observation_height=observation_height,
            observation_width=observation_width,
            episode_length=episode_length,
            action_mode=action_mode,
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

    return {k: dict(v) for k, v in out.items()}
