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
"""通过运行 rollout 并计算指标来在环境上评估策略。

需要：pip install 'lerobot[evaluation]'，以及策略附加依赖（例如 lerobot[pi]）
          和环境附加依赖（例如 lerobot[pusht]），如果在仿真中评估的话。

用法示例：

你想要评估一个来自 hub 的模型（例如：https://huggingface.co/lerobot/diffusion_pusht），
评估 10 个 episode。

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

或者，你想要评估一个来自 LeRobot 训练脚本的模型检查点，评估 10 个 episode。
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

请注意，在这两个示例中，仓库/文件夹应至少包含 `config.json` 和 `model.safetensors` 文件。

你可以在 lerobot/configs/eval.py 的 `EvalPipelineConfig` 中了解此脚本的 CLI 选项。
"""

import concurrent.futures as cf
import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import FeatureType, parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs import (
    check_env_attributes_and_types,
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.lerobot_types import PolicyAction
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGE, OBS_IMAGES, OBS_STR, REWARD
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import _peft_available, register_third_party_plugins, require_package
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    init_logging,
    inside_slurm,
)

if TYPE_CHECKING or _peft_available:
    from peft import PeftModel
else:
    PeftModel = None


logger = logging.getLogger(__name__)


def _env_features_to_dataset_features(env_features: dict) -> dict:
    """将 EnvConfig.features 转换为 LeRobotDataset.create() 所期望的字典格式。"""
    features = {}
    for key, ft in env_features.items():
        shape = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            features[key] = {"dtype": "video", "shape": shape, "names": ["height", "width", "channel"]}
        else:
            features[key] = {"dtype": "float32", "shape": shape, "names": None}
    features["next.reward"] = {"dtype": "float32", "shape": (1,), "names": None}
    features["next.success"] = {"dtype": "bool", "shape": (1,), "names": None}
    features["next.done"] = {"dtype": "bool", "shape": (1,), "names": None}
    return features


def _build_raw_frame(
    raw_obs: dict,
    env_idx: int,
    action: np.ndarray,
    reward: float,
    success: bool,
    done: bool,
    task: str,
    env_features: dict,
) -> dict:
    """基于某个环境索引的原始环境观测构建一个数据集帧。

    帧中的键与 env_features 中的键一致，从而与
    _env_features_to_dataset_features() 创建的数据集模式对齐。
    """
    frame: dict[str, Any] = {}
    for key in env_features:
        if key == ACTION:
            continue
        if key.startswith("next."):
            continue
        if "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict):
            for cam_name, img in raw_obs["pixels"].items():
                candidate = f"{OBS_IMAGES}.{cam_name}"
                if candidate == key:
                    frame[key] = img[env_idx]
            if key in frame:
                continue
        if "pixels" in raw_obs and not isinstance(raw_obs["pixels"], dict) and key in ("pixels", OBS_IMAGE):
            frame[key] = raw_obs["pixels"][env_idx]
            continue
        if key in raw_obs and isinstance(raw_obs[key], np.ndarray):
            val = raw_obs[key][env_idx]
            if val.dtype == np.float64:
                val = val.astype(np.float32)
            frame[key] = val
    frame[ACTION] = action
    frame["next.reward"] = np.atleast_1d(np.float32(reward))
    frame["next.success"] = np.atleast_1d(np.bool_(success))
    frame["next.done"] = np.atleast_1d(np.bool_(done))
    frame["task"] = task
    return frame


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    recording_dir: Path | None = None,
    env_features: dict | None = None,
    recording_repo_id: str | None = None,
    recording_private: bool = False,
    predicted_latents_callback: Callable[[PreTrainedPolicy], None] | None = None,
) -> dict:
    """在一批环境上运行一次批量策略 rollout。

    请注意，批次中的所有环境会一直运行，直到最后一个环境结束。这意味着
    可能需要丢弃部分数据（对于那些不是最先结束的环境）。

    返回的字典包含：
        （可选）"observation"：一个字典，包含映射到各观测键的
            (batch, sequence + 1, *) 张量。请注意，相对于字典中的其他键，
            它多出一个序列元素。这是因为在环境被终止（terminated）或截断（truncated）
            之后还包含了一个额外的观测。
        "action"：一个 (batch, sequence, action_dim) 张量，表示根据观测执行的动作
            （不包含最后的观测）。
        "reward"：一个 (batch, sequence) 张量，表示执行这些动作所获得的奖励。
        "success"：一个 (batch, sequence) 张量，表示成功条件（它唯一可能为 True 的时刻
            是环境被终止/截断时）。
        "done"：一个 (batch, sequence) 张量，表示**累积的**结束条件。对于任意给定的批次元素，
            第一个 True 之后一直到末尾全是 True。这可以用于掩蔽上述序列中多余的元素。

    参数:
        env: 环境批次。
        policy: 策略。必须是一个 PyTorch nn 模块。
        seeds: 环境会在 rollout 开始时设置一次随机种子。如果提供，此参数
            为每个环境指定种子。
        return_observations: 是否在返回的 rollout 数据中包含所有观测。观测
            作为可选项返回，因为缓存它们通常会占用更多内存。默认为 False。
        render_callback: 可选的渲染回调，在环境重置之后以及每一步之后使用。
        predicted_latents_callback: 可选回调，在每次 ``select_action`` 之后以策略
            本身为参数调用。世界模型策略（例如 LingBot-VA）会将预测的视频潜变量
            存放在 ``policy.last_predicted_latents`` 上；这让调用方可以拼接各个分块并只解码一次。
    返回:
        上述字典。
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # 重置策略和环境。
    policy.reset()
    # NEW_ROLLOUT_OPTION 告知 FreezeAfterEpisodeEnd 这是一个真正的新 episode，
    # 而不是 Gymnasium 对已结束的子环境进行的无参数自动重置（autoreset）。
    observation, info = env.reset(seed=seeds, options={NEW_ROLLOUT_OPTION: True})
    if render_callback is not None:
        render_callback(env)

    recording_datasets: list[LeRobotDataset] | None = None
    raw_observation = None
    task_desc = ""
    if recording_dir is not None and env_features is not None:
        features = _env_features_to_dataset_features(env_features)
        fps = env.unwrapped.metadata.get("render_fps", 30)
        recording_datasets = []
        multi_env = env.num_envs > 1
        base_repo_id = recording_repo_id or "eval_recording"
        for i in range(env.num_envs):
            root = str(recording_dir / f"env_{i}") if multi_env else str(recording_dir)
            repo_id = f"{base_repo_id}_env_{i}" if multi_env else base_repo_id
            recording_datasets.append(
                LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=fps,
                    features=features,
                    root=root,
                    use_videos=True,
                )
            )
        raw_observation = deepcopy(observation)
        try:
            task_desc = list(env.call("task_description"))[0]
        except (AttributeError, NotImplementedError):
            task_desc = ""

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # 记录哪些环境已经结束。
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # 使用 slurm 时不希望显示进度条，因为它会使日志变得杂乱
        leave=False,
    )
    check_env_attributes_and_types(env)
    try:
        while not np.all(done) and step < max_steps:
            # 将 numpy 数组转换为张量，并把字典键改为 LeRobot 策略格式。
            observation = preprocess_observation(observation)
            if return_observations:
                all_observations.append(deepcopy(observation))

            # 从子环境推断 "task"（优先使用自然语言描述）。
            # env.call() 对 SyncVectorEnv 和 AsyncVectorEnv 都适用。
            try:
                observation["task"] = list(env.call("task_description"))
            except (AttributeError, NotImplementedError):
                try:
                    observation["task"] = list(env.call("task"))
                except (AttributeError, NotImplementedError):
                    observation["task"] = [""] * env.num_envs

            # 应用环境专属的预处理（例如用于 LIBERO 的 LiberoProcessorStep）
            observation = env_preprocessor(observation)

            observation = preprocessor(observation)
            with torch.inference_mode():
                action = policy.select_action(observation)
            if predicted_latents_callback is not None:
                predicted_latents_callback(policy)
            action = postprocessor(action)

            action_transition = {ACTION: action}
            action_transition = env_postprocessor(action_transition)
            action = action_transition[ACTION]

            # 转换为 CPU / numpy。
            action_numpy: np.ndarray = action.to("cpu").numpy()
            assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

            # 执行下一个动作。
            observation, reward, terminated, truncated, info = env.step(action_numpy)
            if render_callback is not None:
                render_callback(env)

            # VectorEnv 将 is_success 存储在 `info["final_info"][env_index]["is_success"]` 中。
            # 如果没有任何环境结束，则不存在 "final_info"。
            if "final_info" in info:
                final_info = info["final_info"]
                if isinstance(final_info, dict):
                    is_success = final_info.get("is_success", [False] * env.num_envs)
                    successes = (
                        is_success.tolist()
                        if hasattr(is_success, "tolist")
                        else [bool(is_success)] * env.num_envs
                    )
                else:
                    # Gymnasium < 1.0 会将 final_info 返回为按环境排列的序列/对象数组，
                    # 只有刚刚结束的环境所对应的条目才会被设置为字典。
                    successes = []
                    for item in final_info:
                        if isinstance(item, dict) and "is_success" in item:
                            successes.append(bool(item["is_success"]))
                        else:
                            successes.append(False)
            elif "is_success" in info:
                is_success = info["is_success"]
                successes = (
                    is_success.tolist()
                    if hasattr(is_success, "tolist")
                    else [bool(is_success)] * env.num_envs
                )
            else:
                successes = [False] * env.num_envs

            if recording_datasets is not None and raw_observation is not None:
                prev_done = done.copy()
                for env_idx in range(env.num_envs):
                    if prev_done[env_idx]:
                        continue
                    frame = _build_raw_frame(
                        raw_observation,
                        env_idx,
                        action_numpy[env_idx],
                        reward[env_idx],
                        successes[env_idx],
                        bool(terminated[env_idx] | truncated[env_idx]),
                        task_desc,
                        recording_datasets[env_idx].features,
                    )
                    recording_datasets[env_idx].add_frame(frame)
                    if terminated[env_idx] or truncated[env_idx]:
                        recording_datasets[env_idx].save_episode()
                raw_observation = deepcopy(observation)

            # 记录到目前为止哪些环境已经结束。
            # 如果达到最大步数限制，则将该 episode 标记为结束。
            # 这确保 rollout 总是在 `max_steps` 处干净地终止，
            # 并允许一致地触发日志记录/保存（例如视频）。
            done = terminated | truncated | done
            if step + 1 == max_steps:
                done = np.ones_like(done, dtype=bool)

            all_actions.append(torch.from_numpy(action_numpy))
            all_rewards.append(torch.from_numpy(reward))
            all_dones.append(torch.from_numpy(done))
            all_successes.append(torch.tensor(successes))

            step += 1
            running_success_rate = (
                einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
            )
            progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
            progbar.update()
    finally:
        if recording_datasets is not None:
            for ds in recording_datasets:
                ds.finalize()
                if recording_repo_id is not None:
                    if ds.num_episodes > 0:
                        ds.push_to_hub(private=recording_private)
                    else:
                        logging.warning("No episodes recorded for %s — skipping push to hub.", ds.repo_id)

    # 记录最后的观测。
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # 沿第一维堆叠序列，从而得到 (batch, sequence, *) 张量。
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    recording_dir: Path | None = None,
    env_features: dict | None = None,
    recording_repo_id: str | None = None,
    recording_private: bool = False,
    save_predicted_video: bool = False,
) -> dict:
    """
    参数:
        env: 环境批次。
        policy: 策略。
        n_episodes: 要评估的 episode 数量。
        max_episodes_rendered: 要渲染成视频的最大 episode 数量。
        videos_dir: 渲染出的视频的保存位置。
        return_episode_data: 是否返回用于在线训练的 episode 数据。该数据会被并入
            返回字典的 "episodes" 键中。
        start_seed: 第一次单独 rollout 使用的首个种子。后续每次 rollout 的
            种子递增 1。如果未提供，则不手动为环境设置种子。
    返回:
        包含与 rollout 相关的指标和数据的字典。
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    # 世界模型策略（例如 LingBot-VA）通过其配置选择启用预测视频保存。
    save_predicted_video = save_predicted_video or bool(
        getattr(getattr(policy, "config", None), "save_predicted_video", False)
    )

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        if not _peft_available:
            raise exc
        require_package("peft", extra="peft")
        if not isinstance(policy, PeftModel):
            raise exc

    start = time.time()
    # 为直接调用者保留模式状态。eval_policy_all 会在所有任务外围
    # 统一设置模式，从而避免并行评估之间相互竞争。
    was_training = policy.training
    policy.eval()

    # 计算要获得 n_episodes 个 episode 需要多少个批量 rollout。请注意，如果 n_episodes
    # 不能被 env.num_envs 整除，最后一个批次中的部分数据最终会被丢弃。
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # 记录一些指标。
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # 用于保存视频的线程
    n_episodes_rendered = 0  # 用于控制保存视频的正确数量

    # 用于可视化的回调。
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif hasattr(env, "call"):
            # 这里必须渲染所有帧，然后丢弃不需要的帧。
            # 涵盖 AsyncVectorEnv 和 _LazyAsyncVectorEnv（后者包装了前者）。
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if save_predicted_video:
        if not videos_dir:
            raise ValueError("If save_predicted_video is True, videos_dir must be provided.")
        predicted_video_paths: list[str] = []
        n_predicted_rendered = 0

    # 在整个 rollout 过程中收集预测视频潜变量（仅限世界模型策略）。这些潜变量会在
    # rollout 结束后拼接并只解码一次，与上游 LingBot-VA 的可视化路径保持一致。
    def collect_predicted_latents(policy: PreTrainedPolicy):
        latents = getattr(policy, "last_predicted_latents", None)
        if latents is not None:
            pred_latents.append(
                latents.detach().to("cpu") if hasattr(latents, "detach") else torch.as_tensor(latents).cpu()
            )
            policy.last_predicted_latents = None

    if return_episode_data:
        episode_data: dict | None = None

    # 使用 slurm 时不希望显示进度条，因为它会使日志变得杂乱
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # 缓存用于渲染视频的帧。每个元素的形状为 (b, h, w, c)，列表的索引对应
        # rollout 的步数。
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if save_predicted_video:
            pred_latents: list[torch.Tensor] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            recording_dir=recording_dir,
            env_features=env_features,
            recording_repo_id=recording_repo_id,
            recording_private=recording_private,
            predicted_latents_callback=collect_predicted_latents if save_predicted_video else None,
        )

        # 找出每个 rollout 序列中第一次遇到结束条件的位置（此位置之后的结果
        # 不会被纳入）。
        n_steps = rollout_data["done"].shape[1]
        # 注意：这里依赖 argmax 的一个特性：在出现平局时，它会返回第一次出现的位置。
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # 构造一个形状为 (batch, n_steps) 的掩码，用于掩蔽第一次结束之后
        # （按批次元素分别判断）的 rollout 数据。注意这里的 `done_indices + 1`，
        # 以确保保留结束那一步的数据。
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # 扩充指标。
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.extend([None] * env.num_envs)

        # FIXME：episode_data 要么为 None，要么尚不存在
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # 一些健全性检查，以确保我们正确地汇编数据。
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # 拼接 episode 数据。
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # 可能需要渲染视频用于可视化。
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # +1 以捕获最后一个观测
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        # 可能需要保存策略对该批次 rollout 的预测（想象）视频。
        if save_predicted_video and len(pred_latents) > 0:
            predicted_latent = torch.cat(pred_latents, dim=2)
            decoder = getattr(policy, "decode_predicted_latents", None) or getattr(
                policy, "_decode_predicted_video", None
            )
            if decoder is None:
                raise AttributeError(
                    "Policy config requested predicted-video saving, but the policy does not expose "
                    "`decode_predicted_latents` or `_decode_predicted_video`."
                )
            predicted_video = decoder(predicted_latent)
            if hasattr(predicted_video, "detach"):
                predicted_video = predicted_video.detach().to("cpu").numpy()
            videos_dir.mkdir(parents=True, exist_ok=True)
            predicted_video_path = videos_dir / f"pred_episode_{n_predicted_rendered}.mp4"
            predicted_video_paths.append(str(predicted_video_path))
            thread = threading.Thread(
                target=write_video,
                args=(
                    str(predicted_video_path),
                    predicted_video,
                    env.unwrapped.metadata["render_fps"],
                ),
            )
            thread.start()
            threads.append(thread)
            n_predicted_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # 等待所有视频渲染线程结束。
    for thread in threads:
        thread.join()

    # 汇编评估信息。
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    if save_predicted_video:
        info["predicted_video_paths"] = predicted_video_paths

    policy.train(was_training)

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """供 `eval_policy(return_episode_data=True)` 使用的便捷函数。

    将所有 rollout 数据汇编为一个 Hugging Face 数据集。

    数据集推送到 hub 时实现了类似的逻辑（参见：`push_to_hub`）。
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # +2 以包含第一个结束帧和最后一个观测帧。
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # 这里使用 `num_frames - 1`，因为我们暂时还不想包含最后一个观测帧。
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # 对于最后一个观测帧，其他所有键都直接通过复制最后一个值来填充。
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # 检查设备是否可用
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info(f"Making environment (batch_size={cfg.eval.batch_size}, async={cfg.eval.use_async_envs}).")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # 推理设备会自动设置为与检测到的硬件一致，覆盖训练时先前的任何设备设置以确保兼容性。
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # 创建环境专属的预处理器和后处理器（例如用于 LIBERO 环境）
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    recording_dir = Path(cfg.output_dir) / "recordings" if cfg.eval.recording else None
    max_episodes_rendered = 0 if cfg.eval.recording else 10
    videos_dir = None if cfg.eval.recording else Path(cfg.output_dir) / "videos"

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=max_episodes_rendered,
            videos_dir=videos_dir,
            return_episode_data=False,
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            recording_dir=recording_dir,
            env_features=cfg.env.features if cfg.eval.recording else None,
            recording_repo_id=cfg.eval.recording_repo_id,
            recording_private=cfg.eval.recording_private,
        )
        logger.info("Overall Aggregated Metrics:")
        logger.info(info["overall"])

        # 打印每个测试套件（suite）的统计信息
        for task_group, task_group_info in info.items():
            logger.info(f"\nAggregated Metrics for {task_group}:")
            logger.info(task_group_info)
    # 关闭所有向量环境
    close_envs(envs)

    # 保存信息
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


# ---- 单个任务评估返回的类型化负载 ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]
    predicted_video_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths", "predicted_video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    recording_dir: Path | None = None,
    env_features: dict | None = None,
    recording_repo_id: str | None = None,
    recording_private: bool = False,
) -> TaskMetrics:
    """使用提供的向量环境评估某个套件中的一个 task_id。"""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        recording_dir=recording_dir,
        env_features=env_features,
        recording_repo_id=recording_repo_id,
        recording_private=recording_private,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
        predicted_video_paths=task_result.get("predicted_video_paths", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    recording_dir: Path | None = None,
    env_features: dict | None = None,
    recording_repo_id: str | None = None,
    recording_private: bool = False,
):
    """
    针对单个 (task_group, task_id, env) 运行 eval_one。
    返回 (task_group, task_id, task_metrics_dict)。
    此函数特意定义在模块级别，以便于测试。
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    task_recording_dir = None
    task_repo_id = None
    if recording_dir is not None and env_features is not None:
        task_recording_dir = recording_dir / f"{task_group}_{task_id}"
        if recording_repo_id is not None:
            task_repo_id = f"{recording_repo_id}_{task_group}_{task_id}"

    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        recording_dir=task_recording_dir,
        env_features=env_features,
        recording_repo_id=task_repo_id,
        recording_private=recording_private,
    )

    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    metrics.setdefault("predicted_video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    recording_dir: Path | None = None,
    env_features: dict | None = None,
    recording_repo_id: str | None = None,
    recording_private: bool = False,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
) -> dict:
    """
    评估一个嵌套的 `envs` 字典：{task_group: {task_id: vec_env}}。
    此实现会将任务展平，顺序执行或通过 ThreadPoolExecutor 执行，
    累积各组以及总体的统计信息，并返回与单环境评估器相同的聚合指标
    模式（avg_sum_reward / avg_max_reward / pc_success / 计时信息），
    此外还包括每个任务的信息。
    """
    start_t = time.time()

    # 将 envs 展平为 (task_group, task_id, env) 列表
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # 累加器：在每个组的层级以及所有组的整体层级分别跟踪指标
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # 小型内联辅助函数，用于将单个任务的指标累积到累加器中
    def _accumulate_to(group: str, metrics: dict):
        # metrics 预期包含 'sum_rewards'、'max_rewards'、'successes'，以及可选的 'video_paths'，
        # 但 eval_one 可能存储的是按 episode 的列表；为保持稳健，标量或列表都接受。
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        for key in ("video_paths", "predicted_video_paths"):
            paths = metrics.get(key, [])
            if paths:
                group_acc[group][key].extend(paths)
                overall[key].extend(paths)

    # 选择运行方式（顺序还是多线程）
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        recording_dir=recording_dir,
        env_features=env_features,
        recording_repo_id=recording_repo_id,
        recording_private=recording_private,
    )

    # 在启动任何工作进程之前设置共享策略的模式。如果在各个任务
    # 内部恢复模式，可能会出现一个任务启用训练模式、而另一个任务
    # 仍在评估中的情况。
    was_training = policy.training
    policy.eval()
    try:
        if max_parallel_tasks <= 1:
            prefetch_thread: threading.Thread | None = None
            for i, (task_group, task_id, env) in enumerate(tasks):
                if prefetch_thread is not None:
                    prefetch_thread.join()
                    prefetch_thread = None

                try:
                    tg, tid, metrics = task_runner(task_group, task_id, env)
                    _accumulate_to(tg, metrics)
                    per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
                finally:
                    env.close()
                    # 在关闭当前环境*之后*再预取下一个任务的工作进程，以防止
                    # 相邻任务之间的 GPU 显存重叠。
                    if i + 1 < len(tasks):
                        next_env = tasks[i + 1][2]
                        if hasattr(next_env, "_ensure"):
                            prefetch_thread = threading.Thread(target=next_env._ensure, daemon=True)
                            prefetch_thread.start()
        else:
            with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
                fut2meta = {}
                for task_group, task_id, env in tasks:
                    fut = executor.submit(task_runner, task_group, task_id, env)
                    fut2meta[fut] = (task_group, task_id, env)
                for fut in cf.as_completed(fut2meta):
                    tg, tid, env = fut2meta[fut]
                    try:
                        tg, tid, metrics = fut.result()
                        _accumulate_to(tg, metrics)
                        per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
                    finally:
                        env.close()
    finally:
        policy.train(was_training)

    # 计算聚合指标的辅助函数（对列表/标量都稳健）
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # 计算各组的聚合结果
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
            "predicted_video_paths": list(acc["predicted_video_paths"]),
        }

    # 总体聚合结果
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
        "predicted_video_paths": list(overall["predicted_video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
