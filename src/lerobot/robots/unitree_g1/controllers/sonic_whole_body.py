#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""用于 Unitree G1 的 SONIC decoder 全身控制器（仅 token 模式）。

这是 NVIDIA SONIC 部署栈中 *decode*（解码）部分的纯 Python/ONNX 重实现。
encoder（编码器）被有意省略：每个控制节拍由一个输出 token 的 VLA（例如
``nepyope/sonic_walk``）直接提供 64 维隐变量 ``motion_token``，SONIC **decoder**
将 ``token + 近期本体感知历史`` 映射为残差动作；该残差经过缩放后叠加到站立姿态
（``default_angles``）上，从而为机器人的 PD 控制器生成 50 Hz 的关节位置目标。

索引空间：关节存在两种排序——**IsaacLab**（策略/训练顺序）和 **MuJoCo**
（部署顺序）。``ISAACLAB_TO_MUJOCO`` / ``MUJOCO_TO_ISAACLAB``（位于 g1_utils）
负责二者之间的转换。四元数采用标量在前的 ``(w, x, y, z)`` 约定。
"""

from __future__ import annotations

import json
import logging

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from ..g1_utils import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    NUM_MOTORS,
    G1_29_JointIndex,
    get_gravity_orientation,
)
from ..unitree_g1 import RobotController

logger = logging.getLogger(__name__)

CONTROL_DT = 0.02  # 50 Hz 控制周期（秒）
TOKEN_DIM = 64  # decoder 隐变量维度
HISTORY_LEN = 10  # decoder 条件依赖的本体感知帧数

# 隐变量 token 的特征键前缀：action 携带 token，obs 将其回传。
TOKEN_ACTION_PREFIX = "motion_token"  # nosec B105 - 特征键前缀，并非密钥
TOKEN_STATE_PREFIX = "motion_token_state"  # nosec B105 - 特征键前缀，并非密钥

# SONIC decoder 检查点。部署常量（kp/kd、default_angles、action_scale、
# neutral_token）内置在 ONNX 元数据中；参见 upload_sonic_decoder.py。
DEFAULT_SONIC_REPO_ID = "lerobot/sonic_decoder"
# token + HISTORY_LEN 帧（角速度、关节位置、关节速度、上一动作）+ 重力
DECODER_INPUT_DIM = TOKEN_DIM + HISTORY_LEN * (3 + 3 * NUM_MOTORS) + HISTORY_LEN * 3  # 994

# Decoder 文件名映射：完整 decoder（default）或 NVIDIA 蒸馏的低延迟版本。
POLICY_FILES = {
    "default": "model_decoder.onnx",
    "low_latency": "low_latency/model_decoder.onnx",
}


def load_policy(
    repo_id: str = DEFAULT_SONIC_REPO_ID,
    policy_type: str = "default",
) -> tuple[ort.InferenceSession, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 SONIC decoder 及其内置在 ONNX 元数据中的部署常量。

    Args:
        repo_id: Hugging Face Hub 仓库 ID
        policy_type: 为 "default"（完整 decoder）或 "low_latency"（蒸馏版本）

    Returns:
        (decoder, kp, kd, default_angles, action_scale, neutral_token) 元组。
        增益/姿态/缩放均为 IsaacLab 关节顺序的 (29,) float32 数组；
        neutral_token 是 (64,) 的空闲隐变量。
    """
    if policy_type not in POLICY_FILES:
        raise ValueError(f"Unknown policy type: {policy_type}. Choose from: {list(POLICY_FILES.keys())}")

    filename = POLICY_FILES[policy_type]
    logger.info(f"Loading {policy_type.upper()} SONIC decoder from: {repo_id}/{filename}")
    decoder_path = hf_hub_download(repo_id=repo_id, filename=filename)

    decoder = ort.InferenceSession(decoder_path)
    logger.info(f"Decoder loaded: {decoder.get_inputs()[0].shape} → {decoder.get_outputs()[0].shape}")

    # 从 ONNX 元数据中提取部署常量
    model = onnx.load(decoder_path, load_external_data=False)
    metadata = {prop.key: prop.value for prop in model.metadata_props}

    required = ("kp", "kd", "default_angles", "action_scale", "neutral_token")
    missing = [k for k in required if k not in metadata]
    if missing:
        raise ValueError(f"ONNX model must contain {list(required)} in metadata (missing {missing})")

    arr = {k: np.array(json.loads(metadata[k]), dtype=np.float32) for k in required}
    logger.info(f"Loaded SONIC deploy constants from ONNX ({len(arr['kp'])} joints)")
    return decoder, arr["kp"], arr["kd"], arr["default_angles"], arr["action_scale"], arr["neutral_token"]


class SonicWholeBodyController(RobotController):
    """用于 UnitreeG1 后台控制器线程的全身 SONIC decoder 控制器。

    仅 token 部署（绕过 encoder）：每个节拍将最新机器人状态追加到 10 帧历史
    缓冲区，然后把策略提供的 64 维 token 与该历史一起映射为残差动作，叠加到
    ``default_angles`` 上 -> 50 Hz 关节位置目标。
    """

    control_dt = CONTROL_DT

    def __init__(self, policy_type: str = "default"):
        self.decoder, self.kp, self.kd, self.default_angles, self.action_scale, self.neutral_token = (
            load_policy(policy_type=policy_type)
        )
        self.decoder_input = self.decoder.get_inputs()[0].name
        self.default_angles_mj = self.default_angles[MUJOCO_TO_ISAACLAB]

        # 64 维隐变量 token 动作空间；rollout 会把策略的 64 维输出映射到这些键上。
        self.action_ft = {f"{TOKEN_ACTION_PREFIX}.{i}.pos": float for i in range(TOKEN_DIM)}
        # 64 维 token 本体状态，由 rollout 聚合到 observation.state（最后一个 token）。
        self.observation_ft = {f"{TOKEN_STATE_PREFIX}.{i}.pos": float for i in range(TOKEN_DIM)}

        self.reset()
        logger.info("SonicWholeBodyController initialized")

    def reset(self) -> None:
        """为新的回合重置内部状态：持有的 token 和本体感知历史。"""
        self.last_action_mj = np.zeros(NUM_MOTORS, np.float32)
        self.h_q_mj = [np.zeros(NUM_MOTORS, np.float32) for _ in range(HISTORY_LEN)]
        self.h_dq_mj = [np.zeros(NUM_MOTORS, np.float32) for _ in range(HISTORY_LEN)]
        self.h_ang = [np.zeros(3, np.float32) for _ in range(HISTORY_LEN)]
        self.h_act_mj = [np.zeros(NUM_MOTORS, np.float32) for _ in range(HISTORY_LEN)]
        self.h_quat = [np.array([1, 0, 0, 0], np.float32) for _ in range(HISTORY_LEN)]
        self._last_token = None  # 首个节拍会重新播种 neutral token

    def observation_state(self) -> dict[str, float]:
        """将上一次解码的 token 作为 ``observation.state`` 回传，使输出 token 的 VLA
        能基于自身上一个 token 形成闭环。"""
        token = self._last_token if self._last_token is not None else np.zeros(TOKEN_DIM, dtype=np.float32)
        return {f"{TOKEN_STATE_PREFIX}.{i}.pos": float(v) for i, v in enumerate(token)}

    def run_step(self, action: dict, lowstate) -> dict:
        """解码一个控制节拍，输出绝对关节位置目标。

        Args:
            action: 最新的动作快照。必须同时包含全部 64 个 ``motion_token.{i}.pos``
                键才会更新隐变量；任何其他内容（摇杆轴、不完整的分片）都会保持
                先前持有的 token 不变。
            lowstate: 携带关节位置/速度和 IMU 状态的 Unitree 低层状态。

        Returns:
            全部 29 个关节、以 ``<joint>.q`` 为键的绝对关节目标：``default_angles``
            加上 decoder 的残差并乘以 ``action_scale``。
        """
        # Token：从 motion_token.{i}.pos 重组稠密的 64 维隐变量（必须包含全部键）；
        # 否则保持上一个 token（首个真实 token 之前为 neutral，解码结果为站立姿态）。
        keys = [f"{TOKEN_ACTION_PREFIX}.{i}.pos" for i in range(TOKEN_DIM)]
        if action and all(k in action for k in keys):
            self._last_token = np.fromiter(
                (float(action[k]) for k in keys), dtype=np.float32, count=TOKEN_DIM
            )
        elif self._last_token is None:
            self._last_token = self.neutral_token.copy()

        # 从低层状态读取本体感知（IsaacLab 关节顺序）。
        q = np.array([lowstate.motor_state[m.value].q for m in G1_29_JointIndex], np.float32)
        dq = np.array([lowstate.motor_state[m.value].dq for m in G1_29_JointIndex], np.float32)
        quat = np.array(lowstate.imu_state.quaternion, np.float32)  # (w, x, y, z)
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        ang = np.array(lowstate.imu_state.gyroscope, np.float32)

        # 压入 10 帧历史（最新帧在前）。decoder 消费的是 MuJoCo 关节顺序，
        # 因此通过 MUJOCO_TO_ISAACLAB 对 q/dq 重排（已对照 ONNX 验证，切勿翻转）。
        self.h_q_mj = [q[MUJOCO_TO_ISAACLAB] - self.default_angles_mj] + self.h_q_mj[:-1]
        self.h_dq_mj = [dq[MUJOCO_TO_ISAACLAB]] + self.h_dq_mj[:-1]
        self.h_ang = [ang] + self.h_ang[:-1]
        self.h_act_mj = [self.last_action_mj.copy()] + self.h_act_mj[:-1]
        self.h_quat = [quat] + self.h_quat[:-1]

        # 组装 994 维 decoder 输入：token + 从最旧到最新的历史 + 重力。
        obs = np.zeros(DECODER_INPUT_DIM, np.float32)
        obs[:TOKEN_DIM] = self._last_token
        off = TOKEN_DIM
        for hist, sz in (
            (self.h_ang, 3),
            (self.h_q_mj, NUM_MOTORS),
            (self.h_dq_mj, NUM_MOTORS),
            (self.h_act_mj, NUM_MOTORS),
        ):
            for frame in reversed(hist):
                obs[off : off + sz] = frame
                off += sz
        for hquat in reversed(self.h_quat):
            obs[off : off + 3] = get_gravity_orientation(hquat)
            off += 3

        # 解码 -> 残差动作（MuJoCo 顺序），叠加到站立姿态上。
        action_mj = (
            self.decoder.run(None, {self.decoder_input: obs.reshape(1, -1)})[0].squeeze().astype(np.float32)
        )
        self.last_action_mj = action_mj.copy()
        target = self.default_angles + action_mj[ISAACLAB_TO_MUJOCO] * self.action_scale
        return {f"{m.name}.q": float(target[m.value]) for m in G1_29_JointIndex}
