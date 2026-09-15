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
import logging

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from ..g1_utils import (
    REMOTE_AXES,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    get_gravity_orientation,
)
from ..unitree_g1 import RobotController

logger = logging.getLogger(__name__)

DEFAULT_ANGLES = np.zeros(29, dtype=np.float32)
DEFAULT_ANGLES[[0, 6]] = -0.312  # 髋关节俯仰
DEFAULT_ANGLES[[3, 9]] = 0.669  # 膝关节
DEFAULT_ANGLES[[4, 10]] = -0.363  # 踝关节俯仰
DEFAULT_ANGLES[[15, 22]] = 0.2  # 肩关节俯仰
DEFAULT_ANGLES[16] = 0.2  # 左肩 roll
DEFAULT_ANGLES[23] = -0.2  # 右肩 roll
DEFAULT_ANGLES[[18, 25]] = 0.6  # 肘关节

# 控制参数
ACTION_SCALE = 0.25
CONTROL_DT = 0.005  # 200Hz
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
GAIT_PERIOD = 0.5


DEFAULT_HOLOSOMA_REPO_ID = "nepyope/holosoma_locomotion"

# 策略文件名映射
POLICY_FILES = {
    "fastsac": "fastsac_g1_29dof.onnx",
    "ppo": "ppo_g1_29dof.onnx",
}


def load_policy(
    repo_id: str = DEFAULT_HOLOSOMA_REPO_ID,
    policy_type: str = "fastsac",
) -> tuple[ort.InferenceSession, np.ndarray, np.ndarray]:
    """加载 Holosoma 行走策略，并从元数据中提取 KP/KD。

    Args:
        repo_id: Hugging Face Hub 仓库 ID
        policy_type: 为 "fastsac"（默认）或 "ppo"

    Returns:
        (policy, kp, kd) 元组
    """
    if policy_type not in POLICY_FILES:
        raise ValueError(f"Unknown policy type: {policy_type}. Choose from: {list(POLICY_FILES.keys())}")

    filename = POLICY_FILES[policy_type]
    logger.info(f"Loading {policy_type.upper()} policy from: {repo_id}/{filename}")
    policy_path = hf_hub_download(repo_id=repo_id, filename=filename)

    policy = ort.InferenceSession(policy_path)
    logger.info(f"Policy loaded: {policy.get_inputs()[0].shape} → {policy.get_outputs()[0].shape}")

    # 从 ONNX 元数据中提取 KP/KD
    model = onnx.load(policy_path, load_external_data=False)
    metadata = {prop.key: prop.value for prop in model.metadata_props}

    if "kp" not in metadata or "kd" not in metadata:
        raise ValueError("ONNX model must contain 'kp' and 'kd' in metadata")

    kp = np.array(json.loads(metadata["kp"]), dtype=np.float32)
    kd = np.array(json.loads(metadata["kd"]), dtype=np.float32)
    logger.info(f"Loaded KP/KD from ONNX ({len(kp)} joints)")

    return policy, kp, kd


class HolosomaLocomotionController(RobotController):
    """用于 Unitree G1 的 Holosoma 下半身行走控制器。"""

    control_dt = CONTROL_DT

    def __init__(self):
        # 加载策略和增益
        self.policy, self.kp, self.kd = load_policy()

        self.default_angles = DEFAULT_ANGLES
        self.cmd = np.zeros(3, dtype=np.float32)

        # 机器人状态
        self.qj = np.zeros(29, dtype=np.float32)
        self.dqj = np.zeros(29, dtype=np.float32)
        self.obs = np.zeros(100, dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)

        # 步态相位
        self.phase = np.array([[0.0, np.pi]], dtype=np.float32)
        self.phase_dt = 2 * np.pi / ((1.0 / CONTROL_DT) * GAIT_PERIOD)
        self.is_standing = True

        logger.info("HolosomaLocomotionController initialized")

    def reset(self) -> None:
        """为新的回合重置内部状态。"""
        self.cmd[:] = 0.0
        self.qj[:] = 0.0
        self.dqj[:] = 0.0
        self.obs[:] = 0.0
        self.last_action[:] = 0.0
        self.phase = np.array([[0.0, np.pi]], dtype=np.float32)
        self.is_standing = True

    def run_step(self, action: dict, lowstate) -> dict:
        """运行行走控制器的一个步骤。

        Args:
            action: 包含 remote.lx/ly/rx/ry 的动作字典
            lowstate: 包含电机位置/速度和 IMU 的机器人低层状态

        Returns:
            下半身关节（0-14）的动作字典
        """
        if lowstate is None:
            return {}

        lx, ly, rx, _ry = (action.get(k, 0.0) for k in REMOTE_AXES)
        ly = ly if abs(ly) > 0.1 else 0.0
        lx = lx if abs(lx) > 0.1 else 0.0
        rx = rx if abs(rx) > 0.1 else 0.0
        ly = np.clip(ly, -0.3, 0.3)
        lx = np.clip(lx, -0.3, 0.3)
        self.cmd[:] = [ly, -lx, -rx]

        # 从低层状态获取关节位置和速度
        for motor in G1_29_JointIndex:
            idx = motor.value
            self.qj[idx] = lowstate.motor_state[idx].q
            self.dqj[idx] = lowstate.motor_state[idx].dq

        # 对策略隐藏手臂位置（改为展示 DEFAULT_ANGLES）
        # 防止策略对手臂的遥操作动作做出反应
        for arm_joint in G1_29_JointArmIndex:
            self.qj[arm_joint.value] = DEFAULT_ANGLES[arm_joint.value]
            self.dqj[arm_joint.value] = 0.0

        # 将 IMU 数据转换到重力参考系下表示
        quat = lowstate.imu_state.quaternion
        ang_vel = np.array(lowstate.imu_state.gyroscope, dtype=np.float32)
        gravity = get_gravity_orientation(quat)

        # 在策略推理前缩放关节位置和速度
        qj_obs = (self.qj - DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj_obs = self.dqj * DOF_VEL_SCALE
        ang_vel_s = ang_vel * ANG_VEL_SCALE

        # 更新步态相位
        if np.linalg.norm(self.cmd[:2]) < 0.01 and abs(self.cmd[2]) < 0.01:
            self.phase[0, :] = np.pi
            self.is_standing = True
        elif self.is_standing:
            self.phase = np.array([[0.0, np.pi]], dtype=np.float32)
            self.is_standing = False
        else:
            self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi

        sin_ph = np.sin(self.phase[0])
        cos_ph = np.cos(self.phase[0])

        # 构建观测
        self.obs[0:29] = self.last_action
        self.obs[29:32] = ang_vel_s
        self.obs[32] = self.cmd[2]
        self.obs[33:35] = self.cmd[:2]
        self.obs[35:37] = cos_ph
        self.obs[37:66] = qj_obs
        self.obs[66:95] = dqj_obs
        self.obs[95:98] = gravity
        self.obs[98:100] = sin_ph

        # 运行策略推理
        ort_in = {self.policy.get_inputs()[0].name: self.obs.reshape(1, -1).astype(np.float32)}
        raw_action = self.policy.run(None, ort_in)[0].squeeze()
        policy_action = np.clip(raw_action, -100.0, 100.0)
        self.last_action = policy_action.copy()

        # 将动作转换回目标关节位置
        target = DEFAULT_ANGLES + policy_action * ACTION_SCALE

        # 构建动作字典（仅前 15 个关节）
        action_dict = {}
        for i in range(15):
            motor_name = G1_29_JointIndex(i).name
            action_dict[f"{motor_name}.q"] = float(target[i])

        return action_dict
