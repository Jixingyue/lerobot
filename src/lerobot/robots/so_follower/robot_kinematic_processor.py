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

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    TransitionKey,
)
from lerobot.utils.rotation import Rotation

logger = logging.getLogger(__name__)


@ProcessorStepRegistry.register("ee_reference_and_delta")
@dataclass
class EEReferenceAndDelta(RobotActionProcessorStep):
    """
    根据相对增量命令计算末端执行器的目标位姿。

    该步骤接收期望的位置与姿态变化量（`target_*`），并将其叠加到一个参考末端位姿上，
    从而计算出绝对目标位姿。参考位姿通过正运动学由机器人当前关节位置推导得到。

    该处理器可在两种模式下工作：
    1.  `use_latched_reference=True`：在动作首次使能的瞬间“锁存”（保存）参考位姿，
        后续命令均相对于这个固定参考。
    2.  `use_latched_reference=False`：每一步都将参考位姿更新为机器人当前位姿。

    属性:
        kinematics: 用于正运动学计算的机器人运动学模型。
        end_effector_step_sizes: 对输入增量命令进行缩放的字典。
        motor_names: 正运动学计算所需的电机名称列表。
        use_latched_reference: 为 True 时在使能时锁存参考位姿；否则始终以当前位姿
            作为参考。
        reference_ee_pose: 保存锁存参考位姿的内部状态。
        _prev_enabled: 用于检测使能信号上升沿的内部状态。
        _command_when_disabled: 未使能期间保存上一条命令的内部状态。
    """

    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]
    use_latched_reference: bool = (
        True  # 为 True 时在使能时锁存参考；为 False 时始终使用当前位姿
    )
    use_ik_solution: bool = False

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        raw_observation = self.transition.get(TransitionKey.OBSERVATION)

        if raw_observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        observation = raw_observation.copy()

        if self.use_ik_solution and "IK_solution" in self.transition.get(TransitionKey.COMPLEMENTARY_DATA):
            q_raw = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)["IK_solution"]
        else:
            q_raw = np.array(
                [
                    float(v)
                    for k, v in observation.items()
                    if isinstance(k, str)
                    and k.endswith(".pos")
                    and k.removesuffix(".pos") in self.motor_names
                ],
                dtype=float,
            )

        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        # 由实测关节角通过 FK（正运动学）得到当前位姿
        t_curr = self.kinematics.forward_kinematics(q_raw)

        enabled = bool(action.pop("enabled"))
        tx = float(action.pop("target_x"))
        ty = float(action.pop("target_y"))
        tz = float(action.pop("target_z"))
        wx = float(action.pop("target_wx"))
        wy = float(action.pop("target_wy"))
        wz = float(action.pop("target_wz"))
        gripper_vel = float(action.pop("gripper_vel"))

        desired = None

        if enabled:
            ref = t_curr
            if self.use_latched_reference:
                # 锁存参考模式：在上升沿锁存参考位姿
                if not self._prev_enabled or self.reference_ee_pose is None:
                    self.reference_ee_pose = t_curr.copy()
                ref = self.reference_ee_pose if self.reference_ee_pose is not None else t_curr

            delta_p = np.array(
                [
                    tx * self.end_effector_step_sizes["x"],
                    ty * self.end_effector_step_sizes["y"],
                    tz * self.end_effector_step_sizes["z"],
                ],
                dtype=float,
            )
            r_abs = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
            desired = np.eye(4, dtype=float)
            desired[:3, :3] = ref[:3, :3] @ r_abs
            desired[:3, 3] = ref[:3, 3] + delta_p

            self._command_when_disabled = desired.copy()
        else:
            # 未使能时持续发送同一命令，以避免漂移。
            if self._command_when_disabled is None:
                # 如果还从未收到过使能命令，则先冻结当前 FK 位姿一次。
                self._command_when_disabled = t_curr.copy()
            desired = self._command_when_disabled.copy()

        # 写入动作字段
        pos = desired[:3, 3]
        tw = Rotation.from_matrix(desired[:3, :3]).as_rotvec()
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(tw[0])
        action["ee.wy"] = float(tw[1])
        action["ee.wz"] = float(tw[2])
        action["ee.gripper_vel"] = gripper_vel

        self._prev_enabled = enabled
        return action

    def reset(self):
        """重置处理器的内部状态。"""
        self._prev_enabled = False
        self.reference_ee_pose = None
        self._command_when_disabled = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION].pop(f"{feat}", None)

        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_vel"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety(RobotActionProcessorStep):
    """
    将末端执行器位姿裁剪到预定义边界内，并检查不安全的跳变。

    该步骤确保末端执行器目标位姿始终处于安全工作空间内。它还会对命令加以缓和，
    防止相邻步骤之间出现大幅、突然的运动。

    属性:
        end_effector_bounds: 带有 "min" 和 "max" 键的字典，用于位置裁剪。
        max_ee_step_m: 相邻步骤之间允许的最大位置变化量（单位：米）。
        raise_on_jump: 为 ``True``（默认）时，单帧步长超限会抛出 ``ValueError``
            （中止控制循环）。为 ``False`` 时，步长会被限速到 ``max_ee_step_m``，
            并改为记录一条警告——对于实时遥操作，这是更安全的选择，因为瞬时的
            跟踪故障不应导致循环崩溃而让机器人失去控制。
        _last_pos: 保存上一次命令位置的内部状态。
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05
    raise_on_jump: bool = True
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        x = action["ee.x"]
        y = action["ee.y"]
        z = action["ee.z"]
        wx = action["ee.wx"]
        wy = action["ee.wy"]
        wz = action["ee.wz"]
        # TODO(Steven)：ee.gripper_vel 不需要设界

        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present in action"
            )

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)

        # 裁剪位置
        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        # 检查位置跳变
        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                # 将步长钳制到单帧上限（限速）。无论如何都会计算钳制后的值；
                # raise_on_jump 只决定超限步长是中止循环，还是被限速并发出警告。
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)
                if self.raise_on_jump:
                    raise ValueError(f"EE jump {n:.3f}m > {self.max_ee_step_m}m")
                logger.warning(
                    "EE jump %.3fm > %.3fm; rate-limited to the per-frame step "
                    "(likely a transient tracking glitch; if it recurs every frame "
                    "the commanded target is systematically out of workspace).",
                    n,
                    self.max_ee_step_m,
                )

        self._last_pos = pos

        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(twist[0])
        action["ee.wy"] = float(twist[1])
        action["ee.wz"] = float(twist[2])
        return action

    def reset(self):
        """重置上一次已知的位置和姿态。"""
        self._last_pos = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints(RobotActionProcessorStep):
    """
    使用逆运动学（IK）根据末端执行器目标位姿计算期望关节位置。

    该步骤将笛卡尔空间命令（末端执行器的位置和姿态）转换为各个电机对应的
    关节空间命令。

    属性:
        kinematics: 用于逆运动学计算的机器人运动学模型。
        motor_names: 需要计算关节位置的电机名称列表。
        q_curr: 保存上一次关节位置的内部状态，用作 IK 求解器的初始猜测。
        initial_guess_current_joints: 为 True 时，以机器人当前关节状态作为 IK
            猜测；为 False 时，使用上一步的求解结果。
        orientation_weight: 传递给 ``RobotKinematics.inverse_kinematics`` 的
            姿态约束权重。默认为 ``0.01``（与求解器默认值一致，因此现有调用方
            行为不变）。对于欠驱动机械臂，可设为 ``0.0`` 以仅对位置做 IK；
            较小的非零权重可在 5 自由度 SO-101 上实现软姿态 IK——其手腕只能
            部分跟踪姿态（位置占主导）。
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True
    orientation_weight: float = 0.01

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        raw_observation = self.transition.get(TransitionKey.OBSERVATION)
        if raw_observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        observation = raw_observation.copy()

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # 以当前关节角作为初始猜测
            self.q_curr = q_raw
        else:  # 以上一次 ik 求解结果作为初始猜测
            if self.q_curr is None:
                self.q_curr = q_raw

        # 由位置 + 旋转向量（twist）构建期望的 4x4 变换矩阵
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # 计算逆运动学
        q_target = self.kinematics.inverse_kinematics(
            self.q_curr, t_des, orientation_weight=self.orientation_weight
        )
        self.q_curr = q_target

        # TODO：此处对 motor_names 与 q_target 的映射顺序很敏感
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """重置 IK 求解器的初始猜测。"""
        self.q_curr = None


@ProcessorStepRegistry.register("gripper_velocity_to_joint")
@dataclass
class GripperVelocityToJoint(RobotActionProcessorStep):
    """
    将夹爪速度命令转换为夹爪关节目标位置。

    该步骤以当前夹爪位置为起点，对归一化速度命令随时间进行积分以生成位置命令。
    它还支持离散模式：整数动作分别映射为张开、闭合或无操作。

    属性:
        motor_names: 电机名称列表，其中必须包含 'gripper'。
        speed_factor: 将归一化速度命令转换为位置变化量的缩放系数。
        clip_min: 允许的夹爪关节最小位置。
        clip_max: 允许的夹爪关节最大位置。
        discrete_gripper: 为 True 时，将输入解释为离散类别索引
            {0 = 闭合, 1 = 保持, 2 = 张开}，与 `GamepadTeleop.GripperAction` 一致。
    """

    speed_factor: float = 20.0
    clip_min: float = 0.0
    clip_max: float = 100.0
    discrete_gripper: bool = False

    def action(self, action: RobotAction) -> RobotAction:
        raw_observation = self.transition.get(TransitionKey.OBSERVATION)

        gripper_vel = action.pop("ee.gripper_vel")

        if raw_observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        observation = raw_observation.copy()

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.discrete_gripper:
            # 将离散命令 {0=闭合, 1=保持, 2=张开} 映射为带符号速度。
            # 取负以适配 SO100 的符号约定（闭合时关节位置增大）。
            #   0 -> +clip_max（闭合），1 -> 0（保持），2 -> -clip_max（张开）
            gripper_vel = -(gripper_vel - 1) * self.clip_max

        # 计算期望夹爪位置
        delta = gripper_vel * float(self.speed_factor)
        # TODO：这里假设夹爪是机器人中最后指定的关节
        gripper_pos = float(np.clip(q_raw[-1] + delta, self.clip_min, self.clip_max))
        action["ee.gripper_pos"] = gripper_pos

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("ee.gripper_vel", None)
        features[PipelineFeatureType.ACTION]["ee.gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )

        return features


def compute_forward_kinematics_joints_to_ee(
    joints: dict[str, Any], kinematics: RobotKinematics, motor_names: list[str]
) -> dict[str, Any]:
    motor_joint_values = [joints[f"{n}.pos"] for n in motor_names]

    q = np.array(motor_joint_values, dtype=float)
    t = kinematics.forward_kinematics(q)
    pos = t[:3, 3]
    tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    gripper_pos = joints["gripper.pos"]
    for n in motor_names:
        joints.pop(f"{n}.pos")
    joints["ee.x"] = float(pos[0])
    joints["ee.y"] = float(pos[1])
    joints["ee.z"] = float(pos[2])
    joints["ee.wx"] = float(tw[0])
    joints["ee.wy"] = float(tw[1])
    joints["ee.wz"] = float(tw[2])
    joints["ee.gripper_pos"] = float(gripper_pos)
    return joints


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_observation")
@dataclass
class ForwardKinematicsJointsToEEObservation(ObservationProcessorStep):
    """
    使用正运动学（FK）根据关节位置计算末端执行器位姿。

    该步骤通常用于将机器人的笛卡尔位姿加入观测空间，可用于可视化或作为
    策略的输入。

    属性:
        kinematics: 机器人运动学模型。
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def observation(self, observation: RobotObservation) -> RobotObservation:
        return compute_forward_kinematics_joints_to_ee(observation, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 数据集中只使用末端位姿，因此不需要关节位置
        for n in self.motor_names:
            features[PipelineFeatureType.OBSERVATION].pop(f"{n}.pos", None)
        # 指定本步骤中需要存入数据集的数据集特征
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.OBSERVATION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_action")
@dataclass
class ForwardKinematicsJointsToEEAction(RobotActionProcessorStep):
    """
    使用正运动学（FK）根据关节位置计算末端执行器位姿。

    该步骤通常用于将机器人的笛卡尔位姿加入观测空间，可用于可视化或作为
    策略的输入。

    属性:
        kinematics: 机器人运动学模型。
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def action(self, action: RobotAction) -> RobotAction:
        return compute_forward_kinematics_joints_to_ee(action, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 数据集中只使用末端位姿，因此不需要关节位置
        for n in self.motor_names:
            features[PipelineFeatureType.ACTION].pop(f"{n}.pos", None)
        # 在数据集 schema 中将末端执行器特征存为动作
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register(name="forward_kinematics_joints_to_ee")
@dataclass
class ForwardKinematicsJointsToEE(ProcessorStep):
    kinematics: RobotKinematics
    motor_names: list[str]

    def __post_init__(self):
        self.joints_to_ee_action_processor = ForwardKinematicsJointsToEEAction(
            kinematics=self.kinematics, motor_names=self.motor_names
        )
        self.joints_to_ee_observation_processor = ForwardKinematicsJointsToEEObservation(
            kinematics=self.kinematics, motor_names=self.motor_names
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if transition.get(TransitionKey.ACTION) is not None:
            transition = self.joints_to_ee_action_processor(transition)
        if transition.get(TransitionKey.OBSERVATION) is not None:
            transition = self.joints_to_ee_observation_processor(transition)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if features[PipelineFeatureType.ACTION] is not None:
            features = self.joints_to_ee_action_processor.transform_features(features)
        if features[PipelineFeatureType.OBSERVATION] is not None:
            features = self.joints_to_ee_observation_processor.transform_features(features)
        return features


@ProcessorStepRegistry.register("inverse_kinematics_rl_step")
@dataclass
class InverseKinematicsRLStep(ProcessorStep):
    """
    使用逆运动学（IK）根据末端执行器目标位姿计算期望关节位置。

    这是在 InverseKinematicsEEToJoints 步骤的基础上修改而来，用于 RL 流水线。
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = dict(transition)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("Action is required for InverseKinematicsEEToJoints")
        action = dict(action)

        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        raw_observation = new_transition.get(TransitionKey.OBSERVATION)
        if raw_observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        observation = raw_observation.copy()

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # 以当前关节角作为初始猜测
            self.q_curr = q_raw
        else:  # 以上一次 ik 求解结果作为初始猜测
            if self.q_curr is None:
                self.q_curr = q_raw

        # 由位置 + 旋转向量（twist）构建期望的 4x4 变换矩阵
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # 计算逆运动学
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        # TODO：此处对 motor_names 与 q_target 的映射顺序很敏感
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        new_transition[TransitionKey.ACTION] = action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data["IK_solution"] = q_target
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """重置 IK 求解器的初始猜测。"""
        self.q_curr = None
