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

from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import require_package

_placo_runtime_error: ImportError | None = None

if TYPE_CHECKING:
    import placo  # type: ignore[import-not-found]
else:
    try:
        import placo  # type: ignore[import-not-found]
    except ImportError as _placo_import_err:
        placo = None
        _placo_runtime_error = _placo_import_err


def _raise_if_placo_unusable() -> None:
    if placo is None and _placo_runtime_error is not None:
        raise ImportError(
            f"placo is installed but failed to import: {_placo_runtime_error!s}"
        ) from _placo_runtime_error


class RobotKinematics:
    """使用 placo 库进行正运动学和逆运动学计算的机器人运动学。"""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        """
        初始化基于 placo 的运动学求解器。

        Args:
            urdf_path (str): 机器人 URDF 文件的路径
            target_frame_name (str): URDF 中末端执行器坐标系的名称
            joint_names (list[str] | None): 运动学求解器使用的关节名称列表
        """
        require_package("placo", extra="placo-dep")
        _raise_if_placo_unusable()

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # 固定基座

        self.target_frame_name = target_frame_name

        # 设置关节名称
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # 为逆运动学初始化坐标系任务
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        根据构造函数中给定的目标坐标系名称，为给定的关节配置计算正运动学。

        Args:
            joint_pos_deg: 以度为单位的关节位置（numpy 数组）

        Returns:
            末端执行器位姿的 4x4 变换矩阵
        """

        # 将角度转换为弧度
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])

        # 更新 placo 机器人中的关节位置
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        # 更新运动学
        self.robot.update_kinematics()

        # 获取变换矩阵
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        max_iters: int = 8,
    ) -> np.ndarray:
        """
        使用 placo 求解器计算逆运动学。

        Args:
            current_joint_pos: 以度为单位的当前关节位置（用作初始猜测）
            desired_ee_pose: 以 4x4 变换矩阵表示的目标末端执行器位姿
            position_weight: 逆运动学中位置约束的权重
            orientation_weight: 逆运动学中姿态约束的权重，设为 0.0 则仅约束位置
            max_iters: 要运行的 placo 牛顿迭代步数。

        Returns:
            能达到期望末端执行器位姿的以度为单位的关节位置
        """

        # 将当前关节位置转换为弧度作为初始猜测
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])

        # 将当前关节位置设为初始猜测
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[i])

        # 更新坐标系任务的目标位姿
        self.tip_frame.T_world_frame = desired_ee_pose

        # 根据 position_only 标志配置任务
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

        # 求解逆运动学。
        for _ in range(max_iters):
            self.solver.solve(True)
            self.robot.update_kinematics()

        # 提取关节位置
        joint_pos_rad = []
        for joint_name in self.joint_names:
            joint = self.robot.get_joint(joint_name)
            joint_pos_rad.append(joint)

        # 转换回角度
        joint_pos_deg = np.rad2deg(joint_pos_rad)

        # 如果 current_joint_pos 中存在夹爪位置，则保留它
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_pos_deg
