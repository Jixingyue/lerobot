# ------------------------------------------------------------------------------
# Copyright 2025 2toINF and HuggingFace Inc. (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

# =============================================================================
# 注册表
# =============================================================================
ACTION_REGISTRY: dict[str, type[BaseActionSpace]] = {}


def register_action(name: str):
    """用于注册新动作空间的装饰器。"""

    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls

    return _wrap


def build_action_space(name: str, **kwargs) -> BaseActionSpace:
    """按名称实例化一个已注册的动作空间。"""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


# =============================================================================
# 基类
# =============================================================================
class BaseActionSpace(nn.Module):
    """
    所有动作空间定义的抽象基类。

    每个子类需要定义：
      - `dim_action`：动作向量的维度。
      - `gripper_idx`：夹爪通道的索引。
      - `compute_loss(pred, target)`：该空间的监督损失。
      - `preprocess(proprio, action, mode)`：步骤前的修改。
      - `postprocess(action)`：步骤后的修正（例如应用 sigmoid）。
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: tuple[int, ...] = ()

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------------------
    # 核心监督损失
    # ---------------------------------------------------------------------
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """compute_loss 的别名。"""
        return self.compute_loss(pred, target)

    # ---------------------------------------------------------------------
    # 空间级别的钩子
    # ---------------------------------------------------------------------
    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """默认：原样返回。"""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """默认：原样返回。"""
        return action


# =============================================================================
# 工具函数
# =============================================================================
def _ensure_indices_valid(dim_action: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= dim_action]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim dim_action={dim_action}")


# =============================================================================
# 具体实现
# =============================================================================
@register_action("ee6d")
class EE6DActionSpace(BaseActionSpace):
    """末端执行器布局，包含 xyz、6D 旋转和夹爪通道。"""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0

    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        batch_size, seq_len, action_dim = pred.shape
        _ensure_indices_valid(action_dim, self.gripper_idx, "gripper_idx")

        # 夹爪 BCE
        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        # XYZ 位置
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1])
            + self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE

        # 6D 旋转
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1])
            + self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """将 proprio/action 中的夹爪通道置零。"""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """对夹爪 logits 应用 sigmoid。"""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("joint")
class JointActionSpace(BaseActionSpace):
    """关节空间布局，仅包含关节 + 夹爪。"""

    dim_action = 14
    gripper_idx = (6, 13)
    GRIPPER_SCALE = 0.1
    JOINTS_SCALE = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        batch_size, seq_len, action_dim = pred.shape
        _ensure_indices_valid(action_dim, self.gripper_idx, "gripper_idx")

        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        joints_idx = tuple(i for i in range(action_dim) if i not in set(self.gripper_idx))
        joints_loss = self.mse(pred[:, :, joints_idx], target[:, :, joints_idx]) * self.JOINTS_SCALE

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """将 proprio/action 中的夹爪通道置零。"""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """对夹爪 logits 应用 sigmoid。"""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("agibot_ee6d")
class AGIBOTEE6DActionSpace(BaseActionSpace):
    """EE6DActionSpace 的 AGI-bot 变体，对所有分量都使用 MSE。"""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 10.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        batch_size, seq_len, action_dim = pred.shape
        _ensure_indices_valid(action_dim, self.gripper_idx, "gripper_idx")

        gripper_loss = (
            self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx]) * self.GRIPPER_SCALE
        )
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1])
            + self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1])
            + self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """AGIBOT 变体不做预处理。"""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """AGIBOT 不做后处理。"""
        return action


@register_action("franka_joint7")
class FrankaJoint7ActionSpace(BaseActionSpace):
    """
    Franka Panda 关节空间：7 个关节，带夹爪。

    - 真实机器人动作维度：7
    - 面向模型的维度：20（用零填充），
      与期望 20 维输入的预训练 VLA 模型兼容。
    """

    dim_action = 20  # 模型维度
    REAL_DIM = 7  # Franka 的实际关节数

    JOINTS_SCALE = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """将 7 维填充到 20 维（哑通道用零填充）。"""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.REAL_DIM:
            raise ValueError(
                f"Expected last dim to be {self.REAL_DIM} or {self.dim_action}, got {x.size(-1)}"
            )

        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.REAL_DIM]  # 13 个零
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """将模型输出从 20 维裁剪到 7 维。"""
        return x[..., : self.REAL_DIM]

    def compute_loss(self, pred, target):
        """
        pred :  [B, T, 20]
        target : [B, T, 7] 或 [B, T, 20]

        只在前 7 维上计算 MSE。
        """
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)

        assert pred.shape == target.shape

        joints_loss = (
            self.mse(
                pred[:, :, : self.REAL_DIM],  # 只使用前 7 个关节
                target[:, :, : self.REAL_DIM],
            )
            * self.JOINTS_SCALE
        )

        return {"joints_loss": joints_loss}

    def preprocess(self, proprio, action, mode="train"):
        """
        训练期间：
        - 将 [7] 填充到 [20]
        """
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """
        模型预测之后：
        - 将 [20] 裁剪到 [7]，用于真实机器人控制。
        """
        return self._trim_to_real_dim(action)


@register_action("auto")
class AutoActionSpace(BaseActionSpace):
    """
    自动检测的动作空间，可适配任意动作维度。

    - 根据策略特征自动检测真实动作维度
    - 为与预训练模型兼容，模型输出 max_dim 维
    - 损失只在前 real_dim 个维度上计算
    - 后处理时将输出裁剪回 real_dim

    Args:
        real_dim: 来自数据集/策略特征的实际动作维度
        max_dim: 为与预训练 VLA 兼容而使用的模型输出维度
    """

    JOINTS_SCALE = 1.0

    def __init__(self, real_dim: int, max_dim: int):
        super().__init__()
        self.real_dim = real_dim
        self.dim_action = max_dim  # 面向模型的维度
        self.mse = nn.MSELoss()

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """将 real_dim 填充到 max_dim（哑通道用零填充）。"""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.real_dim:
            # 如果维度与两者都不匹配，先填充/裁剪到 real_dim
            if x.size(-1) < self.real_dim:
                pad_shape = list(x.shape[:-1]) + [self.real_dim - x.size(-1)]
                pad = x.new_zeros(pad_shape)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., : self.real_dim]

        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.real_dim]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """将模型输出从 max_dim 裁剪到 real_dim。"""
        return x[..., : self.real_dim]

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        只在前 real_dim 个维度上计算损失。

        pred:   来自模型的 [B, T, max_dim]
        target: [B, T, real_dim] 或 [B, T, max_dim]

        Loss = MSE(pred[:,:,:real_dim], target[:,:,:real_dim])
        """
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"

        # 只在真实维度上计算损失
        joints_loss = (
            self.mse(
                pred[:, :, : self.real_dim],
                target[:, :, : self.real_dim],
            )
            * self.JOINTS_SCALE
        )

        return {"joints_loss": joints_loss}

    def preprocess(self, proprio: torch.Tensor, action: torch.Tensor, mode: str = "train"):
        """
        为模型将动作从 real_dim 填充到 max_dim。
        """
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """
        为真实机器人控制将模型输出从 max_dim 裁剪到 real_dim。
        """
        return self._trim_to_real_dim(action)


@register_action("so101_bimanual")
class BimanualSO101ActionSpace(BaseActionSpace):
    """
    双臂 SO101 机器人：2 条手臂，每条 5 个关节 + 夹爪。

    布局（真实机器人）：
    [left_arm（5 个关节 + 夹爪）, right_arm（5 个关节 + 夹爪）]
    - 左臂： shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    - 右臂： shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

    真实动作维度：12
    面向模型的维度：20（末尾额外的 8 个哑维度）
    """

    # 模型输出 / 训练维度（与预训练策略匹配）
    dim_action = 20

    # 真实机器人动作维度
    REAL_DIM = 12

    # 真实通道与哑通道的索引
    REAL_IDXS = tuple(range(REAL_DIM))  # 0..11
    DUMMY_IDXS = tuple(range(REAL_DIM, dim_action))  # 12..19

    # 夹爪位于真实部分中
    gripper_idx = (5, 11)  # left_gripper 在索引 5，right_gripper 在索引 11
    GRIPPER_SCALE = 1.0
    JOINTS_SCALE = 1.0

    # 左臂和右臂关节的索引（不包含夹爪）
    LEFT_ARM_JOINTS = (0, 1, 2, 3, 4)
    RIGHT_ARM_JOINTS = (6, 7, 8, 9, 10)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    # ---------- 辅助方法 ----------

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """如果最后一维为 REAL_DIM（12），则补零到 dim_action（20）。"""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.REAL_DIM:
            raise ValueError(
                f"Expected last dim to be {self.REAL_DIM} or {self.dim_action}, got {x.size(-1)}"
            )
        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.REAL_DIM]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """为真实机器人只保留前 REAL_DIM（12）维。"""
        return x[..., : self.REAL_DIM]

    # ---------- 损失 ----------

    def compute_loss(self, pred, target):
        """
        pred:  来自模型的 [B, T, 20]
        target: [B, T, 12] 或 [B, T, 20]
        我们将 target 填充到 20 维，并且只在真实维度上计算损失。
        """
        # 确保两者都是 [B, T, 20]
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape == target.shape

        # ---- 对所有真实维度（0–11）计算 MSE ----
        real_dims = 12

        joints_loss = (
            self.mse(
                pred[:, :, :real_dims],
                target[:, :, :real_dims],
            )
            * self.JOINTS_SCALE
        )

        left_arm_loss = self.mse(pred[:, :, :6], target[:, :, :6])
        right_arm_loss = self.mse(pred[:, :, 6:12], target[:, :, 6:12])

        gripper_loss = (
            self.mse(
                pred[:, :, [5, 11]],
                target[:, :, [5, 11]],
            )
            * self.GRIPPER_SCALE
        )

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
            "left_arm_loss": left_arm_loss,
            "right_arm_loss": right_arm_loss,
        }

    # ---------- 预处理 / 后处理 ----------

    def preprocess(self, proprio, action, mode="train"):
        """
        - 如果 proprio/action 是 12 维，则为模型将其填充到 20 维。
        - 将 proprio/action 中的夹爪通道置零，使学习聚焦于关节。
        """
        proprio_m = self._pad_to_model_dim(proprio.clone())
        action_m = self._pad_to_model_dim(action.clone()) if action is not None else None

        proprio_m[..., self.gripper_idx] = 0.0
        if action_m is not None:
            action_m[..., self.gripper_idx] = 0.0

        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """
        - 模型输出 [*, 20]
        - 对夹爪 logits 应用 sigmoid
        - 只为真实机器人返回前 12 维：
          ["left_shoulder_pan.pos",
           "left_shoulder_lift.pos",
           "left_elbow_flex.pos",
           "left_wrist_flex.pos",
           "left_wrist_roll.pos",
           "left_gripper.pos",
           "right_shoulder_pan.pos",
           "right_shoulder_lift.pos",
           "right_elbow_flex.pos",
           "right_wrist_flex.pos",
           "right_wrist_roll.pos",
           "right_gripper.pos"]
        """
        # 确保至少具有真实维度和夹爪
        if action.size(-1) < self.REAL_DIM:
            raise ValueError(f"Expected at least {self.REAL_DIM} dims in action, got {action.size(-1)}")

        # 在模型空间中对夹爪通道（索引 5 和 11）应用 sigmoid
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])

        # 只为环境返回真实的 12 维控制向量
        return self._trim_to_real_dim(action)


# =============================================================================
# 导出
# =============================================================================
__all__ = [
    "BaseActionSpace",
    "build_action_space",
    "register_action",
    "EE6DActionSpace",
    "JointActionSpace",
    "AGIBOTEE6DActionSpace",
    "FrankaJoint7ActionSpace",
    "AutoActionSpace",
    "BimanualSO101ActionSpace",
    "ACTION_REGISTRY",
]
