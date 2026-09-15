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
from dataclasses import dataclass

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX, OBS_STATE, OBS_STR

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="libero_processor")
class LiberoProcessorStep(ObservationProcessorStep):
    """
    将 LIBERO 观测处理为 LeRobot 格式。

    该步骤处理来自 LIBERO 环境的特定观测结构，
    其中包括嵌套的 robot_state 字典和图像观测。

    **状态处理：**
    -   处理 `robot_state` 字典，其中包含嵌套的末端执行器、
        夹爪和关节信息。
    -   提取并拼接：
        - 末端执行器位置（3D）
        - 转换为轴角表示的末端执行器四元数（3D）
        - 夹爪关节位置（2D）
    -   将拼接后的状态映射到 `"observation.state"`。

    **图像处理：**
    -   通过翻转高度和宽度两个维度将图像旋转 180 度。
    -   这是为了适配 HuggingFaceVLA/libero 的相机朝向约定。
    """

    def _process_observation(self, observation):
        """
        处理来自 LIBERO 的图像和 robot_state 观测。
        """
        processed_obs = observation.copy()
        for key in list(processed_obs.keys()):
            if key.startswith(f"{OBS_IMAGES}."):
                img = processed_obs[key]

                # 同时翻转 H 和 W
                img = torch.flip(img, dims=[2, 3])

                processed_obs[key] = img
        # 将 robot_state 处理为扁平的状态向量
        observation_robot_state_str = OBS_PREFIX + "robot_state"
        if observation_robot_state_str in processed_obs:
            robot_state = processed_obs.pop(observation_robot_state_str)

            # 提取各分量
            eef_pos = robot_state["eef"]["pos"]  # (B, 3,)
            eef_quat = robot_state["eef"]["quat"]  # (B, 4,)
            gripper_qpos = robot_state["gripper"]["qpos"]  # (B, 2,)

            # 将四元数转换为轴角表示
            eef_axisangle = self._quat2axisangle(eef_quat)  # (B, 3)
            # 拼接为单个状态向量
            state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)

            # 确保为 float32
            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)

            processed_obs[OBS_STATE] = state
        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        将特征键从 LIBERO 格式转换为 LeRobot 标准格式。
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}

        # 复制非 STATE 特征
        for ft, feats in features.items():
            if ft != FeatureType.STATE:
                new_features[ft] = feats.copy()

        # 重建 STATE 特征
        state_feats = {}

        # 添加新的扁平化状态
        state_feats[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(8,),  # [eef_pos(3), axis_angle(3), gripper(2)]
        )

        new_features[FeatureType.STATE] = state_feats

        return new_features

    def observation(self, observation):
        return self._process_observation(observation)

    def _quat2axisangle(self, quat: torch.Tensor) -> torch.Tensor:
        """
        将批量的四元数转换为轴角格式。
        仅接受形状为 (B, 4) 的 torch 张量。

        Args:
            quat (Tensor): 形状为 (B, 4)、格式为 (x, y, z, w) 的四元数张量

        Returns:
            Tensor: 形状为 (B, 3) 的轴角向量

        Raises:
            TypeError: 如果输入不是 torch 张量
            ValueError: 如果形状不是 (B, 4)
        """

        if not isinstance(quat, torch.Tensor):
            raise TypeError(f"_quat2axisangle expected a torch.Tensor, got {type(quat)}")

        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError(f"_quat2axisangle expected shape (B, 4), got {tuple(quat.shape)}")

        quat = quat.to(dtype=torch.float32)
        device = quat.device
        batch_size = quat.shape[0]

        w = quat[:, 3].clamp(-1.0, 1.0)

        den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))

        result = torch.zeros((batch_size, 3), device=device)

        mask = den > 1e-10

        if mask.any():
            angle = 2.0 * torch.acos(w[mask])  # (M,)
            axis = quat[mask, :3] / den[mask].unsqueeze(1)
            result[mask] = axis * angle.unsqueeze(1)

        return result


@dataclass
@ProcessorStepRegistry.register(name="isaaclab_arena_processor")
class IsaaclabArenaProcessorStep(ObservationProcessorStep):
    """
    将 IsaacLab Arena 观测处理为 LeRobot 格式。

    **状态处理：**
    - 根据 `state_keys` 从 obs["policy"] 中提取状态分量。
    - 拼接为扁平向量并映射到 "observation.state"。

    **图像处理：**
    - 根据 `camera_keys` 从 obs["camera_obs"] 中提取图像。
    - 从 (B, H, W, C) uint8 转换为 (B, C, H, W) float32 [0, 1]。
    - 映射到 "observation.images.<camera_name>"。
    """

    # 可通过 IsaacLabEnv 配置 / 命令行参数配置：--env.state_keys="robot_joint_pos,left_eef_pos"
    state_keys: tuple[str, ...]

    # 可通过 IsaacLabEnv 配置 / 命令行参数配置：--env.camera_keys="robot_pov_cam_rgb"
    camera_keys: tuple[str, ...]

    def _process_observation(self, observation):
        """
        处理来自 IsaacLab Arena 的图像和策略状态观测。
        """
        processed_obs = {}

        if f"{OBS_STR}.camera_obs" in observation:
            camera_obs = observation[f"{OBS_STR}.camera_obs"]

            for cam_name, img in camera_obs.items():
                if cam_name not in self.camera_keys:
                    continue

                img = img.permute(0, 3, 1, 2).contiguous()
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                elif img.dtype != torch.float32:
                    img = img.float()

                processed_obs[f"{OBS_IMAGES}.{cam_name}"] = img

        # 处理策略状态 -> observation.state
        if f"{OBS_STR}.policy" in observation:
            policy_obs = observation[f"{OBS_STR}.policy"]

            # 按顺序收集状态分量
            state_components = []
            for key in self.state_keys:
                if key in policy_obs:
                    component = policy_obs[key]
                    # 展平多余的维度：(B, N, M) -> (B, N*M)
                    if component.dim() > 2:
                        batch_size = component.shape[0]
                        component = component.view(batch_size, -1)
                    state_components.append(component)

            if state_components:
                state = torch.cat(state_components, dim=-1)
                state = state.float()
                processed_obs[OBS_STATE] = state

        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """不用于策略评估。"""
        return features

    def observation(self, observation):
        return self._process_observation(observation)
