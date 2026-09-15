# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import (
    IMAGENET_STATS,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
)

from .configuration_xvla import XVLAConfig
from .utils import rotate6d_to_axis_angle


def make_xvla_pre_post_processors(
    config: XVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为 XVLA 构建 LeRobot 处理器流水线。
    """

    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding=config.pad_language_to,
            padding_side=config.tokenizer_padding_side,
        ),
        XVLAImageToFloatProcessorStep(),
        XVLAImageNetNormalizeProcessorStep(),
        XVLAAddDomainIdProcessorStep(),
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


# XVLA 自定义处理器步骤
@dataclass
class LiberoProcessorStep(ObservationProcessorStep):
    """
    将 LIBERO 观测处理为 LeRobot 格式。

    该步骤处理来自 LIBERO 环境的特定观测结构，
    其中包含嵌套的 robot_state 字典和图像观测。

    **状态处理：**
    -   处理 `robot_state` 字典，其中包含嵌套的末端执行器、
        夹爪和关节信息。
    -   提取并拼接：
        - 末端执行器位置（3D）
        - 转换为轴角的末端执行器四元数（3D）
        - 夹爪关节位置（2D）
    -   将拼接后的状态映射到 `"observation.state"`。

    **图像处理：**
    -   通过同时翻转高度和宽度维度，将图像旋转 180 度。
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

                if key == f"{OBS_IMAGES}.image":
                    # 同时翻转 H 和 W
                    img = torch.flip(img, dims=[2, 3])

                processed_obs[key] = img
        # 将 robot_state 处理为扁平的状态向量
        robot_state_str = OBS_PREFIX + "robot_state"
        if robot_state_str in processed_obs:
            robot_state = processed_obs.pop(robot_state_str)

            # 提取各组成部分
            eef_pos = robot_state["eef"]["pos"]  # (B, 3,)
            eef_mat = robot_state["eef"]["mat"]  # (B, 3, 3)
            eef_rot6d = self._mat_to_rotate6d(eef_mat)  # (B, 6)

            extra = torch.zeros((eef_pos.shape[0], 1), dtype=torch.float32, device=eef_pos.device)

            proprio_state = torch.cat((eef_pos, eef_rot6d, extra), dim=-1)  # (B, 10)
            state = torch.cat((proprio_state, torch.zeros_like(proprio_state)), dim=-1)  # (B, 20)
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
            if ft != PipelineFeatureType.STATE:
                new_features[ft] = feats.copy()

        # 重建 STATE 特征
        state_feats = {}

        # 添加我们新的扁平化状态
        state_feats[OBS_STATE] = PolicyFeature(
            key=OBS_STATE,
            shape=(20,),
            dtype="float32",
        )

        new_features[PipelineFeatureType.STATE] = state_feats

        return new_features

    def _mat_to_rotate6d(self, rot_mats: torch.Tensor) -> torch.Tensor:
        """
        将批量旋转矩阵 (B, 3, 3) 转换为 6D 旋转表示 (B, 6)。

        Args:
            rot_mats (Tensor): 形状为 (B, 3, 3) 的旋转矩阵

        Returns:
            Tensor: 6D 旋转表示，形状 (B, 6)

        Raises:
            TypeError: 输入不是 torch 张量时
            ValueError: 形状不是 (B, 3, 3) 时
        """

        if not isinstance(rot_mats, torch.Tensor):
            raise TypeError(f"mat_to_rot6d expects a torch.Tensor, got {type(rot_mats)}")

        if rot_mats.ndim != 3 or rot_mats.shape[1:] != (3, 3):
            raise ValueError(f"mat_to_rot6d expects shape (B, 3, 3), got {tuple(rot_mats.shape)}")

        rot_mats = rot_mats.to(torch.float32)

        col1 = rot_mats[:, :3, 0]  # (B, 3)
        col2 = rot_mats[:, :3, 1]  # (B, 3)

        rot6d = torch.cat([col1, col2], dim=-1)  # (B, 6)

        return rot6d

    def observation(self, observation):
        return self._process_observation(observation)


@dataclass
@ProcessorStepRegistry.register(name="xvla_image_scale")
class XVLAImageScaleProcessorStep(ProcessorStep):
    """将图像观测乘以 255，以从 [0, 1] 范围转换到 [0, 255] 范围。

    该处理器步骤会将所有图像观测乘以 255，期望图像处于类 uint8
    范围的 XVLA 模型需要这一处理。

    Args:
        image_keys: 包含待缩放图像的观测键列表。
                   如果为 None，将自动检测以 "observation.images." 开头的键。
    """

    image_keys: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将图像观测乘以 255。"""
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition

        # 复制一份观测，以避免修改原始数据
        obs = obs.copy()

        # 确定要缩放哪些键
        keys_to_scale = self.image_keys
        if keys_to_scale is None:
            # 自动检测图像键
            keys_to_scale = [k for k in obs if k.startswith(OBS_IMAGES)]

        # 缩放每张图像
        for key in keys_to_scale:
            if key in obs and isinstance(obs[key], torch.Tensor):
                obs[key] = obs[key] * 255

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        """图像缩放不会改变特征结构。"""
        return features

    def get_config(self) -> dict[str, Any]:
        """返回可序列化的配置。"""
        return {
            "image_keys": self.image_keys,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_image_to_float")
class XVLAImageToFloatProcessorStep(ProcessorStep):
    """将图像观测从 [0, 255] 范围转换到 [0, 1] 范围。

    该处理器步骤会将图像观测除以 255，从类 uint8 的
    [0, 255] 范围转换为浮点的 [0, 1] 范围。通常在加载
    以 uint8 值存储的图像时使用。

    Args:
        image_keys: 包含待转换图像的观测键列表。
                   如果为 None，将自动检测以 "observation.images." 开头的键。
        validate_range: 如果为 True，校验输入值是否在 [0, 255] 范围内（默认：True）

    Raises:
        ValueError: 当 validate_range 为 True 且图像值不在 [0, 255] 范围内时。
    """

    image_keys: list[str] | None = None
    validate_range: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将图像观测从 [0, 255] 转换到 [0, 1]。"""
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition

        # 复制一份观测，以避免修改原始数据
        obs = obs.copy()

        # 确定要转换哪些键
        keys_to_convert = self.image_keys
        if keys_to_convert is None:
            # 自动检测图像键
            keys_to_convert = [k for k in obs if k.startswith(OBS_IMAGES)]

        # 转换每张图像
        for key in keys_to_convert:
            if key in obs and isinstance(obs[key], torch.Tensor):
                tensor = obs[key]

                min_val = tensor.min().item()
                max_val = tensor.max().item()

                if max_val <= 1.0:
                    obs[key] = tensor.float()  # 确保为 float dtype，但不做除法
                    continue
                # 如果要求，则校验值是否在 [0, 255] 范围内
                if self.validate_range and (min_val < 0.0 or max_val > 255.0):
                    raise ValueError(
                        f"Image '{key}' has values outside [0, 255] range: "
                        f"min={min_val:.4f}, max={max_val:.4f}. "
                        f"Cannot convert to [0, 1] range."
                    )

                # 转换为 float 并除以 255
                obs[key] = tensor.float() / 255.0

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        """图像转换不会改变特征结构。"""
        return features

    def get_config(self) -> dict[str, Any]:
        """返回可序列化的配置。"""
        return {
            "image_keys": self.image_keys,
            "validate_range": self.validate_range,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_imagenet_normalize")
class XVLAImageNetNormalizeProcessorStep(ProcessorStep):
    """使用 ImageNet 统计量对图像观测进行归一化。

    该处理器步骤对图像观测应用 ImageNet 归一化（均值和标准差）。
    在归一化之前会校验输入值是否处于 [0, 1] 范围内。

    归一化公式为：(image - mean) / std

    Args:
        image_keys: 包含待归一化图像的观测键列表。
                   如果为 None，将自动检测以 "observation.images." 开头的键。

    Raises:
        ValueError: 当图像值不在 [0, 1] 范围内时。
    """

    image_keys: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """使用 ImageNet 统计量对图像观测进行归一化。"""
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition

        # 复制一份观测，以避免修改原始数据
        obs = obs.copy()

        # 确定要归一化哪些键
        keys_to_normalize = self.image_keys
        if keys_to_normalize is None:
            # 自动检测图像键
            keys_to_normalize = [k for k in obs if k.startswith(OBS_IMAGES)]

        # 对每张图像进行归一化
        for key in keys_to_normalize:
            if key in obs and isinstance(obs[key], torch.Tensor):
                tensor = obs[key]

                # 校验值是否在 [0, 1] 范围内
                min_val = tensor.min().item()
                max_val = tensor.max().item()
                if min_val < 0.0 or max_val > 1.0:
                    raise ValueError(
                        f"Image '{key}' has values outside [0, 1] range: "
                        f"min={min_val:.4f}, max={max_val:.4f}. "
                        f"ImageNet normalization requires input values in [0, 1]."
                    )

                # 应用 ImageNet 归一化
                mean = torch.tensor(IMAGENET_STATS["mean"], device=tensor.device, dtype=tensor.dtype)
                std = torch.tensor(IMAGENET_STATS["std"], device=tensor.device, dtype=tensor.dtype)

                # 扩展 mean/std 以匹配张量维度（例如 BCHW 或 BNCHW）
                while mean.dim() < tensor.dim():
                    mean = mean.unsqueeze(0)
                    std = std.unsqueeze(0)

                # 归一化：(image - mean) / std
                obs[key] = (tensor - mean) / std

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        """ImageNet 归一化不会改变特征结构。"""
        return features

    def get_config(self) -> dict[str, Any]:
        """返回可序列化的配置。"""
        return {
            "image_keys": self.image_keys,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_add_domain_id")
class XVLAAddDomainIdProcessorStep(ProcessorStep):
    """向 complementary data 中添加 domain_id。

    该处理器步骤会向 complementary data 添加一个 domain_id 张量，
    XVLA 用它来识别不同的机器人本体或任务域。

    Args:
        domain_id: 要添加的域 ID（默认：3）
    """

    domain_id: int = 0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """向 complementary data 中添加 domain_id。"""
        new_transition = transition.copy()
        comp = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        comp = {} if comp is None else comp.copy()

        # 从观测张量推断 batch size
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        batch_size = 1
        if obs:
            for v in obs.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.shape[0]
                    break

        # 添加 domain_id 张量
        comp["domain_id"] = torch.tensor([int(self.domain_id)] * batch_size, dtype=torch.long)

        new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return new_transition

    def transform_features(self, features):
        """添加域 ID 不会改变特征结构。"""
        return features

    def get_config(self) -> dict[str, Any]:
        """返回可序列化的配置。"""
        return {
            "domain_id": self.domain_id,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_rotation_6d_to_axis_angle")
class XVLARotation6DToAxisAngleProcessorStep(ProcessorStep):
    """将 6D 旋转表示转换为轴角，并重新组织动作维度。

    该处理器步骤接收带 6D 旋转表示的动作，并将其转换为
    轴角表示，同时按如下方式重新组织动作维度：
    - action[:, :3] -> target_eef（末端执行器位置）
    - action[:, 3:9] -> 6D 旋转（转换为 3D 轴角）
    - action[:, 9:10] -> 夹爪动作

    最终输出：[target_eef (3), axis_angle (3), gripper (1)] = 7 维动作

    Args:
        expected_action_dim: 预期的输入动作维度（默认：10，支持 6D 旋转加额外维度）
    """

    expected_action_dim: int = 10

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """将动作中的 6D 旋转转换为轴角。"""
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is None or not isinstance(action, torch.Tensor):
            return new_transition

        # 转换为 numpy 以便处理
        device = action.device
        dtype = action.dtype
        action_np = action.cpu().numpy()

        # 提取各组成部分
        # action 形状：(B, D)，其中 D >= 10
        target_eef = action_np[:, :3]  # (B, 3)
        rotation_6d = action_np[:, 3:9]  # (B, 6)
        target_act = action_np[:, 9:10]  # (B, 1)

        # 将 6D 旋转转换为轴角
        target_axis = rotate6d_to_axis_angle(rotation_6d)  # (B, 3)

        # 拼接：[eef (3), axis_angle (3), gripper (1)] = 7 维
        action_np = np.concatenate([target_eef, target_axis, target_act], axis=-1)

        # 将夹爪动作转换为 -1 或 1
        action_np[:, -1] = np.where(action_np[:, -1] > 0.5, 1.0, -1.0)

        # 转换回张量
        action = torch.from_numpy(action_np).to(device=device, dtype=dtype)

        new_transition[TransitionKey.ACTION] = action
        return new_transition

    def transform_features(self, features):
        """旋转转换会将动作维度从 10 变为 7。"""
        # 注意：这是简化版本。实际中可能还需要
        # 更新 features 字典中的动作特征形状。
        return features

    def get_config(self) -> dict[str, Any]:
        """返回可序列化的配置。"""
        return {
            "expected_action_dim": self.expected_action_dim,
        }


def make_xvla_libero_pre_post_processors() -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    为配合 LIBERO 环境使用的 XVLA 构建 LeRobot 处理器流水线。
    """
    pre_processor_steps: list[ProcessorStep] = []
    post_processor_steps: list[ProcessorStep] = []
    pre_processor_steps.extend(
        [LiberoProcessorStep(), XVLAImageNetNormalizeProcessorStep(), XVLAAddDomainIdProcessorStep()]
    )
    post_processor_steps.extend([XVLARotation6DToAxisAngleProcessorStep()])
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=pre_processor_steps,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=post_processor_steps,
        ),
    )
