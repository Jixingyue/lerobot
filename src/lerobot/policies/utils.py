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

import logging
from collections import deque

import numpy as np
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.lerobot_types import PolicyAction, RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame


def populate_queues(
    queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None
):
    """将批次中的值追加到各自按键对应的 deque 中（首次使用时填满它们）。

    使用此辅助函数的策略会将该字典保存为 ``self._queues``，这是
    ``PreTrainedPolicy._action_queue_attrs`` 知道如何清除的名称之一。
    """
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # 忽略尚不在队列中的键（由调用者负责确保队列包含所需的键）。
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # 通过多次复制第一个观测值来初始化，直到队列被填满
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # 将最新观测值添加到队列
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """通过检查模块的某个参数来获取模块的设备。

    注意：假设所有参数都在同一设备上
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """通过检查模块的某个参数来获取模块参数的 dtype。

    注意：假设所有参数具有相同的 dtype。
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    根据输入形状计算 PyTorch 模块的输出形状。

    Args:
        module (nn.Module): 一个 PyTorch 模块
        input_shape (tuple): 表示输入形状的元组，例如 (batch_size, channels, height, width)

    Returns:
        tuple: 模块的输出形状。
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)


def log_model_loading_keys(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    """在加载模型时记录缺失和意外的键。

    Args:
        missing_keys (list[str]): 期望存在但未找到的键。
        unexpected_keys (list[str]): 找到但不期望存在的键。
    """
    if missing_keys:
        logging.warning(f"Missing key(s) when loading model: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected key(s) when loading model: {unexpected_keys}")


# TODO(Steven): 将此函数移到合适的预处理器步骤中
def prepare_observation_for_inference(
    observation: dict[str, np.ndarray],
    device: torch.device,
    task: str | None = None,
    robot_type: str | None = None,
) -> RobotObservation:
    """将观测数据转换为模型可用的 PyTorch 张量。

    该函数接收一个 NumPy 数组字典，执行必要的预处理，
    并为模型推理做好准备。步骤包括：
    1. 将 NumPy 数组转换为 PyTorch 张量。
    2. 将每个张量移动到指定的计算设备。
    3. 在该设备上对图像数据进行归一化和维度置换（如果有的话）。
    4. 为每个张量添加批次维度。
    5. 向字典中添加任务和机器人类型信息。

    图像以紧凑的 ``uint8`` 帧形式传输到设备——比其 float32 对应物
    小 4 倍的拷贝——并且 ``/255`` + 置换在设备上运行，
    使控制循环每个 tick 的 CPU 开销保持较低。

    Args:
        observation: 将观测名称（str）映射到 NumPy 数组数据的字典。
            对于图像，期望的格式为 (H, W, C)。
        device: 张量将被移动到的 PyTorch 设备（例如 'cpu' 或 'cuda'）。
        task: 当前任务的可选字符串标识符。
        robot_type: 所用机器人的可选字符串标识符。

    Returns:
        一个字典，其值是为推理预处理好的、位于目标设备上的
        PyTorch 张量。图像张量被重塑为 (C, H, W) 并归一化到 [0, 1] 范围。
    """
    for name in observation:
        tensor = torch.from_numpy(observation[name]).to(device)
        if "image" in name:
            if tensor.dtype == torch.uint8:
                tensor = tensor.type(torch.float32) / 255
            tensor = tensor.permute(2, 0, 1).contiguous()
        observation[name] = tensor.unsqueeze(0)

    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""

    return observation


def build_inference_frame(
    observation: RobotObservation,
    device: torch.device,
    ds_features: dict[str, dict],
    task: str | None = None,
    robot_type: str | None = None,
) -> RobotObservation:
    """从原始观测构建模型可用的观测张量字典。

    该工具函数负责编排将来自环境的原始、非结构化观测
    转换为适合传递给策略模型的结构化、基于张量的格式的过程。

    Args:
        observation: 原始观测字典，可能包含多余的键。
        device: 最终张量的目标 PyTorch 设备。
        ds_features: 指定要从原始观测中提取哪些特征的配置字典。
        task: 当前任务的可选字符串标识符。
        robot_type: 所用机器人的可选字符串标识符。

    Returns:
        可直接用于模型推理的预处理张量字典。
    """
    # 从传入的原始观测中提取正确的键
    observation = build_dataset_frame(ds_features, observation, prefix=OBS_STR)

    # 对观测执行必要的转换
    observation = prepare_observation_for_inference(observation, device, task, robot_type)

    return observation


def make_robot_action(action_tensor: PolicyAction, ds_features: dict[str, dict]) -> RobotAction:
    """将策略的输出张量转换为带名称的动作字典。

    该函数将策略模型的数值输出转换为人类可读且机器人可消费的格式，
    其中动作张量的每个维度都被映射到一个带名称的电机或执行器命令。

    Args:
        action_tensor: 表示策略动作的 PyTorch 张量，
            通常带有批次维度（例如形状 [1, action_dim]）。
        ds_features: 包含元数据的配置字典，其中包括
            与动作张量每个索引对应的名称。

    Returns:
        将动作名称（例如 "joint_1_motor"）映射到其对应浮点值的字典，
        可直接发送给机器人控制器。
    """
    # TODO(Steven): 检查这些步骤是否已存在于所有后处理器策略中
    action_tensor = action_tensor.squeeze(0)
    action_tensor = action_tensor.to("cpu")

    action_names = ds_features[ACTION]["names"]
    act_processed_policy: RobotAction = {
        f"{name}": float(action_tensor[i]) for i, name in enumerate(action_names)
    }
    return act_processed_policy


def raise_feature_mismatch_error(
    provided_features: set[str],
    expected_features: set[str],
) -> None:
    """
    针对数据集/环境与策略配置之间的特征不匹配，抛出标准化的 ValueError。
    """
    missing = expected_features - provided_features
    extra = provided_features - expected_features
    # TODO (jadechoghari): 向用户提供动态的重命名映射建议。
    raise ValueError(
        f"Feature mismatch between dataset/environment and policy config.\n"
        f"- Missing features: {sorted(missing) if missing else 'None'}\n"
        f"- Extra features: {sorted(extra) if extra else 'None'}\n\n"
        f"Please ensure your dataset and policy use consistent feature names.\n"
        f"If your dataset uses different observation keys (e.g., cameras named differently), "
        f"use the `--rename_map` argument, for example:\n"
        f'  --rename_map=\'{{"observation.images.left": "observation.images.camera1", '
        f'"observation.images.top": "observation.images.camera2"}}\''
    )


def validate_visual_features_consistency(
    cfg: PreTrainedConfig,
    features: dict[str, PolicyFeature],
) -> None:
    """
    验证策略配置与提供的数据集/环境特征之间的视觉特征一致性。

    满足以下任一条件即验证通过：
    - 策略期望的视觉特征是数据集的子集（策略使用部分相机，数据集有更多）
    - 数据集提供的视觉特征是策略的子集（策略为灵活性声明了额外的特征）

    Args:
        cfg (PreTrainedConfig): 包含 input_features 和 type 的模型或策略配置。
        features (Dict[str, PolicyFeature]): 特征名称到 PolicyFeature 对象的映射。
    """
    expected_visuals = {k for k, v in cfg.input_features.items() if v.type == FeatureType.VISUAL}
    provided_visuals = {k for k, v in features.items() if v.type == FeatureType.VISUAL}

    # 任一方向是子集即接受
    policy_subset_of_dataset = expected_visuals.issubset(provided_visuals)
    dataset_subset_of_policy = provided_visuals.issubset(expected_visuals)

    if not (policy_subset_of_dataset or dataset_subset_of_policy):
        raise_feature_mismatch_error(provided_visuals, expected_visuals)
