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

"""
本脚本定义了一个处理步骤，用于将环境转移数据移动到指定的 torch 设备，
并转换其浮点精度。
"""

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, PolicyAction, TransitionKey
from lerobot.utils.device_utils import get_safe_torch_device

from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("device_processor")
@dataclass
class DeviceProcessorStep(ProcessorStep):
    """
    将 `EnvTransition` 中的所有张量移动到指定设备，并可选地转换其
    浮点数据类型的处理步骤。

    这对于为 GPU 等硬件上的模型训练或推理准备数据至关重要。

    Attributes:
        device: 张量的目标设备（例如 "cpu"、"cuda"、"cuda:0"）。
        float_dtype: 以字符串表示的目标浮点 dtype（例如 "float32"、"float16"、"bfloat16"）。
                     如果为 None，则不更改 dtype。
    """

    device: str = "cpu"
    float_dtype: str | None = None

    DTYPE_MAPPING = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "half": torch.float16,
        "float": torch.float32,
        "double": torch.float64,
    }

    def __post_init__(self):
        """
        通过将字符串配置转换为 torch 对象来初始化处理器。

        该方法设置 `torch.device`，判断传输是否可以为非阻塞，并验证
        `float_dtype` 字符串，将其转换为 `torch.dtype` 对象。
        """
        self.tensor_device: torch.device = get_safe_torch_device(self.device)
        # 更新设备字符串，以防选择了特定的 GPU（例如 "cuda" -> "cuda:0"）
        self.device = self.tensor_device.type
        self.non_blocking = "cuda" in str(self.device)

        # 验证 float_dtype 字符串并将其转换为 torch dtype
        if self.float_dtype is not None:
            if self.float_dtype not in self.DTYPE_MAPPING:
                raise ValueError(
                    f"Invalid float_dtype '{self.float_dtype}'. Available options: {list(self.DTYPE_MAPPING.keys())}"
                )
            self._target_float_dtype = self.DTYPE_MAPPING[self.float_dtype]
        else:
            self._target_float_dtype = None

    def _process_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        将单个张量移动到目标设备并转换其 dtype。

        如果张量已经位于与目标不同的 CUDA 设备上，则不移动该张量，以此处理多 GPU 场景，
        这在使用 Accelerate 等框架时很有用。

        Args:
            tensor: 输入的 torch.Tensor。

        Returns:
            位于正确设备且具有正确 dtype 的处理后张量。
        """
        # 确定目标设备
        if tensor.is_cuda and self.tensor_device.type == "cuda":
            # 张量和目标都在 GPU 上 - 保留张量原有的 GPU 位置。
            # 这处理了多 GPU 场景，其中 Accelerate 已经为每个进程
            # 将张量放置在正确的 GPU 上。
            target_device = tensor.device
        else:
            # 要么张量在 CPU 上，要么我们配置为使用 CPU。
            # 两种情况下都使用配置的设备。
            target_device = self.tensor_device

        # MPS 变通方案：由于 MPS 不支持 float64，将 float64 转换为 float32
        if target_device.type == "mps" and tensor.dtype == torch.float64:
            tensor = tensor.to(dtype=torch.float32)

        # 仅在必要时才移动
        if tensor.device != target_device:
            tensor = tensor.to(target_device, non_blocking=self.non_blocking)

        # 如果指定了目标浮点 dtype 且张量是浮点类型，则进行转换
        if self._target_float_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=self._target_float_dtype)

        return tensor

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        对环境转移中的所有张量应用设备和 dtype 转换。

        它会遍历转移，找到所有 `torch.Tensor` 对象（包括嵌套在
        `observation` 等字典中的张量），并对其进行处理。

        Args:
            transition: 输入的 `EnvTransition` 对象。

        Returns:
            所有张量都已移动到目标设备和 dtype 的新 `EnvTransition` 对象。
        """
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is not None and not isinstance(action, PolicyAction):
            raise ValueError(f"If action is not None should be a PolicyAction type got {type(action)}")

        simple_tensor_keys = [
            TransitionKey.ACTION,
            TransitionKey.REWARD,
            TransitionKey.DONE,
            TransitionKey.TRUNCATED,
        ]

        dict_tensor_keys = [
            TransitionKey.OBSERVATION,
            TransitionKey.COMPLEMENTARY_DATA,
        ]

        # 处理简单的顶层张量
        for key in simple_tensor_keys:
            value = transition.get(key)
            if isinstance(value, torch.Tensor):
                new_transition[key] = self._process_tensor(value)

        # 处理嵌套在字典中的张量
        for key in dict_tensor_keys:
            data_dict = transition.get(key)
            if data_dict is not None:
                new_data_dict = {
                    k: self._process_tensor(v) if isinstance(v, torch.Tensor) else v
                    for k, v in data_dict.items()
                }
                new_transition[key] = new_data_dict

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        返回处理器的可序列化配置。

        Returns:
            包含 device 和 float_dtype 设置的字典。
        """
        return {"device": self.device, "float_dtype": self.float_dtype}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        原样返回输入特征。

        设备和 dtype 转换不会改变特征的基本定义（例如形状）。

        Args:
            features: 策略特征字典。

        Returns:
            原始的策略特征字典。
        """
        return features
