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

"""实时分块（Real-Time Chunking，RTC）的调试信息处理器。"""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class DebugStep:
    """单个去噪步骤的调试信息容器。

    Attributes:
        step_idx (int): 步骤索引/计数器。
        x_t (Tensor | None): 当前的潜变量/状态张量。
        v_t (Tensor | None): 去噪器给出的速度。
        x1_t (Tensor | None): 去噪预测值（x_t - time * v_t）。
        correction (Tensor | None): 校正梯度张量。
        err (Tensor | None): 加权误差项。
        weights (Tensor | None): 前缀注意力权重。
        guidance_weight (float | Tensor | None): 实际应用的引导权重。
        time (float | Tensor | None): 时间参数。
        inference_delay (int | None): 推理延迟参数。
        execution_horizon (int | None): 执行时域参数。
        metadata (dict[str, Any]): 附加元数据。
    """

    step_idx: int = 0
    x_t: Tensor | None = None
    v_t: Tensor | None = None
    x1_t: Tensor | None = None
    correction: Tensor | None = None
    err: Tensor | None = None
    weights: Tensor | None = None
    guidance_weight: float | Tensor | None = None
    time: float | Tensor | None = None
    inference_delay: int | None = None
    execution_horizon: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_tensors: bool = False) -> dict[str, Any]:
        """将调试步骤转换为字典。

        Args:
            include_tensors (bool): 为 True 时包含张量的值；为 False 时仅包含
                张量的统计信息（shape、mean、std、min、max）。

        Returns:
            该调试步骤的字典表示。
        """
        result = {
            "step_idx": self.step_idx,
            "guidance_weight": (
                self.guidance_weight.item()
                if isinstance(self.guidance_weight, Tensor)
                else self.guidance_weight
            ),
            "time": self.time.item() if isinstance(self.time, Tensor) else self.time,
            "inference_delay": self.inference_delay,
            "execution_horizon": self.execution_horizon,
            "metadata": self.metadata.copy(),
        }

        # 添加张量信息
        tensor_fields = ["x_t", "v_t", "x1_t", "correction", "err", "weights"]
        for field_name in tensor_fields:
            tensor = getattr(self, field_name)
            if tensor is not None:
                if include_tensors:
                    result[field_name] = tensor.detach().cpu()
                else:
                    result[f"{field_name}_stats"] = {
                        "shape": tuple(tensor.shape),
                        "mean": tensor.mean().item(),
                        "std": tensor.std().item(),
                        "min": tensor.min().item(),
                        "max": tensor.max().item(),
                    }

        return result


class Tracker:
    """收集并管理 RTC 处理过程中的调试信息。

    该跟踪器将最近去噪步骤的调试信息存储在字典中，
    以时间作为键，便于高效查找和更新。

    Args:
        enabled (bool): 是否启用调试信息收集。
        maxlen (int | None): 可选的滑动窗口大小。若提供，则只保留最近的
            ``maxlen`` 个调试步骤；若为 ``None``，则保留全部。
    """

    def __init__(self, enabled: bool = False, maxlen: int = 100):
        self.enabled = enabled
        self._steps = {} if enabled else None  # 以时间为键的字典
        self._maxlen = maxlen
        self._step_counter = 0

    def reset(self) -> None:
        """清除所有已记录的调试信息。"""
        if self.enabled and self._steps is not None:
            self._steps.clear()
        self._step_counter = 0

    @torch._dynamo.disable
    def track(
        self,
        time: float | Tensor,
        x_t: Tensor | None = None,
        v_t: Tensor | None = None,
        x1_t: Tensor | None = None,
        correction: Tensor | None = None,
        err: Tensor | None = None,
        weights: Tensor | None = None,
        guidance_weight: float | Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        **metadata,
    ) -> None:
        """跟踪给定时间下某个去噪步骤的调试信息。

        如果已存在该时间对应的步骤，则用新数据更新；否则创建一个新步骤。
        只会更新/设置非 None 的字段。

        注意：该方法被排除在 torch.compile 之外，以避免 .item() 等与编译图
        不兼容的操作导致图中断。

        Args:
            time (float | Tensor): 时间参数，用作标识该步骤的键。
            x_t (Tensor | None): 当前的潜变量/状态张量。
            v_t (Tensor | None): 去噪器给出的速度。
            x1_t (Tensor | None): 去噪预测值。
            correction (Tensor | None): 校正梯度张量。
            err (Tensor | None): 加权误差项。
            weights (Tensor | None): 前缀注意力权重。
            guidance_weight (float | Tensor | None): 实际应用的引导权重。
            inference_delay (int | None): 推理延迟参数。
            execution_horizon (int | None): 执行时域参数。
            **metadata: 要存储的附加元数据。
        """
        if not self.enabled:
            return

        # 将时间转换为 float 并四舍五入，以避免浮点数精度问题
        time_value = time.item() if isinstance(time, Tensor) else time
        time_key = round(time_value, 6)  # 使用四舍五入后的时间作为字典键

        # 检查是否已存在该时间对应的步骤
        if time_key in self._steps:
            # 用非 None 字段更新已有步骤
            existing_step = self._steps[time_key]
            if x_t is not None:
                existing_step.x_t = x_t.detach().clone()
            if v_t is not None:
                existing_step.v_t = v_t.detach().clone()
            if x1_t is not None:
                existing_step.x1_t = x1_t.detach().clone()
            if correction is not None:
                existing_step.correction = correction.detach().clone()
            if err is not None:
                existing_step.err = err.detach().clone()
            if weights is not None:
                existing_step.weights = weights.detach().clone()
            if guidance_weight is not None:
                existing_step.guidance_weight = guidance_weight
            if inference_delay is not None:
                existing_step.inference_delay = inference_delay
            if execution_horizon is not None:
                existing_step.execution_horizon = execution_horizon
            if metadata:
                existing_step.metadata.update(metadata)
        else:
            # 创建新步骤
            step = DebugStep(
                step_idx=self._step_counter,
                x_t=x_t.detach().clone() if x_t is not None else None,
                v_t=v_t.detach().clone() if v_t is not None else None,
                x1_t=x1_t.detach().clone() if x1_t is not None else None,
                correction=correction.detach().clone() if correction is not None else None,
                err=err.detach().clone() if err is not None else None,
                weights=weights.detach().clone() if weights is not None else None,
                guidance_weight=guidance_weight,
                time=time_value,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                metadata=metadata,
            )

            # 添加到字典
            self._steps[time_key] = step
            self._step_counter += 1

            # 若设置了 maxlen，则执行长度限制
            if self._maxlen is not None and len(self._steps) > self._maxlen:
                # 删除最旧的条目（字典中的第一个键——Python 3.7+ 保持插入顺序）
                oldest_key = next(iter(self._steps))
                del self._steps[oldest_key]

    def get_all_steps(self) -> list[DebugStep]:
        """获取所有已记录的调试步骤。

        Returns:
            所有 DebugStep 对象组成的列表（若禁用则可能为空）。
        """
        if not self.enabled or self._steps is None:
            return []

        return list(self._steps.values())

    def __len__(self) -> int:
        """返回已记录的调试步骤数量。"""
        if not self.enabled or self._steps is None:
            return 0
        return len(self._steps)
