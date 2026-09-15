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

"""RTC 调试信息的可视化工具。"""

import torch


class RTCDebugVisualizer:
    """RTC 调试信息可视化器。

    该类提供了一系列方法，用于可视化 Tracker 收集到的调试信息，
    包括各个去噪步骤上的校正量、误差、权重以及引导权重。
    """

    @staticmethod
    def plot_waypoints(
        axes,
        tensor,
        start_from: int = 0,
        color: str = "blue",
        label: str = "",
        alpha: float = 0.7,
        linewidth: float = 2,
        marker: str | None = None,
        markersize: int = 4,
    ):
        """绘制多个维度上的轨迹。

        该函数将张量在多个维度上的值随时间的变化绘制成图，
        每个维度绘制在单独的坐标轴上。

        Args:
            axes: matplotlib 坐标轴数组（每个维度对应一个）。
            tensor: 要绘制的张量（可以是 torch.Tensor 或 numpy 数组）。
                   形状应为 (time_steps, num_dims) 或 (batch, time_steps, num_dims)。
            start_from: x 轴的起始索引。
            color: 曲线的颜色。
            label: 图例标签。
            alpha: 图形的透明度。
            linewidth: 曲线的线宽。
            marker: 数据点的标记样式（如 'o'、's'、'^'）。
            markersize: 标记的大小。
        """
        import numpy as np

        # 处理张量为 None 的情况
        if tensor is None:
            return

        # 必要时将张量转换为 numpy
        tensor_np = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

        # 处理不同的张量形状
        if tensor_np.ndim == 3:
            # 若存在批次维度，则取第一个批次
            tensor_np = tensor_np[0]
        elif tensor_np.ndim == 1:
            # 若为一维，则重塑为 (time_steps, 1)
            tensor_np = tensor_np.reshape(-1, 1)

        # 获取维度
        time_steps, num_dims = tensor_np.shape

        # 创建 x 轴索引
        x_indices = np.arange(start_from, start_from + time_steps)

        # 在对应的坐标轴上绘制每个维度
        num_axes = len(axes) if hasattr(axes, "__len__") else 1
        for dim_idx in range(min(num_dims, num_axes)):
            ax = axes[dim_idx] if hasattr(axes, "__len__") else axes

            # 绘制轨迹
            if marker:
                ax.plot(
                    x_indices,
                    tensor_np[:, dim_idx],
                    color=color,
                    label=label if dim_idx == 0 else "",  # 标签只显示一次
                    alpha=alpha,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=markersize,
                )
            else:
                ax.plot(
                    x_indices,
                    tensor_np[:, dim_idx],
                    color=color,
                    label=label if dim_idx == 0 else "",  # 标签只显示一次
                    alpha=alpha,
                    linewidth=linewidth,
                )

            # 若网格和标签尚不存在，则添加
            if not ax.xaxis.get_label().get_text():
                ax.set_xlabel("Step", fontsize=10)
            if not ax.yaxis.get_label().get_text():
                ax.set_ylabel(f"Dim {dim_idx}", fontsize=10)
            ax.grid(True, alpha=0.3)
