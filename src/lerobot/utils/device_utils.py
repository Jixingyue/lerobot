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
from contextlib import nullcontext

import torch


def auto_select_torch_device() -> torch.device:
    """尝试自动选择一个 torch 设备。"""
    if torch.cuda.is_available():
        logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Metal backend detected, using mps.")
        return torch.device("mps")
    elif torch.xpu.is_available():
        logging.info("Intel XPU backend detected, using xpu.")
        return torch.device("xpu")
    else:
        logging.warning("No accelerated backend detected. Using default cpu, this will be slow.")
        return torch.device("cpu")


# TODO(Steven): 移除 log。log 不应作为参数，这应该由 logger 级别来处理
def get_safe_torch_device(try_device: str, log: bool = False) -> torch.device:
    """给定一个字符串，返回一个 torch.device，并检查该设备是否可用。

    抛出异常：
        ValueError：当请求的设备类别已知但在本机不可用时
            （此前使用 ``AssertionError``，但在 ``python -O`` 下断言会消失，
            容易被误认为是程序员的 bug）。
    """
    try_device = str(try_device)
    if try_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"Requested device {try_device!r} but CUDA is not available.")
        device = torch.device(try_device)
    elif try_device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested device 'mps' but MPS is not available.")
        device = torch.device("mps")
    elif try_device == "xpu":
        if not torch.xpu.is_available():
            raise ValueError("Requested device 'xpu' but XPU is not available.")
        device = torch.device("xpu")
    elif try_device == "cpu":
        device = torch.device("cpu")
        if log:
            logging.warning("Using CPU, this will be slow.")
    else:
        device = torch.device(try_device)
        if log:
            logging.warning(f"Using custom {try_device} device.")
    return device


def resolve_safetensors_device(map_location: str | torch.device) -> str:
    """为 safetensors 加载解析设备字符串，绕开一个设备映射的怪癖。

    safetensors 的加载会把裸字符串 "cuda" 映射到 cuda:0，而不管当前设备是什么
    （不像 torch 的 .to("cuda") 会遵循 torch.cuda.current_device()）。在多 GPU 的
    accelerate/FSDP 下，每个 rank 都会把权重加载到 GPU 0 上，在分片之前就把它撑爆。
    将 "cuda" 解析为具体的当前设备索引，使每个 rank 加载到自己的 GPU 上。
    """
    map_location = str(map_location)
    if map_location == "cuda" and torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return map_location


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """
    mps 目前与 float64 不兼容
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    if device == "xpu" and dtype == torch.float64:
        if hasattr(torch.xpu, "get_device_capability"):
            device_capability = torch.xpu.get_device_capability()
            # 注意：部分 Intel XPU 设备不支持双精度（FP64）。
            # `has_fp64` 标志由 `torch.xpu.get_device_capability()`
            # 在可用时返回；若为 False，为兼容性回退到 float32。
            if not device_capability.get("has_fp64", False):
                logging.warning(f"Device {device} does not support float64, using float32 instead.")
                return torch.float32
        else:
            logging.warning(
                f"Device {device} capability check failed. Assuming no support for float64, using float32 instead."
            )
            return torch.float32
        return dtype
    else:
        return dtype


def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)  # 确保 try_device 是字符串
    if try_device.startswith("cuda"):
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "xpu":
        return torch.xpu.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, mps, xpu or cpu.")


def is_amp_available(device: str):
    if device in ["cuda", "xpu", "cpu"]:
        return True
    elif device == "mps":
        return False
    else:
        raise ValueError(f"Unknown device '{device}.")


def get_autocast_context(device_type: str, dtype: torch.dtype = torch.bfloat16):
    """返回一个设备安全的 autocast 上下文管理器。

    硬编码的 `torch.autocast(dtype=torch.bfloat16)` 在没有 AMP 的后端（MPS）上会报错，
    在 Ampere 之前的 CUDA 上也会行为异常。因此改为：
      - 不支持 AMP（例如 mps）：`nullcontext()`，即使用张量的原生 dtype
      - CPU 且 dtype 是其 autocast 未实现的（尤其是 float32）：`nullcontext()`，因为
        `torch.autocast` 会接受它，然后在每次调用时发出警告并禁用
      - CUDA 在计算能力 < 8.0 时请求 bf16：回退到 fp16
      - 其他情况：`torch.autocast(device_type, dtype)`
    """
    if not is_amp_available(device_type):
        return nullcontext()
    if device_type == "cpu" and dtype not in (torch.bfloat16, torch.float16):
        return nullcontext()
    if (
        device_type == "cuda"
        and dtype == torch.bfloat16
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] < 8
    ):
        dtype = torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype)
