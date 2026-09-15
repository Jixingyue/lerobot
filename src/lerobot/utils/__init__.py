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

"""
仅依赖基础包的轻量级工具模块的公共 API。

重量级的横切模块（train_utils、control_utils）已移动到
``lerobot.common``。``visualization_utils`` 仍保留在此处，
但有意不做重新导出，以避免引入可选依赖。
"""

from .constants import (
    ACTION,
    DEFAULT_FEATURES,
    DONE,
    IMAGENET_STATS,
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
    OBS_STR,
    REWARD,
)
from .decorators import check_if_already_connected, check_if_not_connected
from .device_utils import auto_select_torch_device, get_safe_torch_device, is_torch_device_available
from .errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from .import_utils import is_package_available, require_package

__all__ = [
    # 常量
    "ACTION",
    "DEFAULT_FEATURES",
    "DONE",
    "IMAGENET_STATS",
    "OBS_ENV_STATE",
    "OBS_IMAGE",
    "OBS_IMAGES",
    "OBS_STATE",
    "OBS_STR",
    "REWARD",
    # 设备工具
    "auto_select_torch_device",
    "get_safe_torch_device",
    "is_torch_device_available",
    # 导入守卫
    "is_package_available",
    "require_package",
    # 装饰器
    "check_if_already_connected",
    "check_if_not_connected",
    # 错误
    "DeviceAlreadyConnectedError",
    "DeviceNotConnectedError",
]
