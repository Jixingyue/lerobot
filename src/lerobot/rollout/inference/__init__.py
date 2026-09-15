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

"""推理引擎包——与后端无关的动作生成。

具体后端（``sync``、``rtc`` 等）暴露相同的小型接口，
因此 rollout 策略永远不需要根据所使用的后端进行分支处理。
"""

from .base import InferenceEngine, PolicyQuery, QueryAnswer, QueryKind
from .factory import (
    InferenceEngineConfig,
    RTCInferenceConfig,
    SyncInferenceConfig,
    create_inference_engine,
)
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine

__all__ = [
    "InferenceEngine",
    "InferenceEngineConfig",
    "PolicyQuery",
    "QueryAnswer",
    "QueryKind",
    "RTCInferenceConfig",
    "RTCInferenceEngine",
    "SyncInferenceConfig",
    "SyncInferenceEngine",
    "create_inference_engine",
]
