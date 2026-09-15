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

"""带有可插拔 rollout 策略的策略部署引擎。

一个交互式 rollout 由四个组件构成：:class:`InferenceEngine`（持有策略）、
:class:`RolloutStrategy`（实时 tick 循环）、:class:`RolloutController`（生命周期状态机）
以及 :class:`InteractiveSession`（基于控制器的文本 I/O）。调用关系只指向下方——session ->
controller -> {strategy, engine}，strategy -> engine——并且 ``strategies/`` 和 ``inference/``
下的任何内容都不会引用控制器，因此策略在非交互模式下依然可用。控制器命令只在控制器锁下
记录意图；``serve()`` 是唯一将意图转化为动作的地方。
"""

from lerobot.utils.import_utils import require_package

require_package("datasets", extra="dataset")

from .configs import (
    BaseStrategyConfig,
    DAggerKeyboardConfig,
    DAggerPedalConfig,
    DAggerStrategyConfig,
    EpisodicStrategyConfig,
    HighlightStrategyConfig,
    RolloutConfig,
    RolloutStrategyConfig,
    SentryStrategyConfig,
)
from .context import (
    DatasetContext,
    HardwareContext,
    PolicyContext,
    ProcessorContext,
    RolloutContext,
    RuntimeContext,
    build_rollout_context,
)
from .controller import (
    AskResult,
    LinkedEvent,
    RolloutController,
    RolloutEvent,
)
from .inference import (
    InferenceEngine,
    InferenceEngineConfig,
    QueryAnswer,
    QueryKind,
    RTCInferenceConfig,
    RTCInferenceEngine,
    SyncInferenceConfig,
    SyncInferenceEngine,
    create_inference_engine,
)
from .interactive import InteractiveSession
from .strategies import (
    BaseStrategy,
    DAggerStrategy,
    EpisodicStrategy,
    HighlightStrategy,
    RolloutStrategy,
    SentryStrategy,
    create_strategy,
)

__all__ = [
    "AskResult",
    "BaseStrategy",
    "BaseStrategyConfig",
    "DAggerKeyboardConfig",
    "DAggerPedalConfig",
    "DAggerStrategy",
    "DAggerStrategyConfig",
    "DatasetContext",
    "EpisodicStrategy",
    "EpisodicStrategyConfig",
    "HardwareContext",
    "HighlightStrategy",
    "HighlightStrategyConfig",
    "InferenceEngine",
    "InferenceEngineConfig",
    "InteractiveSession",
    "LinkedEvent",
    "PolicyContext",
    "ProcessorContext",
    "QueryAnswer",
    "QueryKind",
    "RTCInferenceConfig",
    "RTCInferenceEngine",
    "RolloutConfig",
    "RolloutContext",
    "RolloutController",
    "RolloutEvent",
    "RolloutStrategy",
    "RolloutStrategyConfig",
    "RuntimeContext",
    "SentryStrategy",
    "SentryStrategyConfig",
    "SyncInferenceConfig",
    "SyncInferenceEngine",
    "build_rollout_context",
    "create_inference_engine",
    "create_strategy",
]
