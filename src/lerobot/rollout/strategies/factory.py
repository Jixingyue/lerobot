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

"""策略工厂：根据配置的类型名分发到对应的策略类。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import make_device_from_device_class

from .base import BaseStrategy
from .core import RolloutStrategy
from .dagger import DAggerStrategy
from .episodic import EpisodicStrategy
from .highlight import HighlightStrategy
from .sentry import SentryStrategy

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig


def create_strategy(config: RolloutStrategyConfig) -> RolloutStrategy:
    """根据配置对象实例化相应的策略。

    对于内置策略，依据 ``config.type``（通过 ``draccus.ChoiceRegistry``
    注册的名称）进行分发；随后回退到与机器人、相机和遥操作器工厂共用的
    ``<Name>Config`` -> ``<Name>`` 命名约定，因此第三方策略无需修改
    此处代码。
    """
    if config.type == "base":
        return BaseStrategy(config)
    if config.type == "sentry":
        return SentryStrategy(config)
    if config.type == "highlight":
        return HighlightStrategy(config)
    if config.type == "dagger":
        return DAggerStrategy(config)
    if config.type == "episodic":
        return EpisodicStrategy(config)
    try:
        return make_device_from_device_class(config)
    except Exception as e:
        raise ValueError(f"Error creating strategy with config {config}: {e}") from e
