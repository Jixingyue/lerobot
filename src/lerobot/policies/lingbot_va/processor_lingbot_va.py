# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""LingBot-VA 策略的前/后处理器流水线。

预处理器直接透传输入（IDENTITY），后处理器使用内置的 ``UnnormalizerProcessorStep``
（QUANTILES），利用从检查点恢复的逐通道 q01/q99，将策略输出的 ``[-1, 1]``
动作映射回物理单位。
"""

from typing import Any

import torch

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    UnnormalizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_lingbot_va import LingBotVAConfig


def make_lingbot_va_pre_post_processors(
    config: LingBotVAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """构建 LingBot-VA 的前/后处理器流水线。"""

    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.normalize,
        steps.to_device,
    ]

    # 利用从检查点恢复的 q01/q99，将动作从 [-1, 1] 反归一化（QUANTILES）到物理单位。
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
            stats=dataset_stats,
        ),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
