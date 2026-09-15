#!/usr/bin/env python

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
"""可引导的标注流水线，为 LeRobot 数据集生成 ``language_persistent`` 和
``language_events`` 列。

该流水线被分解为三个可独立运行的模块，它们的输出在最终 parquet 重写之前按回合（episode）暂存：

- :mod:`.modules.plan_subtasks_memory`（``plan`` 模块）— 持久化样式
- :mod:`.modules.interjections_and_speech`（``interjections`` 模块）— 事件样式 + 语音
- :mod:`.modules.general_vqa`（``vqa`` 模块）— 事件样式的 VQA 对
"""

from .config import AnnotationPipelineConfig
from .validator import StagingValidator, ValidationReport
from .writer import LanguageColumnsWriter

__all__ = [
    "AnnotationPipelineConfig",
    "LanguageColumnsWriter",
    "StagingValidator",
    "ValidationReport",
]
