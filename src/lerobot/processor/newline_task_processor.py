#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs import PipelineFeatureType, PolicyFeature

from .pipeline import ComplementaryDataProcessorStep, ProcessorStepRegistry


# 注意：注册表名称 "smolvla_new_line_processor" 予以保留，以向后兼容
# 引用该名称的已序列化处理器配置。
@ProcessorStepRegistry.register(name="smolvla_new_line_processor")
class NewLineTaskProcessorStep(ComplementaryDataProcessorStep):
    """
    确保 'task' 描述以换行符结尾的处理器步骤。

    某些分词器（例如 PaliGemma）要求提示末尾有一个
    换行符，因此需要本步骤。它同时处理单个字符串任务和
    字符串任务列表。
    """

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # 同时处理字符串和字符串列表
        if isinstance(task, str):
            # 单个字符串：若无换行则添加
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # 字符串列表：为每个缺少换行的字符串添加换行
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # 如果 task 既不是字符串也不是字符串列表，则保持不变

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
