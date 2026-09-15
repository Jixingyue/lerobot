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
lerobot 配置类型和基础配置类的公共 API。

注意：TrainPipelineConfig、EvalPipelineConfig 和 TrainRLServerPipelineConfig
有意不在这里重新导出，以避免循环依赖
（它们在模块级别导入 lerobot.envs 和 lerobot.policies）。
请直接导入：``from lerobot.configs.train import TrainPipelineConfig``
"""

from .dataset import DatasetRecordConfig
from .default import DatasetConfig, EMAConfig, EvalConfig, JobConfig, PeftConfig, WandBConfig
from .policies import PreTrainedConfig
from .types import (
    FeatureType,
    NormalizationMode,
    PipelineFeatureType,
    PolicyFeature,
    RTCAttentionSchedule,
)
from .video import (
    DEFAULT_DEPTH_UNIT,
    DEPTH_METER_UNIT,
    DEPTH_MILLIMETER_UNIT,
    VALID_VIDEO_CODECS,
    VIDEO_ENCODER_INFO_KEYS,
    DepthEncoderConfig,
    RGBEncoderConfig,
    VideoEncoderConfig,
    depth_encoder_defaults,
    encoder_config_from_video_info,
    infer_depth_unit,
    is_depth_map,
    rgb_encoder_defaults,
)

__all__ = [
    # 类型
    "FeatureType",
    "NormalizationMode",
    "PipelineFeatureType",
    "PolicyFeature",
    "RTCAttentionSchedule",
    # 配置类
    "DatasetRecordConfig",
    "DatasetConfig",
    "EMAConfig",
    "EvalConfig",
    "JobConfig",
    "PeftConfig",
    "PreTrainedConfig",
    "WandBConfig",
    "VideoEncoderConfig",
    "RGBEncoderConfig",
    "DepthEncoderConfig",
    # 默认值
    "rgb_encoder_defaults",
    "depth_encoder_defaults",
    # 工厂函数
    "encoder_config_from_video_info",
    "infer_depth_unit",
    "is_depth_map",
    # 常量
    "DEFAULT_DEPTH_UNIT",
    "DEPTH_METER_UNIT",
    "DEPTH_MILLIMETER_UNIT",
    "VALID_VIDEO_CODECS",
    "VIDEO_ENCODER_INFO_KEYS",
]
