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

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.configs.rewards import RewardModelConfig
from lerobot.utils.constants import OBS_IMAGES

# 来自上游 TOPReward 论文/参考实现（``QwenClient.compute_instruction_reward``）的
# 默认提示词框架。该提示词在给定视频的条件下，
# 对 ``f"{instruction} ... True"`` 中末尾的 ``True`` token
# 进行评分。
DEFAULT_PROMPT_PREFIX = (
    "The above video shows a robot manipulation trajectory that completes the following task: "
)
DEFAULT_PROMPT_SUFFIX_TEMPLATE = (
    "{instruction} Decide whether the above statement is True or not. The answer is: True"
)


@RewardModelConfig.register_subclass("topreward")
@dataclass
class TOPRewardConfig(RewardModelConfig):
    """TOPReward 零样本奖励模型的配置。

    TOPReward 是**零样本**的：它自身没有可学习参数。
    该"模型"是一个通用的视觉-语言模型（默认为
    ``Qwen/Qwen3-VL-8B-Instruct``），配合固定提示词使用，
    以提取 token 的对数概率作为奖励信号。因此没有
    需要托管的微调检查点：``pretrained_path`` 在运行时
    不会被使用——模型身份由 :attr:`vlm_name`（HF Hub id）确定。

    Args:
        vlm_name: 底层 VLM 的 Hugging Face Hub id。必须是
            Qwen3-VL 系列模型（这是本 LeRobot 移植版中
            唯一实现的客户端）。
        torch_dtype: 传递给 VLM 加载器的 Torch dtype 名称
            （``"auto"``、``"bfloat16"``、``"float16"`` 等）。
        attn_implementation: ``transformers`` 的注意力实现
            （例如 ``"flash_attention_2"``、``"sdpa"``）。默认为
            ``None``，让上游选择最佳可用实现。
        image_key: 保存轨迹帧的观测键。
        task_key: 保存任务指令的补充数据键。
        default_task: 当 ``task_key`` 缺失时使用的回退指令。
        max_frames: 每个样本送入 VLM 的帧数上限。
            ``None`` = 使用所有帧。
        fps: 供 Qwen 视频处理器使用的每秒帧数元数据。
        prompt_prefix: 紧跟在视频之后、后缀模板之前
            展示给 VLM 的文本。
        prompt_suffix_template: 追加在 ``prompt_prefix`` 之后的后缀。
            必须包含 ``{instruction}``；VLM 会对
            前缀之后各 token 的对数似然进行评分。
        add_chat_template: 若为 ``True``，在分词前用分词器的
            chat template 包装完整提示词（与上游
            ``add_chat_template=True`` 一致）。
        success_threshold: 可选的 log-prob 阈值。若为有限值，
            :meth:`TOPRewardModel.compute_reward` 将返回
            ``(reward > success_threshold).float()`` 而不是
            原始 log-prob。
        max_input_length: 分词后总输入长度的硬性上限；
            超出该限制的样本会抛出 ``ValueError``。
    """

    # 指向本地 LeRobot 目录或 HF 仓库的路径，其中保存了本
    # TOPRewardConfig 的 ``config.json`` 快照。VLM 权重本身
    # 始终由 ``vlm_name`` 标识。
    pretrained_path: str | None = None

    vlm_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    torch_dtype: str = "auto"
    attn_implementation: str | None = None

    image_key: str = OBS_IMAGES + ".top"
    task_key: str = "task"
    default_task: str | None = None
    max_frames: int | None = 16
    fps: float = 2.0

    prompt_prefix: str = DEFAULT_PROMPT_PREFIX
    prompt_suffix_template: str = DEFAULT_PROMPT_SUFFIX_TEMPLATE
    add_chat_template: bool = False

    success_threshold: float = float("-inf")
    max_input_length: int = 32768

    license: str | None = "mit"  # 与上游 TOPReward 保持一致
    tags: list[str] | None = field(
        default_factory=lambda: ["reward-model", "vision-language", "qwen3-vl", "zero-shot"]
    )

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {self.max_frames}")
        if self.fps <= 0:
            raise ValueError(f"fps must be > 0, got {self.fps}")
        if "{instruction}" not in self.prompt_suffix_template:
            raise ValueError(
                "prompt_suffix_template must contain `{instruction}` so the model "
                "scores the log-likelihood of the task suffix."
            )
        if self.max_input_length <= 0:
            raise ValueError(f"max_input_length must be > 0, got {self.max_input_length}")

        if self.image_key not in self.input_features:
            self.input_features[self.image_key] = PolicyFeature(shape=(3, 224, 224), type=FeatureType.VISUAL)
        self.output_features.setdefault("reward", PolicyFeature(shape=(1,), type=FeatureType.REWARD))

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if self.image_key not in self.input_features:
            raise ValueError(f"TOPReward requires image input feature {self.image_key!r}")
