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

"""TOPReward 预处理/后处理流水线。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    policy_action_to_transition,
)
from lerobot.rewards.topreward.configuration_topreward import (
    DEFAULT_PROMPT_PREFIX,
    DEFAULT_PROMPT_SUFFIX_TEMPLATE,
    TOPRewardConfig,
)
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_PREFIX,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None

TOPREWARD_FEATURE_PREFIX = f"{OBS_PREFIX}topreward."

_TRUE_ANSWER = "True"

TOPREWARD_VLM_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values_videos",
    "video_grid_thw",
    "mm_token_type_ids",
)
TOPREWARD_INPUT_KEYS = TOPREWARD_VLM_INPUT_KEYS + ("labels",)


def _prepare_video_batch(video: Tensor, *, max_frames: int | None) -> Tensor:
    """将视频以 ``(B, T, C, H, W)`` uint8 张量的形式返回，供 Qwen3-VL 使用。"""
    if video.ndim == 4:
        video = video.unsqueeze(1)
    elif video.ndim != 5:
        raise ValueError(
            f"Expected TOPReward frames with shape (B,C,H,W) or (B,T,C,H,W); got {tuple(video.shape)}"
        )

    if max_frames is not None:
        video = video[:, -max_frames:]
    if video.shape[-1] in (1, 3):
        video = video.permute(0, 1, 4, 2, 3)
    elif video.shape[2] not in (1, 3):
        raise ValueError(f"Expected channel dim of size 1 or 3, got shape {tuple(video.shape)}")

    if video.is_floating_point():
        video = video * 255.0

    return video.clamp(0, 255).to(torch.uint8).contiguous()


def _expand_tasks(task: Any, *, batch_size: int, default: str | None) -> list[str]:
    if task is None:
        task = default
    if task is None:
        raise KeyError("TOPReward expected a task description in complementary data")
    if isinstance(task, str):
        return [task] * batch_size
    if isinstance(task, tuple):
        task = list(task)
    if not (isinstance(task, list) and all(isinstance(item, str) for item in task)):
        raise TypeError(f"TOPReward task must be a string or list of strings, got {type(task)}")
    if len(task) == 1 and batch_size > 1:
        return task * batch_size
    if len(task) != batch_size:
        raise ValueError(f"Expected {batch_size} tasks, got {len(task)}")
    return task


@dataclass
@ProcessorStepRegistry.register(name="topreward_encoder")
class TOPRewardEncoderProcessorStep(ProcessorStep):
    """将原始帧 + 任务编码为供 TOPReward 模型使用的 Qwen-VL 张量。

    加载与 ``vlm_name`` 匹配的 :class:`~transformers.AutoProcessor`，
    并构建包含指令后缀的完整聊天提示词。所得的
    ``input_ids``、``attention_mask``、视觉张量和
    ``labels`` 会写入 ``observation.topreward.*`` 命名空间，
    使模型无需重新分词即可评分。

    调用时该步骤读取：

    - ``observation[image_key]``：``(B, T, C, H, W)`` 或 ``(B, C, H, W)`` 帧。
    - ``complementary_data[task_key]``：字符串或字符串列表。

    并将 Qwen-VL 张量及 ``labels`` 写入
    ``observation[f"{TOPREWARD_FEATURE_PREFIX}<name>"]``。
    """

    vlm_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    image_key: str = OBS_IMAGES + ".top"
    task_key: str = "task"
    default_task: str | None = None
    max_frames: int | None = 16
    fps: float = 2.0
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX
    prompt_suffix_template: str = DEFAULT_PROMPT_SUFFIX_TEMPLATE
    add_chat_template: bool = False
    max_length: int = 32768

    _processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        require_package("transformers", extra="topreward")
        self._processor = AutoProcessor.from_pretrained(self.vlm_name, trust_remote_code=True)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        if self.image_key not in observation:
            raise KeyError(f"TOPReward expected image key {self.image_key!r} in observation")

        frames = observation[self.image_key]
        videos = frames.detach().cpu() if isinstance(frames, Tensor) else torch.as_tensor(frames)
        videos = _prepare_video_batch(videos, max_frames=self.max_frames)

        batch_size = videos.shape[0]
        tasks = _expand_tasks(
            complementary.get(self.task_key, self.default_task),
            batch_size=batch_size,
            default=self.default_task,
        )

        encoded = self._encode_batch(videos, tasks, batch_size)

        new_observation = dict(observation)
        for key, value in encoded.items():
            new_observation[f"{TOPREWARD_FEATURE_PREFIX}{key}"] = value

        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def _encode_batch(self, videos: Tensor, tasks: list[str], batch_size) -> dict[str, Any]:
        """将一批 (帧, 任务) 对分词为 Qwen-VL 张量。

        循环仅构建每个样本的聊天字符串。分词、填充、
        视频预处理和 label 构建都是批量进行的。
        """

        texts: list[str] = []
        video_metadata = [
            {
                "total_num_frames": int(videos.shape[1]),
                "fps": float(self.fps),
                "frames_indices": list(range(int(videos.shape[1]))),
            }
            for _ in range(batch_size)
        ]
        eos_token = self._processor.tokenizer.eos_token

        for i in range(batch_size):
            instruction_suffix = self.prompt_suffix_template.format(instruction=tasks[i])
            if self.add_chat_template:
                suffix_for_template = instruction_suffix.removesuffix(_TRUE_ANSWER).rstrip()
                templated_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": videos[i], "fps": self.fps},
                            {"type": "text", "text": f"{self.prompt_prefix}{suffix_for_template}"},
                        ],
                    }
                ]
                prompt_chat = self._processor.apply_chat_template(
                    templated_messages, tokenize=False, add_generation_prompt=True
                )
                full_text = f"{prompt_chat}{_TRUE_ANSWER}"
            else:
                user_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": videos[i], "fps": self.fps},
                            {"type": "text", "text": self.prompt_prefix},
                        ],
                    }
                ]
                prompt_chat = self._processor.apply_chat_template(
                    user_messages, tokenize=False, add_generation_prompt=False
                )
                if eos_token is not None:
                    prompt_chat = prompt_chat.split(eos_token)[0]
                full_text = f"{prompt_chat}{instruction_suffix}"

            texts.append(full_text)

        result = self._processor(
            text=texts,
            videos=videos,
            video_metadata=video_metadata,
            do_sample_frames=False,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        input_ids = result["input_ids"]

        if input_ids.shape[-1] > self.max_length:
            raise ValueError(
                f"TOPReward input length {input_ids.shape[-1]} exceeds max_length "
                f"{self.max_length}; lower `max_frames` or raise `max_length`."
            )

        labels = torch.full_like(input_ids, -100)
        labels[:, -1] = input_ids[:, -1]
        result["labels"] = labels
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "vlm_name": self.vlm_name,
            "image_key": self.image_key,
            "task_key": self.task_key,
            "default_task": self.default_task,
            "max_frames": self.max_frames,
            "fps": self.fps,
            "prompt_prefix": self.prompt_prefix,
            "prompt_suffix_template": self.prompt_suffix_template,
            "add_chat_template": self.add_chat_template,
            "max_length": self.max_length,
        }


def make_topreward_pre_post_processors(
    config: TOPRewardConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """将帧 + 任务预先编码为 Qwen-VL 张量的流水线。

    预处理器在需要时添加批次维度，运行 TOPReward 的
    编码器（对完整提示词分词并输出 ``labels``），并
    将所有内容移动到配置的设备上。后处理器是
    恒等变换，因为 TOPReward 输出单个奖励张量。
    """
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            AddBatchDimensionProcessorStep(),
            TOPRewardEncoderProcessorStep(
                vlm_name=config.vlm_name,
                image_key=config.image_key,
                task_key=config.task_key,
                default_task=config.default_task,
                max_frames=config.max_frames,
                fps=config.fps,
                prompt_prefix=config.prompt_prefix,
                prompt_suffix_template=config.prompt_suffix_template,
                add_chat_template=config.add_chat_template,
                max_length=config.max_input_length,
            ),
            DeviceProcessorStep(device=config.device or "cpu"),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline(
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
    )
    return preprocessor, postprocessor
