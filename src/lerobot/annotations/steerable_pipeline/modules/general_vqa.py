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
"""``vqa`` 模块：按定时节奏进行通用 VQA。

每 ``1/hz`` 秒触发一次发射节拍；每个节拍锚定 ``K`` 个
连续帧，每个锚定帧获得自己的 VQA 对。每个
对都基于该单个锚定帧——没有逐对的帧
窗口。对于多摄像头数据集，每个锚定帧为*每个摄像头*产生
一个 ``(vqa, user)`` + ``(vqa, assistant)`` 对：每个对
针对该摄像头的帧生成，并在发射的行上标记匹配的
``camera`` 字段。解析器通过 ``camera=...`` 消歧；
消费 VQA 的配方通过每个摄像头一个子配方来实现
（参见 ``recipes/pi05_hirobot.yaml``）。

在单个 (frame, camera) 内，我们仍然最多发射一个 ``(vqa, user)``
和一个 ``(vqa, assistant)`` 行，因此解析器约定保持标量。

涵盖的问题类型（按照计划的 ``vqa`` 表）：bbox、keypoint、
count、attribute、spatial。助手的 ``content`` 是一个 JSON 字符串，
其模式取决于问题类型。格式错误的 JSON 会在
:meth:`VlmClient.generate_json` 内部触发一次重试。
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import VqaConfig
from ..frames import FrameProvider, null_provider, to_image_blocks
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord
from ..staging import EpisodeStaging
from ..validator import classify_vqa_answer
from ..vlm_client import VlmClient


def _emission_anchor_indices(frame_timestamps: Sequence[float], hz: float, k: int) -> list[int]:
    """返回 VQA 发射要锚定到的相对帧索引。

    对于每个发射节拍（每 ``1/hz`` 秒），我们从节拍开始锚定 ``k`` 个
    连续帧。节拍落在最近的可用源帧时间戳上。
    """
    if hz <= 0 or k <= 0 or not frame_timestamps:
        return []
    t0 = frame_timestamps[0]
    t_last = frame_timestamps[-1]
    period = 1.0 / hz
    indices: list[int] = []
    t = t0
    while t <= t_last + 1e-9:
        # 找到距离 t 最近的帧的索引
        nearest_i = min(range(len(frame_timestamps)), key=lambda i: abs(frame_timestamps[i] - t))
        for offset in range(k):
            j = nearest_i + offset
            if j >= len(frame_timestamps):
                break
            if not indices or indices[-1] != j:
                indices.append(j)
        t += period
    # 去重同时保持顺序
    seen: set[int] = set()
    deduped: list[int] = []
    for i in indices:
        if i in seen:
            continue
        seen.add(i)
        deduped.append(i)
    return deduped


@dataclass
class GeneralVqaModule:
    """按定时节奏发射有依据的 VQA 对。"""

    vlm: VlmClient
    config: VqaConfig
    seed: int = 1729
    frame_provider: FrameProvider = field(default_factory=null_provider)
    _warned_no_camera: bool = field(default=False, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        if not record.frame_timestamps:
            staging.write("vqa", [])
            return
        rng = random.Random(f"{self.seed}:{record.episode_index}:vqa")
        anchor_idx = _emission_anchor_indices(
            record.frame_timestamps, self.config.vqa_emission_hz, self.config.K
        )
        cameras = self._target_cameras()
        if not cameras:
            # 没有可用的摄像头——与其产生无法通过验证的无标记行，
            # 不如什么都不发射。发出一次响亮的一次性警告，
            # 使这永远不会静默地成为空操作。
            if not self._warned_no_camera:
                logging.getLogger(__name__).warning(
                    "vqa module found no cameras on the frame provider — "
                    "every episode will emit zero VQA rows. Check that the "
                    "dataset declares observation.images.* features in "
                    "meta/info.json; passing --vlm.camera_key=<key> at the "
                    "CLI now also seeds the cameras list as a fallback."
                )
                self._warned_no_camera = True
            staging.write("vqa", [])
            return

        # 先构建所有消息（每个 (frame, camera) 一条），然后将它们
        # 作为单个批量 generate_json 调用发出，以便客户端可以
        # 并发地分发它们。
        per_call: list[tuple[float, str, str, list[dict[str, Any]]]] = []
        for idx in anchor_idx:
            ts = float(record.frame_timestamps[idx])
            qtype = rng.choice(self.config.question_types)
            for camera in cameras:
                messages = self._build_messages(record, qtype, ts, camera)
                # 跳过在此时间戳解码出零帧的摄像头：没有图像时
                # 让 VLM 定位 bbox 毫无意义。
                if not _has_image_block(messages):
                    continue
                per_call.append((ts, camera, qtype, messages))

        if not per_call:
            staging.write("vqa", [])
            return

        results = self.vlm.generate_json([m for _, _, _, m in per_call])

        rows: list[dict[str, Any]] = []
        for (ts, camera, _qtype, _messages), result in zip(per_call, results, strict=True):
            qa = self._postprocess(result)
            if qa is None:
                continue
            question, answer = qa
            rows.append(
                {
                    "role": "user",
                    "content": question,
                    "style": "vqa",
                    "timestamp": ts,
                    "camera": camera,
                    "tool_calls": None,
                }
            )
            rows.append(
                {
                    "role": "assistant",
                    "content": json.dumps(answer, sort_keys=True),
                    "style": "vqa",
                    "timestamp": ts,
                    "camera": camera,
                    "tool_calls": None,
                }
            )
        staging.write("vqa", rows)

    def _target_cameras(self) -> list[str]:
        """返回 ``vqa`` 模块应为每个锚定帧迭代的摄像头。

        默认为提供者暴露的所有摄像头。没有摄像头的数据集
        （或测试/空提供者）产生空列表，这使
        ``run_episode`` 成为空操作。

        当设置了 ``config.restrict_to_default_camera`` 时，VQA 仅基于
        提供者的默认摄像头（单个 ``--vlm.camera_key``
        流）定位，与计划/插话模块匹配，使整个
        流水线聚焦于一个视图。
        """
        all_cameras = list(getattr(self.frame_provider, "camera_keys", []) or [])
        if getattr(self.config, "restrict_to_default_camera", False):
            default = getattr(self.frame_provider, "camera_key", None)
            if default and default in all_cameras:
                return [default]
            # 设置了 ``restrict_to_default_camera``，但配置的默认摄像头
            # 不是提供者暴露的摄像头之一。如果仍然返回它，
            # ``_decode`` 会在帧提取深处引发 KeyError，因此发出警告并
            # 改为回退到所有可用摄像头。
            if default:
                logging.getLogger(__name__).warning(
                    "restrict_to_default_camera is set but camera_key=%r is not in the "
                    "provider's cameras %s; grounding VQA on all available cameras instead.",
                    default,
                    all_cameras,
                )
        return all_cameras

    def _build_messages(
        self,
        record: EpisodeRecord,
        question_type: str,
        frame_timestamp: float,
        camera_key: str,
    ) -> list[dict[str, Any]]:
        prompt = load_prompt("vqa").format(
            episode_task=record.episode_task,
            question_type=question_type,
        )
        images = self.frame_provider.frames_at(record, [frame_timestamp], camera_key=camera_key)
        content = [*to_image_blocks(images), {"type": "text", "text": prompt}]
        return [{"role": "user", "content": content}]

    def _postprocess(self, result: Any) -> tuple[str, dict[str, Any]] | None:
        if not isinstance(result, dict):
            return None
        question = result.get("question")
        answer = result.get("answer")
        if not isinstance(question, str) or not question.strip():
            return None
        if not isinstance(answer, dict):
            return None
        # 验证器将强制检查形状；这里我们只是健全性检查答案
        # 是否匹配*某个*已知形状，以便尽早丢弃垃圾。
        if classify_vqa_answer(answer) is None:
            return None
        return question.strip(), answer


def _has_image_block(messages: list[dict[str, Any]]) -> bool:
    """如果任何用户内容块是已填充的图像块，则返回 True。"""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                return True
    return False
