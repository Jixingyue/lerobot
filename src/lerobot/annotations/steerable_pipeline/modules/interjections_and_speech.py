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
"""``interjections`` 模块：插话 + 配对的语音（EVENT 风格 + 语音原子）。

两个子过程：

1. 在 ``t=0`` 时，仅发射一个语音工具调用原子（对规范任务的
   确认应答）。不发射插话行——规范任务本身已经是来自
   ``meta/tasks.parquet`` 的用户话语。

2. 对于片段中部的打断，发射一个同时间戳的行对：
       {role:user, style:interjection, content:<text>}
       语音原子 (role:assistant, style:None, tool_calls=[say(...)])
   两行都以相同的时间戳放入 ``language_events``。

``plan`` 模块的 :meth:`run_plan_updates` 会复用本模块的
插话时间戳，在同一时刻刷新 ``plan`` 行。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import InterjectionsConfig
from ..frames import FrameProvider, null_provider, to_image_blocks
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord, reconstruct_subtask_spans, snap_to_frame
from ..staging import EpisodeStaging
from ..vlm_client import VlmClient
from ..writer import speech_atom


@dataclass
class InterjectionsAndSpeechModule:
    """生成任务开始时的语音以及片段中部的插话/语音对。"""

    vlm: VlmClient
    config: InterjectionsConfig
    seed: int = 1729
    frame_provider: FrameProvider = field(default_factory=null_provider)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        rows: list[dict[str, Any]] = []
        if record.frame_timestamps:
            t0 = float(record.frame_timestamps[0])
            initial = self._initial_speech(record)
            if initial:
                rows.append(speech_atom(t0, initial))
        # 拉取 ``plan`` 模块为本片段生成的子任务区间，使插话提示词
        # 能够在每个选定的时间戳上以实际正在进行的当前子任务为
        # 依据。``plan`` 模块已先行运行。
        episode_end_t = float(record.frame_timestamps[-1]) if record.frame_timestamps else None
        subtask_spans = reconstruct_subtask_spans(staging.read("plan"), episode_end_t=episode_end_t)
        rows.extend(self._mid_episode_interjections(record, subtask_spans))
        staging.write("interjections", rows)

    @staticmethod
    def _subtask_at(spans: Sequence[dict[str, Any]], t: float) -> str | None:
        current: str | None = None
        for span in spans:
            if float(span["start"]) <= t:
                current = span.get("text")
            else:
                break
        return current

    def _initial_speech(self, record: EpisodeRecord) -> str | None:
        prompt = load_prompt("interjections_initial_speech").format(
            episode_task=record.episode_task,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        result = self.vlm.generate_json([messages])[0]
        if isinstance(result, dict) and isinstance(result.get("text"), str):
            text = result["text"].strip()
            if text:
                return text
        return None

    def _mid_episode_interjections(
        self,
        record: EpisodeRecord,
        subtask_spans: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """生成与实际演示轨迹对齐的插话。

        遥操作数据是冻结的——机器人已经在视频中执行了每一个
        步骤。像"其实跳过擦拭这一步"这样的*反事实*插话与视频
        中随后发生的事情相矛盾，这正是 qwen36moe-10/11 所暴露出
        的低质量插话问题。

        取而代之的做法是，将每个插话锚定在一个子任务边界上，
        并将其写成一个针对*即将进行*的子任务的自然人用户请求。
        机器人可见的下一步行为正是该插话的效果，因此训练信号
        保持一致：插话文本 → 计划刷新 → 动作流全部对齐。
        """
        if self.config.max_interjections_per_episode <= 0:
            return []
        if len(subtask_spans) < 2:
            # 至少需要一次转换（子任务 0 → 子任务 1）。
            return []
        # 每个片段使用确定性的随机数生成器，使重跑在多个 SLURM 作业间保持稳定。
        rng = random.Random(f"{self.seed}:{record.episode_index}:interjection")

        # 边界：除第一个子任务外每个子任务的开始时间
        # （第一个子任务就是 t0，已由初始任务语音原子覆盖）。
        boundaries: list[tuple[float, str, str]] = []
        for i in range(1, len(subtask_spans)):
            ts = float(subtask_spans[i]["start"])
            if ts < self.config.interjection_min_t:
                continue
            prev_text = (subtask_spans[i - 1].get("text") or "").strip()
            next_text = (subtask_spans[i].get("text") or "").strip()
            if not next_text:
                continue
            boundaries.append((ts, prev_text, next_text))
        if not boundaries:
            return []

        n = min(self.config.max_interjections_per_episode, len(boundaries))
        chosen = sorted(rng.sample(boundaries, n), key=lambda b: b[0])

        out: list[dict[str, Any]] = []
        for t, prev_subtask, next_subtask in chosen:
            t_snap = snap_to_frame(t, record.frame_timestamps)
            # 窗口横跨边界，使 VLM 能看到前一个子任务的结尾和
            # 下一个子任务的开头——与策略在训练时看到的条件
            # 信息相同。
            window_ts = self._window_timestamps(t_snap, record.frame_timestamps)
            prompt = load_prompt("interjections_interjection").format(
                episode_task=record.episode_task,
                prev_subtask=prev_subtask or "(starting from initial state)",
                next_subtask=next_subtask,
                timestamp=t_snap,
                window_seconds=self.config.interjection_window_seconds,
            )
            images = self.frame_provider.frames_at(record, window_ts)
            content = [*to_image_blocks(images), {"type": "text", "text": prompt}]
            messages = [{"role": "user", "content": content}]
            result = self.vlm.generate_json([messages])[0]
            if not isinstance(result, dict):
                continue
            interjection_text = result.get("interjection")
            speech_text = result.get("speech")
            if not isinstance(interjection_text, str) or not interjection_text.strip():
                continue
            if not isinstance(speech_text, str) or not speech_text.strip():
                continue
            out.append(
                {
                    "role": "user",
                    "content": interjection_text.strip(),
                    "style": "interjection",
                    "timestamp": t_snap,
                    "tool_calls": None,
                }
            )
            out.append(speech_atom(t_snap, speech_text.strip()))
        return out

    def _window_timestamps(self, t_anchor: float, frame_timestamps: Sequence[float]) -> list[float]:
        """返回以 ``t_anchor`` 为中心的一小组帧时间戳。

        窗口横跨插话所处的子任务边界：大约一半的帧覆盖前一个
        子任务的结尾，另一半覆盖下一个子任务的开头。因此 VLM
        既能看到刚刚完成的内容，也能看到即将开始的内容，而这
        正是写出一个与可见的后续行为相匹配的自然"现在请做 X"
        请求所需的条件信息。
        """
        if not frame_timestamps:
            return [t_anchor]
        n = max(1, int(self.config.interjection_window_frames))
        if n == 1:
            return [t_anchor]
        window = float(self.config.interjection_window_seconds)
        step = window / max(1, n - 1)
        # 将窗口居中于锚点，使一半落在其前，一半落在其后。
        start_offset = -window / 2.0
        targets = [t_anchor + start_offset + step * i for i in range(n)]
        first_ts = float(frame_timestamps[0])
        last_ts = float(frame_timestamps[-1])
        snapped: list[float] = []
        seen: set[float] = set()
        for tgt in targets:
            clamped = min(last_ts, max(first_ts, tgt))
            t = snap_to_frame(clamped, frame_timestamps)
            if t not in seen:
                seen.add(t)
                snapped.append(t)
        return snapped or [t_anchor]
