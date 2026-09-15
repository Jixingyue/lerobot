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

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs.default import JobConfig

# 标注流水线会启动自己的 vLLM 服务器，因此 pod 从官方 vLLM 运行时启动，
# 而不是预构建的 `lerobot-gpu` 训练镜像；`lerobot` 在此之上通过 pip 安装
# （参见 `lerobot.jobs.annotate`）。
DEFAULT_ANNOTATE_JOB_IMAGE = "vllm/vllm-openai:latest"


@dataclass
class AnnotationJobConfig(JobConfig):
    """带有标注运行时默认值的 `JobConfig`。

    添加了 `lerobot_ref`，因为 vLLM 镜像不自带 lerobot：pod 会从 git 安装它，
    而 ref 决定实际执行标注的代码。将其指向分支/标签/SHA 以远程尝试未合并的更改。
    """

    image: str = DEFAULT_ANNOTATE_JOB_IMAGE
    # 标注是对数据集的有界遍历；比训练的"2d"更严格的上限可以防止卡住的 vLLM
    # 服务器浪费一整天的 GPU 时间。
    timeout: str | None = "2h"
    lerobot_ref: str = "main"


@dataclass
class PlanConfig:
    """``plan`` 模块：子任务 + 计划 + 记忆 + 任务增强。"""

    enabled: bool = True

    # 在 t=0 时的 ``task_aug`` 复述（渲染器在其中轮换 ${task}）；0 表示禁用。
    n_task_rephrasings: int = 10

    # 从视频而不是 episode_task 推导任务：off / if_short / always。
    # 仅影响提示词；``meta/tasks.parquet`` 不受影响。
    derive_task_from_video: str = "if_short"
    derive_task_min_words: int = 3

    # --- 帧输入：带时间戳的联系表（始终开启）---------------
    # 子任务描述/分割阶段始终将回合渲染为
    # macrodata/refiner 风格的联系表：采样帧打包成 JPEG 网格，
    # 每帧的时间戳烧录在其角落中，这样 VLM 可以直接引用边界的确切源时间。
    # 这在视觉 token 上比每帧一张图像便宜得多（实际上子任务生成速度约快 2 倍），
    # 这就是为什么默认采样是密集的。
    #
    # ``frames_per_second`` 是采样率：2.0 = 每 0.5 秒一帧。
    frames_per_second: float = 2.0
    # 每次 VLM 调用的帧预算（= 列 × 行 × 表）。当以
    # ``frames_per_second`` 采样的整个回合超过此值时，回合会被
    # 自动分割为连续的窗口，每个窗口包含
    # ``max_frames_per_prompt`` 帧（每个窗口一次 describe→segment 调用，
    # 仍保持完整的 ``frames_per_second`` 密度），然后
    # 各窗口的跨度会被合并 + 拼接成一个连续的覆盖。因此
    # 任何长度的回合始终以完整采样密度被覆盖。
    max_frames_per_prompt: int = 60
    contact_sheet_columns: int = 5
    contact_sheet_frames_per_sheet: int = 20
    contact_sheet_frame_width: int = 224
    contact_sheet_quality: int = 84

    min_subtask_seconds: float = 1.5
    plan_max_steps: int = 8

    # 分割前的仅叙述基础验证——这是防止从任务文本虚构子任务的最佳防御
    # （每个回合 +1 次 VLM 调用）。
    subtask_describe_first: bool = True

    # 种子重标注：在分割之后，使用一个聚焦的过程重新标注每个跨度，
    # 该过程查看前一个/当前/下一个片段的联系表并
    # 最小化地修正种子标签（macrodata 最佳的端到端标注步骤）。
    # 每个子任务花费 +1 次 VLM 调用；默认关闭。
    subtask_seeded_relabel: bool = False
    # 重标注过程中每个片段表均匀采样的帧数。
    subtask_relabel_frames: int = 5

    # 在每个边界处发出 ``style="plan"`` 行；False = 仅子任务 + 记忆。
    emit_plan: bool = True

    # 在每个边界处发出 ``style="memory"`` 行；False = 仅子任务（+ 计划）。
    # ``emit_plan`` 的对称对应项。
    emit_memory: bool = True

    # （子任务跨度始终被拼接到连续的全回合覆盖；不可配置。）

    # 可选的 EgoMimic 风格 5 轴任务增强；替代 n_task_rephrasings。
    task_aug_axes: TaskAugAxesConfig = field(default_factory=lambda: TaskAugAxesConfig())


@dataclass
class TaskAugAxesConfig:
    """5 轴 t=0 任务增强（EgoMimic 风格）：同义词 / 省略手臂 /
    省略朝向 / 省略抓取方法 / 组合。启用时替代 n_task_rephrasings；
    每个变体成为一个 ``task_aug`` 行。没有可省略内容的轴会发出更少的条目。
    默认值（3+3+2+2+2）与 EgoMimic 一致。"""

    enabled: bool = False

    synonym_paraphrase: int = 3
    omit_arm: int = 3
    omit_orientation: int = 2
    omit_grasp_method: int = 2
    combined_omissions: int = 2


@dataclass
class InterjectionsConfig:
    """``interjections`` 模块：插入语 + 配对语音。"""

    enabled: bool = True

    # 每个都会发出一个配对的（插入语，语音）行 + 在该时间戳处的计划刷新。
    max_interjections_per_episode: int = 3
    interjection_min_t: float = 2.0

    # 以时间戳为中心的帧窗口，以便 VLM 看到运动，而不是单帧。
    interjection_window_seconds: float = 2.0
    interjection_window_frames: int = 4


@dataclass
class VqaConfig:
    """``vqa`` 模块：通用 VQA。"""

    enabled: bool = True
    vqa_emission_hz: float = 1.0
    K: int = 1
    """每个发射节拍的连续帧数。VLM 以第一帧为基础，
    因此 K>1 会将过时的标签涂抹到移动的帧上。默认 1（无涂抹）。"""
    question_types: tuple[str, ...] = ("bbox", "keypoint", "count", "attribute", "spatial")

    # True：仅在 --vlm.camera_key 上进行 VQA 基础定位（默认：每个摄像头）。
    restrict_to_default_camera: bool = False


@dataclass
class VlmConfig:
    """共享的 Qwen-VL 客户端配置。"""

    # 仅 ``openai``（兼容 OpenAI 的 vLLM 服务器，当 auto_serve=True 时自动生成）；
    # ``stub`` 用于测试。
    backend: str = "openai"
    model_id: str = "Qwen/Qwen3.6-27B"

    # 兼容 OpenAI 的端点；``EMPTY`` 密钥适用于本地服务器。
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"

    # 如果没有服务器响应 api_base 则生成一个；False = 在远程服务器上快速失败。
    auto_serve: bool = True
    serve_port: int = 8000
    # 覆盖自动生成命令；``{port}`` 按副本替换。
    serve_command: str | None = None

    # 用于轮询路由的独立服务器（每个 GPU 一个）。num_gpus=0 = 每个一个。
    parallel_servers: int = 1
    num_gpus: int = 0
    client_concurrency: int = 16
    serve_ready_timeout_s: float = 600.0

    max_new_tokens: int = 512
    temperature: float = 0.2

    # 自动生成的上下文长度（None → 32768）；其他 vLLM 标志放在 serve_command 中。
    max_model_len: int | None = None

    # 关键帧的摄像头；None → 第一个 ``observation.images.*`` 键。
    camera_key: str | None = None
    # 作为 extra_body.chat_template_kwargs 转发（例如 {"enable_thinking": false}）。
    chat_template_kwargs: dict[str, Any] | None = None

    # OpenAI 风格的思考预算提示（"low"/"medium"/"high"）；设置时转发到
    # 服务器。用于限制思考模型的推理，使其在兼容 OpenAI 的端点上
    # 为实际的 JSON 答案留出 token。
    reasoning_effort: str | None = None


@dataclass
class ExecutorConfig:
    """执行器设置（进程内回合并发；通过 HF Jobs 进行分布式）。"""

    # 每个阶段并发处理的回合数；是使服务器饱和的主要旋钮。
    episode_parallelism: int = 16


@dataclass
class AnnotationPipelineConfig:
    """``lerobot-annotate`` 的顶层配置（原地重写数据分片）。"""

    # Hub 数据集：当 ``root`` 未设置时的下载源；当 push_to_hub 开启且
    # ``new_repo_id`` 未设置时的推送目标。
    repo_id: str | None = None

    # 单独的推送目标（与 LeRobot 编辑工具一致）。未设置 → 原地推送。
    new_repo_id: str | None = None

    root: Path | None = None

    # 默认为 ``<root>/.annotate_staging/``。
    staging_dir: Path | None = None

    seed: int = 1729

    plan: PlanConfig = field(default_factory=PlanConfig)
    interjections: InterjectionsConfig = field(default_factory=InterjectionsConfig)
    vqa: VqaConfig = field(default_factory=VqaConfig)

    vlm: VlmConfig = field(default_factory=VlmConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)

    # 标注运行的位置：省略 / "local" 在本机上标注，任何其他值都是
    # HF Jobs 风格（例如 "h200"）并在那里提交运行。
    # 使用 `hf jobs hardware` 列出风格 + 价格。
    job: AnnotationJobConfig = field(default_factory=AnnotationJobConfig)

    skip_validation: bool = False
    only_episodes: tuple[int, ...] | None = None

    # 转发到 ``decode_video_frames`` 的关键帧解码后端。None → 库默认
    # （torchcodec 可用时使用，否则 PyAV）。或显式指定
    # ``"torchcodec"`` / ``"pyav"``。
    video_backend: str | None = None

    # 上传到 Hub（如果设置则用 new_repo_id，否则 repo_id；必须设置其中一个）。
    push_to_hub: bool = False
    push_private: bool = False
    push_commit_message: str | None = None

    def resolved_staging_dir(self, root: Path) -> Path:
        return self.staging_dir if self.staging_dir is not None else root / ".annotate_staging"
