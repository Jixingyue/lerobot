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
"""在 HF Jobs（HuggingFace GPU）上运行 ``lerobot-annotate``。

与 ``hf.py`` 中的训练提交器形状相同，有一个区别：标注
流水线服务于自己的 VLM，因此 pod 从官方的
``vllm/vllm-openai`` 镜像（没有 lerobot）启动，而不是预构建的
``lerobot-gpu`` 镜像，并在运行前在其上安装 lerobot。

因为没有配置仓库需要暂存，pod 重放用户自己的 CLI
标志——除了仅客户端的 ``--job.*`` 和主机本地的
``--root``（被替换为 ``--repo_id``，以便 pod 从 Hub 拉取数据集）
之外的所有内容。
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import is_dataclass
from typing import TYPE_CHECKING

from huggingface_hub import HfApi, get_token, run_job

from .dataset import ensure_dataset_available

# 包内部重用训练提交器的作业管道：跟踪已提交的作业
# 和转发 argv 对于标注运行是相同的。
from .hf import _pod_forwarded_args, follow_job, resolve_job_tags

if TYPE_CHECKING:
    from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig

LEROBOT_GIT_URL = "https://github.com/huggingface/lerobot.git"

# 镜像 pyproject.toml 中的固定版本。vLLM 镜像否则会自行解析依赖，
# 并拉取 av 18 / datasets 5 / draccus 0.11——每一个都会在导入时
# 破坏 lerobot。`--upgrade-strategy only-if-needed` 保持 vLLM 自己的
# （torch、transformers、...）固定版本不变。
_RUNTIME_REQUIREMENTS = (
    "'datasets>=4.7.0,<5.0.0' 'pyarrow>=21.0.0,<30.0.0' 'av>=15.0.0,<16.0.0' 'draccus==0.10.0' "
    "'pandas>=2.0.0,<3.0.0' jsonlines gymnasium torchcodec mergedeep pyyaml-include toml typing-inspect "
    "openai"
)

# 提交器自行解析而不是逐字转发的标志：`--root`
# 命名仅此机器具有的目录，`--repo_id` 从配置中重新发出，
# 配置文件参数命名本地文件（由 `submit_annotate_to_hf` 预先拒绝）。
# `--job.*` 单独按前缀删除；裸 `--job` 不删除，因此它在此处有条目——
# 它是唯一一个可以将远程 `target` 偷运到 pod 并让作业递归提交自身的参数。
_SUBMITTER_OWNED_ARGS = ("--root", "--repo_id", "--config_path", "--job")


def _local_config_file_args(cfg: AnnotationPipelineConfig) -> list[str]:
    """命名客户端磁盘上配置文件的 CLI 参数。

    draccus 为整个配置暴露 ``--config_path``，并为每个嵌套数据类
    （``--vlm``、``--plan``、``--job``、...）暴露一个 ``--<field>``。pod 没有
    这些文件，因此远程运行必须拒绝它们，而不是静默地
    丢弃它们携带的设置。
    """
    return ["--config_path", *(f"--{name}" for name in vars(cfg) if is_dataclass(getattr(cfg, name)))]


def build_pod_setup(lerobot_ref: str) -> str:
    """将 vLLM 镜像转换为 ``lerobot-annotate`` 运行时的 shell 前言。"""
    spec = f"lerobot @ git+{LEROBOT_GIT_URL}@{lerobot_ref}"
    return (
        # git 用于从仓库安装，ffmpeg 用于解码数据集的视频。
        "apt-get update -qq && apt-get install -y -qq git ffmpeg && "
        f"pip install --no-deps {shlex.quote(spec)} && "
        f"pip install --upgrade-strategy only-if-needed {_RUNTIME_REQUIREMENTS} && "
        # vLLM 的 cudagraph 内存估计过度预留并使 KV 缓存饥饿；
        # PyAV 是服务器可以解码我们帧的视频后端。
        "export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 && "
        "export VLLM_VIDEO_BACKEND=pyav"
    )


def build_pod_command(repo_id: str, lerobot_ref: str, argv: list[str]) -> list[str]:
    """构建 pod 运行的 ``bash -c`` 命令：设置前言，然后标注。

    ``argv`` 是用户的 CLI（``sys.argv[1:]``）减去 ``_SUBMITTER_OWNED_ARGS``
    中的标志；``--repo_id`` 从配置中重新添加，以便 pod
    始终标注我们刚刚确保在 Hub 上可访问的数据集。
    ``--job.target=local`` 阻止 pod 重新分发到自身。
    """
    forwarded = _pod_forwarded_args(argv, drop_names=_SUBMITTER_OWNED_ARGS, drop_prefixes=("--job.",))
    annotate = shlex.join(["lerobot-annotate", f"--repo_id={repo_id}", *forwarded, "--job.target=local"])
    return ["bash", "-c", f"{build_pod_setup(lerobot_ref)} && {annotate}"]


def submit_annotate_to_hf(cfg: AnnotationPipelineConfig) -> None:
    """向 HF Jobs 基础设施提交标注运行。

    解析凭据，确保源数据集可从 pod 访问，
    提交作业，然后尾随其日志直到作业到达终端阶段——
    或使用 ``--job.detach`` 立即返回。Ctrl-C 分离而不
    取消远程作业。
    """
    token = get_token()
    if not token:
        raise RuntimeError("Not logged in to Hugging Face. Run `hf auth login` first.")

    if cfg.repo_id is None:
        raise ValueError(
            "Remote annotation requires --repo_id: the pod downloads the dataset from the Hub, "
            "and --root only names a directory on this machine."
        )

    argv = sys.argv[1:]
    passed = {tok.split("=", 1)[0] for tok in argv}
    used_config_files = sorted(passed.intersection(_local_config_file_args(cfg)))
    if used_config_files:
        raise ValueError(
            f"{', '.join(used_config_files)} cannot be used with a remote --job.target: the pod "
            "cannot read config files from this machine. Pass the settings as CLI flags instead."
        )

    if not cfg.push_to_hub:
        # 作业结束时 pod 的文件系统会被丢弃，因此没有推送的话
        # 运行不会产生任何结果。警告而不是失败：对
        # --only_episodes 的冒烟测试只检查日志是合法的用例。
        print(
            "WARNING: --push_to_hub is off. The annotated dataset lives only on the pod and is "
            "discarded when the job ends. Pass --push_to_hub=true to keep the result."
        )

    api = HfApi(token=token)
    tags = resolve_job_tags(cfg.job.tags)
    ensure_dataset_available(cfg.repo_id, api=api, tags=tags)

    command = build_pod_command(cfg.repo_id, cfg.job.lerobot_ref, argv)

    print(f"Submitting job to HF Jobs (flavor={cfg.job.target}, image={cfg.job.image}) ...")
    job_info = run_job(
        image=cfg.job.image,
        command=command,
        flavor=cfg.job.target,
        secrets={"HF_TOKEN": token},
        timeout=cfg.job.timeout,
        # HF Jobs 标签是键/值；将每个标签暴露为可查询的标签。
        labels=dict.fromkeys(tags, "true"),
    )
    job_id = job_info.id
    job_url = getattr(job_info, "url", None)
    print(f"Job submitted: {job_id}")
    if job_url:
        print(f"  Job page:     {job_url}")
    target_repo_id = cfg.new_repo_id or cfg.repo_id
    if cfg.push_to_hub:
        print(f"  Dataset repo: https://huggingface.co/datasets/{target_repo_id}")
    print(f"  Monitor:      hf jobs logs {job_id}")
    print(f"  Cancel:       hf jobs cancel {job_id}")

    # 没有成功标记：`lerobot-annotate` 在上传日志行之后继续工作
    # （数据集卡片、版本标签），因此完成必须基于阶段。
    if not follow_job(job_id, detach=cfg.job.detach):
        return

    if cfg.push_to_hub:
        print(f"\nAnnotation complete — dataset pushed to https://huggingface.co/datasets/{target_repo_id}")
    else:
        print("\nAnnotation complete. Note: --push_to_hub was off, so the result stayed on the pod.")
