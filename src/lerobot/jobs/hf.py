# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""在 HF Jobs（HuggingFace GPU）上运行 lerobot 训练。

从 lelab 的 runners/hf_cloud.py 移植并简化：没有 UI 日志队列，
没有注册表——只是提交并流式输出到 stdout。
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import netrc
import os
import re
import signal
import sys
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from huggingface_hub import (
    HfApi,
    create_repo,
    fetch_job_logs,
    get_token,
    inspect_job,
    run_job,
    upload_file,
)

from lerobot.common.train_utils import push_checkpoint_to_hub
from lerobot.configs import parser

from .dataset import ensure_dataset_available

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")

_TERMINAL_STAGES = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}

# huggingface_hub 1.x 运行在 httpx 之上：瞬态 HTTP/传输层故障表现为
# httpx.HTTPError，套接字层错误表现为 OSError。只捕获这些可以防止真正的
# bug（TypeError、AttributeError 等）被静默重试或被计为
# 作业失败。
_TRANSIENT_NET_ERRORS = (OSError, httpx.HTTPError)

# 始终附加到远程作业和已推送的数据集，以便在 Hub 上识别
# LeRobot 产生的工作；调用方（例如 LeLab）通过 --job.tags 添加自己的标签。
LEROBOT_TAG = "lerobot"


def resolve_job_tags(extra: list[str] | None) -> list[str]:
    """返回运行的标签列表：lerobot 标签加上所有附加标签，去重，顺序稳定。"""
    tags = [LEROBOT_TAG, *(extra or [])]
    seen: set[str] = set()
    return [t for t in tags if not (t in seen or seen.add(t))]


def resolve_wandb_api_key() -> str | None:
    """用于转发给作业的主机 wandb 密钥：$WANDB_API_KEY，否则 ~/.netrc。"""
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    try:
        rc = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return None
    auth = rc.authenticators("api.wandb.ai")
    if auth is None:
        return None
    _login, _account, password = auth
    return password or None


def build_repo_id(username: str, job_name: str, now: dt.datetime) -> str:
    """为远程运行生成模型仓库 id：<user>/<job_name>_<timestamp>。"""
    slug = _SLUG_RE.sub("-", job_name).strip("-") or "train"
    stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{username}/{slug}_{stamp}"


def build_remote_config_file(cfg, repo_id: str, dest: Path, tags: list[str] | None = None) -> Path:
    """为 pod 写入应用了远程覆盖的 train_config.json。

    pod 运行 `lerobot-train --config_path=<dest>`，并通过 repo_id 将数据集
    下载到自己的缓存中。剥离仅客户端字段，使配置能被训练器镜像接受：
    `job`（纯客户端编排）始终被移除，`save_checkpoint_to_hub` 除非显式启用
    否则被移除——旧的 lerobot 镜像会拒绝未知键，因此默认值使配置
    与已发布的 `lerobot-gpu` 镜像保持兼容。`tags` 被合并到
    policy.tags 中，使 pod 推送的已训练模型也携带它们。
    """
    remote = copy.deepcopy(cfg)
    remote.policy.push_to_hub = True
    remote.policy.repo_id = repo_id
    # 不要固定客户端解析出的设备（例如 "mps"）；让 pod 自动检测其 GPU。
    remote.policy.device = None
    # 丢弃任何主机本地数据集根目录；pod 通过 repo_id 解析数据集。
    remote.dataset.root = None
    if tags:
        existing = list(remote.policy.tags or [])
        remote.policy.tags = existing + [t for t in tags if t not in existing]

    # 编码为规范的、pod 可解析的字典，然后丢弃已发布的
    # 训练器镜像不认识的键。
    data = remote.to_dict()
    data.pop("job", None)
    if not remote.save_checkpoint_to_hub:
        data.pop("save_checkpoint_to_hub", None)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=4))
    return dest


def _stage_config_on_hub(cfg, repo_id: str, token: str, tags: list[str] | None = None) -> str:
    """将 train_config.json 上传到模型仓库，并返回用于 --config_path 的 repo_id。"""
    create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=token)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = build_remote_config_file(cfg, repo_id, Path(tmp) / "train_config.json", tags=tags)
        upload_file(
            path_or_fileobj=config_path,
            path_in_repo="train_config.json",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
    return repo_id


def _tail_logs(
    job_id: str,
    done: threading.Event,
    success_marker: str | None = None,
    success_event: threading.Event | None = None,
) -> None:
    """将作业日志流式输出到 stdout，在流断开时重新连接，直到 done 被设置。

    每次重新连接都会重新获取完整的缓冲日志，因此我们跟踪已打印的
    行数并跳过它们——否则快速失败的作业的回溯会在每次重新连接时
    被重复打印。

    当 `success_marker` 出现在某一行时，设置 `success_event` 和 `done`，
    使调用方能在已训练模型到达 Hub 后立即完成，而不是
    等待平台的运行后收尾（这可能增加约 30 秒）。
    """
    printed = 0
    while not done.is_set():
        try:
            seen = 0
            for line in fetch_job_logs(job_id=job_id, follow=True):
                seen += 1
                if seen <= printed:
                    continue  # 已在前一次连接中显示过
                printed = seen
                # fetch_job_logs 产生的 SSE 数据没有结尾换行符，因此为每个
                # 条目添加一个——否则所有日志行会连成一行。
                print(line.rstrip("\n"), flush=True)
                if success_marker and success_event is not None and success_marker in line:
                    success_event.set()
                    done.set()
                    return
                if done.is_set():
                    return
            # 流干净地关闭了。稍等片刻，让状态轮询器在我们重新连接
            # 之前将作业标记为终态（避免重新尾随缓冲区）。
            if done.wait(3):
                return
        except _TRANSIENT_NET_ERRORS:
            if done.wait(2):
                return


def _poll_until_done(
    job_id: str,
    done: threading.Event,
    poll_interval: float = 5.0,
    status_holder: dict | None = None,
    max_failures: int = 6,
) -> str | None:
    """轮询 inspect_job 直到终态阶段或 `done` 被设置。

    返回终态阶段字符串，如果 `done` 先被设置（分离）或在
    `max_failures` 次连续 inspect_job 错误之后则返回 None。当到达终态阶段
    且提供了 `status_holder` 时，记录 `status_holder["message"]`
    （平台的状态消息，例如 "Job timeout"）。
    """
    failures = 0
    while not done.is_set():
        try:
            info = inspect_job(job_id=job_id)
            failures = 0
            # 在某些 huggingface_hub 版本中 `stage` 是枚举，在其他版本中是普通字符串。
            stage = getattr(info.status.stage, "value", info.status.stage)
            if stage in _TERMINAL_STAGES:
                if status_holder is not None:
                    status_holder["message"] = getattr(info.status, "message", None)
                done.set()
                return stage
        except _TRANSIENT_NET_ERRORS:
            failures += 1
            if failures >= max_failures:
                done.set()
                return None
        done.wait(poll_interval)
    return None


def follow_job(job_id: str, *, detach: bool = False, success_marker: str | None = None) -> bool:
    """监视已提交的作业直到结束，将其日志流式输出到 stdout。

    作业成功完成时返回 True，在没有结论的情况下停止监视时返回 False——
    `detach`，或用户按下 Ctrl-C（这会使作业分离而不是取消远程作业）。
    当作业到达 COMPLETED 以外的终态阶段时引发 RuntimeError。

    `success_marker` 在该字符串出现在日志中时立即完成，而不是等待
    平台的运行后收尾（约 30 秒）。拥有表示"工件已在 Hub 上"的
    日志行的调用方应传入它；没有的话，完成判定基于阶段。
    """
    if detach:
        return False

    done = threading.Event()
    detached = threading.Event()
    marker_seen = threading.Event()
    stage_holder: dict[str, str | None] = {}

    def _poll() -> None:
        stage_holder["stage"] = _poll_until_done(job_id, done, status_holder=stage_holder)

    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()
    log_thread = threading.Thread(
        target=_tail_logs, args=(job_id, done, success_marker, marker_seen), daemon=True
    )
    log_thread.start()

    def _detach(sig, frame):
        detached.set()
        done.set()
        print("\nDetached. Job is still running.")
        print(f"  Monitor: hf jobs logs {job_id}")
        print(f"  Cancel:  hf jobs cancel {job_id}")

    # signal.signal 只在主线程上有效；从工作线程调用时
    # （例如编排框架）跳过 Ctrl-C 分离而不是取消的
    # 处理器，而不是因 ValueError 而崩溃。
    install_sigint = threading.current_thread() is threading.main_thread()
    original_sigint = signal.getsignal(signal.SIGINT) if install_sigint else None
    if install_sigint:
        signal.signal(signal.SIGINT, _detach)
    try:
        # 基于超时的 join，以便 SIGINT 能及时传递到主线程。
        while poll_thread.is_alive():
            poll_thread.join(timeout=0.5)
        log_thread.join(timeout=5)
    finally:
        if install_sigint:
            signal.signal(signal.SIGINT, original_sigint)

    if detached.is_set():
        return False
    if marker_seen.is_set():
        return True

    stage = stage_holder.get("stage")
    if stage != "COMPLETED":
        message = stage_holder.get("message")
        detail = f" ({message})" if message else ""
        raise RuntimeError(
            f"Job {job_id} ended with stage={stage}{detail}. Check logs: hf jobs logs {job_id}"
        )
    return True


def _pod_forwarded_args(
    argv: list[str], drop_names: tuple[str, ...] = (), drop_prefixes: tuple[str, ...] = ()
) -> list[str]:
    """要在 pod 上重放的用户 CLI 覆盖项，减去提交器自己设置的标志。

    处理 `--name=value` 和 `--name value` 两种形式。转发用户的覆盖项（例如
    `--steps`、`--save_checkpoint_to_hub`）使远程恢复的行为与相同的本地命令一致。
    """
    out: list[str] = []
    skip_next = False
    for i, tok in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        name = tok.split("=", 1)[0]
        if name in drop_names or any(name.startswith(p) for p in drop_prefixes):
            if "=" not in tok and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                skip_next = True  # 同时丢弃空格分隔的值
            continue
        out.append(tok)
    return out


def _build_resume_job(cfg: TrainPipelineConfig, username: str) -> tuple[str, list[str]]:
    """解析模型仓库和 pod 命令，以在作业上恢复运行。

    Hub 的 `config_path` 直接从中恢复：其检查点配置已经指向该仓库，
    因此新检查点在那里延续谱系。本地 `config_path` 会先将其检查点上传到
    新的 PRIVATE 仓库，并强制恢复的运行推送回该仓库。pod 命令
    始终携带 `--job.target=local`，使检查点保存的 `job.target` 不会让 pod
    重新分发自身。
    """
    config_path = parser.parse_arg("config_path")
    forwarded = _pod_forwarded_args(
        sys.argv[1:],
        drop_names=("--config_path", "--policy.repo_id", "--policy.push_to_hub", "--dataset.root"),
        drop_prefixes=("--job.",),
    )

    if Path(config_path).exists():
        # 本地检查点：将其暂存到 Hub，使 pod 可以从中恢复，并推送回那里。
        # 解析路径，使 `last` 符号链接以其真实步骤名（数字）上传，
        # pod 的最新检查点查找以此为键。
        checkpoint_dir = Path(cfg.checkpoint_path).resolve()
        source_repo = build_repo_id(username, cfg.job_name or "train", dt.datetime.now(dt.UTC))
        push_checkpoint_to_hub(checkpoint_dir, source_repo, private=True)
        extra = [f"--policy.repo_id={source_repo}", "--policy.push_to_hub=true"]
    else:
        source_repo = config_path
        extra = []

    command = [
        "lerobot-train",
        *forwarded,
        f"--config_path={source_repo}",
        "--job.target=local",
        *extra,
    ]
    return source_repo, command


def submit_to_hf(cfg: TrainPipelineConfig) -> None:
    """向 HF Jobs 基础设施提交训练作业。

    验证 cfg，解析凭据，确保数据集在 Hub 上，然后暂存一个
    清理过的配置（全新运行）或从检查点仓库恢复，提交作业，并尾随日志
    直到完成或立即分离。Ctrl-C 分离而不取消远程作业。
    """
    token = get_token()
    if not token:
        raise RuntimeError("Not logged in to Hugging Face. Run `hf auth login` first.")

    api = HfApi(token=token)
    user_info = api.whoami(token=token)
    username = user_info["name"]

    now = dt.datetime.now(dt.UTC)
    fresh_repo_id: str | None = None
    if not cfg.resume:
        # 在 validate() 之前解析模型仓库并标记推送：只要 push_to_hub 为 True，
        # validate() 就要求设置 repo_id。（恢复则复用检查点的仓库。）
        if cfg.policy is not None:
            base_name = cfg.job_name or cfg.policy.type
            fresh_repo_id = cfg.policy.repo_id or build_repo_id(username, base_name, now)
            cfg.policy.repo_id = fresh_repo_id
            cfg.policy.push_to_hub = True
        else:
            # 基于路径的策略在 validate() 内部解析；回退到通用 slug。
            fresh_repo_id = build_repo_id(username, cfg.job_name or "train", now)

    cfg.validate()

    if cfg.is_reward_model_training:
        raise ValueError(
            "Remote training via --job.target only supports policy training, not reward models. "
            "Run reward-model training locally."
        )

    secrets: dict[str, str] = {"HF_TOKEN": token}
    if cfg.wandb.enable:
        wandb_key = resolve_wandb_api_key()
        if wandb_key is None:
            raise ValueError(
                "wandb is enabled but no WANDB_API_KEY found. "
                "Set it via `export WANDB_API_KEY=...` or add it to ~/.netrc."
            )
        secrets["WANDB_API_KEY"] = wandb_key

    tags = resolve_job_tags(cfg.job.tags)
    # 对于全新运行和恢复运行，数据集都必须能从 pod 访问；仅本地的
    # 数据集在这里以 PRIVATE 推送。由于对两者都适用，提升到恢复/全新分支之前。
    ensure_dataset_available(cfg.dataset.repo_id, api=api, tags=tags)

    if cfg.resume:
        repo_id, command = _build_resume_job(cfg, username)
    else:
        config_repo_id = _stage_config_on_hub(cfg, fresh_repo_id, token, tags=tags)
        repo_id = fresh_repo_id
        command = ["lerobot-train", f"--config_path={config_repo_id}"]

    print(f"Submitting job to HF Jobs (flavor={cfg.job.target}, image={cfg.job.image}) ...")
    job_info = run_job(
        image=cfg.job.image,
        command=command,
        flavor=cfg.job.target,
        secrets=secrets,
        timeout=cfg.job.timeout,
        # HF Jobs 标签是键/值；将每个标签暴露为可查询的标签。
        labels=dict.fromkeys(tags, "true"),
    )
    job_id = job_info.id
    job_url = getattr(job_info, "url", None)
    print(f"Job submitted: {job_id}")
    if job_url:
        print(f"  Job page:   {job_url}")
    print(f"  Model repo: https://huggingface.co/{repo_id}")
    print(f"  Monitor:    hf jobs logs {job_id}")
    print(f"  Cancel:     hf jobs cancel {job_id}")

    # 在模型推送后立即完成，而不是在作业阶段翻转为 COMPLETED 之前
    # 等待平台的运行后收尾。这与
    # lerobot.common.train_utils.publish_trained_model 发出的确切日志行匹配——两者必须保持
    # 同步。如果不再匹配，我们只是回退到基于阶段的完成判定
    # （慢约 30 秒），因此该约定是优化而非正确性要求。
    success_marker = f"Model pushed to https://huggingface.co/{repo_id}"
    if follow_job(job_id, detach=cfg.job.detach, success_marker=success_marker):
        print(f"\nTraining complete — model pushed to https://huggingface.co/{repo_id}")
