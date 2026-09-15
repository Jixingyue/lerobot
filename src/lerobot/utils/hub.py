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

import builtins
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

from huggingface_hub import HfApi
from huggingface_hub.utils import validate_hf_hub_args

from .constants import CHECKPOINTS_DIR

T = TypeVar("T", bound="HubMixin")


# 分片训练的恢复制品（torch DCP 分片目录 + 分片文件）。已发布的模型仓库
# 只携带 safetensors，因此发布时的上传会排除这些内容——而检查点推送
# （其存在是为了恢复，而非分发）则有意不排除。
def find_latest_hub_checkpoint(
    repo_id: str,
    *,
    token: str | bool | None = None,
    revision: str | None = None,
) -> str | None:
    """训练仓库中最新检查点的、相对于仓库根目录的路径。

    训练运行会将检查点推送到 ``checkpoints/<step>/``（参见
    ``push_checkpoint_to_hub``）。此函数列出这些步数目录并返回
    ``checkpoints/<最高步数>``；如果仓库中没有检查点则返回 ``None``。

    参数:
        repo_id (str): 要检查的 Hub 模型仓库。
        token (str | bool | None): Hub 认证令牌。默认为 None（即
            `huggingface-cli login` 缓存的令牌）。
        revision (str | None): 要列出的仓库修订版本。默认为 None（默认分支）。

    返回:
        str | None：相对于仓库的路径 `checkpoints/<最高步数>`；如果仓库
            没有检查点则为 None。
    """
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model", revision=revision, token=token)
    prefix = f"{CHECKPOINTS_DIR}/"
    steps = {
        name for f in files if f.startswith(prefix) and (name := f[len(prefix) :].split("/", 1)[0]).isdigit()
    }
    if not steps:
        return None
    return f"{CHECKPOINTS_DIR}/{max(steps, key=int)}"


class HubMixin:
    """
    一个 Mixin，包含将对象推送到 hub 的功能。

    这类似于 huggingface_hub.ModelHubMixin，但更轻量，对其子类的假设更少
    （尤其是，它不一定是一个模型）。

    继承类必须实现 '_save_pretrained' 和 'from_pretrained'。
    """

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        card_kwargs: dict[str, Any] | None = None,
        **push_to_hub_kwargs,
    ) -> str | None:
        """
        将对象保存到本地目录。

        参数:
            save_directory (`str` 或 `Path`):
                保存该对象的目录路径。
            push_to_hub (`bool`，*可选*，默认为 `False`):
                是否在保存后将对象推送到 Huggingface Hub。
            repo_id (`str`，*可选*):
                你在 Hub 上的仓库 ID。仅在 `push_to_hub=True` 时使用。若未
                提供，将默认使用文件夹名称。
            card_kwargs (`Dict[str, Any]`，*可选*):
                传递给卡片模板以自定义卡片的附加参数。
            push_to_hub_kwargs:
                传递给 [`~HubMixin.push_to_hub`] 方法的附加关键字参数。
        返回:
            `str` 或 `None`：当 `push_to_hub=True` 时为 Hub 上提交的 url，
            否则为 `None`。
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # 保存对象（权重、文件等）
        self._save_pretrained(save_directory)

        # 如有需要，推送到 Hub
        if push_to_hub:
            if repo_id is None:
                repo_id = save_directory.name  # 默认为 `save_directory` 的名称
            return self.push_to_hub(repo_id=repo_id, card_kwargs=card_kwargs, **push_to_hub_kwargs)
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        在子类中重写此方法，以定义如何保存你的对象。

        参数:
            save_directory (`str` 或 `Path`):
                保存该对象文件的目录路径。
        """
        raise NotImplementedError

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> T:
        """
        从 Huggingface Hub 下载该对象并实例化它。

        参数:
            pretrained_name_or_path (`str`、`Path`):
                - 可以是托管在 Hub 上的对象的 `repo_id`（字符串），例如 `lerobot/diffusion_pusht`。
                - 也可以是一个 `directory` 的路径，其中包含通过 `.save_pretrained`
                    保存的对象文件，例如 `../path/to/my_model_directory/`。
            revision (`str`，*可选*):
                Hub 上的修订版本。可以是分支名、git 标签或任意提交 id。
                默认为 `main` 分支上的最新提交。
            force_download (`bool`，*可选*，默认为 `False`):
                是否强制从 Hub（重新）下载文件，覆盖现有缓存。
            proxies (`Dict[str, str]`，*可选*):
                按协议或端点使用的代理服务器字典，例如 `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`。每次请求都会使用这些代理。
            token (`str` 或 `bool`，*可选*):
                用作远程文件 HTTP bearer 授权的令牌。默认使用运行
                `huggingface-cli login` 时缓存的令牌。
            cache_dir (`str`、`Path`，*可选*):
                缓存文件的存储文件夹路径。
            local_files_only (`bool`，*可选*，默认为 `False`):
                若为 `True`，则避免下载文件；如果本地缓存文件存在，则返回其路径。
            kwargs (`Dict`，*可选*):
                初始化对象时传递给它的附加 kwargs。
        """
        raise NotImplementedError

    @validate_hf_hub_args
    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str | None = None,
        private: bool | None = None,
        token: str | None = None,
        branch: str | None = None,
        create_pr: bool | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
        card_kwargs: dict[str, Any] | None = None,
    ) -> str | None:
        """
        将模型检查点上传到 Hub。

        使用 `allow_patterns` 和 `ignore_patterns` 可以精确筛选哪些文件应被推送到 hub。使用
        `delete_patterns` 可以在同一次提交中删除已有的远程文件。更多细节请参见
        [`upload_folder`] 参考文档。

        分布式约定：在每个 rank 上都调用。`save_pretrained` 在所有 rank 上
        运行——对于分片对象，它可能包含一次集合 gather（若用 rank 门控会导致死锁）——
        而仓库创建和上传只在主进程上进行。

        参数:
            repo_id (`str`):
                要推送到的仓库 ID（例如：`"username/my-model"`）。
            commit_message (`str`，*可选*):
                推送时提交的信息。
            private (`bool`，*可选*):
                创建的仓库是否为私有。
                若为 `None`（默认），仓库将是公开的，除非组织默认设为私有。
            token (`str`，*可选*):
                用作远程文件 HTTP bearer 授权的令牌。默认使用运行
                `huggingface-cli login` 时缓存的令牌。
            branch (`str`，*可选*):
                推送模型所使用的 git 分支。默认为 `"main"`。
            create_pr (`boolean`，*可选*):
                是否从 `branch` 针对该提交创建 Pull Request。默认为 `False`。
            allow_patterns (`List[str]` 或 `str`，*可选*):
                若提供，则只推送至少匹配其中一个模式的文件。
            ignore_patterns (`List[str]` 或 `str`，*可选*):
                若提供，则匹配任一模式的文件不会被推送。
            delete_patterns (`List[str]` 或 `str`，*可选*):
                若提供，匹配任一模式的远程文件将从仓库中删除。
            card_kwargs (`Dict[str, Any]`，*可选*):
                传递给卡片模板以自定义卡片的附加参数。

        返回:
            `str` 或 `None`：你的对象在给定仓库中的提交 url；
            在分布式运行的非主 rank 上为 `None`（只有主进程上传）。
        """
        # 懒导入：hub 代码在模块加载时不得导入 distributed 包
        # （configs -> hub 本身就位于 lerobot.distributed 的导入路径上）。
        from lerobot.distributed.utils import is_main_process

        # 分布式约定：`save_pretrained` 在每个 rank 上运行——对于分片策略，
        # 它包含一次集合 gather（用 rank 门控会导致死锁），并且只有
        # 主进程会写入该 rank 的私有临时目录。仓库创建和上传则
        # 仅限主进程。
        if commit_message is None:
            if "Policy" in self.__class__.__name__:
                commit_message = "Upload policy"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            else:
                commit_message = f"Upload {self.__class__.__name__}"

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path, card_kwargs=card_kwargs)
            if not is_main_process():
                return None
            api = HfApi(token=token)
            repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )
