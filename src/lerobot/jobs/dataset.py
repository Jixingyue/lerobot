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
"""使训练数据集可从 HF Job pod 访问。

pod 看不到主机的 ~/.cache/huggingface/lerobot，因此数据集必须
位于 Hub 上：pod 在训练时通过 repo_id 下载它（转发的
HF_TOKEN 覆盖私有数据集）。已在 Hub 上的数据集按原样使用；
仅本地数据集先推送到 PRIVATE 仓库（从不公开）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

if TYPE_CHECKING:
    from huggingface_hub import HfApi


def ensure_dataset_available(repo_id: str, *, api: HfApi, tags: list[str] | None = None) -> None:
    """确保 repo_id 在 Hub 上可解析，先将仅本地数据集推送到私有仓库。

    `tags` 仅在我们推送时附加到数据集（已在 Hub 上的
    数据集保持不变）。如果数据集既不在 Hub 上也不在
    本地缓存中，则引发 RuntimeError。
    """
    if api.repo_exists(repo_id, repo_type="dataset"):
        return

    local_present = (HF_LEROBOT_HOME / repo_id / "meta" / "info.json").is_file()
    if not local_present:
        raise RuntimeError(
            f"Dataset '{repo_id}' is not in the local cache ({HF_LEROBOT_HOME}) and could not be "
            f"reached on the Hub — it may not exist, or be private and inaccessible with your "
            f"token. Record or download it first, or run `hf auth login`."
        )

    print(f"[dataset] '{repo_id}' is local-only; pushing to a PRIVATE Hub repo...")
    LeRobotDataset(repo_id).push_to_hub(private=True, tags=tags)
    print(f"[dataset] '{repo_id}' uploaded (private). The job will download it by repo_id.")
