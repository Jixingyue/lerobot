#!/usr/bin/env python

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
import contextlib
import dataclasses
import importlib.resources
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import numpy as np
import packaging.version
import torch
from huggingface_hub import DatasetCard, DatasetCardData, HfApi

from lerobot.utils.utils import flatten_dict, unflatten_dict

V30_MESSAGE = """
The dataset you requested ({repo_id}) is in {version} format.

We introduced a new format since v3.0 which is not backward compatible with v2.1.
Please, update your dataset to the new format using this command:
```
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id={repo_id}
```

If you already have a converted version uploaded to the hub, then this error might be because of
an older version in your local cache. Consider deleting the cached version and retrying.

If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).
"""

FUTURE_MESSAGE = """
The dataset you requested ({repo_id}) is only available in {version} format.
As we cannot ensure forward compatibility with it, please update your current version of lerobot.
"""

MISSING_VERSION_TAG_MESSAGE = """
Your dataset must be tagged with a codebase version.
Assuming _version_ is the codebase_version value in the info.json, you can run this:
```python
from huggingface_hub import HfApi

hub_api = HfApi()
hub_api.create_tag("{repo_id}", tag="_version_", repo_type="dataset")
```
"""


class CompatibilityError(Exception): ...


class BackwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        if version.major == 2 and version.minor == 1:
            message = V30_MESSAGE.format(repo_id=repo_id, version=version)
        else:
            raise NotImplementedError(
                "Contact the maintainer on [Discord](https://discord.com/invite/s3KuuzsPFb)."
            )
        super().__init__(message)


class ForwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = FUTURE_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)


logger = logging.getLogger(__name__)


DEFAULT_CHUNK_SIZE = 1000  # 每个分片的最大文件数
DEFAULT_DATA_FILE_SIZE_IN_MB = 100  # 每个文件的最大大小
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200  # 每个文件的最大大小

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"

EPISODES_DIR = "meta/episodes"
DATA_DIR = "data"
VIDEO_DIR = "videos"

CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
IMAGE_FILE_PATTERN = "frame-{frame_index:06d}.png"


def resolve_episode_indices(
    episodes: Sequence[int] | None,
    total_episodes: int,
    exclude_episodes: Sequence[int] | None = None,
) -> list[int] | None:
    """针对数据集边界解析可选的 episode 允许列表和排除列表。

    当未请求任何过滤时保留 ``None``，以便调用方保留其
    原生的“所有 episode”快速路径。无效索引会被忽略并发出
    警告，同时保留输入顺序。
    """
    if total_episodes < 0:
        raise ValueError(f"total_episodes must be non-negative, got {total_episodes}")

    if episodes is None and not exclude_episodes:
        return None

    candidates = list(range(total_episodes)) if episodes is None else list(episodes)
    invalid = [episode for episode in candidates if not 0 <= episode < total_episodes]
    if invalid:
        logger.warning(
            "Ignoring episode indices outside the dataset range [0, %d): %s",
            total_episodes,
            invalid,
        )
    candidates = [episode for episode in candidates if 0 <= episode < total_episodes]

    excluded = set(exclude_episodes or [])
    invalid_excluded = sorted(episode for episode in excluded if not 0 <= episode < total_episodes)
    if invalid_excluded:
        logger.warning(
            "Ignoring excluded episode indices outside the dataset range [0, %d): %s",
            total_episodes,
            invalid_excluded,
        )
    excluded = {episode for episode in excluded if 0 <= episode < total_episodes}
    return [episode for episode in candidates if episode not in excluded]


DEPTH_FILE_PATTERN = "frame-{frame_index:06d}.tiff"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
DEFAULT_EPISODES_PATH = EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_DATA_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_VIDEO_PATH = VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode-{episode_index:06d}/" + IMAGE_FILE_PATTERN
DEFAULT_DEPTH_PATH = "images/{image_key}/episode-{episode_index:06d}/" + DEPTH_FILE_PATTERN

LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"


@dataclass
class DatasetInfo:
    """LeRobot 数据集 ``meta/info.json`` 文件的带类型表示。

    取代了此前由 ``load_info()`` 返回、由
    ``create_empty_dataset_info()`` 创建的无类型 ``dict``。
    使用 dataclass 可以提供显式的字段定义、IDE 自动补全，
    以及构造时的校验。
    """

    codebase_version: str
    fps: int
    features: dict[str, dict]

    # Episode / 帧计数器 —— 新数据集从零开始
    total_episodes: int = 0
    total_frames: int = 0
    total_tasks: int = 0

    # 存储设置
    chunks_size: int = field(default=DEFAULT_CHUNK_SIZE)
    data_files_size_in_mb: int = field(default=DEFAULT_DATA_FILE_SIZE_IN_MB)
    video_files_size_in_mb: int = field(default=DEFAULT_VIDEO_FILE_SIZE_IN_MB)

    # 文件路径模板
    data_path: str = field(default=DEFAULT_DATA_PATH)
    video_path: str | None = field(default=DEFAULT_VIDEO_PATH)

    # 持有底层数据文件的格式。``None`` 表示内置的
    # parquet/mp4 布局；任何其他值（例如 "lance"）都会将
    # LeRobotDataset 的数据访问路由到为该格式注册的存储后端。
    storage_format: str | None = None

    # 可选元数据
    robot_type: str | None = None
    splits: dict[str, str] = field(default_factory=dict)
    # 数据集声明的 OpenAI 风格工具模式。``None`` 表示
    # 数据集没有声明任何工具——读取器会回退到 ``DEFAULT_TOOLS``。
    tools: list[dict] | None = None

    def __post_init__(self) -> None:
        # 将特征形状从 list 强制转换为 tuple —— JSON 反序列化
        # 返回的是 list，但代码库的其他部分期望 tuple。
        for ft in self.features.values():
            if isinstance(ft.get("shape"), list):
                ft["shape"] = tuple(ft["shape"])

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.chunks_size <= 0:
            raise ValueError(f"chunks_size must be positive, got {self.chunks_size}")
        if self.data_files_size_in_mb <= 0:
            raise ValueError(f"data_files_size_in_mb must be positive, got {self.data_files_size_in_mb}")
        if self.video_files_size_in_mb <= 0:
            raise ValueError(f"video_files_size_in_mb must be positive, got {self.video_files_size_in_mb}")

    def to_dict(self) -> dict:
        """返回一个可 JSON 序列化的字典。

        将 tuple 形状转换回 list，以便 ``json.dump`` 能够处理。
        当 ``tools`` 和 ``storage_format`` 未设置时将其移除，
        使已有数据集保持干净的 ``info.json``。
        """
        d = dataclasses.asdict(self)
        for ft in d["features"].values():
            if isinstance(ft.get("shape"), tuple):
                ft["shape"] = list(ft["shape"])
        if d.get("tools") is None:
            d.pop("tools", None)
        if d.get("storage_format") is None:
            d.pop("storage_format", None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetInfo":
        """从原始字典构造（例如直接从 JSON 加载的字典）。

        为前向兼容携带额外字段的数据集（例如来自 v2.x 的
        ``total_videos``），未知键会被忽略。当存在此类字段时
        会记录一条警告。
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(k for k in data if k not in known)
        if unknown:
            logger.warning(f"Unknown fields in DatasetInfo: {unknown}. These will be ignored.")
        return cls(**{k: v for k, v in data.items() if k in known})

    # ---------------------------------------------------------------------------
    # 临时的字典风格兼容层
    # 允许已有的 ``info["key"]`` 调用点无需改动即可继续工作。
    # 待所有调用方迁移到属性访问后，删除这些方法。
    # ---------------------------------------------------------------------------
    def __getitem__(self, key: str):
        import warnings

        warnings.warn(
            f"Accessing DatasetInfo with dict-style syntax info['{key}'] is deprecated. "
            f"Use attribute access info.{key} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(self, key)
        except AttributeError as err:
            raise KeyError(key) from err

    def __setitem__(self, key: str, value) -> None:
        import warnings

        warnings.warn(
            f"Setting DatasetInfo with dict-style syntax info['{key}'] = ... is deprecated. "
            f"Use attribute assignment info.{key} = ... instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not hasattr(self, key):
            raise KeyError(f"DatasetInfo has no field '{key}'")
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """检查某字段是否存在（字典风格接口）。"""
        return hasattr(self, key)

    def get(self, key: str, default=None):
        """获取属性值，找不到时回退到默认值（字典风格接口）。"""
        try:
            return getattr(self, key)
        except AttributeError:
            return default


def has_legacy_hub_download_metadata(root: Path) -> bool:
    """当 *root* 看起来像遗留的 Hub ``local_dir`` 镜像时返回 ``True``。

    ``snapshot_download(local_dir=...)`` 会将轻量元数据存储在
    ``<local_dir>/.cache/huggingface/download/`` 下。该目录的
    存在是一个可靠标志，表明数据集是用旧的、非版本安全的
    ``local_dir`` 模式下载的，应当改为通过快照缓存重新获取。
    """
    return (root / ".cache" / "huggingface" / "download").exists()


def update_chunk_file_indices(chunk_idx: int, file_idx: int, chunks_size: int) -> tuple[int, int]:
    if file_idx == chunks_size - 1:
        file_idx = 0
        chunk_idx += 1
    else:
        file_idx += 1
    return chunk_idx, file_idx


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    """将包含张量或 numpy 数组的字典序列化为 JSON 兼容形式。

    会将 torch.Tensor、np.ndarray 和 np.generic 类型转换为列表或
    Python 原生类型。

    Args:
        stats (dict): 可能包含不可序列化数值类型的字典。

    Returns:
        dict: 所有值都已转换为 JSON 可序列化类型的字典。

    Raises:
        NotImplementedError: 当某个值的类型不受支持时。
    """
    serialized_dict = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, (torch.Tensor | np.ndarray)):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, list) and isinstance(value[0], (int | float | list)):
            serialized_dict[key] = value
        elif isinstance(value, np.generic):
            serialized_dict[key] = value.item()
        elif isinstance(value, (int | float)):
            serialized_dict[key] = value
        else:
            raise NotImplementedError(f"The value '{value}' of type '{type(value)}' is not supported.")
    return unflatten_dict(serialized_dict)


def is_valid_version(version: str) -> bool:
    """检查一个字符串是否为有效的 PEP 440 版本号。

    Args:
        version (str): 待检查的版本字符串。

    Returns:
        bool: 版本字符串有效时返回 True，否则返回 False。
    """
    try:
        packaging.version.parse(version)
        return True
    except packaging.version.InvalidVersion:
        return False


def check_version_compatibility(
    repo_id: str,
    version_to_check: str | packaging.version.Version,
    current_version: str | packaging.version.Version,
    enforce_breaking_major: bool = True,
) -> None:
    """检查数据集与当前代码库之间的版本兼容性。

    Args:
        repo_id (str): 用于日志记录的仓库 ID。
        version_to_check (str | packaging.version.Version): 数据集的版本。
        current_version (str | packaging.version.Version): 代码库的当前版本。
        enforce_breaking_major (bool): 若为 True，主版本不匹配时抛出错误。

    Raises:
        BackwardCompatibilityError: 当数据集版本来自更新的、不兼容的
            代码库主版本时。
    """
    v_check = (
        packaging.version.parse(version_to_check)
        if not isinstance(version_to_check, packaging.version.Version)
        else version_to_check
    )
    v_current = (
        packaging.version.parse(current_version)
        if not isinstance(current_version, packaging.version.Version)
        else current_version
    )
    if v_check.major < v_current.major and enforce_breaking_major:
        raise BackwardCompatibilityError(repo_id, v_check)
    elif v_check.minor < v_current.minor:
        logging.warning(FUTURE_MESSAGE.format(repo_id=repo_id, version=v_check))


def get_repo_versions(repo_id: str, *, token: str | bool | None = None) -> list[packaging.version.Version]:
    """返回给定 Hub 仓库上可用的有效版本（分支和标签）。

    Args:
        repo_id (str): Hugging Face Hub 上的仓库 ID。
        token: 用于 Hub 请求的认证令牌。可传入字符串令牌；
            ``True`` 表示要求使用本地存储的令牌；``False``
            表示禁用认证；``None`` 表示使用 Hugging Face Hub 的默认行为。

    Returns:
        list[packaging.version.Version]: 找到的有效版本列表。
    """
    api = HfApi() if token is None else HfApi(token=token)
    repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
    repo_refs = [b.name for b in repo_refs.branches + repo_refs.tags]
    repo_versions = []
    for ref in repo_refs:
        with contextlib.suppress(packaging.version.InvalidVersion):
            repo_versions.append(packaging.version.parse(ref))

    return repo_versions


def get_safe_version(
    repo_id: str,
    version: str | packaging.version.Version,
    *,
    token: str | bool | None = None,
) -> str:
    """若仓库上存在指定版本则返回它，否则返回最新的兼容版本。

    如果找不到确切版本，则查找主版本号相同且次版本号
    小于或等于目标次版本号的最新版本。

    Args:
        repo_id (str): Hugging Face Hub 上的仓库 ID。
        version (str | packaging.version.Version): 目标版本。
        token: 转发给 Hub 版本查询的认证令牌。

    Returns:
        str: 用作 revision 的安全版本字符串（例如 "v1.2.3"）。

    Raises:
        RuntimeError: 当仓库没有任何版本标签时。
        BackwardCompatibilityError: 当只有更旧主版本的版本可用时。
        ForwardCompatibilityError: 当只有更新主版本的版本可用时。
    """
    target_version = (
        packaging.version.parse(version) if not isinstance(version, packaging.version.Version) else version
    )
    hub_versions = get_repo_versions(repo_id) if token is None else get_repo_versions(repo_id, token=token)

    if not hub_versions:
        raise RuntimeError(MISSING_VERSION_TAG_MESSAGE.format(repo_id=repo_id))

    if target_version in hub_versions:
        return f"v{target_version}"

    compatibles = [
        v for v in hub_versions if v.major == target_version.major and v.minor <= target_version.minor
    ]
    if compatibles:
        return_version = max(compatibles)
        if return_version < target_version:
            logging.warning(f"Revision {version} for {repo_id} not found, using version v{return_version}")
        return f"v{return_version}"

    lower_major = [v for v in hub_versions if v.major < target_version.major]
    if lower_major:
        raise BackwardCompatibilityError(repo_id, max(lower_major))

    upper_versions = [v for v in hub_versions if v > target_version]
    assert len(upper_versions) > 0
    raise ForwardCompatibilityError(repo_id, min(upper_versions))


def create_branch(repo_id: str, *, branch: str, repo_type: str | None = None) -> None:
    """在已有的 Hugging Face 仓库上创建分支。

    如果分支已存在，则先删除再创建。

    Args:
        repo_id (str): 仓库的 ID。
        branch (str): 要创建的分支名。
        repo_type (str | None): 仓库类型（例如 "dataset"）。
    """
    api = HfApi()

    branches = api.list_repo_refs(repo_id, repo_type=repo_type).branches
    refs = [branch.ref for branch in branches]
    ref = f"refs/heads/{branch}"
    if ref in refs:
        api.delete_branch(repo_id, repo_type=repo_type, branch=branch)

    api.create_branch(repo_id, repo_type=repo_type, branch=branch)


def create_lerobot_dataset_card(
    tags: list | None = None,
    dataset_info: DatasetInfo | None = None,
    **kwargs,
) -> DatasetCard:
    """为 LeRobot 数据集创建 `DatasetCard`。

    关键字参数用于替换卡片模板中的值。
    注意：如果指定 `license`，它必须是
    https://huggingface.co/docs/hub/repositories-licenses
    上有效的许可证标识符。

    Args:
        tags (list | None): 要添加到数据集卡片的标签列表。
        dataset_info (DatasetInfo | None): 数据集的 info 对象，
            将展示在卡片上。
        **kwargs: 用于填充卡片模板的额外关键字参数。

    Returns:
        DatasetCard: 生成的数据集卡片对象。
    """
    card_tags = ["LeRobot"]

    if tags:
        card_tags += tags
    if dataset_info:
        dataset_structure = "[meta/info.json](meta/info.json):\n"
        dataset_structure += f"```json\n{json.dumps(dataset_info.to_dict(), indent=4)}\n```\n"
        kwargs = {**kwargs, "dataset_structure": dataset_structure}
    card_data = DatasetCardData(
        license=kwargs.get("license"),
        tags=card_tags,
        task_categories=["robotics"],
        configs=[
            {
                "config_name": "default",
                "data_files": "data/*/*.parquet",
            }
        ],
    )

    card_template = (importlib.resources.files("lerobot.datasets") / "card_template.md").read_text()

    return DatasetCard.from_template(
        card_data=card_data,
        template_str=card_template,
        **kwargs,
    )


def is_float_in_list(target, float_list, threshold=1e-6):
    return any(abs(target - x) <= threshold for x in float_list)


def find_float_index(target, float_list, threshold=1e-6):
    for i, x in enumerate(float_list):
        if abs(target - x) <= threshold:
            return i
    return -1


def safe_shard(dataset: datasets.IterableDataset, index: int, num_shards: int) -> datasets.Dataset:
    """
    安全地对数据集进行分片。
    """
    shard_idx = min(dataset.num_shards, index + 1) - 1

    return dataset.shard(num_shards, index=shard_idx)
