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
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from datasets.table import embed_table_storage
from PIL import Image as PILImage
from torchvision import transforms

from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.utils import SuppressProgressBars, flatten_dict, unflatten_dict

from .language import LANGUAGE_COLUMNS
from .utils import (
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_EPISODES_PATH,
    DEFAULT_TASKS_PATH,
    EPISODES_DIR,
    INFO_PATH,
    STATS_PATH,
    DatasetInfo,
    serialize_dict,
)


def get_parquet_file_size_in_mb(parquet_path: str | Path) -> float:
    metadata = pq.read_metadata(parquet_path)
    total_uncompressed_size = 0
    for row_group in range(metadata.num_row_groups):
        rg_metadata = metadata.row_group(row_group)
        for column in range(rg_metadata.num_columns):
            col_metadata = rg_metadata.column(column)
            total_uncompressed_size += col_metadata.total_uncompressed_size
    return total_uncompressed_size / (1024**2)


def get_hf_dataset_size_in_mb(hf_ds: Dataset) -> int:
    return hf_ds.data.nbytes // (1024**2)


def load_nested_dataset(
    pq_dir: Path, features: datasets.Features | None = None, episodes: list[int] | None = None
) -> Dataset:
    """在给定目录 {pq_dir}/chunk-xxx/file-xxx.parquet 中查找 parquet 文件
    将 parquet 文件转换为缓存文件夹中的 pyarrow 内存映射，以高效使用 RAM
    拼接所有 pyarrow 引用并返回 HF Dataset 格式

    Args:
        pq_dir: 包含 parquet 文件的目录
        features: 可选的特征 schema，用于确保图像等复杂类型加载的一致性
        episodes: 可选的待过滤 episode 索引列表。使用 PyArrow 谓词下推以提高效率。
    """
    paths = sorted(pq_dir.glob("*/*.parquet"))
    if len(paths) == 0:
        raise FileNotFoundError(f"Provided directory does not contain any parquet file: {pq_dir}")

    with SuppressProgressBars():
        # 为了效率，我们使用 .from_parquet() 的内存映射加载
        filters = pa_ds.field("episode_index").isin(episodes) if episodes is not None else None
        return Dataset.from_parquet([str(path) for path in paths], filters=filters, features=features)


def get_parquet_num_frames(parquet_path: str | Path) -> int:
    metadata = pq.read_metadata(parquet_path)
    return metadata.num_rows


def get_file_size_in_mb(file_path: Path) -> float:
    """获取文件在磁盘上的大小（以兆字节为单位）。

    Args:
        file_path (Path): 文件路径。
    """
    file_size_bytes = file_path.stat().st_size
    return file_size_bytes / (1024**2)


def embed_images(dataset: datasets.Dataset) -> datasets.Dataset:
    """在保存到 Parquet 之前，将图像字节嵌入数据集表中。

    此函数通过将图像对象转换为可存储在 Arrow/Parquet 中的
    嵌入格式，为 Hugging Face 数据集的序列化做准备。

    Args:
        dataset (datasets.Dataset): 输入数据集，可能包含图像特征。

    Returns:
        datasets.Dataset: 图像已嵌入表存储的数据集。
    """
    # 在保存到 parquet 之前，将图像字节嵌入表中
    format = dataset.format
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(embed_table_storage, batched=False)
    dataset = dataset.with_format(**format)
    return dataset


def write_info(info: DatasetInfo, local_dir: Path) -> None:
    write_json(info.to_dict(), local_dir / INFO_PATH)


def load_info(local_dir: Path) -> DatasetInfo:
    """从标准文件路径加载数据集信息元数据。

    Args:
        local_dir (Path): 数据集的根目录。

    Returns:
        DatasetInfo: 带类型的数据集信息对象。
    """
    raw = load_json(local_dir / INFO_PATH)
    return DatasetInfo.from_dict(raw)


def write_stats(stats: dict, local_dir: Path) -> None:
    """将数据集统计量序列化并写入其标准文件路径。

    Args:
        stats (dict): 统计量字典（可包含张量/numpy 数组）。
        local_dir (Path): 数据集的根目录。
    """
    serialized_stats = serialize_dict(stats)
    write_json(serialized_stats, local_dir / STATS_PATH)


def cast_stats_to_numpy(stats: dict) -> dict[str, dict[str, np.ndarray]]:
    """递归地将统计量字典中的数值转换为 numpy 数组。

    Args:
        stats (dict): 统计量字典。

    Returns:
        dict: 值已转换为 numpy 数组的统计量字典。
    """
    stats = {key: np.atleast_1d(np.array(value)) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_stats(local_dir: Path) -> dict[str, dict[str, np.ndarray]] | None:
    """加载数据集统计量并将数值转换为 numpy 数组。

    如果统计量文件不存在，则返回 None。

    Args:
        local_dir (Path): 数据集的根目录。

    Returns:
        统计量字典；如果未找到文件则返回 None。
    """
    if not (local_dir / STATS_PATH).exists():
        return None
    stats = load_json(local_dir / STATS_PATH)
    return cast_stats_to_numpy(stats)


def write_tasks(tasks: pandas.DataFrame, local_dir: Path) -> None:
    path = local_dir / DEFAULT_TASKS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(path)


def load_tasks(local_dir: Path) -> pandas.DataFrame:
    tasks = pd.read_parquet(local_dir / DEFAULT_TASKS_PATH)
    tasks.index.name = "task"
    return tasks


def write_episodes(episodes: Dataset, local_dir: Path) -> None:
    """以 LeRobot v3.0 格式将 episode 元数据写入 parquet 文件。
    此函数将 episode 级别的元数据写入单个 parquet 文件。
    主要用于数据集转换（v2.1 → v3.0）和测试 fixture。

    Args:
        episodes: 包含 episode 元数据的 HuggingFace Dataset
        local_dir: 数据集存储的根目录
    """
    episode_size_mb = get_hf_dataset_size_in_mb(episodes)
    if episode_size_mb > DEFAULT_DATA_FILE_SIZE_IN_MB:
        raise NotImplementedError(
            f"Episodes dataset is too large ({episode_size_mb} MB) to write to a single file. "
            f"The current limit is {DEFAULT_DATA_FILE_SIZE_IN_MB} MB. "
            "This function only supports single-file episode metadata. "
        )

    fpath = local_dir / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(fpath)


def load_episodes(local_dir: Path) -> datasets.Dataset:
    episodes = load_nested_dataset(local_dir / EPISODES_DIR)
    # 选择包含对 episode 数据和视频引用的 episode 特征/列
    # （例如 tasks、dataset_from_index、dataset_to_index、data/chunk_index、data/file_index 等）
    # 这是为了加快对这些数据的访问，而不必加载 episode 统计量。
    episodes = episodes.select_columns([key for key in episodes.features if not key.startswith("stats/")])
    return episodes


def load_image_as_numpy(
    fpath: str | Path, dtype: np.dtype = np.float32, channel_first: bool = True
) -> np.ndarray:
    """从文件加载图像为 numpy 数组。

    Args:
        fpath (str | Path): 图像文件路径。
        dtype (np.dtype): 输出数组的期望数据类型。如果是浮点类型，
            像素会被缩放到 [0, 1]。仅用于 RGB 图像。
        channel_first (bool): 如果为 True，将图像转换为 (C, H, W) 格式。
            否则保持 (H, W, C) 格式。

    Returns:
        np.ndarray: 以 numpy 数组表示的图像。
    """
    is_depth = fpath.endswith(".tiff") or fpath.endswith(".tif")
    if is_depth:
        # 保留原生深度 dtype（uint16 -> "I;16"，float32 -> "F"）。
        img = PILImage.open(fpath)
        img_array = np.array(img)
    else:
        img = PILImage.open(fpath).convert("RGB")
        img_array = np.array(img, dtype=dtype)
        if np.issubdtype(dtype, np.floating):
            img_array /= 255.0
    if channel_first:  # (H, W, C) -> (C, H, W)
        img_array = img_array[np.newaxis, ...] if img_array.ndim == 2 else np.transpose(img_array, (2, 0, 1))
    return img_array


# 16 位无符号深度图的 PIL 模式。
UINT16_PIL_MODES = {"I;16", "I;16B", "I;16L"}


def pil_to_chw_tensor(img: PILImage.Image) -> torch.Tensor:
    """将 PIL 图像转换为通道在前的张量。

    ``uint16`` 深度图会以原生单位变为 ``float32 (1, H, W)``（``ToTensor``
    会使其溢出为 ``int16``）；所有其他模式使用标准的 ``ToTensor`` 路径。
    """
    if img.mode in UINT16_PIL_MODES:
        return torch.from_numpy(np.array(img, dtype=np.float32))[None, ...]
    return transforms.ToTensor()(img)


def hf_transform_to_torch(items_dict: dict[str, list[Any]]) -> dict[str, list[torch.Tensor | str]]:
    """将 Hugging Face 数据集的一个批次转换为 torch 张量。

    此转换函数将 Hugging Face 数据集格式（pyarrow）的条目
    转换为 torch 张量。RGB 图像从 PIL 对象（H, W, C, uint8）
    转换为 torch 图像表示（C, H, W, float32），范围在 [0, 1] 内。
    深度图以原生单位返回为 float32 (1, H, W)。其他
    类型会被转换为 torch.tensor。

    Args:
        items_dict (dict): 表示来自 Hugging Face 数据集的
            一个批次数据的字典。

    Returns:
        dict: 条目已转换为 torch 张量的批次。
    """
    for key in items_dict:
        if key in LANGUAGE_COLUMNS:
            continue
        first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            items_dict[key] = [pil_to_chw_tensor(img) for img in items_dict[key]]
        elif first_item is None or isinstance(first_item, dict):
            pass
        else:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict


def write_table_one_row_group_per_episode(table: pa.Table, path: Path) -> None:
    """写入 ``table``，每个 episode 对应一个 parquet 行组（按 episode 顺序）。

    使分片保持对随机访问友好（``read_row_group(i)`` 获取 episode i），
    与录制写入器保持一致。``table`` 必须携带连续的
    ``episode_index`` 列。
    """
    episode_index = table.column("episode_index").to_numpy(zero_copy_only=False)
    starts = np.concatenate(([0], np.nonzero(np.diff(episode_index))[0] + 1))
    writer = pq.ParquetWriter(str(path), table.schema, compression="snappy", use_dictionary=True)
    try:
        for start, stop in zip(starts, np.append(starts[1:], len(episode_index)), strict=True):
            writer.write_table(table.slice(start, stop - start))  # 一个 episode -> 一个行组
    finally:
        writer.close()


def to_parquet_with_hf_images(
    df: pandas.DataFrame, path: Path, features: datasets.Features | None = None
) -> None:
    """将带 HF 编码图像的 DataFrame 写入 parquet，每个 episode 对应一个行组。

    图像会先嵌入 arrow 表中（``ParquetWriter.write_table``
    不像 ``Dataset.to_parquet`` 那样嵌入外部图像文件）。
    ``features`` 在 parquet schema 中将图像列的类型设为 ``Image()``。
    """
    ds = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=features)
    ds = embed_images(ds)
    table = ds.with_format("arrow")[:]
    if "episode_index" in table.column_names:
        write_table_one_row_group_per_episode(table, path)
    else:
        # 没有 episode 边界可供行组对齐——保持单次写入。
        pq.write_table(table, str(path))


def to_parquet_one_row_group_per_episode(df: pandas.DataFrame, path: Path) -> None:
    """将（非图像的）DataFrame 写入 parquet，每个 episode 对应一个行组。"""
    table = pa.Table.from_pandas(df, preserve_index=False)
    if "episode_index" in table.column_names:
        write_table_one_row_group_per_episode(table, path)
    else:
        pq.write_table(table, str(path))


def item_to_torch(item: dict) -> dict:
    """在合适的情况下将字典中的所有条目转换为 PyTorch 张量。

    此函数用于将流式数据集中的条目转换为 PyTorch 张量。

    Args:
        item (dict): 来自数据集的条目字典。

    Returns:
        dict: 所有类张量条目已转换为 torch.Tensor 的字典。
    """
    skip_keys = {"task", *LANGUAGE_COLUMNS}
    for key, val in item.items():
        if key in skip_keys:
            continue
        if isinstance(val, PILImage.Image):
            item[key] = pil_to_chw_tensor(val)
        elif isinstance(val, (np.ndarray | list)):
            # 将 numpy 数组和列表转换为 torch 张量
            item[key] = torch.tensor(val)
    return item
