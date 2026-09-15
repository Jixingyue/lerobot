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
"""训练用于动作编码的 FAST 分词器。

此脚本：
1. 从 LeRobotDataset 加载动作分块（含 episode 采样）
2. 可选地应用相对变换（相对动作与绝对动作）
3. 提取指定的动作维度用于编码
4. 应用归一化（MEAN_STD、MIN_MAX、QUANTILES 或其他模式）
5. 在动作分块上训练 FAST 分词器（对 DCT 系数做 BPE）
6. 将分词器保存到输出目录
7. 可选地将分词器推送到 Hugging Face Hub
8. 报告压缩统计信息

示例：

```shell
lerobot-train-tokenizer \
    --repo_id=user/dataset_name \
    --action_horizon=10 \
    --max_episodes=100 \
    --sample_fraction=0.1 \
    --encoded_dims="0:6" \
    --relative_dims="0,1,2,3,4,5" \
    --use_relative_transform=true \
    --state_key="observation.state" \
    --normalization_mode="QUANTILES" \
    --vocab_size=1024 \
    --scale=10.0 \
    --output_dir="./fast_tokenizer_dataset_name" \
    --push_to_hub=true \
    --hub_repo_id="user/fast_tokenizer_dataset_name" \
    --hub_private=false
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from huggingface_hub import HfApi

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None

from lerobot.configs import NormalizationMode, parser
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class TokenizerTrainingConfig:
    """训练 FAST 分词器的配置。"""

    # LeRobot 数据集仓库 ID
    repo_id: str
    # 数据集根目录（默认：~/.cache/huggingface/lerobot）
    root: str | None = None
    # 每个分块中未来动作的数量
    action_horizon: int = 10
    # 最多使用多少个 episode（None = 数据集中的所有 episode）
    max_episodes: int | None = None
    # 每个 episode 中采样分块的比例
    sample_fraction: float = 0.1
    # 以逗号分隔的待编码维度范围（例如 "0:6,7:23"）
    encoded_dims: str = "0:6,7:23"
    # 以逗号分隔的、用于相对变换的维度索引（例如 "0,1,2,3,4,5"）
    relative_dims: str | None = None
    # 是否应用相对变换（相对动作还是绝对动作）
    use_relative_transform: bool = False
    # 状态观测对应的数据集键（默认："observation.state"）
    state_key: str = OBS_STATE
    # 归一化模式（MEAN_STD、MIN_MAX、QUANTILES、QUANTILE10、IDENTITY）
    normalization_mode: str = "QUANTILES"
    # FAST 词表大小（BPE 词表大小）
    vocab_size: int = 1024
    # DCT 缩放系数（默认：10.0）
    scale: float = 10.0
    # 分词器保存目录（默认：./fast_tokenizer_{repo_id}）
    output_dir: str | None = None
    # 是否将分词器推送到 Hugging Face Hub
    push_to_hub: bool = False
    # Hub 仓库 ID（例如 "username/tokenizer-name"）。为 None 时使用 output_dir 的名称
    hub_repo_id: str | None = None
    # 是否在 Hub 上创建私有仓库
    hub_private: bool = False


def apply_relative_transform(
    state: np.ndarray, actions: np.ndarray, relative_dims: list[int] | None
) -> np.ndarray:
    """对指定维度应用相对变换。

    参数:
        state: 当前状态 [D]
        actions: 未来动作 [D]
        relative_dims: 需要应用相对变换的维度索引列表

    返回:
        变换后的动作 [D]
    """
    if relative_dims is None or len(relative_dims) == 0:
        return actions

    relative_actions = actions.copy()
    for dim in relative_dims:
        relative_actions[dim] = actions[dim] - state[dim]

    return relative_actions


def apply_normalization(
    data: np.ndarray,
    stats: dict[str, np.ndarray],
    mode: NormalizationMode,
    eps: float = 1e-8,
) -> np.ndarray:
    """根据指定模式对数据应用归一化。

    参数:
        data: 待归一化的数据 [N, H, D] 或 [D]
        stats: 统计信息字典（mean、std、min、max、q01、q99、q10、q90）
        mode: 要应用的归一化模式
        eps: 用于数值稳定性的小 epsilon

    返回:
        与输入形状相同的归一化后数据
    """
    if mode == NormalizationMode.IDENTITY:
        return data

    if mode == NormalizationMode.MEAN_STD:
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None or std is None:
            raise ValueError("MEAN_STD mode requires 'mean' and 'std' in stats")
        return (data - mean) / np.maximum(std, eps)

    if mode == NormalizationMode.MIN_MAX:
        min_val = stats.get("min")
        max_val = stats.get("max")
        if min_val is None or max_val is None:
            raise ValueError("MIN_MAX mode requires 'min' and 'max' in stats")
        denom = np.maximum(max_val - min_val, eps)
        return 2.0 * (data - min_val) / denom - 1.0

    if mode == NormalizationMode.QUANTILES:
        q01 = stats.get("q01")
        q99 = stats.get("q99")
        if q01 is None or q99 is None:
            raise ValueError("QUANTILES mode requires 'q01' and 'q99' in stats")
        denom = np.maximum(q99 - q01, eps)
        # 先裁剪到分位数范围，再归一化到 [-1, 1]
        clipped = np.clip(data, q01, q99)
        return 2.0 * (clipped - q01) / denom - 1.0

    if mode == NormalizationMode.QUANTILE10:
        q10 = stats.get("q10")
        q90 = stats.get("q90")
        if q10 is None or q90 is None:
            raise ValueError("QUANTILE10 mode requires 'q10' and 'q90' in stats")
        denom = np.maximum(q90 - q10, eps)
        # 先裁剪到分位数范围，再归一化到 [-1, 1]
        clipped = np.clip(data, q10, q90)
        return 2.0 * (clipped - q10) / denom - 1.0

    raise ValueError(f"Unsupported normalization mode: {mode}")


def process_episode(args):
    """处理单个 episode 并返回动作分块。"""
    dataset, ep_idx, action_horizon, relative_dims, sample_fraction, state_key, use_relative_transform = args

    try:
        # 获取 episode 信息
        ep_info = dataset.meta.episodes[ep_idx]
        from_idx = ep_info["dataset_from_index"]
        to_idx = ep_info["dataset_to_index"]
        ep_length = to_idx - from_idx

        if ep_length < action_horizon:
            return None

        # 加载 episode 中的所有帧
        # 如果数据集带有 episode 过滤，则需要使用索引映射
        states = []
        actions = []

        for abs_idx in range(from_idx, to_idx):
            # 如有必要，将绝对索引映射为相对索引
            if dataset.reader._absolute_to_relative_idx is not None:
                if abs_idx not in dataset.reader._absolute_to_relative_idx:
                    # 该 episode 的帧不在过滤后的数据集中
                    return None
                rel_idx = dataset.reader._absolute_to_relative_idx[abs_idx]
            else:
                rel_idx = abs_idx

            frame = dataset.get_raw_item(rel_idx)

            # 获取状态（可能来自 observation.state 或其他状态键）
            if state_key in frame:
                state = (
                    frame[state_key].numpy()
                    if torch.is_tensor(frame[state_key])
                    else np.array(frame[state_key])
                )
            else:
                # 如果没有状态键，则使用零（不做相对变换）
                state = np.zeros_like(
                    frame[ACTION].numpy() if torch.is_tensor(frame[ACTION]) else np.array(frame[ACTION])
                )

            action = frame[ACTION].numpy() if torch.is_tensor(frame[ACTION]) else np.array(frame[ACTION])

            states.append(state)
            actions.append(action)

        states = np.array(states)
        actions = np.array(actions)

        # 创建动作分块（滑动窗口）
        # 一个分块中的所有动作都相对于该分块中的第一个状态
        action_chunks = []

        for i in range(len(states) - action_horizon + 1):
            current_state = states[i]  # 分块中的第一个状态
            future_absolute_actions = actions[i : i + action_horizon]

            if use_relative_transform:
                # 相对动作
                relative_chunk = np.zeros_like(future_absolute_actions)
                for t in range(action_horizon):
                    relative_chunk[t] = apply_relative_transform(
                        current_state,
                        future_absolute_actions[t],
                        relative_dims,
                    )
                action_chunks.append(relative_chunk)
            else:
                # 绝对动作（不做相对变换）
                action_chunks.append(future_absolute_actions)

        if len(action_chunks) == 0:
            return None

        action_chunks = np.array(action_chunks)

        # 对分块进行采样
        if sample_fraction < 1.0:
            n_chunks = len(action_chunks)
            n_samples = max(1, int(n_chunks * sample_fraction))
            episode_seed = hash(ep_idx) % (2**31)
            rng = np.random.RandomState(episode_seed)
            indices = rng.choice(n_chunks, size=n_samples, replace=False)
            action_chunks = action_chunks[indices]

        return action_chunks

    except Exception:
        logger.exception("Error processing episode %s", ep_idx)
        return None


def train_fast_tokenizer(
    action_chunks: np.ndarray,
    vocab_size: int = 1024,
    scale: float = 10.0,
) -> AutoProcessor:
    """
    在动作分块上训练 FAST 分词器（对 DCT 系数做 BPE）。

    使用 .fit() 方法在给定数据上训练一个新的分词器。

    参数:
        action_chunks: 动作分块数组 [N, H, D]，其中 N=分块数，H=horizon，D=action_dim
        vocab_size: BPE 词表大小
        scale: 用于量化的 DCT 缩放系数

    返回:
        训练好的 FAST 分词器
    """
    logger.info(f"Training FAST tokenizer on {len(action_chunks)} action chunks...")
    logger.info(f"Action chunk shape: {action_chunks.shape}")
    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"DCT scale: {scale}")

    # 下载分词器源代码（而非预训练权重）
    # 我们将在自己的数据上训练一个新的分词器
    base_tokenizer = AutoProcessor.from_pretrained("lerobot/fast-action-tokenizer", trust_remote_code=True)

    # 将 action_chunks 数组转换为数组列表（.fit() 所要求的格式）
    action_data_list = [action_chunks[i] for i in range(len(action_chunks))]

    # 使用 .fit() 在我们的动作数据上训练新分词器
    # 这会在 DCT 系数上训练 BPE 分词器
    logger.info("Training new tokenizer (this may take a few minutes)...")
    tokenizer = base_tokenizer.fit(
        action_data_list,
        scale=scale,
        vocab_size=vocab_size,
        time_horizon=action_chunks.shape[1],  # action_horizon
        action_dim=action_chunks.shape[2],  # 编码维度数
    )
    logger.info("✓ Tokenizer training complete!")

    # 验证它可以正常工作
    sample_chunk = action_chunks[0]
    encoded = tokenizer(sample_chunk[None])[0]
    if isinstance(encoded, list):
        encoded = np.array(encoded)
    logger.info(f"Sample encoding: {len(encoded)} tokens for chunk shape {sample_chunk.shape}")

    return tokenizer


def compute_compression_stats(tokenizer, action_chunks: np.ndarray):
    """计算压缩统计信息。"""
    logger.info("\nComputing compression statistics...")

    # 为统计信息采样（为提高速度最多使用 1000 个分块）
    sample_size = min(1000, len(action_chunks))
    sample_indices = np.random.RandomState(42).choice(len(action_chunks), size=sample_size, replace=False)
    sample_chunks = action_chunks[sample_indices]

    token_lengths = []
    for chunk in sample_chunks:
        encoded = tokenizer(chunk[None])[0]
        if isinstance(encoded, list):
            token_lengths.append(len(encoded))
        else:
            token_lengths.append(encoded.shape[0] if hasattr(encoded, "shape") else len(encoded))

    token_lengths = np.array(token_lengths)

    # 压缩比：(H * D) / 平均 token 数
    input_size = action_chunks.shape[1] * action_chunks.shape[2]
    avg_tokens = np.mean(token_lengths)
    compression_ratio = input_size / avg_tokens

    stats = {
        "compression_ratio": float(compression_ratio),
        "mean_token_length": float(np.mean(token_lengths)),
        "p99_token_length": float(np.percentile(token_lengths, 99)),
        "min_token_length": float(np.min(token_lengths)),
        "max_token_length": float(np.max(token_lengths)),
    }

    logger.info("Compression Statistics:")
    logger.info(f"  Average compression ratio: {stats['compression_ratio']:.2f}x")
    logger.info(f"  Mean token length: {stats['mean_token_length']:.1f}")
    logger.info(f"  P99 token length: {stats['p99_token_length']:.0f}")
    logger.info(f"  Min token length: {stats['min_token_length']:.0f}")
    logger.info(f"  Max token length: {stats['max_token_length']:.0f}")

    return stats


@parser.wrap()
def train_tokenizer(cfg: TokenizerTrainingConfig):
    """
    训练用于动作编码的 FAST 分词器。

    参数:
        cfg: 包含所有配置参数的 TokenizerTrainingConfig 数据类
    """
    # 加载数据集
    logger.info(f"Loading dataset: {cfg.repo_id}")
    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root)
    logger.info(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    # 解析归一化模式
    try:
        norm_mode = NormalizationMode(cfg.normalization_mode)
    except ValueError as err:
        raise ValueError(
            f"Invalid normalization_mode: {cfg.normalization_mode}. "
            f"Must be one of: {', '.join([m.value for m in NormalizationMode])}"
        ) from err
    logger.info(f"Normalization mode: {norm_mode.value}")

    # 解析编码维度
    encoded_dim_ranges = []
    for range_str in cfg.encoded_dims.split(","):
        start, end = map(int, range_str.strip().split(":"))
        encoded_dim_ranges.append((start, end))

    total_encoded_dims = sum(end - start for start, end in encoded_dim_ranges)
    logger.info(f"Encoding {total_encoded_dims} dimensions: {cfg.encoded_dims}")

    # 解析相对变换维度
    relative_dim_list = None
    if cfg.relative_dims is not None and cfg.relative_dims.strip():
        relative_dim_list = [int(d.strip()) for d in cfg.relative_dims.split(",")]
        logger.info(f"Relative dimensions: {relative_dim_list}")
    else:
        logger.info("No relative dimensions specified")

    logger.info(f"Use relative transform: {cfg.use_relative_transform}")
    if cfg.use_relative_transform and (relative_dim_list is None or len(relative_dim_list) == 0):
        logger.warning(
            "Warning: use_relative_transform=True but no relative_dims specified. "
            "No relative transform will be applied."
        )

    logger.info(f"Action horizon: {cfg.action_horizon}")
    logger.info(f"State key: {cfg.state_key}")

    # 确定要处理的 episode 数量
    num_episodes = dataset.num_episodes
    if cfg.max_episodes is not None:
        num_episodes = min(cfg.max_episodes, num_episodes)

    logger.info(f"Processing {num_episodes} episodes...")

    # 顺序处理各个 episode（以避免数据集的 pickle 问题）
    all_chunks = []
    for ep_idx in range(num_episodes):
        if ep_idx % 10 == 0:
            logger.info(f"  Processing episode {ep_idx}/{num_episodes}...")

        chunks = process_episode(
            (
                dataset,
                ep_idx,
                cfg.action_horizon,
                relative_dim_list,
                cfg.sample_fraction,
                cfg.state_key,
                cfg.use_relative_transform,
            )
        )
        if chunks is not None:
            all_chunks.append(chunks)

    # 拼接所有分块
    all_chunks = np.concatenate(all_chunks, axis=0)
    logger.info(f"Collected {len(all_chunks)} action chunks")

    # 首先只提取待编码维度（在归一化之前）
    encoded_chunks = []
    for start, end in encoded_dim_ranges:
        encoded_chunks.append(all_chunks[:, :, start:end])
    encoded_chunks = np.concatenate(encoded_chunks, axis=-1)  # [N, H, D_encoded]
    logger.info(f"Extracted {encoded_chunks.shape[-1]} encoded dimensions")

    # 对编码维度应用归一化
    logger.info("\nBefore normalization - overall stats:")
    logger.info(f"  Min: {np.min(encoded_chunks):.4f}, Max: {np.max(encoded_chunks):.4f}")
    logger.info(f"  Mean: {np.mean(encoded_chunks):.4f}, Std: {np.std(encoded_chunks):.4f}")

    # 从数据集中获取归一化统计信息
    norm_stats = dataset.meta.stats
    if norm_stats is not None and ACTION in norm_stats:
        action_stats = norm_stats[ACTION]

        # 构建编码维度的索引
        encoded_dim_indices = []
        for start, end in encoded_dim_ranges:
            encoded_dim_indices.extend(range(start, end))
        encoded_dim_indices = np.array(encoded_dim_indices)

        # 仅提取编码维度对应的统计信息
        encoded_stats = {}
        for stat_name, stat_values in action_stats.items():
            if isinstance(stat_values, (list, np.ndarray)):
                stat_array = np.array(stat_values)
                if len(stat_array) > max(encoded_dim_indices):
                    encoded_stats[stat_name] = stat_array[encoded_dim_indices]

        if encoded_stats:
            logger.info(f"\nNormalization stats for encoded dimensions (mode: {norm_mode.value}):")
            for stat_name, stat_values in encoded_stats.items():
                logger.info(
                    f"  {stat_name}: shape={stat_values.shape}, "
                    f"range=[{np.min(stat_values):.4f}, {np.max(stat_values):.4f}]"
                )

            # 根据所选模式应用归一化
            try:
                encoded_chunks = apply_normalization(encoded_chunks, encoded_stats, norm_mode, eps=1e-8)
                logger.info(f"\nApplied {norm_mode.value} normalization")
            except ValueError as e:
                logger.warning(f"Warning: {e}. Using raw actions without normalization.")

            logger.info("\nAfter normalization - overall stats:")
            logger.info(f"  Min: {np.min(encoded_chunks):.4f}, Max: {np.max(encoded_chunks):.4f}")
            logger.info(f"  Mean: {np.mean(encoded_chunks):.4f}, Std: {np.std(encoded_chunks):.4f}")

            logger.info("\nPer-dimension stats (after normalization):")
            for d in range(encoded_chunks.shape[-1]):
                dim_data = encoded_chunks[:, :, d]
                logger.info(
                    f"  Dim {d}: min={np.min(dim_data):7.4f}, max={np.max(dim_data):7.4f}, "
                    f"mean={np.mean(dim_data):7.4f}, std={np.std(dim_data):7.4f}"
                )
        else:
            logger.warning("Warning: Could not extract stats for encoded dimensions, using raw actions")
    else:
        logger.warning("Warning: No normalization stats found in dataset, using raw actions")

    logger.info(f"Encoded chunks shape: {encoded_chunks.shape}")

    # 训练 FAST 分词器
    tokenizer = train_fast_tokenizer(
        encoded_chunks,
        vocab_size=cfg.vocab_size,
        scale=cfg.scale,
    )

    # 计算压缩统计信息
    compression_stats = compute_compression_stats(tokenizer, encoded_chunks)

    # 保存分词器
    output_dir = cfg.output_dir
    if output_dir is None:
        output_dir = f"fast_tokenizer_{cfg.repo_id.replace('/', '_')}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(output_path)

    # 保存元数据
    metadata = {
        "repo_id": cfg.repo_id,
        "vocab_size": cfg.vocab_size,
        "scale": cfg.scale,
        "encoded_dims": cfg.encoded_dims,
        "encoded_dim_ranges": encoded_dim_ranges,
        "total_encoded_dims": total_encoded_dims,
        "relative_dims": cfg.relative_dims,
        "relative_dim_list": relative_dim_list,
        "use_relative_transform": cfg.use_relative_transform,
        "state_key": cfg.state_key,
        "normalization_mode": norm_mode.value,
        "action_horizon": cfg.action_horizon,
        "num_training_chunks": len(encoded_chunks),
        "compression_stats": compression_stats,
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nSaved FAST tokenizer to {output_path}")
    logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")

    # 如果有要求，则推送到 Hugging Face Hub
    if cfg.push_to_hub:
        # 确定 hub 仓库 ID
        hub_repo_id = cfg.hub_repo_id
        if hub_repo_id is None:
            hub_repo_id = output_path.name
            logger.info(f"\nNo hub_repo_id provided, using: {hub_repo_id}")

        logger.info(f"\nPushing tokenizer to Hugging Face Hub: {hub_repo_id}")
        logger.info(f"   Private: {cfg.hub_private}")

        try:
            # 使用分词器自身的 push_to_hub 方法
            tokenizer.push_to_hub(
                repo_id=hub_repo_id,
                private=cfg.hub_private,
                commit_message=f"Upload FAST tokenizer trained on {cfg.repo_id}",
            )

            # 另外单独上传 metadata.json 文件
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(output_path / "metadata.json"),
                path_in_repo="metadata.json",
                repo_id=hub_repo_id,
                repo_type="model",
                commit_message="Upload tokenizer metadata",
            )

            logger.info(f"Successfully pushed tokenizer to: https://huggingface.co/{hub_repo_id}")
        except Exception as e:
            logger.error(f"Error pushing to hub: {e}")
            logger.error("   Make sure you're logged in with `huggingface-cli login`")


def main():
    """CLI 入口点，负责解析参数并运行分词器训练。"""
    init_logging()
    train_tokenizer()


if __name__ == "__main__":
    main()
