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

"""
一个通用脚本，用于将带有内置归一化层的 LeRobot 策略迁移到新的
基于流水线的处理器系统。

本脚本执行以下步骤：
1.  从本地路径或 Hugging Face Hub 加载一个预训练策略模型及其配置。
2.  扫描模型的状态字典，为所有特征提取归一化统计量（例如 mean、
    std、min、max）。
3.  创建两个新的处理器流水线：
    - 一个预处理器，对输入（观察）和输出（动作）进行归一化。
    - 一个后处理器，在推理时对输出（动作）进行反归一化。
4.  从模型的状态字典中移除原有的归一化层，
    得到一个“干净”的模型。
5.  将新的干净模型、预处理器、后处理器以及生成的
    模型卡保存到一个新目录。
6.  可选地将所有新产物推送到 Hugging Face Hub。

Usage:
    python src/lerobot/processor/migrate_policy_normalization.py \
        --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
        --push-to-hub \
        --branch main

Note: 本表现在使用 `lerobot.policies.factory` 中现代的 `make_pre_post_processors` 和
`make_policy_config` 工厂函数来创建处理器和配置，
确保与当前代码库的一致性。

该脚本从旧模型的 state_dict 中提取归一化统计量，使用工厂函数创建干净的
处理器流水线，并保存一个与新 PolicyProcessorPipeline 架构兼容的
迁移后模型。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file as load_safetensors

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.utils.constants import ACTION


def extract_normalization_stats(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """
    扫描模型的 state_dict，查找并提取归一化统计量。

    本函数根据一组预定义的模式识别与归一化层对应的键（例如
    用于 mean、std、min、max 的键），并将它们组织到一个嵌套字典中。

    Args:
        state_dict: 预训练策略模型的状态字典。

    Returns:
        一个嵌套字典，外层键为特征名（例如
        'observation.state'），内层键为统计量类型（'mean'、'std'），
        映射到其对应的张量值。
    """
    stats = {}

    # 定义需要匹配的模式及其要移除的前缀
    normalization_patterns = [
        "normalize_inputs.buffer_",
        "unnormalize_outputs.buffer_",
        "normalize_targets.buffer_",
        "normalize.",  # 必须位于 normalize_* 模式之后
        "unnormalize.",  # 必须位于 unnormalize_* 模式之后
        "input_normalizer.",
        "output_normalizer.",
        "normalalize_inputs.",
        "unnormalize_outputs.",
        "normalize_targets.",
        "unnormalize_targets.",
    ]

    # 处理 state_dict 中的每个键
    for key, tensor in state_dict.items():
        # 尝试每个模式
        for pattern in normalization_patterns:
            if key.startswith(pattern):
                # 提取模式之后的剩余部分
                remaining = key[len(pattern) :]
                parts = remaining.split(".")

                # 至少需要特征名和统计量类型
                if len(parts) >= 2:
                    # 最后一部分是统计量类型（mean、std、min、max 等）
                    stat_type = parts[-1]
                    # 其余部分都是特征名
                    feature_name = ".".join(parts[:-1]).replace("_", ".")

                    # 添加到 stats
                    if feature_name not in stats:
                        stats[feature_name] = {}
                    stats[feature_name][stat_type] = tensor.clone()

                # 只处理第一个匹配的模式
                break

    return stats


def detect_features_and_norm_modes(
    config: dict[str, Any], stats: dict[str, dict[str, torch.Tensor]]
) -> tuple[dict[str, PolicyFeature], dict[FeatureType, NormalizationMode]]:
    """
    从模型配置和统计量推断策略特征和归一化模式。

    本函数首先尝试直接从策略的配置文件中查找特征定义和归一化
    映射。如果这些信息不存在，则从提取的归一化统计量中推断，
    使用张量形状来确定特征形状，并通过特定统计量键的存在
    （例如 'mean'/'std' 还是 'min'/'max'）来确定归一化模式。
    如果无法推断，则应用合理的默认值。

    Args:
        config: 来自 `config.json` 的策略配置字典。
        stats: 从模型 state_dict 中提取的归一化统计量。

    Returns:
        一个元组，包含：
        - 一个将特征名映射到 `PolicyFeature` 对象的字典。
        - 一个将 `FeatureType` 枚举映射到 `NormalizationMode` 枚举的字典。
    """
    features = {}
    norm_modes = {}

    # 首先，检查配置中是否有 normalization_mapping
    if "normalization_mapping" in config:
        print(f"Found normalization_mapping in config: {config['normalization_mapping']}")
        # 从 config 中提取归一化模式
        for feature_type_str, mode_str in config["normalization_mapping"].items():
            # 将字符串转换为 FeatureType 枚举
            try:
                if feature_type_str == "VISUAL":
                    feature_type = FeatureType.VISUAL
                elif feature_type_str == "STATE":
                    feature_type = FeatureType.STATE
                elif feature_type_str == "ACTION":
                    feature_type = FeatureType.ACTION
                else:
                    print(f"Warning: Unknown feature type '{feature_type_str}', skipping")
                    continue
            except (AttributeError, ValueError):
                print(f"Warning: Could not parse feature type '{feature_type_str}', skipping")
                continue

            # 将字符串转换为 NormalizationMode 枚举
            try:
                if mode_str == "MEAN_STD":
                    mode = NormalizationMode.MEAN_STD
                elif mode_str == "MIN_MAX":
                    mode = NormalizationMode.MIN_MAX
                elif mode_str == "IDENTITY":
                    mode = NormalizationMode.IDENTITY
                else:
                    print(
                        f"Warning: Unknown normalization mode '{mode_str}' for feature type '{feature_type_str}'"
                    )
                    continue
            except (AttributeError, ValueError):
                print(f"Warning: Could not parse normalization mode '{mode_str}', skipping")
                continue

            norm_modes[feature_type] = mode

    # 尝试从 config 中提取
    if "features" in config:
        for key, feature_config in config["features"].items():
            shape = feature_config.get("shape", feature_config.get("dim"))
            shape = (shape,) if isinstance(shape, int) else tuple(shape)

            # 确定特征类型
            if "image" in key or "visual" in key:
                feature_type = FeatureType.VISUAL
            elif "state" in key:
                feature_type = FeatureType.STATE
            elif ACTION in key:
                feature_type = FeatureType.ACTION
            else:
                feature_type = FeatureType.STATE  #  默认
            features[key] = PolicyFeature(feature_type, shape)

    # 如果 config 中没有特征，则从 stats 推断
    if not features:
        for key, stat_dict in stats.items():
            # 从任意一个统计量张量获取形状
            tensor = next(iter(stat_dict.values()))
            shape = tuple(tensor.shape)

            # 根据键确定特征类型
            if "image" in key or "visual" in key or "pixels" in key:
                feature_type = FeatureType.VISUAL
            elif "state" in key or "joint" in key or "position" in key:
                feature_type = FeatureType.STATE
            elif ACTION in key:
                feature_type = FeatureType.ACTION
            else:
                feature_type = FeatureType.STATE

            features[key] = PolicyFeature(feature_type, shape)

    # 如果归一化模式不在 config 中，则根据可用的 stats 确定
    if not norm_modes:
        for key, stat_dict in stats.items():
            if key in features:
                if "mean" in stat_dict and "std" in stat_dict:
                    feature_type = features[key].type
                    if feature_type not in norm_modes:
                        norm_modes[feature_type] = NormalizationMode.MEAN_STD
                elif "min" in stat_dict and "max" in stat_dict:
                    feature_type = features[key].type
                    if feature_type not in norm_modes:
                        norm_modes[feature_type] = NormalizationMode.MIN_MAX

    # 如果未检测到，则使用默认的归一化模式
    if FeatureType.VISUAL not in norm_modes:
        norm_modes[FeatureType.VISUAL] = NormalizationMode.MEAN_STD
    if FeatureType.STATE not in norm_modes:
        norm_modes[FeatureType.STATE] = NormalizationMode.MIN_MAX
    if FeatureType.ACTION not in norm_modes:
        norm_modes[FeatureType.ACTION] = NormalizationMode.MEAN_STD

    return features, norm_modes


def remove_normalization_layers(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    创建一个移除了所有与归一化相关层的新 state_dict。

    本函数过滤原始状态字典，排除任何与一组预定义的、
    与归一化模块相关的模式的键。

    Args:
        state_dict: 原始的模型状态字典。

    Returns:
        一个只包含核心模型权重、不含任何
        归一化参数的新状态字典。
    """
    new_state_dict = {}

    # 需要移除的模式
    remove_patterns = [
        "normalize_inputs.",
        "unnormalize_outputs.",
        "normalize_targets.",  # 为 target 归一化新增的模式
        "normalize.",
        "unnormalize.",
        "input_normalizer.",
        "output_normalizer.",
        "normalizer.",
    ]

    for key, tensor in state_dict.items():
        should_remove = any(pattern in key for pattern in remove_patterns)
        if not should_remove:
            new_state_dict[key] = tensor

    return new_state_dict


def clean_state_dict(
    state_dict: dict[str, torch.Tensor], remove_str: str = "._orig_mod"
) -> dict[str, torch.Tensor]:
    """
    从 state dict 的所有键中移除一个子串（例如 '._orig_mod'）。

    Args:
        state_dict (dict): 原始的 state dict。
        remove_str (str): 要从键中移除的子串。

    Returns:
        dict: 一个键已清洗的新 state dict。
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace(remove_str, "")
        new_state_dict[new_k] = v
    return new_state_dict


def load_state_dict_with_missing_key_handling(
    policy: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    policy_type: str,
    known_missing_keys_whitelist: dict[str, list[str]],
) -> list[str]:
    """
    以优雅的方式处理缺失键，将 state dict 加载到策略中。

    本函数使用 strict=False 加载 state dict，过滤掉白名单中
    的缺失键，并对发现的任何问题提供详细的报告。

    Args:
        policy: 要加载 state dict 的策略模型。
        state_dict: 要加载的已清洗状态字典。
        policy_type: 策略的类型（用于白名单查找）。
        known_missing_keys_whitelist: 将策略类型映射到已知可接受
                                     的缺失键列表的字典。

    Returns:
        不在白名单中的问题缺失键列表。
    """
    # 使用 strict=False 加载已清洗的 state dict，以捕获缺失/多余的键
    load_result = policy.load_state_dict(state_dict, strict=False)

    # 检查缺失的键
    missing_keys = load_result.missing_keys
    unexpected_keys = load_result.unexpected_keys

    # 过滤掉白名单中的缺失键
    policy_type_lower = policy_type.lower()
    whitelisted_keys = known_missing_keys_whitelist.get(policy_type_lower, [])
    problematic_missing_keys = [key for key in missing_keys if key not in whitelisted_keys]

    if missing_keys:
        if problematic_missing_keys:
            print(f"WARNING: Found {len(problematic_missing_keys)} unexpected missing keys:")
            for key in problematic_missing_keys:
                print(f"   - {key}")

        if len(missing_keys) > len(problematic_missing_keys):
            whitelisted_missing = [key for key in missing_keys if key in whitelisted_keys]
            print(f"INFO: Found {len(whitelisted_missing)} expected missing keys (whitelisted):")
            for key in whitelisted_missing:
                print(f"   - {key}")

    if unexpected_keys:
        print(f"WARNING: Found {len(unexpected_keys)} unexpected keys:")
        for key in unexpected_keys:
            print(f"   - {key}")

    if not missing_keys and not unexpected_keys:
        print("Successfully loaded cleaned state dict into policy model (all keys matched)")
    else:
        print("State dict loaded with some missing/unexpected keys (see details above)")

    return problematic_missing_keys


def convert_features_to_policy_features(features_dict: dict[str, dict]) -> dict[str, PolicyFeature]:
    """
    将特征字典从旧的配置格式转换为新的 `PolicyFeature` 格式。

    Args:
        features_dict: 旧格式的特征字典，其值为
                       简单的字典（例如 `{"shape": [7]}`）。

    Returns:
        一个将特征名映射到 `PolicyFeature` dataclass 对象的字典。
    """
    converted_features = {}

    for key, feature_dict in features_dict.items():
        # 根据键确定特征类型
        if "image" in key or "visual" in key:
            feature_type = FeatureType.VISUAL
        elif "state" in key:
            feature_type = FeatureType.STATE
        elif ACTION in key:
            feature_type = FeatureType.ACTION
        else:
            feature_type = FeatureType.STATE

        # 从特征字典获取形状
        shape = feature_dict.get("shape", feature_dict.get("dim"))
        shape = (shape,) if isinstance(shape, int) else tuple(shape) if shape is not None else ()

        converted_features[key] = PolicyFeature(feature_type, shape)

    return converted_features


def display_migration_summary_with_warnings(problematic_missing_keys: list[str]) -> None:
    """
    显示最终的迁移总结，并对问题的缺失键发出警告。

    Args:
        problematic_missing_keys: 不在白名单中的缺失键列表。
    """
    if not problematic_missing_keys:
        return

    print("\n" + "=" * 60)
    print("IMPORTANT: MIGRATION COMPLETED WITH WARNINGS")
    print("=" * 60)
    print(
        f"The migration was successful, but {len(problematic_missing_keys)} unexpected missing keys were found:"
    )
    print()
    for key in problematic_missing_keys:
        print(f"   - {key}")
    print()
    print("These missing keys may indicate:")
    print("  • The model architecture has changed")
    print("  • Some components were not properly saved in the original model")
    print("  • The migration script needs to be updated for this policy type")
    print()
    print("What to do next:")
    print("  1. Test your migrated model carefully to ensure it works as expected")
    print("  2. If you encounter issues, please open an issue at:")
    print("     https://github.com/huggingface/lerobot/issues")
    print("  3. Include this migration log and the missing keys listed above")
    print()
    print("If the model works correctly despite these warnings, the missing keys")
    print("might be expected for your policy type and can be added to the whitelist.")
    print("=" * 60)


def load_model_from_hub(
    repo_id: str, revision: str | None = None
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any] | None]:
    """
    从 Hugging Face Hub 下载并加载模型的 state_dict 和配置。

    Args:
        repo_id: Hub 上的仓库 ID（例如 'lerobot/aloha'）。
        revision: 要使用的具体 git 修订版（分支、标签或提交哈希）。

    Returns:
        一个元组，包含模型的状态字典、策略配置，
        以及训练配置（若未找到 train_config.json 则为 None）。
    """
    # 下载文件。
    safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)

    # 加载 state_dict
    state_dict = load_safetensors(safetensors_path)

    # 加载 config
    with open(config_path) as f:
        config = json.load(f)

    # 尝试加载 train_config（可选）
    train_config = None
    try:
        train_config_path = hf_hub_download(repo_id=repo_id, filename="train_config.json", revision=revision)
        with open(train_config_path) as f:
            train_config = json.load(f)
    except FileNotFoundError:
        print("train_config.json not found - continuing without training configuration")

    return state_dict, config, train_config


def main():
    parser = argparse.ArgumentParser(
        description="Migrate policy models with normalization layers to new pipeline system"
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        required=True,
        help="Path to pretrained model (hub repo or local directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for migrated model (default: same as pretrained-path)",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Push migrated model to hub")
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="Hub repository ID for pushing (default: same as pretrained-path)",
    )
    parser.add_argument("--revision", type=str, default=None, help="Revision of the model to load")
    parser.add_argument("--private", action="store_true", help="Make the hub repository private")
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Git branch to use when pushing to hub. If specified, a PR will be created automatically (default: push directly to main)",
    )

    args = parser.parse_args()

    # 加载模型和配置
    print(f"Loading model from {args.pretrained_path}...")
    if os.path.isdir(args.pretrained_path):
        # 本地目录
        state_dict = load_safetensors(os.path.join(args.pretrained_path, "model.safetensors"))
        with open(os.path.join(args.pretrained_path, "config.json")) as f:
            config = json.load(f)

        # 尝试加载 train_config（可选）
        train_config = None
        train_config_path = os.path.join(args.pretrained_path, "train_config.json")
        if os.path.exists(train_config_path):
            with open(train_config_path) as f:
                train_config = json.load(f)
        else:
            print("train_config.json not found - continuing without training configuration")
    else:
        # Hub 仓库
        state_dict, config, train_config = load_model_from_hub(args.pretrained_path, args.revision)

    # 提取归一化统计量
    print("Extracting normalization statistics...")
    stats = extract_normalization_stats(state_dict)

    print(f"Found normalization statistics for: {list(stats.keys())}")

    # 检测输入特征和归一化模式
    print("Detecting features and normalization modes...")
    features, norm_map = detect_features_and_norm_modes(config, stats)

    print(f"Detected features: {list(features.keys())}")
    print(f"Normalization modes: {norm_map}")

    # 从 state_dict 中移除归一化层
    print("Removing normalization layers from model...")
    new_state_dict = remove_normalization_layers(state_dict)
    new_state_dict = clean_state_dict(new_state_dict, remove_str="._orig_mod")

    removed_keys = set(state_dict.keys()) - set(new_state_dict.keys())
    if removed_keys:
        print(f"Removed {len(removed_keys)} normalization layer keys")

    # 确定输出路径
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if os.path.isdir(args.pretrained_path):
            output_dir = Path(args.pretrained_path).parent / f"{Path(args.pretrained_path).name}_migrated"
        else:
            output_dir = Path(f"./{args.pretrained_path.replace('/', '_')}_migrated")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 从 config 中提取策略类型
    if "type" not in config:
        raise ValueError("Policy type not found in config.json. The config must contain a 'type' field.")

    policy_type = config["type"]
    print(f"Detected policy type: {policy_type}")

    # 清理 config —— 移除不应传给 config 构造器的字段
    cleaned_config = dict(config)

    # 移除不属于 config 类构造器的字段
    fields_to_remove = ["normalization_mapping", "type"]
    for field in fields_to_remove:
        if field in cleaned_config:
            print(f"Removing '{field}' field from config")
            del cleaned_config[field]

    # 如果存在，则将 input_features 和 output_features 转换为 PolicyFeature 对象
    if "input_features" in cleaned_config:
        cleaned_config["input_features"] = convert_features_to_policy_features(
            cleaned_config["input_features"]
        )
    if "output_features" in cleaned_config:
        cleaned_config["output_features"] = convert_features_to_policy_features(
            cleaned_config["output_features"]
        )

    # 向 config 添加归一化映射
    cleaned_config["normalization_mapping"] = norm_map

    # 使用工厂创建策略配置
    print(f"Creating {policy_type} policy configuration...")
    policy_config = make_policy_config(policy_type, **cleaned_config)

    # 使用工厂创建策略实例
    print(f"Instantiating {policy_type} policy...")
    policy_class = get_policy_class(policy_type)
    policy = policy_class(policy_config)

    # 定义某些策略类型可接受的已知缺失键白名单（例如权重绑定）
    known_missing_keys_whitelist = {
        "pi0": ["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"],
        # 根据需要在此处添加其他策略类型及其已知的缺失键
    }

    # 以优雅处理缺失键的方式加载 state dict
    problematic_missing_keys = load_state_dict_with_missing_key_handling(
        policy=policy,
        state_dict=new_state_dict,
        policy_type=policy_type,
        known_missing_keys_whitelist=known_missing_keys_whitelist,
    )
    policy.to(torch.float32)
    # 使用工厂创建预处理器和后处理器
    print("Creating preprocessor and postprocessor using make_pre_post_processors...")
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_config, dataset_stats=stats)

    # 若要推送到 hub，则确定 hub 仓库 ID
    hub_repo_id = None
    if args.push_to_hub:
        if args.hub_repo_id:
            hub_repo_id = args.hub_repo_id
        else:
            if not os.path.isdir(args.pretrained_path):
                # 使用同一仓库，加 "_migrated" 后缀
                hub_repo_id = f"{args.pretrained_path}_migrated"
            else:
                raise ValueError("--hub-repo-id must be specified when pushing local model to hub")

    # 首先将所有组件保存到本地目录
    print(f"Saving preprocessor to {output_dir}...")
    preprocessor.save_pretrained(output_dir)

    print(f"Saving postprocessor to {output_dir}...")
    postprocessor.save_pretrained(output_dir)

    print(f"Saving model to {output_dir}...")
    policy.save_pretrained(output_dir)

    # 生成并保存模型卡
    print("Generating model card...")
    # 从原始 config 获取元数据
    dataset_repo_id = "unknown"
    if train_config is not None:
        dataset_repo_id = train_config.get("repo_id", "unknown")
    license = config.get("license", "apache-2.0")

    tags = config.get("tags", ["robotics", "lerobot", policy_type]) or ["robotics", "lerobot", policy_type]
    tags = set(tags).union({"robotics", "lerobot", policy_type})
    tags = list(tags)

    # 通过自由函数助手生成模型卡（PreTrainedPolicy.generate_model_card 已在
    # publisher 重设计中移除），然后应用上面恢复的元数据 —— 迁移后的
    # 策略配置不携带原始仓库的卡片字段。
    from lerobot.common.train_utils import generate_model_card

    card = generate_model_card(policy.config)
    card.data.datasets = dataset_repo_id
    card.data.license = license
    card.data.tags = sorted(tags)

    # 本地保存模型卡
    card.save(str(output_dir / "README.md"))
    print(f"Model card saved to {output_dir / 'README.md'}")
    # 若已请求，则在单个操作中将所有文件推送到 hub
    if args.push_to_hub and hub_repo_id:
        api = HfApi()

        # 确定是否应创建 PR（若指定了分支则自动创建）
        create_pr = args.branch is not None
        target_location = f"branch '{args.branch}'" if args.branch else "main branch"

        print(f"Pushing all migrated files to {hub_repo_id} on {target_location}...")

        # 在单个提交中上传所有文件，若指定了分支则自动创建 PR
        commit_message = "Migrate policy to PolicyProcessorPipeline system"
        commit_description = None

        if create_pr:
            # 为 PR 正文单独设置 commit 描述
            commit_description = """**Automated Policy Migration to PolicyProcessorPipeline**

This PR migrates your model to the new LeRobot policy format using the modern PolicyProcessorPipeline architecture.

## What Changed

### **New Architecture - PolicyProcessorPipeline**
Your model now uses external PolicyProcessorPipeline components for data processing instead of built-in normalization layers. This provides:
- **Modularity**: Separate preprocessing and postprocessing pipelines
- **Flexibility**: Easy to swap, configure, and debug processing steps
- **Compatibility**: Works with the latest LeRobot ecosystem

### **Normalization Extraction**
We've extracted normalization statistics from your model's state_dict and removed the built-in normalization layers:
- **Extracted patterns**: `normalize_inputs.*`, `unnormalize_outputs.*`, `normalize.*`, `unnormalize.*`, `input_normalizer.*`, `output_normalizer.*`
- **Statistics preserved**: Mean, std, min, max values for all features
- **Clean model**: State dict now contains only core model weights

### **Files Added**
- **preprocessor_config.json**: Configuration for input preprocessing pipeline
- **postprocessor_config.json**: Configuration for output postprocessing pipeline
- **model.safetensors**: Clean model weights without normalization layers
- **config.json**: Updated model configuration
- **train_config.json**: Training configuration
- **README.md**: Updated model card with migration information

### **Benefits**
- **Backward Compatible**: Your model behavior remains identical
- **Future Ready**: Compatible with latest LeRobot features and updates
- **Debuggable**: Easy to inspect and modify processing steps
- **Portable**: Processors can be shared and reused across models

### **Usage**
```python
# Load your migrated model
from lerobot.policies import get_policy_class
from lerobot.processor import PolicyProcessorPipeline

# The preprocessor and postprocessor are now external
preprocessor = PolicyProcessorPipeline.from_pretrained("your-model-repo", config_filename="preprocessor_config.json")
postprocessor = PolicyProcessorPipeline.from_pretrained("your-model-repo", config_filename="postprocessor_config.json")
policy = get_policy_class("your-policy-type").from_pretrained("your-model-repo")

# Process data through the pipeline
processed_batch = preprocessor(raw_batch)
action = policy(processed_batch)
final_action = postprocessor(action)
```

*Generated automatically by the LeRobot policy migration script*"""

        upload_kwargs = {
            "repo_id": hub_repo_id,
            "folder_path": output_dir,
            "repo_type": "model",
            "commit_message": commit_message,
            "revision": args.branch,
            "create_pr": create_pr,
            "allow_patterns": ["*.json", "*.safetensors", "*.md"],
            "ignore_patterns": ["*.tmp", "*.log"],
        }

        # 若创建 PR，则为 PR 正文添加 commit_description
        if create_pr and commit_description:
            upload_kwargs["commit_description"] = commit_description

        api.upload_folder(**upload_kwargs)

        if create_pr:
            print("All files pushed and pull request created successfully!")
        else:
            print("All files pushed to main branch successfully!")

    print("\nMigration complete!")
    print(f"Migrated model saved to: {output_dir}")
    if args.push_to_hub and hub_repo_id:
        if args.branch:
            print(
                f"Successfully pushed all files to branch '{args.branch}' and created PR on https://huggingface.co/{hub_repo_id}"
            )
        else:
            print(f"Successfully pushed to https://huggingface.co/{hub_repo_id}")
        if args.branch:
            print(f"\nView the branch at: https://huggingface.co/{hub_repo_id}/tree/{args.branch}")
            print(
                f"View the PR at: https://huggingface.co/{hub_repo_id}/discussions (look for the most recent PR)"
            )
        else:
            print(f"\nView the changes at: https://huggingface.co/{hub_repo_id}")

    # 显示关于任何问题的缺失键的最终总结
    display_migration_summary_with_warnings(problematic_missing_keys)


if __name__ == "__main__":
    main()
