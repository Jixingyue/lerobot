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
"""离线将 DCP 格式的检查点转换为可分发的 safetensors 模型。

以单进程运行（无需 GPU，无需进程组）。示例：

```bash
lerobot-convert-dcp --checkpoint_dir=outputs/train/run/checkpoints/005000
lerobot-convert-dcp --checkpoint_dir=... --delete_dcp=true --push_to_hub=user/my-policy
```

`--push_to_hub` 会将转换后的目录作为模型仓库发布，并支持优雅降级：
核心产物（model.safetensors、config.json、processor 文件）始终会上传；
只有当 `train_config.json`（以及其中指定的数据集）可访问时，README
模型卡片才会补充训练/数据集元数据，否则会输出一条 WARNING 明确列出被跳过的内容。
DCP 分片产物永远不会上传 — 发布的仓库只包含 safetensors。
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi

from lerobot.configs import parser
from lerobot.distributed.checkpoint import dcp_to_safetensors
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.utils import init_logging


@dataclass
class ConvertDcpConfig:
    """离线 DCP 到 safetensors 检查点转换的 CLI 配置。"""

    # 检查点步骤目录（包含 pretrained_model/），或者就是
    # pretrained_model 目录本身。
    checkpoint_dir: Path
    # 在转换成功后删除 DCP 分片目录。
    delete_dcp: bool = False
    # 将转换后的目录发布到该 Hub 仓库 id（例如 "user/my-policy"）。
    push_to_hub: str | None = None
    private: bool | None = None


def _locate_pretrained_dir(checkpoint_dir: Path) -> Path:
    """从用户提供的检查点路径解析出 pretrained_model/ 目录。

    参数：
        checkpoint_dir (Path): 检查点步骤目录（包含 `pretrained_model/`），或者就是
            `pretrained_model` 目录本身。

    返回：
        Path: 存在时返回嵌套的 `pretrained_model/` 目录，否则原样返回
        `checkpoint_dir`。
    """
    nested = checkpoint_dir / PRETRAINED_MODEL_DIR
    return nested if nested.is_dir() else checkpoint_dir


def _publish_converted(pretrained_dir: Path, repo_id: str, private: bool | None) -> None:
    """尽力发布转换后的检查点目录，支持优雅降级。

    核心产物（model.safetensors、config.json、processor 文件）始终会上传；
    只有当 `train_config.json`（以及其中指定的数据集）可访问时，README
    模型卡片才会补充训练/数据集元数据，否则会输出一条 WARNING 列出被跳过的内容。
    DCP 分片产物不会包含在上传中。

    参数：
        pretrained_dir (Path): 要上传的转换后的 `pretrained_model/` 目录。
        repo_id (str): 目标 Hub 模型仓库 id（例如 "user/my-policy"）；不存在时会创建。
        private (bool | None): 传递给 `create_repo` 的仓库可见性；None 表示保持 Hub
            （或已有仓库的）默认设置。
    """
    from lerobot.common.train_utils import generate_model_card
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig

    train_cfg = None
    dataset_meta = None
    if (pretrained_dir / TRAIN_CONFIG_NAME).is_file():
        try:
            train_cfg = TrainPipelineConfig.from_pretrained(pretrained_dir)
        except Exception as e:  # noqa: BLE001 — 降级处理，绝不阻塞上传
            logging.warning(f"Could not parse {TRAIN_CONFIG_NAME} ({e}); README will lack training metadata.")
    else:
        logging.warning(f"{TRAIN_CONFIG_NAME} missing; README will lack training metadata.")
    if train_cfg is not None:
        try:
            from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

            dataset_meta = LeRobotDatasetMetadata(
                repo_id=train_cfg.dataset.repo_id,
                root=train_cfg.dataset.root,
                revision=train_cfg.dataset.revision,
            )
        except Exception as e:  # noqa: BLE001
            logging.warning(
                f"Dataset '{train_cfg.dataset.repo_id}' unreachable ({e}); README will lack dataset metadata."
            )
    try:
        model_cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
        card = generate_model_card(model_cfg, cfg=train_cfg, dataset_meta=dataset_meta)
        card.save(str(pretrained_dir / "README.md"))
    except Exception as e:  # noqa: BLE001
        logging.warning(f"Could not build the model card ({e}); publishing without README.")

    api = HfApi()
    repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(pretrained_dir),
        commit_message="Upload converted policy (DCP -> safetensors)",
        allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
        # 除非传入了 --delete_dcp，否则检查点会保留其 DCP 分片目录；
        # 上面的允许列表既不包含 `.distcp` 分片，也不包含其 `.metadata` 附属文件。
        ignore_patterns=["*.tmp", "*.log"],
    )
    logging.info(f"Model pushed to {commit_info.repo_url.url}")


@parser.wrap()
def convert_checkpoint(cfg: ConvertDcpConfig) -> Path:
    """将检查点的 DCP 分片合并为 `model.safetensors`，然后可选地发布它。

    参数：
        cfg (ConvertDcpConfig): 转换选项 — 要转换的检查点目录、是否在合并成功后
            删除 DCP 分片，以及可选的用于发布转换后目录的 Hub 仓库 id
            （和可见性）。

    返回：
        Path: 合并后的 `model.safetensors` 文件的路径。

    异常：
        FileNotFoundError: 如果检查点没有 DCP 分片目录，即它不是以
            `checkpoint_format=dcp`（或 `safetensors_dcp`）保存的。
    """
    from accelerate.utils.constants import FSDP_MODEL_NAME

    pretrained_dir = _locate_pretrained_dir(cfg.checkpoint_dir)
    dcp_dir = pretrained_dir / f"{FSDP_MODEL_NAME}_0"
    if not dcp_dir.is_dir():
        raise FileNotFoundError(
            f"No DCP shard directory at {dcp_dir}. Point --checkpoint_dir at a checkpoint "
            "saved with checkpoint_format=dcp (or safetensors_dcp)."
        )
    logging.info(f"Merging {dcp_dir} -> {pretrained_dir / 'model.safetensors'}")
    safetensors_path = dcp_to_safetensors(dcp_dir, pretrained_dir, delete_dcp=cfg.delete_dcp)
    if cfg.push_to_hub:
        _publish_converted(pretrained_dir, cfg.push_to_hub, cfg.private)
    return safetensors_path


def main() -> None:
    """`lerobot-convert-dcp` 的控制台入口：初始化日志并运行转换。"""
    init_logging()
    convert_checkpoint()


if __name__ == "__main__":
    main()
