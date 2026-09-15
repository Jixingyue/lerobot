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
"""按回合暂存。

每个模块将其原始输出作为 JSONL 文件写入
``<staging_dir>/episode_{ep:06d}/<module>.jsonl``。写入器读回此暂存树
并将行分区到两个语言列中。

这里首选 JSONL 而不是 parquet，因为暂存产物旨在
可供人类检查、易于在提示词迭代之间进行差异比较，并且可以简单地追加。
最终数据集格式是 parquet；暂存只是一个中间产物。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ModuleName = str

_MODULES: tuple[ModuleName, ...] = (
    "plan",
    "interjections",
    "vqa",
)


@dataclass
class EpisodeStaging:
    """单个回合的暂存模块输出的文件系统布局。"""

    root: Path
    episode_index: int

    @property
    def episode_dir(self) -> Path:
        return self.root / f"episode_{self.episode_index:06d}"

    def path_for(self, module: ModuleName) -> Path:
        if module not in _MODULES:
            raise ValueError(f"Unknown module {module!r}; expected one of {_MODULES}")
        return self.episode_dir / f"{module}.jsonl"

    def write(self, module: ModuleName, rows: Iterable[dict[str, Any]]) -> Path:
        path = self.path_for(module)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 原子替换：写入中途崩溃否则会留下一个半写的 JSONL 文件，
        # ``read()`` 随后无法解析它。写入兄弟 .tmp 并重命名，
        # 以便目标路径始终只指向完整的文件。
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                f.write("\n")
        tmp_path.replace(path)
        return path

    def read(self, module: ModuleName) -> list[dict[str, Any]]:
        path = self.path_for(module)
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def read_all(self) -> dict[ModuleName, list[dict[str, Any]]]:
        return {m: self.read(m) for m in _MODULES}

    def has(self, module: ModuleName) -> bool:
        return self.path_for(module).exists()
