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
"""以纯文本形式加载的提示词模板。

每个使用位置对应一个文件。模板使用 ``str.format(**vars)`` 替换；
我们在这里有意避免使用 jinja2，以便模板可以在普通编辑器中查看，
并且能够干净地通过 ``ruff format`` 往返处理。
"""

from __future__ import annotations

import os
from pathlib import Path

_DIR = Path(__file__).parent


def load(name: str) -> str:
    """从 ``prompts/`` 目录读取提示词模板 ``name.txt``。

    当 ``LEROBOT_PROMPT_OVERRIDE_<name>`` 环境变量被设置为
    非空值时，它优先于打包的文件。这使得提示词搜索（例如
    GEPA）无需重新构建包即可将候选模板注入远程作业；
    覆盖值必须保留调用点所格式化的相同 ``{placeholder}``
    字段。
    """
    override = os.environ.get(f"LEROBOT_PROMPT_OVERRIDE_{name}")
    if override and override.strip():
        return override
    path = _DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")
