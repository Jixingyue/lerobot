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

"""让 docstring 中的示例真正运行的 doctest 管道。

改编自 `transformers.testing_utils`。标准库的两个限制使这成为必要：

1. Ruff 配置了 `docstring-code-format = true`，它会重新格式化 docstring 内的代码，
   并移除结束围栏前的空行。标准库的 `_EXAMPLE_RE` 随后会把 ` ``` ` 吞进
   期望输出组中，导致所有带输出的示例都失败。[`LeRobotDocTestParser`] 修补了
   正则表达式，使其在围栏处停止匹配。
2. `doctest.DocTestFinder` 对 `@property` 和 `functools.wraps` 对象会报告错误的
   行号（https://bugs.python.org/issue17446）。我们的硬件 API 大量使用属性——
   `observation_features`、`action_features`、`is_connected`、`is_calibrated`
   都是抽象属性——因此 [`LeRobotDoctestModule`] 在定位示例前先对它们解包。

两个环境变量可按内容跳过整个示例块：

- `SKIP_CUDA_DOCTEST=1` 跳过需要 GPU 的示例。
- `SKIP_HARDWARE_DOCTEST=1` 跳过需要实体机器人或 Hub 下载的示例。

两者都是基于示例源码的启发式判断，刻意做得粗糙：被不必要跳过的示例
没有任何代价，而在没有相应硬件的机器上运行的示例则会挂起或失败。
"""

import doctest
import functools
import inspect
import os
import re
import sys
from collections.abc import Iterable

from _pytest.doctest import (
    DoctestItem,
    DoctestModule,
    _get_checker,
    _get_continue_on_failure,
    _get_runner,
    get_optionflags,
)
from _pytest.nodes import Collector
from _pytest.outcomes import skip

# 这些调用的进度条本来会被拿去和期望输出比较。前瞻部分会放过
# 已经带有指令的行。
_NOISY_CALL_PATTERN = re.compile(r"(>>> (?!.*# doctest:).*(?:load_dataset|LeRobotDataset)\(.*)")

_CUDA_PATTERN = re.compile(r"cuda|to\(0\)|device=0")

# 串口、视频设备，以及与真实硬件通信的 connect/scan 调用。
_HARDWARE_PATTERN = re.compile(r"/dev/tty|/dev/video|COM\d|\.connect\(|find_cameras\(|find_port\(")

# 任何通过网络访问 Hub 的内容。
_HUB_PATTERN = re.compile(r"from_pretrained\(|push_to_hub\(|snapshot_download\(|load_dataset\(")


def preprocess_string(string: str, skip_cuda_tests: bool, skip_hardware_tests: bool) -> str:
    """预处理 docstring 或 `.mdx` 文件，使其可被 doctest 运行。

    参数：
        string (`str`)：
            `.mdx` 的整个文件内容，或 Python 文件的单个 docstring。
            两者都可能包含多个围栏代码示例。
        skip_cuda_tests (`bool`)：
            是否剔除看起来需要 GPU 的示例。
        skip_hardware_tests (`bool`)：
            是否剔除看起来需要机器人或 Hub 下载的示例。

    返回值：
        `str`：在嘈杂调用处注入 `# doctest: +IGNORE_RESULT` 后的输入；
        若示例被跳过则返回空字符串——此时完全不会为它收集 doctest。
    """
    # 只与示例行匹配，不与周围的文字匹配，以免仅仅*描述* CUDA
    # 或串口的 docstring 被误判为使用了它们。
    example_lines = "\n".join(
        line for line in string.splitlines() if line.lstrip().startswith((">>>", "..."))
    )
    if not example_lines:
        return string

    if skip_cuda_tests and _CUDA_PATTERN.search(example_lines):
        return ""
    if skip_hardware_tests and (
        _HARDWARE_PATTERN.search(example_lines) or _HUB_PATTERN.search(example_lines)
    ):
        return ""

    return _NOISY_CALL_PATTERN.sub(r"\1 # doctest: +IGNORE_RESULT", string)


class LeRobotDocTestParser(doctest.DocTestParser):
    """能理解围栏式、自动格式化代码块的 `DocTestParser`。

    Ruff 的 `docstring-code-format` 会移除结束围栏前的空行，之后标准库的
    `_EXAMPLE_RE` 会把围栏本身读作期望输出的一部分，导致所有带输出的示例
    都失败。下面的正则就是标准库的版本加上一条在围栏处停止匹配的子句。
    """

    # fmt: off
    _EXAMPLE_RE = re.compile(r'''
        # Source consists of a PS1 line followed by zero or more PS2 lines.
        (?P<source>
            (?:^(?P<indent> [ ]*) >>>    .*)    # PS1 line
            (?:\n           [ ]*  \.\.\. .*)*)  # PS2 lines
        \n?
        # Want consists of any non-blank lines that do not start with PS1.
        (?P<want> (?:(?![ ]*$)    # Not a blank line
             (?![ ]*>>>)          # Not a line starting with PS1
             (?:(?!```).)*        # Stop at a closing fence: formatting drops the blank line before it
             (?:\n|$)  # Match a new line or end of string
          )*)
        ''', re.MULTILINE | re.VERBOSE
    )
    # fmt: on

    skip_cuda_tests: bool = os.environ.get("SKIP_CUDA_DOCTEST", "0") == "1"
    skip_hardware_tests: bool = os.environ.get("SKIP_HARDWARE_DOCTEST", "0") == "1"

    def parse(self, string, name="<string>"):
        """先预处理 `string`，再按标准库的方式解析。

        参数：
            string (`str`)：
                要解析的 docstring 或文件内容。
            name (`str`, *可选*，默认为 `"<string>"`)：
                失败信息中使用的名称。

        返回值：
            `list`：示例与穿插的文本，与 `doctest.DocTestParser.parse` 的返回一致。
        """
        string = preprocess_string(string, self.skip_cuda_tests, self.skip_hardware_tests)
        return super().parse(string, name)


class LeRobotDoctestModule(DoctestModule):
    """使用 [`LeRobotDocTestParser`] 进行收集的 pytest `DoctestModule`。

    `doctest.DocTestFinder` 在类定义时就绑定了默认解析器，因此在 `conftest.py`
    中给 `doctest.DocTestParser` 打补丁无法影响到 pytest 构建的 finder。必须显式
    传入解析器，这意味着要重新实现 `collect`。它与 pytest 自身的实现一致。
    """

    def collect(self) -> Iterable[DoctestItem]:
        """收集本模块中的 doctest。

        返回值：
            `Iterable[DoctestItem]`：每个含示例的 docstring 一个条目。示例被
            `preprocess_string` 剔除的 docstring 不产生任何条目。
        """

        class MockAwareDocTestFinder(doctest.DocTestFinder):
            """能为属性和被包装的可调用对象报告正确行号的 doctest finder。"""

            # 上游已在 CPython 3.11.9 / 3.12.3 修复；为旧版解释器保留。我们的硬件 API
            # 大量使用属性（`observation_features`、`is_connected` 等），因此这里的行号
            # 若出错，所有失败都会指向装饰器。https://github.com/python/cpython/issues/61648
            def _find_lineno(self, obj, source_lines):
                if isinstance(obj, property):
                    obj = getattr(obj, "fget", obj)
                if hasattr(obj, "__wrapped__"):
                    obj = inspect.unwrap(obj)
                return super()._find_lineno(obj, source_lines)

            if sys.version_info < (3, 13):
                # 否则 `cached_property` 永远不会被视为当前模块的一部分，
                # 其中的示例会被静默跳过。https://github.com/python/cpython/issues/107995
                def _from_module(self, module, object):
                    if isinstance(object, functools.cached_property):
                        object = object.func
                    return super()._from_module(module, object)

        try:
            module = self.obj
        except Collector.CollectError:
            if self.config.getvalue("doctest_ignore_import_errors"):
                skip(f"unable to import module {self.path!r}")
            else:
                raise

        # Doctest 通过 `getfixture` 和 autouse 支持 fixture。
        self.session._fixturemanager.parsefactories(self)

        finder = MockAwareDocTestFinder(parser=LeRobotDocTestParser())
        optionflags = get_optionflags(self.config)
        runner = _get_runner(
            verbose=False,
            optionflags=optionflags,
            checker=_get_checker(),
            continue_on_failure=_get_continue_on_failure(self.config),
        )
        for test in finder.find(module, module.__name__):
            if test.examples:  # 跳过没有示例的 docstring，以及被解析器剔除的块。
                yield DoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)
