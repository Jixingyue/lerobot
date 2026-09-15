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
import importlib
import inspect
import json
import pkgutil
import sys
import tempfile
from argparse import ArgumentError
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from pathlib import Path
from pkgutil import ModuleInfo
from types import ModuleType
from typing import Any, TypeVar, cast

import draccus
import yaml  # type: ignore[import-untyped]
from draccus.help_formatter import SimpleHelpFormatter
from draccus.utils import DecodingError
from draccus.wrappers import DataclassWrapper
from draccus.wrappers.choice_wrapper import ChoiceWrapper, UnionWrapper
from draccus.wrappers.field_wrapper import FieldWrapper
from draccus.wrappers.suppressing_argparse import SuppressingArgumentParser
from draccus.wrappers.wrapper import AggregateWrapper, Wrapper

from lerobot.utils.utils import has_method

F = TypeVar("F", bound=Callable[..., object])

PATH_KEY = "path"
PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"

# 用于存储从 YAML/JSON 配置文件中提取的 path 参数，这样即使它们
# 没有通过 CLI 传入，get_path_arg() 也能找到它们。
_config_path_args: dict[str, str] = {}

# 用于存储非 path 的 YAML 覆盖项，以便 validate() 将它们传递给 from_pretrained。
_config_yaml_overrides: dict[str, list[str]] = {}


def _flatten_to_cli_args(d: dict, prefix: str = "") -> list[str]:
    """将嵌套字典递归展平为 CLI 风格的参数（例如 {"lr": 1e-4} -> ["--lr=0.0001"]）。"""
    args = []
    for key, value in d.items():
        if key in (PATH_KEY, draccus.CHOICE_TYPE_KEY):
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            value = str(value).lower()
        if isinstance(value, dict):
            args.extend(_flatten_to_cli_args(value, full_key))
        elif value is not None and not isinstance(value, list):
            args.append(f"--{full_key}={value}")
    return args


def get_cli_overrides(field_name: str, args: Sequence[str] | None = None) -> list[str] | None:
    """在指定的嵌套属性层级解析来自 CLI 的参数。

    例如，假设主脚本的调用方式为：
    python myscript.py --arg1=1 --arg2.subarg1=abc --arg2.subarg2=some/path

    如果在 myscript.py 执行期间调用，get_cli_overrides("arg2") 将返回：
    ["--subarg1=abc" "--subarg2=some/path"]
    """
    if args is None:
        args = sys.argv[1:]
    attr_level_args = []
    detect_string = f"--{field_name}."
    excluded_names = (draccus.CHOICE_TYPE_KEY, PATH_KEY)
    for index, arg in enumerate(args):
        if not arg.startswith(detect_string):
            continue

        denested_arg = arg.removeprefix(detect_string)
        if denested_arg.split("=", maxsplit=1)[0] in excluded_names:
            continue

        attr_level_args.append(f"--{denested_arg}")
        if "=" not in arg and index + 1 < len(args) and not args[index + 1].startswith("--"):
            attr_level_args.append(args[index + 1])

    return attr_level_args


def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    if args is None:
        args = sys.argv[1:]
    option = f"--{arg_name}"
    for index, arg in enumerate(args):
        if arg.startswith(f"{option}="):
            return arg.removeprefix(f"{option}=")
        if arg == option and index + 1 < len(args) and not args[index + 1].startswith("--"):
            return args[index + 1]
    return None


def parse_plugin_args(plugin_arg_suffix: str, args: Sequence[str]) -> dict[str, str]:
    """从命令行参数中解析插件相关的参数。

    此函数从命令行参数中提取匹配指定后缀模式的参数。
    它接受 '--key=value' 和 '--key value' 格式的参数，并以字典形式返回。

    Args:
        plugin_arg_suffix (str): 用于识别插件相关参数的后缀。
        cli_args (Sequence[str]): 要解析的命令行参数序列。

    Returns:
        dict: 包含解析后的插件参数的字典，其中：
            - 键是参数名（如果存在 '--' 前缀则移除）
            - 值是对应的参数值

    Example:
        >>> args = ["--env.discover_packages_path=my_package", "--other_arg=value"]
        >>> parse_plugin_args("discover_packages_path", args)
        {'env.discover_packages_path': 'my_package'}
    """
    plugin_args = {}
    for index, arg in enumerate(args):
        if not arg.startswith("--"):
            continue

        key, separator, value = arg[2:].partition("=")
        if plugin_arg_suffix not in key:
            continue
        if not separator:
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                continue
            value = args[index + 1]
        plugin_args[key] = value
    return plugin_args


class PluginLoadError(Exception):
    """当插件加载失败时抛出。"""


def load_plugin(plugin_path: str) -> None:
    """从给定的 Python 包路径加载并初始化插件。

    此函数通过导入插件的包及其所有子模块来尝试加载插件。
    插件注册预期发生在包初始化期间，即当包被导入时，gym 环境
    应该被注册，配置类应该使用 `register_subclass` 装饰器注册到其父类。

    Args:
        plugin_path (str): 插件的 Python 包路径（例如 "mypackage.plugins.myplugin"）

    Raises:
        PluginLoadError: 如果由于导入错误或包路径无效而无法加载插件。

    Examples:
        >>> load_plugin("external_plugin.core")  # 从外部包加载插件

    Notes:
        - 插件包应在导入期间处理自身的注册
        - 插件包中的所有子模块都会被导入
        - 实现遵循 Python 打包指南中的插件发现模式

    See Also:
        https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
    """
    try:
        package_module = importlib.import_module(plugin_path, __package__)
    except (ImportError, ModuleNotFoundError) as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e

    def iter_namespace(ns_pkg: ModuleType) -> Iterable[ModuleInfo]:
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    try:
        for _finder, pkg_name, _ispkg in iter_namespace(package_module):
            importlib.import_module(pkg_name)
    except ImportError as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e


def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    result = parse_arg(f"{field_name}.{PATH_KEY}", args)
    if result is None:
        result = _config_path_args.get(field_name)
    return result


def get_yaml_overrides(field_name: str) -> list[str]:
    return _config_yaml_overrides.get(field_name, [])


def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)


def _register_scoped_actions(
    wrapper: Wrapper, parser: SuppressingArgumentParser, cli_args: Sequence[str]
) -> None:
    """类似于 draccus 自身的 Wrapper.register_actions，但对于 ChoiceType 字段，
    只递归进入已选择的子类（根据 CLI 的 `.type` 参数），而不是每个已注册的选项。

    这镜像了 draccus 0.11.x 内部的 wrapper 遍历逻辑，因为其公共解析器会在解析命令行之前
    急切地注册每个选项。更新 draccus 时请保持同步。
    """
    if isinstance(wrapper, ChoiceWrapper):
        group = parser.add_argument_group(title=wrapper.title, description=wrapper.description)
        children = wrapper._children
        arg_name = f"{wrapper.dest}.{draccus.CHOICE_TYPE_KEY}" if wrapper.dest else draccus.CHOICE_TYPE_KEY
        group.add_argument(
            f"--{arg_name}",
            choices=list(children.keys()),
            help=f"Which type of {wrapper.title} to use",
            required=wrapper.required,
        )
        selected = get_type_arg(wrapper.dest, cli_args) if wrapper.dest else None
        if selected in children:
            _register_scoped_actions(children[selected], parser, cli_args)
    elif isinstance(wrapper, DataclassWrapper):
        group = parser.add_argument_group(title=wrapper.title, description=wrapper.description)
        for child in wrapper._children:
            if isinstance(child, AggregateWrapper):
                parser.add_argument(
                    f"--{child.name}", type=str, required=False, help=f"Config file for {child.name}"
                )
                _register_scoped_actions(child, parser, cli_args)
            elif isinstance(child, FieldWrapper):
                child.add_action(group)
    elif isinstance(wrapper, UnionWrapper):
        group = parser.add_argument_group(title=wrapper.title, description=wrapper.description)
        has_field_wrapper = False
        for child in wrapper._children:
            if isinstance(child, (DataclassWrapper, ChoiceWrapper)):
                _register_scoped_actions(child, parser, cli_args)
            elif isinstance(child, FieldWrapper):
                has_field_wrapper = True
        if has_field_wrapper:
            group.add_argument(f"--{wrapper.dest}", required=False)
    else:
        wrapper.register_actions(parser)


def print_scoped_help(config_class: type, cli_args: Sequence[str]) -> None:
    """打印范围限定到 CLI 上已解析选项（例如 --env.type=pusht）的 --help 输出，
    而不是 draccus 默认的展开每个 ChoiceType 字段的所有已注册子类。"""
    parser = SuppressingArgumentParser(formatter_class=SimpleHelpFormatter)
    parser.add_argument(
        f"--{draccus.utils.CONFIG_ARG}", type=str, help="Path for a config file to parse with draccus"
    )
    _register_scoped_actions(DataclassWrapper(config_class), parser, cli_args)
    parser.print_help()


def filter_arg(field_to_filter: str, args: Sequence[str] | None = None) -> list[str]:
    if args is None:
        return []
    option = f"--{field_to_filter}"
    filtered_args = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == option:
            index += 1
            if index < len(args) and not args[index].startswith("--"):
                index += 1
            continue
        if arg.startswith(f"{option}="):
            index += 1
            continue
        filtered_args.append(arg)
        index += 1
    return filtered_args


def filter_path_args(fields_to_filter: str | list[str], args: Sequence[str] | None = None) -> list[str]:
    """
    过滤与带有特定 path 参数的字段相关的命令行参数。

    Args:
        fields_to_filter (str | list[str]): 需要过滤其参数的单个字符串或字符串列表。
        args (Sequence[str] | None): 要过滤的命令行参数序列。
            默认为 None。

    Returns:
        list[str]: 过滤后的参数列表，与指定字段相关的参数已被移除。

    Raises:
        ArgumentError: 如果同一字段同时指定了 path 参数（例如 `--field_name.path`）
            和 type 参数（例如 `--field_name.type`）。
    """
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]

    filtered_args = [] if args is None else list(args)

    for field in fields_to_filter:
        if get_path_arg(field, args):
            if get_type_arg(field, args):
                raise ArgumentError(
                    argument=None,
                    message=f"Cannot specify both --{field}.{PATH_KEY} and --{field}.{draccus.CHOICE_TYPE_KEY}",
                )
            option_prefix = f"--{field}."
            retained_args = []
            index = 0
            while index < len(filtered_args):
                arg = filtered_args[index]
                if arg.startswith(option_prefix):
                    index += 1
                    if (
                        "=" not in arg
                        and index < len(filtered_args)
                        and not filtered_args[index].startswith("--")
                    ):
                        index += 1
                    continue
                retained_args.append(arg)
                index += 1
            filtered_args = retained_args

    return filtered_args


def extract_path_fields_from_config(config_path: str, path_fields: list[str]) -> str:
    """在 draccus 处理之前，从 YAML/JSON 配置中提取 `path` 字段。

    当用户在 YAML 配置中指定例如 ``policy.path: lerobot/smolvla_base`` 时，
    draccus 会失败，因为 ``path`` 不是策略配置类的有效字段。
    此函数提取这些 path 值，将它们存储在 ``_config_path_args`` 中供
    ``get_path_arg()`` 稍后检索，并返回清理后的临时配置文件路径。
    """
    config_file = Path(config_path)
    suffix = config_file.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_file) as f:
            config_data = json.load(f)
    else:
        return config_path

    if not isinstance(config_data, dict):
        return config_path

    modified = False
    for field in path_fields:
        if field in config_data and isinstance(config_data[field], dict) and PATH_KEY in config_data[field]:
            _config_path_args[field] = str(config_data[field].pop(PATH_KEY))
            remaining = config_data[field]
            if remaining:
                _config_yaml_overrides[field] = _flatten_to_cli_args(remaining)
            del config_data[field]
            modified = True

    if not modified:
        return config_path

    # 将清理后的配置写入临时文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        if suffix in (".yaml", ".yml"):
            yaml.dump(config_data, tmp, default_flow_style=False)
        else:
            json.dump(config_data, tmp, indent=2)
    return tmp.name


def wrap(config_path: Path | None = None) -> Callable[[F], F]:
    """
    HACK：类似于 draccus.wrap，但额外做了三件事：
        - 会从 CLI 中移除 '.path' 参数，以便稍后处理。
        - 如果传入了 'config_path' 且主配置类有 'from_pretrained' 方法，
          会从那里初始化它，以允许直接从 hub 获取配置
        - 会加载 CLI 参数中指定的插件。这些插件通常会注册自己的配置类子类，
          这样 draccus 就能根据 CLI 的 '.type' 参数找到要实例化的正确类
    """

    def wrapper_outer(fn: F) -> F:
        @wraps(fn)
        def wrapper_inner(*args: Any, **kwargs: Any) -> Any:
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]
            if len(args) > 0 and type(args[0]) is argtype:
                cfg = args[0]
                args = args[1:]
            else:
                cli_args = sys.argv[1:]
                plugin_args = parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, cli_args)
                for plugin_cli_arg, plugin_path in plugin_args.items():
                    try:
                        load_plugin(plugin_path)
                    except PluginLoadError as e:
                        # 将相关的 CLI 参数添加到错误消息中
                        raise PluginLoadError(f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}") from e
                    cli_args = filter_arg(plugin_cli_arg, cli_args)
                if "--help" in cli_args or "-h" in cli_args:
                    print_scoped_help(argtype, cli_args)
                    sys.exit(0)
                config_path_cli = parse_arg("config_path", cli_args)
                if has_method(argtype, "__get_path_fields__"):
                    path_fields = argtype.__get_path_fields__()
                    cli_args = filter_path_args(path_fields, cli_args)
                    # 同时从 YAML/JSON 配置文件中提取 path 字段
                    if config_path_cli:
                        config_path_cli = extract_path_fields_from_config(config_path_cli, path_fields)
                try:
                    if has_method(argtype, "from_pretrained") and config_path_cli:
                        cli_args = filter_arg("config_path", cli_args)
                        cfg = argtype.from_pretrained(config_path_cli, cli_args=cli_args)
                    else:
                        if config_path_cli:
                            cli_args = filter_arg("config_path", cli_args)
                        cfg = draccus.parse(
                            config_class=argtype,
                            config_path=config_path_cli or config_path,
                            args=cli_args,
                        )
                except DecodingError as e:
                    print(f"error: {e}", file=sys.stderr)
                    sys.exit(1)
            response = fn(cfg, *args, **kwargs)
            return response

        return cast(F, wrapper_inner)

    return cast(Callable[[F], F], wrapper_outer)
