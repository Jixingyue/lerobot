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
import importlib
import importlib.metadata
import logging
from typing import Any

from draccus.choice_types import ChoiceRegistry


def is_package_available(
    pkg_name: str, import_name: str | None = None, return_version: bool = False
) -> tuple[bool, str] | bool:
    """
    检查包的 spec 是否存在，并获取其版本以避免导入到本地目录。

    参数:
        pkg_name: 通过 pip 安装时使用的包名（例如 "python-can"）。
        import_name: 导入该包时实际使用的名称（例如 "can"）。
                     未提供时默认为 pkg_name。
        return_version: 是否返回版本字符串。
    """
    if import_name is None:
        import_name = pkg_name

    # 使用导入名称检查模块 spec 是否存在
    package_exists = importlib.util.find_spec(import_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # 获取包版本的主要方法
            package_version = importlib.metadata.version(pkg_name)

        except importlib.metadata.PackageNotFoundError:
            # 回退方法：仅适用于 "torch" 以及包含 "dev" 的版本
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(import_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # 检查版本是否包含 "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # 如果无法导入该包，则视为不可用
                    package_exists = False
            else:
                # 对于 "torch" 以外的包，不尝试回退，直接设为不可用
                package_exists = False
        logging.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


def get_safe_default_video_backend():
    logger = logging.getLogger(__name__)
    # 尽管 torchcodec 已安装，它在运行时仍可能无法加载。
    if importlib.util.find_spec("torchcodec"):
        try:
            importlib.import_module("torchcodec")
            return "torchcodec"
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning(
                f"{e}\n'torchcodec' is installed but cannot be loaded (see the error above). "
                "Falling back to 'pyav' as a default decoder."
            )
            return "pyav"
    else:
        logger.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"


_require_package_cache: dict[str, bool] = {}


def require_package(pkg_name: str, extra: str, import_name: str | None = None) -> None:
    """当某个可选功能所需的包缺失时，抛出信息明确的 ImportError。"""
    cache_key = import_name or pkg_name
    if cache_key not in _require_package_cache:
        _require_package_cache[cache_key] = is_package_available(pkg_name, import_name)
    if not _require_package_cache[cache_key]:
        raise ImportError(
            f"'{pkg_name}' is required but not installed. Install it with: "
            f"pip install 'lerobot[{extra}]' (or uv pip install 'lerobot[{extra}]')"
        )


# ── 集中管理的可用性标志 ────────────────────────────────────────
# 所有可选依赖的检查都放在这里，以便代码库的其他部分
# 可以直接 ``from lerobot.utils.import_utils import _foo_available``。
# 不要在其他模块中临时定义 ``is_package_available(...)`` 调用。

# 机器学习 / 训练
_lancedb_available = is_package_available("lancedb")
_transformers_available = is_package_available("transformers")
_peft_available = is_package_available("peft")
_scipy_available = is_package_available("scipy")
_diffusers_available = is_package_available("diffusers")
_torchdiffeq_available = is_package_available("torchdiffeq")

# 硬件 SDK
_serial_available = is_package_available("pyserial", import_name="serial")
_deepdiff_available = is_package_available("deepdiff")
_dynamixel_sdk_available = is_package_available("dynamixel-sdk", import_name="dynamixel_sdk")
_feetech_sdk_available = is_package_available("feetech-servo-sdk", import_name="scservo_sdk")
_reachy2_sdk_available = is_package_available("reachy2_sdk")
_can_available = is_package_available("python-can", "can")
_motorbridge_available = is_package_available("motorbridge")
_motorbridge_smart_servo_available = is_package_available(
    "motorbridge-smart-servo", import_name="motorbridge_smart_servo"
)
_unitree_sdk_available = is_package_available("unitree-sdk2py", "unitree_sdk2py")
_pyrealsense2_available = is_package_available("pyrealsense2") or is_package_available(
    "pyrealsense2-macosx", import_name="pyrealsense2"
)
_zmq_available = is_package_available("pyzmq", import_name="zmq")
_hebi_available = is_package_available("hebi-py", import_name="hebi")
_teleop_available = is_package_available("teleop")
_placo_available = is_package_available("placo")
_hidapi_available = is_package_available("hidapi", import_name="hid")

# 数据 / 序列化
_datasets_available = is_package_available("datasets")
_pandas_available = is_package_available("pandas")
_faker_available = is_package_available("faker")

# 视频编码 / 解码
_av_available = is_package_available("av")

# 其他
_pynput_available = is_package_available("pynput")
_pygame_available = is_package_available("pygame")
_qwen_vl_utils_available = is_package_available("qwen-vl-utils", import_name="qwen_vl_utils")
_grpc_available = is_package_available("grpcio", import_name="grpc")
_wallx_deps_available = (
    _transformers_available and _peft_available and _torchdiffeq_available and _qwen_vl_utils_available
)


def make_device_from_device_class(config: ChoiceRegistry) -> Any:
    """
    根据对象的 `ChoiceRegistry` 配置动态实例化该对象。

    此工厂利用 `config` 对象类型中的模块路径和类名来定位并实例化
    相应的设备类（而不是配置类）。
    它通过从配置类名末尾去掉 'Config' 来推导设备类名，并尝试在设备实现
    通常所在的几个候选模块中进行查找。
    """
    if not isinstance(config, ChoiceRegistry):
        raise ValueError(f"Config should be an instance of `ChoiceRegistry`, got {type(config)}")

    config_cls = config.__class__
    module_path = config_cls.__module__  # 典型情况：lerobot_teleop_mydevice.config_mydevice
    config_name = config_cls.__name__  # 典型情况：MyDeviceConfig

    # 推导设备类名（去掉 "Config"）
    if not config_name.endswith("Config"):
        raise ValueError(f"Config class name '{config_name}' does not end with 'Config'")

    device_class_name = config_name[:-6]  # 典型情况：MyDeviceConfig -> MyDevice

    # 构造用于搜索设备类的候选模块列表
    parts = module_path.split(".")
    parent_module = ".".join(parts[:-1]) if len(parts) > 1 else module_path
    candidates = [
        module_path,  # 配置自身所在的模块（单文件插件）
        parent_module,  # 典型情况：lerobot_teleop_mydevice
        parent_module + "." + device_class_name.lower(),  # 典型情况：lerobot_teleop_mydevice.mydevice
    ]

    # 处理名为 "config_xxx" 的模块——尝试将该部分替换为 "xxx"
    last = parts[-1] if parts else ""
    if last.startswith("config_"):
        candidates.append(".".join(parts[:-1] + [last.replace("config_", "")]))

    # 在保持顺序的同时去重
    seen: set[str] = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    tried: list[str] = []
    for candidate in candidates:
        tried.append(candidate)
        try:
            module = importlib.import_module(candidate)
        except ImportError:
            continue

        if hasattr(module, device_class_name):
            cls = getattr(module, device_class_name)
            if callable(cls):
                try:
                    return cls(config)
                except TypeError as e:
                    raise TypeError(
                        f"Failed to instantiate '{device_class_name}' from module '{candidate}': {e}"
                    ) from e

    raise ImportError(
        f"Could not locate device class '{device_class_name}' for config '{config_name}'. "
        f"Tried modules: {tried}. Ensure your device class name is the config class name without "
        f"'Config' and that it's importable from one of those modules."
    )


def register_third_party_plugins() -> None:
    """
    发现并导入第三方 LeRobot 插件，以便它们能够自行注册。

    此函数使用 `importlib.metadata` 查找环境中已安装的
    （包括可编辑安装的）以 'lerobot_robot_'、'lerobot_camera_'、
    'lerobot_teleoperator_'、'lerobot_policy_'、'lerobot_env_' 或
    'lerobot_strategy_' 开头的包，并导入它们。
    """
    prefixes = (
        "lerobot_robot_",
        "lerobot_camera_",
        "lerobot_teleoperator_",
        "lerobot_policy_",
        "lerobot_env_",
        "lerobot_strategy_",
    )
    imported: list[str] = []
    failed: list[str] = []

    def attempt_import(module_name: str):
        try:
            importlib.import_module(module_name)
            imported.append(module_name)
            logging.info("Imported third-party plugin: %s", module_name)
        except Exception:
            logging.exception("Could not import third-party plugin: %s", module_name)
            failed.append(module_name)

    for dist in importlib.metadata.distributions():
        dist_name = dist.metadata.get("Name")
        if not dist_name:
            continue
        if dist_name.startswith(prefixes):
            attempt_import(dist_name)

    logging.debug("Third-party plugin import summary: imported=%s failed=%s", imported, failed)
