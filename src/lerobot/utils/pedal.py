# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""基于 evdev 的通用脚踏板监听器。

调用方提供一个回调，接收按下的键码（例如 ``"KEY_A"``）
以及可选的设备路径。监听器在守护线程中运行，
当 :mod:`evdev` 未安装或设备不可用时静默不做任何操作。
策略相关的按键映射逻辑由调用方负责。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_PEDAL_DEVICE = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"


def start_pedal_listener(
    on_press: Callable[[str], None],
    device_path: str = DEFAULT_PEDAL_DEVICE,
) -> threading.Thread | None:
    """启动一个守护线程，将脚踏板按键码转发给 ``on_press``。

    参数
    ----------
    on_press:
        每次脚踏板按下事件时以按下的键码字符串（例如 ``"KEY_A"``）
        调用的回调。回调在监听线程中运行，必须是线程安全的。
    device_path:
        Linux 输入设备路径（例如 ``/dev/input/by-id/...``）。

    返回值
    -------
    已启动的守护 :class:`threading.Thread`；当 :mod:`evdev`
    未安装时返回 ``None``（可选依赖；静默不做任何操作）。
    """
    try:
        from evdev import InputDevice, categorize, ecodes
    except ImportError:
        return None

    def pedal_reader() -> None:
        try:
            dev = InputDevice(device_path)
            logger.info("Pedal connected: %s", dev.name)
            for ev in dev.read_loop():
                if ev.type != ecodes.EV_KEY:
                    continue
                key = categorize(ev)
                code = key.keycode
                if isinstance(code, (list, tuple)):
                    code = code[0]
                if key.keystate != 1:  # 仅处理按键按下事件
                    continue
                try:
                    on_press(code)
                except Exception as cb_err:  # pragma: no cover - 防御性处理
                    logger.warning("Pedal callback error: %s", cb_err)
        except (FileNotFoundError, PermissionError):
            pass
        except Exception as e:
            logger.warning("Pedal error: %s", e)

    thread = threading.Thread(target=pedal_reader, daemon=True, name="PedalListener")
    thread.start()
    return thread
