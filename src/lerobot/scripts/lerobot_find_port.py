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
查找与你的 MotorsBus 关联的 USB 端口的辅助工具。

示例：

```shell
lerobot-find-port
```
"""

import platform
import time
from pathlib import Path


def find_available_ports():
    from lerobot.utils.import_utils import require_package

    require_package("pyserial", extra="hardware", import_name="serial")
    from serial.tools import list_ports

    if platform.system() == "Windows":
        # 使用 pyserial 列出 COM 端口
        ports = [port.device for port in list_ports.comports()]
    else:  # Linux/macOS
        # 在基于 Unix 的系统上列出 /dev/tty* 端口
        ports = [str(path) for path in Path("/dev").glob("tty*")]
    return ports


def find_port():
    print("Finding all available ports for the MotorsBus.")
    ports_before = find_available_ports()
    print("Ports before disconnecting:", ports_before)

    print("Remove the USB cable from your MotorsBus and press Enter when done.")
    input()  # 等待用户断开设备连接

    time.sleep(0.5)  # 留出一些时间让端口被释放
    ports_after = find_available_ports()
    ports_diff = list(set(ports_before) - set(ports_after))

    if len(ports_diff) == 1:
        port = ports_diff[0]
        print(f"The port of this MotorsBus is '{port}'")
        print("Reconnect the USB cable.")
    elif len(ports_diff) == 0:
        raise OSError(f"Could not detect the port. No difference was found ({ports_diff}).")
    else:
        raise OSError(f"Could not detect the port. More than one port was found ({ports_diff}).")


def main():
    find_port()


if __name__ == "__main__":
    main()
