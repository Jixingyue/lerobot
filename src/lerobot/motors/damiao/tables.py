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

"""Damiao 电机的配置表。"""

from enum import IntEnum


# 电机类型定义
class MotorType(IntEnum):
    DM3507 = 0
    DM4310 = 1
    DM4310_48V = 2
    DM4340 = 3
    DM4340_48V = 4
    DM6006 = 5
    DM8006 = 6
    DM8009 = 7
    DM10010L = 8
    DM10010 = 9
    DMH3510 = 10
    DMH6215 = 11
    DMG6220 = 12


# 控制模式
class ControlMode(IntEnum):
    MIT = 1
    POS_VEL = 2
    VEL = 3
    TORQUE_POS = 4


# 电机变量 ID（RID）
class MotorVariable(IntEnum):
    UV_VALUE = 0
    KT_VALUE = 1
    OT_VALUE = 2
    OC_VALUE = 3
    ACC = 4
    DEC = 5
    MAX_SPD = 6
    MST_ID = 7
    ESC_ID = 8
    TIMEOUT = 9
    CTRL_MODE = 10
    DAMP = 11
    INERTIA = 12
    HW_VER = 13
    SW_VER = 14
    SN = 15
    NPP = 16
    RS = 17
    LS = 18
    FLUX = 19
    GR = 20
    PMAX = 21
    VMAX = 22
    TMAX = 23
    I_BW = 24
    KP_ASR = 25
    KI_ASR = 26
    KP_APR = 27
    KI_APR = 28
    OV_VALUE = 29
    GREF = 30
    DETA = 31
    V_BW = 32
    IQ_C1 = 33
    VL_C1 = 34
    CAN_BR = 35
    SUB_VER = 36
    U_OFF = 50
    V_OFF = 51
    K1 = 52
    K2 = 53
    M_OFF = 54
    DIR = 55
    P_M = 80
    XOUT = 81


# 电机限位参数 [PMAX, VMAX, TMAX]
# PMAX：最大位置（rad）
# VMAX：最大速度（rad/s）
# TMAX：最大力矩（N·m）
MOTOR_LIMIT_PARAMS = {
    MotorType.DM3507: (12.5, 30, 10),
    MotorType.DM4310: (12.5, 30, 10),
    MotorType.DM4310_48V: (12.5, 50, 10),
    MotorType.DM4340: (12.5, 8, 28),
    MotorType.DM4340_48V: (12.5, 10, 28),
    MotorType.DM6006: (12.5, 45, 20),
    MotorType.DM8006: (12.5, 45, 40),
    MotorType.DM8009: (12.5, 45, 54),
    MotorType.DM10010L: (12.5, 25, 200),
    MotorType.DM10010: (12.5, 20, 200),
    MotorType.DMH3510: (12.5, 280, 1),
    MotorType.DMH6215: (12.5, 45, 10),
    MotorType.DMG6220: (12.5, 45, 10),
}

# 电机型号名称
MODEL_NAMES = {
    MotorType.DM3507: "dm3507",
    MotorType.DM4310: "dm4310",
    MotorType.DM4310_48V: "dm4310_48v",
    MotorType.DM4340: "dm4340",
    MotorType.DM4340_48V: "dm4340_48v",
    MotorType.DM6006: "dm6006",
    MotorType.DM8006: "dm8006",
    MotorType.DM8009: "dm8009",
    MotorType.DM10010L: "dm10010l",
    MotorType.DM10010: "dm10010",
    MotorType.DMH3510: "dmh3510",
    MotorType.DMH6215: "dmh6215",
    MotorType.DMG6220: "dmg6220",
}

# 电机分辨率表（每转编码器计数值）
MODEL_RESOLUTION = {
    "dm3507": 65536,
    "dm4310": 65536,
    "dm4310_48v": 65536,
    "dm4340": 65536,
    "dm4340_48v": 65536,
    "dm6006": 65536,
    "dm8006": 65536,
    "dm8009": 65536,
    "dm10010l": 65536,
    "dm10010": 65536,
    "dmh3510": 65536,
    "dmh6215": 65536,
    "dmg6220": 65536,
}

# Damiao 电机支持的 CAN 波特率
AVAILABLE_BAUDRATES = [
    125000,  # 0: 125 kbps
    200000,  # 1: 200 kbps
    250000,  # 2: 250 kbps
    500000,  # 3: 500 kbps
    1000000,  # 4: 1 mbps（OpenArms 的默认值）
    2000000,  # 5: 2 mbps
    2500000,  # 6: 2.5 mbps
    3200000,  # 7: 3.2 mbps
    4000000,  # 8: 4 mbps
    5000000,  # 9: 5 mbps
]
DEFAULT_BAUDRATE = 1000000  # 1 Mbps 是 OpenArms 的标准值

# 默认超时时间（毫秒）
DEFAULT_TIMEOUT_MS = 1000

# OpenArms 专用配置
# 基于：https://docs.openarm.dev/software/setup/configure-test
# OpenArms 每条手臂有 7 个自由度（双臂共 14 个）
OPENARMS_ARM_MOTOR_IDS = {
    "joint_1": {"send": 0x01, "recv": 0x11},  # J1 - 肩部旋转
    "joint_2": {"send": 0x02, "recv": 0x12},  # J2 - 肩部抬升
    "joint_3": {"send": 0x03, "recv": 0x13},  # J3 - 肘部弯曲
    "joint_4": {"send": 0x04, "recv": 0x14},  # J4 - 腕部弯曲
    "joint_5": {"send": 0x05, "recv": 0x15},  # J5 - 腕部滚转
    "joint_6": {"send": 0x06, "recv": 0x16},  # J6 - 腕部俯仰
    "joint_7": {"send": 0x07, "recv": 0x17},  # J7 - 腕部旋转
}

OPENARMS_GRIPPER_MOTOR_IDS = {
    "gripper": {"send": 0x08, "recv": 0x18},  # J8 - 夹爪
}

# OpenArms 的默认电机类型
OPENARMS_DEFAULT_MOTOR_TYPES = {
    "joint_1": MotorType.DM8009,  # 肩部旋转 - 高力矩
    "joint_2": MotorType.DM8009,  # 肩部抬升 - 高力矩
    "joint_3": MotorType.DM4340,  # 肩部转动
    "joint_4": MotorType.DM4340,  # 肘部弯曲
    "joint_5": MotorType.DM4310,  # 腕部滚转
    "joint_6": MotorType.DM4310,  # 腕部俯仰
    "joint_7": MotorType.DM4310,  # 腕部旋转
    "gripper": MotorType.DM4310,  # 夹爪
}

# MIT 控制参数范围
MIT_KP_RANGE = (0.0, 500.0)
MIT_KD_RANGE = (0.0, 5.0)

# CAN 帧命令 ID
CAN_CMD_ENABLE = 0xFC
CAN_CMD_DISABLE = 0xFD
CAN_CMD_SET_ZERO = 0xFE
CAN_CMD_REFRESH = 0xCC
CAN_CMD_QUERY_PARAM = 0x33
CAN_CMD_WRITE_PARAM = 0x55
CAN_CMD_SAVE_PARAM = 0xAA

# 参数操作使用的 CAN ID
CAN_PARAM_ID = 0x7FF
