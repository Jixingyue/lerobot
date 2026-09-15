#!/usr/bin/env python

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

from __future__ import annotations

import importlib
import logging
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.import_utils import _unitree_sdk_available, require_package

from ..robot import Robot
from .config_unitree_g1 import UnitreeG1Config
from .g1_kinematics import G1_29_ArmIK
from .g1_utils import (
    NUM_MOTORS,
    REMOTE_AXES,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    default_remote_input,
)

if TYPE_CHECKING or _unitree_sdk_available:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize as _SDKChannelFactoryInitialize,
        ChannelPublisher as _SDKChannelPublisher,
        ChannelSubscriber as _SDKChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        LowCmd_ as hg_LowCmd,
        LowState_ as hg_LowState,
    )
    from unitree_sdk2py.utils.crc import CRC
else:
    _SDKChannelFactoryInitialize = None
    _SDKChannelPublisher = None
    _SDKChannelSubscriber = None
    unitree_hg_msg_dds__LowCmd_ = None
    hg_LowCmd = None
    hg_LowState = None
    CRC = None

logger = logging.getLogger(__name__)


@runtime_checkable
class RobotController(Protocol):
    """驱动 ``UnitreeG1`` 后台控制线程的控制器接口。

    涵盖运动控制器（GR00T、Holosoma）和全身控制器（SONIC）。

    每个周期，机器人将最新的 lowstate 以及传入动作的快照交给控制器，
    并发布其返回的绝对关节目标，键名为 ``<joint>.q``。该接口定义在此处
    而非 ``controllers/`` 中，这样导入机器人时不会引入控制器实现及其
    onnxruntime 依赖。

    控制器还可以暴露以下任意属性，机器人在存在时会拾取它们：

    - ``kp`` / ``kd``: ``(29,)`` 形状的 PD 增益，随目标一起发布，覆盖配置。
    - ``default_angles``: ``(29,)`` 形状的初始姿态，残差动作施加于其上。
    - ``action_ft`` / ``observation_ft``: 特征字典，接管机器人默认的
      29 自由度动作空间和本体感知状态（SONIC 的 64 维潜在 token）。
    - ``observation_state()``: ``observation_ft`` 中声明的键的当前值。
    """

    control_dt: float
    """以秒为单位的控制周期；设置机器人控制器线程的运行速率。"""

    def run_step(self, action: dict, lowstate) -> dict:
        """将一个 lowstate 加动作映射为以 ``<joint>.q`` 为键的绝对关节目标。"""
        ...

    def reset(self) -> None:
        """丢弃每个回合的状态，例如历史缓冲区和保持的指令。"""
        ...


def make_robot_controller(name: str | None) -> RobotController | None:
    """按类名实例化机器人控制器。如果 name 为 None 则返回 None。"""
    if name is None:
        return None
    controllers = {
        "GrootLocomotionController": "lerobot.robots.unitree_g1.controllers.gr00t_locomotion",
        "HolosomaLocomotionController": "lerobot.robots.unitree_g1.controllers.holosoma_locomotion",
        "SonicWholeBodyController": "lerobot.robots.unitree_g1.controllers.sonic_whole_body",
    }
    module_path = controllers.get(name)
    if module_path is None:
        raise ValueError(f"Unknown controller: {name!r}. Available: {list(controllers)}")
    module = importlib.import_module(module_path)
    return getattr(module, name)()


# DDS 主题名称遵循 Unitree SDK 命名规范
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowState = "rt/lowstate"


@dataclass
class MotorState:
    q: float | None = None  # 位置
    dq: float | None = None  # 速度
    tau_est: float | None = None  # 估计力矩
    temperature: float | None = None  # 电机温度


@dataclass
class IMUState:
    quaternion: np.ndarray | None = None  # [w, x, y, z]
    gyroscope: np.ndarray | None = None  # [x, y, z] 角速度（rad/s）
    accelerometer: np.ndarray | None = None  # [x, y, z] 线加速度（m/s²）
    rpy: np.ndarray | None = None  # [roll, pitch, yaw]（rad）
    temperature: float | None = None  # IMU 温度


# g1 观测类
@dataclass
class G1_29_LowState:  # noqa: N801
    motor_state: list[MotorState] = field(default_factory=lambda: [MotorState() for _ in G1_29_JointIndex])
    imu_state: IMUState = field(default_factory=IMUState)
    wireless_remote: bytes | None = None  # 原始无线遥控器数据
    mode_machine: int = 0  # 机器人模式


class UnitreeG1(Robot):
    config_class = UnitreeG1Config
    name = "unitree_g1"

    def __init__(self, config: UnitreeG1Config):
        require_package("unitree-sdk2py", extra="unitree_g1", import_name="unitree_sdk2py")
        super().__init__(config)

        logger.info("Initialize UnitreeG1...")

        self.config = config
        self.control_dt = config.control_dt

        # 初始化相机配置（基于 ZMQ）— 实际连接在 connect() 中
        self._cameras = make_cameras_from_configs(config.cameras)

        # 根据模式导入通道类
        if config.is_simulation:
            self._ChannelFactoryInitialize = _SDKChannelFactoryInitialize
            self._ChannelPublisher = _SDKChannelPublisher
            self._ChannelSubscriber = _SDKChannelSubscriber
        else:
            from .unitree_sdk2_socket import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )

            self._ChannelFactoryInitialize = ChannelFactoryInitialize
            self._ChannelPublisher = ChannelPublisher
            self._ChannelSubscriber = ChannelSubscriber

        # 初始化状态变量
        self.sim_env = None
        self._env_wrapper = None
        self._lowstate = None
        self._lowstate_lock = threading.Lock()
        # 保护共享的 lowcmd 消息：控制器线程、send_action()、reset() 和关闭路径
        # 都通过它发布，即使更新被中断也能携带有效的 CRC。
        self._lowcmd_lock = threading.Lock()
        # 决定谁可以在一段时间内驱动关节：一个控制器周期，或整个重置扫描。
        # 粒度比 _lowcmd_lock 更粗，后者只保证单条指令的原子性。
        self._control_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.subscribe_thread = None

        self.arm_ik = G1_29_ArmIK() if config.gravity_compensation else None

        # 动态加载的控制器
        self.controller: RobotController | None = make_robot_controller(config.controller)
        # 控制器线程状态
        self._controller_thread = None
        self._controller_action_lock = threading.Lock()
        self.controller_input = default_remote_input()
        self.controller_output = {}

    def _subscribe_lowstate(self):  # 以 250Hz 轮询机器人状态
        while not self._shutdown_event.is_set():
            start_time = time.time()

            # 如果处于仿真模式则步进仿真
            if self.config.is_simulation and self.sim_env is not None:
                self.sim_env.step()

            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()

                # 使用 jointindex 捕获电机状态
                for joint in G1_29_JointIndex:
                    lowstate.motor_state[joint].q = msg.motor_state[joint].q
                    lowstate.motor_state[joint].dq = msg.motor_state[joint].dq
                    lowstate.motor_state[joint].tau_est = msg.motor_state[joint].tau_est
                    lowstate.motor_state[joint].temperature = msg.motor_state[joint].temperature

                # 捕获 IMU 状态
                lowstate.imu_state.quaternion = list(msg.imu_state.quaternion)
                lowstate.imu_state.gyroscope = list(msg.imu_state.gyroscope)
                lowstate.imu_state.accelerometer = list(msg.imu_state.accelerometer)
                lowstate.imu_state.rpy = list(msg.imu_state.rpy)
                lowstate.imu_state.temperature = msg.imu_state.temperature

                # 捕获无线遥控器数据
                lowstate.wireless_remote = msg.wireless_remote

                # 捕获 mode_machine
                lowstate.mode_machine = msg.mode_machine

                with self._lowstate_lock:
                    self._lowstate = lowstate

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))  # 维持恒定的控制周期
            time.sleep(sleep_time)

    def publish_lowcmd(
        self,
        action: RobotAction,
        kp: np.ndarray | list[float] | None = None,
        kd: np.ndarray | list[float] | None = None,
        tau: np.ndarray | list[float] | None = None,
    ) -> None:  # 在每次请求时写入机器人指令
        with self._lowcmd_lock:
            for motor in G1_29_JointIndex:
                key = f"{motor.name}.q"
                if key in action:
                    self.msg.motor_cmd[motor.value].q = action[key]
                    self.msg.motor_cmd[motor.value].qd = 0
                    self.msg.motor_cmd[motor.value].kp = (
                        kp[motor.value] if kp is not None else self.kp[motor.value]
                    )
                    self.msg.motor_cmd[motor.value].kd = (
                        kd[motor.value] if kd is not None else self.kd[motor.value]
                    )
                    self.msg.motor_cmd[motor.value].tau = tau[motor.value] if tau is not None else 0.0

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # 声明自身本体感知状态的控制器（SONIC 的 64-D token 回显）会替换
        # 原始关节位置而非在其基础上扩展，就像 action_features 将动作空间
        # 交给控制器那样。
        controller_ft = getattr(self.controller, "observation_ft", None)
        proprio_ft = self._motors_ft if controller_ft is None else dict(controller_ft)
        return {**proprio_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # 完全未配置控制器：原始 29-DoF 关节遥操作。
        if self.controller is None:
            return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

        # 全身控制器（SONIC）：64-D 潜在 token。
        controller_ft = getattr(self.controller, "action_ft", None)
        if controller_ft is not None:
            return dict(controller_ft)

        # 运动控制器（GR00T / Holosoma）：手臂关节目标 + 摇杆轴。
        # TODO：让 GR00T/Holosoma 也声明各自的 action_features，这样每个
        # 控制器都声明自己的动作空间，这个兵底分支就可以去掉。
        arm_features = {f"{G1_29_JointArmIndex(motor).name}.q": float for motor in G1_29_JointArmIndex}
        remote_features = dict.fromkeys(REMOTE_AXES, float)
        return {**arm_features, **remote_features}

    def _controller_loop(self):
        """以后台线程形式按策略的 control_dt 运行控制器。"""
        control_dt = self.controller.control_dt
        logger.info(f"Controller loop starting with control_dt={control_dt} ({1.0 / control_dt:.1f}Hz)")

        loop_count = 0
        last_log_time = time.time()

        while not self._shutdown_event.is_set():
            start_time = time.time()

            with self._lowstate_lock:
                lowstate = self._lowstate

            if lowstate is not None and self.controller is not None:
                loop_count += 1
                if time.time() - last_log_time >= 5.0:  # 每 5 秒记录一次
                    actual_hz = loop_count / (time.time() - last_log_time)
                    logger.info(
                        f"Controller actual rate: {actual_hz:.1f}Hz (target: {1.0 / control_dt:.1f}Hz)"
                    )
                    loop_count = 0
                    last_log_time = time.time()
                # 读取控制器输入快照
                with self._controller_action_lock:
                    controller_input = dict(self.controller_input)

                # 运行一步控制器并将其作为一个控制周期发布，这样复位扫描
                # 就不会把自己的目标与本 tick 的目标交错在一起。
                with self._control_lock:
                    controller_action = self.controller.run_step(controller_input, lowstate)

                    # 写入控制器输出快照
                    with self._controller_action_lock:
                        self.controller_output = dict(controller_action)

                    ctrl_kp = self.controller.kp if hasattr(self.controller, "kp") else None
                    ctrl_kd = self.controller.kd if hasattr(self.controller, "kd") else None
                    self.publish_lowcmd(controller_action, kp=ctrl_kp, kd=ctrl_kd)

            elapsed = time.time() - start_time
            sleep_time = max(0, control_dt - elapsed)
            time.sleep(sleep_time)

    def calibrate(self) -> None:
        # TODO: 实现 g1_29 标定
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:  # 连接到 DDS
        # 初始化 DDS 通道和仿真环境
        if self.config.is_simulation:
            from lerobot.envs import make_env

            self._ChannelFactoryInitialize(0, "lo")
            self._env_wrapper = make_env("lerobot/unitree-g1-mujoco", trust_remote_code=True)
            # 从字典结构中提取实际的 gym 环境
            self.sim_env = self._env_wrapper["hub_env"][0].envs[0]
        else:
            self._ChannelFactoryInitialize(0, config=self.config)

        # 初始化直接电机控制接口
        self.lowcmd_publisher = self._ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = self._ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()

        # 启动订阅线程以读取机器人状态
        self.subscribe_thread = threading.Thread(target=self._subscribe_lowstate)
        self.subscribe_thread.start()

        # 连接相机
        for cam in self._cameras.values():
            if not cam.is_connected:
                cam.connect()

        logger.info(f"Connected {len(self._cameras)} camera(s).")

        # 初始化 lowcmd 消息
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0

        # 等待第一条状态消息到达
        lowstate = None
        deadline = time.time() + 10.0
        while lowstate is None:
            with self._lowstate_lock:
                lowstate = self._lowstate
            if lowstate is None:
                if time.time() > deadline:
                    raise TimeoutError("Timed out waiting for robot state (10s)")
                logger.warning("[UnitreeG1] Waiting for robot state...")
                time.sleep(0.01)
        logger.info("[UnitreeG1] Connected to robot.")
        self.msg.mode_machine = lowstate.mode_machine

        # 优先使用当前控制器的增益（例如 SONIC 从其 ONNX 加载 kp/kd）；
        # 否则回退到配置默认值。
        if self.controller is not None and hasattr(self.controller, "kp"):
            self.kp = np.array(self.controller.kp, dtype=np.float32)
            self.kd = np.array(self.controller.kd, dtype=np.float32)
        else:
            self.kp = np.array(self.config.kp, dtype=np.float32)
            self.kd = np.array(self.config.kd, dtype=np.float32)

        for joint in G1_29_JointIndex:
            self.msg.motor_cmd[joint].mode = 1
            self.msg.motor_cmd[joint].kp = self.kp[joint.value]
            self.msg.motor_cmd[joint].kd = self.kd[joint.value]
            self.msg.motor_cmd[joint].q = lowstate.motor_state[joint.value].q

        # 在控制器接管前平滑过渡到它的 home 姿态，以免第一条指令
        # 从连接时的姿态突然跳变。reset() 会自行选取那个姿态。
        if self.controller is not None and hasattr(self.controller, "default_angles"):
            self.reset()

        # 若已启用则启动控制器线程
        if self.controller is not None:
            self._controller_thread = threading.Thread(target=self._controller_loop, daemon=True)
            self._controller_thread.start()
            fps = int(1.0 / self.controller.control_dt)
            logger.info(f"Controller thread started ({fps}Hz)")

    def _send_zero_torque(self) -> None:
        """在关闭前发送一个零增益指令，使关节变为无源。"""
        try:
            with self._lowstate_lock:
                lowstate = self._lowstate
            if lowstate is None:
                return
            action = {f"{motor.name}.q": lowstate.motor_state[motor.value].q for motor in G1_29_JointIndex}
            zero_gains = np.zeros(29, dtype=np.float32)
            self.publish_lowcmd(action, kp=zero_gains, kd=zero_gains, tau=zero_gains)
            logger.info("Sent zero-torque command for safe shutdown")
        except Exception as e:
            logger.warning(f"Failed to send zero-torque on disconnect: {e}")

    def disconnect(self):
        # 向线程发出停止信号并解除任何等待
        self._shutdown_event.set()

        # 等待控制器线程结束。它必须在变为无源之前停止，
        # 否则一个已在途的 tick 会重新加强关节刚度，而零力矩就不是
        # 机器人的最后一条指令。
        if self._controller_thread is not None:
            self._controller_thread.join(timeout=2.0)
            if self._controller_thread.is_alive():
                logger.warning("Controller thread did not stop cleanly")

        # 将机器人置于无源模式
        if not self.config.is_simulation:
            self._send_zero_torque()

        # 等待订阅线程结束
        if self.subscribe_thread is not None:
            self.subscribe_thread.join(timeout=2.0)
            if self.subscribe_thread.is_alive():
                logger.warning("Subscribe thread did not stop cleanly")

        # 关闭仿真环境
        if self.config.is_simulation and self.sim_env is not None:
            try:
                # 先强制终止图像发布子进程，以避免长时间等待
                if hasattr(self.sim_env, "simulator") and hasattr(self.sim_env.simulator, "sim_env"):
                    sim_env_inner = self.sim_env.simulator.sim_env
                    if hasattr(sim_env_inner, "image_publish_process"):
                        proc = sim_env_inner.image_publish_process
                        if proc.process and proc.process.is_alive():
                            logger.info("Force-terminating image publish subprocess...")
                            proc.stop_event.set()
                            proc.process.terminate()
                            proc.process.join(timeout=1)
                            if proc.process.is_alive():
                                proc.process.kill()
                self.sim_env.close()
            except Exception as e:
                logger.warning(f"Error closing sim_env: {e}")
            self.sim_env = None
            self._env_wrapper = None

        # 断开相机连接
        for cam in self._cameras.values():
            cam.disconnect()

    def get_observation(self) -> RobotObservation:
        with self._lowstate_lock:
            lowstate = self._lowstate
        if lowstate is None:
            return {}

        obs = {}

        # 电机 - 所有关节的 q、dq、tau
        for motor in G1_29_JointIndex:
            name = motor.name
            idx = motor.value
            obs[f"{name}.q"] = lowstate.motor_state[idx].q
            obs[f"{name}.dq"] = lowstate.motor_state[idx].dq
            obs[f"{name}.tau"] = lowstate.motor_state[idx].tau_est

        # IMU - 陀螺仪
        if lowstate.imu_state.gyroscope:
            obs["imu.gyro.x"] = lowstate.imu_state.gyroscope[0]
            obs["imu.gyro.y"] = lowstate.imu_state.gyroscope[1]
            obs["imu.gyro.z"] = lowstate.imu_state.gyroscope[2]

        # IMU - 加速度计
        if lowstate.imu_state.accelerometer:
            obs["imu.accel.x"] = lowstate.imu_state.accelerometer[0]
            obs["imu.accel.y"] = lowstate.imu_state.accelerometer[1]
            obs["imu.accel.z"] = lowstate.imu_state.accelerometer[2]

        # IMU - 四元数
        if lowstate.imu_state.quaternion:
            obs["imu.quat.w"] = lowstate.imu_state.quaternion[0]
            obs["imu.quat.x"] = lowstate.imu_state.quaternion[1]
            obs["imu.quat.y"] = lowstate.imu_state.quaternion[2]
            obs["imu.quat.z"] = lowstate.imu_state.quaternion[3]

        # IMU - rpy（滚转/俯仰/偏航）
        if lowstate.imu_state.rpy:
            obs["imu.rpy.roll"] = lowstate.imu_state.rpy[0]
            obs["imu.rpy.pitch"] = lowstate.imu_state.rpy[1]
            obs["imu.rpy.yaw"] = lowstate.imu_state.rpy[2]

        # 无线遥控器（供遥操作器使用的原始字节）
        if lowstate.wireless_remote:
            obs["wireless_remote"] = lowstate.wireless_remote

        # 控制器贡献的观测（例如 SONIC 将其上一个解码出的 token 作为
        # observation.state 回显，从而使输出 token 的 VLA 能针对自己先前的 token 闭环）。
        if self.controller is not None and hasattr(self.controller, "observation_state"):
            obs.update(self.controller.observation_state())

        # 相机 - 从 ZMQ 相机读取图像
        for cam_name, cam in self._cameras.items():
            if getattr(cam, "use_rgb", True):
                obs[cam_name] = cam.read_latest()
            if getattr(cam, "use_depth", False):
                obs[f"{cam_name}_depth"] = cam.read_latest_depth()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        action_to_publish = action
        if self.controller is not None:
            # 控制器线程拥有腿部/腰部的所有权。这里我们只更新摇杆输入，
            # 并发布来自遥操作器的手臂目标。
            self._update_controller_action(action)
            arm_prefixes = tuple(j.name for j in G1_29_JointArmIndex)
            action_to_publish = {
                key: value
                for key, value in action.items()
                if key.endswith(".q") and key.startswith(arm_prefixes)
            }
            if not action_to_publish:
                # 这里没有手臂相关的项，因此发布只会以调用方的速率、在控制器
                # 之上重新发送控制器线程自己的上一条指令并附带新的 CRC。
                # 仅含 token 的动作（SONIC 策略）每一步都会走到这里。
                return action

        tau = None
        if self.config.gravity_compensation and self.arm_ik is not None:
            tau = np.zeros(29, dtype=np.float32)
            action_np = np.array(
                [
                    action_to_publish.get(f"{joint.name}.q", self.msg.motor_cmd[joint.value].q)
                    for joint in G1_29_JointArmIndex
                ],
                dtype=np.float32,
            )
            arm_tau = self.arm_ik.solve_tau(action_np)
            arm_start_idx = G1_29_JointArmIndex.kLeftShoulderPitch.value
            for joint in G1_29_JointArmIndex:
                local_idx = joint.value - arm_start_idx
                tau[joint.value] = arm_tau[local_idx]

        self.publish_lowcmd(action_to_publish, tau=tau)
        return action

    def _update_controller_action(self, action: RobotAction) -> None:
        """将传入的遥操作动作值转发进 ``controller_input``；每个控制器
        只读取它所能理解的键。"""
        with self._controller_action_lock:
            for key, value in action.items():
                if isinstance(key, str) and value is not None:
                    self.controller_input[key] = value

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        with self._lowstate_lock:
            return self._lowstate is not None

    @property
    def _motors_ft(self) -> dict[str, type]:
        """所有 29 个关节的关节位置。"""
        return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

    @property
    def cameras(self) -> dict:
        return self._cameras

    def reset(
        self,
        control_dt: float | None = None,
        default_positions: list[float] | None = None,
    ) -> None:  # 将机器人移动到默认位置
        if control_dt is None:
            control_dt = self.config.control_dt
        if default_positions is None:
            # 当控制器拥有自己的 home 姿态时归位到它：策略正是围绕该姿态
            # 训练的（SONIC 从 ONNX 元数据中读取它），而配置中的默认值
            # 只是供原始关节遥操作使用的通用回退选项。
            controller_home = getattr(self.controller, "default_angles", None)
            source = self.config.default_positions if controller_home is None else controller_home
            default_positions = np.array(source, dtype=np.float32)

        # 在整个扫描过程中保持控制权限。否则控制器线程会持续发布它自己的
        # 目标，机器人就会同时被两个写入者驱动。
        with self._control_lock:
            if self.config.is_simulation and self.sim_env is not None:
                self.sim_env.reset()
                self.publish_lowcmd(
                    {f"{motor.name}.q": float(default_positions[motor.value]) for motor in G1_29_JointIndex}
                )
            else:
                total_time = 3.0
                num_steps = int(total_time / control_dt)

                # 获取当前状态
                obs = self.get_observation()

                # 记录当前位置
                init_dof_pos = np.zeros(NUM_MOTORS, dtype=np.float32)
                for motor in G1_29_JointIndex:
                    init_dof_pos[motor.value] = obs[f"{motor.name}.q"]

                # 插值到默认位置
                for step in range(num_steps):
                    start_time = time.time()

                    alpha = step / num_steps
                    action_dict = {}
                    for motor in G1_29_JointIndex:
                        target_pos = default_positions[motor.value]
                        interp_pos = init_dof_pos[motor.value] * (1 - alpha) + target_pos * alpha
                        action_dict[f"{motor.name}.q"] = float(interp_pos)

                    self.publish_lowcmd(action_dict)

                    # 保持恒定的控制速率
                    elapsed = time.time() - start_time
                    sleep_time = max(0, control_dt - elapsed)
                    time.sleep(sleep_time)

            # 在线程仍被拦止期间重置控制器内部状态（步态相位、观测历史等），
            # 使其无法在我们清理缓冲区的同时半途重新填充。
            if self.controller is not None and hasattr(self.controller, "reset"):
                self.controller.reset()

        logger.info("Reached default position")
