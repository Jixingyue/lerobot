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

"""
该模块负责处理外骨骼中使用的霍尔效应传感器的校准。
由于磁铁/传感器放置不精确，每个关节都有一对输出 sin 和 cos 值的 ADC 通道，
这些值会随着关节旋转而描绘出一个椭圆。我们将该椭圆拟合为单位圆，
并计算单位圆的 arctan2 以得到关节角度。
然后我们存储每个关节的椭圆参数和零位偏移，供运行时使用。
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _serial_available, require_package

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


ADC_MAX = 2**12 - 1
ADC_HALF = ADC_MAX / 2

# 外骨骼关节名称 -> ADC 通道对。TODO: 添加腕部俯仰和腕部偏航
JOINTS = {
    "shoulder_pitch": (0, 1),
    "shoulder_yaw": (2, 3),
    "shoulder_roll": (4, 5),
    "elbow_flex": (6, 7),
    "wrist_roll": (14, 15),
}


@dataclass
class ExoskeletonJointCalibration:
    name: str  # 关节名称
    center_fit: list[float]  # 椭圆中心
    T: list[list[float]]  # 2x2 变换矩阵
    zero_offset: float = 0.0  # 中立位姿的角度


@dataclass
class ExoskeletonCalibration:
    """外骨骼手臂的完整校准数据。"""

    version: int = 2
    side: str = ""
    adc_max: int = ADC_MAX
    joints: list[ExoskeletonJointCalibration] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "side": self.side,
            "adc_max": self.adc_max,
            "joints": [
                {
                    "name": j.name,
                    "center_fit": j.center_fit,
                    "T": j.T,
                    "zero_offset": j.zero_offset,
                }
                for j in self.joints
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExoskeletonCalibration:
        joints = [
            ExoskeletonJointCalibration(
                name=j["name"],
                center_fit=j["center_fit"],
                T=j["T"],
                zero_offset=j.get("zero_offset", 0.0),
            )
            for j in data.get("joints", [])
        ]
        return cls(
            version=data.get("version", 2),
            side=data.get("side", ""),
            adc_max=data.get("adc_max", ADC_MAX),
            joints=joints,
        )


@dataclass(frozen=True)
class CalibParams:
    fit_every: float = 0.15
    min_fit_points: int = 60
    fit_window: int = 900
    max_fit_points: int = 300
    trim_low: float = 0.05
    trim_high: float = 0.95
    median_window: int = 5
    history: int = 3500
    draw_hz: float = 120.0
    sample_count: int = 50


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi]。"""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def joint_z_and_angle(raw16: list[int], j: ExoskeletonJointCalibration) -> tuple[np.ndarray, float]:
    """
    对每个关节应用校准：原始值 → 居中 → 椭圆转圆 → 角度。
    """
    pair = JOINTS[j.name]
    s, c = raw16[pair[0]], raw16[pair[1]]  # 获取 sin 和 cos
    p = np.array([float(c) - ADC_HALF, float(s) - ADC_HALF])  # 将原始值居中
    z = np.asarray(j.T) @ (
        p - np.asarray(j.center_fit)
    )  # 将椭圆居中并逆变换矩阵以得到单位圆坐标
    ang = float(np.arctan2(z[1], z[0])) - j.zero_offset  # 计算角度并应用零位偏移
    return z, normalize_angle(-ang)  # 确保范围为 [-pi, pi]


def exo_raw_to_angles(raw16: list[int], calib: ExoskeletonCalibration) -> dict[str, float]:
    """使用校准将原始传感器读数转换为关节角度。"""
    return {j.name: joint_z_and_angle(raw16, j)[1] for j in calib.joints}


def run_exo_calibration(
    ser: serial.Serial,
    side: str,
    save_path: Path,
    params: CalibParams | None = None,
) -> ExoskeletonCalibration:
    """
    对外骨骼手臂运行交互式校准。
    """
    require_package("pyserial", extra="unitree_g1", import_name="serial")
    try:
        import cv2
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Calibration requires matplotlib and opencv-python. "
            "Install with: pip install matplotlib opencv-python"
        ) from e

    from .exo_serial import read_raw_from_serial

    params = params or CalibParams()
    joint_list = list(JOINTS.items())  # 将字典转换为列表以便索引
    logger.info(f"Starting calibration for {side} exoskeleton arm")

    def running_median(win: deque) -> float:
        return float(np.median(np.fromiter(win, dtype=float)))

    def read_joint_point(raw16: list[int], pair: tuple[int, int]):
        s, c = raw16[pair[0]], raw16[pair[1]]
        return float(c) - ADC_HALF, float(s) - ADC_HALF, float(s), float(c)

    def select_fit_subset(xs, ys):
        """选择并过滤用于椭圆拟合的点。按半径修剪离群点并进行降采样。"""
        n = min(params.fit_window, len(xs))
        if n <= 0:
            return None, None
        x = np.asarray(list(xs)[-n:], dtype=float)  # 最近的 n 个样本
        y = np.asarray(list(ys)[-n:], dtype=float)
        r = np.sqrt(x * x + y * y)  # 到原点的半径
        if len(r) >= 20:
            lo, hi = np.quantile(r, params.trim_low), np.quantile(r, params.trim_high)  # 离群点边界
            keep = (r >= lo) & (r <= hi)
            x, y = x[keep], y[keep]  # 移除离群点
        if len(x) > params.max_fit_points:
            idx = np.linspace(0, len(x) - 1, params.max_fit_points).astype(int)  # 均匀降采样
            x, y = x[idx], y[idx]
        return x, y

    def fit_ellipse_opencv(x, y):
        """使用 OpenCV 将椭圆拟合到 (x,y) 点。返回中心、轴、旋转矩阵和轮廓。"""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if len(x) < 5:
            return None
        pts = np.stack([x, y], axis=1).astype(np.float32).reshape(-1, 1, 2)
        try:
            (xc, yc), (w, h), angle_deg = cv2.fitEllipse(pts)  # 返回中心、轴、旋转角度（度）
        except cv2.error:
            return None
        a, b = float(w) * 0.5, float(h) * 0.5  # 得到椭圆的长短半轴
        phi = np.deg2rad(float(angle_deg))  # 转为弧度
        if b > a:  # 确保长轴为 a
            a, b = b, a
            phi += np.pi / 2.0
        if not np.isfinite(a) or not np.isfinite(b) or a <= 1e-6 or b <= 1e-6:
            return None
        cp, sp = float(np.cos(phi)), float(np.sin(phi))  #
        rot = np.array([[cp, -sp], [sp, cp]], dtype=float)  # 2x2 旋转矩阵
        center = np.array([float(xc), float(yc)], dtype=float)  # 偏移向量
        tt = np.linspace(0, 2 * np.pi, 360)
        outline = (rot @ np.stack([a * np.cos(tt), b * np.sin(tt)])).T + center  # 用于可视化
        return {"center": center, "a": a, "b": b, "R": rot, "ex": outline[:, 0], "ey": outline[:, 1]}

    # 设置 matplotlib
    plt.ion()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.set_xlabel("cos - center")
    ax0.set_ylabel("sin - center")
    ax0.grid(True, alpha=0.25)
    ax0.set_aspect("equal", adjustable="box")
    ax1.set_title("Unit circle + angle")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True, alpha=0.25)
    ax1.set_aspect("equal", adjustable="box")
    tt = np.linspace(0, 2 * np.pi, 360)
    ax1.plot(np.cos(tt), np.sin(tt), "k-", linewidth=1)
    ax0.set_xlim(-2200, 2200)
    ax0.set_ylim(-2200, 2200)
    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-1.4, 1.4)

    sc0 = ax0.scatter([], [], s=6, animated=True)
    (ell_line,) = ax0.plot([], [], "r-", linewidth=2, animated=True)
    sc1 = ax1.scatter([], [], s=6, animated=True)
    (radius_line,) = ax1.plot([], [], "g-", linewidth=2, animated=True)
    angle_text = ax1.text(
        0.02, 0.98, "", transform=ax1.transAxes, va="top", ha="left", fontsize=12, animated=True
    )

    fig.canvas.draw()
    bg0 = fig.canvas.copy_from_bbox(ax0.bbox)
    bg1 = fig.canvas.copy_from_bbox(ax1.bbox)

    # 状态
    joints_out = []
    joint_idx = 0
    phase = "ellipse"
    advance_requested = False
    zero_samples = []

    def on_key(event):
        nonlocal advance_requested
        if event.key in ("n", "N", "enter", " "):
            advance_requested = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    def reset_state():
        return {
            "xs": deque(maxlen=params.history),
            "ys": deque(maxlen=params.history),
            "xu": deque(maxlen=params.history),
            "yu": deque(maxlen=params.history),
            "win_s": deque(maxlen=params.median_window),
            "win_c": deque(maxlen=params.median_window),
            "ellipse_cache": None,
            "T": None,
            "center_fit": None,
            "have_transform": False,
            "latest_z": None,
            "last_fit": 0.0,
        }

    state = reset_state()
    last_draw = 0.0
    name, pair = joint_list[joint_idx]
    fig.canvas.manager.set_window_title(f"[{joint_idx + 1}/{len(joint_list)}] {name} - ELLIPSE")
    ax0.set_title(f"{name} raw (filtered)")
    logger.info(f"[{joint_idx + 1}/{len(joint_list)}] Calibrating {name}")
    logger.info("Step 1: Move joint around to map ellipse, then press 'n'")

    try:
        while plt.fignum_exists(fig.number):
            name, pair = joint_list[joint_idx]

            # 处理校准 GUI 状态：椭圆 → 零位姿 → 下一个关节 -> 椭圆 -> ...
            if phase == "ellipse" and advance_requested and state["have_transform"]:
                joints_out.append(
                    {
                        "name": name,
                        "center_fit": state["center_fit"].tolist(),
                        "T": state["T"].tolist(),
                    }
                )
                logger.info(f"  -> Ellipse saved for {name}")
                phase, zero_samples, advance_requested = "zero_pose", [], False
                fig.canvas.manager.set_window_title(f"[{joint_idx + 1}/{len(joint_list)}] {name} - ZERO POSE")
                ax0.set_title(f"{name} - hold zero pose")
                fig.canvas.draw()
                bg0, bg1 = fig.canvas.copy_from_bbox(ax0.bbox), fig.canvas.copy_from_bbox(ax1.bbox)
                logger.info(f"Step 2: Hold {name} in zero position, then press 'n'")

            elif phase == "ellipse" and advance_requested and not state["have_transform"]:
                logger.info("  (Need valid fit first - keep moving the joint)")
                advance_requested = False

            elif phase == "zero_pose" and advance_requested:
                if len(zero_samples) >= params.sample_count:
                    zero_offset = float(np.mean(zero_samples[-params.sample_count :]))
                    joints_out[-1]["zero_offset"] = zero_offset
                    logger.info(f"  -> {name} zero: {zero_offset:+.3f} rad ({np.degrees(zero_offset):+.1f}°)")
                    joint_idx += 1
                    advance_requested = False

                    if joint_idx >= len(joint_list):
                        # 所有关节已完成
                        calib = ExoskeletonCalibration(
                            version=2,
                            side=side,
                            adc_max=ADC_MAX,
                            joints=[
                                ExoskeletonJointCalibration(
                                    name=j["name"],
                                    center_fit=j["center_fit"],
                                    T=j["T"],
                                    zero_offset=j.get("zero_offset", 0.0),
                                )
                                for j in joints_out
                            ],
                        )
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(save_path, "w") as f:
                            json.dump(calib.to_dict(), f, indent=2)
                        logger.info(f"Saved calibration to {save_path}")
                        logger.info("Calibration complete!")
                        plt.close(fig)
                        return calib

                    # 下一个关节
                    phase, state = "ellipse", reset_state()
                    name, pair = joint_list[joint_idx]
                    fig.canvas.manager.set_window_title(
                        f"[{joint_idx + 1}/{len(joint_list)}] {name} - ELLIPSE"
                    )
                    ax0.set_title(f"{name} raw (filtered)")
                    fig.canvas.draw()
                    bg0, bg1 = fig.canvas.copy_from_bbox(ax0.bbox), fig.canvas.copy_from_bbox(ax1.bbox)
                    logger.info(f"[{joint_idx + 1}/{len(joint_list)}] Calibrating {name}")
                    logger.info("Step 1: Move joint around to map ellipse, then press 'n'")
                else:
                    logger.info(
                        f"  (Collecting samples: {len(zero_samples)}/{params.sample_count} - hold still)"
                    )
                    advance_requested = False

            # 读取传感器
            raw16 = read_raw_from_serial(ser)
            if raw16 is not None:
                x_raw, y_raw, s_raw, c_raw = read_joint_point(raw16, pair)

                if phase == "ellipse":
                    if state["have_transform"]:
                        z = state["T"] @ (np.array([x_raw, y_raw]) - state["center_fit"])
                        state["xu"].append(float(z[0]))
                        state["yu"].append(float(z[1]))
                        state["latest_z"] = (float(z[0]), float(z[1]))
                    state["win_s"].append(s_raw)
                    state["win_c"].append(c_raw)
                    if len(state["win_s"]) >= max(3, params.median_window):
                        state["ys"].append(running_median(state["win_s"]) - ADC_HALF)
                        state["xs"].append(running_median(state["win_c"]) - ADC_HALF)
                else:
                    jdata = joints_out[-1]
                    z = np.array(jdata["T"]) @ (np.array([x_raw, y_raw]) - np.array(jdata["center_fit"]))
                    zero_samples.append(float(np.arctan2(z[1], z[0])))
                    state["latest_z"] = (float(z[0]), float(z[1]))

            # 椭圆拟合
            t = time.time()
            if (
                phase == "ellipse"
                and (t - state["last_fit"]) >= params.fit_every
                and len(state["xs"]) >= params.min_fit_points
            ):
                xfit, yfit = select_fit_subset(state["xs"], state["ys"])
                if xfit is not None and len(xfit) >= params.min_fit_points:
                    fit = fit_ellipse_opencv(xfit, yfit)
                    if fit is not None:
                        state["center_fit"] = fit["center"]
                        state["T"] = np.diag([1.0 / fit["a"], 1.0 / fit["b"]]) @ fit["R"].T
                        state["ellipse_cache"] = (fit["ex"], fit["ey"])
                        state["have_transform"] = True
                state["last_fit"] = t

            # 绘图
            if (t - last_draw) >= 1.0 / params.draw_hz:
                fig.canvas.restore_region(bg0)
                fig.canvas.restore_region(bg1)

                if phase == "ellipse":
                    sc0.set_offsets(np.c_[state["xs"], state["ys"]] if state["xs"] else np.empty((0, 2)))
                    ax0.draw_artist(sc0)
                    ell_line.set_data(*state["ellipse_cache"] if state["ellipse_cache"] else ([], []))
                    ax0.draw_artist(ell_line)
                    sc1.set_offsets(np.c_[state["xu"], state["yu"]] if state["xu"] else np.empty((0, 2)))
                    ax1.draw_artist(sc1)
                    if state["latest_z"]:
                        zx, zy = state["latest_z"]
                        radius_line.set_data([0.0, zx], [0.0, zy])
                        ang = float(np.arctan2(zy, zx))
                        angle_text.set_text(
                            f"angle: {ang:+.3f} rad  ({np.degrees(ang):+.1f}°)\nmove {name}, press 'n' to advance"
                        )
                    else:
                        radius_line.set_data([], [])
                        angle_text.set_text("(waiting for fit)")
                else:
                    sc0.set_offsets(np.empty((0, 2)))
                    ax0.draw_artist(sc0)
                    ell_line.set_data([], [])
                    ax0.draw_artist(ell_line)
                    if state["latest_z"]:
                        zx, zy = state["latest_z"]
                        sc1.set_offsets([[zx, zy]])
                        radius_line.set_data([0.0, zx], [0.0, zy])
                        ang = float(np.arctan2(zy, zx))
                        angle_text.set_text(
                            f"Zero pose for {name}\nangle: {ang:+.3f} rad\nsamples: {len(zero_samples)}/{params.sample_count}\nhold still, press 'n'"
                        )
                    else:
                        sc1.set_offsets(np.empty((0, 2)))
                        radius_line.set_data([], [])
                        angle_text.set_text("(waiting for data)")
                    ax1.draw_artist(sc1)

                ax1.draw_artist(radius_line)
                ax1.draw_artist(angle_text)
                fig.canvas.blit(ax0.bbox)
                fig.canvas.blit(ax1.bbox)
                fig.canvas.flush_events()
                last_draw = t

            plt.pause(0.001)

    finally:
        plt.close(fig)
