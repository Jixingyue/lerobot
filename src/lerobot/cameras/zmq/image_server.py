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
通过 ZMQ 流式传输相机图像。
使用 lerobot 的 OpenCVCamera 进行捕获，将图像编码为 base64 并通过 ZMQ 发送。
"""

import base64
import contextlib
import json
import logging
import threading
import time
from collections import deque

import cv2
import numpy as np
import zmq

from ..configs import ColorMode
from ..opencv import OpenCVCamera, OpenCVCameraConfig

logger = logging.getLogger(__name__)


def encode_image(image: np.ndarray, quality: int = 80) -> str:
    """将 RGB 图像编码为 base64 JPEG 字符串。"""
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode("utf-8")


class CameraCaptureThread:
    """持续从相机捕获并编码帧的后台线程。"""

    def __init__(self, camera: OpenCVCamera, name: str):
        self.camera = camera
        self.name = name
        self.latest_encoded: str | None = None  # 预编码的 base64 JPEG
        self.latest_timestamp: float = 0.0
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self):
        """启动捕获线程。"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """停止捕获线程。"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _capture_loop(self):
        """以相机的原生速率持续捕获并编码帧。"""
        while self.running:
            try:
                frame = self.camera.read()  # 以相机的原生速率阻塞
                timestamp = time.time()
                # 在捕获线程中立即编码（这是耗时的部分）
                encoded = encode_image(frame)
                with self.frame_lock:
                    self.latest_encoded = encoded
                    self.latest_timestamp = timestamp
            except Exception as e:
                logger.warning(f"Camera {self.name} capture error: {e}")
                time.sleep(0.01)

    def get_latest(self) -> tuple[str | None, float]:
        """获取最新的编码帧及其时间戳。"""
        with self.frame_lock:
            return self.latest_encoded, self.latest_timestamp


class ImageServer:
    def __init__(self, config: dict, port: int = 5555):
        # fps 控制发布循环的速率（帧通过 ZMQ 发送的频率），而不是相机的捕获速率
        self.fps = config.get("fps", 30)
        self.cameras: dict[str, OpenCVCamera] = {}
        self.capture_threads: dict[str, CameraCaptureThread] = {}

        for name, cfg in config.get("cameras", {}).items():
            shape = cfg.get("shape", [480, 640])
            cam_config = OpenCVCameraConfig(
                index_or_path=cfg.get("device_id", 0),
                fps=self.fps,
                width=shape[1],
                height=shape[0],
                fourcc=cfg.get("fourcc", "MJPG"),
                color_mode=ColorMode.RGB,
            )
            camera = OpenCVCamera(cam_config)
            camera.connect()
            self.cameras[name] = camera
            logger.info(f"Camera {name}: {shape[1]}x{shape[0]}")

            # 为该相机创建捕获线程
            capture_thread = CameraCaptureThread(camera, name)
            self.capture_threads[name] = capture_thread

        # ZMQ PUB 套接字
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")

        logger.info(f"ImageServer running on port {port}")

    def run(self):
        frame_count = 0
        frame_times = deque(maxlen=60)
        last_published_ts: dict[str, float] = {}

        # 启动所有捕获线程
        for capture_thread in self.capture_threads.values():
            capture_thread.start()

        # 等待首批帧被捕获并编码
        logger.info("Waiting for cameras to start capturing...")
        for name, capture_thread in self.capture_threads.items():
            while capture_thread.get_latest()[0] is None:
                time.sleep(0.01)
            logger.info(f"Camera {name} ready (capture + encode in background)")

        try:
            while True:
                t0 = time.time()

                # 构建消息
                message = {"timestamps": {}, "images": {}}
                for name, capture_thread in self.capture_threads.items():
                    encoded, timestamp = capture_thread.get_latest()
                    if encoded is not None and timestamp > last_published_ts.get(name, 0.0):
                        message["timestamps"][name] = timestamp
                        message["images"][name] = encoded
                        last_published_ts[name] = timestamp

                # 以 JSON 字符串发送（缓冲区满时抑制异常）
                with contextlib.suppress(zmq.Again):
                    self.socket.send_string(json.dumps(message), zmq.NOBLOCK)

                frame_count += 1
                frame_times.append(time.time() - t0)

                if frame_count % 60 == 0:
                    logger.debug(f"FPS: {len(frame_times) / sum(frame_times):.1f}")

                sleep = (1.0 / self.fps) - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            pass
        finally:
            for capture_thread in self.capture_threads.values():
                capture_thread.stop()
            for cam in self.cameras.values():
                cam.disconnect()
            self.socket.close()
            self.context.term()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {"fps": 30, "cameras": {"head_camera": {"device_id": 4, "shape": [480, 640]}}}
    ImageServer(config, port=5555).run()
