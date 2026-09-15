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
import logging
import multiprocessing
import queue
import threading
from pathlib import Path

import numpy as np
import PIL.Image
import torch

logger = logging.getLogger(__name__)


def safe_stop_image_writer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException:
            dataset = kwargs.get("dataset")
            writer = getattr(dataset, "writer", None) if dataset else None
            if writer is not None and writer.image_writer is not None:
                logger.warning("Waiting for image writer to terminate...")
                writer.image_writer.stop()
            raise

    return wrapper


def squeeze_single_channel(array: np.ndarray) -> np.ndarray:
    """去除首尾的单元素通道维度：``(1, H, W)`` / ``(H, W, 1)`` -> ``(H, W)``。

    与 ``array.squeeze()`` 不同，此函数只移除通道轴，绝不会移除大小为 1 的 ``H`` 或 ``W``。
    """
    if array.ndim == 3:
        if array.shape[0] == 1:
            return array[0]
        if array.shape[-1] == 1:
            return array[..., 0]
    return array


def image_array_to_pil_image(image_array: np.ndarray, range_check: bool = True) -> PIL.Image.Image:
    """将 NumPy 数组转换为 PIL 图像，并保留灰度图的精度。

    按形状划分的行为：

    - ``(H, W)`` 或 ``(1, H, W)`` / ``(H, W, 1)``：单通道灰度图。
      使用匹配的 PIL 模式（``I;16`` / ``F``）保留原生 dtype。
      这是原始深度图所走的路径（不做缩放、裁剪或降精度转换）。
    - ``(3, H, W)`` / ``(H, W, 3)``：RGB。通道在前的输入会被转置为
      通道在后。``[0, 1]`` 范围内的浮点输入会被缩放到 ``uint8``
      （既有行为，由 ``range_check`` 控制）。

    其他形状/通道数会抛出 ``NotImplementedError`` 或
    ``ValueError``。
    """
    # TODO(CarolinePascal): 4 维 RGB-D 图像
    if image_array.ndim not in (2, 3):
        raise ValueError(f"The array has {image_array.ndim} dimensions, but 2 or 3 is expected for an image.")

    # 将 3D 单通道输入压缩为 2D，这样无论调用方输出的是
    # (H, W)、(1, H, W) 还是 (H, W, 1)，深度图都能正常工作。
    image_array = squeeze_single_channel(image_array)

    if image_array.ndim == 2:
        if image_array.dtype not in [np.uint16, np.float32]:
            raise ValueError(
                f"Unsupported single-channel image dtype: {image_array.dtype}. "
                f"Supported dtypes: {sorted(str(d) for d in [np.uint16, np.float32])}."
            )
        return PIL.Image.fromarray(np.ascontiguousarray(image_array))

    # 3D 路径：必须是 RGB（3 通道），通道在前或通道在后均可。
    if image_array.shape[0] == 3:
        # 从 pytorch 约定 (C, H, W) 转置为 (H, W, C)
        image_array = image_array.transpose(1, 2, 0)

    elif image_array.shape[-1] != 3:
        raise NotImplementedError(
            f"The image has {image_array.shape[-1]} channels, but 3 is required for now."
        )

    if image_array.dtype != np.uint8:
        if range_check:
            max_ = image_array.max().item()
            min_ = image_array.min().item()
            if max_ > 1.0 or min_ < 0.0:
                raise ValueError(
                    "The image data type is float, which requires values in the range [0.0, 1.0]. "
                    f"However, the provided range is [{min_}, {max_}]. Please adjust the range or "
                    "provide a uint8 image with values in the range [0, 255]."
                )

        image_array = (image_array * 255).astype(np.uint8)

    return PIL.Image.fromarray(image_array)


def save_kwargs_for_path(fpath: Path, compress_level: int) -> dict:
    """为 :meth:`PIL.Image.Image.save` 选择合适的格式特定 kwargs。

    PNG 使用 ``compress_level``（0-9，zlib）。TIFF 使用 ``compression``（raw）保存无损原始深度图。
    """
    suffix = Path(fpath).suffix.lower()
    if suffix == ".png":
        return {"compress_level": compress_level}
    if suffix in (".tif", ".tiff"):
        return {"compression": "raw"}
    else:
        raise ValueError(f"Unsupported image file extension: {suffix}")


def write_image(image: np.ndarray | PIL.Image.Image, fpath: Path, compress_level: int = 1):
    """
    将 NumPy 数组或 PIL 图像保存到文件。

    此函数同时处理 NumPy 数组和 PIL 图像对象，会在保存前将
    前者转换为 PIL 图像。它包含对保存操作的错误处理。
    输出格式根据 *fpath* 扩展名推断：``.png`` → 带 ``compress_level``
    的 PNG，``.tiff`` / ``.tif`` → 无损原始深度图（TIFF）。

    Args:
        image (np.ndarray | PIL.Image.Image): 要保存的图像数据。
        fpath (Path): 图像的目标文件路径。
        compress_level (int, optional): 保存图像的压缩级别，
            与 PIL.Image.save() 所用的一致。默认为 1。
            有关默认值选择依据的更多细节，
            请参考：https://github.com/huggingface/lerobot/pull/2135。

    Raises:
        TypeError: 如果输入的 'image' 不是 NumPy 数组或
            PIL.Image.Image 对象。

    Side Effects:
        如果图像写入过程因任何原因失败，则记录错误消息。
    """
    try:
        if isinstance(image, np.ndarray):
            img = image_array_to_pil_image(image)
        elif isinstance(image, PIL.Image.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        img.save(fpath, **save_kwargs_for_path(fpath, compress_level))
    except Exception as e:
        logger.error("Error writing image %s: %s", fpath, e)


def worker_thread_loop(queue: queue.Queue):
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break
        image_array, fpath, compress_level = item
        write_image(image_array, fpath, compress_level)
        queue.task_done()


def worker_process(queue: queue.Queue, num_threads: int):
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker_thread_loop, args=(queue,))
        t.daemon = True
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


class AsyncImageWriter:
    """
    此类抽象了进程或/和线程的初始化，用于异步将图像保存到磁盘，
    这对于以高帧率控制机器人和记录数据至关重要。

    当 `num_processes=0` 时，它会创建大小为 `num_threads` 的线程池。
    当 `num_processes>0` 时，它会创建大小为 `num_processes` 的进程池，
    其中每个子进程都会启动自己大小为 `num_threads` 的线程池。

    最优的进程数和线程数取决于你的计算机性能。
    我们建议每个相机使用 4 个线程且进程数为 0。如果 fps 不稳定，
    可以尝试增加或减少线程数。如果仍不稳定，可以尝试使用 1 个子进程或更多。
    """

    def __init__(self, num_processes: int = 0, num_threads: int = 1):
        self.num_processes = num_processes
        self.num_threads = num_threads
        self.queue = None
        self.threads = []
        self.processes = []
        self._stopped = False

        if num_threads <= 0 and num_processes <= 0:
            raise ValueError("Number of threads and processes must be greater than zero.")

        if self.num_processes == 0:
            # 使用多线程
            self.queue = queue.Queue()
            for _ in range(self.num_threads):
                t = threading.Thread(target=worker_thread_loop, args=(self.queue,))
                t.daemon = True
                t.start()
                self.threads.append(t)
        else:
            # 使用多进程
            self.queue = multiprocessing.JoinableQueue()
            for _ in range(self.num_processes):
                p = multiprocessing.Process(target=worker_process, args=(self.queue, self.num_threads))
                p.daemon = True
                p.start()
                self.processes.append(p)

    def save_image(
        self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path, compress_level: int = 1
    ):
        if isinstance(image, torch.Tensor):
            # 将张量转换为 numpy 数组，以减少主进程耗时
            image = image.cpu().numpy()
        self.queue.put((image, fpath, compress_level))

    def wait_until_done(self):
        self.queue.join()

    def stop(self):
        if self._stopped:
            return

        if self.num_processes == 0:
            for _ in self.threads:
                self.queue.put(None)
            for t in self.threads:
                t.join()
        else:
            num_nones = self.num_processes * self.num_threads
            for _ in range(num_nones):
                self.queue.put(None)
            for p in self.processes:
                p.join()
                if p.is_alive():
                    p.terminate()
            self.queue.close()
            self.queue.join_thread()

        self._stopped = True
