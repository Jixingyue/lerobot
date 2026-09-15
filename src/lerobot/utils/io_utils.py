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
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]


def load_json(fpath: Path) -> Any:
    """从 JSON 文件加载数据。

    参数：
        fpath (Path)：JSON 文件的路径。

    返回值：
        Any：从 JSON 文件加载的数据。
    """
    with open(fpath, encoding="utf-8") as f:
        return json.load(f)


def write_json(data: JsonLike, fpath: Path) -> None:
    """将可 JSON 序列化的数据写入文件。

    如果父目录不存在则创建。

    参数：
        data：要写入的可 JSON 序列化数据。
        fpath (Path)：输出 JSON 文件的路径。
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_video(video_path: str | Path, stacked_frames: list, fps: int) -> None:
    """使用 libx264 将一系列 RGB 帧写入 MP4 视频文件。

    参数：
        video_path：输出文件路径。
        stacked_frames：HWC uint8 numpy 数组（RGB）列表。
        fps：输出视频的每秒帧数。
    """
    from .import_utils import require_package

    require_package("av", extra="av-dep")
    import av

    with av.open(str(video_path), mode="w") as container:
        orig_height, orig_width = stacked_frames[0].shape[:2]
        # yuv420p 要求尺寸为偶数；必要时裁剪一个像素
        height = orig_height if orig_height % 2 == 0 else orig_height - 1
        width = orig_width if orig_width % 2 == 0 else orig_width - 1
        if height != orig_height or width != orig_width:
            logger.warning(
                "Frame dimensions %dx%d are not even; cropping to %dx%d for yuv420p compatibility.",
                orig_width,
                orig_height,
                width,
                height,
            )
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_array in stacked_frames:
            if height != orig_height or width != orig_width:
                frame_array = frame_array[:height, :width]
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:
    """
    从 `fpath` 加载 JSON 数据，并递归地用对应的值填充 `obj`
    （严格匹配结构和类型）。
    `obj` 中的元组在 JSON 数据中应为列表，
    加载时会被转换回元组。
    """
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    def _deserialize(target, source):
        """
        用 `source` 中的数据递归覆盖 `target` 中的结构，
        并对结构和类型进行严格检查。
        返回更新后的 `target`（对元组尤其重要）。
        """

        # 如果目标是字典，source 也必须是字典。
        if isinstance(target, dict):
            if not isinstance(source, dict):
                raise TypeError(f"Type mismatch: expected dict, got {type(source)}")

            # 检查两者的键集合是否完全一致。
            if target.keys() != source.keys():
                raise ValueError(
                    f"Dictionary keys do not match.\nExpected: {target.keys()}, got: {source.keys()}"
                )

            # 递归更新每个键。
            for k in target:
                target[k] = _deserialize(target[k], source[k])

            return target

        # 如果目标是列表，source 也必须是列表。
        elif isinstance(target, list):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list, got {type(source)}")

            # 检查长度
            if len(target) != len(source):
                raise ValueError(f"List length mismatch: expected {len(target)}, got {len(source)}")

            # 递归更新每个元素。
            for i in range(len(target)):
                target[i] = _deserialize(target[i], source[i])

            return target

        # 如果目标是元组，JSON 中的 source 必须是列表，
        # 我们会将其转换回元组。
        elif isinstance(target, tuple):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list (for tuple), got {type(source)}")

            if len(target) != len(source):
                raise ValueError(f"Tuple length mismatch: expected {len(target)}, got {len(source)}")

            # 转换每个元素，组成新的元组。
            converted_items = []
            for t_item, s_item in zip(target, source, strict=False):
                converted_items.append(_deserialize(t_item, s_item))

            # 返回一个全新的元组（Python 中元组不可变）。
            return tuple(converted_items)

        # 否则，处理的是"原始值"（int、float、str、bool、None）。
        else:
            # 检查类型是否完全一致。若要求 1:1 匹配，则：
            if type(target) is not type(source):
                raise TypeError(f"Type mismatch: expected {type(target)}, got {type(source)}")
            return source

    # 执行就地/递归反序列化
    updated_obj = _deserialize(obj, data)
    return updated_obj
