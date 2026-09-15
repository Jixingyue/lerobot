#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
Wall-X 工具函数。

包含用于 Wall-X 跨本体机器人控制模型的数据处理工具、文本格式化函数和辅助类。
"""

import random
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch

from lerobot.utils.import_utils import _transformers_available, _wallx_deps_available

if TYPE_CHECKING or _transformers_available:
    from transformers import BatchFeature
else:
    BatchFeature = None

if TYPE_CHECKING or _wallx_deps_available:
    from qwen_vl_utils.vision_process import smart_resize
else:
    smart_resize = None

from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_functional

from lerobot.utils.constants import OBS_IMAGES

from .constant import (
    CAMERA_NAME_MAPPING,
    IMAGE_FACTOR,
    MAX_PIXELS,
    MIN_PIXELS,
    RESOLUTION,
)


def preprocesser_call(
    processor,
    images: list | Any | None = None,
    text: str | list[str] | None = None,
    videos: list | Any | None = None,
    device: torch.device | str | None = None,
    padding: bool | str = False,
    truncation: bool | None = None,
    max_length: int | None = None,
    return_tensors: str = "pt",
    target_spans: list[list[tuple[int, int]]] | None = None,
) -> BatchFeature:
    """Wall-X 模型的统一预处理函数，处理文本、图像和视频输入。

    将输入处理为适合多模态 transformer 模型的格式，包括：
    - 文本分词和特殊 token 处理
    - 通过图像处理器处理图像/视频
    - 注意力掩码和标签生成
    - 填充和截断处理

    Args:
        processor: 包含分词器和图像处理器的多模态处理器
        images: 输入图像（PIL、numpy 数组或 torch 张量）
        text: 要分词的文本或文本列表
        videos: 输入视频（numpy 数组或 torch 张量）
        device: 运行图像/视频预处理的设备
        padding: 是否将序列填充到相同长度
        truncation: 是否截断超过 max_length 的序列
        max_length: 截断/填充的最大长度
        return_tensors: 返回张量的格式（'pt'、'np' 等）
        target_spans: 每个 prompt 中文本监督 token 对应的字符区间。
            ``None`` 保留原本仅针对动作的 assistant 标签路径；
            为某个样本传入空列表则显式禁用该样本的文本监督。

    Returns:
        包含处理后输入的 BatchFeature，其键包括：
        - input_ids: 分词后的文本
        - attention_mask: 文本的注意力掩码
        - pixel_values: 处理后的图像像素
        - pixel_values_videos: 处理后的视频帧
        - image_grid_thw: 提供给 LLM 的图像网格维度
        - video_grid_thw: 提供给 LLM 的视频网格维度
        - labels: 带掩码的训练标签
    """
    # 处理图像输入
    if images is not None and len(images) > 0:
        image_inputs = processor.image_processor(
            images=images,
            return_tensors=return_tensors,
            device=device,
        )
        image_grid_thw = image_inputs["image_grid_thw"]
    else:
        image_inputs = {}
        image_grid_thw = None

    # 处理视频输入
    if videos is not None:
        videos_inputs = processor.image_processor(
            videos=videos,
            return_tensors=return_tensors,
            device=device,
        )
        video_grid_thw = videos_inputs["video_grid_thw"]
    else:
        videos_inputs = {}
        video_grid_thw = None

    # 确保文本输入为列表格式
    if not isinstance(text, list):
        text = [text]

    if target_spans is not None:
        if len(target_spans) != len(text):
            raise ValueError("WALL-X needs one target-span list for each prompt.")
        target_spans = [list(spans) for spans in target_spans]

    def replace_placeholder_tokens(
        prompt: str,
        placeholder: str,
        token_count: int,
        spans: list[tuple[int, int]] | None,
    ) -> tuple[str, list[tuple[int, int]] | None]:
        position = prompt.find(placeholder)
        if position < 0:
            return prompt, spans
        replacement = "<|placeholder|>" * token_count
        placeholder_end = position + len(placeholder)
        if spans is not None:
            for start, end in spans:
                if start < placeholder_end and end > position:
                    raise ValueError(
                        "WALL-X text-supervision spans cannot contain image or video placeholders."
                    )
            # 分词器接收的是最终的 prompt——每个临时的 ``<|placeholder|>`` 之后
            # 都会被还原为 ``placeholder``。因此按最终的长度变化来平移 spans，
            # 而不是按临时展开后的长度变化。
            delta = len(placeholder) * (token_count - 1)
            spans = [
                (
                    start + (delta if start >= placeholder_end else 0),
                    end + (delta if end >= placeholder_end else 0),
                )
                for start, end in spans
            ]
        return prompt.replace(placeholder, replacement, 1), spans

    # 处理文本中的图像占位符 token。
    if image_grid_thw is not None:
        merge_length = processor.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while "<|image_pad|>" in text[i]:
                # 添加边界检查以避免索引溢出
                if index >= len(image_grid_thw):
                    print(
                        f"Warning: Number of image placeholders ({index + 1}) "
                        f"exceeds actual images ({len(image_grid_thw)}), "
                        f"skipping remaining placeholder processing"
                    )
                    break
                # 用实际的 token 数量替换图像占位符
                token_count = (image_grid_thw[index].prod() // merge_length).item()
                updated_text, updated_spans = replace_placeholder_tokens(
                    text[i],
                    "<|image_pad|>",
                    token_count,
                    target_spans[i] if target_spans is not None else None,
                )
                text[i] = updated_text
                if target_spans is not None:
                    target_spans[i] = updated_spans
                index += 1
            text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

    # 处理文本中的视频占位符 token
    if video_grid_thw is not None:
        merge_length = processor.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while "<|video_pad|>" in text[i]:
                # 用实际的 token 数量替换视频占位符
                token_count = (video_grid_thw[index].prod() // merge_length).item()
                updated_text, updated_spans = replace_placeholder_tokens(
                    text[i],
                    "<|video_pad|>",
                    token_count,
                    target_spans[i] if target_spans is not None else None,
                )
                text[i] = updated_text
                if target_spans is not None:
                    target_spans[i] = updated_spans
                index += 1
            text[i] = text[i].replace("<|placeholder|>", "<|video_pad|>")

    # 对完整的输入文本进行分词
    text_inputs = processor.tokenizer(
        text,
        return_tensors=return_tensors,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_offsets_mapping=target_spans is not None,
    )

    # 获取用于生成标签的 pad token ID
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    labels = torch.full_like(text_inputs.input_ids, -100)
    if target_spans is not None:
        offsets = text_inputs.pop("offset_mapping")
        for row, spans in enumerate(target_spans):
            for start, end in spans:
                overlap = (
                    (offsets[row, :, 1] > start)
                    & (offsets[row, :, 0] < end)
                    & text_inputs.attention_mask[row].bool()
                )
                labels[row, overlap] = text_inputs.input_ids[row, overlap]
    else:
        # 保留原本仅针对动作的标签生成路径。
        assistant_marker = "<|im_start|>assistant\n"
        im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_tokens = processor.tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids

        for i in range(len(text)):
            assistant_regions = []
            parts = text[i].split(assistant_marker)
            num_left_pads = 0
            for token_id in text_inputs.input_ids[i]:
                if token_id == pad_token_id:
                    num_left_pads += 1
                else:
                    break
            current_pos = num_left_pads

            for j, part in enumerate(parts):
                part_tokens = processor.tokenizer(part, add_special_tokens=False).input_ids
                if j == 0:
                    current_pos += len(part_tokens)
                    continue
                for k in range(current_pos + 1, len(text_inputs.input_ids[i])):
                    if text_inputs.input_ids[i][k] == im_end_token_id:
                        assistant_regions.append((current_pos + len(assistant_tokens), k + 2))
                        break
                current_pos += len(part_tokens) + 3

            for start, end in assistant_regions:
                labels[i][start:end] = text_inputs.input_ids[i][start:end]

    # 在标签中掩码掉特殊动作 token
    action_token_id = processor.tokenizer.encode("<|action|>")[0]
    propri_token_id = processor.tokenizer.encode("<|propri|>")[0]
    labels[labels == action_token_id] = -100
    labels[labels == propri_token_id] = -100
    labels[labels == pad_token_id] = -100

    # 如果所有标签都无效，则将 labels 设为 None 以跳过交叉熵损失
    if (labels != -100).any().item():
        text_inputs["labels"] = labels
    else:
        text_inputs["labels"] = None

    return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})


def _wall_x_resize_dimensions(height: int, width: int) -> tuple[int, int, int, int]:
    """以 ``(H, W, H, W)`` 的形式返回 Wall-X 缩放的中间尺寸和最终尺寸。"""
    if RESOLUTION == -1:
        intermediate_height, intermediate_width = height, width
    elif width > height:
        intermediate_width = RESOLUTION
        intermediate_height = int(RESOLUTION * height / width)
    else:
        intermediate_height = RESOLUTION
        intermediate_width = int(RESOLUTION * width / height)

    resized_height, resized_width = smart_resize(
        intermediate_height,
        intermediate_width,
        factor=IMAGE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    return intermediate_height, intermediate_width, resized_height, resized_width


def _resize_wall_x_image_batch(images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """在不离开当前设备的情况下，对 BCHW 相机批次进行量化和缩放。"""
    if images.ndim != 4:
        raise ValueError(f"Wall-X images must be BCHW tensors, got shape {tuple(images.shape)}")

    original_height, original_width = images.shape[-2:]
    intermediate_height, intermediate_width, resized_height, resized_width = _wall_x_resize_dimensions(
        original_height, original_width
    )

    if images.is_floating_point():
        images = (images * 255).to(torch.uint8)
    elif images.dtype != torch.uint8:
        raise TypeError(f"Wall-X images must be floating point or uint8, got {images.dtype}")

    if images.shape[-2:] != (intermediate_height, intermediate_width):
        images = tv_functional.resize(
            images,
            [intermediate_height, intermediate_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    if images.shape[-2:] != (resized_height, resized_width):
        images = tv_functional.resize(
            images,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    return images, (original_height, original_width, resized_height, resized_width)


def prepare_wall_x_image_inputs(
    batch: dict[str, Any], image_keys: list[str]
) -> tuple[list[list[torch.Tensor]], dict[str, tuple[int, int, int, int]]]:
    """缩放每个相机批次，并恢复以样本为主序、相机为次序的排列。"""
    resized_by_key: dict[str, torch.Tensor] = {}
    dimensions_by_key: dict[str, tuple[int, int, int, int]] = {}
    for key in image_keys:
        resized_by_key[key], dimensions_by_key[key] = _resize_wall_x_image_batch(batch[key])

    batch_size = batch[image_keys[0]].shape[0]
    image_inputs = [[resized_by_key[key][index] for key in image_keys] for index in range(batch_size)]
    return image_inputs, dimensions_by_key


def process_grounding_points(
    text: str,
    orig_height: int,
    orig_width: int,
    resized_height: int,
    resized_width: int,
    model_type: str,
) -> str:
    """根据图像缩放处理文本中的定位点（grounding point）坐标。

    针对不同的模型类型（qwen2、qwen2_5），调整 <point> 标签中的坐标值，
    以匹配缩放后的图像尺寸。

    Args:
        text: 包含带坐标的 <point> 标签的输入文本
        orig_height: 原始图像高度
        orig_width: 原始图像宽度
        resized_height: 缩放后的图像高度
        resized_width: 缩放后的图像宽度
        model_type: 用于坐标处理的模型类型（'qwen2' 或 'qwen2_5'）

    Returns:
        坐标值经过调整的文本
    """
    # 用于匹配 <point> 标签及其内容的正则表达式
    point_pattern = re.compile(r"<point>(.*?)</point>")

    def process_match(match):
        """处理单个 point 匹配并调整坐标。"""
        coords_str = match.group(1)
        try:
            # 从字符串中提取坐标
            coords = list(map(int, re.findall(r"\d+", coords_str)))

            # 计算缩放比例因子
            scale_w = resized_width / orig_width
            scale_h = resized_height / orig_height

            if len(coords) == 2:
                x, y = coords
                if model_type == "qwen2_5":
                    # Qwen2.5 使用像素坐标
                    new_x = max(0, min(round(x * scale_w), resized_width - 1))
                    new_y = max(0, min(round(y * scale_h), resized_height - 1))
                elif model_type == "qwen2":
                    # Qwen2 归一化到 [0, 1000) 范围
                    new_x = max(0, min(999.999, (x / orig_width) * 1000))
                    new_y = max(0, min(999.999, (y / orig_height) * 1000))
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                coords = [new_x, new_y]

            elif len(coords) == 4:
                x1, y1, x2, y2 = coords
                if model_type == "qwen2_5":
                    new_x1 = max(0, min(round(x1 * scale_w), resized_width - 1))
                    new_y1 = max(0, min(round(y1 * scale_h), resized_height - 1))
                    new_x2 = max(0, min(round(x2 * scale_w), resized_width - 1))
                    new_y2 = max(0, min(round(y2 * scale_h), resized_height - 1))
                elif model_type == "qwen2":
                    new_x1 = max(0, min(999.999, (x1 / orig_width) * 1000))
                    new_y1 = max(0, min(999.999, (y1 / orig_height) * 1000))
                    new_x2 = max(0, min(999.999, (x2 / orig_width) * 1000))
                    new_y2 = max(0, min(999.999, (y2 / orig_height) * 1000))
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                coords = [new_x1, new_y1, new_x2, new_y2]

            # 返回处理后的 point 标签
            return f"<point>[{', '.join(map(str, coords))}]</point>"

        except (ValueError, TypeError):
            # 如果处理失败，则返回原始内容
            return match.group(0)

    # 替换所有匹配的 point 标签
    processed_text = point_pattern.sub(process_match, text)
    return processed_text


def get_frame_instruction(
    instruction_info: dict[str, Any],
    frame_idx: int | None = None,
    truncate_keys: list[str] | None = None,
) -> tuple[dict[str, Any], int | None]:
    """从指令字典中提取特定帧的指令。

    Args:
        instruction_info: 包含指令各组成部分的字典
        frame_idx: 当前帧索引
        truncate_keys: 一旦找到就会触发截断的键

    Returns:
        元组 (frame_instruction_dict, split_end_frame)
    """
    if truncate_keys is None:
        truncate_keys = [
            "subtask_generation",
            "distribute",
            "subtask_generation_zh",
            "distribute_zh",
        ]

    instruction_for_frame = {}
    split_end = None

    for key, value in instruction_info.items():
        if isinstance(value, dict):
            # 处理针对特定帧范围的指令
            for frame_range, frame_instruction in value.items():
                start_frame, end_frame = map(int, frame_range.split(" "))
                if start_frame <= frame_idx < end_frame or (start_frame == frame_idx):
                    instruction_for_frame[key] = frame_instruction
                    if truncate_keys is not None and split_end is None and key in truncate_keys:
                        split_end = end_frame + 1
                    break
        else:
            instruction_for_frame[key] = value

    return instruction_for_frame, split_end


def get_task_instruction(
    frame_instruction_info: dict[str, Any], priority_order: OrderedDict | None = None
) -> str:
    """使用按优先级采样的方式，从可用的指令字段构建任务指令。

    Args:
        frame_instruction_info: 包含指令字段的字典
        priority_order: 指定每个字段采样概率的 OrderedDict

    Returns:
        组合了各优先级成分的指令字符串
    """
    # 默认优先级设置
    default_priority_order = OrderedDict(
        {
            "subtask_generation": 0.25,
            "subtask_generation_zh": 0.25,
            "distribute": 0.25,
            "distribute_zh": 0.25,
        }
    )

    priority_order = OrderedDict(priority_order) if priority_order is not None else default_priority_order

    got_instruction = False
    task_instruction = ""

    # 根据优先级概率采样指令成分
    for key, prob in priority_order.items():
        if key in frame_instruction_info and frame_instruction_info[key] != "":
            if got_instruction and random.random() >= prob:
                continue

            task_instruction += f"\n{frame_instruction_info[key]}"
            got_instruction = True
            break

    # 如果没有找到任何优先级成分，则回退到基础指令
    if not got_instruction:
        task_instruction = frame_instruction_info.get("instruction", "")

    return task_instruction


def get_wallx_normal_text(
    instruction_info: dict[str, Any],
    action_chunk_size: int,
    frame_idx: int,
    priority_order: OrderedDict | None = None,
    img_keys: list[str] | None = None,
    generate_subtask_ratio: float = 0.0,
) -> tuple[str, bool]:
    """为 Wall-X 模型构建完整的多模态 prompt 文本。

    使用特殊 token 对输入进行格式化，包括：
    - 系统消息
    - 用户观测（带图像占位符）
    - 任务指令
    - 本体感知 prompt
    - 助手回复（带动作 token）

    Args:
        instruction_info: 包含指令各组成部分的字典
        action_chunk_size: 要生成的动作 token 数量
        frame_idx: 当前帧索引
        priority_order: 指令采样的优先级顺序
        img_keys: 图像键列表
        generate_subtask_ratio: 生成子任务而不是动作的概率

    Returns:
        元组 (formatted_prompt_text, is_subtask_generation)
    """
    # 用于格式化的特殊 token
    role_start_symbol = "<|im_start|>"
    role_end_symbol = "<|im_end|>"
    vision_start_symbol = "<|vision_start|>"
    vision_end_symbol = "<|vision_end|>"
    image_pad_symbol = "<|image_pad|>"
    propri_symbol = "<|propri|>"
    action_symbol = "<|action|>"
    action_fast_symbol = "<|action_fast|>"

    # 系统开场白
    prologue = f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"

    # 带观测的用户请求
    user_request = f"{role_start_symbol}user\nObservation:"
    if img_keys:
        img_keys = img_key_mapping(img_keys)
        for key in img_keys:
            user_request += f" {key}: {vision_start_symbol}{image_pad_symbol}{vision_end_symbol}"
    user_request += "\nInstruction:"

    # 获取特定帧的指令
    frame_instruction_info, _ = get_frame_instruction(instruction_info, frame_idx=frame_idx)

    generate_subtask = False
    priority_keys = ["subtask_generation", "distribute"]

    # 决定是生成子任务还是生成动作
    if (
        bool(set(frame_instruction_info.keys()) & set(priority_keys))
        and random.random() < generate_subtask_ratio
    ):
        # 生成子任务（等价于 VQA 任务）
        instruction = frame_instruction_info.get("instruction", "")
        text_prompt = "\nPredict the next action in language.\n"
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"

        # 从优先级键中找到输出指令
        for key in priority_keys:
            if key in frame_instruction_info:
                output_instruction = frame_instruction_info[key]
                break

        assistant_output = f"{role_start_symbol}assistant\n{output_instruction}\n{role_end_symbol}"
        generate_subtask = True
    else:
        # 生成动作
        instruction = get_task_instruction(frame_instruction_info, priority_order=priority_order)
        text_prompt = f"\nPredict the next action in robot action.\nProprioception: {propri_symbol}\n"
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
        assistant_output = f"{role_start_symbol}assistant\n{action_fast_symbol}{role_end_symbol}\n{action_symbol * action_chunk_size}"

    complete_text = prologue + user_message + assistant_output
    return complete_text, generate_subtask


def img_key_mapping(img_keys: list[str]) -> list[str]:
    """将图像键映射为相机名称。

    Args:
        img_keys: 图像键列表

    Returns:
        相机名称列表
    """
    processed_img_keys = []
    for key in img_keys:
        key = key.replace(OBS_IMAGES + ".", "")
        if key in CAMERA_NAME_MAPPING:
            key = CAMERA_NAME_MAPPING[key]
        else:
            key = key.replace("_", " ") if "view" in key else key + " view"
        processed_img_keys.append(key)
    return processed_img_keys


def get_action_tokens(normalized_actions: torch.Tensor | list, action_tokenizer) -> list[list[str]]:
    """将归一化后的动作转换为动作 token 字符串。

    Args:
        normalized_actions: 归一化后的动作数组/张量
        action_tokenizer: 用于将动作转换为 token 的分词器

    Returns:
        每个样本对应的动作 token 字符串列表组成的列表
    """
    if isinstance(normalized_actions, torch.Tensor):
        normalized_actions = normalized_actions.cpu().numpy()

    all_action_tokens = []
    for i in range(len(normalized_actions)):
        if isinstance(normalized_actions[i], torch.Tensor):
            normalized_actions[i] = normalized_actions[i].cpu().numpy()

        token_id = action_tokenizer(normalized_actions[i])
        action_tokens = [f"<|action_token_{j}|>" for j in token_id[0]]
        all_action_tokens.append(action_tokens)

    return all_action_tokens


def pad_action_token_strs(
    actions_token_lists: list[list[str]],
    pad_token: str = "<|endoftext|>",  # nosec B107
) -> list[str]:
    """将动作 token 列表填充到相同长度，并拼接成字符串。

    Args:
        actions_token_lists: 每个样本的动作 token 列表
        pad_token: 用于填充的 token

    Returns:
        填充后的动作 token 字符串列表
    """
    max_len = max(len(tokens) for tokens in actions_token_lists)
    padded_action_strs = []

    for tokens in actions_token_lists:
        padded_tokens = tokens + ["<|im_end|>\n"] + [pad_token] * (max_len - len(tokens))
        padded_action_strs.append("".join(padded_tokens))

    return padded_action_strs


def replace_action_token(
    text: list[str],
    norm_action: torch.Tensor | None,
    action_tokenizer,
    dof_masks: torch.Tensor | None = None,
) -> list[str]:
    """用实际的动作 token 替换文本中的动作占位符。

    Args:
        text: 带动作占位符的文本字符串列表
        norm_action: 归一化后的动作张量
        action_tokenizer: 用于将动作转换为 token 的分词器
        dof_masks: 自由度掩码

    Returns:
        动作 token 被替换后的文本字符串列表
    """
    if action_tokenizer is not None and norm_action is not None:
        # 根据块大小和 DOF 掩码提取动作
        norm_action = [action[:32, dof_masks[i, 0].bool()] for i, action in enumerate(norm_action)]

        # 转换为动作 token 并进行填充
        actions_fast_tokens = get_action_tokens(norm_action, action_tokenizer)
        actions_fast_token_strs = pad_action_token_strs(actions_fast_tokens)

        # 用实际的 token 替换动作占位符
        actions_fast_token_idx = 0
        for i in range(len(text)):
            if "<|action_fast|>" in text[i]:
                text[i] = text[i].replace(
                    "<|action_fast|><|im_end|>\n",
                    actions_fast_token_strs[actions_fast_token_idx],
                )
                actions_fast_token_idx += 1

        # 移除剩余的动作占位符
        text = [t.replace("<|action|>", "") for t in text]
    else:
        # 当没有可用分词器时，移除动作占位符
        text = [t.replace("<|action_fast|><|im_end|>\n", "") for t in text]

    return text
