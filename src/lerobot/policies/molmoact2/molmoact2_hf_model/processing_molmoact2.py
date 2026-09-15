# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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
MolmoAct2 的处理器类。
"""

import numpy as np
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.video_utils import VideoInput

from .image_processing_molmoact2 import MolmoAct2ImageProcessor, MolmoAct2ImagesKwargs
from .video_processing_molmoact2 import MolmoAct2VideoProcessor, MolmoAct2VideoProcessorKwargs

logger = logging.get_logger(__name__)


# 特殊 token，由于预处理器会使用它们，我们使用的任何分词器中都必须包含这些 token
IMAGE_PATCH_TOKEN = "<im_patch>"  # nosec B105  # 插入高分辨率 token 的位置
IMAGE_LOW_RES_TOKEN = "<im_low>"  # nosec B105  # 插入低分辨率 token 的位置
IM_START_TOKEN = "<im_start>"  # nosec B105
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"  # nosec B105
FRAME_START_TOKEN = "<frame_start>"  # nosec B105
IM_END_TOKEN = "<im_end>"  # nosec B105
FRAME_END_TOKEN = "<frame_end>"  # nosec B105
IM_COL_TOKEN = "<im_col>"  # nosec B105
IMAGE_PROMPT = "<|image|>"
VIDEO_PROMPT = "<|video|>"

IMAGE_TOKENS = [
    IMAGE_PATCH_TOKEN,
    IM_COL_TOKEN,
    IM_START_TOKEN,
    LOW_RES_IMAGE_START_TOKEN,
    FRAME_START_TOKEN,
    IM_END_TOKEN,
    FRAME_END_TOKEN,
    IMAGE_LOW_RES_TOKEN,
]


class MolmoAct2ProcessorKwargs(ProcessingKwargs, total=False):
    """MolmoAct2 处理器的 kwargs"""

    images_kwargs: MolmoAct2ImagesKwargs
    videos_kwargs: MolmoAct2VideoProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class MolmoAct2Processor(ProcessorMixin):
    attributes = ["image_processor", "video_processor", "tokenizer"]
    optional_attributes = [
        "chat_template",
        "time_mode",
        "image_use_col_tokens",
        "use_single_crop_col_tokens",
        "use_single_crop_start_token",
        "video_use_col_tokens",
        "use_frame_special_tokens",
    ]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor: MolmoAct2ImageProcessor = None,
        video_processor: MolmoAct2VideoProcessor = None,
        tokenizer: AutoTokenizer = None,
        chat_template: str | None = None,
        image_use_col_tokens: bool | None = True,
        use_single_crop_col_tokens: bool | None = None,
        use_single_crop_start_token: bool | None = True,
        video_use_col_tokens: bool | None = False,
        use_frame_special_tokens: bool | None = True,
        **kwargs,
    ) -> None:
        super().__init__(
            image_processor,
            video_processor,
            tokenizer,
            chat_template=chat_template,
        )
        self.image_use_col_tokens = image_use_col_tokens
        self.use_single_crop_col_tokens = use_single_crop_col_tokens
        self.use_single_crop_start_token = use_single_crop_start_token
        self.video_use_col_tokens = video_use_col_tokens
        self.use_frame_special_tokens = use_frame_special_tokens

        self.image_placeholder_token = IMAGE_PROMPT
        self.video_placeholder_token = VIDEO_PROMPT
        self.image_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in IMAGE_TOKENS]

    def get_image_tokens(self, image_grid: np.ndarray):
        resized_h, resized_w, height, width = image_grid
        if int(height) == 0 or int(width) == 0:
            per_row = np.full(resized_w, IMAGE_PATCH_TOKEN)
            use_single_crop_col_tokens = (
                self.image_use_col_tokens
                if self.use_single_crop_col_tokens is None
                else self.use_single_crop_col_tokens
            )
            if use_single_crop_col_tokens:
                per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
            joint = [
                [IM_START_TOKEN],
                np.tile(per_row, [resized_h]),
                [IM_END_TOKEN],
            ]
            return np.concatenate(joint)
        per_row = np.full(width, IMAGE_PATCH_TOKEN)
        if self.image_use_col_tokens:
            per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
        joint = [
            [IM_START_TOKEN],
            np.tile(per_row, [height]),
            [IM_END_TOKEN],
        ]
        per_row = np.full(resized_w, IMAGE_PATCH_TOKEN)
        use_single_crop_col_tokens = (
            self.image_use_col_tokens
            if self.use_single_crop_col_tokens is None
            else self.use_single_crop_col_tokens
        )
        image_start_token = LOW_RES_IMAGE_START_TOKEN if self.use_single_crop_start_token else IM_START_TOKEN
        if use_single_crop_col_tokens:
            per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
        joint = [
            [image_start_token],
            np.tile(per_row, [resized_h]),
            [IM_END_TOKEN],
        ] + joint

        return np.concatenate(joint)

    def get_video_string(
        self,
        video_grid: np.ndarray,
        timestamps: np.ndarray,
    ):
        if self.use_frame_special_tokens:
            start_token_id = FRAME_START_TOKEN
            end_token_id = FRAME_END_TOKEN
        else:
            start_token_id = IM_START_TOKEN
            end_token_id = IM_END_TOKEN

        num_frames, h, w = video_grid
        video_string: str = ""
        for frame_idx, frame_time in enumerate(timestamps):
            # `per-frame-compact` 时间模式
            prev_space = " " if frame_idx > 0 else ""
            frame_prefix = prev_space + f"{frame_time:.1f} "  # 在图像 token 前后显式添加空白

            video_string += frame_prefix
            per_row = np.full(w, IMAGE_PATCH_TOKEN)
            if self.video_use_col_tokens:
                per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
            extra_tokens = np.tile(per_row, [h])
            video_tokens = [
                [start_token_id],
                extra_tokens,
                [end_token_id],
            ]
            video_string += "".join(np.concatenate(video_tokens, 0))

        return video_string

    def insert_bos(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        bos_token_id: int,
        pad_token_id: int,
    ):
        """
        参数：
            input_ids: 左填充的 [B, S] 数组
            attention_mask: [B, S] 数组（0 表示填充，1 表示有效）
            bos_token_id: int
            pad_token_id: int
        返回：
            input_ids_out: [B, S] 或 [B, S+1] 数组，必要时插入 bos
            attention_mask_out: 与 input_ids_out 形状相同
        """

        need_to_expand = len(input_ids.shape) == 1
        if need_to_expand:
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]

        B, S = input_ids.shape  # noqa: N806

        # 处理零长度序列
        if S == 0:
            new_input_ids = np.full((B, 1), bos_token_id, dtype=input_ids.dtype)
            new_attention_mask = np.ones((B, 1), dtype=attention_mask.dtype)
            if need_to_expand:
                new_input_ids = new_input_ids[0]
                new_attention_mask = new_attention_mask[0]
            return new_input_ids, new_attention_mask

        first_valid_index = (attention_mask == 1).argmax(axis=-1)  # [B]
        bos_already_present = np.all(input_ids[np.arange(B), first_valid_index] == bos_token_id)

        if bos_already_present:
            if need_to_expand:
                input_ids = input_ids[0]
                attention_mask = attention_mask[0]
            return input_ids, attention_mask
        else:
            new_input_ids = np.full((B, S + 1), pad_token_id, dtype=input_ids.dtype)
            new_attention_mask = np.zeros((B, S + 1), dtype=attention_mask.dtype)

            src_idx = np.tile(np.arange(S), (B, 1))  # [B, S]
            valid_mask = src_idx >= first_valid_index[:, None]  # [B, S]
            tgt_idx = src_idx + 1  # 右移
            batch_idx = np.tile(np.arange(B)[:, None], (1, S))  # [B, S]

            # 展平有效位置
            flat_vals = input_ids[valid_mask]
            flat_batch = batch_idx[valid_mask]
            flat_tgt = tgt_idx[valid_mask]

            new_input_ids[flat_batch, flat_tgt] = flat_vals
            new_attention_mask[flat_batch, flat_tgt] = 1

            insert_pos = first_valid_index
            new_input_ids[np.arange(B), insert_pos] = bos_token_id
            new_attention_mask[np.arange(B), insert_pos] = 1

            if need_to_expand:
                new_input_ids = new_input_ids[0]
                new_attention_mask = new_attention_mask[0]

            return new_input_ids, new_attention_mask

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        **kwargs: Unpack[MolmoAct2ProcessorKwargs],
    ) -> BatchFeature:
        """

        参数：
            text (`str`, `list[str]`, `list[list[str]]`)：
                要编码的序列或序列批次。每个序列可以是字符串或字符串列表
                （预分词的字符串）。如果序列以字符串列表（预分词）的形式提供，必须设置
                `is_split_into_words=True`（以消除与序列批次的歧义）。
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`)：
                要准备的图像或图像批次。每张图像可以是 PIL 图像、NumPy 数组或 PyTorch
                张量。支持通道优先和通道最后两种格式。
            videos (`dict[str, Any]` 或 `list[dict[str, Any]]`)：
                要准备的视频或视频批次。每个视频可以是包含以下键的字典：
                - `"frames"`：形状为 (T, H, W, 3) 的 `np.ndarray`
                - `"timestamps"`：形状为 (T,) 的 `np.ndarray`
                - `"sampled_fps"`：`float`（可选）
                - `"sampling_augmentation"`：`str`（可选）
            return_tensors (`str` 或 [`~utils.TensorType`]，*可选*)：
                如果设置，将返回特定框架的张量。可接受的值为：
                - `'tf'`：返回 TensorFlow `tf.constant` 对象。
                - `'pt'`：返回 PyTorch `torch.Tensor` 对象。
                - `'np'`：返回 NumPy `np.ndarray` 对象。
                - `'jax'`：返回 JAX `jnp.ndarray` 对象。

        返回：
            `BatchFeature`：包含以下字段的 [`BatchFeature`]：
            - **input_ids** -- 要输入模型的 token id 列表。当 `text` 不为 `None` 时返回。
            - **attention_mask** -- 指定模型应关注哪些 token 的索引列表（当
              `return_attention_mask=True`，或 *"attention_mask"* 在 `self.model_input_names` 中且 `text` 不为 `None` 时返回）。
            - **pixel_values** -- 要输入模型的像素值。当 `images` 不为 `None` 时返回。
            - **image_token_pooling** -- `image_tokens` 中每个 token 做池化时所用 `image_grids` 中块的索引。
              当 `images` 不为 `None` 时返回。
            - **image_grids** -- 图像网格。当 `images` 不为 `None` 时返回。
            - **image_num_crops** -- 每张图像的裁剪块数量。当 `images` 不为 `None` 时返回。
            - **pixel_values_videos** -- 要输入模型的视频像素值。当 `videos` 不为 `None` 时返回。
            - **video_token_pooling** -- `video_tokens` 中每个 token 做池化时所用 `video_grids` 中块的索引。
              当 `videos` 不为 `None` 时返回。
            - **video_grids** -- 视频网格。当 `videos` 不为 `None` 时返回。
        """

        output_kwargs = self._merge_kwargs(
            MolmoAct2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            image_grids = image_inputs["image_grids"]
        else:
            image_inputs = {}
            image_grids = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grids = videos_inputs["video_grids"]
            # 如果用户没有请求视频元数据，则将其弹出
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grids = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # 下面的代码会就地修改 text

        if image_grids is not None:
            index = 0
            for i in range(len(text)):
                num_images = text[i].count(self.image_placeholder_token)
                image_grids_i = image_grids[index : index + num_images]
                for image_grid in image_grids_i:
                    image_tokens = self.get_image_tokens(image_grid)
                    image_string = "".join(image_tokens)
                    text[i] = text[i].replace(self.image_placeholder_token, image_string, 1)
                index += num_images

        if video_grids is not None:
            index = 0
            for i in range(len(text)):
                num_videos = text[i].count(self.video_placeholder_token)
                assert num_videos in {0, 1}, "At most one video is supported for now"
                video_grids_i = video_grids[index : index + num_videos]
                metadata_i = video_metadata[index : index + num_videos]
                for video_grid, metadata in zip(video_grids_i, metadata_i, strict=False):
                    video_string = self.get_video_string(
                        video_grid,
                        metadata.timestamps,
                    )
                    text[i] = text[i].replace(self.video_placeholder_token, video_string, 1)
                index += num_videos

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)

        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        input_ids, attention_mask = self.insert_bos(
            input_ids, attention_mask, bos, self.tokenizer.pad_token_id
        )

        if return_mm_token_type_ids:
            image_tokens = np.array(self.image_token_ids).astype(input_ids.dtype)
            token_type_ids = np.any(input_ids[:, :, None] == image_tokens[None, None, :], axis=-1)
            text_inputs["token_type_ids"] = token_type_ids.tolist()

        text_inputs["input_ids"] = input_ids.tolist()
        text_inputs["attention_mask"] = attention_mask.tolist()

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs},
            tensor_type=return_tensors,
        )

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        对模型输出进行后处理以解码文本。

        参数：
            generated_outputs (`torch.Tensor` 或 `np.ndarray`)：
                模型 `generate` 函数的输出。期望输出为形状 `(batch_size, sequence_length)`
                或 `(sequence_length,)` 的张量。
            skip_special_tokens (`bool`，*可选*，默认为 `True`)：
                是否移除输出中的特殊 token。该参数会传递给分词器的 `batch_decode` 方法。
            clean_up_tokenization_spaces (`bool`，*可选*，默认为 `False`)：
                是否清理分词产生的空格。该参数会传递给分词器的 `batch_decode` 方法。
            **kwargs：
                要传递给分词器 `batch_decode` 方法的额外参数。

        返回：
            `list[str]`：解码后的文本。
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


MolmoAct2Processor.register_for_auto_class()
