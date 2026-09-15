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


"""MolmoAct2 的视频处理器类"""

import os
import warnings
from collections.abc import Callable
from contextlib import redirect_stdout
from functools import partial
from io import BytesIO
from urllib.parse import urlparse

import einops
import numpy as np
import requests
import torch
import torchvision.transforms
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
    validate_kwargs,
)
from transformers.processing_utils import Unpack, VideosKwargs
from transformers.utils import (
    TensorType,
    is_av_available,
    is_decord_available,
    is_torchcodec_available,
    is_yt_dlp_available,
    logging,
    to_numpy,
)
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import (
    VideoInput,
    VideoMetadata,
    is_valid_video,
    make_batched_metadata,
    make_batched_videos,
)

logger = logging.get_logger(__name__)

MAX_VIDEO_FPS = 8


def normalize_image(
    image: np.ndarray,
    image_mean: list[float],
    image_std: list[float],
) -> np.ndarray:
    if np.allclose(image_mean, [0.5, 0.5, 0.5]) and np.allclose(image_std, [0.5, 0.5, 0.5]):
        return image * np.asarray(2.0, dtype=np.float32) - np.asarray(1.0, dtype=np.float32)
    image -= np.array(image_mean, dtype=np.float32)[None, None, :]
    image /= np.array(image_std, dtype=np.float32)[None, None, :]
    return image


def resize_image(
    image: np.ndarray,
    desired_output_size: list[int],
    resample: PILImageResampling,
) -> np.ndarray:
    if len(image.shape) == 3:
        is_video = False
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    else:
        is_video = True
        image = torch.permute(torch.from_numpy(image), [0, 3, 1, 2])
    dtype = image.dtype
    if torch.is_floating_point(image):
        in_min = 0.0
        in_max = 1.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            resample,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(dtype)
    else:
        assert image.dtype == torch.uint8, (
            f"SigLIP expects float images or uint8 images, but got {image.dtype}"
        )
        in_min = 0.0
        in_max = 255.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            resample,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0, 255).to(dtype)

    resized = resized.to(torch.float32)
    resized = (resized - in_min) / (in_max - in_min)

    if is_video:
        resized = torch.permute(resized, [0, 2, 3, 1]).numpy()
    else:
        resized = torch.permute(resized, [1, 2, 0]).numpy()

    return resized


def build_resized_image(
    image: np.ndarray,
    base_image_input_size: list[int],
    resample: PILImageResampling,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    resized = resize_image(
        image,
        base_image_input_size,
        resample,
    )
    resized = normalize_image(resized, image_mean, image_std)
    if len(resized.shape) == 3:
        resized = np.expand_dims(resized, 0)
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    resize_idx = np.arange(crop_patch_w * crop_patch_h).reshape([crop_patch_h, crop_patch_w])
    return resized, resize_idx


def batch_pixels_to_patches(array: np.ndarray, patch_size: int) -> np.ndarray:
    """将 [n_images, h, w, 3] 的图像重塑为 [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, h, w = array.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(array, [n_crops, h_patches * w_patches, patch_size * patch_size])
        return array
    else:
        n_crops, h, w, c = array.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
        array = np.transpose(array, [0, 1, 3, 2, 4, 5])
        array = np.reshape(array, [n_crops, h_patches * w_patches, patch_size * patch_size * c])
        return array


def arange_for_pooling(
    idx_arr: np.ndarray,
    pool_h: int,
    pool_w: int,
) -> np.ndarray:
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(
        idx_arr,
        [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
        mode="constant",
        constant_values=-1,
    )
    return einops.rearrange(idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)


def image_to_patches_and_grids(
    image: ImageInput,
    base_image_input_size: list[int],
    resample: PILImageResampling,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
    image_pooling_w: int,
    image_pooling_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :return image_grids，每张图像池化后的形状
    :return crops，要用 ViT 处理的图像裁剪块
    :return pooled_patch_idx，对于 `image_tokens` 中的每个 patch_id token，
                                为该 token 做池化时所用 `crops` 中块的索引，用 -1 屏蔽
    """
    if isinstance(base_image_input_size, int):
        base_image_input_size = (base_image_input_size, base_image_input_size)

    pooling_w = image_pooling_w
    pooling_h = image_pooling_h

    resized, resize_idx = build_resized_image(
        image,
        base_image_input_size,
        resample,
        image_mean,
        image_std,
        image_patch_size,
    )
    pooling_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
    h, w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape([-1, pooling_h * pooling_w])
    image_grid = [h, w]
    return (
        image_grid,
        batch_pixels_to_patches(resized, image_patch_size),
        pooling_idx,
    )


def get_candidate_target_fps(
    video_fps: int | float,
    sampling_fps: int | float,
    max_fps: int | float = MAX_VIDEO_FPS,
) -> list[float]:
    """
    返回 `video_fps` 的因子中仍为 `sampling_fps` 倍数的那部分子集。

    示例：
        >>> get_candidate_target_fps(video_fps=6, sampling_fps=2)
        [2, 6]
        >>> get_candidate_target_fps(video_fps=5, sampling_fps=1)
        [1, 5]
        >>> get_candidate_target_fps(video_fps=2, sampling_fps=2)
        [2]
        >>> get_candidate_target_fps(video_fps=5, sampling_fps=2)
        Traceback (most recent call last):
            ...
        ValueError: sampling_fps=2 must divide video_fps=5 to produce consistent frame steps.
    """
    video_fps = int(video_fps)
    sampling_fps = int(sampling_fps)
    max_fps = int(max_fps)

    if sampling_fps is None:
        raise ValueError("sampling_fps must be provided")
    if video_fps <= 0 or sampling_fps <= 0:
        raise ValueError(f"video_fps and sampling_fps must be positive (got {video_fps}, {sampling_fps})")
    if video_fps % sampling_fps != 0:
        raise ValueError(f"sampling_fps={sampling_fps} must divide video_fps={video_fps}.")

    candidates = []
    for candidate in range(sampling_fps, video_fps + 1, sampling_fps):
        if candidate > max_fps:
            break
        if video_fps % candidate == 0:
            candidates.append(float(candidate))

    return candidates


def read_video_decord(
    video_path,
    sample_timestamps_fn: Callable,
    **kwargs,
) -> np.ndarray:
    """
    使用 Decord 后端解码视频。

    参数：
        video_path (`str`)：
            视频文件的路径。
        sample_timestamps_fn (`Callable`)：
            一个可调用函数，返回应对视频进行采样的时间戳。

    返回：
        tuple[`np.array`, `VideoMetadata`]：包含以下内容的元组：
            - RGB 格式的帧的 Numpy 数组（形状：[num_frames, height, width, 3]）。
            - `VideoMetadata` 对象。
    """
    # 延迟导入 decord
    import importlib

    decord = importlib.import_module("decord")

    vr = decord.VideoReader(uri=video_path, ctx=decord.cpu(0))  # decord 在 gpu 上有问题
    video_fps = vr.get_avg_fps()
    total_num_frames = len(vr)
    time_stamps = vr.get_frame_timestamp(list(range(len(vr))))
    duration = time_stamps[-1][1] - time_stamps[0][0]

    metadata = VideoMetadata(
        total_num_frames=int(total_num_frames),
        fps=float(video_fps),
        duration=float(duration),
        video_backend="decord",
    )

    target_timestamps = sample_timestamps_fn(metadata=metadata, **kwargs)
    target_timestamps = np.array(target_timestamps)
    offset = time_stamps[0, 0]

    ix = np.searchsorted(time_stamps[:, 1], target_timestamps + offset, side="right")
    ix = np.minimum(ix, len(time_stamps) - 1)

    video = vr.get_batch(ix).asnumpy()
    metadata.update(
        {
            "frames_indices": target_timestamps * video_fps,
            "height": video.shape[1],
            "width": video.shape[2],
        }
    )
    return video, metadata


def read_video_torchcodec(
    video_path,
    sample_timestamps_fn: Callable,
    **kwargs,
) -> np.ndarray:
    """
    使用 torchcodec 解码器解码视频。

    参数：
        video_path (`str`)：
            视频文件的路径。
        sample_timestamps_fn (`Callable`)：
            一个可调用函数，返回应对视频进行采样的时间戳。

    返回：
        tuple[`np.array`, `VideoMetadata`]：包含以下内容的元组：
            - RGB 格式的帧的 Numpy 数组（形状：[num_frames, height, width, 3]）。
            - `VideoMetadata` 对象。
    """
    # 延迟导入 torchcodec
    import importlib

    torchcodec = importlib.import_module("torchcodec")

    decoder = torchcodec.decoders.VideoDecoder(
        video_path,
        # 有趣的是，当我们加载整个视频时，`exact` 模式比近似模式耗时更少
        seek_mode="exact",
        # 允许 FFmpeg 自行决定线程数以提高效率
        num_ffmpeg_threads=0,
    )
    # 如果第一帧的起始时间 > 0，我们实际上会从该时间点开始截取视频，
    # 因为（大多数）视频播放器也会跳到该时间点
    time_offset = decoder.metadata.begin_stream_seconds_from_content
    # 注意，这个时长确实假设了我们从 `time_offset` 开始播放
    duration = decoder.metadata.duration_seconds

    metadata = VideoMetadata(
        total_num_frames=decoder.metadata.num_frames,
        fps=decoder.metadata.average_fps,
        duration=duration,
        video_backend="torchcodec",
        height=decoder.metadata.height,
        width=decoder.metadata.width,
    )

    target_timestamps = sample_timestamps_fn(metadata=metadata, **kwargs)

    # 浮点数/舍入问题可能导致 `target_timestamps` 非常轻微地
    # 越界，为了处理这种情况，我们先做合理性检查再对其进行裁剪
    assert all(x >= 0 for x in target_timestamps)
    assert all(x < duration + 1e-6 for x in target_timestamps)
    # 加上 1e-6 的余量，因为即使请求的是精确的边界值，
    # torchcodec 也可能抛出越界错误，不过我们仍然应该能拿到首帧/末帧
    max_timestamp = decoder.metadata.end_stream_seconds_from_content - 1e-6
    min_timestamp = decoder.metadata.begin_stream_seconds_from_content + 1e-6
    # 注意，这里避免使用 numpy 运算，以减少浮点精度问题
    timestamps = [x + time_offset for x in target_timestamps]
    timestamps = [max(min_timestamp, min(max_timestamp, x)) for x in timestamps]

    video = (
        decoder.get_frames_played_at(timestamps).data.numpy().transpose(0, 2, 3, 1)
    )  # 转换为 THWC 格式
    target_timestamps = np.array(target_timestamps)
    metadata.frames_indices = target_timestamps * metadata.fps

    return video, metadata


def read_video_pyav(
    video_path,
    sample_timestamps_fn: Callable,
    **kwargs,
) -> np.ndarray:
    """
    使用 PyAV 后端解码视频。

    参数：
        video_path (`str`)：
            视频文件的路径。
        sample_timestamps_fn (`Callable`)：
            一个可调用函数，返回应对视频进行采样的时间戳。

    返回：
        tuple[`np.array`, `VideoMetadata`]：包含以下内容的元组：
            - RGB 格式的帧的 Numpy 数组（形状：[num_frames, height, width, 3]）。
            - `VideoMetadata` 对象。
    """
    # 延迟导入 av
    import importlib

    av = importlib.import_module("av")

    with av.open(video_path) as container:
        video_stream = container.streams.video[0]
        fps = video_stream.average_rate or video_stream.guessed_rate
        it = container.decode(video=0)
        frames = list(it)

        stream = container.streams.video[0]
        start = frames[0].pts * stream.time_base
        container_end = stream.duration
        if container_end is not None:
            container_end *= stream.time_base
        if container_end is None or container_end < frames[-1].pts:
            # 流时长有问题，因此直接使用帧的 PTS，
            # 并猜测最后一帧的时长
            end = frames[-1].pts * stream.time_base + 1 / fps
        else:
            end = container_end
        duration = float(end - start)

        metadata = VideoMetadata(
            total_num_frames=len(frames),
            fps=float(fps),
            duration=float(duration),
            video_backend="pyav",
            height=video_stream.height,
            width=video_stream.width,
        )

        target_timestamps = sample_timestamps_fn(metadata=metadata, **kwargs)
        offset = float(start)

        target_timestamps = np.array(target_timestamps)
        end_time_stamps = np.array([float(frame.pts * stream.time_base) for frame in frames[1:]] + [duration])
        indices = np.searchsorted(end_time_stamps, target_timestamps + offset, side="right")
        indices = np.minimum(indices, len(end_time_stamps) - 1)

        video = np.stack(
            [frames[i].to_ndarray(format="rgb24", channel_last=True) for i in indices],
            axis=0,
        )

        metadata.frames_indices = target_timestamps * fps

        return video, metadata


VIDEO_DECODERS = {
    "decord": read_video_decord,
    "torchcodec": read_video_torchcodec,
    "pyav": read_video_pyav,
}


def load_video(
    video: VideoInput,
    backend: str = "decord",
    sample_timestamps_fn: Callable | None = None,
    **kwargs,
):
    """
    将 `video` 加载为 numpy 数组。

    参数：
        video (`VideoInput`)：
            要转换为 numpy 数组格式的视频。可以是视频链接或本地路径。
        backend (`str`，*可选*，默认为 `"decord"`)：
            加载视频时使用的后端。可以是 ["decord", "pyav", ""torchcodec"] 中的任意一个。默认为 "decord"。
        sample_timestamps_fn (`Callable`)：
            一个可调用函数，返回应对视频进行采样的时间戳。
    """

    # 如果提供的是数组或 `PIL` 帧，则提前返回
    if not isinstance(video, str):
        metadata = [None] * len(video)
        return video, metadata

    if urlparse(video).netloc in ["www.youtube.com", "youtube.com"]:
        if not is_yt_dlp_available():
            raise ImportError("To load a video from YouTube url you have  to install `yt_dlp` first.")
        # 延迟导入 yt_dlp
        import importlib

        yt_dlp = importlib.import_module("yt_dlp")

        buffer = BytesIO()
        with redirect_stdout(buffer), yt_dlp.YoutubeDL() as f:
            f.download([video])
        bytes_obj = buffer.getvalue()
        file_obj = BytesIO(bytes_obj)
    elif video.startswith("http://") or video.startswith("https://"):
        file_obj = BytesIO(requests.get(video, timeout=10).content)
    elif os.path.isfile(video):
        file_obj = video
    else:
        raise TypeError(
            "Incorrect format used for video. Should be an url linking to an video or a local path."
        )

    # 也可以用 decord 加载，但不能用 cv2/torchvision
    # 在 url 链接的情况下两者都会失败
    video_is_url = video.startswith("http://") or video.startswith("https://")
    if video_is_url and backend == "opencv":
        raise ValueError("If you are trying to load a video from URL, you cannot use 'opencv' as backend")

    if (
        (not is_decord_available() and backend == "decord")
        or (not is_torchcodec_available() and backend == "torchcodec")
        or (not is_av_available() and backend == "pyav")
    ):
        raise ImportError(
            f"You chose backend={backend} for loading the video but the required library is not found in your environment "
            f"Make sure to install {backend} before loading the video."
        )

    video_decoder = VIDEO_DECODERS[backend]
    video, metadata = video_decoder(file_obj, sample_timestamps_fn, **kwargs)
    return video, metadata


def get_target_fps(
    video_fps: float,
    max_frames: int,
    total_frames: int,
    frame_sample_mode: str,
    candidate_target_fps: tuple[float],
) -> float:
    """
    获取能最好地覆盖整个视频且采样帧数最多的目标 fps
    """
    num_frames_sampled = 0
    selected_target_fps = None
    for target_fps in candidate_target_fps:
        step_size = max(int(video_fps / target_fps), 1)
        num_frames_sampled_at_fps = int(total_frames / step_size)
        if num_frames_sampled == 0:
            if "uniform" in frame_sample_mode and num_frames_sampled_at_fps > max_frames:
                break
            selected_target_fps = target_fps
            num_frames_sampled = num_frames_sampled_at_fps

        else:
            # 候选采样 fps 递增，因此帧数不会减少
            assert num_frames_sampled <= num_frames_sampled_at_fps
            if num_frames_sampled_at_fps > max_frames:
                # 选择能覆盖整个视频的采样 fps
                continue

            elif num_frames_sampled_at_fps > num_frames_sampled:
                # 两者都小于 max_frames，选择采样帧密度更高的那个
                selected_target_fps = target_fps
                num_frames_sampled = num_frames_sampled_at_fps
    return selected_target_fps


def get_frame_times_and_chosen_fps(selected_target_fps, total_frames, max_frames, video_fps):
    if selected_target_fps is None:
        frame_indices = np.linspace(0, total_frames, max_frames, endpoint=False, dtype=int)
    else:
        step_size = max(int(video_fps / selected_target_fps), 1)
        frame_indices = np.arange(0, total_frames, step_size)
    if len(frame_indices) > max_frames:
        frame_indices = frame_indices[:max_frames]
    return selected_target_fps, frame_indices


class MolmoAct2VideoProcessorKwargs(VideosKwargs, total=False):
    patch_size: int | None
    pooling_size: list[int] | None
    frame_sample_mode: str | None
    max_fps: int | None
    sampling_fps: int | None


class MolmoAct2VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    size = {"height": 378, "width": 378}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 14
    pooling_size = [3, 3]
    do_sample_frames = True
    frame_sample_mode = "uniform_last_frame"
    max_fps = 2
    sampling_fps = 2
    valid_kwargs = MolmoAct2VideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_token_pooling", "video_grids"]

    def __init__(self, **kwargs: Unpack[MolmoAct2VideoProcessorKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (
            self.size.get("height", None) is None or self.size.get("width", None) is None
        ):
            raise ValueError("size must contain 'height' and 'width' keys.")

    def _further_process_kwargs(
        self,
        size: SizeDict | None = None,
        **kwargs,
    ) -> dict:
        """
        更新那些在校验前需要进一步处理的 kwargs。
        子类可以重写此方法以自定义 kwargs 的处理。
        """
        if size is not None and ("height" not in size or "width" not in size):
            raise ValueError("size must contain 'height' and 'width' keys.")

        return super()._further_process_kwargs(size=size, **kwargs)

    def sample_times(
        self,
        metadata: VideoMetadata,
        frame_sample_mode: str,
        num_frames: int,
        max_fps: int | None = None,
        sampling_fps: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        当传入数组形式的视频时，进行基于时间的采样
        参数：
            metadata (`VideoMetadata`)：
                视频的元数据，包含总时长、fps 和总帧数等信息。
            frame_sample_mode (`str`，*可选*)：
                帧采样模式。默认为 `self.frame_sample_mode`。
            num_frames (`int`，*可选*)：
                最多采样的帧数。默认为 `self.num_frames`。
            man_fps (`int`，*可选*)：
                采样的最大每秒帧数。
            sampling_fps (`int`，*可选*)：
                每秒采样帧数。默认为 `self.sampling_fps`。
                当 `frame_sample_mode` 为 `"fps"` 时使用。
        """
        frame_sample_mode = frame_sample_mode or self.frame_sample_mode
        num_frames = num_frames or self.num_frames
        sampling_fps = sampling_fps or self.sampling_fps

        duration = metadata.duration or metadata.total_num_frames / metadata.fps
        if frame_sample_mode == "fps":
            candidate_target_fps = get_candidate_target_fps(metadata.fps, sampling_fps)
            # 尝试越来越大的 FPS，直到遇到无法覆盖整个视频的取值
            target_fps = candidate_target_fps[0]
            for candidate_fps in candidate_target_fps[1:]:
                if num_frames / candidate_fps < duration:
                    break
                target_fps = candidate_fps
            times = np.arange(0, num_frames) / target_fps
            times = times[times < duration]
            return times
        elif frame_sample_mode == "uniform_last_frame":
            if max_fps is not None:
                max_duration = (num_frames - 1) / max_fps  # -1 是为了包含最后一帧
                if max_duration < duration:
                    times = np.linspace(0, duration, num=num_frames, endpoint=True, dtype=np.float64)
                else:
                    times = np.arange(0.0, stop=duration, step=1 / max_fps)
                    times = np.concatenate([times, [duration]], axis=0)
                    assert len(times) <= num_frames
            else:
                times = np.linspace(0, duration, num=num_frames, endpoint=True, dtype=np.float64)
            return times
        else:
            raise NotImplementedError(frame_sample_mode)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        frame_sample_mode: str | None = None,
        num_frames: int | None = None,
        max_fps: int | None = None,
        sampling_fps: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        当传入数组形式的视频时，进行基于帧的采样
        参数：
            metadata (`VideoMetadata`)：
                视频的元数据，包含总时长、fps 和总帧数等信息。
            frame_sample_mode (`str`，*可选*)：
                帧采样模式。默认为 `self.frame_sample_mode`。
            num_frames (`int`，*可选*)：
                最多采样的帧数。默认为 `self.num_frames`。
            max_fps (`int`，*可选*)：
                采样的最大每秒帧数。
            sampling_fps (`int`，*可选*)：
                每秒采样帧数。默认为 `self.sampling_fps`。
                当 `frame_sample_mode` 为 `"fps"` 时使用。
        """
        frame_sample_mode = frame_sample_mode or self.frame_sample_mode
        num_frames = num_frames or self.num_frames
        sampling_fps = sampling_fps or self.sampling_fps

        total_num_frames = metadata.total_num_frames
        if frame_sample_mode == "uniform_last_frame" and max_fps is not None:
            duration = total_num_frames / metadata.fps
            if total_num_frames <= 2:
                return np.arange(total_num_frames).astype(int)
            if duration > (num_frames - 1) / max_fps:  # -1 是为了包含最后一帧
                # 均匀采样回退方案
                indices = np.linspace(
                    0,
                    total_num_frames - 1,
                    num=min(num_frames, total_num_frames),
                    endpoint=True,
                ).astype(int)
                return indices
            else:
                float_indices = np.arange(
                    0.0,
                    stop=total_num_frames - 1,
                    step=float(metadata.fps / max_fps),
                )
                if np.round(float_indices[-1]) != total_num_frames - 1:
                    float_indices = np.concatenate([float_indices, [total_num_frames - 1]], axis=0)
                indices = np.round(float_indices).astype(int)
                assert indices[-1] < total_num_frames
                assert len(float_indices) <= num_frames
                return indices
        elif frame_sample_mode == "uniform_last_frame":
            indices = np.linspace(
                0,
                total_num_frames - 1,
                num=min(num_frames, total_num_frames),
                endpoint=True,
            ).astype(int)
            return indices
        elif frame_sample_mode == "fps":
            candidate_target_fps = get_candidate_target_fps(metadata.fps, sampling_fps)
            selected_target_fps = get_target_fps(
                metadata.fps,
                num_frames,
                total_num_frames,
                frame_sample_mode,
                candidate_target_fps,
            )
            _, indices = get_frame_times_and_chosen_fps(
                selected_target_fps,
                total_num_frames,
                num_frames,
                metadata.fps,
            )
            return indices
        else:
            raise NotImplementedError(frame_sample_mode)

    def fetch_videos(self, video_url_or_urls: str | list[str] | list[list[str]], sample_timestamps_fn=None):
        """
        将单个或一组 url 转换为对应的 `np.array` 对象。

        如果传入单个 url，返回值将是单个对象。如果传入列表，则返回对象列表。
        """
        if (not is_decord_available()) and (not is_torchcodec_available()) and (not is_av_available()):
            raise ImportError(
                "MolmoAct2VideoProcessor requires `decord`, `torchcodec`, or `av` to be installed."
            )

        if is_decord_available():
            backend = "decord"
        elif is_torchcodec_available():
            warnings.warn(
                "`decord` is not installed and cannot be used to decode the video by default. "
                "Falling back to `torchcodec`.",
                stacklevel=2,
            )
            backend = "torchcodec"
        else:
            warnings.warn(
                "`decord` is not installed and cannot be used to decode the video by default. "
                "Falling back to `PyAV`.",
                stacklevel=2,
            )
            backend = "pyav"

        if isinstance(video_url_or_urls, list):
            return list(
                zip(
                    *[
                        self.fetch_videos(x, sample_timestamps_fn=sample_timestamps_fn)
                        for x in video_url_or_urls
                    ],
                    strict=False,
                )
            )
        else:
            return load_video(video_url_or_urls, backend=backend, sample_timestamps_fn=sample_timestamps_fn)

    def _decode_and_sample_videos(
        self,
        videos: VideoInput,
        video_metadata: VideoMetadata | dict,
        do_sample_frames: bool | None = None,
        sample_indices_fn: Callable | None = None,
        sample_timestamps_fn: Callable | None = None,
    ):
        """
        解码输入视频，并在需要时对帧进行采样。
        """
        videos = make_batched_videos(videos)
        video_metadata = make_batched_metadata(videos, video_metadata=video_metadata)

        # 如果传入的是数组形式的视频，则进行基于帧的采样
        # 否则，进行带解码的基于时间的采样
        if is_valid_video(videos[0]) and do_sample_frames:
            assert video_metadata[0].fps is not None, "FPS must be provided for video input"
            sampled_videos = []
            sampled_metadata = []
            for video, metadata in zip(videos, video_metadata, strict=False):
                indices = sample_indices_fn(metadata=metadata)
                metadata.frames_indices = indices
                sampled_videos.append(video[indices])
                sampled_metadata.append(metadata)
            videos = sampled_videos
            video_metadata = sampled_metadata
        elif not is_valid_video(videos[0]):
            if sample_indices_fn is None:
                logger.warning(
                    "do_sample_frames is False, but video array is not provided: "
                    "Will decode the video and sample frames using MolmoAct2's default sampling mode"
                )
            if isinstance(videos[0], list):
                raise ValueError("A list of images is not supported for video input!")
            else:
                videos, video_metadata = self.fetch_videos(videos, sample_timestamps_fn=sample_timestamps_fn)

        return videos, video_metadata

    def _prepare_input_videos(
        self,
        videos: VideoInput,
        **kwargs,
    ) -> list[np.ndarray]:
        processed_videos = [to_numpy(video) for video in videos]
        return processed_videos

    def preprocess(
        self,
        videos: VideoInput,
        **kwargs: Unpack[MolmoAct2VideoProcessorKwargs],
    ) -> BatchFeature:
        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=list(self.valid_kwargs.__annotations__.keys()) + ["return_tensors"],
        )

        # 从 self 设置默认 kwargs。这确保了当用户未提供某个 kwarg 时，
        # 它会从实例获取默认值，或被设为 None。
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        do_sample_frames = kwargs.pop("do_sample_frames")
        video_metadata = kwargs.pop("video_metadata")

        sample_indices_fn = partial(self.sample_frames, **kwargs) if do_sample_frames else None
        sample_timestamps_fn = partial(self.sample_times, **kwargs)
        videos, video_metadata = self._decode_and_sample_videos(
            videos,
            video_metadata=video_metadata,
            do_sample_frames=do_sample_frames,
            sample_indices_fn=sample_indices_fn,
            sample_timestamps_fn=sample_timestamps_fn,
        )
        videos = self._prepare_input_videos(videos=videos)

        kwargs = self._further_process_kwargs(**kwargs)

        return_metadata = kwargs.pop("return_metadata")
        preprocessed_videos = self._preprocess(videos=videos, **kwargs)
        if return_metadata:
            preprocessed_videos["video_metadata"] = video_metadata
        return preprocessed_videos

    def _preprocess(
        self,
        videos: list[np.ndarray],
        size: SizeDict | None = None,
        resample: PILImageResampling | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool | None = None,
        patch_size: int | None = None,
        pooling_size: list[int] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        为模型预处理视频。
        参数：
            videos (`VideoInput`)：
                要预处理的视频。
            size (`SizeDict`，*可选*，默认为 `self.size`)：
                图像缩放后的尺寸。
            resample (`PILImageResampling`，*可选*，默认为 `self.resample`)：
                缩放图像时使用的重采样滤波器。可以是 `PILImageResampling` 枚举之一。
                仅在 `do_resize` 设为 `True` 时生效。
            image_mean (`float` 或 `list[float]`，*可选*，默认为 `self.image_mean`)：
                归一化使用的图像均值。仅在 `do_normalize` 设为 `True` 时生效。
            image_std (`float` 或 `list[float]`，*可选*，默认为 `self.image_std`)：
                归一化使用的图像标准差。仅在 `do_normalize` 设为
                `True` 时生效。
            do_convert_rgb (`bool`，*可选*，默认为 `self.do_convert_rgb`)：
                是否将图像转换为 RGB。
            patch_size (`int`，*可选*，默认为 `self.patch_size`)：
                视觉编码器的空间块尺寸。
            pooling_size (`list[int]`，*可选*，默认为 `self.pooling_size`)：
                视觉适配器的池化尺寸。
            return_tensors (`str` 或 `TensorType`，*可选*)：
                要返回的张量类型。可以是以下之一：
                - 未设置：返回 `np.ndarray` 列表。
                - `TensorType.TENSORFLOW` 或 `'tf'`：返回 `tf.Tensor` 类型的批次。
                - `TensorType.PYTORCH` 或 `'pt'`：返回 `torch.Tensor` 类型的批次。
                - `TensorType.NUMPY` 或 `'np'`：返回 `np.ndarray` 类型的批次。
                - `TensorType.JAX` 或 `'jax'`：返回 `jax.numpy.ndarray` 类型的批次。

        返回：
            包含以下键的 `BatchFeature`：
                - `pixel_values_videos`：预处理后的视频。
                - `video_token_pooling`：`video_tokens` 中每个 token 做池化时所用 `crops` 中块的索引。
                - `video_grids`：视频网格。
        """
        if size.height is None or size.width is None:
            raise ValueError("size must contain 'height' and 'width' keys.")

        base_image_input_size = [size.height, size.width]

        resample = resample or self.resample
        image_mean = image_mean or self.image_mean
        image_std = image_std or self.image_std
        do_convert_rgb = do_convert_rgb or self.do_convert_rgb

        patch_size = patch_size or self.patch_size
        pooling_size = pooling_size or self.pooling_size

        image_pooling_h, image_pooling_w = pooling_size

        batch_grids = []
        batch_crops = []
        batch_pooled_patches_idx = []

        for video in videos:
            all_crops = []
            pooled_patches_idx = []

            for frame in video:
                image_grid, crops, pooled_idx = image_to_patches_and_grids(
                    frame,
                    base_image_input_size,
                    resample,
                    image_mean,
                    image_std,
                    patch_size,
                    image_pooling_w,
                    image_pooling_h,
                )
                offset = sum(np.prod(x.shape[:2]) for x in all_crops)
                pooled_idx_with_offset = np.where(pooled_idx >= 0, pooled_idx + offset, pooled_idx)
                pooled_patches_idx.append(pooled_idx_with_offset)
                all_crops.append(crops)

            video_grid = np.array([len(video), image_grid[0], image_grid[1]])
            all_crops = np.concatenate(all_crops, 0)
            pooled_patches_idx = np.concatenate(pooled_patches_idx, 0)

            batch_grids.append(video_grid)
            batch_crops.append(all_crops)
            batch_pooled_patches_idx.append(pooled_patches_idx)

        video_grids = np.stack(batch_grids, 0)
        pixel_values_videos = np.concatenate(batch_crops, 0)
        video_token_pooling = np.concatenate(batch_pooled_patches_idx, 0)

        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_token_pooling": video_token_pooling,
            "video_grids": video_grids,
        }

        return BatchFeature(data, tensor_type=return_tensors)


MolmoAct2VideoProcessor.register_for_auto_class()
