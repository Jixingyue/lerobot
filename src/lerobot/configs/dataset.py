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

"""``lerobot-record`` 和 ``lerobot-rollout`` 共用的数据集录制配置。"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .video import DepthEncoderConfig, RGBEncoderConfig, depth_encoder_defaults, rgb_encoder_defaults


@dataclass
class DatasetRecordConfig:
    # 数据集标识。按约定应为 '{hf_username}/{dataset_name}' 形式（例如 `lerobot/test`）。
    repo_id: str = ""
    # 对录制期间所执行任务的简短而准确的描述（例如 "Pick the Lego block and drop it in the box on the right."）。
    single_task: str = ""
    # 数据集存储的根目录（例如 'dataset/path'）。如果为 None，默认为 $HF_LEROBOT_HOME/repo_id。
    root: str | Path | None = None
    # 限制每秒帧数。
    fps: int = 30
    # 每个 episode 数据录制的秒数。
    episode_time_s: int | float = 60
    # 每个 episode 之后重置环境的秒数。
    reset_time_s: int | float = 60
    # 要录制的 episode 数量。
    num_episodes: int = 50
    # 将数据集中的帧编码为视频
    video: bool = True
    # 将数据集上传到 Hugging Face hub。
    push_to_hub: bool = True
    # 如果为 True，则以私有方式上传；如果为 None，则遵循 Hub 上的组织默认设置（仅影响组织）。
    private: bool | None = None
    # 为 Hub 上的数据集添加标签。
    tags: list[str] | None = None
    # 负责将帧保存为 PNG 的子进程数量。设为 0 表示仅使用线程；
    # 设为 ≥1 表示使用子进程，每个子进程内部使用线程写入图像。最佳进程数
    # 和线程数取决于你的系统。我们推荐 0 个进程、每个相机 4 个线程。
    # 如果 fps 不稳定，请调整线程数。如果仍不稳定，可尝试使用 1 个或更多子进程。
    num_image_writer_processes: int = 0
    # 每个相机将帧作为 png 图像写入磁盘的线程数。
    # 线程过多可能因主线程被阻塞而导致遥操作 fps 不稳定。
    # 线程不足可能导致相机 fps 过低。
    num_image_writer_threads_per_camera: int = 4
    # 批量编码视频之前要录制的 episode 数量
    # 设为 1 表示立即编码（默认行为），设为更大的值表示批量编码
    video_encoding_batch_size: int = 1
    # 相机 MP4 的视频编码器设置（编解码器、质量、GOP 等）。通过 CLI 嵌套键调整，
    # 例如 ``--dataset.rgb_encoder.vcodec=h264``（参见 ``RGBEncoderConfig``）。
    rgb_encoder: RGBEncoderConfig = field(default_factory=rgb_encoder_defaults)
    # 深度图 MP4 的视频编码器设置（编解码器、质量、GOP 等）。通过 CLI 嵌套键调整。
    depth_encoder: DepthEncoderConfig = field(default_factory=depth_encoder_defaults)
    # 启用流式视频编码：在采集过程中实时编码帧，而不是
    # 先写入 PNG 图像。使 save_episode() 几乎即时完成。更多信息见文档：https://huggingface.co/docs/lerobot/streaming_video_encoding
    streaming_encoding: bool = False
    # 使用流式编码时每个相机缓冲的最大帧数。
    # 在 30fps 下约为 1 秒的缓冲。当编码器跟不上时提供背压（backpressure）。
    encoder_queue_maxsize: int = 30
    # 每个编码器实例的线程数。None = 自动（编解码器默认值）。
    # 较低的值可降低 CPU 使用率；对 libsvtav1 映射为 'lp'（通过 svtav1-params），对 h264/hevc 映射为 'threads'。
    encoder_threads: int | None = None
    # 跳过向 repo_id 追加日期时间标签，保持用户提供的名称不变
    # （例如为后续 `lerobot-edit-dataset merge` 准备的自管理版本化名称）。
    no_stamp: bool = False

    def stamp_repo_id(self) -> None:
        """向 ``repo_id`` 追加日期时间标签，使每次录制会话都有唯一的名称。

        必须在数据集*创建*时显式调用——恢复录制时不能调用，
        因为必须保留已有的 ``repo_id``（已加盖时间戳）。
        当设置了 ``no_stamp`` 时此操作无效，以保留用户管理的 ``repo_id``。
        """
        if self.no_stamp:
            return
        if self.repo_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.repo_id = f"{self.repo_id}_{timestamp}"
