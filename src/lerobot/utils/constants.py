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
# 键名
import os
from pathlib import Path

from huggingface_hub.constants import HF_HOME

OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_ENV_STATE = OBS_STR + ".environment_state"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"
OBS_LANGUAGE = OBS_STR + ".language"
OBS_LANGUAGE_TOKENS = OBS_LANGUAGE + ".tokens"
OBS_LANGUAGE_ATTENTION_MASK = OBS_LANGUAGE + ".attention_mask"
OBS_LANGUAGE_CAUSAL_MARKS = OBS_LANGUAGE + ".causal_marks"
OBS_LANGUAGE_SUBTASK = OBS_STR + ".subtask"
OBS_LANGUAGE_SUBTASK_TOKENS = OBS_LANGUAGE_SUBTASK + ".tokens"
OBS_LANGUAGE_SUBTASK_ATTENTION_MASK = OBS_LANGUAGE_SUBTASK + ".attention_mask"

ACTION = "action"
ACTION_PREFIX = ACTION + "."
ACTION_TOKENS = ACTION + ".tokens"
ACTION_TOKEN_MASK = ACTION + ".token_mask"
ACTION_CODE_TOKEN_MASK = ACTION + ".code_token_mask"
REWARD = "next.reward"
TRUNCATED = "next.truncated"
DONE = "next.done"
SUCCESS = "next.success"
INFO = "info"

# 描述文本生成请求的补充数据键，由 rollout 推理引擎在预处理器
# 运行之前设置，以便某个处理步骤能够格式化提示词：
# QUERY_KIND 表示请求的内容类型（"vqa"、"next_subtask" 等），
# QUERY_TEXT 是请求本身。普通动作推理时不存在这些键。
QUERY_KIND = "query_kind"
QUERY_TEXT = "query_text"

# 原始语义语言数据集列。放在这里是为了让轻量级策略处理器
# 无需导入可选的 datasets/pyarrow 技术栈。
LANGUAGE_PERSISTENT = "language_persistent"
LANGUAGE_EVENTS = "language_events"
MESSAGES_RENDERED = "messages_rendered"

ROBOTS = "robots"
TELEOPERATORS = "teleoperators"

# 文件与目录
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
ALGORITHM_DIR = "algorithm"
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"

if "LEROBOT_HOME" in os.environ:
    raise ValueError(
        f"You have a 'LEROBOT_HOME' environment variable set to '{os.getenv('LEROBOT_HOME')}'.\n"
        "'LEROBOT_HOME' is deprecated, please use 'HF_LEROBOT_HOME' instead."
    )

# 缓存目录
default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()
# LeRobot 自己的支持版本回退的 Hub 缓存（不是系统级的 ~/.cache/huggingface/hub/）。
# 用作 ``snapshot_download`` 的 ``cache_dir`` 参数，使不同版本的
# 数据集存储在相互隔离的快照目录中。
HF_LEROBOT_HUB_CACHE = HF_LEROBOT_HOME / "hub"

# 标定目录
default_calibration_path = HF_LEROBOT_HOME / "calibration"
HF_LEROBOT_CALIBRATION = Path(os.getenv("HF_LEROBOT_CALIBRATION", default_calibration_path)).expanduser()


# 数据集元特征（由录制流水线自动填充）
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}

# ImageNet 归一化常量
IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}

# 流式数据集
LOOKBACK_BACKTRACKTABLE = 100
LOOKAHEAD_BACKTRACKTABLE = 100

# openpi
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38  # TODO(pepijn): 扩展支持 fp8 模型时修改此值

# LIBERO 观测键的常量
LIBERO_KEY_EEF_POS = "robot_state/eef/pos"
LIBERO_KEY_EEF_QUAT = "robot_state/eef/quat"
LIBERO_KEY_EEF_MAT = "robot_state/eef/mat"
LIBERO_KEY_EEF_AXISANGLE = "robot_state/eef/axisangle"
LIBERO_KEY_GRIPPER_QPOS = "robot_state/gripper/qpos"
LIBERO_KEY_GRIPPER_QVEL = "robot_state/gripper/qvel"
LIBERO_KEY_JOINTS_POS = "robot_state/joints/pos"
LIBERO_KEY_JOINTS_VEL = "robot_state/joints/vel"
LIBERO_KEY_PIXELS_AGENTVIEW = "pixels/agentview_image"
LIBERO_KEY_PIXELS_EYE_IN_HAND = "pixels/robot0_eye_in_hand_image"
