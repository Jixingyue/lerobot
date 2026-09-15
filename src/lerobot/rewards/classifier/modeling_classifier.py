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

import logging

import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_IMAGE, REWARD

from ..pretrained import PreTrainedRewardModel
from .configuration_classifier import RewardClassifierConfig


class ClassifierOutput:
    """分类器输出的包装类，附带额外的元数据。"""

    def __init__(
        self,
        logits: Tensor,
        probabilities: Tensor | None = None,
        hidden_states: Tensor | None = None,
    ):
        self.logits = logits
        self.probabilities = probabilities
        self.hidden_states = hidden_states

    def __repr__(self):
        return (
            f"ClassifierOutput(logits={self.logits}, "
            f"probabilities={self.probabilities}, "
            f"hidden_states={self.hidden_states})"
        )


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        可学习空间嵌入的 PyTorch 实现

        Args:
            height: 输入特征的空间高度
            width: 输入特征的空间宽度
            channel: 输入通道数
            num_features: 输出嵌入的维度数
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        空间嵌入的前向传播

        Args:
            features: 形状为 [B, H, W, C] 的输入张量；无批次维度时为 [H, W, C]
        Returns:
            形状为 [B, C*F] 的输出张量；无批次维度时为 [C*F]
        """

        features = features.last_hidden_state

        original_shape = features.shape
        if features.dim() == 3:
            features = features.unsqueeze(0)  # 添加批次维度

        features_expanded = features.unsqueeze(-1)  # [B, H, W, C, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, H, W, C, F]

        # 逐元素相乘并在空间维度上归约
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # 对 H、W 求和

        # 重塑形状以合并通道维度和特征维度
        output = output.view(output.size(0), -1)  # [B, C*F]

        # 移除批次维度
        if len(original_shape) == 3:
            output = output.squeeze(0)

        return output


class Classifier(PreTrainedRewardModel):
    """构建在预训练编码器之上的图像分类器。"""

    name = "reward_classifier"
    config_class = RewardClassifierConfig

    def __init__(
        self,
        config: RewardClassifierConfig,
        **kwargs,
    ):
        from transformers import AutoModel

        super().__init__(config)
        self.config = config

        # 设置编码器
        encoder = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True)
        # 如果给定的是多模态模型，则提取视觉模型
        if hasattr(encoder, "vision_model"):
            logging.info("Multimodal model detected - using vision encoder only")
            self.encoder = encoder.vision_model
            self.vision_config = encoder.config.vision_config
        else:
            self.encoder = encoder
            self.vision_config = getattr(encoder, "config", None)

        # 从配置中获取模型类型
        self.is_cnn = self.config.model_type == "cnn"

        # 对于 CNN，初始化骨干网络
        if self.is_cnn:
            self._setup_cnn_backbone()

        self._freeze_encoder()

        # 从 input_features 中提取图像键
        self.image_keys = [
            key.replace(".", "_") for key in config.input_features if key.startswith(OBS_IMAGE)
        ]

        if self.is_cnn:
            self.encoders = nn.ModuleDict()
            for image_key in self.image_keys:
                encoder = self._create_single_encoder()
                self.encoders[image_key] = encoder

        self._build_classifier_head()

    def _setup_cnn_backbone(self):
        """设置 CNN 编码器"""
        if hasattr(self.encoder, "fc"):
            self.feature_dim = self.encoder.fc.in_features
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        elif hasattr(self.encoder.config, "hidden_sizes"):
            self.feature_dim = self.encoder.config.hidden_sizes[-1]  # 最后一个通道维度
        else:
            raise ValueError("Unsupported CNN architecture")

    def _freeze_encoder(self) -> None:
        """冻结编码器参数。"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _create_single_encoder(self):
        encoder = nn.Sequential(
            self.encoder,
            SpatialLearnedEmbeddings(
                height=4,
                width=4,
                channel=self.feature_dim,
                num_features=self.config.image_embedding_pooling_dim,
            ),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.feature_dim * self.config.image_embedding_pooling_dim, self.config.latent_dim),
            nn.LayerNorm(self.config.latent_dim),
            nn.Tanh(),
        )

        return encoder

    def _build_classifier_head(self) -> None:
        """初始化分类头结构。"""
        # 根据模型类型获取输入维度
        if self.is_cnn:
            input_dim = self.config.latent_dim
        else:  # Transformer 模型
            if hasattr(self.encoder.config, "hidden_size"):
                input_dim = self.encoder.config.hidden_size
            else:
                raise ValueError("Unsupported transformer architecture since hidden_size is not found")

        self.classifier_head = nn.Sequential(
            nn.Linear(input_dim * self.config.num_cameras, self.config.hidden_dim),
            nn.Dropout(self.config.dropout_rate),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.hidden_dim,
                1 if self.config.num_classes == 2 else self.config.num_classes,
            ),
        )

    def _get_encoder_output(self, x: torch.Tensor, image_key: str) -> torch.Tensor:
        """从编码器中提取合适的输出。"""
        with torch.no_grad():
            if self.is_cnn:
                # HF ResNet 会在内部应用池化
                outputs = self.encoders[image_key](x)
                return outputs
            else:  # Transformer 模型
                outputs = self.encoder(x)
                return outputs.last_hidden_state[:, 0, :]

    def extract_images_and_labels(self, batch: dict[str, Tensor]) -> tuple[list, Tensor]:
        """从批次中提取图像张量和标签张量。"""
        # 检查 OBS_IMAGE 和 OBS_IMAGES 两种前缀
        images = [batch[key] for key in self.config.input_features if key.startswith(OBS_IMAGE)]
        labels = batch[REWARD]

        return images, labels

    def predict(self, xs: list) -> ClassifierOutput:
        """用于推理的分类器前向传播。"""
        encoder_outputs = torch.hstack(
            [self._get_encoder_output(x, img_key) for x, img_key in zip(xs, self.image_keys, strict=True)]
        )
        logits = self.classifier_head(encoder_outputs)

        if self.config.num_classes == 2:
            logits = logits.squeeze(-1)
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)

        return ClassifierOutput(logits=logits, probabilities=probabilities, hidden_states=encoder_outputs)

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        """根据图像观测，成功返回 1.0，失败返回 0.0。"""
        images = [batch[key] for key in self.config.input_features if key.startswith(OBS_IMAGE)]
        output = self.predict(images)

        if self.config.num_classes == 2:
            return (output.probabilities > 0.5).float()
        else:
            return torch.argmax(output.probabilities, dim=1).float()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """与 train.py 兼容的标准训练前向传播。"""
        # 提取图像和标签
        images, labels = self.extract_images_and_labels(batch)

        # 获取预测结果
        outputs = self.predict(images)

        # 计算损失
        if self.config.num_classes == 2:
            # 二分类
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)
            predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
        else:
            # 多分类
            loss = nn.functional.cross_entropy(outputs.logits, labels.long())
            predictions = torch.argmax(outputs.logits, dim=1)

        # 计算准确率用于日志记录
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        # 返回损失和用于日志记录的指标
        output_dict = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

        return loss, output_dict

    def predict_reward(self, batch, threshold=0.5):
        """评估方法。返回预测的奖励，决策阈值作为参数传入。"""
        # 从批次字典中提取图像
        images = [batch[key] for key in self.config.input_features if key.startswith(OBS_IMAGE)]

        if self.config.num_classes == 2:
            probs = self.predict(images).probabilities
            logging.debug(f"Predicted reward images: {probs}")
            return (probs > threshold).float()
        else:
            return torch.argmax(self.predict(images).probabilities, dim=1)
