"""适用于 MFCC/Mel/Log-Mel 的轻量卷积基线。"""

from __future__ import annotations

import torch
from torch import nn
from pydantic import Field

from ser_lib.core.config import StrictConfig
from ser_lib.data.types import SERBatch, TensorSpec
from ser_lib.data.validation import ModelSpec
from ser_lib.models.base import ModelOutput, SERModel
from ser_lib.models.registry import ModelDescriptor, model_registry

class CNNBaselineConfig(StrictConfig):
    feature_dim: int = Field(ge=1)
    num_classes: int = Field(ge=2)
    hidden_dim: int = Field(default=128, ge=1)
    dropout: float = Field(default=0.2, ge=0, lt=1)


class CNNBaseline(SERModel):
    """保持时间分辨率的一维卷积分类器。

    输入 ``features`` 为 ``[B,F,T]``。卷积沿时间轴进行，随后根据 mask 做
    masked mean pooling，因此支持同一 batch 内不同有效长度。
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or num_classes < 2 or hidden_dim < 1:
            raise ValueError("feature_dim/hidden_dim 必须为正，num_classes 必须 >= 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            model_id="cnn_baseline",
            required_inputs={
                "features": TensorSpec(layout="FT", feature_dim=self.feature_dim)
            },
            supports_masks=True,
            supports_variable_length=True,
            num_classes=self.num_classes,
        )

    @property
    def model_config(self) -> dict[str, int | float]:
        return CNNBaselineConfig(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).model_dump(mode="json")

    def forward(self, batch: SERBatch) -> ModelOutput:
        try:
            features = batch.inputs["features"]
        except KeyError:
            raise ValueError("CNNBaseline 需要 batch.inputs['features']") from None
        if features.dim() != 3 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"CNNBaseline 期望 [B,{self.feature_dim},T]，实际 {tuple(features.shape)}"
            )
        if features.shape[0] < 1 or features.shape[-1] < 1:
            raise ValueError("CNNBaseline 不接受空 batch 或零长度时间轴")
        if not features.is_floating_point():
            raise ValueError(f"CNNBaseline features 必须是浮点 tensor，实际 {features.dtype}")
        encoded = self.encoder(features)
        mask = batch.masks.get("features")
        if mask is None:
            embeddings = encoded.mean(dim=-1)
        else:
            if mask.shape != (features.shape[0], features.shape[-1]):
                raise ValueError(
                    f"features mask 期望 {(features.shape[0], features.shape[-1])}，"
                    f"实际 {tuple(mask.shape)}"
                )
            if torch.any(mask.sum(dim=-1) == 0):
                raise ValueError("CNNBaseline 的每个样本必须至少包含一个有效时间步")
            weights = mask.to(encoded.dtype).unsqueeze(1)
            embeddings = (encoded * weights).sum(dim=-1) / weights.sum(dim=-1)
        return ModelOutput(logits=self.classifier(embeddings), embeddings=embeddings)


model_registry.register(
    "cnn_baseline", CNNBaseline, config_model=CNNBaselineConfig,
    descriptor=ModelDescriptor(
        id="cnn_baseline", display_name="CNN 基线",
        description="适用于 MFCC/Mel/Log-Mel 的轻量时间卷积分类器。",
        config_schema=CNNBaselineConfig.model_json_schema(),
        input_layouts={"features": "FT"},
    ),
)


__all__ = ["CNNBaseline", "CNNBaselineConfig"]
