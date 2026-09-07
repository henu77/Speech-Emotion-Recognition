"""适用于帧级声学特征的轻量 Transformer 编码器。"""

from __future__ import annotations

import math
from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import nn

from ser_lib.core.config import StrictConfig
from ser_lib.data.types import SERBatch, TensorSpec
from ser_lib.data.validation import ModelSpec
from ser_lib.models.base import ModelOutput, SERModel
from ser_lib.models.registry import ModelDescriptor, model_registry


class TransformerBaselineConfig(StrictConfig):
    feature_dim: int = Field(ge=1)
    num_classes: int = Field(ge=2)
    d_model: int = Field(default=128, ge=4)
    num_heads: int = Field(default=4, ge=1)
    num_layers: int = Field(default=2, ge=1)
    feedforward_dim: int = Field(default=256, ge=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    activation: Literal["relu", "gelu"] = "gelu"
    norm_first: bool = False

    @model_validator(mode="after")
    def _validate_attention_dimensions(self) -> "TransformerBaselineConfig":
        if self.d_model % self.num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除")
        if self.feedforward_dim < self.d_model:
            raise ValueError("feedforward_dim 必须 >= d_model")
        return self


def _sinusoidal_positions(length: int, dimension: int, tensor: torch.Tensor) -> torch.Tensor:
    """动态生成位置编码，避免固定最大序列长度和 artifact 状态膨胀。"""
    positions = torch.arange(length, device=tensor.device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=tensor.device, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, device=tensor.device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(positions * frequencies[:dimension // 2])
    return encoding.to(dtype=tensor.dtype)


class TransformerBaseline(SERModel):
    """投影声学帧、加入正弦位置编码并进行 masked mean pooling。"""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "gelu",
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        config = TransformerBaselineConfig(
            feature_dim=feature_dim,
            num_classes=num_classes,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        for key, value in config.model_dump().items():
            setattr(self, key, value)
        self.input_projection = nn.Linear(self.feature_dim, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=self.norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=self.num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            model_id="transformer_baseline",
            required_inputs={
                "features": TensorSpec(layout="FT", feature_dim=self.feature_dim)
            },
            supports_masks=True,
            supports_variable_length=True,
            num_classes=self.num_classes,
        )

    @property
    def model_config(self) -> dict[str, int | float | bool | str]:
        return TransformerBaselineConfig(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            activation=self.activation,
            norm_first=self.norm_first,
        ).model_dump(mode="json")

    def _valid_mask(self, batch: SERBatch, features: torch.Tensor) -> torch.Tensor:
        batch_size, _, time = features.shape
        lengths = batch.lengths.get("features")
        mask = batch.masks.get("features")
        if lengths is None and mask is None:
            return torch.ones(batch_size, time, dtype=torch.bool, device=features.device)
        if lengths is not None:
            if lengths.shape != (batch_size,):
                raise ValueError(f"features lengths 必须是 [B]，实际 {tuple(lengths.shape)}")
            if torch.any(lengths <= 0) or torch.any(lengths > time):
                raise ValueError("features lengths 必须位于 [1,T]")
            length_mask = torch.arange(time, device=features.device).unsqueeze(0) < (
                lengths.to(features.device).unsqueeze(1)
            )
            if mask is None:
                return length_mask
        if mask is None or mask.shape != (batch_size, time):
            raise ValueError("features mask 必须是 [B,T]")
        mask = mask.to(device=features.device, dtype=torch.bool)
        if lengths is not None and not torch.equal(mask, length_mask):
            raise ValueError("features mask 必须是由 lengths 定义的连续前缀 mask")
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("features mask 每行至少包含一个有效帧")
        return mask

    def forward(self, batch: SERBatch) -> ModelOutput:
        if "features" not in batch.inputs:
            raise ValueError("TransformerBaseline 需要 batch.inputs['features']")
        features = batch.inputs["features"]
        if features.dim() != 3 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"TransformerBaseline 期望 [B,{self.feature_dim},T]，"
                f"实际 {tuple(features.shape)}"
            )
        if features.shape[0] < 1 or features.shape[-1] < 1:
            raise ValueError("TransformerBaseline 不接受空 batch 或零长度时间轴")
        if not features.is_floating_point():
            raise ValueError("TransformerBaseline features 必须是浮点 tensor")
        valid = self._valid_mask(batch, features)
        sequence = self.input_projection(features.transpose(1, 2))
        sequence = sequence + _sinusoidal_positions(
            sequence.shape[1], self.d_model, sequence
        ).unsqueeze(0)
        encoded = self.encoder(sequence, src_key_padding_mask=~valid)
        weights = valid.unsqueeze(-1).to(encoded.dtype)
        embeddings = (encoded * weights).sum(dim=1) / weights.sum(dim=1)
        embeddings = self.output_norm(embeddings)
        return ModelOutput(logits=self.classifier(embeddings), embeddings=embeddings)


model_registry.register(
    "transformer_baseline",
    TransformerBaseline,
    config_model=TransformerBaselineConfig,
    descriptor=ModelDescriptor(
        id="transformer_baseline",
        display_name="轻量 Transformer 基线",
        description="带动态正弦位置编码和 mask pooling 的帧级 Transformer 分类器。",
        config_schema=TransformerBaselineConfig.model_json_schema(),
        input_layouts={"features": "FT"},
    ),
)


__all__ = ["TransformerBaseline", "TransformerBaselineConfig"]
