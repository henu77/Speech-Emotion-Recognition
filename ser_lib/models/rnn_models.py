"""适用于帧级声学表示的循环神经网络基线。"""

from __future__ import annotations

import torch
from pydantic import Field, model_validator
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ser_lib.core.config import StrictConfig
from ser_lib.data.types import SERBatch, TensorSpec
from ser_lib.data.validation import ModelSpec
from ser_lib.models.base import ModelOutput, SERModel
from ser_lib.models.registry import ModelDescriptor, model_registry


class GRUBaselineConfig(StrictConfig):
    feature_dim: int = Field(ge=1)
    num_classes: int = Field(ge=2)
    hidden_dim: int = Field(default=128, ge=1)
    num_layers: int = Field(default=1, ge=1)
    bidirectional: bool = True
    dropout: float = Field(default=0.0, ge=0, lt=1)

    @model_validator(mode="after")
    def _dropout_requires_multiple_layers(self) -> "GRUBaselineConfig":
        # PyTorch GRU 在单层时忽略内部 dropout；配置中拒绝这种隐式降级。
        if self.num_layers == 1 and self.dropout != 0:
            raise ValueError("num_layers=1 时 dropout 必须为 0")
        return self


class GRUBaseline(SERModel):
    """使用 packed sequence 忽略 padding 的 GRU 分类基线。

    输入 ``features`` 为 ``[B,F,T]``，输出 embedding 是最后一层最终隐藏状态；
    双向模式会拼接正向与反向状态。
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        config = GRUBaselineConfig(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.feature_dim = config.feature_dim
        self.num_classes = config.num_classes
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.dropout = config.dropout
        self.encoder = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        embedding_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        self.classifier = nn.Linear(embedding_dim, self.num_classes)

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            model_id="gru_baseline",
            required_inputs={
                "features": TensorSpec(layout="FT", feature_dim=self.feature_dim)
            },
            supports_masks=True,
            supports_variable_length=True,
            num_classes=self.num_classes,
        )

    @property
    def model_config(self) -> dict[str, int | float | bool]:
        return GRUBaselineConfig(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
        ).model_dump(mode="json")

    def _lengths(self, batch: SERBatch, features: torch.Tensor) -> torch.Tensor:
        lengths = batch.lengths.get("features")
        mask = batch.masks.get("features")
        if lengths is None:
            lengths = mask.sum(dim=-1) if mask is not None else torch.full(
                (features.shape[0],), features.shape[-1], dtype=torch.long,
                device=features.device,
            )
        if lengths.shape != (features.shape[0],):
            raise ValueError(f"features lengths 必须是 [B]，实际 {tuple(lengths.shape)}")
        if torch.any(lengths <= 0) or torch.any(lengths > features.shape[-1]):
            raise ValueError("features lengths 必须位于 [1,T]")
        if mask is not None:
            if mask.shape != (features.shape[0], features.shape[-1]):
                raise ValueError("features mask 必须是 [B,T]")
            if not torch.equal(mask.sum(dim=-1).to(lengths.device), lengths):
                raise ValueError("features mask 的有效数量必须与 lengths 一致")
        return lengths

    def forward(self, batch: SERBatch) -> ModelOutput:
        if "features" not in batch.inputs:
            raise ValueError("GRUBaseline 需要 batch.inputs['features']")
        features = batch.inputs["features"]
        if features.dim() != 3 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"GRUBaseline 期望 [B,{self.feature_dim},T]，实际 {tuple(features.shape)}"
            )
        if features.shape[0] < 1 or features.shape[-1] < 1:
            raise ValueError("GRUBaseline 不接受空 batch 或零长度时间轴")
        if not features.is_floating_point():
            raise ValueError(f"GRUBaseline features 必须是浮点 tensor，实际 {features.dtype}")
        lengths = self._lengths(batch, features)
        packed = pack_padded_sequence(
            features.transpose(1, 2), lengths.detach().cpu(),
            batch_first=True, enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        if self.bidirectional:
            embeddings = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            embeddings = hidden[-1]
        return ModelOutput(logits=self.classifier(embeddings), embeddings=embeddings)


model_registry.register(
    "gru_baseline",
    GRUBaseline,
    config_model=GRUBaselineConfig,
    descriptor=ModelDescriptor(
        id="gru_baseline",
        display_name="GRU 基线",
        description="使用 packed sequence 的单向或双向 GRU 时序分类器。",
        config_schema=GRUBaselineConfig.model_json_schema(),
        input_layouts={"features": "FT"},
    ),
)


__all__ = ["GRUBaseline", "GRUBaselineConfig"]
