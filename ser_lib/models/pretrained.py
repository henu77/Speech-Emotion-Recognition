"""可选的 Hugging Face 预训练语音编码器适配。"""

from __future__ import annotations

import importlib
from typing import Any, Literal

import torch
import torch.nn.functional as F
from pydantic import Field, model_validator
from torch import nn

from ser_lib.core.config import StrictConfig
from ser_lib.data.types import SERBatch, TensorSpec
from ser_lib.data.validation import ModelSpec
from ser_lib.models.base import ModelOutput, SERModel
from ser_lib.models.registry import ModelDescriptor, model_registry


class HFAudioClassifierConfig(StrictConfig):
    num_classes: int = Field(ge=2)
    pretrained_model_name_or_path: str | None = None
    encoder_config: dict[str, Any] | None = None
    local_files_only: bool = True
    revision: str | None = None
    freeze_encoder: bool = False
    dropout: float = Field(default=0.1, ge=0, lt=1)
    pooling: Literal["mean", "max"] = "mean"
    expected_sample_rate: int = Field(default=16000, ge=1000, le=192000)

    @model_validator(mode="after")
    def _exactly_one_encoder_source(self) -> "HFAudioClassifierConfig":
        supplied = sum(value is not None for value in (
            self.pretrained_model_name_or_path, self.encoder_config
        ))
        if supplied != 1:
            raise ValueError(
                "pretrained_model_name_or_path 与 encoder_config 必须且只能提供一个"
            )
        if self.pretrained_model_name_or_path == "":
            raise ValueError("pretrained_model_name_or_path 不能为空")
        return self


def _transformers():
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "hf_audio_classifier 需要可选依赖；请安装 ser-lib[pretrained]"
        ) from exc


class HFAudioClassifier(SERModel):
    """为 Hugging Face AutoModel 语音编码器增加掩码池化与分类头。

    默认 ``local_files_only=True``，构造期间不会隐式下载；本适配器始终禁用
    ``trust_remote_code``，仅支持 Transformers 已注册的受控模型架构。
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_model_name_or_path: str | None = None,
        encoder_config: dict[str, Any] | None = None,
        local_files_only: bool = True,
        revision: str | None = None,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
        pooling: Literal["mean", "max"] = "mean",
        expected_sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        config = HFAudioClassifierConfig(
            num_classes=num_classes,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            encoder_config=encoder_config,
            local_files_only=local_files_only,
            revision=revision,
            freeze_encoder=freeze_encoder,
            dropout=dropout,
            pooling=pooling,
            expected_sample_rate=expected_sample_rate,
        )
        transformers = _transformers()
        if config.encoder_config is not None:
            raw = dict(config.encoder_config)
            model_type = raw.pop("model_type", None)
            if not isinstance(model_type, str) or not model_type:
                raise ValueError("encoder_config 必须包含非空 model_type")
            hf_config = transformers.AutoConfig.for_model(model_type, **raw)
            self.encoder = transformers.AutoModel.from_config(hf_config)
        else:
            self.encoder = transformers.AutoModel.from_pretrained(
                config.pretrained_model_name_or_path,
                local_files_only=config.local_files_only,
                revision=config.revision,
                trust_remote_code=False,
            )
        serialized = self.encoder.config.to_dict()
        if not isinstance(serialized, dict) or not serialized.get("model_type"):
            raise ValueError("预训练编码器 config.to_dict() 必须包含 model_type")
        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("预训练编码器配置缺少合法 hidden_size")
        self.num_classes = config.num_classes
        self.encoder_config = serialized
        self.local_files_only = config.local_files_only
        self.revision = config.revision
        self.freeze_encoder = config.freeze_encoder
        self.dropout_probability = config.dropout
        self.pooling = config.pooling
        self.expected_sample_rate = config.expected_sample_rate
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, config.num_classes)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            model_id="hf_audio_classifier",
            required_inputs={"waveform": TensorSpec(layout="T")},
            supports_masks=True,
            supports_variable_length=True,
            num_classes=self.num_classes,
            expected_sample_rate=self.expected_sample_rate,
        )

    @property
    def model_config(self) -> dict[str, Any]:
        # 固化架构配置而不是外部模型路径，使 artifact 可离线、自包含地恢复。
        return HFAudioClassifierConfig(
            num_classes=self.num_classes,
            encoder_config=self.encoder_config,
            local_files_only=True,
            revision=None,
            freeze_encoder=self.freeze_encoder,
            dropout=self.dropout_probability,
            pooling=self.pooling,
            expected_sample_rate=self.expected_sample_rate,
        ).model_dump(mode="json")

    def train(self, mode: bool = True) -> "HFAudioClassifier":
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _mask(self, batch: SERBatch, waveform: torch.Tensor) -> torch.Tensor:
        batch_size, time = waveform.shape
        lengths = batch.lengths.get("waveform")
        mask = batch.masks.get("waveform")
        if lengths is None and mask is None:
            return torch.ones(batch_size, time, dtype=torch.bool, device=waveform.device)
        if lengths is not None:
            if lengths.shape != (batch_size,):
                raise ValueError("waveform lengths 必须是 [B]")
            if torch.any(lengths <= 0) or torch.any(lengths > time):
                raise ValueError("waveform lengths 必须位于 [1,T]")
            expected = torch.arange(time, device=waveform.device).unsqueeze(0) < (
                lengths.to(waveform.device).unsqueeze(1)
            )
            if mask is None:
                return expected
        if mask is None or mask.shape != (batch_size, time):
            raise ValueError("waveform mask 必须是 [B,T]")
        mask = mask.to(device=waveform.device, dtype=torch.bool)
        if lengths is not None and not torch.equal(mask, expected):
            raise ValueError("waveform mask 必须是由 lengths 定义的连续前缀 mask")
        if torch.any(mask.sum(1) == 0):
            raise ValueError("waveform mask 每行至少包含一个有效采样点")
        return mask

    def forward(self, batch: SERBatch) -> ModelOutput:
        waveform = batch.inputs.get("waveform")
        if waveform is None:
            raise ValueError("HFAudioClassifier 需要 batch.inputs['waveform']")
        if waveform.dim() != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1:
            raise ValueError("HFAudioClassifier 期望非空 waveform [B,T]")
        if not waveform.is_floating_point():
            raise ValueError("waveform 必须是浮点 tensor")
        valid = self._mask(batch, waveform)
        output = self.encoder(
            input_values=waveform,
            attention_mask=valid.to(torch.long),
            return_dict=True,
        )
        hidden = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            raise ValueError("预训练编码器必须返回 last_hidden_state [B,T,D]")
        encoded_valid = F.interpolate(
            valid.unsqueeze(1).to(torch.float32), size=hidden.shape[1], mode="nearest"
        ).squeeze(1).to(torch.bool)
        if self.pooling == "mean":
            weights = encoded_valid.unsqueeze(-1).to(hidden.dtype)
            embeddings = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        else:
            embeddings = hidden.masked_fill(~encoded_valid.unsqueeze(-1), float("-inf")).max(1).values
        return ModelOutput(
            logits=self.classifier(self.dropout(embeddings)), embeddings=embeddings
        )


model_registry.register(
    "hf_audio_classifier",
    HFAudioClassifier,
    config_model=HFAudioClassifierConfig,
    descriptor=ModelDescriptor(
        id="hf_audio_classifier",
        display_name="Hugging Face 语音编码器分类器",
        description="可选依赖、默认仅本地加载且 artifact 可离线恢复的语音编码器适配。",
        config_schema=HFAudioClassifierConfig.model_json_schema(),
        input_layouts={"waveform": "T"},
        status="optional",
    ),
)


__all__ = ["HFAudioClassifier", "HFAudioClassifierConfig"]
