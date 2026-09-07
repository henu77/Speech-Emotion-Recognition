"""谱图类表示：Spectrogram、MelSpectrogram、LogMel、MFCC（设计文档 §9.2）。

统一输出 key ``features``，layout ``FT``（``[F, Tm]``）；Mel 参数转换逻辑
封装在组件内部，不暴露给 Dataset。
"""

from __future__ import annotations

import torchaudio.transforms as T
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ser_lib.data.errors import RepresentationError
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.representations.base import Representation
from ser_lib.data.types import (
    LAYOUT_FT,
    AudioData,
    RepresentationOutput,
    TensorSpec,
)


class SpectralConfigBase(BaseModel):
    """谱图类表示的公共参数（对齐 torchaudio 默认值）。"""

    model_config = ConfigDict(extra="forbid")

    sample_rate: int = Field(default=16000, ge=1000, le=192000)
    n_fft: int = Field(default=400, ge=32, le=8192)
    win_length: int | None = Field(default=None, ge=32, le=8192)
    hop_length: int = Field(default=200, ge=1, le=4096)
    f_min: float = Field(default=0.0, ge=0.0)
    f_max: float | None = Field(default=None, ge=0.0)
    center: bool = True
    pad_mode: str = "reflect"

    @model_validator(mode="after")
    def _validate_params(self) -> "SpectralConfigBase":
        win = self.win_length if self.win_length is not None else self.n_fft
        if win > self.n_fft:
            raise ValueError(f"win_length ({win}) 不能大于 n_fft ({self.n_fft})")
        if self.f_max is not None and self.f_max <= self.f_min:
            raise ValueError(f"f_max ({self.f_max}) 必须大于 f_min ({self.f_min})")
        return self


class SpectrogramConfig(SpectralConfigBase):
    """线性谱图参数。power 固定为 2.0（幅度谱使用 power=None 的场景第一版不暴露）。"""

    power: float = Field(default=2.0, ge=0.0, le=3.0)


class MelConfig(SpectralConfigBase):
    """Mel 谱公共参数。"""

    n_mels: int = Field(default=80, ge=16, le=512)
    power: float = Field(default=2.0, ge=1.0, le=3.0)


class LogMelConfig(MelConfig):
    """Log-Mel 参数。``top_db`` 传给 AmplitudeToDB。"""

    top_db: float = Field(default=80.0, ge=10.0, le=120.0)


class MFCCConfig(SpectralConfigBase):
    """MFCC 参数；Mel 参数封装在组件内部。"""

    n_mels: int = Field(default=80, ge=16, le=512)
    n_mfcc: int = Field(default=40, ge=10, le=128)
    mel_norm: str | None = "slaney"
    mel_scale: str = "htk"

    @model_validator(mode="after")
    def _validate_mfcc(self) -> "MFCCConfig":
        if self.n_mfcc > self.n_mels:
            raise ValueError(f"n_mfcc ({self.n_mfcc}) 不能大于 n_mels ({self.n_mels})")
        return self


class _SpectralRepresentationBase(Representation):
    """谱图类表示公共实现：单声道输入 + 采样率校验 + ``features`` 输出。"""

    def __init__(self, config: SpectralConfigBase) -> None:
        super().__init__()
        self.config = config
        self._expected_sample_rate = config.sample_rate

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        return {
            "features": TensorSpec(
                layout=LAYOUT_FT,
                feature_dim=self._feature_dim,
            )
        }

    @property
    def _feature_dim(self) -> int:
        raise NotImplementedError

    def _to_output(self, features: "T.Tensor", audio: AudioData) -> RepresentationOutput:
        # [1, T] 输入 -> [F, Tm]（去掉 batch/channel 维）
        if features.dim() != 3:
            raise RepresentationError(
                f"谱图提取输出维度错误: 期望 3D [C, F, T]，实际 "
                f"{features.dim()}D {tuple(features.shape)}",
                path=audio.source_path,
                component=self.descriptor.id,
                stage="representation",
            )
        features = features[0]  # [F, Tm]
        return RepresentationOutput(
            inputs={"features": features},
            lengths={"features": int(features.shape[1])},
        )


class SpectrogramRepresentation(_SpectralRepresentationBase):
    """线性幅度谱（power 谱）。"""

    descriptor = ComponentDescriptor(
        id="spectrogram",
        display_name="线性谱图",
        category="representation",
        description="输出 [F, T] 幂谱/幅度谱。",
        config_schema=SpectrogramConfig.model_json_schema(),
    )

    def __init__(self, **params) -> None:
        config = SpectrogramConfig(**params)
        super().__init__(config)
        self.transform = T.Spectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            power=config.power,
            center=config.center,
            pad_mode=config.pad_mode,
        )

    @property
    def _feature_dim(self) -> int:
        return self.config.n_fft // 2 + 1

    def forward(self, audio: AudioData) -> RepresentationOutput:
        self._require_mono(audio)
        self._require_sample_rate(audio)
        return self._to_output(self.transform(audio.waveform), audio)


class MelSpectrogramRepresentation(_SpectralRepresentationBase):
    """Mel 谱。"""

    descriptor = ComponentDescriptor(
        id="mel_spectrogram",
        display_name="Mel 谱",
        category="representation",
        description="输出 [F, T] Mel 谱。",
        config_schema=MelConfig.model_json_schema(),
    )

    def __init__(self, **params) -> None:
        config = MelConfig(**params)
        super().__init__(config)
        self.transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=config.power,
            center=config.center,
            pad_mode=config.pad_mode,
        )

    @property
    def _feature_dim(self) -> int:
        return self.config.n_mels

    def forward(self, audio: AudioData) -> RepresentationOutput:
        self._require_mono(audio)
        self._require_sample_rate(audio)
        return self._to_output(self.transform(audio.waveform), audio)


class LogMelRepresentation(_SpectralRepresentationBase):
    """Log-Mel 谱：Mel 谱后接 AmplitudeToDB。"""

    descriptor = ComponentDescriptor(
        id="log_mel",
        display_name="Log-Mel 谱",
        category="representation",
        description="输出 [F, T] Log-Mel 谱（AmplitudeToDB, top_db 可配）。",
        config_schema=LogMelConfig.model_json_schema(),
    )

    def __init__(self, **params) -> None:
        config = LogMelConfig(**params)
        super().__init__(config)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=config.power,
            center=config.center,
            pad_mode=config.pad_mode,
        )
        self.db_transform = T.AmplitudeToDB(top_db=config.top_db)

    @property
    def _feature_dim(self) -> int:
        return self.config.n_mels

    def forward(self, audio: AudioData) -> RepresentationOutput:
        self._require_mono(audio)
        self._require_sample_rate(audio)
        mel = self.mel_transform(audio.waveform)
        return self._to_output(self.db_transform(mel), audio)


class MFCCRepresentation(_SpectralRepresentationBase):
    """MFCC：Mel 参数转换逻辑封装在组件内部。"""

    descriptor = ComponentDescriptor(
        id="mfcc",
        display_name="MFCC",
        category="representation",
        description="输出 [n_mfcc, T] MFCC 特征。",
        config_schema=MFCCConfig.model_json_schema(),
    )

    def __init__(self, **params) -> None:
        config = MFCCConfig(**params)
        super().__init__(config)
        self.transform = T.MFCC(
            sample_rate=config.sample_rate,
            n_mfcc=config.n_mfcc,
            norm=config.mel_norm,
            melkwargs={
                "n_fft": config.n_fft,
                "win_length": config.win_length,
                "hop_length": config.hop_length,
                "f_min": config.f_min,
                "f_max": config.f_max,
                "n_mels": config.n_mels,
                "center": config.center,
                "pad_mode": config.pad_mode,
                "norm": config.mel_norm,
                "mel_scale": config.mel_scale,
            },
        )

    @property
    def _feature_dim(self) -> int:
        return self.config.n_mfcc

    def forward(self, audio: AudioData) -> RepresentationOutput:
        self._require_mono(audio)
        self._require_sample_rate(audio)
        return self._to_output(self.transform(audio.waveform), audio)


__all__ = [
    "SpectrogramConfig",
    "MelConfig",
    "LogMelConfig",
    "MFCCConfig",
    "SpectrogramRepresentation",
    "MelSpectrogramRepresentation",
    "LogMelRepresentation",
    "MFCCRepresentation",
]
