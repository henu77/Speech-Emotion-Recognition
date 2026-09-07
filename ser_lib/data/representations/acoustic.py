"""声学帧级特征表示：F0、RMS、ZCR、谱特征与全局音质向量。

输出协议：

- 帧级特征堆叠为 ``inputs["features"]`` ``[T, D]``（layout ``TD``），
  仅当各帧级特征时间长度一致时堆叠；不一致时抛出
  ``RepresentationError``，禁止在 Dataset 中无条件插值或裁剪“凑齐”长度
  （设计文档 §2.4）。需要不等长时间轴时，请用多个独立的
  ``AcousticFeatures`` 子表示通过 :class:`CompositeRepresentation` 组合。
- 全局音质向量 ``jitter_shimmer_hnr`` 输出 ``inputs["jitter_shimmer_hnr"]``
  ``[3]``（layout ``D``）， utterance 级，不与帧级特征堆叠。
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ser_lib.data.errors import RepresentationError
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.representations.base import Representation
from ser_lib.data.types import (
    LAYOUT_D,
    LAYOUT_TD,
    AudioData,
    RepresentationOutput,
    TensorSpec,
)

AcousticFeatureName = Literal[
    "f0", "rms", "zcr", "spectral_centroid", "spectral_rolloff",
    "spectral_flatness", "spectral_flux", "delta", "jitter_shimmer_hnr",
]

# 各帧级特征的输出维度（堆叠后 [T, D] 的 D）
_FRAME_FEATURE_DIMS: dict[str, int] = {
    "f0": 1,
    "rms": 1,
    "zcr": 1,
    "spectral_centroid": 1,
    "spectral_rolloff": 1,
    "spectral_flatness": 1,
    "spectral_flux": 1,
    "delta": 3,
}
GLOBAL_FEATURE_DIMS: dict[str, int] = {
    "jitter_shimmer_hnr": 3,
}


class AcousticFeaturesConfig(BaseModel):
    """AcousticFeatures 参数。"""

    model_config = ConfigDict(extra="forbid")

    features: list[AcousticFeatureName] = Field(..., min_length=1)
    sample_rate: int = Field(default=16000, ge=1000, le=192000)
    hop_length: int = Field(default=256, ge=1, le=4096)
    # RMS / ZCR 的窗口长度（采样点）
    win_length: int = Field(default=400, ge=2, le=8192)
    # 谱特征的 FFT 参数
    n_fft: int = Field(default=1024, ge=32, le=8192)
    roll_percent: float = Field(default=0.85, gt=0.0, lt=1.0)
    # Delta 的差分窗长（奇数 >= 3）
    delta_win_length: int = Field(default=5, ge=3, le=21)

    @model_validator(mode="after")
    def _validate(self) -> "AcousticFeaturesConfig":
        if len(set(self.features)) != len(self.features):
            raise ValueError(f"features 存在重复项: {self.features}")
        if self.delta_win_length % 2 == 0:
            raise ValueError(f"delta_win_length 必须是奇数，实际: {self.delta_win_length}")
        return self


class _FrameFeature(nn.Module):
    """帧级特征基类：输入 [1, T]，输出 [T_f] 或 [T_f, D]。"""

    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _PitchF0(_FrameFeature):
    def __init__(self, sample_rate: int, hop_length: int) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        pitch = torchaudio.functional.detect_pitch_frequency(
            waveform,
            sample_rate=self.sample_rate,
            frame_time=self.hop_length / self.sample_rate,
            win_length=int(self.sample_rate * 0.03),
            freq_low=50,
            freq_high=800,
        )
        # [1, frames] -> [frames]
        return pitch[0]


class _RMS(_FrameFeature):
    def __init__(self, win_length: int, hop_length: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(
            kernel_size=win_length, stride=hop_length, padding=win_length // 2
        )

    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform.unsqueeze(1)  # [1, 1, T]
        rms = torch.sqrt(self.pool(x ** 2) + 1e-8)
        return rms[0, 0]  # [frames]


class _ZeroCrossingRate(_FrameFeature):
    def __init__(self, win_length: int, hop_length: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(
            kernel_size=win_length, stride=hop_length, padding=win_length // 2
        )

    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        signs = torch.sign(waveform)
        diffs = torch.abs(signs[:, 1:] - signs[:, :-1])
        diffs = F.pad(diffs, (1, 0)).unsqueeze(1)  # [1, 1, T]
        zcr = 0.5 * self.pool(diffs)
        return zcr[0, 0]  # [frames]


class _StftFeature(_FrameFeature):
    """基于 STFT 幅度谱的帧级特征公共基类。"""

    def __init__(self, sample_rate: int, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("_window", torch.hann_window(n_fft), persistent=False)
        freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        self.register_buffer("_freqs", freqs, persistent=False)

    def _magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._window.to(waveform.device),
            return_complex=True,
        ).abs()  # [1, F, frames]

    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _SpectralCentroid(_StftFeature):
    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        S = self._magnitude(waveform)
        freqs = self._freqs.to(waveform.device).unsqueeze(-1)
        centroid = torch.sum(S * freqs, dim=-2) / (torch.sum(S, dim=-2) + 1e-8)
        return centroid[0]  # [frames]


class _SpectralRolloff(_StftFeature):
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int,
                 roll_percent: float) -> None:
        super().__init__(sample_rate, n_fft, hop_length)
        self.roll_percent = roll_percent

    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        S = self._magnitude(waveform)
        total = torch.sum(S, dim=-2, keepdim=True)
        threshold = total * self.roll_percent
        mask = torch.cumsum(S, dim=-2) >= threshold
        idx = torch.argmax(mask.to(torch.int8), dim=-2)
        freqs = self._freqs.to(waveform.device)
        return freqs[idx][0]  # [frames]


class _SpectralFlatness(_StftFeature):
    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        S = self._magnitude(waveform)
        power = S ** 2 + 1e-10
        gmean = torch.exp(torch.mean(torch.log(power), dim=-2))
        amean = torch.mean(power, dim=-2)
        return (gmean / (amean + 1e-10))[0]  # [frames]


class _SpectralFlux(_StftFeature):
    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        S = self._magnitude(waveform)
        diff = F.relu(S[..., 1:] - S[..., :-1])
        flux = torch.sum(diff ** 2, dim=-2)
        return F.pad(flux, (1, 0))[0]  # [frames]


class _Delta(_FrameFeature):
    """对波形按 legacy 语义计算一阶/二阶差分并拼接，输出 [T, 3]。"""

    def __init__(self, win_length: int) -> None:
        super().__init__()
        self.transform = T.ComputeDeltas(win_length=win_length)

    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        # 将 [1, T] 视作单通道特征矩阵 [B=1, C=1, T]
        matrix = waveform.unsqueeze(1)
        delta = self.transform(matrix)
        delta_delta = self.transform(delta)
        stacked = torch.cat([matrix, delta, delta_delta], dim=1)  # [1, 3, T]
        return stacked[0].transpose(0, 1)  # [T, 3]


class _JitterShimmerHNR(nn.Module):
    """utterance 级音质向量 [3] = (jitter, shimmer, hnr)。

    shimmer 与 HNR 需要精确周期检测（Praat 内核），第一版保留 0 占位并在
    descriptor 中声明为近似实现；jitter 基于帧级 F0 的相对微扰。
    """

    def compute(self, waveform: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        valid = pitch > 0
        if int(valid.sum()) < 2:
            return torch.zeros(3, dtype=waveform.dtype, device=waveform.device)
        T0 = 1.0 / (pitch[valid] + 1e-5)
        jitter = torch.mean(torch.abs(T0[1:] - T0[:-1])) / (torch.mean(T0) + 1e-8)
        return torch.stack(
            [jitter, torch.zeros_like(jitter), torch.zeros_like(jitter)]
        )


class AcousticFeatures(Representation):
    """声学帧级/全局特征表示（详见模块 docstring 的输出协议）。"""

    descriptor = ComponentDescriptor(
        id="acoustic_features",
        display_name="声学特征",
        category="representation",
        description="帧级声学特征 (f0/rms/zcr/spectral_*/delta) 堆叠为 [T, D]；"
                    "utterance 级音质向量 jitter_shimmer_hnr 单独输出 [3]。",
        config_schema=AcousticFeaturesConfig.model_json_schema(),
    )

    def __init__(self, **params) -> None:
        config = AcousticFeaturesConfig(**params)
        super().__init__()
        self.config = config
        self._expected_sample_rate = config.sample_rate

        self._frame_modules: dict[str, _FrameFeature] = {}
        for name in config.features:
            if name == "f0":
                self._frame_modules[name] = _PitchF0(config.sample_rate, config.hop_length)
            elif name == "rms":
                self._frame_modules[name] = _RMS(config.win_length, config.hop_length)
            elif name == "zcr":
                self._frame_modules[name] = _ZeroCrossingRate(config.win_length, config.hop_length)
            elif name == "spectral_centroid":
                self._frame_modules[name] = _SpectralCentroid(
                    config.sample_rate, config.n_fft, config.hop_length)
            elif name == "spectral_rolloff":
                self._frame_modules[name] = _SpectralRolloff(
                    config.sample_rate, config.n_fft, config.hop_length, config.roll_percent)
            elif name == "spectral_flatness":
                self._frame_modules[name] = _SpectralFlatness(
                    config.sample_rate, config.n_fft, config.hop_length)
            elif name == "spectral_flux":
                self._frame_modules[name] = _SpectralFlux(
                    config.sample_rate, config.n_fft, config.hop_length)
            elif name == "delta":
                self._frame_modules[name] = _Delta(config.delta_win_length)

        self._has_global = "jitter_shimmer_hnr" in config.features
        if self._has_global:
            self._quality = _JitterShimmerHNR()
        # 供 nn.Module 管理 buffer/module 注册
        self._feature_modules = nn.ModuleDict(
            {f"feat_{k}": v for k, v in self._frame_modules.items()}
        )

    # ------------------------------------------------------------------

    @property
    def output_specs(self) -> dict[str, TensorSpec]:
        specs: dict[str, TensorSpec] = {}
        if self._frame_names:
            feature_dim = sum(_FRAME_FEATURE_DIMS[name] for name in self._frame_names)
            specs["features"] = TensorSpec(layout=LAYOUT_TD, feature_dim=feature_dim)
        if self._has_global:
            specs["jitter_shimmer_hnr"] = TensorSpec(
                layout=LAYOUT_D, feature_dim=GLOBAL_FEATURE_DIMS["jitter_shimmer_hnr"]
            )
        return specs

    @property
    def _frame_names(self) -> list[str]:
        return [name for name in self.config.features if name in _FRAME_FEATURE_DIMS]

    def forward(self, audio: AudioData) -> RepresentationOutput:
        self._require_mono(audio)
        self._require_sample_rate(audio)
        waveform = audio.waveform  # [1, T]

        inputs: dict[str, torch.Tensor] = {}
        lengths: dict[str, int] = {}

        frame_tensors: list[torch.Tensor] = []
        if self._frame_names:
            computed: dict[str, torch.Tensor] = {}
            for name in self._frame_names:
                value = self._frame_modules[name].compute(waveform)
                if value.dim() == 1:
                    value = value.unsqueeze(-1)  # [T_f] -> [T_f, 1]
                computed[name] = value
                frame_tensors.append(value)

            ref = frame_tensors[0].shape[0]
            mismatched = {
                name: value.shape[0]
                for name, value in computed.items()
                if value.shape[0] != ref
            }
            if mismatched:
                raise RepresentationError(
                    f"帧级特征时间长度不一致: {mismatched}（参考帧数 {ref}）。"
                    f"禁止自动插值或裁剪对齐；如需不等长时间轴，请将各特征拆分为"
                    f"独立的 AcousticFeatures 子表示并通过 CompositeRepresentation 组合",
                    path=audio.source_path,
                    component=self.descriptor.id,
                    stage="representation",
                )
            features = torch.cat(frame_tensors, dim=-1)  # [T, D]
            inputs["features"] = features
            lengths["features"] = int(features.shape[0])

        if self._has_global:
            pitch = None
            if "f0" in self._frame_modules:
                pitch = self._frame_modules["f0"].compute(waveform)
            else:
                pitch = _PitchF0(self.config.sample_rate, self.config.hop_length).compute(waveform)
            quality = self._quality.compute(waveform, pitch)
            inputs["jitter_shimmer_hnr"] = quality
            # 全局向量不进入 lengths

        return RepresentationOutput(inputs=inputs, lengths=lengths)
