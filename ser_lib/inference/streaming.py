"""与设备和 UI 无关的纯 PCM 流式 SER 核心。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from ser_lib.data.types import AudioData
from ser_lib.inference.offline import EmotionPredictor, PredictionResult


@dataclass(frozen=True, slots=True)
class StreamingConfig:
    input_sample_rate: int = 16000
    window_ms: int = 2000
    hop_ms: int = 500
    silence_rms_threshold: float = 0.0
    suppress_silence: bool = True
    smoothing_alpha: float = 1.0
    max_chunk_ms: int = 10000

    def __post_init__(self) -> None:
        if self.input_sample_rate <= 0:
            raise ValueError("input_sample_rate 必须 > 0")
        if self.window_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("window_ms 和 hop_ms 必须 > 0")
        if self.hop_ms > self.window_ms:
            raise ValueError("hop_ms 不能大于 window_ms")
        if self.silence_rms_threshold < 0:
            raise ValueError("silence_rms_threshold 必须 >= 0")
        if not 0 < self.smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha 必须在 (0, 1] 内")
        if self.max_chunk_ms <= 0:
            raise ValueError("max_chunk_ms 必须 > 0")


@dataclass(frozen=True, slots=True)
class StreamingPrediction:
    sequence: int
    start_ms: float
    end_ms: float
    silent: bool
    rms: float
    prediction: PredictionResult | None


@dataclass(frozen=True, slots=True)
class StreamingLatency:
    window_ms: float
    hop_ms: float
    resampler_lookahead_ms: float
    first_result_ms: float


class _LinearResampler:
    """分块方式无关、仅保留下一插值点所需状态的线性重采样器。"""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._buffer = torch.empty(0, dtype=torch.float32)
        self._buffer_start = 0
        self._input_count = 0
        self._output_count = 0

    def push(self, samples: torch.Tensor, *, final: bool = False) -> torch.Tensor:
        if samples.numel():
            self._buffer = torch.cat((self._buffer, samples.cpu()))
            self._input_count += int(samples.numel())
        output: list[torch.Tensor] = []
        while True:
            numerator = self._output_count * self.source_rate
            left = numerator // self.target_rate
            remainder = numerator % self.target_rate
            right = left + (1 if remainder else 0)
            if right >= self._input_count:
                if not final or left >= self._input_count:
                    break
                right = left
                remainder = 0
            local_left = left - self._buffer_start
            local_right = right - self._buffer_start
            fraction = remainder / self.target_rate
            value = self._buffer[local_left] * (1.0 - fraction)
            if local_right != local_left:
                value = value + self._buffer[local_right] * fraction
            output.append(value)
            self._output_count += 1
        next_left = (self._output_count * self.source_rate) // self.target_rate
        discard = max(0, min(next_left - self._buffer_start, self._buffer.numel()))
        if discard:
            self._buffer = self._buffer[discard:]
            self._buffer_start += discard
        return torch.stack(output) if output else torch.empty(0, dtype=torch.float32)

    def reset(self) -> None:
        self._buffer = torch.empty(0, dtype=torch.float32)
        self._buffer_start = 0
        self._input_count = 0
        self._output_count = 0


class StreamingEmotionRecognizer:
    """同步消费 PCM，并为每个完整窗口返回一次预测。"""

    def __init__(self, predictor: EmotionPredictor, config: StreamingConfig) -> None:
        self.predictor = predictor
        self.config = config
        self.target_rate = predictor.audio_loader.config.target_sample_rate
        self.window_samples = round(config.window_ms * self.target_rate / 1000)
        self.hop_samples = round(config.hop_ms * self.target_rate / 1000)
        if self.window_samples < 1 or self.hop_samples < 1:
            raise ValueError("window_ms/hop_ms 在目标采样率下不足一个采样点")
        self._resampler = _LinearResampler(config.input_sample_rate, self.target_rate)
        self._buffer = torch.empty(0, dtype=torch.float32)
        self._consumed = 0
        self._sequence = 0
        self._smoothed: torch.Tensor | None = None
        self._closed = False
        self._flushed = False

    @property
    def buffered_samples(self) -> int:
        return int(self._buffer.numel())

    @property
    def latency(self) -> StreamingLatency:
        lookahead = 0.0 if self.config.input_sample_rate == self.target_rate else (
            1000.0 / self.config.input_sample_rate
        )
        return StreamingLatency(
            float(self.config.window_ms), float(self.config.hop_ms), lookahead,
            float(self.config.window_ms) + lookahead,
        )

    def push_pcm(self, pcm: torch.Tensor | Sequence[float]) -> list[StreamingPrediction]:
        if self._closed:
            raise RuntimeError("流式会话已关闭")
        if self._flushed:
            raise RuntimeError("流式会话已 flush；请 reset 后再输入")
        samples = torch.as_tensor(pcm, dtype=torch.float32)
        if samples.dim() == 2:
            samples = samples.mean(dim=0)
        if samples.dim() != 1:
            raise ValueError("PCM 必须是 [T] 或 [C,T]")
        if not torch.isfinite(samples).all():
            raise ValueError("PCM 包含 NaN/Inf")
        maximum = round(self.config.max_chunk_ms * self.config.input_sample_rate / 1000)
        if samples.numel() > maximum:
            raise BufferError(f"PCM chunk 超过 max_chunk_ms={self.config.max_chunk_ms}")
        converted = self._resampler.push(samples.contiguous())
        if converted.numel():
            self._buffer = torch.cat((self._buffer, converted))
        return self._drain()

    def flush(self, *, pad_final: bool = False) -> list[StreamingPrediction]:
        if self._closed:
            return []
        if self._flushed:
            return []
        self._flushed = True
        tail = self._resampler.push(torch.empty(0), final=True)
        if tail.numel():
            self._buffer = torch.cat((self._buffer, tail))
        results = self._drain()
        if pad_final and self._buffer.numel():
            self._buffer = torch.nn.functional.pad(
                self._buffer, (0, self.window_samples - self._buffer.numel())
            )
            results.extend(self._drain())
            self._buffer = torch.empty(0, dtype=torch.float32)
        return results

    def _drain(self) -> list[StreamingPrediction]:
        results = []
        while self._buffer.numel() >= self.window_samples:
            window = self._buffer[:self.window_samples]
            results.append(self._predict_window(window))
            self._buffer = self._buffer[self.hop_samples:]
            self._consumed += self.hop_samples
        return results

    def _predict_window(self, window: torch.Tensor) -> StreamingPrediction:
        rms = float(window.square().mean().sqrt())
        silent = rms <= self.config.silence_rms_threshold
        prediction = None
        uid = f"stream-{self._sequence:08d}"
        if not (silent and self.config.suppress_silence):
            audio = AudioData(
                waveform=window.unsqueeze(0), sample_rate=self.target_rate,
                source_path=Path("<stream>"), original_sample_rate=self.target_rate,
                num_frames=self.window_samples,
            )
            raw = self.predictor.predict_audio(audio, uid=uid)
            probabilities = torch.tensor(raw.probabilities)
            alpha = self.config.smoothing_alpha
            self._smoothed = probabilities if self._smoothed is None else (
                alpha * probabilities + (1 - alpha) * self._smoothed
            )
            label_id = int(self._smoothed.argmax())
            prediction = PredictionResult(
                uid, label_id, self.predictor.labels.get(label_id, str(label_id)),
                float(self._smoothed[label_id]), self._smoothed.tolist(),
            )
        result = StreamingPrediction(
            sequence=self._sequence,
            start_ms=self._consumed * 1000.0 / self.target_rate,
            end_ms=(self._consumed + self.window_samples) * 1000.0 / self.target_rate,
            silent=silent, rms=rms, prediction=prediction,
        )
        self._sequence += 1
        return result

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("流式会话已关闭")
        self._resampler.reset()
        self._buffer = torch.empty(0, dtype=torch.float32)
        self._consumed = 0
        self._sequence = 0
        self._smoothed = None
        self._flushed = False

    def close(self) -> None:
        self._buffer = torch.empty(0, dtype=torch.float32)
        self._smoothed = None
        self._closed = True


__all__ = [
    "StreamingConfig", "StreamingPrediction", "StreamingLatency",
    "StreamingEmotionRecognizer",
]
