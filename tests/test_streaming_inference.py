from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ser_lib.inference import (
    PredictionResult,
    StreamingConfig,
    StreamingEmotionRecognizer,
)


class FakePredictor:
    def __init__(self, target_rate: int, probabilities=(0.25, 0.75)) -> None:
        self.audio_loader = SimpleNamespace(
            config=SimpleNamespace(target_sample_rate=target_rate)
        )
        self.labels = {0: "neutral", 1: "happy"}
        self.probabilities = list(probabilities)
        self.windows: list[torch.Tensor] = []

    def predict_audio(self, audio, *, uid):
        self.windows.append(audio.waveform.squeeze(0).clone())
        label = int(torch.tensor(self.probabilities).argmax())
        return PredictionResult(
            uid, label, self.labels[label], self.probabilities[label],
            list(self.probabilities),
        )


def _session(predictor, **updates):
    return StreamingEmotionRecognizer(
        predictor,
        StreamingConfig(
            input_sample_rate=updates.pop("input_sample_rate", 8),
            window_ms=updates.pop("window_ms", 500),
            hop_ms=updates.pop("hop_ms", 250),
            max_chunk_ms=updates.pop("max_chunk_ms", 2000),
            **updates,
        ),
    )


def test_streaming_windows_are_independent_of_chunk_partition():
    pcm = torch.arange(10, dtype=torch.float32)
    whole_predictor = FakePredictor(8)
    chunked_predictor = FakePredictor(8)
    whole = _session(whole_predictor)
    chunked = _session(chunked_predictor)

    whole_results = whole.push_pcm(pcm)
    chunked_results = []
    for part in (pcm[:1], pcm[1:4], pcm[4:5], pcm[5:]):
        chunked_results.extend(chunked.push_pcm(part))

    assert len(whole_results) == len(chunked_results) == 4
    assert all(
        torch.equal(left, right)
        for left, right in zip(whole_predictor.windows, chunked_predictor.windows)
    )
    assert [item.start_ms for item in whole_results] == [0, 250, 500, 750]
    assert whole.buffered_samples < whole.window_samples


def test_stateful_resampling_is_chunk_partition_invariant():
    pcm = torch.linspace(-1, 1, 12)
    one_predictor = FakePredictor(4)
    many_predictor = FakePredictor(4)
    one = _session(
        one_predictor, input_sample_rate=8, window_ms=1000, hop_ms=500
    )
    many = _session(
        many_predictor, input_sample_rate=8, window_ms=1000, hop_ms=500
    )
    one_results = one.push_pcm(pcm) + one.flush()
    many_results = []
    for part in (pcm[:2], pcm[2:7], pcm[7:9], pcm[9:]):
        many_results.extend(many.push_pcm(part))
    many_results.extend(many.flush())

    assert len(one_results) == len(many_results)
    assert all(
        torch.equal(left, right)
        for left, right in zip(one_predictor.windows, many_predictor.windows)
    )


def test_silence_suppression_and_probability_smoothing():
    predictor = FakePredictor(8, probabilities=(0.8, 0.2))
    session = _session(
        predictor, silence_rms_threshold=0.01, smoothing_alpha=0.5
    )
    silent = session.push_pcm(torch.zeros(4))[0]
    assert silent.silent is True
    assert silent.prediction is None
    assert not predictor.windows

    first = session.push_pcm(torch.ones(2))[0]
    assert first.prediction is not None
    predictor.probabilities = [0.2, 0.8]
    second = session.push_pcm(torch.ones(2))[0]
    assert second.prediction.probabilities == pytest.approx([0.5, 0.5])


def test_backpressure_reset_flush_and_close_lifecycle():
    predictor = FakePredictor(8)
    session = _session(predictor, max_chunk_ms=500)
    with pytest.raises(BufferError, match="max_chunk_ms"):
        session.push_pcm(torch.ones(5))
    session.push_pcm(torch.ones(3))
    assert session.flush(pad_final=True)
    assert session.flush() == []
    with pytest.raises(RuntimeError, match="flush"):
        session.push_pcm(torch.ones(1))

    session.reset()
    assert session.buffered_samples == 0
    assert session.push_pcm(torch.ones(4))[0].sequence == 0
    latency = session.latency
    assert latency.first_result_ms == 500
    session.close()
    assert session.buffered_samples == 0
    with pytest.raises(RuntimeError, match="关闭"):
        session.push_pcm(torch.ones(1))


def test_long_running_session_keeps_buffer_bounded():
    session = _session(FakePredictor(8))
    for _ in range(100):
        session.push_pcm(torch.ones(2))
        assert session.buffered_samples < session.window_samples
