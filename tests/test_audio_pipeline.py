import math
import struct
import wave
from pathlib import Path

import pytest
import torch

from ser_lib.data.audio import AudioLoader, AudioLoaderConfig
from ser_lib.data.collate import SERCollator
from ser_lib.data.config import BatchingConfig
from ser_lib.data.dataset import SERDataset
from ser_lib.data.errors import AudioNotFoundError, InvalidAudioSegmentError
from ser_lib.data.pipeline import SamplePipeline
from ser_lib.data.representations.waveform import RawWaveform
from ser_lib.data.types import AudioRecord


def _write_pcm_wav(path: Path, *, sample_rate=8000, seconds=0.1, channels=1):
    frame_count = int(sample_rate * seconds)
    frames = bytearray()
    for index in range(frame_count):
        value = int(8000 * math.sin(2 * math.pi * 220 * index / sample_rate))
        frames.extend(struct.pack("<h", value) * channels)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)


def test_audio_loader_resamples_and_preserves_channel_axis(tmp_path: Path):
    path = tmp_path / "中文 音频.wav"
    _write_pcm_wav(path, sample_rate=8000, channels=2)
    loader = AudioLoader(AudioLoaderConfig(target_sample_rate=16000, mono=True))
    audio = loader.load(AudioRecord(uid="a", audio_path=path))
    assert audio.waveform.dtype == torch.float32
    assert audio.waveform.shape[0] == 1
    assert 1500 <= audio.waveform.shape[-1] <= 1700
    assert audio.original_sample_rate == 8000
    assert audio.sample_rate == 16000


def test_audio_loader_reads_segment_without_loading_contract_change(tmp_path: Path):
    path = tmp_path / "audio.wav"
    _write_pcm_wav(path, sample_rate=8000, seconds=0.2)
    loader = AudioLoader(AudioLoaderConfig(target_sample_rate=8000))
    audio = loader.load(AudioRecord(uid="segment", audio_path=path, start_ms=50, end_ms=100))
    assert audio.waveform.shape == (1, 400)


def test_audio_loader_reports_missing_file_with_uid(tmp_path: Path):
    with pytest.raises(AudioNotFoundError, match="uid=missing"):
        AudioLoader().load(AudioRecord(uid="missing", audio_path=tmp_path / "none.wav"))


def test_audio_loader_rejects_start_past_end_of_file(tmp_path: Path):
    path = tmp_path / "audio.wav"
    _write_pcm_wav(path, seconds=0.1)
    with pytest.raises(InvalidAudioSegmentError, match="片段起点超出"):
        AudioLoader(AudioLoaderConfig(target_sample_rate=8000)).load(
            AudioRecord(uid="bad", audio_path=path, start_ms=200)
        )


def test_minimal_waveform_pipeline_end_to_end(tmp_path: Path):
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    _write_pcm_wav(first, seconds=0.05)
    _write_pcm_wav(second, seconds=0.08)
    loader = AudioLoader(AudioLoaderConfig(target_sample_rate=8000))
    pipeline = SamplePipeline(RawWaveform())
    dataset = SERDataset(
        [
            AudioRecord(uid="a", audio_path=first, label=0),
            AudioRecord(uid="b", audio_path=second, label=1),
        ],
        loader,
        pipeline,
    )
    batch = SERCollator(pipeline.output_specs, BatchingConfig(type="dynamic"))(
        [dataset[0], dataset[1]]
    )
    assert batch.inputs["waveform"].shape == (2, 640)
    assert batch.lengths["waveform"].tolist() == [400, 640]
    assert batch.masks["waveform"].sum(dim=1).tolist() == [400, 640]
    assert batch.labels.tolist() == [0, 1]
