from pathlib import Path

import torch

from ser_lib.data.cache import CachedRepresentation
from ser_lib.data.config import (
    AudioSettings, BatchingConfig, CacheSettings, ComponentConfig, DataConfig,
)
from ser_lib.data.pipeline import build_pipeline
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.representations.base import Representation
from ser_lib.data.types import AudioData, RepresentationOutput, TensorSpec


class CountingRepresentation(Representation):
    descriptor = ComponentDescriptor(
        id="counting", display_name="Counting", category="representation", version="1"
    )

    def __init__(self):
        super().__init__()
        self.calls = 0

    @property
    def output_specs(self):
        return {"features": TensorSpec(layout="FT", feature_dim=1)}

    def forward(self, audio):
        self.calls += 1
        value = audio.waveform.mean(0, keepdim=True)
        return RepresentationOutput({"features": value}, {"features": value.shape[-1]})


def _audio(path: Path, values=None):
    return AudioData(
        waveform=torch.tensor(values or [[0.0, 1.0, 2.0]], dtype=torch.float32),
        sample_rate=16000,
        source_path=path,
        original_sample_rate=16000,
        num_frames=3,
    )


def test_cached_representation_hits_and_content_change_misses(tmp_path: Path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    inner = CountingRepresentation()
    cached = CachedRepresentation(inner, tmp_path / "cache")
    first = cached(_audio(source))
    second = cached(_audio(source))
    assert inner.calls == 1
    assert torch.equal(first.inputs["features"], second.inputs["features"])
    cached(_audio(source, [[0.0, 1.0, 3.0]]))
    assert inner.calls == 2
    assert cached.entry_count() == 2
    assert cached.size_bytes() > 0


def test_corrupted_cache_entry_is_recomputed(tmp_path: Path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    inner = CountingRepresentation()
    cached = CachedRepresentation(inner, tmp_path / "cache")
    cached(_audio(source))
    entry = next((tmp_path / "cache").glob("*/*.pt"))
    entry.write_bytes(b"broken")
    output = cached(_audio(source))
    assert inner.calls == 2
    assert output.lengths == {"features": 3}


def test_configured_cache_wraps_representation(tmp_path: Path):
    config = DataConfig(
        manifest=tmp_path / "unused.yaml",
        audio=AudioSettings(target_sample_rate=16000),
        cache=CacheSettings(enabled=True, directory=tmp_path / "cache"),
        representation=ComponentConfig(type="waveform"),
        batching=BatchingConfig(type="dynamic"),
    )
    pipeline = build_pipeline(config, train=False)
    assert isinstance(pipeline.representation, CachedRepresentation)


def test_cache_rejects_training_waveform_augmentation(tmp_path: Path):
    config = DataConfig(
        manifest=tmp_path / "unused.yaml",
        cache=CacheSettings(enabled=True, directory=tmp_path / "cache"),
        representation=ComponentConfig(type="waveform"),
        waveform_transforms=[
            ComponentConfig(type="gaussian_noise", probability=0.5)
        ],
        batching=BatchingConfig(type="dynamic"),
    )
    try:
        build_pipeline(config, train=True)
    except ValueError as exc:
        assert "禁止" in str(exc)
    else:
        raise AssertionError("随机增强后的缓存应被拒绝")
