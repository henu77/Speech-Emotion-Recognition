from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from ser_lib.artifacts import export_model_artifact, load_model_artifact
from ser_lib.data import BatchingConfig, SERCollator, SERSample, TensorSpec
from ser_lib.data.config import AudioSettings, ComponentConfig, DataConfig
from ser_lib.engine import Trainer, TrainerConfig
from ser_lib.models import (
    TransformerBaseline,
    TransformerBaselineConfig,
    model_registry,
)


SPECS = {"features": TensorSpec(layout="FT", feature_dim=4)}


def _samples(first: torch.Tensor, second: torch.Tensor | None = None):
    tensors = [first] if second is None else [first, second]
    return SERCollator(SPECS, BatchingConfig(type="dynamic"))([
        SERSample(
            f"sample-{index}", {"features": value},
            {"features": value.shape[-1]}, index % 2, {},
        )
        for index, value in enumerate(tensors)
    ])


def _data_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        manifest=tmp_path / "unused.yaml",
        audio=AudioSettings(target_sample_rate=16000),
        representation=ComponentConfig(
            type="log_mel",
            params={"sample_rate": 16000, "n_mels": 16, "n_fft": 128,
                    "win_length": 128, "hop_length": 64, "f_max": 8000},
        ),
        batching=BatchingConfig(type="dynamic"),
        labels={0: {"en": "neutral"}, 1: {"en": "happy"}},
    )


def test_transformer_registry_config_and_forward_contract():
    model = model_registry.create(
        "transformer_baseline", feature_dim=4, num_classes=2, d_model=8,
        num_heads=2, num_layers=1, feedforward_dim=16, dropout=0,
    )
    batch = _samples(torch.randn(4, 5), torch.randn(4, 3))
    output = model(batch)
    assert output.logits.shape == (2, 2)
    assert output.embeddings.shape == (2, 8)
    rebuilt = model_registry.create("transformer_baseline", **model.model_config)
    assert rebuilt.model_config == model.model_config


def test_transformer_rejects_invalid_attention_dimensions():
    with pytest.raises(ValidationError, match="num_heads"):
        TransformerBaselineConfig(
            feature_dim=4, num_classes=2, d_model=10, num_heads=3
        )


def test_transformer_padding_does_not_change_valid_sample_embedding():
    torch.manual_seed(8)
    first = torch.randn(4, 3)
    model = TransformerBaseline(
        feature_dim=4, num_classes=2, d_model=8, num_heads=2,
        num_layers=1, feedforward_dim=16, dropout=0,
    ).eval()
    with torch.no_grad():
        alone = model(_samples(first)).embeddings[0]
        mixed = model(_samples(first, torch.randn(4, 8))).embeddings[0]
    assert torch.allclose(alone, mixed, atol=1e-6)


def test_transformer_rejects_non_prefix_mask():
    batch = _samples(torch.randn(4, 4), torch.randn(4, 2))
    batch.masks["features"][1] = torch.tensor([True, False, True, False])
    model = TransformerBaseline(
        feature_dim=4, num_classes=2, d_model=8, num_heads=2,
        num_layers=1, feedforward_dim=16,
    )
    with pytest.raises(ValueError, match="连续前缀"):
        model(batch)


def test_transformer_trains_one_step():
    model = TransformerBaseline(
        feature_dim=4, num_classes=2, d_model=8, num_heads=2,
        num_layers=1, feedforward_dim=16, dropout=0,
    )
    result = Trainer(model, TrainerConfig(epochs=1)).train_epoch([
        _samples(torch.randn(4, 4), torch.randn(4, 3))
    ], epoch=1)
    assert result.sample_count == 2
    assert result.loss > 0


def test_transformer_artifact_round_trip(tmp_path: Path):
    model = TransformerBaseline(
        feature_dim=16, num_classes=2, d_model=8, num_heads=2,
        num_layers=1, feedforward_dim=16, dropout=0,
    )
    directory = export_model_artifact(
        tmp_path / "transformer", model, model_name="transformer_baseline",
        data_config=_data_config(tmp_path), labels={0: "neutral", 1: "happy"},
    )
    loaded = load_model_artifact(directory)
    assert isinstance(loaded.model, TransformerBaseline)
    assert loaded.model.model_config == model.model_config
