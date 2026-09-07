from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from ser_lib.artifacts import export_model_artifact, load_model_artifact
from ser_lib.data import BatchingConfig, SERCollator, SERSample, TensorSpec
from ser_lib.data.config import AudioSettings, ComponentConfig, DataConfig
from ser_lib.engine import Trainer, TrainerConfig
from ser_lib.models import GRUBaseline, GRUBaselineConfig, model_registry


SPECS = {"features": TensorSpec(layout="FT", feature_dim=4)}


def _batch(lengths: tuple[int, ...] = (5, 3)):
    torch.manual_seed(22)
    samples = [
        SERSample(
            f"sample-{index}", {"features": torch.randn(4, length)},
            {"features": length}, index % 2, {},
        )
        for index, length in enumerate(lengths)
    ]
    return SERCollator(SPECS, BatchingConfig(type="dynamic"))(samples)


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


def test_gru_registry_config_and_forward_contract():
    model = model_registry.create(
        "gru_baseline", feature_dim=4, num_classes=2, hidden_dim=6,
        num_layers=2, bidirectional=True, dropout=0.1,
    )
    output = model(_batch())
    assert output.logits.shape == (2, 2)
    assert output.embeddings is not None
    assert output.embeddings.shape == (2, 12)
    assert model_registry.create("gru_baseline", **model.model_config).model_config == model.model_config


def test_gru_rejects_silent_single_layer_dropout():
    with pytest.raises(ValidationError, match="num_layers=1"):
        GRUBaselineConfig(feature_dim=4, num_classes=2, num_layers=1, dropout=0.1)


def test_gru_ignores_dynamic_padding():
    torch.manual_seed(4)
    model = GRUBaseline(feature_dim=4, num_classes=2, hidden_dim=5, bidirectional=True)
    model.eval()
    single = _batch((3,))
    mixed = _batch((3, 8))
    with torch.no_grad():
        single_embedding = model(single).embeddings
        mixed_embedding = model(mixed).embeddings
    assert single_embedding is not None and mixed_embedding is not None
    assert torch.allclose(single_embedding[0], mixed_embedding[0], atol=1e-6)


def test_gru_rejects_mask_length_disagreement():
    batch = _batch()
    batch.masks["features"][0, 0] = False
    model = GRUBaseline(feature_dim=4, num_classes=2, hidden_dim=5)
    with pytest.raises(ValueError, match="lengths"):
        model(batch)


def test_gru_trains_one_step():
    model = GRUBaseline(feature_dim=4, num_classes=2, hidden_dim=5)
    result = Trainer(model, TrainerConfig(epochs=1)).train_epoch([_batch()], epoch=1)
    assert result.sample_count == 2
    assert result.loss > 0


def test_gru_artifact_round_trip(tmp_path: Path):
    model = GRUBaseline(feature_dim=16, num_classes=2, hidden_dim=5)
    directory = export_model_artifact(
        tmp_path / "gru", model, model_name="gru_baseline",
        data_config=_data_config(tmp_path), labels={0: "neutral", 1: "happy"},
    )
    loaded = load_model_artifact(directory)
    assert isinstance(loaded.model, GRUBaseline)
    assert loaded.model.model_config == model.model_config
