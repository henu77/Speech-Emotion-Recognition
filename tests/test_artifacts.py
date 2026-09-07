from pathlib import Path

import pytest
import torch

from ser_lib.artifacts import export_model_artifact, load_model_artifact
from ser_lib.data.config import AudioSettings, BatchingConfig, ComponentConfig, DataConfig
from ser_lib.models import CNNBaseline


def _config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        manifest=tmp_path / "unused-dataset.yaml",
        audio=AudioSettings(target_sample_rate=16000),
        representation=ComponentConfig(
            type="log_mel",
            params={"sample_rate": 16000, "n_mels": 16, "n_fft": 128,
                    "win_length": 128, "hop_length": 64, "f_max": 8000},
        ),
        batching=BatchingConfig(type="dynamic"),
        labels={0: {"en": "neutral"}, 1: {"en": "happy"}},
    )


def test_model_artifact_round_trip(tmp_path: Path):
    torch.manual_seed(13)
    model = CNNBaseline(feature_dim=16, num_classes=2, hidden_dim=4, dropout=0)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    directory = export_model_artifact(
        tmp_path / "artifact", model,
        model_name="cnn_baseline",
        model_params={"feature_dim": 16, "num_classes": 2, "hidden_dim": 4, "dropout": 0},
        data_config=_config(tmp_path), labels={0: "neutral", 1: "happy"},
    )
    loaded = load_model_artifact(directory)
    assert loaded.manifest.labels == {0: "neutral", 1: "happy"}
    for key, value in loaded.model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_model_artifact_detects_modified_weights(tmp_path: Path):
    model = CNNBaseline(feature_dim=16, num_classes=2, hidden_dim=4)
    directory = export_model_artifact(
        tmp_path / "artifact", model, model_name="cnn_baseline",
        model_params={"feature_dim": 16, "num_classes": 2, "hidden_dim": 4},
        data_config=_config(tmp_path), labels={0: "neutral", 1: "happy"},
    )
    with (directory / "model_state.pt").open("ab") as output:
        output.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_model_artifact(directory)
