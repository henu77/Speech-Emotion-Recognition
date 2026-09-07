import hashlib
import json
from pathlib import Path

import pytest
import torch

from ser_lib import __version__
from ser_lib.artifacts import (
    ModelCard,
    export_model_artifact,
    load_model_artifact,
    verify_model_artifact,
)
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
    assert loaded.manifest.schema_version == 2
    assert loaded.manifest.weights_format == "safetensors"
    assert (directory / "weights.safetensors").is_file()
    assert (directory / "README.md").is_file()
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
    with (directory / "weights.safetensors").open("ab") as output:
        output.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_model_artifact(directory)


def test_model_artifact_can_derive_model_config(tmp_path: Path):
    model = CNNBaseline(feature_dim=16, num_classes=2, hidden_dim=5, dropout=0.1)
    directory = export_model_artifact(
        tmp_path / "artifact", model, model_name="cnn_baseline",
        data_config=_config(tmp_path), labels={0: "neutral", 1: "happy"},
    )
    loaded = load_model_artifact(directory)
    assert loaded.manifest.model_params == model.model_config


def test_model_artifact_rejects_mismatched_declared_config(tmp_path: Path):
    model = CNNBaseline(feature_dim=16, num_classes=2, hidden_dim=5)
    with pytest.raises(ValueError, match="实际配置"):
        export_model_artifact(
            tmp_path / "artifact", model, model_name="cnn_baseline",
            model_params={"feature_dim": 16, "num_classes": 2, "hidden_dim": 6},
            data_config=_config(tmp_path), labels={0: "neutral", 1: "happy"},
        )


def test_artifact_verifies_all_files_and_model_card(tmp_path: Path):
    model = CNNBaseline(feature_dim=16, num_classes=2, hidden_dim=5)
    directory = export_model_artifact(
        tmp_path / "artifact", model, model_name="cnn_baseline",
        data_config=_config(tmp_path), labels={0: "neutral", 1: "happy"},
        model_card=ModelCard(
            description="A test model", dataset="synthetic", language=["zh"],
            license="MIT", limitations=["test only"],
        ),
    )
    manifest = verify_model_artifact(directory)
    assert set(manifest.files_sha256) == {
        "weights.safetensors", "data_config.json", "model_config.json",
        "labels.json", "metrics.json", "README.md",
    }
    assert "A test model" in (directory / "README.md").read_text(encoding="utf-8")
    (directory / "metrics.json").write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="metrics.json"):
        verify_model_artifact(directory)


def test_artifact_refuses_to_overwrite_existing_directory(tmp_path: Path):
    target = tmp_path / "artifact"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        export_model_artifact(
            target, CNNBaseline(feature_dim=16, num_classes=2),
            model_name="cnn_baseline", data_config=_config(tmp_path),
            labels={0: "neutral", 1: "happy"},
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_artifact_manifest_rejects_weight_path_traversal(tmp_path: Path):
    model = CNNBaseline(feature_dim=16, num_classes=2)
    directory = export_model_artifact(
        tmp_path / "artifact", model, model_name="cnn_baseline",
        data_config=_config(tmp_path), labels={0: "neutral", 1: "happy"},
    )
    manifest_path = directory / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["weights_file"] = "../outside.safetensors"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        verify_model_artifact(directory)


def test_legacy_pytorch_artifact_requires_explicit_trust(tmp_path: Path):
    model = CNNBaseline(feature_dim=16, num_classes=2, hidden_dim=5, dropout=0)
    directory = tmp_path / "legacy"
    directory.mkdir()
    weights = directory / "model_state.pt"
    torch.save(model.state_dict(), weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "library_version": __version__,
        "model_name": "cnn_baseline",
        "model_params": model.model_config,
        "weights_file": "model_state.pt",
        "weights_sha256": digest,
        "preprocessing": _config(tmp_path).model_dump(mode="json"),
        "labels": {"0": "neutral", "1": "happy"},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="allow_legacy_pickle"):
        load_model_artifact(directory)
    loaded = load_model_artifact(directory, allow_legacy_pickle=True)
    assert loaded.manifest.schema_version == 1
