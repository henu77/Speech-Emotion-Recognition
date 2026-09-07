from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from ser_lib.artifacts import export_model_artifact, load_model_artifact
from ser_lib.data import (
    BatchingConfig,
    CompatibilityError,
    SERCollator,
    SERSample,
    TensorSpec,
    validate_compatibility,
)
from ser_lib.data.config import AudioSettings, ComponentConfig, DataConfig
from ser_lib.engine import Trainer, TrainerConfig
from ser_lib.models import HFAudioClassifier, HFAudioClassifierConfig, model_registry


class FakeConfig:
    model_type = "fake_audio"
    hidden_size = 4

    def __init__(self, **values):
        self.hidden_size = values.get("hidden_size", 4)

    def to_dict(self):
        return {"model_type": self.model_type, "hidden_size": self.hidden_size}


class FakeEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or FakeConfig()
        self.projection = nn.Linear(1, self.config.hidden_size, bias=False)

    def forward(self, *, input_values, attention_mask, return_dict):
        assert return_dict is True
        assert attention_mask.shape == input_values.shape
        return SimpleNamespace(last_hidden_state=self.projection(input_values.unsqueeze(-1)))


@pytest.fixture
def fake_transformers(monkeypatch):
    calls = {}

    class AutoConfig:
        @staticmethod
        def for_model(model_type, **values):
            calls["for_model"] = (model_type, values)
            return FakeConfig(**values)

    class AutoModel:
        @staticmethod
        def from_config(config):
            calls["from_config"] = True
            return FakeEncoder(config)

        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["from_pretrained"] = (name, kwargs)
            return FakeEncoder()

    module = SimpleNamespace(AutoConfig=AutoConfig, AutoModel=AutoModel)
    monkeypatch.setattr("ser_lib.models.pretrained._transformers", lambda: module)
    return calls


def _batch(lengths=(6, 4)):
    samples = [
        SERSample(
            f"sample-{index}", {"waveform": torch.randn(length)},
            {"waveform": length}, index % 2, {},
        )
        for index, length in enumerate(lengths)
    ]
    return SERCollator(
        {"waveform": TensorSpec(layout="T")}, BatchingConfig(type="dynamic")
    )(samples)


def _data_config(tmp_path: Path):
    return DataConfig(
        manifest=tmp_path / "unused.yaml",
        audio=AudioSettings(target_sample_rate=16000),
        representation=ComponentConfig(type="waveform"),
        batching=BatchingConfig(type="dynamic"),
        labels={0: {"en": "neutral"}, 1: {"en": "happy"}},
    )


def test_pretrained_config_requires_exactly_one_source():
    with pytest.raises(ValidationError, match="只能提供一个"):
        HFAudioClassifierConfig(num_classes=2)
    with pytest.raises(ValidationError, match="只能提供一个"):
        HFAudioClassifierConfig(
            num_classes=2, pretrained_model_name_or_path="local",
            encoder_config={"model_type": "fake_audio"},
        )


def test_pretrained_load_is_local_and_disables_remote_code(fake_transformers):
    model = HFAudioClassifier(
        num_classes=2, pretrained_model_name_or_path="local/model", dropout=0
    )
    name, options = fake_transformers["from_pretrained"]
    assert name == "local/model"
    assert options["local_files_only"] is True
    assert options["trust_remote_code"] is False
    assert model.model_config["pretrained_model_name_or_path"] is None
    assert model.model_config["encoder_config"]["model_type"] == "fake_audio"


def test_pretrained_forward_train_and_padding_contract(fake_transformers):
    torch.manual_seed(5)
    model = HFAudioClassifier(
        num_classes=2, encoder_config={"model_type": "fake_audio", "hidden_size": 4},
        dropout=0,
    )
    output = model(_batch())
    assert output.logits.shape == (2, 2)
    assert output.embeddings.shape == (2, 4)
    result = Trainer(model, TrainerConfig(epochs=1)).train_epoch([_batch()], epoch=1)
    assert result.sample_count == 2


def test_pretrained_padding_does_not_change_valid_embedding(fake_transformers):
    model = HFAudioClassifier(
        num_classes=2, encoder_config={"model_type": "fake_audio", "hidden_size": 4},
        dropout=0,
    ).eval()
    first = torch.randn(5)
    collator = SERCollator(
        {"waveform": TensorSpec(layout="T")}, BatchingConfig(type="dynamic")
    )
    single = collator([SERSample("first", {"waveform": first}, {"waveform": 5}, 0, {})])
    mixed = collator([
        SERSample("first", {"waveform": first}, {"waveform": 5}, 0, {}),
        SERSample("long", {"waveform": torch.randn(9)}, {"waveform": 9}, 1, {}),
    ])
    with torch.no_grad():
        assert torch.allclose(
            model(single).embeddings[0], model(mixed).embeddings[0], atol=1e-6
        )


def test_freeze_encoder_keeps_it_in_eval_mode(fake_transformers):
    model = HFAudioClassifier(
        num_classes=2, encoder_config={"model_type": "fake_audio", "hidden_size": 4},
        freeze_encoder=True,
    )
    model.train()
    assert model.encoder.training is False
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.classifier.parameters())


def test_optional_dependency_error_is_actionable(monkeypatch):
    import importlib

    original = importlib.import_module

    def missing(name, package=None):
        if name == "transformers":
            raise ImportError(name)
        return original(name, package)

    monkeypatch.setattr("ser_lib.models.pretrained.importlib.import_module", missing)
    with pytest.raises(ImportError, match=r"ser-lib\[pretrained\]"):
        HFAudioClassifier(
            num_classes=2, encoder_config={"model_type": "fake_audio"}
        )


def test_pretrained_registry_and_artifact_round_trip(fake_transformers, tmp_path: Path):
    model = model_registry.create(
        "hf_audio_classifier", num_classes=2,
        encoder_config={"model_type": "fake_audio", "hidden_size": 4}, dropout=0,
    )
    directory = export_model_artifact(
        tmp_path / "hf", model, model_name="hf_audio_classifier",
        data_config=_data_config(tmp_path), labels={0: "neutral", 1: "happy"},
    )
    loaded = load_model_artifact(directory)
    assert isinstance(loaded.model, HFAudioClassifier)
    assert loaded.model.model_config == model.model_config
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[key], value)


def test_pretrained_declares_required_sample_rate(fake_transformers):
    model = HFAudioClassifier(
        num_classes=2, encoder_config={"model_type": "fake_audio", "hidden_size": 4},
        expected_sample_rate=8000,
    )
    assert model.model_spec.expected_sample_rate == 8000
    assert model.model_config["expected_sample_rate"] == 8000
    with pytest.raises(CompatibilityError, match="采样率不一致"):
        validate_compatibility(
            {"waveform": TensorSpec(layout="T")}, model.model_spec,
            BatchingConfig(type="dynamic"), num_classes=2, sample_rate=16000,
        )
