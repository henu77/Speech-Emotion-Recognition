from pathlib import Path
import math
import struct
import wave

import torch
import pytest
from pydantic import ValidationError

from ser_lib.data.collate import SERCollator
from ser_lib.data.audio import AudioLoader, AudioLoaderConfig
from ser_lib.data.config import BatchingConfig
from ser_lib.data.pipeline import SamplePipeline
from ser_lib.data.representations.spectral import LogMelRepresentation
from ser_lib.data.types import SERSample, TensorSpec
from ser_lib.data.validation import validate_compatibility
from ser_lib.engine.checkpoint import load_checkpoint, save_checkpoint
from ser_lib.engine.evaluator import evaluate
from ser_lib.engine.trainer import Trainer, TrainerConfig
from ser_lib.models.base import ModelOutput
from ser_lib.models.cnn_models import CNNBaseline, CNNBaselineConfig
from ser_lib.models.registry import model_registry
from ser_lib.inference.offline import EmotionPredictor


def _batch():
    specs = {"features": TensorSpec(layout="FT", feature_dim=4)}
    samples = [
        SERSample("a", {"features": torch.randn(4, 5)}, {"features": 5}, 0, {}),
        SERSample("b", {"features": torch.randn(4, 3)}, {"features": 3}, 1, {}),
    ]
    return SERCollator(specs, BatchingConfig(type="dynamic"))(samples)


def test_cnn_baseline_forward_and_registry():
    model = model_registry.create("cnn_baseline", feature_dim=4, num_classes=3, hidden_dim=8)
    output = model(_batch())
    assert output.logits.shape == (2, 3)
    assert output.embeddings.shape == (2, 8)


def test_cnn_config_round_trip_and_parameter_count():
    model = CNNBaseline(feature_dim=4, num_classes=3, hidden_dim=8, dropout=0.1)
    rebuilt = model_registry.create("cnn_baseline", **model.model_config)
    assert rebuilt.model_config == model.model_config
    assert model.parameter_count() == sum(p.numel() for p in model.parameters())
    assert model.parameter_count(trainable_only=True) == model.parameter_count()
    with pytest.raises(ValidationError):
        CNNBaselineConfig(feature_dim=4, num_classes=3, unknown=True)


def test_model_registry_exposes_and_validates_cnn_contract():
    descriptor = model_registry.descriptor("cnn_baseline")
    assert descriptor["input_layouts"] == {"features": "FT"}
    assert descriptor["status"] == "stable"
    params = model_registry.validate_config(
        "cnn_baseline", {"feature_dim": 4, "num_classes": 2}
    )
    assert params == {"feature_dim": 4, "num_classes": 2, "hidden_dim": 128, "dropout": 0.2}


def test_model_output_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="logits"):
        ModelOutput(torch.randn(2))
    with pytest.raises(ValueError, match="embeddings"):
        ModelOutput(torch.randn(2, 3), embeddings=torch.randn(1, 4))
    with pytest.raises(ValueError, match="loss"):
        ModelOutput(torch.randn(2, 3), loss=torch.randn(2))


def test_cnn_rejects_all_padding_sample():
    batch = _batch()
    batch.masks["features"][0] = False
    model = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=8)
    with pytest.raises(ValueError, match="有效时间步"):
        model(batch)


def test_model_compatibility_accepts_log_mel_shape_contract():
    model = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=8)
    validate_compatibility(
        {"features": TensorSpec(layout="FT", feature_dim=4)},
        model.model_spec,
        BatchingConfig(type="dynamic"),
        num_classes=2,
    )


def test_trainer_and_evaluator_complete_one_batch():
    torch.manual_seed(7)
    model = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=8, dropout=0)
    result = Trainer(model, TrainerConfig(epochs=1)).train_epoch([_batch()], epoch=1)
    assert result.sample_count == 2
    assert result.loss > 0
    metrics = evaluate(model, [_batch()], num_classes=2)
    assert metrics.sample_count == 2
    assert metrics.confusion_matrix.shape == (2, 2)
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.macro_f1 <= 1
    assert 0 <= metrics.uar <= 1


def test_checkpoint_round_trip(tmp_path: Path):
    torch.manual_seed(3)
    model = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=8)
    optimizer = torch.optim.AdamW(model.parameters())
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    path = save_checkpoint(tmp_path / "checkpoint.pt", model, optimizer, epoch=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    payload = load_checkpoint(path, model, optimizer)
    assert payload["epoch"] == 2
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_log_mel_cnn_offline_prediction(tmp_path: Path):
    path = tmp_path / "predict.wav"
    sample_rate = 16000
    frames = bytearray()
    for index in range(1600):
        value = int(6000 * math.sin(2 * math.pi * 220 * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)

    representation = LogMelRepresentation(
        sample_rate=sample_rate, n_fft=256, win_length=256,
        hop_length=80, n_mels=16, f_max=8000,
    )
    pipeline = SamplePipeline(representation)
    collator = SERCollator(pipeline.output_specs, BatchingConfig(type="dynamic"))
    model = CNNBaseline(feature_dim=16, num_classes=2, hidden_dim=8, dropout=0)
    predictor = EmotionPredictor(
        model, AudioLoader(AudioLoaderConfig(target_sample_rate=sample_rate)),
        pipeline, collator, labels={0: "neutral", 1: "happy"},
    )
    result = predictor.predict_file(path)
    assert result.emotion in {"neutral", "happy"}
    assert len(result.probabilities) == 2
    assert abs(sum(result.probabilities) - 1.0) < 1e-6
