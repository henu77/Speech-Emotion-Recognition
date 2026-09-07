from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from ser_lib.core import CancellationToken, MetricEvent, OperationCancelled, ProgressEvent
from ser_lib.data import BatchingConfig, SERCollator, SERSample, TensorSpec
from ser_lib.data.config import AudioSettings, ComponentConfig, DataConfig
from ser_lib.data.errors import CompatibilityError
from ser_lib.engine import (
    ExperimentConfig,
    ModelConfig,
    Trainer,
    TrainerConfig,
    build_experiment_components,
    load_experiment_config,
    parse_optimizer_config,
    parse_scheduler_config,
)
from ser_lib.models import CNNBaseline


def _data_config(manifest: Path) -> DataConfig:
    return DataConfig(
        manifest=manifest,
        audio=AudioSettings(target_sample_rate=16000),
        representation=ComponentConfig(
            type="log_mel", params={"sample_rate": 16000, "n_mels": 16}
        ),
        batching=BatchingConfig(type="dynamic"),
        labels={0: {"en": "neutral"}, 1: {"en": "happy"}},
    )


def _experiment(tmp_path: Path, **trainer_updates) -> ExperimentConfig:
    params = {"feature_dim": 4, "num_classes": 2, "hidden_dim": 6, "dropout": 0}
    return ExperimentConfig(
        data=_data_config(tmp_path / "dataset.yaml"),
        model=ModelConfig(type="cnn_baseline", params=params),
        trainer=TrainerConfig(epochs=1, **trainer_updates),
        optimizer={"type": "sgd", "params": {"learning_rate": 0.1}},
        scheduler={"type": "step", "params": {"step_size": 1, "gamma": 0.5}},
        output_dir=tmp_path / "run",
    )


def _batch():
    samples = [
        SERSample("a", {"features": torch.randn(4, 5)}, {"features": 5}, 0, {}),
        SERSample("b", {"features": torch.randn(4, 3)}, {"features": 3}, 1, {}),
    ]
    return SERCollator(
        {"features": TensorSpec(layout="FT", feature_dim=4)},
        BatchingConfig(type="dynamic"),
    )(samples)


def test_experiment_config_rejects_unknown_optimizer_and_scheduler():
    with pytest.raises(ValidationError, match="optimizer"):
        ExperimentConfig(
            data=_data_config(Path("dataset.yaml")),
            model=ModelConfig(type="cnn_baseline"),
            optimizer={"type": "unknown", "params": {}},
        )
    with pytest.raises(ValueError, match="未知 scheduler"):
        parse_scheduler_config({"type": "unknown", "params": {}})
    with pytest.raises(ValueError, match="未知字段"):
        parse_optimizer_config({"type": "adamw", "params": {}, "typo": True})


def test_load_experiment_config_resolves_paths_from_config_file(tmp_path: Path):
    path = tmp_path / "configs" / "experiment.yaml"
    path.parent.mkdir()
    path.write_text(
        """schema_version: 1
data:
  schema_version: 1
  manifest: ../dataset.yaml
  representation:
    type: log_mel
    params: {sample_rate: 16000, n_mels: 16}
model:
  type: cnn_baseline
  params: {feature_dim: 16, num_classes: 2}
trainer:
  checkpoint_dir: checkpoints
output_dir: run
""",
        encoding="utf-8",
    )
    config = load_experiment_config(path)
    assert config.data.manifest == (path.parent / "../dataset.yaml").resolve()
    assert config.output_dir == (path.parent / "run").resolve()
    assert config.trainer.checkpoint_dir == (path.parent / "checkpoints").resolve()


def test_trainer_accumulates_gradients_emits_events_and_steps_scheduler(tmp_path: Path):
    events = []
    model = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=6, dropout=0)
    experiment = _experiment(tmp_path, gradient_accumulation_steps=2)
    trainer = Trainer.from_experiment(model, experiment, event_callback=events.append)
    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    history = trainer.fit([_batch(), _batch(), _batch()])
    assert history[0].optimizer_steps == 2
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr * 0.5)
    assert sum(isinstance(item, ProgressEvent) for item in events) == 3
    assert sum(isinstance(item, MetricEvent) for item in events) == 2


def test_trainer_cancellation_happens_before_mutating_model(tmp_path: Path):
    token = CancellationToken()
    token.cancel()
    model = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=6, dropout=0)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    trainer = Trainer.from_experiment(model, _experiment(tmp_path), cancellation=token)
    with pytest.raises(OperationCancelled):
        trainer.fit([_batch()])
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())


def test_trainer_rejects_amp_on_cpu_and_model_config_mismatch(tmp_path: Path):
    model = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=6, dropout=0)
    with pytest.raises(ValueError, match="AMP"):
        Trainer(model, TrainerConfig(amp=True, device="cpu"))
    mismatched = _experiment(tmp_path).model_copy(update={
        "model": ModelConfig(
            type="cnn_baseline",
            params={"feature_dim": 4, "num_classes": 2, "hidden_dim": 7, "dropout": 0},
        )
    })
    with pytest.raises(ValueError, match="实际配置"):
        Trainer.from_experiment(model, mismatched)


def test_experiment_preflight_builds_compatible_runtime(tmp_path: Path):
    config = _experiment(tmp_path).model_copy(update={
        "data": _data_config(tmp_path / "dataset.yaml"),
        "model": ModelConfig(
            type="cnn_baseline",
            params={"feature_dim": 16, "num_classes": 2, "hidden_dim": 6, "dropout": 0},
        ),
    })
    runtime = build_experiment_components(config)
    assert runtime.model.model_spec.model_id == "cnn_baseline"
    assert runtime.pipeline.output_specs["features"].feature_dim == 16


def test_experiment_preflight_rejects_incompatible_feature_dimension(tmp_path: Path):
    config = _experiment(tmp_path).model_copy(update={
        "data": _data_config(tmp_path / "dataset.yaml"),
        "model": ModelConfig(
            type="cnn_baseline",
            params={"feature_dim": 32, "num_classes": 2, "hidden_dim": 6, "dropout": 0},
        ),
    })
    with pytest.raises(CompatibilityError, match="feature_dim"):
        build_experiment_components(config)
