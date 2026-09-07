from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from ser_lib.core import CancellationToken, OperationCancelled
from ser_lib.data import BatchingConfig, SERCollator, SERSample, TensorSpec
from ser_lib.data.config import AudioSettings, ComponentConfig, DataConfig
from ser_lib.engine import (
    ExperimentConfig,
    ModelConfig,
    Trainer,
    TrainerConfig,
    load_checkpoint,
    save_checkpoint,
)
from ser_lib.models import CNNBaseline


def _batch():
    generator = torch.Generator().manual_seed(99)
    samples = [
        SERSample(
            "a", {"features": torch.randn(4, 5, generator=generator)},
            {"features": 5}, 0, {},
        ),
        SERSample(
            "b", {"features": torch.randn(4, 3, generator=generator)},
            {"features": 3}, 1, {},
        ),
    ]
    return SERCollator(
        {"features": TensorSpec(layout="FT", feature_dim=4)},
        BatchingConfig(type="dynamic"),
    )(samples)


def _experiment(tmp_path: Path) -> ExperimentConfig:
    trainer = TrainerConfig(
        epochs=2, seed=7, checkpoint_dir=tmp_path / "checkpoints"
    )
    return ExperimentConfig(
        data=DataConfig(
            manifest=tmp_path / "unused.yaml",
            audio=AudioSettings(),
            representation=ComponentConfig(
                type="log_mel", params={"sample_rate": 16000, "n_mels": 16}
            ),
            batching=BatchingConfig(type="dynamic"),
            labels={0: {"en": "neutral"}, 1: {"en": "happy"}},
        ),
        model=ModelConfig(
            type="cnn_baseline",
            params={"feature_dim": 4, "num_classes": 2, "hidden_dim": 6, "dropout": 0},
        ),
        trainer=trainer,
        optimizer={"type": "sgd", "params": {"learning_rate": 0.05}},
        scheduler={"type": "step", "params": {"step_size": 1, "gamma": 0.5}},
        output_dir=tmp_path,
    )


def test_checkpoint_restores_rng_scheduler_and_scalars(tmp_path: Path):
    model = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=6, dropout=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()
    scheduler.step()
    random.seed(123)
    torch.manual_seed(123)
    path = save_checkpoint(
        tmp_path / "state.pt", model, optimizer, scheduler=scheduler, epoch=3,
        trainer_config={"epochs": 4},
    )
    expected_python = random.random()
    expected_torch = torch.rand(3)
    random.seed(999)
    torch.manual_seed(999)
    optimizer.param_groups[0]["lr"] = 9.0
    payload = load_checkpoint(
        path, model, optimizer, scheduler=scheduler,
        expected_trainer_config={"epochs": 4},
    )
    assert payload["format_version"] == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert random.random() == pytest.approx(expected_python)
    assert torch.equal(torch.rand(3), expected_torch)


def test_cancel_after_epoch_then_resume_matches_continuous_training(tmp_path: Path):
    torch.manual_seed(5)
    initial = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=6, dropout=0)
    initial_state = {key: value.detach().clone() for key, value in initial.state_dict().items()}
    batches = [_batch()]

    continuous_model = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=6, dropout=0)
    continuous_model.load_state_dict(initial_state)
    continuous = Trainer.from_experiment(continuous_model, _experiment(tmp_path / "continuous"))
    continuous.fit(batches)

    token = CancellationToken()
    interrupted_model = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=6, dropout=0)
    interrupted_model.load_state_dict(initial_state)
    interrupted_config = _experiment(tmp_path / "resumed")
    interrupted = Trainer.from_experiment(
        interrupted_model, interrupted_config, cancellation=token
    )
    with pytest.raises(OperationCancelled):
        interrupted.fit(batches, on_epoch_end=lambda _: token.cancel())
    checkpoint = interrupted_config.trainer.checkpoint_dir / "epoch-0001.pt"
    assert checkpoint.is_file()

    resumed_model = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=6, dropout=0)
    resumed = Trainer.from_experiment(resumed_model, interrupted_config)
    resumed.resume_from(checkpoint)
    history = resumed.fit(batches)
    assert [item.epoch for item in history] == [2]
    assert resumed.last_completed_epoch == 2
    for key, value in continuous_model.state_dict().items():
        assert torch.allclose(value, resumed_model.state_dict()[key], atol=1e-7)
    assert continuous.optimizer.param_groups[0]["lr"] == pytest.approx(
        resumed.optimizer.param_groups[0]["lr"]
    )


def test_checkpoint_rejects_model_or_trainer_config_before_loading(tmp_path: Path):
    source = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=6, dropout=0)
    path = save_checkpoint(tmp_path / "state.pt", source, None, epoch=1,
                           trainer_config={"epochs": 2})
    target = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=7, dropout=0)
    before = {key: value.detach().clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="模型配置"):
        load_checkpoint(path, target)
    assert all(torch.equal(value, before[key]) for key, value in target.state_dict().items())

    matching = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=6, dropout=0)
    before_matching = {
        key: value.detach().clone() for key, value in matching.state_dict().items()
    }
    with pytest.raises(ValueError, match="trainer_config"):
        load_checkpoint(path, matching, expected_trainer_config={"epochs": 3})
    assert all(
        torch.equal(value, before_matching[key])
        for key, value in matching.state_dict().items()
    )


def test_format_v1_checkpoint_remains_loadable(tmp_path: Path):
    model = CNNBaseline(feature_dim=4, num_classes=2, hidden_dim=6, dropout=0)
    path = tmp_path / "v1.pt"
    torch.save({
        "format_version": 1,
        "model_id": "cnn_baseline",
        "model_state": model.state_dict(),
        "optimizer_state": None,
        "epoch": 1,
        "metrics": {},
        "metadata": {},
    }, path)
    assert load_checkpoint(path, model)["epoch"] == 1
