from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ser_lib.core import CancellationToken, OperationCancelled, ProgressEvent
from ser_lib.data import SERBatch, TensorSpec
from ser_lib.data.validation import ModelSpec
from ser_lib.engine import evaluate, write_evaluation_report
from ser_lib.models import ModelOutput, SERModel


class ScoreModel(SERModel):
    @property
    def model_spec(self):
        return ModelSpec(
            model_id="score_test",
            required_inputs={"scores": TensorSpec(layout="D", feature_dim=3)},
            supports_masks=False,
            supports_variable_length=False,
            num_classes=3,
        )

    @property
    def model_config(self):
        return {}

    def forward(self, batch):
        return ModelOutput(logits=batch.inputs["scores"])


def _batch(labels=(0, 1, 1, 2)):
    scores = torch.tensor([
        [4.0, 1.0, 0.0],
        [3.0, 2.0, 0.0],
        [0.0, 4.0, 1.0],
        [0.0, 1.0, 4.0],
    ])[:len(labels)]
    return SERBatch(
        inputs={"scores": scores}, lengths={}, masks={},
        labels=torch.tensor(labels, dtype=torch.long),
        uids=[f"sample-{index}" for index in range(len(labels))],
        metadata=[{} for _ in labels],
    )


def test_evaluate_returns_known_ser_metrics_and_predictions():
    events = []
    model = ScoreModel()
    model.train()
    result = evaluate(
        model, [_batch()], num_classes=3,
        labels={0: "neutral", 1: "happy", 2: "sad"},
        event_callback=events.append,
    )
    assert result.accuracy == pytest.approx(0.75)
    assert result.war == pytest.approx(0.75)
    assert result.uar == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert result.macro_f1 == pytest.approx((2 / 3 + 2 / 3 + 1.0) / 3)
    assert result.loss > 0
    assert result.confusion_matrix.tolist() == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert [item.support for item in result.per_class] == [1, 2, 1]
    assert [item.label_name for item in result.per_class] == ["neutral", "happy", "sad"]
    assert len(result.predictions) == 4
    assert result.predictions[1].predicted == 0
    assert sum(result.predictions[0].probabilities) == pytest.approx(1.0)
    assert len(events) == 1 and isinstance(events[0], ProgressEvent)
    assert model.training is True


def test_write_evaluation_report_is_json_safe(tmp_path: Path):
    result = evaluate(ScoreModel(), [_batch()], num_classes=3)
    directory = write_evaluation_report(tmp_path / "evaluation", result)
    summary = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["sample_count"] == 4
    assert summary["confusion_matrix"] == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert len(rows) == 4
    assert rows[0]["uid"] == "sample-0"


def test_evaluate_rejects_bad_label_mapping_and_out_of_range_target():
    with pytest.raises(ValueError, match="labels 必须覆盖"):
        evaluate(ScoreModel(), [_batch()], num_classes=3, labels={0: "zero"})
    with pytest.raises(ValueError, match="超出"):
        evaluate(ScoreModel(), [_batch((0, 3))], num_classes=3)


def test_evaluate_honors_cancellation_and_restores_mode():
    token = CancellationToken()
    token.cancel()
    model = ScoreModel()
    model.train()
    with pytest.raises(OperationCancelled):
        evaluate(model, [_batch()], num_classes=3, cancellation=token)
    assert model.training is True
