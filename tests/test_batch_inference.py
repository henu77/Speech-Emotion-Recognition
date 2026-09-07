from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from ser_lib.core import CancellationToken, OperationCancelled, ProgressEvent
from ser_lib.data import (
    AudioRecord,
    BatchingConfig,
    DatasetManifest,
    ManifestMeta,
    SERCollator,
    SERSample,
    TensorSpec,
)
from ser_lib.data.validation import ModelSpec
from ser_lib.inference import (
    BatchEmotionPredictor,
    EmotionPredictor,
    PredictionResult,
    write_batch_predictions,
)
from ser_lib.models import ModelOutput, SERModel


class FakePredictor:
    def __init__(self) -> None:
        self.records = []

    def predict_record(self, record):
        self.records.append(record)
        if not record.audio_path.is_file():
            raise FileNotFoundError(f"missing: {record.audio_path}")
        return PredictionResult(record.uid, 1, "happy", 0.75, [0.25, 0.75])


class PassThroughLoader:
    def load(self, record):
        return record


class UIDPipeline:
    output_specs = {"features": TensorSpec(layout="FT", feature_dim=1)}

    def __call__(self, _audio, record):
        value = 1.0 if record.uid.startswith("positive") else 0.0
        tensor = torch.full((1, 5), value)
        return SERSample(record.uid, {"features": tensor}, {"features": 5}, None, {})


class CountingModel(SERModel):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    @property
    def model_spec(self):
        return ModelSpec(
            "counting", {"features": TensorSpec(layout="FT", feature_dim=1)},
            supports_masks=True, supports_variable_length=True, num_classes=2,
        )

    @property
    def model_config(self):
        return {}

    def forward(self, batch):
        self.forward_calls += 1
        mean = batch.inputs["features"].mean(dim=(1, 2))
        return ModelOutput(torch.stack((-mean, mean), dim=-1))


def test_batch_predict_records_collects_failures_and_progress(tmp_path: Path):
    valid = tmp_path / "valid.wav"
    valid.write_bytes(b"not decoded by fake")
    records = [
        AudioRecord("valid", valid),
        AudioRecord("missing", tmp_path / "missing.wav"),
    ]
    events = []
    result = BatchEmotionPredictor(FakePredictor()).predict_records(
        records, fail_fast=False, event_callback=events.append
    )
    assert result.total == 2
    assert result.succeeded == 1
    assert result.failed == 1
    assert result.failures[0].error_type == "FileNotFoundError"
    assert [event.completed for event in events if isinstance(event, ProgressEvent)] == [1, 2]
    assert events[-1].fraction == 1.0


def test_batch_predict_fail_fast_and_cancellation(tmp_path: Path):
    missing = AudioRecord("missing", tmp_path / "missing.wav")
    with pytest.raises(FileNotFoundError):
        BatchEmotionPredictor(FakePredictor()).predict_records([missing])
    token = CancellationToken()
    token.cancel()
    with pytest.raises(OperationCancelled):
        BatchEmotionPredictor(FakePredictor()).predict_records(
            [missing], fail_fast=False, cancellation=token
        )


def test_directory_prediction_filters_extensions_and_is_deterministic(tmp_path: Path):
    (tmp_path / "b.WAV").write_bytes(b"b")
    (tmp_path / "a.wav").write_bytes(b"a")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.flac").write_bytes(b"c")
    fake = FakePredictor()
    result = BatchEmotionPredictor(fake).predict_directory(tmp_path, recursive=True)
    assert result.succeeded == 3
    assert [record.audio_path.name for record in fake.records] == ["a.wav", "b.WAV", "c.flac"]
    assert len({prediction.uid for prediction in result.predictions}) == 3


def test_manifest_prediction_resolves_paths_and_preserves_segments(tmp_path: Path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    meta = ManifestMeta("demo", tmp_path, tmp_path / "dataset.yaml")
    record = AudioRecord("segment", Path("audio.wav"), start_ms=10, end_ms=20)
    manifest = DatasetManifest(meta, [record])
    fake = FakePredictor()
    result = BatchEmotionPredictor(fake).predict_manifest(manifest)
    assert result.succeeded == 1
    assert fake.records[0].audio_path == audio.resolve()
    assert fake.records[0].start_ms == 10
    assert fake.records[0].end_ms == 20


def test_batch_prediction_jsonl_and_csv_exports(tmp_path: Path):
    valid = tmp_path / "valid.wav"
    valid.write_bytes(b"valid")
    result = BatchEmotionPredictor(FakePredictor()).predict_records([
        AudioRecord("ok", valid),
        AudioRecord("bad", tmp_path / "missing.wav"),
    ], fail_fast=False)
    jsonl_path = write_batch_predictions(tmp_path / "predictions.jsonl", result)
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows] == ["succeeded", "failed"]
    assert rows[0]["probabilities"] == [0.25, 0.75]

    csv_path = write_batch_predictions(tmp_path / "predictions.csv", result)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert [row["status"] for row in csv_rows] == ["succeeded", "failed"]
    assert json.loads(csv_rows[0]["probabilities"]) == [0.25, 0.75]

    with pytest.raises(ValueError, match="jsonl 或 csv"):
        write_batch_predictions(tmp_path / "predictions.txt", result)


def test_real_batching_uses_one_forward_and_groups_sliding_windows():
    model = CountingModel()
    collator = SERCollator(
        UIDPipeline.output_specs,
        BatchingConfig(
            type="sliding", primary_key="features",
            sliding={"window_size": 2, "stride": 2},
        ),
    )
    predictor = EmotionPredictor(
        model, PassThroughLoader(), UIDPipeline(), collator,
        labels={0: "neutral", 1: "happy"}, window_aggregation="mean_logits",
    )
    records = [
        AudioRecord("neutral-1", Path("unused-a.wav")),
        AudioRecord("positive-1", Path("unused-b.wav")),
    ]
    results = predictor.predict_records(records)
    assert model.forward_calls == 1
    assert [result.uid for result in results] == ["neutral-1", "positive-1"]
    assert [result.label_id for result in results] == [0, 1]


def test_batch_wrapper_respects_batch_size_for_real_predictor():
    model = CountingModel()
    predictor = EmotionPredictor(
        model, PassThroughLoader(), UIDPipeline(),
        SERCollator(UIDPipeline.output_specs, BatchingConfig(type="dynamic")),
        labels={0: "neutral", 1: "happy"},
    )
    records = [AudioRecord(f"positive-{index}", Path("unused.wav")) for index in range(5)]
    result = BatchEmotionPredictor(predictor).predict_records(records, batch_size=2)
    assert result.succeeded == 5
    assert model.forward_calls == 3
