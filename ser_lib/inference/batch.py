"""批量离线推理与 JSONL/CSV 结果导出。"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ser_lib.core.events import CancellationCheck, EventCallback, ProgressEvent
from ser_lib.data.manifest import DatasetManifest
from ser_lib.data.types import AudioRecord
from ser_lib.inference.offline import EmotionPredictor, PredictionResult


DEFAULT_AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a"})


@dataclass(frozen=True, slots=True)
class PredictionFailure:
    uid: str
    audio_path: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class BatchPredictionResult:
    predictions: tuple[PredictionResult, ...]
    failures: tuple[PredictionFailure, ...]
    total: int

    def __post_init__(self) -> None:
        if self.total != len(self.predictions) + len(self.failures):
            raise ValueError("BatchPredictionResult.total 与成功/失败数量不一致")

    @property
    def succeeded(self) -> int:
        return len(self.predictions)

    @property
    def failed(self) -> int:
        return len(self.failures)


class BatchEmotionPredictor:
    """在单文件预测器之上提供来源枚举和逐条失败策略。"""

    def __init__(self, predictor: EmotionPredictor) -> None:
        self.predictor = predictor

    def predict_records(
        self,
        records: Iterable[AudioRecord],
        *,
        fail_fast: bool = True,
        batch_size: int = 16,
        event_callback: EventCallback | None = None,
        cancellation: CancellationCheck | None = None,
    ) -> BatchPredictionResult:
        if batch_size < 1:
            raise ValueError("batch_size 必须 >= 1")
        materialized = list(records)
        predictions: list[PredictionResult] = []
        failures: list[PredictionFailure] = []
        completed = 0

        def report() -> None:
            if event_callback is not None:
                event_callback(ProgressEvent(
                    stage="batch_predict",
                    completed=completed,
                    total=len(materialized),
                    message=f"succeeded={len(predictions)}, failed={len(failures)}",
                ))

        def predict_one(record: AudioRecord) -> None:
            nonlocal completed
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            try:
                predictions.append(self.predictor.predict_record(record))
            except Exception as exc:
                if fail_fast:
                    raise
                failures.append(PredictionFailure(
                    uid=record.uid,
                    audio_path=str(record.audio_path),
                    error_type=type(exc).__name__,
                    message=str(exc),
                ))
            completed += 1
            report()

        batch_method = getattr(self.predictor, "predict_records", None)
        for start in range(0, len(materialized), batch_size):
            chunk = materialized[start:start + batch_size]
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            if batch_method is None or len(chunk) == 1:
                for record in chunk:
                    predict_one(record)
                continue
            try:
                chunk_results = batch_method(chunk)
                if len(chunk_results) != len(chunk):
                    raise ValueError("批量预测返回数量与输入记录数不一致")
            except Exception:
                if fail_fast:
                    raise
                # 逐条重试以隔离坏文件；有效项可能被重新预处理，但结果不会重复写入。
                for record in chunk:
                    predict_one(record)
            else:
                predictions.extend(chunk_results)
                for _ in chunk:
                    completed += 1
                    report()
        return BatchPredictionResult(tuple(predictions), tuple(failures), len(materialized))

    def predict_files(
        self,
        paths: Iterable[Path | str],
        **kwargs,
    ) -> BatchPredictionResult:
        records = [
            AudioRecord(uid=f"{Path(path).stem or 'audio'}-{index:06d}", audio_path=Path(path))
            for index, path in enumerate(paths, start=1)
        ]
        return self.predict_records(records, **kwargs)

    def predict_directory(
        self,
        directory: Path | str,
        *,
        recursive: bool = True,
        extensions: Sequence[str] | None = None,
        **kwargs,
    ) -> BatchPredictionResult:
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"批量推理目录不存在或不是目录: {root}")
        normalized = {
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in (extensions or DEFAULT_AUDIO_EXTENSIONS)
        }
        iterator = root.rglob("*") if recursive else root.glob("*")
        paths = sorted(
            (path for path in iterator if path.is_file() and path.suffix.lower() in normalized),
            key=lambda path: path.as_posix().casefold(),
        )
        return self.predict_files(paths, **kwargs)

    def predict_manifest(
        self,
        manifest: DatasetManifest | Path | str,
        *,
        split: str | None = None,
        **kwargs,
    ) -> BatchPredictionResult:
        dataset = manifest if isinstance(manifest, DatasetManifest) else DatasetManifest.load(manifest)
        return self.predict_records(dataset.resolved_records(split), **kwargs)


def write_batch_predictions(
    path: Path | str,
    result: BatchPredictionResult,
    *,
    format: Literal["jsonl", "csv"] | None = None,
) -> Path:
    """原子写入批量结果；失败记录与成功记录使用同一输出文件。"""
    target = Path(path)
    output_format = format or target.suffix.lower().lstrip(".")
    if output_format not in {"jsonl", "csv"}:
        raise ValueError("批量预测输出格式必须是 jsonl 或 csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        if output_format == "jsonl":
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for prediction in result.predictions:
                    row = {"status": "succeeded", **asdict(prediction)}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                for failure in result.failures:
                    row = {"status": "failed", **asdict(failure)}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            columns = [
                "status", "uid", "label_id", "emotion", "confidence",
                "probabilities", "audio_path", "error_type", "message",
            ]
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for prediction in result.predictions:
                    writer.writerow({
                        "status": "succeeded",
                        "uid": prediction.uid,
                        "label_id": prediction.label_id,
                        "emotion": prediction.emotion,
                        "confidence": prediction.confidence,
                        "probabilities": json.dumps(prediction.probabilities),
                    })
                for failure in result.failures:
                    writer.writerow({"status": "failed", **asdict(failure)})
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "PredictionFailure", "BatchPredictionResult", "BatchEmotionPredictor",
    "write_batch_predictions",
]
