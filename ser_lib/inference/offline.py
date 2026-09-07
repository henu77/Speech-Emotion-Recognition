"""复用训练 Pipeline 的离线推理入口。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Literal, Mapping
import torch
from ser_lib.data.audio import AudioLoader
from ser_lib.data.collate import SERCollator
from ser_lib.data.pipeline import SamplePipeline
from ser_lib.data.types import AudioData, AudioRecord
from ser_lib.engine.trainer import move_batch_to_device
from ser_lib.models.base import SERModel

@dataclass(frozen=True, slots=True)
class PredictionResult:
    uid: str
    label_id: int
    emotion: str
    confidence: float
    probabilities: list[float]

class EmotionPredictor:
    def __init__(self, model: SERModel, audio_loader: AudioLoader,
                 pipeline: SamplePipeline, collator: SERCollator,
                 labels: Mapping[int, str] | None = None, *,
                 device: str | torch.device = "cpu",
                 window_aggregation: Literal["mean_logits", "mean_probabilities", "max_confidence"] | None = None) -> None:
        self.model, self.audio_loader = model, audio_loader
        self.pipeline, self.collator = pipeline, collator
        self.labels, self.device = dict(labels or {}), torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("推理请求 CUDA，但当前环境不可用")
        self.window_aggregation = window_aggregation
        model.to(self.device)

    @torch.inference_mode()
    def predict_file(self, path: Path | str, *, uid: str | None = None) -> PredictionResult:
        source = Path(path)
        record = AudioRecord(uid=uid or source.stem or "prediction", audio_path=source)
        return self.predict_record(record)

    @torch.inference_mode()
    def predict_record(self, record: AudioRecord) -> PredictionResult:
        """预测已解析路径的记录，并保留 UID 与片段范围。"""
        return self.predict_records([record])[0]

    @torch.inference_mode()
    def predict_audio(
        self, audio: AudioData, *, uid: str = "stream"
    ) -> PredictionResult:
        """预测已在内存中的标准 AudioData，供流式核心等调用方复用。"""
        record = AudioRecord(uid=uid, audio_path=audio.source_path)
        return self._predict_samples([self.pipeline(audio, record)], [record])[0]

    @torch.inference_mode()
    def predict_records(self, records: Sequence[AudioRecord]) -> list[PredictionResult]:
        """在一次模型 forward 中预测多个记录，并正确聚合各自的滑窗。"""
        if not records:
            return []
        samples = [
            self.pipeline(self.audio_loader.load(record), record)
            for record in records
        ]
        return self._predict_samples(samples, records)

    def _predict_samples(self, samples, records) -> list[PredictionResult]:
        batch = move_batch_to_device(self.collator(samples), self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            logits = self.model(batch).logits
        finally:
            self.model.train(was_training)
        if batch.window_map is None:
            if logits.shape[0] != len(records):
                raise ValueError("模型输出行数与输入记录数不一致")
            groups = [logits[index:index + 1] for index in range(len(records))]
        else:
            groups = []
            for index in range(len(records)):
                selected = logits[batch.window_map == index]
                if selected.shape[0] == 0:
                    raise ValueError(f"记录 {records[index].uid!r} 没有对应的滑窗输出")
                groups.append(selected)
        results = []
        for record, group in zip(records, groups):
            probabilities = self._aggregate(group).cpu()
            label_id = int(probabilities.argmax())
            results.append(PredictionResult(
                record.uid, label_id, self.labels.get(label_id, str(label_id)),
                float(probabilities[label_id]), [float(v) for v in probabilities.tolist()],
            ))
        return results

    def _aggregate(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2 or logits.shape[0] < 1:
            raise ValueError(f"模型 logits 必须是非空 [N,C]，实际 {tuple(logits.shape)}")
        if logits.shape[0] == 1:
            return torch.softmax(logits[0], -1)
        if self.window_aggregation is None:
            raise ValueError("单文件产生多个窗口输出，必须配置 window_aggregation")
        if self.window_aggregation == "mean_logits":
            return torch.softmax(logits.mean(0), -1)
        per_window = torch.softmax(logits, -1)
        if self.window_aggregation == "mean_probabilities":
            return per_window.mean(0)
        if self.window_aggregation == "max_confidence":
            return per_window[per_window.max(dim=1).values.argmax()]
        raise ValueError(f"未知 window_aggregation: {self.window_aggregation!r}")

__all__ = ["EmotionPredictor", "PredictionResult"]
