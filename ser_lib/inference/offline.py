"""复用训练 Pipeline 的离线推理入口。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping
import torch
from ser_lib.data.audio import AudioLoader
from ser_lib.data.collate import SERCollator
from ser_lib.data.pipeline import SamplePipeline
from ser_lib.data.types import AudioRecord
from ser_lib.engine.trainer import move_batch_to_device
from ser_lib.models.base import SERModel

@dataclass(frozen=True)
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
        self.window_aggregation = window_aggregation
        model.to(self.device)

    @torch.inference_mode()
    def predict_file(self, path: Path | str, *, uid: str | None = None) -> PredictionResult:
        source = Path(path)
        record = AudioRecord(uid=uid or source.stem or "prediction", audio_path=source)
        sample = self.pipeline(self.audio_loader.load(record), record)
        batch = move_batch_to_device(self.collator([sample]), self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            logits = self.model(batch).logits
        finally:
            self.model.train(was_training)
        probabilities = self._aggregate(logits).cpu()
        label_id = int(probabilities.argmax())
        return PredictionResult(
            record.uid, label_id, self.labels.get(label_id, str(label_id)),
            float(probabilities[label_id]), [float(v) for v in probabilities.tolist()],
        )

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
