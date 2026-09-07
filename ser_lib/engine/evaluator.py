"""SER 分类模型评估、样本预测和机器可读报告。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ser_lib.core.events import CancellationCheck, EventCallback, ProgressEvent
from ser_lib.data.types import SERBatch
from ser_lib.engine.trainer import move_batch_to_device
from ser_lib.models.base import SERModel


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    label_id: int
    label_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    uid: str
    target: int
    predicted: int
    confidence: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    accuracy: float
    macro_f1: float
    uar: float
    confusion_matrix: torch.Tensor
    sample_count: int
    loss: float
    war: float
    per_class: tuple[ClassMetrics, ...]
    predictions: tuple[PredictionRecord, ...]

    def summary_dict(self) -> dict[str, object]:
        """返回不含样本明细的 JSON-safe 聚合报告。"""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "war": self.war,
            "uar": self.uar,
            "macro_f1": self.macro_f1,
            "sample_count": self.sample_count,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class": [asdict(item) for item in self.per_class],
        }


def _validate_labels(labels: Mapping[int, str] | None, num_classes: int) -> dict[int, str]:
    if labels is None:
        return {index: str(index) for index in range(num_classes)}
    normalized = dict(labels)
    expected = set(range(num_classes))
    if set(normalized) != expected:
        raise ValueError(
            f"labels 必须覆盖 0..{num_classes - 1}，实际: {sorted(normalized)}"
        )
    if any(not isinstance(name, str) or not name for name in normalized.values()):
        raise ValueError("labels 名称必须是非空字符串")
    return normalized


@torch.inference_mode()
def evaluate(
    model: SERModel,
    batches: Iterable[SERBatch],
    *,
    num_classes: int,
    device: str | torch.device = "cpu",
    labels: Mapping[int, str] | None = None,
    event_callback: EventCallback | None = None,
    cancellation: CancellationCheck | None = None,
) -> EvaluationResult:
    """评估分类模型，并保留每个 batch 行对应的样本预测。

    Sliding collator 产生的窗口被视为独立行；原始样本级窗口聚合属于推理层，
    评估器不会根据重复 UID 隐式猜测聚合策略。
    """
    if num_classes < 2:
        raise ValueError("num_classes 必须 >= 2")
    label_names = _validate_labels(labels, num_classes)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("评估请求 CUDA，但当前环境不可用")
    model.to(target_device)
    was_training = model.training
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    records: list[PredictionRecord] = []
    total_loss = 0.0
    total_samples = 0
    try:
        for batch_index, batch in enumerate(batches, start=1):
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            if batch.labels is None:
                raise ValueError("评估 batch 必须包含 labels")
            batch = move_batch_to_device(batch, target_device)
            labels_tensor = batch.labels
            assert labels_tensor is not None
            output = model(batch)
            if output.logits.shape != (labels_tensor.shape[0], num_classes):
                raise ValueError(
                    f"模型 logits 期望 [B,{num_classes}]，实际 {tuple(output.logits.shape)}"
                )
            if torch.any(labels_tensor < 0) or torch.any(labels_tensor >= num_classes):
                raise ValueError("评估标签超出 [0, num_classes) 范围")
            loss = output.loss if output.loss is not None else F.cross_entropy(
                output.logits, labels_tensor
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("评估 loss 为 NaN/Inf")
            probabilities = output.logits.softmax(dim=-1)
            confidence, predictions = probabilities.max(dim=-1)
            flat = labels_tensor * num_classes + predictions
            counts = torch.bincount(flat, minlength=num_classes * num_classes)
            confusion += counts.reshape(num_classes, num_classes).cpu()
            count = int(labels_tensor.shape[0])
            total_loss += float(loss) * count
            total_samples += count
            for index, uid in enumerate(batch.uids):
                records.append(PredictionRecord(
                    uid=uid,
                    target=int(labels_tensor[index]),
                    predicted=int(predictions[index]),
                    confidence=float(confidence[index]),
                    probabilities=tuple(float(value) for value in probabilities[index].cpu()),
                ))
            if event_callback is not None:
                event_callback(ProgressEvent(
                    stage="evaluate_batch", completed=batch_index,
                    message=f"samples={total_samples}",
                ))
    finally:
        model.train(was_training)

    total = int(confusion.sum().item())
    if total == 0:
        raise ValueError("评估数据为空")
    true_positive = confusion.diag().to(torch.float64)
    support = confusion.sum(dim=1).to(torch.float64)
    predicted_count = confusion.sum(dim=0).to(torch.float64)
    recall = true_positive / support.clamp_min(1)
    precision = true_positive / predicted_count.clamp_min(1)
    denominator = precision + recall
    f1 = torch.where(
        denominator > 0,
        2 * precision * recall / denominator.clamp_min(torch.finfo(torch.float64).eps),
        torch.zeros_like(denominator),
    )
    present = support > 0
    accuracy = float(true_positive.sum().item() / total)
    # WAR 是按 support 加权的 recall；单标签分类中与 accuracy 数值相同。
    war = float((recall * support).sum().item() / total)
    per_class = tuple(
        ClassMetrics(
            label_id=index,
            label_name=label_names[index],
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index in range(num_classes)
    )
    return EvaluationResult(
        accuracy=accuracy,
        macro_f1=float(f1[present].mean()),
        uar=float(recall[present].mean()),
        confusion_matrix=confusion,
        sample_count=total,
        loss=total_loss / total_samples,
        war=war,
        per_class=per_class,
        predictions=tuple(records),
    )


def write_evaluation_report(directory: Path | str, result: EvaluationResult) -> Path:
    """原子写入 ``metrics.json`` 与 ``predictions.jsonl``。"""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    metrics_path = target / "metrics.json"
    metrics_tmp = target / "metrics.json.tmp"
    metrics_tmp.write_text(
        json.dumps(result.summary_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    predictions_path = target / "predictions.jsonl"
    predictions_tmp = target / "predictions.jsonl.tmp"
    with predictions_tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for record in result.predictions:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    metrics_tmp.replace(metrics_path)
    predictions_tmp.replace(predictions_path)
    return target


__all__ = [
    "ClassMetrics", "PredictionRecord", "EvaluationResult",
    "evaluate", "write_evaluation_report",
]
