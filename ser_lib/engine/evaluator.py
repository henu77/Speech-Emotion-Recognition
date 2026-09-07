"""SER 分类模型评估。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ser_lib.data.types import SERBatch
from ser_lib.engine.trainer import move_batch_to_device
from ser_lib.models.base import SERModel


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    macro_f1: float
    uar: float
    confusion_matrix: torch.Tensor
    sample_count: int


@torch.inference_mode()
def evaluate(
    model: SERModel,
    batches: Iterable[SERBatch],
    *,
    num_classes: int,
    device: str | torch.device = "cpu",
) -> EvaluationResult:
    if num_classes < 2:
        raise ValueError("num_classes 必须 >= 2")
    target_device = torch.device(device)
    model.to(target_device)
    was_training = model.training
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    try:
        for batch in batches:
            if batch.labels is None:
                raise ValueError("评估 batch 必须包含 labels")
            batch = move_batch_to_device(batch, target_device)
            predictions = model(batch).logits.argmax(dim=-1)
            labels = batch.labels
            if predictions.shape != labels.shape:
                raise ValueError("模型预测数量与标签数量不一致")
            flat = labels * num_classes + predictions
            counts = torch.bincount(flat, minlength=num_classes * num_classes)
            confusion += counts.reshape(num_classes, num_classes).cpu()
    finally:
        model.train(was_training)

    total = int(confusion.sum().item())
    if total == 0:
        raise ValueError("评估数据为空")
    tp = confusion.diag().to(torch.float64)
    support = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    recall = tp / support.clamp_min(1)
    precision = tp / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float64).eps)
    present = support > 0
    return EvaluationResult(
        accuracy=float(tp.sum().item() / total),
        macro_f1=float(f1[present].mean().item()),
        uar=float(recall[present].mean().item()),
        confusion_matrix=confusion,
        sample_count=total,
    )


__all__ = ["EvaluationResult", "evaluate"]
