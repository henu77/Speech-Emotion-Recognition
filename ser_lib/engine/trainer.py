"""最小、表示无关的 SER 分类训练器。"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable
import torch
import torch.nn.functional as F
from ser_lib.data.types import SERBatch
from ser_lib.models.base import SERModel

@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    gradient_clip_norm: float | None = None
    checkpoint_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("epochs>=1、learning_rate>0、weight_decay>=0")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm 必须 > 0")

@dataclass(frozen=True)
class EpochResult:
    epoch: int
    loss: float
    accuracy: float
    sample_count: int

def move_batch_to_device(batch: SERBatch, device: torch.device) -> SERBatch:
    return replace(
        batch,
        inputs={k: v.to(device) for k, v in batch.inputs.items()},
        lengths={k: v.to(device) for k, v in batch.lengths.items()},
        masks={k: v.to(device) for k, v in batch.masks.items()},
        labels=batch.labels.to(device) if batch.labels is not None else None,
        window_map=batch.window_map.to(device) if batch.window_map is not None else None,
    )

class Trainer:
    def __init__(self, model: SERModel, config: TrainerConfig | None = None, *,
                 optimizer: torch.optim.Optimizer | None = None) -> None:
        self.model, self.config = model, config or TrainerConfig()
        self.device = torch.device(self.config.device)
        model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_epoch(self, batches: Iterable[SERBatch], *, epoch: int) -> EpochResult:
        self.model.train()
        total_loss = 0.0
        total_correct = total_samples = 0
        for batch in batches:
            if batch.labels is None:
                raise ValueError("训练 batch 必须包含 labels")
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch)
            loss = output.loss if output.loss is not None else F.cross_entropy(output.logits, batch.labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"训练 loss 非有限值: {loss.item()}")
            loss.backward()
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            count = int(batch.labels.shape[0])
            total_samples += count
            total_loss += float(loss.detach()) * count
            total_correct += int((output.logits.detach().argmax(-1) == batch.labels).sum())
        if total_samples == 0:
            raise ValueError("训练数据为空")
        return EpochResult(epoch, total_loss / total_samples,
                           total_correct / total_samples, total_samples)

    def fit(self, train_batches: Iterable[SERBatch] | Callable[[], Iterable[SERBatch]], *,
            on_epoch_end: Callable[[EpochResult], None] | None = None) -> list[EpochResult]:
        from ser_lib.engine.checkpoint import save_checkpoint
        history = []
        for epoch in range(1, self.config.epochs + 1):
            batches = train_batches() if callable(train_batches) else train_batches
            result = self.train_epoch(batches, epoch=epoch)
            history.append(result)
            if self.config.checkpoint_dir is not None:
                save_checkpoint(
                    self.config.checkpoint_dir / f"epoch-{epoch:04d}.pt", self.model,
                    self.optimizer, epoch=epoch,
                    metrics={"loss": result.loss, "accuracy": result.accuracy},
                )
            if on_epoch_end:
                on_epoch_end(result)
        return history

__all__ = ["TrainerConfig", "EpochResult", "Trainer", "move_batch_to_device"]
