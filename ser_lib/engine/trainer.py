"""表示无关、可观测且可取消的 SER 分类训练器。"""

from __future__ import annotations

import random
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from ser_lib.core.events import (
    CancellationCheck,
    EventCallback,
    MetricEvent,
    ProgressEvent,
)
from ser_lib.data.types import SERBatch
from ser_lib.engine.config import ExperimentConfig, TrainerConfig
from ser_lib.engine.optim import (
    SchedulerConfig,
    build_optimizer,
    build_scheduler,
    parse_optimizer_config,
    parse_scheduler_config,
)
from ser_lib.models.base import SERModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    loss: float
    accuracy: float
    sample_count: int
    optimizer_steps: int = 0
    validation: dict[str, float] | None = None


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """为 Python 与 PyTorch 设置可复现 seed。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def move_batch_to_device(batch: SERBatch, device: torch.device) -> SERBatch:
    """将 batch 中的 tensor 移动到目标设备，保留元数据。"""
    return replace(
        batch,
        inputs={key: value.to(device) for key, value in batch.inputs.items()},
        lengths={key: value.to(device) for key, value in batch.lengths.items()},
        masks={key: value.to(device) for key, value in batch.masks.items()},
        labels=batch.labels.to(device) if batch.labels is not None else None,
        window_map=batch.window_map.to(device) if batch.window_map is not None else None,
    )


class Trainer:
    def __init__(
        self,
        model: SERModel,
        config: TrainerConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        loss_fn: torch.nn.Module | None = None,
        event_callback: EventCallback | None = None,
        cancellation: CancellationCheck | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainerConfig()
        try:
            self.device = torch.device(self.config.device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"无效训练设备: {self.config.device!r}") from exc
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("配置请求 CUDA，但当前环境不可用")
        if self.config.amp and self.device.type != "cuda":
            raise ValueError("AMP 当前仅支持 CUDA 设备")
        seed_everything(self.config.seed, deterministic=self.config.deterministic)
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = scheduler
        self.loss_fn = loss_fn.to(self.device) if loss_fn is not None else None
        self.event_callback = event_callback
        self.cancellation = cancellation
        self._scaler = torch.cuda.amp.GradScaler() if self.config.amp else None
        self.last_completed_epoch = 0
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.epochs_without_improvement = 0

    @classmethod
    def from_experiment(
        cls,
        model: SERModel,
        experiment: ExperimentConfig,
        *,
        event_callback: EventCallback | None = None,
        cancellation: CancellationCheck | None = None,
    ) -> "Trainer":
        """按白名单实验配置构造 optimizer、scheduler 和 Trainer。"""
        from ser_lib.models.registry import model_registry

        if model.model_spec.model_id != experiment.model.type:
            raise ValueError(
                f"实验模型 {experiment.model.type!r} 与实例声明 "
                f"{model.model_spec.model_id!r} 不一致"
            )
        expected_model_config = model_registry.validate_config(
            experiment.model.type, experiment.model.params
        )
        if expected_model_config != model.model_config:
            raise ValueError("实验 model.params 与模型实例的实际配置不一致")
        optimizer_config = parse_optimizer_config(experiment.optimizer)
        optimizer = build_optimizer(model.parameters(), optimizer_config)
        scheduler_config: SchedulerConfig | None = parse_scheduler_config(experiment.scheduler)
        scheduler = build_scheduler(optimizer, scheduler_config)
        from ser_lib.engine.objectives import ClassificationLoss

        num_classes = model.model_spec.num_classes
        if num_classes is None:
            raise ValueError("分类训练要求模型声明 num_classes")
        return cls(
            model,
            experiment.trainer,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=ClassificationLoss(experiment.loss, num_classes),
            event_callback=event_callback,
            cancellation=cancellation,
        )

    def _emit(self, event) -> None:
        if self.event_callback is not None:
            self.event_callback(event)

    def _check_cancelled(self) -> None:
        if self.cancellation is not None:
            self.cancellation.raise_if_cancelled()

    def _optimizer_step(self) -> None:
        if self._scaler is not None:
            self._scaler.unscale_(self.optimizer)
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
        if self._scaler is None:
            self.optimizer.step()
        else:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def train_epoch(self, batches: Iterable[SERBatch], *, epoch: int) -> EpochResult:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        optimizer_steps = 0
        pending_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(batches, start=1):
            self._check_cancelled()
            if batch.labels is None:
                raise ValueError("训练 batch 必须包含 labels")
            batch = move_batch_to_device(batch, self.device)
            labels = batch.labels
            assert labels is not None
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.config.amp,
            ):
                output = self.model(batch)
                loss = (
                    self.loss_fn(output.logits, labels)
                    if self.loss_fn is not None
                    else output.loss if output.loss is not None
                    else F.cross_entropy(output.logits, labels)
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"训练 loss 非有限值: {loss.item()}")
            scaled_loss = loss / self.config.gradient_accumulation_steps
            if self._scaler is None:
                scaled_loss.backward()
            else:
                self._scaler.scale(scaled_loss).backward()
            pending_batches += 1
            if pending_batches == self.config.gradient_accumulation_steps:
                self._optimizer_step()
                optimizer_steps += 1
                pending_batches = 0

            count = int(labels.shape[0])
            total_samples += count
            total_loss += float(loss.detach()) * count
            total_correct += int((output.logits.detach().argmax(-1) == labels).sum())
            self._emit(ProgressEvent(
                stage="train_batch", completed=batch_index,
                message=f"epoch={epoch}",
            ))

        if pending_batches:
            self._optimizer_step()
            optimizer_steps += 1
        if total_samples == 0:
            raise ValueError("训练数据为空")

        result = EpochResult(
            epoch=epoch,
            loss=total_loss / total_samples,
            accuracy=total_correct / total_samples,
            sample_count=total_samples,
            optimizer_steps=optimizer_steps,
        )
        self._emit(MetricEvent("loss", result.loss, step=epoch, split="train"))
        self._emit(MetricEvent("accuracy", result.accuracy, step=epoch, split="train"))
        return result

    def fit(
        self,
        train_batches: Iterable[SERBatch] | Callable[[], Iterable[SERBatch]],
        *,
        val_batches: Iterable[SERBatch] | Callable[[], Iterable[SERBatch]] | None = None,
        on_epoch_end: Callable[[EpochResult], None] | None = None,
        start_epoch: int | None = None,
    ) -> list[EpochResult]:
        from ser_lib.engine.checkpoint import save_checkpoint

        if self.config.early_stopping_patience is not None and val_batches is None:
            raise ValueError("启用 early stopping 时必须提供 val_batches")
        first_epoch = self.last_completed_epoch + 1 if start_epoch is None else start_epoch
        if first_epoch < 1:
            raise ValueError("start_epoch 必须 >= 1")
        history: list[EpochResult] = []
        for epoch in range(first_epoch, self.config.epochs + 1):
            self._check_cancelled()
            batches = train_batches() if callable(train_batches) else train_batches
            result = self.train_epoch(batches, epoch=epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            self.last_completed_epoch = epoch
            improved = False
            if val_batches is not None and epoch % self.config.validation_interval == 0:
                from ser_lib.engine.evaluator import evaluate

                validation_batches = val_batches() if callable(val_batches) else val_batches
                num_classes = self.model.model_spec.num_classes
                if num_classes is None:
                    raise ValueError("验证要求模型声明 num_classes")
                validation_result = evaluate(
                    self.model,
                    validation_batches,
                    num_classes=num_classes,
                    device=self.device,
                    event_callback=self.event_callback,
                    cancellation=self.cancellation,
                    loss_fn=self.loss_fn,
                )
                validation = {
                    "loss": validation_result.loss,
                    "accuracy": validation_result.accuracy,
                    "war": validation_result.war,
                    "uar": validation_result.uar,
                    "macro_f1": validation_result.macro_f1,
                }
                result = replace(result, validation=validation)
                for name, value in validation.items():
                    self._emit(MetricEvent(name, value, step=epoch, split="val"))
                monitored = validation[self.config.monitor.removeprefix("val_")]
                improved = self._is_improved(monitored)
                if improved:
                    self.best_metric = monitored
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
            history.append(result)
            logger.info(
                "epoch=%d train_loss=%.6f train_accuracy=%.4f validation=%s",
                epoch, result.loss, result.accuracy, result.validation,
            )
            self._check_cancelled()
            if self.config.checkpoint_dir is not None:
                metrics = {"loss": result.loss, "accuracy": result.accuracy}
                if result.validation is not None:
                    metrics.update({f"val_{k}": v for k, v in result.validation.items()})
                metadata = {
                    "best_metric": self.best_metric,
                    "best_epoch": self.best_epoch,
                    "epochs_without_improvement": self.epochs_without_improvement,
                    "monitor": self.config.monitor,
                }
                save_checkpoint(
                    self.config.checkpoint_dir / f"epoch-{epoch:04d}.pt",
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    scheduler=self.scheduler,
                    scaler=self._scaler,
                    metrics=metrics,
                    metadata=metadata,
                    trainer_config=self.config.model_dump(mode="json"),
                )
                if self.config.save_last:
                    save_checkpoint(
                        self.config.checkpoint_dir / "last.pt", self.model, self.optimizer,
                        epoch=epoch, scheduler=self.scheduler, scaler=self._scaler,
                        metrics=metrics, metadata=metadata,
                        trainer_config=self.config.model_dump(mode="json"),
                    )
                if improved and self.config.save_best:
                    save_checkpoint(
                        self.config.checkpoint_dir / "best.pt", self.model, self.optimizer,
                        epoch=epoch, scheduler=self.scheduler, scaler=self._scaler,
                        metrics=metrics, metadata=metadata,
                        trainer_config=self.config.model_dump(mode="json"),
                    )
            if on_epoch_end is not None:
                on_epoch_end(result)
            if (
                self.config.early_stopping_patience is not None
                and self.epochs_without_improvement >= self.config.early_stopping_patience
            ):
                break
        return history

    def _is_improved(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        delta = self.config.early_stopping_min_delta
        if self.config.monitor == "val_loss":
            return value < self.best_metric - delta
        return value > self.best_metric + delta

    def resume_from(self, path, *, restore_rng: bool = True) -> dict:
        """恢复训练状态，并使下次 ``fit`` 从 checkpoint 的下一 epoch 开始。"""
        from ser_lib.engine.checkpoint import load_checkpoint

        payload = load_checkpoint(
            path,
            self.model,
            self.optimizer,
            scheduler=self.scheduler,
            scaler=self._scaler,
            map_location=self.device,
            restore_rng=restore_rng,
            expected_trainer_config=self.config.model_dump(mode="json"),
        )
        epoch = payload.get("epoch")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("checkpoint epoch 非法")
        self.last_completed_epoch = epoch
        metadata = payload.get("metadata") or {}
        if metadata.get("monitor") in (None, self.config.monitor):
            best_metric = metadata.get("best_metric")
            best_epoch = metadata.get("best_epoch")
            without_improvement = metadata.get("epochs_without_improvement", 0)
            self.best_metric = float(best_metric) if best_metric is not None else None
            self.best_epoch = int(best_epoch) if best_epoch is not None else None
            self.epochs_without_improvement = int(without_improvement)
        return payload


__all__ = [
    "TrainerConfig", "EpochResult", "Trainer", "move_batch_to_device", "seed_everything"
]
