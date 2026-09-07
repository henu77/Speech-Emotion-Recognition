"""可信本地训练 checkpoint 的原子保存与完整恢复。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from ser_lib.models.base import SERModel


CHECKPOINT_FORMAT_VERSION = 2


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path | str,
    model: SERModel,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    metrics: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
    trainer_config: dict[str, Any] | None = None,
) -> Path:
    """原子保存继续训练所需状态。

    Checkpoint 使用 pickle，只能加载由本库在可信本地环境生成的文件；用于分发
    的模型必须使用 artifact。
    """
    if epoch < 0:
        raise ValueError("checkpoint epoch 不能为负数")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_id": model.model_spec.model_id,
        "model_config": model.model_config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "rng_state": _rng_state(),
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
        "metadata": dict(metadata or {}),
        "trainer_config": dict(trainer_config or {}),
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_checkpoint(
    path: Path | str,
    model: SERModel,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
    expected_trainer_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """加载可信 checkpoint；兼容格式 v1，完整恢复格式 v2。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint 顶层结构必须是映射")
    version = payload.get("format_version")
    if version not in (1, CHECKPOINT_FORMAT_VERSION):
        raise ValueError(f"不支持的 checkpoint 格式: {version!r}")
    if payload.get("model_id") != model.model_spec.model_id:
        raise ValueError("checkpoint 与当前模型类型不一致")
    saved_model_config = payload.get("model_config")
    if saved_model_config is not None and saved_model_config != model.model_config:
        raise ValueError("checkpoint 与当前模型配置不一致")
    if expected_trainer_config is not None:
        saved_trainer_config = payload.get("trainer_config")
        if saved_trainer_config and saved_trainer_config != expected_trainer_config:
            raise ValueError("checkpoint trainer_config 与当前训练配置不一致")
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint 缺少合法 model_state")
    model.load_state_dict(model_state)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng and version >= 2 and isinstance(payload.get("rng_state"), dict):
        _restore_rng_state(payload["rng_state"])
    return payload


__all__ = ["CHECKPOINT_FORMAT_VERSION", "save_checkpoint", "load_checkpoint"]
