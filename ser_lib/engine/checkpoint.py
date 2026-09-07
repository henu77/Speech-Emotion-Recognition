"""可信本地训练 checkpoint 的原子保存与恢复。"""
from pathlib import Path
from typing import Any
import torch
from ser_lib.models.base import SERModel

def save_checkpoint(path: Path | str, model: SERModel,
                    optimizer: torch.optim.Optimizer | None, *, epoch: int,
                    metrics: dict[str, float] | None = None,
                    metadata: dict[str, Any] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save({
        "format_version": 1, "model_id": model.model_spec.model_id,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "epoch": int(epoch), "metrics": dict(metrics or {}),
        "metadata": dict(metadata or {}),
    }, temporary)
    temporary.replace(target)
    return target

def load_checkpoint(path: Path | str, model: SERModel,
                    optimizer: torch.optim.Optimizer | None = None, *,
                    map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """仅加载来源可信、由本库生成的训练 checkpoint。"""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("不支持的 checkpoint 格式")
    if payload.get("model_id") != model.model_spec.model_id:
        raise ValueError("checkpoint 与当前模型类型不一致")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
