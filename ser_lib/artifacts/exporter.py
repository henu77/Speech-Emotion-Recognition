"""导出模型 state_dict、预处理配置和标签为独立目录。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import torch
from ser_lib import __version__
from ser_lib.artifacts.manifest import ModelArtifactManifest
from ser_lib.data.config import DataConfig
from ser_lib.models.base import SERModel

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def export_model_artifact(
    directory: Path | str, model: SERModel, *, model_name: str,
    model_params: Mapping[str, Any], data_config: DataConfig,
    labels: Mapping[int, str], metrics: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    weights = target / "model_state.pt"
    weights_tmp = target / "model_state.pt.tmp"
    # 只保存 tensor state_dict；加载端强制 weights_only=True。
    torch.save(model.state_dict(), weights_tmp)
    weights_tmp.replace(weights)
    manifest = ModelArtifactManifest(
        library_version=__version__, model_name=model_name,
        model_params=dict(model_params), weights_sha256=_sha256(weights),
        preprocessing=data_config.model_dump(mode="json"), labels=dict(labels),
        metrics=dict(metrics or {}), metadata=dict(metadata or {}),
    )
    manifest_path = target / "manifest.json"
    temporary = target / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return target
