"""将模型、预处理和标签原子导出为安全 artifact。"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from safetensors.torch import save_file

from ser_lib import __version__
from ser_lib.artifacts.manifest import ModelArtifactManifest, ModelCard
from ser_lib.data.config import DataConfig
from ser_lib.models.base import SERModel
from ser_lib.models.registry import model_registry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _model_card_markdown(card: ModelCard, model_name: str, labels: Mapping[int, str]) -> str:
    languages = ", ".join(card.language) if card.language else "未声明"
    limitations = "\n".join(f"- {item}" for item in card.limitations) or "- 未声明"
    label_text = "\n".join(f"- {index}: {name}" for index, name in sorted(labels.items()))
    return (
        f"# {model_name}\n\n{card.description or '暂无模型描述。'}\n\n"
        f"## Intended use\n\n{card.intended_use or '未声明'}\n\n"
        f"## Training data\n\n{card.dataset or '未声明'}\n\n"
        f"## Language\n\n{languages}\n\n"
        f"## License\n\n{card.license or '未声明'}\n\n"
        f"## Labels\n\n{label_text}\n\n"
        f"## Limitations\n\n{limitations}\n"
    )


def export_model_artifact(
    directory: Path | str,
    model: SERModel,
    *,
    model_name: str,
    model_params: Mapping[str, Any] | None = None,
    data_config: DataConfig,
    labels: Mapping[int, str],
    metrics: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
    model_card: ModelCard | Mapping[str, Any] | None = None,
) -> Path:
    """导出 schema v2 artifact；目标必须不存在，避免覆盖已有模型。"""
    if model.model_spec.model_id != model_name:
        raise ValueError(
            f"model_name={model_name!r} 与模型声明 {model.model_spec.model_id!r} 不一致"
        )
    resolved_params = model.model_config if model_params is None else dict(model_params)
    validated_params = model_registry.validate_config(model_name, resolved_params)
    if validated_params != model.model_config:
        raise ValueError("model_params 与模型实例的实际配置不一致")
    normalized_labels = dict(labels)
    card = model_card if isinstance(model_card, ModelCard) else ModelCard(**dict(model_card or {}))

    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"artifact 目标已存在，拒绝覆盖: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        weights = temporary / "weights.safetensors"
        state = {
            key: tensor.detach().cpu().contiguous()
            for key, tensor in model.state_dict().items()
        }
        save_file(state, weights, metadata={"model_name": model_name, "library": "ser_lib"})

        preprocessing = data_config.model_dump(mode="json")
        _write_json(temporary / "data_config.json", preprocessing)
        _write_json(temporary / "model_config.json", validated_params)
        _write_json(temporary / "labels.json", normalized_labels)
        _write_json(temporary / "metrics.json", dict(metrics or {}))
        (temporary / "README.md").write_text(
            _model_card_markdown(card, model_name, normalized_labels), encoding="utf-8"
        )
        component_files = [
            "weights.safetensors", "data_config.json", "model_config.json",
            "labels.json", "metrics.json", "README.md",
        ]
        hashes = {name: _sha256(temporary / name) for name in component_files}
        manifest = ModelArtifactManifest(
            schema_version=2,
            library_version=__version__,
            model_name=model_name,
            model_params=validated_params,
            input_specs={
                key: {
                    "layout": spec.layout,
                    "dtype": str(spec.dtype).removeprefix("torch."),
                    "feature_dim": spec.feature_dim,
                    "time_axis": spec.time_axis,
                    "pad_value": spec.pad_value,
                }
                for key, spec in model.model_spec.required_inputs.items()
            },
            weights_file="weights.safetensors",
            weights_format="safetensors",
            weights_sha256=hashes["weights.safetensors"],
            files_sha256=hashes,
            preprocessing=preprocessing,
            labels=normalized_labels,
            metrics=dict(metrics or {}),
            model_card=card,
            metadata=dict(metadata or {}),
        )
        _write_json(temporary / "manifest.json", manifest.model_dump(mode="json"))
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


__all__ = ["export_model_artifact"]
