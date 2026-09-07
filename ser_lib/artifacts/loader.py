"""校验并加载模型 artifact。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from ser_lib import __version__
from ser_lib.artifacts.manifest import ModelArtifactManifest
from ser_lib.data.audio import AudioLoader
from ser_lib.data.collate import SERCollator, build_collator
from ser_lib.data.config import DataConfig
from ser_lib.data.pipeline import SamplePipeline, build_components
from ser_lib.data.validation import validate_compatibility
from ser_lib.models.base import SERModel
from ser_lib.models.registry import model_registry


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    manifest: ModelArtifactManifest
    model: SERModel
    audio_loader: AudioLoader
    pipeline: SamplePipeline
    collator: SERCollator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _major(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"artifact library_version 非法: {version!r}") from exc


def _safe_component_path(source: Path, name: str) -> Path:
    candidate = source / name
    if candidate.parent.resolve() != source.resolve():
        raise ValueError(f"artifact 文件名越出根目录: {name!r}")
    return candidate


def verify_model_artifact(directory: Path | str) -> ModelArtifactManifest:
    """验证 manifest、版本、文件存在性与全部校验和，不加载模型。"""
    source = Path(directory)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"artifact manifest 不存在: {manifest_path}")
    try:
        manifest = ModelArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"artifact manifest 无法读取或校验失败: {manifest_path}") from exc
    if manifest.schema_version >= 2 and _major(manifest.library_version) != _major(__version__):
        raise ValueError(
            f"artifact 需要 ser_lib {manifest.library_version}，当前版本 {__version__} 不兼容"
        )
    weights = _safe_component_path(source, manifest.weights_file)
    if not weights.is_file():
        raise FileNotFoundError(f"模型权重不存在: {weights}")
    if _sha256(weights) != manifest.weights_sha256:
        raise ValueError("模型权重 SHA-256 校验失败，文件可能损坏或被修改")
    if manifest.schema_version >= 2:
        if not manifest.files_sha256:
            raise ValueError("schema v2 artifact 缺少 files_sha256")
        if manifest.files_sha256.get(manifest.weights_file) != manifest.weights_sha256:
            raise ValueError("weights_sha256 与 files_sha256 不一致")
        for name, expected in manifest.files_sha256.items():
            path = _safe_component_path(source, name)
            if not path.is_file():
                raise FileNotFoundError(f"artifact 组成文件不存在: {path}")
            if _sha256(path) != expected:
                raise ValueError(f"artifact 文件 SHA-256 校验失败: {name}")
    return manifest


def _validate_external_metadata(source: Path, manifest: ModelArtifactManifest) -> None:
    if manifest.schema_version < 2:
        return
    expected = {
        "data_config.json": manifest.preprocessing,
        "model_config.json": manifest.model_params,
        "labels.json": {str(key): value for key, value in manifest.labels.items()},
        "metrics.json": manifest.metrics,
    }
    for name, embedded in expected.items():
        actual = json.loads((source / name).read_text(encoding="utf-8"))
        if actual != embedded:
            raise ValueError(f"artifact {name} 与 manifest 内容不一致")


def load_model_artifact(
    directory: Path | str,
    *,
    map_location: str | torch.device = "cpu",
    allow_legacy_pickle: bool = False,
) -> LoadedArtifact:
    """验证并加载 artifact；旧 v1 pickle 必须显式授权。"""
    source = Path(directory)
    manifest = verify_model_artifact(source)
    _validate_external_metadata(source, manifest)
    target_device = torch.device(map_location)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("artifact 加载请求 CUDA，但当前环境不可用")
    data_config = DataConfig.model_validate(manifest.preprocessing)
    model = model_registry.create(manifest.model_name, **manifest.model_params)
    audio_loader, pipeline = build_components(data_config, train=False)
    validate_compatibility(
        pipeline.output_specs,
        model.model_spec,
        data_config.batching,
        num_classes=len(manifest.labels),
        sample_rate=data_config.audio.target_sample_rate,
    )
    collator = build_collator(pipeline.output_specs, data_config.batching)
    weights = source / manifest.weights_file
    if manifest.weights_format == "safetensors":
        state = load_file(weights, device="cpu")
    elif manifest.weights_format == "pytorch" and allow_legacy_pickle:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    else:
        raise ValueError(
            "旧 PyTorch artifact 可能包含 pickle；仅可信文件可设置 allow_legacy_pickle=True"
        )
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("artifact 权重不是合法 tensor state_dict")
    model.load_state_dict(state, strict=True)
    model.to(target_device)
    return LoadedArtifact(manifest, model, audio_loader, pipeline, collator)


__all__ = ["LoadedArtifact", "verify_model_artifact", "load_model_artifact"]
