"""校验并加载模型 artifact。"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import torch
from ser_lib.artifacts.manifest import ModelArtifactManifest
from ser_lib.data.collate import SERCollator, build_collator
from ser_lib.data.config import DataConfig
from ser_lib.data.pipeline import SamplePipeline, build_components
from ser_lib.data.audio import AudioLoader
from ser_lib.data.validation import validate_compatibility
from ser_lib.models.base import SERModel
from ser_lib.models.registry import model_registry

@dataclass(frozen=True)
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

def load_model_artifact(directory: Path | str, *,
                        map_location: str | torch.device = "cpu") -> LoadedArtifact:
    source = Path(directory)
    manifest_path = source / "manifest.json"
    manifest = ModelArtifactManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    weights = source / manifest.weights_file
    if not weights.is_file():
        raise FileNotFoundError(f"模型权重不存在: {weights}")
    actual_hash = _sha256(weights)
    if actual_hash != manifest.weights_sha256:
        raise ValueError("模型权重 SHA-256 校验失败，文件可能损坏或被修改")
    data_config = DataConfig.model_validate(manifest.preprocessing)
    model = model_registry.create(manifest.model_name, **manifest.model_params)
    validate_compatibility(
        # build_components 后再取准确 output spec；这里提前构建以复用注册验证。
        (pipeline_pair := build_components(data_config, train=False))[1].output_specs,
        model.model_spec, data_config.batching, num_classes=len(manifest.labels),
    )
    audio_loader, pipeline = pipeline_pair
    collator = build_collator(pipeline.output_specs, data_config.batching)
    state = torch.load(weights, map_location=map_location, weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError("artifact 权重不是合法 state_dict")
    model.load_state_dict(state)
    model.to(torch.device(map_location))
    return LoadedArtifact(manifest, model, audio_loader, pipeline, collator)
