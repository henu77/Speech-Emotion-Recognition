"""CLI 的薄编排层；所有领域行为委托给公开库 API。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

from torch.utils.data import DataLoader

from ser_lib.artifacts import (
    ModelCard,
    ModelArtifactManifest,
    export_model_artifact,
    load_model_artifact,
    verify_model_artifact,
)
from ser_lib.data import DatasetManifest, SERDataset
from ser_lib.engine import (
    Trainer,
    build_experiment_components,
    evaluate,
    load_checkpoint,
    load_experiment_config,
    write_evaluation_report,
)
from ser_lib.inference import (
    BatchEmotionPredictor,
    EmotionPredictor,
    write_batch_predictions,
)


def _labels(meta_labels: dict[int, dict[str, Any]]) -> dict[int, str]:
    """选择稳定展示名，优先英文，其次中文，最后使用标签 ID。"""
    return {
        index: str(values.get("en") or values.get("zh") or index)
        for index, values in sorted(meta_labels.items())
    }


def _loader(manifest, split, components, *, batch_size, workers, shuffle=False):
    records = manifest.resolved_records(split)
    dataset = SERDataset(records, components.audio_loader, components.pipeline)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=components.collator,
    )


def train_experiment(
    config_path: Path,
    *,
    split: str,
    batch_size: int,
    workers: int,
    resume: Path | None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    if config.trainer.checkpoint_dir is None:
        config = config.model_copy(update={
            "trainer": config.trainer.model_copy(update={
                "checkpoint_dir": config.output_dir / "checkpoints"
            })
        })
    components = build_experiment_components(config, train=True)
    manifest = DatasetManifest.load(config.data.manifest)
    batches = _loader(
        manifest, split, components, batch_size=batch_size, workers=workers, shuffle=True
    )
    trainer = Trainer.from_experiment(components.model, config)
    if resume is not None:
        trainer.resume_from(resume)
    history = trainer.fit(lambda: batches)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.output_dir / "history.json"
    temporary = history_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps([asdict(item) for item in history], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(history_path)
    last_checkpoint = (
        cast(Path, config.trainer.checkpoint_dir)
        / f"epoch-{trainer.last_completed_epoch:04d}.pt"
        if trainer.last_completed_epoch else resume
    )
    return {
        "output_dir": str(config.output_dir),
        "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
        "history": [asdict(item) for item in history],
        "resumed_from": str(resume) if resume else None,
    }


def evaluate_artifact(
    artifact: Path,
    *,
    manifest_path: Path | None,
    split: str,
    batch_size: int,
    workers: int,
    device: str,
    output: Path,
) -> dict[str, Any]:
    loaded = load_model_artifact(artifact, map_location=device)
    manifest = DatasetManifest.load(manifest_path or loaded.manifest.preprocessing["manifest"])
    batches = _loader(manifest, split, loaded, batch_size=batch_size, workers=workers)
    result = evaluate(
        loaded.model,
        batches,
        num_classes=len(loaded.manifest.labels),
        device=device,
        labels=loaded.manifest.labels,
    )
    write_evaluation_report(output, result)
    return {"output_dir": str(output), **result.summary_dict()}


def predict_artifact(
    artifact: Path,
    *,
    source: Path,
    split: str | None,
    batch_size: int,
    device: str,
    output: Path,
    keep_going: bool,
    recursive: bool,
    window_aggregation: Literal[
        "mean_logits", "mean_probabilities", "max_confidence"
    ] | None,
) -> dict[str, Any]:
    loaded = load_model_artifact(artifact, map_location=device)
    predictor = EmotionPredictor(
        loaded.model,
        loaded.audio_loader,
        loaded.pipeline,
        loaded.collator,
        loaded.manifest.labels,
        device=device,
        window_aggregation=window_aggregation,
    )
    batch = BatchEmotionPredictor(predictor)
    if source.is_dir():
        result = batch.predict_directory(
            source, recursive=recursive, batch_size=batch_size,
            fail_fast=not keep_going,
        )
    elif source.suffix.lower() in {".yaml", ".yml"}:
        result = batch.predict_manifest(
            source, split=split, batch_size=batch_size, fail_fast=not keep_going
        )
    else:
        result = batch.predict_files(
            [source], batch_size=batch_size, fail_fast=not keep_going
        )
    write_batch_predictions(output, result)
    return {
        "output": str(output),
        "total": result.total,
        "succeeded": result.succeeded,
        "failed": result.failed,
    }


def export_checkpoint_artifact(
    config_path: Path,
    checkpoint: Path,
    destination: Path,
    *,
    model_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    components = build_experiment_components(config, train=False)
    payload = load_checkpoint(
        checkpoint, components.model, map_location="cpu", restore_rng=False
    )
    manifest = DatasetManifest.load(config.data.manifest)
    source_labels = config.data.labels or manifest.meta.labels
    labels = _labels(source_labels)
    expected_classes = components.model.model_config.get("num_classes")
    if expected_classes is not None and len(labels) != expected_classes:
        raise ValueError(
            f"标签数 {len(labels)} 与模型 num_classes={expected_classes} 不一致"
        )
    target = export_model_artifact(
        destination,
        components.model,
        model_name=config.model.type,
        data_config=config.data,
        labels=labels,
        metrics=payload.get("metrics") or {},
        metadata={"checkpoint_epoch": payload.get("epoch")},
        model_card=ModelCard(**(model_card or {})),
    )
    return {"artifact": str(target), "model": config.model.type, "labels": labels}


def inspect_artifact(path: Path, *, verify: bool) -> dict[str, Any]:
    if verify:
        manifest = verify_model_artifact(path)
    else:
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest 不存在: {manifest_path}")
        manifest = ModelArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    return manifest.model_dump(mode="json")


__all__ = [
    "train_experiment", "evaluate_artifact", "predict_artifact",
    "export_checkpoint_artifact", "inspect_artifact",
]
