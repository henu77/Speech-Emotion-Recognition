"""不复制领域逻辑的 ``ser`` 命令行界面。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import yaml
from pydantic import ValidationError

from ser_lib import __version__
from ser_lib.core import ConfigurationError, SERError
from ser_lib.data import (
    DatasetManifest,
    ManifestError,
    default_registry,
    profile_manifest_audio,
)
from ser_lib.models import model_registry
from ser_lib.cli.workflows import (
    evaluate_artifact,
    export_checkpoint_artifact,
    inspect_artifact,
    predict_artifact,
    train_experiment,
)


EXIT_OK = 0
EXIT_CONFIG = 3
EXIT_DATA = 4
EXIT_RUNTIME = 5
COMPONENT_NAMESPACES = (
    "importer", "representation", "waveform_transform", "feature_transform"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ser", description="SER 基础库命令行工具")
    parser.add_argument("--version", action="version", version=f"ser-lib {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    components = commands.add_parser("components", help="查看已注册组件")
    component_commands = components.add_subparsers(dest="components_command", required=True)
    component_list = component_commands.add_parser("list", help="列出组件 descriptor")
    component_list.add_argument(
        "--kind", choices=(*COMPONENT_NAMESPACES, "model", "all"), default="all"
    )
    component_list.add_argument("--json", action="store_true", dest="as_json")

    dataset = commands.add_parser("dataset", help="数据集导入与检查")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)

    scan = dataset_commands.add_parser("scan", help="扫描外部数据但不写入")
    _add_importer_arguments(scan, destination=False)

    convert = dataset_commands.add_parser("import", help="转换为标准 manifest")
    _add_importer_arguments(convert, destination=True)

    validate = dataset_commands.add_parser("validate", help="校验标准 dataset.yaml")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--split")
    validate.add_argument("--check-files", action="store_true")
    validate.add_argument("--json", action="store_true", dest="as_json")

    stats = dataset_commands.add_parser("stats", help="查看 manifest 轻量统计")
    stats.add_argument("manifest", type=Path)
    stats.add_argument("--split")
    stats.add_argument("--probe-audio", action="store_true")
    stats.add_argument("--fail-fast", action="store_true")
    stats.add_argument("--json", action="store_true", dest="as_json")

    train = commands.add_parser("train", help="按 ExperimentConfig 训练模型")
    train.add_argument("config", type=Path)
    train.add_argument("--split", default="train")
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--resume", type=Path)
    train.add_argument("--json", action="store_true", dest="as_json")

    evaluate = commands.add_parser("evaluate", help="评估安全模型 artifact")
    evaluate.add_argument("artifact", type=Path)
    evaluate.add_argument("--manifest", type=Path)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--json", action="store_true", dest="as_json")

    predict = commands.add_parser("predict", help="使用 artifact 进行离线推理")
    predict.add_argument("artifact", type=Path)
    predict.add_argument("source", type=Path, help="音频文件、目录或 dataset.yaml")
    predict.add_argument("--split")
    predict.add_argument("--batch-size", type=int, default=16)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument("--keep-going", action="store_true")
    predict.add_argument("--no-recursive", action="store_true")
    predict.add_argument(
        "--window-aggregation",
        choices=("mean_logits", "mean_probabilities", "max_confidence"),
    )
    predict.add_argument("--json", action="store_true", dest="as_json")

    artifact = commands.add_parser("artifact", help="检查、校验或导出模型 artifact")
    artifact_commands = artifact.add_subparsers(dest="artifact_command", required=True)
    for name, help_text in (("inspect", "查看 artifact manifest"), ("verify", "校验 artifact 完整性")):
        operation = artifact_commands.add_parser(name, help=help_text)
        operation.add_argument("artifact", type=Path)
        operation.add_argument("--json", action="store_true", dest="as_json")
    export = artifact_commands.add_parser("export", help="从可信 checkpoint 导出 artifact")
    export.add_argument("--config", required=True, type=Path)
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--destination", required=True, type=Path)
    export.add_argument("--model-card", type=Path, help="模型卡 JSON/YAML 对象")
    export.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _add_importer_arguments(parser: argparse.ArgumentParser, *, destination: bool) -> None:
    parser.add_argument("--importer", required=True, choices=default_registry.names("importer"))
    parser.add_argument("--source", required=True, type=Path)
    if destination:
        parser.add_argument("--destination", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--params", help="Importer 参数 JSON 对象")
    group.add_argument("--params-file", type=Path, help="Importer 参数 JSON/YAML 文件")
    parser.add_argument("--json", action="store_true", dest="as_json")


def _load_params(inline: str | None, file: Path | None) -> dict[str, Any]:
    if inline is not None:
        try:
            value = json.loads(inline)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"--params 不是合法 JSON: {exc}") from exc
    elif file is not None:
        try:
            value = yaml.safe_load(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"无法读取参数文件 {file}: {exc}") from exc
    else:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError("Importer 参数必须是 JSON/YAML 对象")
    return dict(value)


def _emit(value: Any, *, as_json: bool, stream: TextIO) -> None:
    if as_json:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            rendered = json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else item
            stream.write(f"{key}: {rendered}\n")
    elif isinstance(value, list):
        for item in value:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        stream.write(f"{value}\n")


def _component_list(args, stream: TextIO) -> int:
    result: dict[str, list[dict[str, Any]]] = {}
    kinds = COMPONENT_NAMESPACES if args.kind == "all" else (args.kind,)
    for kind in kinds:
        if kind == "model":
            result["model"] = model_registry.descriptors()
        else:
            result[kind] = default_registry.json_safe_descriptors(kind)
    if args.kind == "all":
        result["model"] = model_registry.descriptors()
    _emit(result, as_json=args.as_json, stream=stream)
    return EXIT_OK


def _importer(args):
    return default_registry.create("importer", args.importer), _load_params(
        args.params, args.params_file
    )


def _dataset_scan(args, stream: TextIO) -> int:
    importer, params = _importer(args)
    preview = importer.scan(args.source, params)
    summary = preview.summary()
    _emit(summary, as_json=args.as_json, stream=stream)
    return EXIT_OK if preview.ok else EXIT_DATA


def _dataset_import(args, stream: TextIO) -> int:
    importer, params = _importer(args)
    manifest = importer.convert(args.source, args.destination, params)
    result = {
        "manifest": str(manifest.meta.yaml_path),
        "dataset_id": manifest.meta.dataset_id,
        "stats": manifest.stats(),
    }
    _emit(result, as_json=args.as_json, stream=stream)
    return EXIT_OK


def _split_stats(manifest: DatasetManifest, split: str | None) -> dict[str, Any]:
    if split is None:
        return manifest.stats()
    if split not in manifest.meta.splits:
        raise ManifestError(
            f"未知 split={split!r}，可用: {sorted(manifest.meta.splits)}",
            path=manifest.meta.yaml_path,
        )
    records = manifest.get_records(split)
    labels: dict[str, int] = {}
    for record in records:
        key = "unlabeled" if record.label is None else str(record.label)
        labels[key] = labels.get(key, 0) + 1
    return {
        "dataset_id": manifest.meta.dataset_id,
        "split": split,
        "total": len(records),
        "labels": labels,
        "num_classes": manifest.meta.num_classes or None,
    }


def _dataset_validate(args, stream: TextIO) -> int:
    manifest = DatasetManifest.load(args.manifest)
    records = manifest.get_records(args.split)
    missing = []
    if args.check_files:
        missing = [
            {"uid": record.uid, "path": str(manifest.resolve_audio_path(record))}
            for record in records
            if not manifest.resolve_audio_path(record).is_file()
        ]
    result = {
        "valid": not missing,
        "selected_count": len(records),
        "missing_files": missing,
        "stats": _split_stats(manifest, args.split),
    }
    _emit(result, as_json=args.as_json, stream=stream)
    return EXIT_OK if not missing else EXIT_DATA


def _dataset_stats(args, stream: TextIO) -> int:
    manifest = DatasetManifest.load(args.manifest)
    result = _split_stats(manifest, args.split)
    if args.probe_audio:
        profile = profile_manifest_audio(
            manifest, split=args.split, fail_fast=args.fail_fast
        )
        result["audio_profile"] = profile.to_dict()
        exit_code = EXIT_OK if profile.failed_records == 0 else EXIT_DATA
    else:
        exit_code = EXIT_OK
    _emit(result, as_json=args.as_json, stream=stream)
    return exit_code


def _validate_runtime_args(args) -> None:
    if hasattr(args, "batch_size") and args.batch_size < 1:
        raise ConfigurationError("--batch-size 必须 >= 1")
    if hasattr(args, "workers") and args.workers < 0:
        raise ConfigurationError("--workers 必须 >= 0")


def _load_mapping_file(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _load_params(None, path)


def _run_workflow(args, stream: TextIO) -> int:
    _validate_runtime_args(args)
    if args.command == "train":
        result = train_experiment(
            args.config, split=args.split, batch_size=args.batch_size,
            workers=args.workers, resume=args.resume,
        )
    elif args.command == "evaluate":
        result = evaluate_artifact(
            args.artifact, manifest_path=args.manifest, split=args.split,
            batch_size=args.batch_size, workers=args.workers,
            device=args.device, output=args.output,
        )
    elif args.command == "predict":
        result = predict_artifact(
            args.artifact, source=args.source, split=args.split,
            batch_size=args.batch_size, device=args.device, output=args.output,
            keep_going=args.keep_going, recursive=not args.no_recursive,
            window_aggregation=args.window_aggregation,
        )
    elif args.artifact_command in {"inspect", "verify"}:
        result = inspect_artifact(
            args.artifact, verify=args.artifact_command == "verify"
        )
    else:
        result = export_checkpoint_artifact(
            args.config, args.checkpoint, args.destination,
            model_card=_load_mapping_file(args.model_card),
        )
    _emit(result, as_json=args.as_json, stream=stream)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "components":
            return _component_list(args, sys.stdout)
        if args.command in {"train", "evaluate", "predict", "artifact"}:
            return _run_workflow(args, sys.stdout)
        handlers = {
            "scan": _dataset_scan,
            "import": _dataset_import,
            "validate": _dataset_validate,
            "stats": _dataset_stats,
        }
        return handlers[args.dataset_command](args, sys.stdout)
    except (ConfigurationError, ValidationError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"配置错误: {exc}\n")
        return EXIT_CONFIG
    except SERError as exc:
        sys.stderr.write(f"数据错误 [{exc.code}]: {exc}\n")
        return EXIT_DATA
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"错误: {exc}\n")
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())
