"""CSV importer：映射音频路径列、标签列与可选元数据列（设计文档 §6.3 #2）。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from ser_lib.data.importers.base import ImportIssue, ImportPreview
from ser_lib.data.manifest import DatasetManifest, write_jsonl
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.types import AudioRecord


class CsvImportConfig(BaseModel):
    """csv importer 参数。"""

    model_config = ConfigDict(extra="forbid")

    audio_path_column: str = Field(default="audio_path", min_length=1)
    label_column: str | None = Field(default="label")
    # 标签值是字符串时的名称->id 映射；缺省按发现值排序从 0 分配
    label_mapping: dict[str, int] | None = None
    speaker_column: str | None = None
    # 额外保留进 metadata 的列名
    metadata_columns: list[str] = Field(default_factory=list)
    uid_column: str | None = None
    uid_prefix: str = Field(default="audio", min_length=1)
    delimiter: str = Field(default=",", min_length=1, max_length=1)
    encoding: str = "utf-8-sig"
    # 相对音频路径的解析基准目录
    root: Path | None = None


class CsvImporter:
    """从 CSV/TSV 导入。"""

    descriptor = ComponentDescriptor(
        id="csv",
        display_name="CSV 导入",
        category="importer",
        description="映射 CSV 的音频路径列、标签列与可选元数据列，生成标准 manifest。",
        config_schema=CsvImportConfig.model_json_schema(),
    )

    def scan(self, source: Path, config: Mapping[str, Any]) -> ImportPreview:
        cfg = CsvImportConfig(**dict(config))
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(f"CSV 文件不存在: {source}")

        preview = ImportPreview(importer_id=self.descriptor.id)

        with open(source, "r", encoding=cfg.encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=cfg.delimiter)
            if reader.fieldnames is None:
                preview.issues.append(
                    ImportIssue(entry_index=None, path=source, stage="scan",
                                message="CSV 为空或没有表头")
                )
                return preview

            required = [cfg.audio_path_column]
            if cfg.label_column:
                required.append(cfg.label_column)
            missing = [c for c in required if c not in reader.fieldnames]
            if missing:
                preview.issues.append(
                    ImportIssue(
                        entry_index=None, path=source, stage="scan",
                        message=f"缺少必需列: {missing}，实际表头: {reader.fieldnames}",
                    )
                )
                return preview

            rows = list(reader)

        # 标签映射发现（仅当列存在时）
        label_names: set[str] = set()
        if cfg.label_column:
            for row in rows:
                value = (row.get(cfg.label_column) or "").strip()
                if value:
                    label_names.add(value)
        if cfg.label_mapping is not None:
            label_mapping = dict(cfg.label_mapping)
            unknown = label_names - set(label_mapping)
            if unknown:
                preview.issues.append(
                    ImportIssue(entry_index=None, path=source, stage="scan",
                                message=f"以下标签值未在 label_mapping 中声明: {sorted(unknown)}")
                )
        elif label_names and all(_is_int(name) for name in label_names):
            label_mapping = {name: int(name) for name in label_names}
        else:
            label_mapping = {name: idx for idx, name in enumerate(sorted(label_names))}

        for index, row in enumerate(rows):
            path_value = (row.get(cfg.audio_path_column) or "").strip()
            if not path_value:
                preview.issues.append(
                    ImportIssue(entry_index=index, path=source, stage="validate",
                                message="音频路径为空，跳过该行")
                )
                continue
            audio_path = Path(path_value)
            if not audio_path.is_absolute() and cfg.root is not None:
                audio_path = Path(cfg.root) / audio_path

            label: int | None = None
            if cfg.label_column:
                raw_label = (row.get(cfg.label_column) or "").strip()
                if raw_label:
                    if cfg.label_mapping is not None or not _is_int(raw_label):
                        label = label_mapping.get(raw_label)
                        if label is None:
                            preview.issues.append(
                                ImportIssue(entry_index=index, path=source, stage="validate",
                                            message=f"未知标签值 '{raw_label}'，跳过该行")
                            )
                            continue
                    else:
                        label = int(raw_label)

            uid: str | None = None
            if cfg.uid_column and (row.get(cfg.uid_column) or "").strip():
                uid = row[cfg.uid_column].strip()
            else:
                uid = f"{cfg.uid_prefix}-{index:06d}"

            speaker = None
            if cfg.speaker_column and (row.get(cfg.speaker_column) or "").strip():
                speaker = row[cfg.speaker_column].strip()

            metadata = {
                column: row[column]
                for column in cfg.metadata_columns
                if column in row and row[column] is not None
            }

            preview.records.append(
                AudioRecord(
                    uid=uid,
                    audio_path=audio_path,
                    label=label,
                    speaker_id=speaker,
                    metadata=metadata,
                )
            )

        preview.label_mapping = label_mapping
        return preview

    def convert(
        self, source: Path, destination: Path, config: Mapping[str, Any]
    ) -> DatasetManifest:
        cfg = CsvImportConfig(**dict(config))
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        preview = self.scan(source, config)
        if not preview.ok:
            issues = "; ".join(str(i) for i in preview.issues[:10])
            raise ValueError(f"扫描发现 {len(preview.issues)} 个错误，取消导入: {issues}")

        write_jsonl(preview.records, destination / "manifest.jsonl")
        root = cfg.root if cfg.root is not None else source.resolve().parent
        labels_yaml = {
            str(label_id): {"en": name}
            for name, label_id in sorted(preview.label_mapping.items(), key=lambda kv: kv[1])
        }
        (destination / "dataset.yaml").write_text(
            _simple_yaml(
                {
                    "schema_version": 1,
                    "dataset_id": self.descriptor.id,
                    "root": str(root),
                    "splits": {"default": "manifest.jsonl"},
                    "labels": labels_yaml,
                }
            ),
            encoding="utf-8",
        )
        return DatasetManifest.load(destination / "dataset.yaml")


def _is_int(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def _simple_yaml(doc: Mapping[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(dict(doc), allow_unicode=True, sort_keys=False)


__all__ = ["CsvImportConfig", "CsvImporter"]
