"""JSONL importer：校验标准或近似标准 manifest（设计文档 §6.3 #3）。

近似标准（legacy）记录兼容规则：

- 缺少 ``uid`` 时用 ``{uid_prefix}-{index:06d}`` 确定性生成；
- ``start_time_ms`` / ``end_time_ms`` 映射为 ``start_ms`` / ``end_ms``；
- ``sample_rate`` / ``sr`` 映射为 ``sample_rate_hint``；
- 其余未识别字段原样保留进 ``metadata``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from ser_lib.data.errors import ManifestError
from ser_lib.data.importers.base import ImportIssue, ImportPreview
from ser_lib.data.manifest import DatasetManifest, parse_record, write_jsonl
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.types import AudioRecord

# 标准 manifest 顶层字段；其余字段进入 metadata（近似标准模式）
STANDARD_FIELDS = frozenset(
    {"uid", "audio_path", "label", "start_ms", "end_ms", "speaker_id",
     "sample_rate_hint", "metadata"}
)
LEGACY_ALIASES = {
    "start_time_ms": "start_ms",
    "end_time_ms": "end_ms",
    "sample_rate": "sample_rate_hint",
    "sr": "sample_rate_hint",
}


class JsonlImportConfig(BaseModel):
    """jsonl importer 参数。"""

    model_config = ConfigDict(extra="forbid")

    uid_prefix: str = Field(default="audio", min_length=1)
    # 相对音频路径的解析基准；None 表示相对 JSONL 所在目录
    root: Path | None = None


def normalize_raw_record(
    raw: Mapping[str, Any],
    *,
    index: int,
    uid_prefix: str,
) -> dict[str, Any]:
    """把标准或近似标准记录规范化为标准字段结构。

    未知字段保留进 metadata；不会丢失信息，也不会静默丢弃字段。
    """
    entry: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    for key, value in raw.items():
        if key in ("uid", "audio_path", "label", "speaker_id", "metadata"):
            if value is not None:
                entry[key] = value
        elif key in LEGACY_ALIASES:
            canonical = LEGACY_ALIASES[key]
            if value is not None and canonical not in entry:
                entry[canonical] = value
        elif key in ("start_ms", "end_ms", "sample_rate_hint"):
            if value is not None:
                entry[key] = value
        else:
            metadata[key] = value

    if entry.get("metadata"):
        metadata.update(dict(entry.pop("metadata")))
    if metadata:
        entry["metadata"] = metadata

    if not entry.get("uid"):
        entry["uid"] = f"{uid_prefix}-{index:06d}"
    return entry


def normalize_raw_records(
    raw_records: list[Mapping[str, Any]], *, uid_prefix: str
) -> list[AudioRecord]:
    """规范化整批记录并构造 AudioRecord（供兼容层复用）。"""
    records: list[AudioRecord] = []
    seen_uids: set[str] = set()
    for index, raw in enumerate(raw_records):
        entry = normalize_raw_record(raw, index=index, uid_prefix=uid_prefix)
        record = parse_record(entry, source=Path("<normalized>"), line_number=index + 1)
        if record.uid in seen_uids:
            raise ManifestError(f"UID 重复: '{record.uid}'（第 {index + 1} 条）", uid=record.uid)
        seen_uids.add(record.uid)
        records.append(record)
    return records


class JsonlImporter:
    """校验并导入标准/近似标准 JSONL manifest。"""

    descriptor = ComponentDescriptor(
        id="jsonl",
        display_name="JSONL 导入",
        category="importer",
        description="校验标准或近似标准的 JSONL manifest，规范化字段并生成标准 manifest。",
        config_schema=JsonlImportConfig.model_json_schema(),
    )

    def scan(self, source: Path, config: Mapping[str, Any]) -> ImportPreview:
        cfg = JsonlImportConfig(**dict(config))
        source = Path(source)
        preview = ImportPreview(importer_id=self.descriptor.id)

        raw_records: list[dict[str, Any]] = []
        with open(source, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    preview.issues.append(
                        ImportIssue(
                            entry_index=len(raw_records), path=source, stage="scan",
                            message=f"JSON 解析失败 (第 {line_number} 行): {exc.msg}",
                        )
                    )
                    raw_records.append({})
                    continue
                raw_records.append(raw)

        for index, raw in enumerate(raw_records):
            if not raw:
                continue
            entry = normalize_raw_record(raw, index=index, uid_prefix=cfg.uid_prefix)
            try:
                record = parse_record(entry, source=source, line_number=index + 1)
            except ManifestError as exc:
                preview.issues.append(
                    ImportIssue(
                        entry_index=index, path=source, stage="validate",
                        message=str(exc),
                    )
                )
                continue
            preview.records.append(record)

        return preview

    def convert(
        self, source: Path, destination: Path, config: Mapping[str, Any]
    ) -> DatasetManifest:
        cfg = JsonlImportConfig(**dict(config))
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        preview = self.scan(source, config)
        if not preview.ok:
            issues = "; ".join(str(i) for i in preview.issues[:10])
            raise ValueError(f"扫描发现 {len(preview.issues)} 个错误，取消导入: {issues}")

        root = cfg.root if cfg.root is not None else source.resolve().parent
        write_jsonl(preview.records, destination / "manifest.jsonl")
        (destination / "dataset.yaml").write_text(
            _simple_yaml(
                {
                    "schema_version": 1,
                    "dataset_id": self.descriptor.id,
                    "root": str(root),
                    "splits": {"default": "manifest.jsonl"},
                }
            ),
            encoding="utf-8",
        )
        return DatasetManifest.load(destination / "dataset.yaml")


def _simple_yaml(doc: Mapping[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(dict(doc), allow_unicode=True, sort_keys=False)


__all__ = [
    "JsonlImportConfig",
    "JsonlImporter",
    "normalize_raw_record",
    "normalize_raw_records",
    "STANDARD_FIELDS",
]
