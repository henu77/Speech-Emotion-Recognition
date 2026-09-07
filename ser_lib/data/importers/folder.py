"""目录扫描 importer：按目录结构与文件名规则导入（设计文档 §6.3）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ser_lib.data.importers.base import ImportIssue, ImportPreview
from ser_lib.data.manifest import DatasetManifest, load_meta, write_jsonl
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.types import AudioRecord

DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wv", ".aiff")


class FolderImportConfig(BaseModel):
    """folder importer 参数。"""

    model_config = ConfigDict(extra="forbid")

    audio_extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_AUDIO_EXTENSIONS))
    # 标签来源目录层级：0 = 音频文件所在目录（默认），1 = 上一级
    label_dir_level: int = Field(default=0, ge=0, le=2)
    # 说话人来源目录层级；None 表示不提取
    speaker_dir_level: int | None = Field(default=None, ge=0, le=3)
    # 标签名 -> id 映射；缺省时按发现到的标签名排序后从 0 分配
    label_mapping: dict[str, int] | None = None
    # 生成 uid 的前缀
    uid_prefix: str = Field(default="audio", min_length=1)
    # 写入 manifest 的音频路径是否相对 source 目录（引用模式，默认 true）
    relative_paths: bool = True

    @field_validator("audio_extensions")
    @classmethod
    def _normalize_ext(cls, v: list[str]) -> list[str]:
        exts = []
        for ext in v:
            ext = ext.lower()
            if not ext.startswith("."):
                ext = "." + ext
            exts.append(ext)
        return exts


class FolderImporter:
    """从目录结构导入：默认以音频所在目录名作为标签。"""

    descriptor = ComponentDescriptor(
        id="folder",
        display_name="目录导入",
        category="importer",
        description="扫描目录树，按目录名推导标签与说话人，生成标准 manifest。",
        config_schema=FolderImportConfig.model_json_schema(),
    )

    def scan(self, source: Path, config: Mapping[str, Any]) -> ImportPreview:
        cfg = FolderImportConfig(**dict(config))
        source = Path(source).resolve()
        if not source.is_dir():
            raise NotADirectoryError(f"导入源不是目录: {source}")

        preview = ImportPreview(importer_id=self.descriptor.id)

        # 第一遍：发现标签名
        found: list[tuple[Path, str, str | None]] = []
        label_names: set[str] = set()
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in set(cfg.audio_extensions):
                continue
            try:
                label_name = path.relative_to(source).parts[-1 - cfg.label_dir_level]
            except IndexError:
                preview.issues.append(
                    ImportIssue(
                        entry_index=None, path=path, stage="scan",
                        message=f"目录层级不足，无法提取标签 (label_dir_level={cfg.label_dir_level})",
                    )
                )
                continue
            speaker: str | None = None
            if cfg.speaker_dir_level is not None:
                try:
                    speaker = path.relative_to(source).parts[-1 - cfg.speaker_dir_level]
                except IndexError:
                    speaker = None
            found.append((path, label_name, speaker))
            label_names.add(label_name)

        if cfg.label_mapping is not None:
            unknown = label_names - set(cfg.label_mapping)
            if unknown:
                preview.issues.append(
                    ImportIssue(
                        entry_index=None, path=source, stage="scan",
                        message=f"以下标签目录未在 label_mapping 中声明: {sorted(unknown)}",
                    )
                )
            label_mapping = dict(cfg.label_mapping)
        else:
            label_mapping = {name: idx for idx, name in enumerate(sorted(label_names))}
            if not label_mapping:
                preview.issues.append(
                    ImportIssue(
                        entry_index=None, path=source, stage="scan",
                        message="未发现任何音频文件",
                    )
                )

        # 第二遍：生成记录（不读取音频内容）
        for index, (path, label_name, speaker) in enumerate(found):
            label = label_mapping.get(label_name)
            if label is None:
                preview.issues.append(
                    ImportIssue(
                        entry_index=index, path=path, stage="scan",
                        message=f"未知标签 '{label_name}'，跳过该条目",
                    )
                )
                continue
            audio_path = path.relative_to(source) if cfg.relative_paths else path.resolve()
            metadata: dict[str, Any] = {"label_name": label_name}
            preview.records.append(
                AudioRecord(
                    uid=f"{cfg.uid_prefix}-{index:06d}",
                    audio_path=Path(audio_path),
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
        cfg = FolderImportConfig(**dict(config))
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        preview = self.scan(source, config)
        if not preview.ok:
            issues = "; ".join(str(i) for i in preview.issues[:10])
            raise ValueError(f"扫描发现 {len(preview.issues)} 个错误，取消导入: {issues}")

        records = preview.records
        if not cfg.relative_paths:
            records = [
                AudioRecord(
                    uid=r.uid,
                    audio_path=Path(source) / r.audio_path,
                    label=r.label,
                    speaker_id=r.speaker_id,
                    metadata=dict(r.metadata),
                )
                for r in records
            ]
        write_jsonl(records, destination / "manifest.jsonl")

        labels_yaml = {
            str(label_id): {"en": name}
            for name, label_id in sorted(preview.label_mapping.items(), key=lambda kv: kv[1])
        }
        root = source.resolve() if cfg.relative_paths else "."
        (destination / "dataset.yaml").write_text(
            _dataset_yaml(self.descriptor.id, root, {"default": "manifest.jsonl"}, labels_yaml),
            encoding="utf-8",
        )
        return DatasetManifest.load(destination / "dataset.yaml")


def _dataset_yaml(dataset_id: str, root: Any, splits: Mapping[str, str],
                  labels: Mapping[str, Any]) -> str:
    """渲染最小 dataset.yaml 文本。"""
    import yaml

    doc = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "root": str(root),
        "splits": dict(splits),
        "labels": dict(labels),
    }
    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)


__all__ = ["FolderImportConfig", "FolderImporter", "DEFAULT_AUDIO_EXTENSIONS"]
