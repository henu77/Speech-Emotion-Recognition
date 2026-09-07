"""CASIA 导入器：把 data/casia_process.py 的扫描逻辑迁移为标准适配器。

CASIA 目录结构::

    <root>/<speaker>/<emotion>/<utt>.wav

标签映射与原脚本 ``data/casia_process.py`` 保持一致，不改变现有
CASIA 数据划分内容。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from ser_lib.data.importers.base import ImportIssue, ImportPreview
from ser_lib.data.importers.folder import DEFAULT_AUDIO_EXTENSIONS, _dataset_yaml
from ser_lib.data.manifest import DatasetManifest, write_jsonl
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.types import AudioRecord

# 与 data/casia_process.py:15 保持一致
CASIA_EMOTION_MAPPING: dict[str, int] = {
    "neutral": 0,
    "happy": 1,
    "angry": 2,
    "sad": 3,
    "surprise": 4,
    "fear": 5,
}

CASIA_EMOTION_ZH: dict[str, str] = {
    "neutral": "平静",
    "happy": "高兴",
    "angry": "愤怒",
    "sad": "悲伤",
    "surprise": "惊吓",
    "fear": "恐惧",
}


class CasiaImportConfig(BaseModel):
    """casia importer 参数。"""

    model_config = ConfigDict(extra="forbid")

    audio_extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_AUDIO_EXTENSIONS))
    # 覆盖默认 CASIA 情感映射（一般不需要）
    label_mapping: dict[str, int] | None = None


class CasiaImporter:
    """CASIA（说话人/情感两级目录）导入适配器。"""

    descriptor = ComponentDescriptor(
        id="casia",
        display_name="CASIA 导入",
        category="importer",
        description="按 <root>/<speaker>/<emotion>/<utt>.wav 结构扫描 CASIA 数据集，"
                    "标签映射与 data/casia_process.py 一致。",
        config_schema=CasiaImportConfig.model_json_schema(),
    )

    def scan(self, source: Path, config: Mapping[str, Any]) -> ImportPreview:
        cfg = CasiaImportConfig(**dict(config))
        source = Path(source).resolve()
        if not source.is_dir():
            raise NotADirectoryError(f"导入源不是目录: {source}")

        label_mapping = dict(cfg.label_mapping or CASIA_EMOTION_MAPPING)
        extensions = {ext.lower() for ext in cfg.audio_extensions}
        preview = ImportPreview(importer_id=self.descriptor.id)

        index = 0
        for speaker_dir in sorted(source.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for emotion_dir in sorted(speaker_dir.iterdir()):
                if not emotion_dir.is_dir():
                    continue
                emotion = emotion_dir.name.lower()
                label = label_mapping.get(emotion)
                if label is None:
                    preview.issues.append(
                        ImportIssue(
                            entry_index=None, path=emotion_dir, stage="scan",
                            message=f"发现未知情感目录 '{emotion}'，已跳过（与原脚本行为一致）",
                        )
                    )
                    continue
                for audio_file in sorted(emotion_dir.glob("*")):
                    if not audio_file.is_file() or audio_file.suffix.lower() not in extensions:
                        continue
                    preview.records.append(
                        AudioRecord(
                            uid=f"casia-{index:06d}",
                            audio_path=audio_file.relative_to(source),
                            label=label,
                            speaker_id=speaker_dir.name,
                            metadata={"emotion_text": emotion},
                        )
                    )
                    index += 1

        preview.label_mapping = label_mapping
        return preview

    def convert(
        self, source: Path, destination: Path, config: Mapping[str, Any]
    ) -> DatasetManifest:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        preview = self.scan(source, config)
        if not preview.records:
            issues = "; ".join(str(i) for i in preview.issues[:10])
            raise ValueError(f"未发现任何 CASIA 音频: {issues or '目录为空'}")

        write_jsonl(preview.records, destination / "manifest.jsonl")
        labels_yaml = {
            str(label_id): {"en": name, "zh": CASIA_EMOTION_ZH.get(name, name)}
            for name, label_id in sorted(preview.label_mapping.items(), key=lambda kv: kv[1])
        }
        (destination / "dataset.yaml").write_text(
            _dataset_yaml("casia", source.resolve(), {"default": "manifest.jsonl"}, labels_yaml),
            encoding="utf-8",
        )
        return DatasetManifest.load(destination / "dataset.yaml")


__all__ = [
    "CasiaImportConfig",
    "CasiaImporter",
    "CASIA_EMOTION_MAPPING",
    "CASIA_EMOTION_ZH",
]
