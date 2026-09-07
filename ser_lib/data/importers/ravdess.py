"""RAVDESS 官方七段文件名格式适配器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict

from ser_lib.data.importers.base import ImportIssue, ImportPreview
from ser_lib.data.importers.folder import _dataset_yaml
from ser_lib.data.manifest import DatasetManifest, write_jsonl
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.types import AudioRecord


RAVDESS_EMOTIONS = {
    "01": (0, "neutral"),
    "02": (1, "calm"),
    "03": (2, "happy"),
    "04": (3, "sad"),
    "05": (4, "angry"),
    "06": (5, "fearful"),
    "07": (6, "disgust"),
    "08": (7, "surprised"),
}


class RavdessImportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocal_channel: Literal["speech", "song", "all"] = "speech"
    relative_paths: bool = True


class RavdessImporter:
    descriptor = ComponentDescriptor(
        id="ravdess",
        display_name="RAVDESS 导入",
        category="importer",
        description="解析 RAVDESS 官方 7 段文件名，仅引用本地 audio-only WAV。",
        config_schema=RavdessImportConfig.model_json_schema(),
    )

    def scan(self, source: Path, config: Mapping[str, Any]) -> ImportPreview:
        cfg = RavdessImportConfig(**dict(config))
        source = Path(source).resolve()
        if not source.is_dir():
            raise NotADirectoryError(f"导入源不是目录: {source}")
        preview = ImportPreview(importer_id=self.descriptor.id)
        preview.warnings.append(
            "RAVDESS 官方数据采用 CC BY-NC-SA 4.0；使用者须自行确认许可条件。"
        )
        channel_code = {"speech": "01", "song": "02", "all": None}[cfg.vocal_channel]
        for index, path in enumerate(sorted(source.rglob("*.wav"))):
            fields = path.stem.split("-")
            if len(fields) != 7 or any(len(field) != 2 or not field.isdigit() for field in fields):
                preview.issues.append(ImportIssue(
                    entry_index=index, path=path, stage="filename",
                    message="文件名不是 RAVDESS 七段两位数字格式",
                ))
                continue
            modality, channel, emotion, intensity, statement, repetition, actor = fields
            if modality != "03":
                continue
            if channel_code is not None and channel != channel_code:
                continue
            emotion_entry = RAVDESS_EMOTIONS.get(emotion)
            actor_number = int(actor)
            if emotion_entry is None:
                preview.issues.append(ImportIssue(
                    entry_index=index, path=path, stage="filename",
                    message="文件名包含超出 RAVDESS 官方范围的情感编码",
                ))
                continue
            invalid = (
                channel not in {"01", "02"}
                or intensity not in {"01", "02"}
                or statement not in {"01", "02"}
                or repetition not in {"01", "02"}
                or actor_number not in range(1, 25)
                or (emotion == "01" and intensity != "01")
            )
            if invalid:
                preview.issues.append(ImportIssue(
                    entry_index=index, path=path, stage="filename",
                    message="文件名包含超出 RAVDESS 官方范围的字段编码",
                ))
                continue
            label, emotion_name = emotion_entry
            audio_path = path.relative_to(source) if cfg.relative_paths else path
            preview.records.append(AudioRecord(
                uid=f"ravdess-{path.stem}",
                audio_path=audio_path,
                label=label,
                speaker_id=f"actor-{actor}",
                metadata={
                    "emotion_text": emotion_name,
                    "vocal_channel": "speech" if channel == "01" else "song",
                    "intensity": "normal" if intensity == "01" else "strong",
                    "statement": int(statement),
                    "repetition": int(repetition),
                    "actor_gender": "male" if actor_number % 2 else "female",
                },
            ))
        preview.label_mapping = {
            name: label for label, name in RAVDESS_EMOTIONS.values()
        }
        if not preview.records and not preview.issues:
            preview.issues.append(ImportIssue(
                entry_index=None, path=source, stage="scan",
                message="未发现符合筛选条件的 RAVDESS audio-only WAV",
            ))
        return preview

    def convert(
        self, source: Path, destination: Path, config: Mapping[str, Any]
    ) -> DatasetManifest:
        cfg = RavdessImportConfig(**dict(config))
        preview = self.scan(source, config)
        if not preview.ok or not preview.records:
            issues = "; ".join(str(item) for item in preview.issues[:10])
            raise ValueError(f"RAVDESS 扫描失败，取消导入: {issues}")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        write_jsonl(preview.records, destination / "manifest.jsonl")
        labels = {
            str(label): {"en": name}
            for label, name in RAVDESS_EMOTIONS.values()
        }
        root = Path(source).resolve() if cfg.relative_paths else "."
        (destination / "dataset.yaml").write_text(
            _dataset_yaml("ravdess", root, {"default": "manifest.jsonl"}, labels),
            encoding="utf-8",
        )
        return DatasetManifest.load(destination / "dataset.yaml")


__all__ = ["RavdessImporter", "RavdessImportConfig", "RAVDESS_EMOTIONS"]
