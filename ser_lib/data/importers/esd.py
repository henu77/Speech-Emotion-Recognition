"""Emotional Speech Dataset (ESD) directory importer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict

from ser_lib.data.importers.base import ImportIssue, ImportPreview
from ser_lib.data.importers.csemotions import (
    _automatic_speaker_splits,
    _validate_speaker_splits,
)
from ser_lib.data.manifest import DatasetManifest, ManifestMeta
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.types import AudioRecord


ESD_LABELS = {"Neutral": 0, "Happy": 1, "Angry": 2, "Sad": 3, "Surprise": 4}
ESD_ZH = {"Neutral": "中性", "Happy": "快乐", "Angry": "愤怒", "Sad": "悲伤", "Surprise": "惊讶"}


class EsdImportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    languages: list[Literal["zh", "en"]] = ["zh", "en"]
    encoding: str = "utf-8-sig"
    label_mapping: dict[str, int] | None = None
    speaker_splits: dict[str, list[str]] | None = None


def _language(speaker: str) -> str | None:
    if speaker.isdigit() and 1 <= int(speaker) <= 10:
        return "zh"
    if speaker.isdigit() and 11 <= int(speaker) <= 20:
        return "en"
    return None


def _speaker_splits(
    configured: dict[str, list[str]] | None, speakers: set[str]
) -> dict[str, list[str]]:
    if configured is not None:
        return _validate_speaker_splits(configured, speakers)
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for language in ("zh", "en"):
        members = {speaker for speaker in speakers if _language(speaker) == language}
        if not members:
            continue
        splits = _automatic_speaker_splits(members)
        for split in result:
            result[split].extend(splits[split])
    return result


def _transcripts(path: Path, encoding: str) -> tuple[dict[str, str], list[ImportIssue]]:
    result: dict[str, str] = {}
    issues: list[ImportIssue] = []
    if not path.is_file():
        return result, [ImportIssue(None, path, "transcript", "说话人文本文件不存在")]
    for line_number, raw in enumerate(path.read_text(encoding=encoding).splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.split("\t")
        if len(fields) < 2 or not fields[0].strip():
            issues.append(ImportIssue(line_number, path, "transcript", "文本行不是制表符分隔格式"))
            continue
        uid = fields[0].strip()
        if uid in result:
            issues.append(ImportIssue(line_number, path, "transcript", f"文本 UID 重复: {uid}"))
            continue
        result[uid] = fields[1].strip()
    return result, issues


class EsdImporter:
    descriptor = ComponentDescriptor(
        id="esd",
        display_name="ESD 导入",
        category="importer",
        description="解析 ESD 20 位中英语者目录，并生成语言分层、说话人独立划分。",
        config_schema=EsdImportConfig.model_json_schema(),
    )

    def scan(self, source: Path, config: Mapping[str, Any]) -> ImportPreview:
        cfg = EsdImportConfig(**dict(config))
        source = Path(source).resolve()
        if not source.is_dir():
            raise NotADirectoryError(f"ESD 目录不存在: {source}")
        preview = ImportPreview(importer_id=self.descriptor.id)
        mapping = dict(cfg.label_mapping or ESD_LABELS)
        selected_languages = set(cfg.languages)
        seen: set[str] = set()

        for speaker_dir in sorted(path for path in source.iterdir() if path.is_dir()):
            speaker = speaker_dir.name
            language = _language(speaker)
            if language is None:
                preview.warnings.append(f"忽略非 ESD 说话人目录: {speaker_dir}")
                continue
            if language not in selected_languages:
                continue
            texts, issues = _transcripts(speaker_dir / f"{speaker}.txt", cfg.encoding)
            preview.issues.extend(issues)
            for emotion, label in mapping.items():
                emotion_dir = speaker_dir / emotion
                if not emotion_dir.is_dir():
                    preview.issues.append(ImportIssue(None, emotion_dir, "directory", f"缺少情感目录: {emotion}"))
                    continue
                for audio in sorted(emotion_dir.glob("*.wav")):
                    uid = audio.stem
                    if uid in seen:
                        preview.issues.append(ImportIssue(None, audio, "uid", f"UID 重复: {uid}"))
                        continue
                    seen.add(uid)
                    if not uid.startswith(f"{speaker}_"):
                        preview.issues.append(ImportIssue(None, audio, "filename", "文件名说话人前缀与目录不一致"))
                        continue
                    text = texts.get(uid)
                    if text is None:
                        preview.issues.append(ImportIssue(None, audio, "transcript", "音频在文本文件中没有转写"))
                        continue
                    preview.records.append(AudioRecord(
                        uid=f"esd-{uid}",
                        audio_path=audio.relative_to(source),
                        label=label,
                        speaker_id=speaker,
                        metadata={"emotion_text": emotion.casefold(), "text": text, "language": language},
                    ))
        preview.label_mapping = mapping
        if not preview.records and not preview.issues:
            preview.issues.append(ImportIssue(None, source, "scan", "未发现符合条件的 ESD WAV"))
        return preview

    def convert(self, source: Path, destination: Path, config: Mapping[str, Any]) -> DatasetManifest:
        cfg = EsdImportConfig(**dict(config))
        source = Path(source).resolve()
        destination = Path(destination).resolve()
        preview = self.scan(source, config)
        if not preview.ok or not preview.records:
            issues = "; ".join(str(issue) for issue in preview.issues[:10])
            raise ValueError(f"ESD 扫描失败: {issues or '没有记录'}")
        speakers = {record.speaker_id for record in preview.records if record.speaker_id}
        split_speakers = _speaker_splits(cfg.speaker_splits, speakers)
        speaker_to_split = {speaker: split for split, members in split_speakers.items() for speaker in members}
        destination.mkdir(parents=True, exist_ok=True)
        meta = ManifestMeta(
            dataset_id="esd",
            root=source,
            yaml_path=destination / "dataset.yaml",
            splits={name: destination / f"{name}.jsonl" for name in split_speakers},
            labels={label: {"en": emotion.casefold(), "zh": ESD_ZH.get(emotion, emotion)} for emotion, label in preview.label_mapping.items()},
        )
        assignments = {record.uid: speaker_to_split[record.speaker_id] for record in preview.records if record.speaker_id}
        DatasetManifest(meta, preview.records, assignments).write()
        return DatasetManifest.load(destination / "dataset.yaml")


__all__ = ["ESD_LABELS", "ESD_ZH", "EsdImportConfig", "EsdImporter"]
