"""BAAI EmotionTalk JSON/WAV importer."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict

from ser_lib.data.importers.base import ImportIssue, ImportPreview
from ser_lib.data.importers.csemotions import _validate_speaker_splits
from ser_lib.data.manifest import DatasetManifest, ManifestMeta
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.types import AudioRecord


EMOTIONTALK_LABELS = {
    "neutral": 0,
    "happy": 1,
    "angry": 2,
    "sad": 3,
    "surprised": 4,
    "fearful": 5,
    "disgusted": 6,
}
EMOTIONTALK_ZH = {
    "neutral": "中性", "happy": "快乐", "angry": "愤怒", "sad": "悲伤",
    "surprised": "惊讶", "fearful": "恐惧", "disgusted": "厌恶",
}
EMOTIONTALK_OFFICIAL_SPLITS = {
    "val": {"G00001", "G00012"},
    "test": {"G00003", "G00015"},
}


class EmotionTalkImportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    json_directory: str = "json"
    audio_directory: str = "wav"
    encoding: str = "utf-8"
    label_mapping: dict[str, int] | None = None
    split_strategy: Literal["speaker_independent", "official_dialogue"] = "speaker_independent"
    speaker_splits: dict[str, list[str]] | None = None


def _automatic_splits(speakers: set[str]) -> dict[str, list[str]]:
    if len(speakers) < 3:
        raise ValueError("EmotionTalk 说话人独立划分至少需要 3 位说话人")
    ordered = sorted(speakers)
    val_count = max(1, round(len(ordered) * 0.2))
    test_count = max(1, round(len(ordered) * 0.2))
    train_count = len(ordered) - val_count - test_count
    return {
        "train": ordered[:train_count],
        "val": ordered[train_count:train_count + val_count],
        "test": ordered[train_count + val_count:],
    }


def _safe_relative_audio(raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    posix = PurePosixPath(raw.strip())
    if posix.is_absolute() or ".." in posix.parts or posix.suffix.casefold() != ".wav":
        return None
    return Path(*posix.parts)


class EmotionTalkImporter:
    descriptor = ComponentDescriptor(
        id="emotiontalk",
        display_name="BAAI EmotionTalk 导入",
        category="importer",
        description="解析逐句 JSON/WAV、标注者置信度及描述，支持说话人独立或官方对话划分。",
        config_schema=EmotionTalkImportConfig.model_json_schema(),
    )

    def scan(self, source: Path, config: Mapping[str, Any]) -> ImportPreview:
        cfg = EmotionTalkImportConfig(**dict(config))
        source = Path(source).resolve()
        if not source.is_dir():
            raise NotADirectoryError(f"EmotionTalk 目录不存在: {source}")
        json_root = source / cfg.json_directory
        audio_root = source / cfg.audio_directory
        if not json_root.is_dir() or not audio_root.is_dir():
            raise NotADirectoryError("EmotionTalk 必须包含 json 和 wav 目录")
        mapping = dict(cfg.label_mapping or EMOTIONTALK_LABELS)
        if set(mapping) != set(EMOTIONTALK_LABELS) or len(set(mapping.values())) != len(mapping):
            raise ValueError("EmotionTalk label_mapping 必须完整覆盖七类情感且标签不能重复")
        preview = ImportPreview(importer_id=self.descriptor.id)
        if cfg.split_strategy == "official_dialogue":
            preview.warnings.append("官方对话划分会让部分 speaker_id 跨 split；严格说话人泛化实验请使用默认策略。")

        seen: set[str] = set()
        for index, annotation in enumerate(sorted(json_root.rglob("*.json"))):
            try:
                payload = json.loads(annotation.read_text(encoding=cfg.encoding))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                preview.issues.append(ImportIssue(index, annotation, "json", "无法读取 JSON", str(exc)))
                continue
            emotion = payload.get("emotion_result")
            speaker = payload.get("speaker_id")
            relative_audio = _safe_relative_audio(payload.get("file_path"))
            if emotion not in mapping or not isinstance(speaker, str) or not speaker or relative_audio is None:
                preview.issues.append(ImportIssue(index, annotation, "schema", "emotion_result/speaker_id/file_path 非法"))
                continue
            expected_json = json_root / relative_audio.with_suffix(".json")
            if annotation.resolve() != expected_json.resolve():
                preview.issues.append(ImportIssue(index, annotation, "path", "JSON 路径与 file_path 不对应"))
                continue
            audio = audio_root / relative_audio
            if not audio.is_file():
                preview.issues.append(ImportIssue(index, audio, "audio", "JSON 对应音频不存在"))
                continue
            uid = f"emotiontalk-{relative_audio.stem}"
            if uid in seen:
                preview.issues.append(ImportIssue(index, annotation, "uid", f"UID 重复: {uid}"))
                continue
            seen.add(uid)
            paragraphs = payload.get("paragraphs") if isinstance(payload.get("paragraphs"), dict) else {}
            metadata = {
                "language": "zh",
                "text": payload.get("content", ""),
                "emotion_text": emotion,
                "dialogue_id": relative_audio.parts[0],
                "turn_group": relative_audio.parent.name,
                "start_sec": paragraphs.get("startTime"),
                "end_sec": paragraphs.get("endTime"),
                "duration_sec": paragraphs.get("duration"),
                "annotator_votes": payload.get("data", {}),
                "descriptions": payload.get("sourceAttr", {}),
            }
            preview.records.append(AudioRecord(
                uid=uid,
                audio_path=Path(cfg.audio_directory) / relative_audio,
                label=mapping[emotion],
                speaker_id=speaker,
                metadata=metadata,
            ))
        preview.label_mapping = mapping
        if not preview.records and not preview.issues:
            preview.issues.append(ImportIssue(None, json_root, "scan", "未发现 EmotionTalk JSON"))
        return preview

    def convert(self, source: Path, destination: Path, config: Mapping[str, Any]) -> DatasetManifest:
        cfg = EmotionTalkImportConfig(**dict(config))
        source = Path(source).resolve()
        destination = Path(destination).resolve()
        preview = self.scan(source, config)
        if not preview.ok or not preview.records:
            issues = "; ".join(str(issue) for issue in preview.issues[:10])
            raise ValueError(f"EmotionTalk 扫描失败: {issues or '没有记录'}")
        if cfg.split_strategy == "official_dialogue":
            if cfg.speaker_splits is not None:
                raise ValueError("official_dialogue 策略不能同时配置 speaker_splits")
            assignments = {}
            for record in preview.records:
                dialogue = str(record.metadata["dialogue_id"])
                split = "val" if dialogue in EMOTIONTALK_OFFICIAL_SPLITS["val"] else (
                    "test" if dialogue in EMOTIONTALK_OFFICIAL_SPLITS["test"] else "train"
                )
                assignments[record.uid] = split
        else:
            speakers = {record.speaker_id for record in preview.records if record.speaker_id}
            splits = (_validate_speaker_splits(cfg.speaker_splits, speakers)
                      if cfg.speaker_splits is not None else _automatic_splits(speakers))
            speaker_to_split = {speaker: split for split, members in splits.items() for speaker in members}
            assignments = {record.uid: speaker_to_split[record.speaker_id] for record in preview.records if record.speaker_id}
        destination.mkdir(parents=True, exist_ok=True)
        meta = ManifestMeta(
            dataset_id="emotiontalk",
            root=source,
            yaml_path=destination / "dataset.yaml",
            splits={name: destination / f"{name}.jsonl" for name in ("train", "val", "test")},
            labels={label: {"en": emotion, "zh": EMOTIONTALK_ZH[emotion]} for emotion, label in preview.label_mapping.items()},
        )
        DatasetManifest(meta, preview.records, assignments).write()
        return DatasetManifest.load(destination / "dataset.yaml")


__all__ = ["EMOTIONTALK_LABELS", "EmotionTalkImportConfig", "EmotionTalkImporter"]
