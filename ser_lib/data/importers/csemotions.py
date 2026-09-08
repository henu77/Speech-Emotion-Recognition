"""CSEMOTIONS metadata importer with speaker-independent splits."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from ser_lib.data.importers.base import ImportIssue, ImportPreview
from ser_lib.data.manifest import DatasetManifest, ManifestMeta
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.types import AudioRecord


CSEMOTIONS_LABELS = {
    "neutral": 0,
    "happy": 1,
    "angry": 2,
    "sad": 3,
    "surprise": 4,
    "fearful": 5,
    "playfulness": 6,
}
CSEMOTIONS_ZH = {
    "neutral": "中性", "happy": "快乐", "angry": "愤怒", "sad": "悲伤",
    "surprise": "惊讶", "fearful": "恐惧", "playfulness": "俏皮",
}


class CsemotionsImportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata_file: str = "csemotions_metadata.csv"
    audio_directory: str = "wav_data"
    encoding: str = "utf-8-sig"
    label_mapping: dict[str, int] | None = None
    speaker_splits: dict[str, list[str]] | None = None


def _gender(speaker: str) -> str:
    lowered = speaker.casefold()
    if lowered.startswith("female"):
        return "female"
    if lowered.startswith("male"):
        return "male"
    return "unknown"


def _automatic_speaker_splits(speakers: set[str]) -> dict[str, list[str]]:
    if len(speakers) < 3:
        raise ValueError("说话人独立 train/val/test 划分至少需要 3 位说话人")
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    groups: dict[str, list[str]] = {}
    for speaker in sorted(speakers, key=str.casefold):
        groups.setdefault(_gender(speaker), []).append(speaker)
    for members in groups.values():
        count = len(members)
        if count >= 3:
            val_count = max(1, round(count * 0.2))
            test_count = max(1, round(count * 0.2))
            train_count = count - val_count - test_count
            if train_count < 1:
                train_count, val_count, test_count = count - 2, 1, 1
            result["train"].extend(members[:train_count])
            result["val"].extend(members[train_count:train_count + val_count])
            result["test"].extend(members[train_count + val_count:])
        else:
            # Small/unknown demographic groups are distributed deterministically.
            for index, speaker in enumerate(members):
                result[("train", "val", "test")[index % 3]].append(speaker)
    if any(not result[name] for name in result):
        ordered = sorted(speakers, key=str.casefold)
        result = {"train": ordered[:-2], "val": [ordered[-2]], "test": [ordered[-1]]}
    return result


def _validate_speaker_splits(
    configured: dict[str, list[str]] | None, speakers: set[str]
) -> dict[str, list[str]]:
    if configured is None:
        return _automatic_speaker_splits(speakers)
    if set(configured) != {"train", "val", "test"}:
        raise ValueError("speaker_splits 必须且只能包含 train、val、test")
    flattened = [speaker for name in ("train", "val", "test") for speaker in configured[name]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("speaker_splits 中说话人重复")
    if set(flattened) != speakers:
        raise ValueError(
            f"speaker_splits 必须覆盖全部说话人；缺失={sorted(speakers - set(flattened))}，"
            f"未知={sorted(set(flattened) - speakers)}"
        )
    if any(not configured[name] for name in configured):
        raise ValueError("train、val、test 均至少需要一位说话人")
    return {name: list(configured[name]) for name in ("train", "val", "test")}


class CsemotionsImporter:
    descriptor = ComponentDescriptor(
        id="csemotions",
        display_name="CSEMOTIONS 导入",
        category="importer",
        description="解析官方 metadata CSV，并生成性别平衡、说话人独立的划分。",
        config_schema=CsemotionsImportConfig.model_json_schema(),
    )

    def scan(self, source: Path, config: Mapping[str, Any]) -> ImportPreview:
        cfg = CsemotionsImportConfig(**dict(config))
        source = Path(source).resolve()
        if not source.is_dir():
            raise NotADirectoryError(f"CSEMOTIONS 目录不存在: {source}")
        metadata_path = source / cfg.metadata_file
        audio_root = source / cfg.audio_directory
        if not metadata_path.is_file():
            raise FileNotFoundError(f"CSEMOTIONS metadata 不存在: {metadata_path}")
        if not audio_root.is_dir():
            raise NotADirectoryError(f"CSEMOTIONS 音频目录不存在: {audio_root}")

        preview = ImportPreview(importer_id=self.descriptor.id)
        mapping = dict(cfg.label_mapping or CSEMOTIONS_LABELS)
        seen_uids: set[str] = set()
        with metadata_path.open("r", encoding=cfg.encoding, newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"file_name", "text", "emotion", "speaker", "duration_sec"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                preview.issues.append(ImportIssue(
                    None, metadata_path, "header", f"metadata 缺少列: {sorted(missing)}"
                ))
                return preview
            for index, row in enumerate(reader, start=2):
                file_name = (row.get("file_name") or "").strip()
                emotion = (row.get("emotion") or "").strip().casefold()
                speaker = (row.get("speaker") or "").strip()
                audio_path = Path(cfg.audio_directory) / file_name
                resolved_audio = source / audio_path
                if not file_name or not speaker or emotion not in mapping:
                    preview.issues.append(ImportIssue(
                        index, metadata_path, "row",
                        f"非法 file_name/speaker/emotion: {file_name!r}/{speaker!r}/{emotion!r}",
                    ))
                    continue
                if not resolved_audio.is_file():
                    preview.issues.append(ImportIssue(
                        index, resolved_audio, "audio", "metadata 对应音频不存在"
                    ))
                    continue
                uid = f"csemotions-{Path(file_name).stem}"
                if uid in seen_uids:
                    preview.issues.append(ImportIssue(index, metadata_path, "uid", f"UID 重复: {uid}"))
                    continue
                seen_uids.add(uid)
                try:
                    duration = float(row["duration_sec"])
                    if duration <= 0:
                        raise ValueError
                except (TypeError, ValueError):
                    preview.issues.append(ImportIssue(
                        index, metadata_path, "duration", f"非法 duration_sec: {row['duration_sec']!r}"
                    ))
                    continue
                preview.records.append(AudioRecord(
                    uid=uid,
                    audio_path=audio_path,
                    label=mapping[emotion],
                    speaker_id=speaker,
                    metadata={
                        "emotion_text": emotion,
                        "text": row["text"],
                        "duration_sec": duration,
                        "language": "zh",
                        "gender": _gender(speaker),
                    },
                ))
        preview.label_mapping = mapping
        return preview

    def convert(
        self, source: Path, destination: Path, config: Mapping[str, Any]
    ) -> DatasetManifest:
        cfg = CsemotionsImportConfig(**dict(config))
        source = Path(source).resolve()
        destination = Path(destination).resolve()
        preview = self.scan(source, config)
        if not preview.ok or not preview.records:
            issues = "; ".join(str(issue) for issue in preview.issues[:10])
            raise ValueError(f"CSEMOTIONS 扫描失败: {issues or '没有记录'}")
        speakers = {record.speaker_id for record in preview.records if record.speaker_id}
        split_speakers = _validate_speaker_splits(cfg.speaker_splits, speakers)
        speaker_to_split = {
            speaker: split for split, members in split_speakers.items() for speaker in members
        }
        destination.mkdir(parents=True, exist_ok=True)
        meta = ManifestMeta(
            dataset_id="csemotions",
            root=source,
            yaml_path=destination / "dataset.yaml",
            splits={name: destination / f"{name}.jsonl" for name in split_speakers},
            labels={
                label: {"en": emotion, "zh": CSEMOTIONS_ZH.get(emotion, emotion)}
                for emotion, label in preview.label_mapping.items()
            },
        )
        record_splits = {
            record.uid: speaker_to_split[record.speaker_id]
            for record in preview.records if record.speaker_id is not None
        }
        DatasetManifest(meta, preview.records, record_splits).write()
        return DatasetManifest.load(destination / "dataset.yaml")


__all__ = [
    "CSEMOTIONS_LABELS", "CSEMOTIONS_ZH", "CsemotionsImportConfig",
    "CsemotionsImporter",
]
