"""CREMA-D AudioWAV filename and demographic metadata importer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from ser_lib.data.importers.base import ImportIssue, ImportPreview
from ser_lib.data.importers.csemotions import (
    _automatic_speaker_splits,
    _validate_speaker_splits,
)
from ser_lib.data.manifest import DatasetManifest, ManifestMeta
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.types import AudioRecord


CREMA_D_EMOTIONS = {
    "NEU": (0, "neutral"),
    "HAP": (1, "happy"),
    "ANG": (2, "angry"),
    "SAD": (3, "sad"),
    "FEA": (4, "fearful"),
    "DIS": (5, "disgust"),
}
CREMA_D_INTENSITIES = {"XX": "unspecified", "LO": "low", "MD": "medium", "HI": "high"}
CREMA_D_SENTENCES = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}
CREMA_D_ZH = {
    "neutral": "中性", "happy": "快乐", "angry": "愤怒", "sad": "悲伤",
    "fearful": "恐惧", "disgust": "厌恶",
}


class CremaDImportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_directory: str = "AudioWAV"
    demographics_file: str | None = "VideoDemographics.csv"
    encoding: str = "utf-8-sig"
    label_mapping: dict[str, int] | None = None
    speaker_splits: dict[str, list[str]] | None = None


def _read_demographics(path: Path, encoding: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"ActorID", "Age", "Sex", "Race", "Ethnicity"}
        if required - set(reader.fieldnames or []):
            raise ValueError(f"CREMA-D demographics 缺少列: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            actor = (row.get("ActorID") or "").strip()
            if not actor or actor in result:
                raise ValueError(f"CREMA-D demographics ActorID 为空或重复: {actor!r}")
            age_text = (row.get("Age") or "").strip()
            result[actor] = {
                "age": int(age_text) if age_text.isdigit() else None,
                "gender": (row.get("Sex") or "unknown").strip().casefold(),
                "race": (row.get("Race") or "unknown").strip(),
                "ethnicity": (row.get("Ethnicity") or "unknown").strip(),
            }
    return result


def _speaker_splits(
    configured: dict[str, list[str]] | None,
    speakers: set[str],
    demographics: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    if configured is not None:
        return _validate_speaker_splits(configured, speakers)
    groups: dict[str, set[str]] = {}
    for speaker in speakers:
        gender = str(demographics.get(speaker, {}).get("gender", "unknown"))
        groups.setdefault(gender, set()).add(speaker)
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for members in groups.values():
        if len(members) < 3:
            return _automatic_speaker_splits(speakers)
        split = _automatic_speaker_splits(members)
        for name in result:
            result[name].extend(split[name])
    return result


class CremaDImporter:
    descriptor = ComponentDescriptor(
        id="crema_d",
        display_name="CREMA-D 导入",
        category="importer",
        description="解析 AudioWAV 文件名和人口统计元数据，生成性别分层、说话人独立划分。",
        config_schema=CremaDImportConfig.model_json_schema(),
    )

    def scan(self, source: Path, config: Mapping[str, Any]) -> ImportPreview:
        cfg = CremaDImportConfig(**dict(config))
        source = Path(source).resolve()
        if not source.is_dir():
            raise NotADirectoryError(f"CREMA-D 目录不存在: {source}")
        audio_root = source / cfg.audio_directory
        if not audio_root.is_dir():
            raise NotADirectoryError(f"CREMA-D 音频目录不存在: {audio_root}")
        demographics_path = source / cfg.demographics_file if cfg.demographics_file else None
        demographics = _read_demographics(demographics_path, cfg.encoding) if demographics_path else {}
        mapping = dict(cfg.label_mapping or {code: label for code, (label, _) in CREMA_D_EMOTIONS.items()})
        if set(mapping) != set(CREMA_D_EMOTIONS):
            raise ValueError("CREMA-D label_mapping 必须覆盖且只能包含 NEU/HAP/ANG/SAD/FEA/DIS")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("CREMA-D label_mapping 的整数标签不能重复")
        preview = ImportPreview(importer_id=self.descriptor.id)
        preview.warnings.append("CREMA-D 使用 ODbL/DbCL；发布衍生数据或模型前请确认适用义务。")
        if demographics_path and not demographics:
            preview.warnings.append(f"未找到人口统计文件，将不写入相关元数据: {demographics_path}")

        for index, audio in enumerate(sorted(audio_root.glob("*.wav"))):
            fields = audio.stem.split("_")
            if len(fields) != 4:
                preview.issues.append(ImportIssue(index, audio, "filename", "文件名不是四段 CREMA-D 格式"))
                continue
            actor, sentence, emotion, intensity = fields
            if not (actor.isdigit() and sentence in CREMA_D_SENTENCES and emotion in mapping and intensity in CREMA_D_INTENSITIES):
                preview.issues.append(ImportIssue(index, audio, "filename", "文件名包含未知 actor/sentence/emotion/intensity"))
                continue
            if demographics and actor not in demographics:
                preview.issues.append(ImportIssue(index, audio, "demographics", f"找不到演员 {actor} 的人口统计记录"))
                continue
            _, emotion_name = CREMA_D_EMOTIONS[emotion]
            metadata: dict[str, Any] = {
                "language": "en",
                "text": CREMA_D_SENTENCES[sentence],
                "sentence_code": sentence,
                "emotion_text": emotion_name,
                "intensity": CREMA_D_INTENSITIES[intensity],
            }
            metadata.update(demographics.get(actor, {}))
            preview.records.append(AudioRecord(
                uid=f"crema-d-{audio.stem}",
                audio_path=audio.relative_to(source),
                label=mapping[emotion],
                speaker_id=actor,
                metadata=metadata,
            ))
        preview.label_mapping = mapping
        if not preview.records and not preview.issues:
            preview.issues.append(ImportIssue(None, audio_root, "scan", "未发现 CREMA-D WAV"))
        return preview

    def convert(self, source: Path, destination: Path, config: Mapping[str, Any]) -> DatasetManifest:
        cfg = CremaDImportConfig(**dict(config))
        source = Path(source).resolve()
        destination = Path(destination).resolve()
        preview = self.scan(source, config)
        if not preview.ok or not preview.records:
            issues = "; ".join(str(issue) for issue in preview.issues[:10])
            raise ValueError(f"CREMA-D 扫描失败: {issues or '没有记录'}")
        demographics_path = source / cfg.demographics_file if cfg.demographics_file else None
        demographics = _read_demographics(demographics_path, cfg.encoding) if demographics_path else {}
        speakers = {record.speaker_id for record in preview.records if record.speaker_id}
        split_speakers = _speaker_splits(cfg.speaker_splits, speakers, demographics)
        speaker_to_split = {speaker: split for split, members in split_speakers.items() for speaker in members}
        destination.mkdir(parents=True, exist_ok=True)
        labels = {
            label: {"en": CREMA_D_EMOTIONS[code][1], "zh": CREMA_D_ZH[CREMA_D_EMOTIONS[code][1]]}
            for code, label in preview.label_mapping.items()
        }
        meta = ManifestMeta(
            dataset_id="crema-d",
            root=source,
            yaml_path=destination / "dataset.yaml",
            splits={name: destination / f"{name}.jsonl" for name in split_speakers},
            labels=labels,
        )
        assignments = {record.uid: speaker_to_split[record.speaker_id] for record in preview.records if record.speaker_id}
        DatasetManifest(meta, preview.records, assignments).write()
        return DatasetManifest.load(destination / "dataset.yaml")


__all__ = ["CREMA_D_EMOTIONS", "CremaDImportConfig", "CremaDImporter"]
