from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from ser_lib.data.importers.emotiontalk import EmotionTalkImporter


def _record(root: Path, dialogue: str, speaker: str, index: int, emotion: str) -> None:
    stem = f"{dialogue}_01_01_{index:03d}"
    relative = Path(dialogue) / f"{dialogue}_01" / f"{dialogue}_01_01" / f"{stem}.wav"
    audio = root / "wav" / relative
    audio.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 160)
    annotation = root / "json" / relative.with_suffix(".json")
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text(json.dumps({
        "data": {"A": {"emotion": emotion, "Confidence_degree": "8"}},
        "speaker_id": speaker,
        "emotion_result": emotion,
        "content": f"text-{index}",
        "paragraphs": {"startTime": 1.0, "endTime": 2.0, "duration": 1.0},
        "sourceAttr": {"emo_cap": "description"},
        "file_path": relative.as_posix(),
    }, ensure_ascii=False), encoding="utf-8")


def _fixture(root: Path) -> None:
    emotions = ["neutral", "happy", "angry", "sad", "surprised", "fearful", "disgusted"]
    for index, speaker in enumerate(("01", "02", "03", "04", "05", "06"), start=1):
        _record(root, f"G{index:05d}", speaker, index, emotions[index - 1])


def test_emotiontalk_scan_and_speaker_independent_convert(tmp_path: Path) -> None:
    source = tmp_path / "emotiontalk"
    _fixture(source)

    preview = EmotionTalkImporter().scan(source, {})
    manifest = EmotionTalkImporter().convert(source, tmp_path / "standard", {})

    assert preview.ok and len(preview.records) == 6
    assert preview.records[0].metadata["annotator_votes"]
    assert preview.records[0].metadata["descriptions"]
    assert manifest.meta.num_classes == 7
    split_speakers = [
        {record.speaker_id for record in manifest.get_records(split)}
        for split in ("train", "val", "test")
    ]
    assert all(split_speakers)
    assert split_speakers[0].isdisjoint(split_speakers[1])
    assert split_speakers[0].isdisjoint(split_speakers[2])
    assert split_speakers[1].isdisjoint(split_speakers[2])


def test_emotiontalk_official_dialogue_split(tmp_path: Path) -> None:
    source = tmp_path / "emotiontalk"
    _fixture(source)
    _record(source, "G00012", "12", 12, "neutral")
    _record(source, "G00015", "15", 15, "neutral")

    manifest = EmotionTalkImporter().convert(
        source, tmp_path / "standard", {"split_strategy": "official_dialogue"}
    )

    assert {record.metadata["dialogue_id"] for record in manifest.get_records("val")} == {
        "G00001", "G00012"
    }
    assert {record.metadata["dialogue_id"] for record in manifest.get_records("test")} == {
        "G00003", "G00015"
    }


def test_emotiontalk_rejects_unsafe_audio_path(tmp_path: Path) -> None:
    source = tmp_path / "emotiontalk"
    _fixture(source)
    annotation = next((source / "json").rglob("*.json"))
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload["file_path"] = "../escape.wav"
    annotation.write_text(json.dumps(payload), encoding="utf-8")

    preview = EmotionTalkImporter().scan(source, {})

    assert not preview.ok
    assert any(issue.stage == "schema" for issue in preview.issues)
    with pytest.raises(ValueError, match="EmotionTalk 扫描失败"):
        EmotionTalkImporter().convert(source, tmp_path / "standard", {})
