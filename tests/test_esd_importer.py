from __future__ import annotations

import wave
from pathlib import Path

import pytest

from ser_lib.data.importers.esd import ESD_LABELS, EsdImporter


def _fixture(root: Path) -> None:
    for speaker in ("0001", "0002", "0003", "0011", "0012", "0013"):
        speaker_dir = root / speaker
        lines = []
        for index, emotion in enumerate(ESD_LABELS, start=1):
            uid = f"{speaker}_{index:06d}"
            audio = speaker_dir / emotion / f"{uid}.wav"
            audio.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\0\0" * 160)
            lines.append(f"{uid}\ttext-{uid}\t{emotion}")
        (speaker_dir / f"{speaker}.txt").write_text("\n".join(lines), encoding="utf-8")


def test_esd_scan_and_language_stratified_convert(tmp_path: Path) -> None:
    source = tmp_path / "esd"
    _fixture(source)

    preview = EsdImporter().scan(source, {})
    manifest = EsdImporter().convert(source, tmp_path / "standard", {})

    assert preview.ok and len(preview.records) == 30
    assert {record.metadata["language"] for record in preview.records} == {"zh", "en"}
    assert manifest.meta.num_classes == 5
    for split in ("train", "val", "test"):
        records = manifest.get_records(split)
        assert {record.metadata["language"] for record in records} == {"zh", "en"}
    split_speakers = [
        {record.speaker_id for record in manifest.get_records(split)}
        for split in ("train", "val", "test")
    ]
    assert split_speakers[0].isdisjoint(split_speakers[1])
    assert split_speakers[0].isdisjoint(split_speakers[2])
    assert split_speakers[1].isdisjoint(split_speakers[2])


def test_esd_language_filter(tmp_path: Path) -> None:
    source = tmp_path / "esd"
    _fixture(source)

    preview = EsdImporter().scan(source, {"languages": ["zh"]})

    assert preview.ok and len(preview.records) == 15
    assert {record.speaker_id for record in preview.records} == {"0001", "0002", "0003"}


def test_esd_reports_missing_transcript_entry(tmp_path: Path) -> None:
    source = tmp_path / "esd"
    _fixture(source)
    transcript = source / "0001" / "0001.txt"
    transcript.write_text("", encoding="utf-8")

    preview = EsdImporter().scan(source, {})

    assert not preview.ok
    assert any(issue.stage == "transcript" for issue in preview.issues)
    with pytest.raises(ValueError, match="ESD 扫描失败"):
        EsdImporter().convert(source, tmp_path / "standard", {})
