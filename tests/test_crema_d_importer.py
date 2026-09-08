from __future__ import annotations

import csv
import wave
from pathlib import Path

import pytest

from ser_lib.data.importers.crema_d import CREMA_D_EMOTIONS, CremaDImporter


def _fixture(root: Path) -> None:
    audio_root = root / "AudioWAV"
    audio_root.mkdir(parents=True)
    actors = [("1001", "Male"), ("1002", "Male"), ("1003", "Male"),
              ("1004", "Female"), ("1005", "Female"), ("1006", "Female")]
    with (root / "VideoDemographics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ActorID", "Age", "Sex", "Race", "Ethnicity"])
        writer.writeheader()
        for actor, gender in actors:
            writer.writerow({"ActorID": actor, "Age": 30, "Sex": gender,
                             "Race": "Unknown", "Ethnicity": "Unknown"})
    for actor, _ in actors:
        for emotion in CREMA_D_EMOTIONS:
            audio = audio_root / f"{actor}_IEO_{emotion}_XX.wav"
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\0\0" * 160)


def test_crema_d_scan_and_demographic_stratified_convert(tmp_path: Path) -> None:
    source = tmp_path / "crema-d"
    _fixture(source)

    preview = CremaDImporter().scan(source, {})
    manifest = CremaDImporter().convert(source, tmp_path / "standard", {})

    assert preview.ok and len(preview.records) == 36
    assert preview.records[0].metadata["text"] == "It's eleven o'clock."
    assert manifest.meta.num_classes == 6
    for split in ("train", "val", "test"):
        records = manifest.get_records(split)
        assert {record.metadata["gender"] for record in records} == {"male", "female"}
    split_speakers = [
        {record.speaker_id for record in manifest.get_records(split)}
        for split in ("train", "val", "test")
    ]
    assert split_speakers[0].isdisjoint(split_speakers[1])
    assert split_speakers[0].isdisjoint(split_speakers[2])
    assert split_speakers[1].isdisjoint(split_speakers[2])


def test_crema_d_can_scan_without_demographics(tmp_path: Path) -> None:
    source = tmp_path / "crema-d"
    _fixture(source)
    (source / "VideoDemographics.csv").unlink()

    preview = CremaDImporter().scan(source, {})

    assert preview.ok and len(preview.records) == 36
    assert any("人口统计" in warning for warning in preview.warnings)
    assert "gender" not in preview.records[0].metadata


def test_crema_d_rejects_invalid_filename(tmp_path: Path) -> None:
    source = tmp_path / "crema-d"
    _fixture(source)
    invalid = source / "AudioWAV" / "bad.wav"
    invalid.write_bytes(b"not-a-wave")

    preview = CremaDImporter().scan(source, {})

    assert not preview.ok
    assert any(issue.path == invalid and issue.stage == "filename" for issue in preview.issues)
    with pytest.raises(ValueError, match="CREMA-D 扫描失败"):
        CremaDImporter().convert(source, tmp_path / "standard", {})
