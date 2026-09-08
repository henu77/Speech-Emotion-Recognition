from __future__ import annotations

import csv
import wave
from pathlib import Path

import pytest

from ser_lib.data.importers.csemotions import CsemotionsImporter


def _fixture(root: Path) -> None:
    audio = root / "wav_data"
    audio.mkdir(parents=True)
    rows = []
    speakers = ["female001", "female002", "female003", "male001", "male002", "male003"]
    emotions = ["neutral", "happy", "angry", "sad", "surprise", "fearful"]
    for index, (speaker, emotion) in enumerate(zip(speakers, emotions)):
        name = f"{index:05d}_{speaker}_{emotion}.wav"
        with wave.open(str(audio / name), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * 160)
        rows.append({
            "file_name": name,
            "text": f"text-{index}",
            "emotion": emotion,
            "speaker": speaker,
            "duration_sec": "0.01",
        })
    with (root / "csemotions_metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_csemotions_scan_and_speaker_independent_convert(tmp_path: Path):
    source = tmp_path / "source"
    _fixture(source)
    importer = CsemotionsImporter()

    preview = importer.scan(source, {})
    manifest = importer.convert(source, tmp_path / "standard", {})

    assert preview.ok and len(preview.records) == 6
    assert preview.records[0].metadata["language"] == "zh"
    assert manifest.meta.num_classes == 7
    assert manifest.stats()["splits"] == {"train": 2, "val": 2, "test": 2}
    split_speakers = {
        split: {record.speaker_id for record in manifest.get_records(split)}
        for split in ("train", "val", "test")
    }
    assert split_speakers["train"].isdisjoint(split_speakers["val"])
    assert split_speakers["train"].isdisjoint(split_speakers["test"])
    assert split_speakers["val"].isdisjoint(split_speakers["test"])
    assert all(manifest.resolve_audio_path(record).is_file() for record in manifest.records)


def test_csemotions_scan_reports_missing_audio(tmp_path: Path):
    source = tmp_path / "source"
    _fixture(source)
    missing = next((source / "wav_data").glob("*.wav"))
    missing.unlink()

    preview = CsemotionsImporter().scan(source, {})

    assert not preview.ok
    assert preview.issues[0].stage == "audio"
    with pytest.raises(ValueError, match="扫描失败"):
        CsemotionsImporter().convert(source, tmp_path / "standard", {})


def test_csemotions_custom_splits_must_cover_each_speaker(tmp_path: Path):
    source = tmp_path / "source"
    _fixture(source)
    with pytest.raises(ValueError, match="覆盖全部说话人"):
        CsemotionsImporter().convert(
            source,
            tmp_path / "standard",
            {"speaker_splits": {"train": ["female001"], "val": ["male001"],
                                 "test": ["male002"]}},
        )
