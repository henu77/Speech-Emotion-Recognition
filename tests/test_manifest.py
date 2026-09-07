from pathlib import Path

import pytest

from ser_lib.data.errors import ManifestError
from ser_lib.data.manifest import DatasetManifest, read_jsonl, write_jsonl
from ser_lib.data.types import AudioRecord


def test_jsonl_round_trip_preserves_unicode_and_segments(tmp_path: Path):
    path = tmp_path / "清单.jsonl"
    expected = AudioRecord(
        uid="样本-1",
        audio_path=Path("语音 文件/a.wav"),
        label=2,
        start_ms=10,
        end_ms=20,
        speaker_id="说话人",
        metadata={"language": "zh"},
    )
    write_jsonl([expected], path)
    assert read_jsonl(path) == [expected]


def test_read_jsonl_rejects_duplicate_uid(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"uid":"same","audio_path":"a.wav","label":0}\n'
        '{"uid":"same","audio_path":"b.wav","label":1}\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="UID 重复"):
        read_jsonl(path)


def test_dataset_manifest_rejects_uid_repeated_across_splits(tmp_path: Path):
    for split in ("train", "val"):
        (tmp_path / f"{split}.jsonl").write_text(
            '{"uid":"same","audio_path":"a.wav","label":0}\n',
            encoding="utf-8",
        )
    (tmp_path / "dataset.yaml").write_text(
        "schema_version: 1\n"
        "dataset_id: demo\n"
        "root: .\n"
        "splits:\n"
        "  train: train.jsonl\n"
        "  val: val.jsonl\n"
        "labels:\n"
        "  0: {en: neutral}\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="跨 split 重复"):
        DatasetManifest.load(tmp_path / "dataset.yaml")


def test_manifest_resolves_audio_relative_to_declared_root(tmp_path: Path):
    root = tmp_path / "音频 root"
    root.mkdir()
    (tmp_path / "train.jsonl").write_text(
        '{"uid":"a","audio_path":"nested/a.wav","label":0}\n',
        encoding="utf-8",
    )
    (tmp_path / "dataset.yaml").write_text(
        "schema_version: 1\n"
        "dataset_id: demo\n"
        "root: './音频 root'\n"
        "splits: {train: train.jsonl}\n"
        "labels: {0: {en: neutral}}\n",
        encoding="utf-8",
    )
    manifest = DatasetManifest.load(tmp_path / "dataset.yaml")
    assert manifest.resolve_audio_path(manifest.records[0]) == (root / "nested/a.wav").resolve()
