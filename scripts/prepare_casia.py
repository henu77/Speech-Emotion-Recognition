"""Convert a CASIA directory into a speaker-independent standard manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from ser_lib.data.importers.casia import (
    CASIA_EMOTION_MAPPING,
    CASIA_EMOTION_ZH,
    CasiaImporter,
)
from ser_lib.data.manifest import DatasetManifest, ManifestMeta


def prepare(source: Path, destination: Path) -> DatasetManifest:
    source = source.resolve()
    destination = destination.resolve()
    preview = CasiaImporter().scan(source, {})
    if not preview.ok or not preview.records:
        raise ValueError(f"CASIA 扫描失败: {preview.summary()}")

    speakers = sorted(
        {record.speaker_id for record in preview.records if record.speaker_id},
        key=str.casefold,
    )
    if len(speakers) < 4:
        raise ValueError(f"说话人独立 train/val/test 至少需要 4 人，实际 {speakers}")
    train_count = max(2, round(len(speakers) * 0.5))
    val_count = max(1, round(len(speakers) * 0.25))
    if train_count + val_count >= len(speakers):
        val_count = 1
        train_count = len(speakers) - 2
    split_speakers = {
        "train": speakers[:train_count],
        "val": speakers[train_count:train_count + val_count],
        "test": speakers[train_count + val_count:],
    }
    speaker_to_split = {
        speaker: split_name
        for split_name, members in split_speakers.items()
        for speaker in members
    }
    record_splits = {
        record.uid: speaker_to_split[record.speaker_id]
        for record in preview.records
        if record.speaker_id is not None
    }
    destination.mkdir(parents=True, exist_ok=True)
    meta = ManifestMeta(
        dataset_id="casia-speaker-independent",
        root=source,
        yaml_path=destination / "dataset.yaml",
        splits={name: destination / f"{name}.jsonl" for name in split_speakers},
        labels={
            label: {"en": emotion, "zh": CASIA_EMOTION_ZH[emotion]}
            for emotion, label in CASIA_EMOTION_MAPPING.items()
        },
    )
    manifest = DatasetManifest(meta, preview.records, record_splits)
    manifest.write()
    loaded = DatasetManifest.load(destination / "dataset.yaml")
    print(f"manifest={loaded.meta.yaml_path}")
    print(f"speakers={split_speakers}")
    print(f"stats={loaded.stats()}")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/CASIA"))
    parser.add_argument("--destination", type=Path, default=Path("data/casia-standard"))
    args = parser.parse_args()
    prepare(args.source, args.destination)


if __name__ == "__main__":
    main()
