import math
import struct
import wave
from pathlib import Path

from data.base_processor import DatasetProcessor
from ser_lib.data.config import load_data_config
from ser_lib.data.manifest import DatasetManifest


def _wav(path: Path):
    frames = bytearray()
    for index in range(800):
        frames.extend(struct.pack("<h", int(4000 * math.sin(2 * math.pi * 220 * index / 8000))))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(frames)


class DemoProcessor(DatasetProcessor):
    def _extract_samples(self):
        return [
            {"audio_path": path, "label": index % 2, "speaker_id": f"s{index}",
             "emotion_text": "neutral" if index % 2 == 0 else "happy"}
            for index, path in enumerate(sorted(self.raw_data_dir.glob("*.wav")))
        ]

    def _split_strategy(self, records):
        return {"train": records[:2], "val": records[2:3], "test": records[3:]}


def test_processor_writes_only_new_pipeline_formats(tmp_path: Path):
    source = tmp_path / "音频"
    output = tmp_path / "workspace"
    source.mkdir()
    for index in range(4):
        _wav(source / f"{index}.wav")
    DemoProcessor(
        str(source), str(output), "DEMO", {"neutral": 0, "happy": 1},
        {"neutral": "平静", "happy": "高兴"},
    ).process()

    manifest = DatasetManifest.load(output / "dataset.yaml")
    assert len(manifest.records) == 4
    assert len({record.uid for record in manifest.records}) == 4
    assert all(not record.audio_path.is_absolute() for record in manifest.records)
    for filename in ("data_waveform.yaml", "data_log_mel.yaml", "data_acoustic.yaml"):
        config = load_data_config(output / filename)
        assert config.manifest == (output / "dataset.yaml").resolve()
    assert not any(
        name in {path.name for path in output.glob("*.yaml")}
        for name in ("waveform_dataset.yaml", "spectrogram_dataset.yaml", "feature_dataset.yaml")
    )
