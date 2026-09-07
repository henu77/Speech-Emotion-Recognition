from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import pytest

from ser_lib.benchmark import (
    BenchmarkResult,
    compare_benchmarks,
    load_benchmark_result,
    run_benchmark,
    write_benchmark_result,
)
from ser_lib.data import profile_manifest_audio
from ser_lib.data.importers.ravdess import RavdessImporter


def _wav(path: Path, *, rate=8000, channels=1, frames=800):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(struct.pack("<h", 100) * channels * frames)


def test_ravdess_scan_parses_official_filename_fields(tmp_path: Path):
    source = tmp_path / "ravdess"
    speech = source / "Actor_01" / "03-01-05-02-01-02-01.wav"
    song = source / "Actor_02" / "03-02-03-01-02-01-02.wav"
    _wav(speech)
    _wav(song)

    preview = RavdessImporter().scan(source, {})
    assert preview.ok
    assert len(preview.records) == 1
    record = preview.records[0]
    assert record.label == 4
    assert record.speaker_id == "actor-01"
    assert record.metadata == {
        "emotion_text": "angry", "vocal_channel": "speech",
        "intensity": "strong", "statement": 1, "repetition": 2,
        "actor_gender": "male",
    }
    assert "CC BY-NC-SA 4.0" in preview.warnings[0]


def test_ravdess_filter_malformed_and_convert(tmp_path: Path):
    source = tmp_path / "source"
    _wav(source / "Actor_02" / "03-01-03-01-02-01-02.wav")
    _wav(source / "bad.wav")
    importer = RavdessImporter()
    preview = importer.scan(source, {})
    assert not preview.ok
    assert preview.issues[0].stage == "filename"
    with pytest.raises(ValueError, match="取消导入"):
        importer.convert(source, tmp_path / "bad-output", {})

    (source / "bad.wav").unlink()
    manifest = importer.convert(source, tmp_path / "standard", {})
    assert manifest.meta.dataset_id == "ravdess"
    assert manifest.meta.num_classes == 8
    assert manifest.resolve_audio_path(manifest.records[0]).is_file()


def test_ravdess_rejects_out_of_range_official_codes(tmp_path: Path):
    source = tmp_path / "source"
    _wav(source / "03-01-01-02-01-01-25.wav")
    preview = RavdessImporter().scan(source, {})
    assert not preview.ok
    assert "官方范围" in preview.issues[0].message


def test_manifest_audio_profile_reports_properties_and_failures(tmp_path: Path):
    _wav(tmp_path / "mono.wav", rate=8000, channels=1, frames=800)
    _wav(tmp_path / "stereo.wav", rate=16000, channels=2, frames=3200)
    (tmp_path / "records.jsonl").write_text(
        "\n".join((
            json.dumps({"uid": "mono", "audio_path": "mono.wav", "label": 0}),
            json.dumps({"uid": "stereo", "audio_path": "stereo.wav", "label": 1,
                        "start_ms": 50, "end_ms": 150}),
            json.dumps({"uid": "missing", "audio_path": "missing.wav", "label": 0}),
        )) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "dataset.yaml").write_text(
        "schema_version: 1\ndataset_id: profile\nroot: .\n"
        "splits: {test: records.jsonl}\n"
        "labels: {0: {en: neutral}, 1: {en: happy}}\n",
        encoding="utf-8",
    )
    profile = profile_manifest_audio(tmp_path / "dataset.yaml", split="test")
    assert profile.total_records == 3
    assert profile.probed_records == 2
    assert profile.failed_records == 1
    assert profile.total_duration_seconds == pytest.approx(0.2)
    assert profile.sample_rates == {"16000": 1, "8000": 1}
    assert profile.channels == {"1": 1, "2": 1}
    assert profile.failures[0].uid == "missing"


def test_repeatable_benchmark_result_round_trip_and_comparison(tmp_path: Path):
    counter = 0

    def operation():
        nonlocal counter
        counter += 1

    result = run_benchmark("noop", operation, iterations=4, warmup_iterations=2)
    assert counter == 6
    assert result.iterations == 4
    assert result.median_seconds >= 0
    path = write_benchmark_result(tmp_path / "result.json", result)
    assert load_benchmark_result(path) == result

    baseline = BenchmarkResult(
        "forward", 10, 2, 1.0, 2.0, 1.0, 100,
        environment={"machine": "same"},
    )
    current = BenchmarkResult(
        "forward", 10, 2, 1.05, 2.5, 1.0, 105,
        environment={"machine": "same"},
    )
    comparison = compare_benchmarks(current, baseline, threshold_percent=10)
    assert comparison.passed is False
    assert comparison.regressions["p95_seconds"] == pytest.approx(25)


def test_benchmark_refuses_cross_environment_comparison():
    first = BenchmarkResult("x", 1, 0, 1, 1, 1, 1, environment={"host": "a"})
    second = BenchmarkResult("x", 1, 0, 1, 1, 1, 1, environment={"host": "b"})
    with pytest.raises(ValueError, match="环境不同"):
        compare_benchmarks(first, second)
