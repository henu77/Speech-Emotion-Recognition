"""对标准 manifest 做不解码整段音频的可重复属性探测。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torchaudio

from ser_lib.core.events import CancellationCheck, EventCallback, ProgressEvent
from ser_lib.data.manifest import DatasetManifest


@dataclass(frozen=True, slots=True)
class AudioProbeFailure:
    uid: str
    path: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class DatasetAudioProfile:
    dataset_id: str
    split: str | None
    total_records: int
    probed_records: int
    failed_records: int
    total_duration_seconds: float
    min_duration_seconds: float | None
    max_duration_seconds: float | None
    mean_duration_seconds: float | None
    sample_rates: dict[str, int]
    channels: dict[str, int]
    failures: tuple[AudioProbeFailure, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_manifest_audio(
    manifest: DatasetManifest | Path | str,
    *,
    split: str | None = None,
    fail_fast: bool = False,
    event_callback: EventCallback | None = None,
    cancellation: CancellationCheck | None = None,
) -> DatasetAudioProfile:
    """使用音频 header 统计时长、采样率、声道和损坏/缺失文件。"""
    dataset = manifest if isinstance(manifest, DatasetManifest) else DatasetManifest.load(manifest)
    records = dataset.get_records(split)
    durations: list[float] = []
    sample_rates: dict[str, int] = {}
    channels: dict[str, int] = {}
    failures: list[AudioProbeFailure] = []
    for index, record in enumerate(records, start=1):
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        path = dataset.resolve_audio_path(record)
        try:
            info = torchaudio.info(str(path))
            rate = int(info.sample_rate)
            channel_count = int(info.num_channels)
            if rate <= 0 or int(info.num_frames) <= 0 or channel_count <= 0:
                raise ValueError("音频 header 包含非正采样率、帧数或声道数")
            start = (record.start_ms or 0) / 1000.0
            end = record.end_ms / 1000.0 if record.end_ms is not None else (
                int(info.num_frames) / rate
            )
            duration = max(0.0, min(end, int(info.num_frames) / rate) - start)
            if duration <= 0:
                raise ValueError("记录片段没有有效时长")
            durations.append(duration)
            sample_rates[str(rate)] = sample_rates.get(str(rate), 0) + 1
            channels[str(channel_count)] = channels.get(str(channel_count), 0) + 1
        except Exception as exc:
            if fail_fast:
                raise
            failures.append(AudioProbeFailure(
                uid=record.uid,
                path=str(path),
                error_type=type(exc).__name__,
                message=str(exc),
            ))
        if event_callback is not None:
            event_callback(ProgressEvent(
                stage="dataset_audio_profile", completed=index, total=len(records)
            ))
    total_duration = sum(durations)
    return DatasetAudioProfile(
        dataset_id=dataset.meta.dataset_id,
        split=split,
        total_records=len(records),
        probed_records=len(durations),
        failed_records=len(failures),
        total_duration_seconds=total_duration,
        min_duration_seconds=min(durations) if durations else None,
        max_duration_seconds=max(durations) if durations else None,
        mean_duration_seconds=total_duration / len(durations) if durations else None,
        sample_rates=dict(sorted(sample_rates.items())),
        channels=dict(sorted(channels.items())),
        failures=tuple(failures),
    )


__all__ = ["AudioProbeFailure", "DatasetAudioProfile", "profile_manifest_audio"]
