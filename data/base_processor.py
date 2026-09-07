"""将外部 SER 语料转换为新数据流水线使用的标准工作区。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torchaudio
import yaml
from tqdm import tqdm

from ser_lib.data.config import (
    AudioSettings,
    BatchingConfig,
    ComponentConfig,
    DataConfig,
    FixedBatching,
)
from ser_lib.data.manifest import write_jsonl
from ser_lib.data.types import AudioRecord


class DatasetProcessor(ABC):
    """数据集适配器的离线生成基类。

    子类只实现原始目录解析与数据划分；基类负责音频校验、标准 JSONL、
    ``dataset.yaml``、数据报告和三份新流水线配置。
    """

    def __init__(
        self,
        raw_data_dir: str,
        output_meta_dir: str,
        dataset_name: str,
        emotion_mapping: dict[str, int],
        emotion_mapping_zh: dict[str, str] | None = None,
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir).resolve()
        self.output_meta_dir = Path(output_meta_dir).resolve()
        self.dataset_name = dataset_name
        self.emotion_mapping = dict(emotion_mapping)
        self.emotion_mapping_zh = dict(emotion_mapping_zh or {})

    @abstractmethod
    def _extract_samples(self) -> list[dict[str, Any]]:
        """返回 audio_path、label、speaker_id 和可选 metadata。"""

    @abstractmethod
    def _split_strategy(
        self, data_list: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """返回 train/val/test 数据划分。"""

    def process(self) -> None:
        self.output_meta_dir.mkdir(parents=True, exist_ok=True)
        raw_samples = sorted(
            self._extract_samples(), key=lambda item: str(item["audio_path"])
        )
        valid: list[dict[str, Any]] = []
        print(f"发现 {len(raw_samples)} 条候选记录，开始校验音频")
        for item in tqdm(raw_samples, desc="音频校验", unit="file"):
            path = Path(item["audio_path"]).resolve()
            try:
                info = torchaudio.info(str(path))
                if info.num_frames <= 0 or info.sample_rate <= 0:
                    raise ValueError("空音频或非法采样率")
            except Exception as exc:
                print(f"跳过无效音频: {path} | {exc}")
                continue
            normalized = dict(item)
            normalized["audio_path"] = path
            normalized["duration"] = round(info.num_frames / info.sample_rate, 3)
            valid.append(normalized)
        if not valid:
            raise ValueError("没有可用音频，未生成数据集")

        splits = self._split_strategy(valid)
        self._validate_splits(splits)
        records_by_split = self._write_manifests(splits)
        self._write_dataset_yaml()
        self._write_pipeline_configs()
        self._write_report(records_by_split)
        print(f"数据集已生成: {self.output_meta_dir}")

    def _validate_splits(self, splits: dict[str, list[dict[str, Any]]]) -> None:
        missing = {"train", "val", "test"} - set(splits)
        if missing:
            raise ValueError(f"数据划分缺少: {sorted(missing)}")
        seen: set[str] = set()
        for split, items in splits.items():
            for item in items:
                path = str(Path(item["audio_path"]).resolve())
                if path in seen:
                    raise ValueError(f"同一音频跨 split 重复: {path}")
                seen.add(path)

    def _write_manifests(
        self, splits: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[AudioRecord]]:
        result: dict[str, list[AudioRecord]] = {}
        counter = 0
        for split in ("train", "val", "test"):
            records = []
            for item in splits[split]:
                counter += 1
                path = Path(item["audio_path"]).resolve()
                try:
                    relative_path = path.relative_to(self.raw_data_dir)
                except ValueError:
                    relative_path = path
                metadata = dict(item.get("metadata") or {})
                for key in ("emotion_text", "duration"):
                    if key in item:
                        metadata[key] = item[key]
                records.append(
                    AudioRecord(
                        uid=f"{self.dataset_name.lower()}-{counter:08d}",
                        audio_path=relative_path,
                        label=int(item["label"]),
                        speaker_id=item.get("speaker_id"),
                        metadata=metadata,
                    )
                )
            write_jsonl(records, self.output_meta_dir / f"{split}.jsonl")
            result[split] = records
        return result

    @property
    def labels(self) -> dict[int, dict[str, str]]:
        return {
            label: {
                "en": emotion,
                "zh": self.emotion_mapping_zh.get(emotion, emotion),
            }
            for emotion, label in sorted(self.emotion_mapping.items(), key=lambda x: x[1])
        }

    def _write_dataset_yaml(self) -> None:
        document = {
            "schema_version": 1,
            "dataset_id": self.dataset_name.lower(),
            "root": self.raw_data_dir.as_posix(),
            "splits": {name: f"{name}.jsonl" for name in ("train", "val", "test")},
            "labels": self.labels,
        }
        self._write_yaml(self.output_meta_dir / "dataset.yaml", document)

    def _write_pipeline_configs(self) -> None:
        common = {
            "manifest": Path("dataset.yaml"),
            "dataset_id": self.dataset_name.lower(),
            "labels": self.labels,
            "audio": AudioSettings(target_sample_rate=16000),
        }
        configs = {
            "data_waveform.yaml": DataConfig(
                **common,
                representation=ComponentConfig(type="waveform"),
                batching=BatchingConfig(type="dynamic"),
            ),
            "data_log_mel.yaml": DataConfig(
                **common,
                representation=ComponentConfig(
                    type="log_mel",
                    params={
                        "sample_rate": 16000,
                        "n_fft": 1024,
                        "win_length": 1024,
                        "hop_length": 256,
                        "n_mels": 80,
                        "f_min": 0.0,
                        "f_max": 8000.0,
                        "power": 2.0,
                        "top_db": 80.0,
                    },
                ),
                batching=BatchingConfig(
                    type="fixed",
                    fixed=FixedBatching(max_lengths={"features": 300}),
                ),
            ),
            "data_acoustic.yaml": DataConfig(
                **common,
                representation=ComponentConfig(
                    type="acoustic_features",
                    params={
                        "sample_rate": 16000,
                        "features": ["f0", "rms", "zcr", "spectral_centroid"],
                        "hop_length": 256,
                    },
                ),
                batching=BatchingConfig(type="dynamic"),
            ),
        }
        for filename, config in configs.items():
            self._write_yaml(
                self.output_meta_dir / filename,
                config.model_dump(mode="json", exclude_none=True),
            )

    @staticmethod
    def _write_yaml(path: Path, document: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _write_report(self, splits: dict[str, list[AudioRecord]]) -> None:
        all_records = [record for records in splits.values() for record in records]
        durations = [float(record.metadata.get("duration", 0)) for record in all_records]
        label_counts: dict[int, int] = {}
        for record in all_records:
            label_counts[record.label] = label_counts.get(record.label, 0) + 1
        lines = [
            f"# {self.dataset_name} 数据集报告",
            "",
            f"- 有效音频：{len(all_records)}",
            f"- 总时长：{sum(durations) / 3600:.2f} 小时",
            f"- 最大时长：{max(durations):.3f} 秒",
            "",
            "## 数据划分",
            "",
        ]
        lines.extend(f"- {name}: {len(records)}" for name, records in splits.items())
        lines.extend(["", "## 标签分布", ""])
        lines.extend(
            f"- {self.labels[label]['en']} ({label}): {count}"
            for label, count in sorted(label_counts.items())
        )
        (self.output_meta_dir / "data_report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
