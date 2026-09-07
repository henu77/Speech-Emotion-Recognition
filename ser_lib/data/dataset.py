"""SERDataset：唯一的核心 Dataset（设计文档 §10）。

Dataset 只组装样本：``record → audio_loader.load(record) → pipeline(audio,
record)``。它不读取 YAML、不判断表示类型、不理解模型结构；组件在构造时
创建一次，不在 ``__getitem__`` 中重建 torchaudio transform；实例可被
DataLoader worker pickle，CPU tensor 由训练循环负责搬运到 GPU。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from torch.utils.data import Dataset

from ser_lib.data.audio import AudioLoader
from ser_lib.data.errors import SERDataError
from ser_lib.data.pipeline import SamplePipeline
from ser_lib.data.types import AudioRecord, SERSample


class SERDataset(Dataset[SERSample]):
    """核心 SER 数据集。

    Args:
        records: 音频记录序列（音频路径通常应已解析为绝对路径；相对路径
            需提供 ``base_dir``）。
        audio_loader: 共享的 AudioLoader。
        pipeline: 共享的 SamplePipeline。
        base_dir: 可选的相对路径解析基准目录（通常是 manifest root）。
        strict: 是否做运行时样本契约校验（§10.1；测试与开发默认开启）。
    """

    def __init__(
        self,
        records: Sequence[AudioRecord],
        audio_loader: AudioLoader,
        pipeline: SamplePipeline,
        *,
        base_dir: Path | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        if len(records) == 0:
            raise SERDataError(
                "SERDataset 需要至少一条记录；空数据集应在任务启动前失败"
            )
        self._records: tuple[AudioRecord, ...] = tuple(records)
        self.audio_loader = audio_loader
        self.pipeline = pipeline
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.pipeline.validate_contract = strict

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> SERSample:
        record = self._records[index]
        audio = self.audio_loader.load(record, base_dir=self.base_dir)
        return self.pipeline(audio, record)

    @property
    def records(self) -> tuple[AudioRecord, ...]:
        """数据集内的记录（只读视图）。"""
        return self._records

    def get_labels(self) -> list[int | None]:
        """返回全部标签列表，用于计算类别权重或平衡采样。"""
        return [record.label for record in self._records]
