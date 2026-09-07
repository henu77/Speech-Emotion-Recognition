"""Importer 公共协议与预览结构（设计文档 §6.3）。

Importer 把外部数据格式转换为标准 manifest，必须区分 ``scan()``（预览，
不写盘）与 ``convert()``（确认后写入目标目录）。扫描按条目收集错误，
不因单个损坏文件终止全部扫描，也不把音频内容加载进内存。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from ser_lib.data.manifest import DatasetManifest
from ser_lib.data.registry import ComponentDescriptor
from ser_lib.data.types import AudioRecord


@dataclass(frozen=True)
class ImportIssue:
    """单条导入错误或告警，保留原因以便调用方汇总展示。"""

    entry_index: int | None
    path: Path | None
    stage: str
    message: str
    detail: str | None = None

    def __str__(self) -> str:  # pragma: no cover - 展示用途
        location = f" [{self.path}]" if self.path is not None else ""
        index = f" #{self.entry_index}" if self.entry_index is not None else ""
        detail = f" ({self.detail})" if self.detail else ""
        return f"{self.stage}{index}{location}: {self.message}{detail}"


@dataclass
class ImportPreview:
    """``scan()`` 的结果：预览记录、标签映射与错误汇总。"""

    importer_id: str
    records: list[AudioRecord] = field(default_factory=list)
    label_mapping: dict[str, int] = field(default_factory=dict)
    issues: list[ImportIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """是否没有错误级 issue。"""
        return not self.issues

    def summary(self) -> dict[str, Any]:
        """返回可序列化的预览摘要。"""
        return {
            "importer": self.importer_id,
            "num_records": len(self.records),
            "label_mapping": dict(self.label_mapping),
            "num_issues": len(self.issues),
            "issues": [
                {
                    "entry_index": issue.entry_index,
                    "path": str(issue.path) if issue.path else None,
                    "stage": issue.stage,
                    "message": issue.message,
                    "detail": issue.detail,
                }
                for issue in self.issues
            ],
            "warnings": list(self.warnings),
        }


class DatasetImporter(Protocol):
    """Importer 协议（设计文档 §6.3）。"""

    descriptor: ComponentDescriptor

    def scan(self, source: Path, config: Mapping[str, Any]) -> ImportPreview:
        """解析外部数据源并返回预览；不写入任何文件。"""
        ...

    def convert(
        self, source: Path, destination: Path, config: Mapping[str, Any]
    ) -> DatasetManifest:
        """把外部数据源转换为标准 manifest 并写入 destination 目录。"""
        ...
