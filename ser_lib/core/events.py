"""与传输方式无关的进度、指标、日志和取消协议。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from ser_lib.core.exceptions import OperationCancelled


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: str
    completed: int
    total: int | None = None
    message: str = ""
    timestamp: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("ProgressEvent.stage 不能为空")
        if self.completed < 0 or (self.total is not None and self.total < 0):
            raise ValueError("completed/total 不能为负数")
        if self.total is not None and self.completed > self.total:
            raise ValueError("completed 不能大于 total")

    @property
    def fraction(self) -> float | None:
        if self.total is None or self.total == 0:
            return None
        return self.completed / self.total


@dataclass(frozen=True, slots=True)
class MetricEvent:
    name: str
    value: float
    step: int | None = None
    split: str | None = None
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class LogEvent:
    level: str
    message: str
    stage: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utc_now)


LibraryEvent = ProgressEvent | MetricEvent | LogEvent
EventCallback = Callable[[LibraryEvent], None]


class CancellationCheck(Protocol):
    """长操作只依赖此协议，不依赖具体调度器。"""

    @property
    def is_cancelled(self) -> bool: ...

    def raise_if_cancelled(self) -> None: ...


class CancellationToken:
    """可在线程间安全共享的协作式取消令牌。"""

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise OperationCancelled("操作已取消")
