"""桌面端本地长任务管理器。"""
from __future__ import annotations
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

TERMINAL_STATES = {"completed", "failed", "cancelled"}

@dataclass
class JobRecord:
    id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

class JobContext:
    def __init__(self, manager: "JobManager", job_id: str, cancel_event: threading.Event):
        self._manager, self.job_id, self._cancel_event = manager, job_id, cancel_event

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled("任务已取消")

    def report(self, progress: float, message: str = "") -> None:
        self._manager._report(self.job_id, progress, message)

class JobCancelled(Exception):
    pass

class JobManager:
    """线程安全任务状态机；训练任务后续可替换为独立进程执行器。"""
    def __init__(self, max_workers: int = 2) -> None:
        if max_workers < 1:
            raise ValueError("max_workers 必须 >= 1")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ser-job")
        self._lock = threading.RLock()
        self._records: dict[str, JobRecord] = {}
        self._cancellations: dict[str, threading.Event] = {}
        self._futures: dict[str, Future] = {}

    def submit(self, kind: str, function: Callable[[JobContext], Any]) -> str:
        job_id = uuid.uuid4().hex
        event = threading.Event()
        with self._lock:
            self._records[job_id] = JobRecord(id=job_id, kind=kind)
            self._cancellations[job_id] = event
            self._futures[job_id] = self._executor.submit(self._run, job_id, function, event)
        return job_id

    def _run(self, job_id: str, function: Callable[[JobContext], Any], event: threading.Event) -> None:
        self._update(job_id, status="running")
        context = JobContext(self, job_id, event)
        try:
            context.raise_if_cancelled()
            result = function(context)
            context.raise_if_cancelled()
        except JobCancelled:
            self._update(job_id, status="cancelled", message="任务已取消")
        except Exception as exc:
            self._update(job_id, status="failed", error=str(exc), message="任务失败")
        else:
            self._update(job_id, status="completed", progress=1.0, result=result, message="任务完成")

    def _report(self, job_id: str, progress: float, message: str) -> None:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress 必须位于 [0,1]")
        self._update(job_id, progress=float(progress), message=message)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            record = self._records[job_id]
            for key, value in values.items():
                setattr(record, key, value)
            record.updated_at = datetime.now(timezone.utc).isoformat()

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._records:
                raise KeyError(job_id)
            return self._records[job_id].to_dict()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [record.to_dict() for record in self._records.values()]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._records:
                raise KeyError(job_id)
            if self._records[job_id].status in TERMINAL_STATES:
                return False
            self._cancellations[job_id].set()
            self._futures[job_id].cancel()
            return True

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
