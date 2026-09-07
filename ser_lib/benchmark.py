"""可序列化的微基准结果与回归比较工具。"""

from __future__ import annotations

import json
import platform
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    iterations: int
    warmup_iterations: int
    median_seconds: float
    p95_seconds: float
    operations_per_second: float
    peak_memory_bytes: int
    environment: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    passed: bool
    regressions: dict[str, float]
    threshold_percent: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_benchmark(
    name: str,
    operation: Callable[[], Any],
    *,
    iterations: int = 20,
    warmup_iterations: int = 3,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """在同一进程执行同步操作，报告中位数、p95 和 Python 峰值内存。"""
    if not name:
        raise ValueError("benchmark name 不能为空")
    if iterations < 1 or warmup_iterations < 0:
        raise ValueError("iterations 必须 >= 1，warmup_iterations 必须 >= 0")
    for _ in range(warmup_iterations):
        operation()
    timings = []
    tracemalloc.start()
    try:
        for _ in range(iterations):
            started = time.perf_counter()
            operation()
            timings.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    ordered = sorted(timings)
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.999) - 1))
    total = sum(timings)
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        median_seconds=statistics.median(timings),
        p95_seconds=ordered[p95_index],
        operations_per_second=iterations / total if total > 0 else 0.0,
        peak_memory_bytes=peak,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        metadata=dict(metadata or {}),
    )


def compare_benchmarks(
    current: BenchmarkResult,
    baseline: BenchmarkResult,
    *,
    threshold_percent: float = 10.0,
) -> BenchmarkComparison:
    """比较越低越好的延迟/内存指标，环境不同则拒绝误判为可比结果。"""
    if threshold_percent < 0:
        raise ValueError("threshold_percent 必须 >= 0")
    if current.name != baseline.name:
        raise ValueError("只能比较同名 benchmark")
    if current.environment != baseline.environment:
        raise ValueError("benchmark 运行环境不同，不能直接比较")
    regressions = {}
    for field_name in ("median_seconds", "p95_seconds", "peak_memory_bytes"):
        old = float(getattr(baseline, field_name))
        new = float(getattr(current, field_name))
        regressions[field_name] = 0.0 if old == 0 and new == 0 else (
            float("inf") if old == 0 else (new - old) / old * 100.0
        )
    return BenchmarkComparison(
        passed=all(value <= threshold_percent for value in regressions.values()),
        regressions=regressions,
        threshold_percent=threshold_percent,
    )


def write_benchmark_result(path: Path | str, result: BenchmarkResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_benchmark_result(path: Path | str) -> BenchmarkResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark 文件顶层必须是对象")
    return BenchmarkResult(**raw)


__all__ = [
    "BenchmarkResult", "BenchmarkComparison", "run_benchmark",
    "compare_benchmarks", "write_benchmark_result", "load_benchmark_result",
]
