"""按核心包检查 coverage.py JSON，避免全局数字掩盖局部退化。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_THRESHOLDS = {
    "ser_lib/core/": 85.0,
    "ser_lib/artifacts/": 85.0,
    "ser_lib/engine/": 80.0,
    "ser_lib/inference/": 80.0,
    "ser_lib/models/": 80.0,
    "ser_lib/data/": 60.0,
    "ser_lib/cli/": 65.0,
}


def package_coverage(report: dict, prefix: str) -> float:
    normalized_prefix = prefix.replace("\\", "/")
    statements = covered = 0
    for name, details in report.get("files", {}).items():
        if name.replace("\\", "/").startswith(normalized_prefix):
            summary = details["summary"]
            statements += int(summary["num_statements"])
            covered += int(summary["covered_lines"])
    if statements == 0:
        raise ValueError(f"coverage 报告中没有匹配文件: {prefix}")
    return covered / statements * 100.0


def check_report(report: dict, thresholds: dict[str, float]) -> list[str]:
    failures = []
    for prefix, minimum in thresholds.items():
        actual = package_coverage(report, prefix)
        print(f"{prefix}: {actual:.2f}% (minimum {minimum:.2f}%)")
        if actual + 1e-9 < minimum:
            failures.append(f"{prefix} {actual:.2f}% < {minimum:.2f}%")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, nargs="?", default=Path("coverage.json"))
    args = parser.parse_args()
    raw = json.loads(args.report.read_text(encoding="utf-8"))
    failures = check_report(raw, DEFAULT_THRESHOLDS)
    if failures:
        print("coverage threshold failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
