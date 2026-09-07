from __future__ import annotations

import pytest

from scripts.check_coverage import check_report, package_coverage


def _report(covered=8, statements=10):
    return {
        "files": {
            "ser_lib\\core\\config.py": {
                "summary": {
                    "covered_lines": covered,
                    "num_statements": statements,
                }
            }
        }
    }


def test_package_coverage_normalizes_windows_paths():
    assert package_coverage(_report(), "ser_lib/core/") == 80


def test_coverage_policy_reports_under_threshold(capsys):
    failures = check_report(_report(7), {"ser_lib/core/": 75})
    assert failures == ["ser_lib/core/ 70.00% < 75.00%"]
    assert "minimum 75.00%" in capsys.readouterr().out


def test_coverage_policy_rejects_missing_package():
    with pytest.raises(ValueError, match="没有匹配文件"):
        package_coverage(_report(), "ser_lib/missing/")
