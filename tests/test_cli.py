from __future__ import annotations

import json
import importlib
import subprocess
import sys
from pathlib import Path

from ser_lib.cli.main import EXIT_CONFIG, EXIT_DATA, EXIT_OK, main

cli_main_module = importlib.import_module("ser_lib.cli.main")


def test_components_list_json(capsys):
    assert main(["components", "list", "--kind", "all", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "folder" in {item["id"] for item in payload["importer"]}
    assert "log_mel" in {item["id"] for item in payload["representation"]}
    assert "cnn_baseline" in {item["id"] for item in payload["model"]}
    assert "gru_baseline" in {item["id"] for item in payload["model"]}


def test_dataset_cli_scan_import_validate_and_stats(tmp_path: Path, capsys):
    source = tmp_path / "source"
    (source / "happy").mkdir(parents=True)
    (source / "sad").mkdir()
    (source / "happy" / "a.wav").write_bytes(b"fixture")
    (source / "sad" / "b.wav").write_bytes(b"fixture")
    params = json.dumps({"uid_prefix": "demo"})

    assert main([
        "dataset", "scan", "--importer", "folder", "--source", str(source),
        "--params", params, "--json",
    ]) == EXIT_OK
    scan = json.loads(capsys.readouterr().out)
    assert scan["num_records"] == 2
    assert scan["num_issues"] == 0

    destination = tmp_path / "standard"
    assert main([
        "dataset", "import", "--importer", "folder", "--source", str(source),
        "--destination", str(destination), "--params", params, "--json",
    ]) == EXIT_OK
    imported = json.loads(capsys.readouterr().out)
    manifest = Path(imported["manifest"])
    assert manifest.is_file()
    assert imported["stats"]["total"] == 2

    assert main([
        "dataset", "validate", str(manifest), "--split", "default",
        "--check-files", "--json",
    ]) == EXIT_OK
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True
    assert validated["selected_count"] == 2

    assert main([
        "dataset", "stats", str(manifest), "--split", "default", "--json",
    ]) == EXIT_OK
    stats = json.loads(capsys.readouterr().out)
    assert stats["total"] == 2
    assert stats["labels"] == {"0": 1, "1": 1}


def test_dataset_validate_reports_missing_files(tmp_path: Path, capsys):
    (tmp_path / "records.jsonl").write_text(
        '{"uid":"missing","audio_path":"missing.wav","label":0}\n', encoding="utf-8"
    )
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text(
        """schema_version: 1
dataset_id: missing
root: .
splits: {test: records.jsonl}
labels:
  0: {en: neutral}
""",
        encoding="utf-8",
    )
    assert main([
        "dataset", "validate", str(manifest), "--check-files", "--json"
    ]) == EXIT_DATA
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert payload["missing_files"][0]["uid"] == "missing"


def test_cli_configuration_error_has_stable_exit_code(tmp_path: Path, capsys):
    assert main([
        "dataset", "scan", "--importer", "folder", "--source", str(tmp_path),
        "--params", "not-json",
    ]) == EXIT_CONFIG
    assert "配置错误" in capsys.readouterr().err


def test_module_entrypoint_reports_version():
    completed = subprocess.run(
        [sys.executable, "-m", "ser_lib.cli", "--version"],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0
    assert "ser-lib" in completed.stdout


def test_train_cli_dispatches_validated_options(monkeypatch, tmp_path: Path, capsys):
    captured = {}

    def fake(config, **kwargs):
        captured.update(config=config, **kwargs)
        return {"output_dir": "runs/demo", "history": []}

    monkeypatch.setattr(cli_main_module, "train_experiment", fake)
    assert main([
        "train", str(tmp_path / "experiment.yaml"), "--split", "training",
        "--batch-size", "8", "--workers", "2", "--json",
    ]) == EXIT_OK
    assert captured["split"] == "training"
    assert captured["batch_size"] == 8
    assert json.loads(capsys.readouterr().out)["output_dir"] == "runs/demo"


def test_predict_cli_dispatches_source_policy(monkeypatch, tmp_path: Path, capsys):
    captured = {}

    def fake(artifact, **kwargs):
        captured.update(artifact=artifact, **kwargs)
        return {"total": 2, "succeeded": 1, "failed": 1}

    monkeypatch.setattr(cli_main_module, "predict_artifact", fake)
    assert main([
        "predict", str(tmp_path / "model"), str(tmp_path / "audio"),
        "--output", str(tmp_path / "predictions.jsonl"), "--batch-size", "4",
        "--keep-going", "--no-recursive", "--json",
    ]) == EXIT_OK
    assert captured["keep_going"] is True
    assert captured["recursive"] is False
    assert json.loads(capsys.readouterr().out)["failed"] == 1


def test_artifact_export_loads_model_card(monkeypatch, tmp_path: Path, capsys):
    card = tmp_path / "card.yaml"
    card.write_text("description: demo model\nlicense: MIT\n", encoding="utf-8")
    captured = {}

    def fake(config, checkpoint, destination, **kwargs):
        captured.update(kwargs)
        return {"artifact": str(destination)}

    monkeypatch.setattr(cli_main_module, "export_checkpoint_artifact", fake)
    assert main([
        "artifact", "export", "--config", str(tmp_path / "experiment.yaml"),
        "--checkpoint", str(tmp_path / "epoch.pt"),
        "--destination", str(tmp_path / "artifact"),
        "--model-card", str(card), "--json",
    ]) == EXIT_OK
    assert captured["model_card"]["license"] == "MIT"
    assert json.loads(capsys.readouterr().out)["artifact"].endswith("artifact")


def test_cli_rejects_invalid_worker_and_batch_counts(capsys):
    assert main(["train", "config.yaml", "--batch-size", "0"]) == EXIT_CONFIG
    assert "batch-size" in capsys.readouterr().err
    assert main(["train", "config.yaml", "--workers", "-1"]) == EXIT_CONFIG
    assert "workers" in capsys.readouterr().err
