from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest
from pydantic import Field, ValidationError

from ser_lib.core import (
    CancellationToken,
    ConfigurationError,
    OperationCancelled,
    ProgressEvent,
    SERError,
    StrictConfig,
    configure_library_logging,
    get_logger,
    load_versioned_config,
    load_yaml_mapping,
    require_schema_version,
    resolve_config_path,
)
from ser_lib.data.errors import ManifestError, SERDataError


class ExampleConfig(StrictConfig):
    schema_version: int = 1
    name: str = Field(min_length=1)


def test_strict_config_rejects_unknown_fields_and_is_frozen():
    with pytest.raises(ValidationError):
        ExampleConfig(schema_version=1, name="demo", typo=True)
    config = ExampleConfig(schema_version=1, name="demo")
    with pytest.raises(ValidationError):
        config.name = "changed"


def test_resolve_config_path_does_not_depend_on_cwd(tmp_path: Path):
    assert resolve_config_path("nested/config.yaml", base_dir=tmp_path) == (
        tmp_path / "nested/config.yaml"
    ).resolve()


def test_load_yaml_mapping_and_versioned_config(tmp_path: Path):
    path = tmp_path / "配置.yaml"
    path.write_text("schema_version: 1\nname: example\n", encoding="utf-8")
    raw, source = load_yaml_mapping(path)
    assert raw["name"] == "example"
    assert source == path.resolve()
    assert load_versioned_config(path, ExampleConfig).name == "example"


@pytest.mark.parametrize("value", [None, True, "1"])
def test_require_schema_version_rejects_non_integer(value):
    with pytest.raises(ConfigurationError, match="schema_version"):
        require_schema_version({"schema_version": value}, supported={1})


def test_require_schema_version_reports_supported_versions():
    with pytest.raises(ConfigurationError) as caught:
        require_schema_version({"schema_version": 3}, supported={1, 2})
    assert caught.value.details == {"actual": 3, "supported": [1, 2]}


def test_ser_error_has_stable_structured_form():
    error = SERError("failed", code="example_error", details={"item": 2})
    assert error.to_dict() == {
        "code": "example_error",
        "message": "failed",
        "details": {"item": 2},
    }
    data_error = ManifestError("bad manifest", uid="a")
    assert isinstance(data_error, SERDataError)
    assert isinstance(data_error, SERError)
    assert data_error.details["uid"] == "a"


def test_progress_event_validates_counts_and_calculates_fraction():
    assert ProgressEvent("decode", completed=2, total=4).fraction == 0.5
    assert ProgressEvent("scan", completed=0).fraction is None
    with pytest.raises(ValueError, match="大于"):
        ProgressEvent("decode", completed=5, total=4)


def test_cancellation_token_is_idempotent_and_raises_domain_error():
    token = CancellationToken()
    assert token.is_cancelled is False
    token.cancel()
    token.cancel()
    assert token.is_cancelled is True
    with pytest.raises(OperationCancelled):
        token.raise_if_cancelled()


def test_logging_configuration_does_not_change_root_logger():
    root = logging.getLogger()
    previous_level = root.level
    stream = io.StringIO()
    handler = configure_library_logging("INFO", stream=stream)
    try:
        get_logger("test").info("hello")
        assert "ser_lib.test" in stream.getvalue()
        assert "hello" in stream.getvalue()
        assert root.level == previous_level
    finally:
        logging.getLogger("ser_lib").removeHandler(handler)
