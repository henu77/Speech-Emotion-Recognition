"""可移植模型 artifact 的版本化 manifest。"""

from __future__ import annotations

from pathlib import PurePath
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ser_lib.core.config import StrictConfig


class ModelCard(StrictConfig):
    """最小模型卡；未知信息允许留空，但字段不可隐式发明。"""

    description: str = ""
    intended_use: str = ""
    dataset: str = ""
    language: list[str] = Field(default_factory=list)
    license: str = ""
    limitations: list[str] = Field(default_factory=list)


class ModelArtifactManifest(StrictConfig):
    schema_version: int = 2
    library_version: str
    model_name: str
    model_params: dict[str, Any]
    input_specs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    weights_file: str = "weights.safetensors"
    weights_format: Literal["safetensors", "pytorch"] = "safetensors"
    weights_sha256: str
    files_sha256: dict[str, str] = Field(default_factory=dict)
    preprocessing: dict[str, Any]
    labels: dict[int, str]
    metrics: dict[str, float] = Field(default_factory=dict)
    model_card: ModelCard = Field(default_factory=ModelCard)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _legacy_defaults(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("schema_version", 1) == 1:
            updated = dict(value)
            updated.setdefault("weights_file", "model_state.pt")
            updated.setdefault("weights_format", "pytorch")
            return updated
        return value

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: int) -> int:
        if value not in (1, 2):
            raise ValueError(f"不支持 artifact schema_version={value}")
        return value

    @field_validator("weights_file")
    @classmethod
    def _safe_weights_name(cls, value: str) -> str:
        path = PurePath(value)
        if not value or path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
            raise ValueError("weights_file 必须是 artifact 根目录内的普通文件名")
        return value

    @field_validator("weights_sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise ValueError("weights_sha256 必须是 64 位十六进制 SHA-256")
        return value.lower()

    @field_validator("labels")
    @classmethod
    def _contiguous_labels(cls, value: dict[int, str]) -> dict[int, str]:
        if sorted(value) != list(range(len(value))):
            raise ValueError("artifact labels 必须从 0 开始连续")
        if len(value) < 2 or any(not name for name in value.values()):
            raise ValueError("artifact 至少需要两个非空标签")
        return value

    @model_validator(mode="after")
    def _format_matches_file(self) -> "ModelArtifactManifest":
        if self.schema_version == 1:
            return self
        if self.weights_format == "safetensors" and not self.weights_file.endswith(".safetensors"):
            raise ValueError("safetensors 权重文件必须使用 .safetensors 后缀")
        if self.weights_format == "pytorch" and not self.weights_file.endswith((".pt", ".pth")):
            raise ValueError("pytorch 权重文件必须使用 .pt 或 .pth 后缀")
        return self


__all__ = ["ModelCard", "ModelArtifactManifest"]
