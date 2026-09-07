"""可移植模型包的版本化 manifest。"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

class ModelArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    library_version: str
    model_name: str
    model_params: dict[str, Any]
    weights_file: str = "model_state.pt"
    weights_sha256: str
    preprocessing: dict[str, Any]
    labels: dict[int, str]
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
