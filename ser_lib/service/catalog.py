"""桌面端可消费的组件目录与兼容性服务。"""
from __future__ import annotations
from typing import Any, Mapping
from ser_lib.data.config import DataConfig
from ser_lib.data.errors import SERDataError
from ser_lib.data.pipeline import build_pipeline
from ser_lib.data.registry import default_registry
from ser_lib.data.validation import validate_compatibility
from ser_lib.models.registry import model_registry

def component_catalog() -> dict[str, list[dict[str, Any]]]:
    return {
        namespace: default_registry.json_safe_descriptors(namespace)
        for namespace in ("importer", "representation", "waveform_transform", "feature_transform")
    } | {"models": model_registry.descriptors()}

def check_compatibility(data: DataConfig | Mapping[str, Any], model_name: str,
                        model_params: Mapping[str, Any]) -> dict[str, Any]:
    try:
        config = data if isinstance(data, DataConfig) else DataConfig.model_validate(data)
        pipeline = build_pipeline(config, train=False)
        model = model_registry.create(model_name, **dict(model_params))
        validate_compatibility(
            pipeline.output_specs, model.model_spec, config.batching,
            num_classes=config.num_classes,
        )
        return {"compatible": True, "errors": []}
    except (SERDataError, ValueError) as exc:
        return {"compatible": False, "errors": [str(exc)]}
