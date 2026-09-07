"""Transform 子包：注册全部 transform 到默认注册表。"""

from __future__ import annotations

from ser_lib.data.registry import Registry, default_registry
from ser_lib.data.transforms.base import (
    FeatureTransformPipeline,
    RandomApply,
    WaveformTransformPipeline,
    validate_feature_transform_layouts,
)
from ser_lib.data.transforms.feature import (
    SPEC_MASKING_DESCRIPTOR,
    SpecMasking,
    SpecMaskingConfig,
)
from ser_lib.data.transforms.waveform import WAVEFORM_TRANSFORM_SPECS

__all__ = [
    "RandomApply",
    "WaveformTransformPipeline",
    "FeatureTransformPipeline",
    "validate_feature_transform_layouts",
    "SpecMasking",
    "SpecMaskingConfig",
    "SPEC_MASKING_DESCRIPTOR",
    "register_transforms",
]


def register_transforms(registry: Registry | None = None) -> None:
    """注册波形级与特征级 transform（默认注册到 default_registry）。"""
    registry = registry or default_registry
    for name, (factory, config_model, descriptor) in WAVEFORM_TRANSFORM_SPECS.items():
        registry.register(
            namespace="waveform_transform", name=name,
            factory=factory, config_model=config_model, descriptor=descriptor,
        )
    registry.register(
        namespace="feature_transform", name="spec_masking",
        factory=SpecMasking, config_model=SpecMaskingConfig,
        descriptor=SPEC_MASKING_DESCRIPTOR,
    )
