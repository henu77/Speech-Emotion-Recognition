"""Representation 子包：全部表示组件与注册入口。"""

from __future__ import annotations

from ser_lib.data.registry import Registry, default_registry
from ser_lib.data.representations.acoustic import (
    AcousticFeatures,
    AcousticFeaturesConfig,
)
from ser_lib.data.representations.base import Representation
from ser_lib.data.representations.composite import (
    CompositeConfig,
    CompositeRepresentation,
)
from ser_lib.data.representations.spectral import (
    LogMelConfig,
    LogMelRepresentation,
    MFCCConfig,
    MFCCRepresentation,
    MelConfig,
    MelSpectrogramRepresentation,
    SpectrogramConfig,
    SpectrogramRepresentation,
)
from ser_lib.data.representations.waveform import RawWaveform, RawWaveformConfig

__all__ = [
    "Representation",
    "RawWaveform",
    "RawWaveformConfig",
    "SpectrogramRepresentation",
    "SpectrogramConfig",
    "MelSpectrogramRepresentation",
    "MelConfig",
    "LogMelRepresentation",
    "LogMelConfig",
    "MFCCRepresentation",
    "MFCCConfig",
    "AcousticFeatures",
    "AcousticFeaturesConfig",
    "CompositeRepresentation",
    "CompositeConfig",
    "register_representations",
]


def register_representations(registry: Registry | None = None) -> None:
    """把全部表示注册到注册表（默认注册到 default_registry）。"""
    registry = registry or default_registry
    registry.register(
        namespace="representation", name="waveform",
        factory=RawWaveform, config_model=RawWaveformConfig,
        descriptor=RawWaveform.descriptor,
    )
    registry.register(
        namespace="representation", name="spectrogram",
        factory=SpectrogramRepresentation, config_model=SpectrogramConfig,
        descriptor=SpectrogramRepresentation.descriptor,
    )
    registry.register(
        namespace="representation", name="mel_spectrogram",
        factory=MelSpectrogramRepresentation, config_model=MelConfig,
        descriptor=MelSpectrogramRepresentation.descriptor,
    )
    registry.register(
        namespace="representation", name="log_mel",
        factory=LogMelRepresentation, config_model=LogMelConfig,
        descriptor=LogMelRepresentation.descriptor,
    )
    registry.register(
        namespace="representation", name="mfcc",
        factory=MFCCRepresentation, config_model=MFCCConfig,
        descriptor=MFCCRepresentation.descriptor,
    )
    registry.register(
        namespace="representation", name="acoustic_features",
        factory=AcousticFeatures, config_model=AcousticFeaturesConfig,
        descriptor=AcousticFeatures.descriptor,
    )
    registry.register(
        namespace="representation", name="composite",
        factory=CompositeRepresentation, config_model=CompositeConfig,
        descriptor=CompositeRepresentation.descriptor,
    )
