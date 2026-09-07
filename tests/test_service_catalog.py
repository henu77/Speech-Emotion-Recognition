from pathlib import Path

from ser_lib.data.config import AudioSettings, BatchingConfig, ComponentConfig, DataConfig
from ser_lib.service import check_compatibility, component_catalog


def _data(feature_dim=16):
    return DataConfig(
        manifest=Path("unused.yaml"),
        audio=AudioSettings(target_sample_rate=16000),
        representation=ComponentConfig(
            type="log_mel",
            params={"sample_rate": 16000, "n_mels": feature_dim},
        ),
        batching=BatchingConfig(type="dynamic"),
        labels={0: {"en": "neutral"}, 1: {"en": "happy"}},
    )


def test_component_catalog_contains_stable_data_and_model_components():
    catalog = component_catalog()
    assert "log_mel" in {item["id"] for item in catalog["representation"]}
    assert "cnn_baseline" in {item["id"] for item in catalog["models"]}
    assert "pitch_shift" not in {item["id"] for item in catalog["waveform_transform"]}


def test_compatibility_service_reports_success_and_dimension_failure():
    success = check_compatibility(
        _data(), "cnn_baseline", {"feature_dim": 16, "num_classes": 2}
    )
    assert success == {"compatible": True, "errors": []}
    failure = check_compatibility(
        _data(), "cnn_baseline", {"feature_dim": 32, "num_classes": 2}
    )
    assert failure["compatible"] is False
    assert "feature_dim" in failure["errors"][0]
