from pathlib import Path

import pytest

from ser_lib.engine import build_experiment_components, load_experiment_config


@pytest.mark.parametrize(
    ("file_name", "model_type", "feature_dim", "manifest_directory"),
    [
        ("cnn_logmel.yaml", "cnn_baseline", 64, "standard"),
        ("gru_mfcc.yaml", "gru_baseline", 40, "standard"),
        ("transformer_logmel.yaml", "transformer_baseline", 64, "standard"),
        ("casia_cnn_logmel.yaml", "cnn_baseline", 64, "casia-standard"),
        ("csemotions_cnn_logmel.yaml", "cnn_baseline", 64, "csemotions-standard"),
        ("esd_cnn_logmel.yaml", "cnn_baseline", 64, "esd-standard"),
        ("crema_d_cnn_logmel.yaml", "cnn_baseline", 64, "crema-d-standard"),
        ("emotiontalk_cnn_logmel.yaml", "cnn_baseline", 64, "emotiontalk-standard"),
    ],
)
def test_release_config_builds_components(
    file_name: str, model_type: str, feature_dim: int, manifest_directory: str
) -> None:
    config_path = Path(__file__).parents[1] / "configs" / file_name

    config = load_experiment_config(config_path)
    components = build_experiment_components(config, train=True)

    assert config.model.type == model_type
    assert components.model.model_spec.required_inputs["features"].feature_dim == feature_dim
    assert config.data.manifest == (
        config_path.parent / f"../data/{manifest_directory}/dataset.yaml"
    ).resolve()
