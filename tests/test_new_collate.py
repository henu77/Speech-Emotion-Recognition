import pytest
import torch

from ser_lib.data.collate import SERCollator
from ser_lib.data.config import BatchingConfig, FixedBatching, SlidingBatching
from ser_lib.data.errors import CollationError
from ser_lib.data.types import SERSample, TensorSpec


def _sample(uid, inputs, lengths, label=0):
    return SERSample(uid=uid, inputs=inputs, lengths=lengths, label=label, metadata={"uid": uid})


def test_dynamic_collates_independent_time_axes():
    specs = {
        "mel": TensorSpec(layout="FT", feature_dim=2),
        "prosody": TensorSpec(layout="TD", feature_dim=3),
        "global": TensorSpec(layout="D", feature_dim=4),
    }
    samples = [
        _sample("a", {"mel": torch.ones(2, 3), "prosody": torch.ones(2, 3), "global": torch.ones(4)}, {"mel": 3, "prosody": 2}),
        _sample("b", {"mel": torch.ones(2, 5), "prosody": torch.ones(4, 3), "global": torch.ones(4)}, {"mel": 5, "prosody": 4}),
    ]
    batch = SERCollator(specs, BatchingConfig(type="dynamic"))(samples)
    assert batch.inputs["mel"].shape == (2, 2, 5)
    assert batch.inputs["prosody"].shape == (2, 4, 3)
    assert batch.inputs["global"].shape == (2, 4)
    assert batch.masks["mel"].tolist() == [[True, True, True, False, False], [True] * 5]
    assert "global" not in batch.lengths


def test_fixed_requires_length_for_each_temporal_key():
    with pytest.raises(CollationError, match="缺失"):
        SERCollator(
            {"a": TensorSpec(layout="T"), "b": TensorSpec(layout="T")},
            BatchingConfig(type="fixed", fixed=FixedBatching(max_lengths={"a": 3})),
        )


def test_sliding_repeats_non_temporal_inputs_and_metadata():
    specs = {
        "features": TensorSpec(layout="FT", feature_dim=2),
        "global": TensorSpec(layout="D", feature_dim=1),
    }
    samples = [
        _sample("a", {"features": torch.arange(14.0).reshape(2, 7), "global": torch.tensor([10.0])}, {"features": 7}, 1),
        _sample("b", {"features": torch.ones(2, 2), "global": torch.tensor([20.0])}, {"features": 2}, 0),
    ]
    config = BatchingConfig(
        type="sliding",
        sliding=SlidingBatching(window_size=4, stride=3),
        primary_key="features",
    )
    batch = SERCollator(specs, config)(samples)
    assert batch.inputs["features"].shape == (3, 2, 4)
    assert batch.inputs["global"].flatten().tolist() == [10.0, 10.0, 20.0]
    assert batch.window_map.tolist() == [0, 0, 1]
    assert batch.uids == ["a", "a", "b"]
    assert batch.labels.tolist() == [1, 1, 0]


def test_sliding_expands_uid_for_multiple_windows():
    sample = _sample("a", {"waveform": torch.arange(8.0)}, {"waveform": 8}, 1)
    config = BatchingConfig(
        type="sliding", sliding=SlidingBatching(window_size=4, stride=2)
    )
    batch = SERCollator({"waveform": TensorSpec(layout="T")}, config)([sample])
    assert batch.inputs["waveform"].shape == (3, 4)
    assert batch.uids == ["a", "a", "a"]
    assert len(batch.metadata) == 3


def test_collator_rejects_partially_labeled_batch():
    samples = [
        _sample("a", {"waveform": torch.ones(2)}, {"waveform": 2}, 0),
        _sample("b", {"waveform": torch.ones(2)}, {"waveform": 2}, None),
    ]
    with pytest.raises(CollationError, match="部分样本有标签"):
        SERCollator({"waveform": TensorSpec(layout="T")}, BatchingConfig())(samples)
