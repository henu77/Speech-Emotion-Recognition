import pytest
import torch

from ser_lib.data.types import AudioRecord, SERBatch, TensorSpec


def test_audio_record_treats_missing_start_as_zero():
    record = AudioRecord(uid="a", audio_path=__import__("pathlib").Path("a.wav"), end_ms=10)
    assert record.start_ms is None


def test_audio_record_rejects_non_positive_end_from_zero():
    with pytest.raises(ValueError, match="end_ms > start_ms"):
        AudioRecord(uid="a", audio_path=__import__("pathlib").Path("a.wav"), end_ms=0)


def test_tensor_spec_rejects_feature_dim_for_plain_waveform():
    with pytest.raises(ValueError, match="不允许配置 feature_dim"):
        TensorSpec(layout="T", feature_dim=1)


def test_ser_batch_rejects_misaligned_metadata():
    with pytest.raises(ValueError, match="uids/metadata"):
        SERBatch(
            inputs={"waveform": torch.zeros(2, 4)},
            lengths={"waveform": torch.tensor([4, 4])},
            masks={"waveform": torch.ones(2, 4, dtype=torch.bool)},
            labels=torch.tensor([0, 1]),
            uids=["only-one"],
            metadata=[{}, {}],
        )


def test_ser_batch_rejects_non_boolean_mask():
    with pytest.raises(ValueError, match="bool tensor"):
        SERBatch(
            inputs={"waveform": torch.zeros(1, 4)},
            lengths={"waveform": torch.tensor([4])},
            masks={"waveform": torch.ones(1, 4)},
            labels=None,
            uids=["a"],
            metadata=[{}],
        )
