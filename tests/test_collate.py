import torch

from ser_lib.dataset.collate import build_collate_fn


def test_waveform_dynamic_mask_returns_unified_batch_contract():
    collate_fn = build_collate_fn(
        {
            "audio_processing": {"strategy": "dynamic_mask"},
            "transforms": {"batch_level": {"train": []}},
        }
    )

    batch = [
        ({"raw_waveform": torch.tensor([1.0, 2.0, 3.0])}, torch.tensor(1), 3),
        ({"raw_waveform": torch.tensor([4.0, 5.0])}, torch.tensor(0), 2),
    ]

    output = collate_fn(batch)

    assert set(output.keys()) == {"inputs", "labels", "lengths", "mask", "meta"}
    assert output["inputs"]["raw_waveform"].shape == (2, 3)
    assert output["labels"].shape == (2,)
    assert output["lengths"].tolist() == [3, 2]
    assert output["mask"].dtype == torch.bool
    assert output["mask"].shape == (2, 3)
    assert output["meta"]["dataset_type"] == "waveform"
    assert output["meta"]["collate_strategy"] == "dynamic_mask"
    assert output["meta"]["window_counts"] is None
    assert output["meta"]["original_labels"] is None


def test_spectrogram_truncate_pad_outputs_cnn_shape():
    collate_fn = build_collate_fn(
        {
            "spectrogram": {"type": "LogMelSpectrogram"},
            "audio_processing": {"strategy": "truncate_pad", "max_frames": 5},
            "transforms": {"batch_level": {"train": []}},
        }
    )

    batch = [
        ({"logmelspectrogram": torch.ones(3, 4)}, torch.tensor(1), 4),
        ({"logmelspectrogram": torch.ones(3, 6)}, torch.tensor(0), 6),
    ]

    output = collate_fn(batch)

    assert output["inputs"]["logmelspectrogram"].shape == (2, 1, 3, 5)
    assert output["labels"].tolist() == [1, 0]
    assert output["lengths"].tolist() == [4, 5]
    assert output["mask"] is None
    assert output["meta"]["dataset_type"] == "spectrogram"
    assert output["meta"]["collate_strategy"] == "truncate_pad"


def test_feature_dynamic_mask_promotes_scalar_features_to_last_dim():
    collate_fn = build_collate_fn(
        {
            "features": {"selected_features": {"f0": {"type": "F0"}}},
            "audio_processing": {"strategy": "dynamic_mask"},
            "transforms": {"batch_level": {"train": []}},
        }
    )

    batch = [
        (
            {
                "f0": torch.tensor([1.0, 2.0, 3.0]),
                "rms": torch.tensor([[1.0, 2.0, 3.0]]),
            },
            torch.tensor(1),
            3,
        ),
        (
            {
                "f0": torch.tensor([4.0, 5.0]),
                "rms": torch.tensor([[4.0, 5.0]]),
            },
            torch.tensor(0),
            2,
        ),
    ]

    output = collate_fn(batch)

    assert output["inputs"]["f0"].shape == (2, 3, 1)
    assert output["inputs"]["rms"].shape == (2, 3, 1)
    assert output["lengths"].tolist() == [3, 2]
    assert output["mask"].shape == (2, 3)
    assert output["meta"]["dataset_type"] == "feature"
    assert output["meta"]["collate_strategy"] == "dynamic_mask"


def test_waveform_sliding_window_records_window_metadata():
    collate_fn = build_collate_fn(
        {
            "audio_processing": {
                "strategy": "sliding_window",
                "window_size": 4,
                "stride": 2,
            },
            "transforms": {"batch_level": {"train": []}},
        }
    )

    batch = [
        ({"raw_waveform": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])}, torch.tensor(1), 5),
        ({"raw_waveform": torch.tensor([6.0, 7.0])}, torch.tensor(0), 2),
    ]

    output = collate_fn(batch)

    assert output["inputs"]["raw_waveform"].shape == (3, 4)
    assert output["labels"].tolist() == [1, 1, 0]
    assert output["lengths"].tolist() == [4, 4, 2]
    assert output["mask"] is None
    assert output["meta"]["window_counts"].tolist() == [2, 1]
    assert output["meta"]["original_labels"].tolist() == [1, 0]
