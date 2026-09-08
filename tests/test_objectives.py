from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError

from ser_lib.engine import (
    ClassificationLoss,
    LossConfig,
    SamplingConfig,
    build_weighted_sampler,
)


def test_focal_gamma_zero_matches_cross_entropy():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    focal = ClassificationLoss(
        LossConfig(type="focal", focal_gamma=0.0), num_classes=2
    )
    assert focal(logits, targets) == pytest.approx(F.cross_entropy(logits, targets))


def test_class_weight_validation_and_weighted_loss():
    loss = ClassificationLoss(
        LossConfig(type="cross_entropy", class_weights=[1.0, 3.0]), num_classes=2
    )
    logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    targets = torch.tensor([0, 1])
    assert loss(logits, targets) > 0
    with pytest.raises(ValueError, match="长度"):
        ClassificationLoss(LossConfig(class_weights=[1.0]), num_classes=2)
    with pytest.raises(ValidationError, match="大于 0"):
        LossConfig(class_weights=[1.0, 0.0])


def test_weighted_sampler_balances_total_class_mass_and_is_deterministic():
    config = SamplingConfig(type="weighted", num_samples=8)
    first = build_weighted_sampler([0, 0, 0, 1], num_classes=2, config=config, seed=7)
    second = build_weighted_sampler([0, 0, 0, 1], num_classes=2, config=config, seed=7)
    assert first is not None and second is not None
    assert first.weights[:3].sum() == pytest.approx(first.weights[3])
    assert list(first) == list(second)


def test_weighted_sampler_rejects_unlabeled_data():
    with pytest.raises(ValueError, match="都有标签"):
        build_weighted_sampler(
            [0, None], num_classes=2,
            config=SamplingConfig(type="weighted"), seed=1,
        )
