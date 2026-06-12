import pytest

from lab_utils.eval.image_bce import image_bce_metrics


def test_image_bce_metrics_balanced_accuracy_and_tiers():
    m = image_bce_metrics(
        logits=[-3.0, -2.0, 2.0, 3.0],
        labels=[0, 0, 1, 1],
        areas=[0.0, 0.0, 0.04, 0.2],
        threshold=0.5,
    )
    assert m.n_total == 4
    assert m.bal_acc == pytest.approx(1.0)
    assert m.tier_stats["tiny"]["n"] == 1
    assert m.tier_stats["medium"]["n"] == 1


def test_image_bce_metrics_validates_lengths():
    with pytest.raises(ValueError):
        image_bce_metrics([1.0], [1, 0])
