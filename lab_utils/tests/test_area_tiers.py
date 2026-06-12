from lab_utils.data.area_tiers import area_tier, area_tier_labels, with_area_tier


def test_area_tier_edges():
    assert area_tier(0.0) == "tiny"
    assert area_tier(0.05) == "tiny"
    assert area_tier(0.051) == "small"
    assert area_tier(0.149) == "small"
    assert area_tier(0.15) == "medium"
    assert area_tier(0.299) == "medium"
    assert area_tier(0.30) == "large"


def test_area_tier_labels_optional_real():
    assert area_tier_labels() == ("tiny", "small", "medium", "large")
    assert area_tier_labels(include_real=True)[0] == "real"


def test_with_area_tier_does_not_mutate_items():
    items = [{"blob_area_actual": 0.2}]
    out = with_area_tier(items)
    assert out[0]["area_tier"] == "medium"
    assert "area_tier" not in items[0]
