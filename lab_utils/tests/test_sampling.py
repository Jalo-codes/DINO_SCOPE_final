"""Tests for lab_utils.data.sampling."""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lab_utils.data.sampling import (
    build_case_balanced_quick_val_items,
    build_quick_val_items,
    build_shared_tau_calibration_items,
    deterministic_subsample,
    is_real,
    is_splice,
    items_for_source,
    reals_subsample,
    splice_balance_weights,
    stable_item_sort_key,
    val_mix_counts,
    val_source_counts,
)


def _item(source: str, kind: str, case: str, name: str) -> dict:
    return {'source': source, 'kind': kind, 'case_id': case, 'img': name}


# ── splice_balance_weights ────────────────────────────────────────────────


def test_balance_one_splice_one_real_equal_weights():
    items = [
        _item('imd2020', 'imd_splice', 'a', 'a.jpg'),
        _item('imd2020', 'imd_real',   'b', 'b.jpg'),
    ]
    assert splice_balance_weights(items) == [0.5, 0.5]


def test_balance_skewed_distribution():
    items = [_item('imd2020', 'imd_splice', f'c{i}', f's{i}.jpg') for i in range(3)]
    items.append(_item('imd2020', 'imd_real', 'r', 'r.jpg'))
    out = splice_balance_weights(items)
    expected = [0.5 / 3, 0.5 / 3, 0.5 / 3, 0.5]
    assert all(abs(a - b) < 1e-9 for a, b in zip(out, expected))


def test_balance_returns_stats_when_requested():
    items = [
        _item('imd2020', 'imd_splice', 'a', 'a.jpg'),
        _item('imd2020', 'imd_real',   'b', 'b.jpg'),
    ]
    weights, stats = splice_balance_weights(items, return_stats=True)
    assert stats['splice_pos'] == 1
    assert stats['single_region'] == 1
    assert stats['target_splice_frac'] == 0.5
    assert stats['target_single_frac'] == 0.5


def test_balance_biased_target():
    items = [_item('imd2020', 'imd_splice', f's{i}', f's{i}.jpg') for i in range(3)]
    items.append(_item('imd2020', 'imd_real', 'r', 'r.jpg'))
    w = splice_balance_weights(items, target_splice_frac=0.8)
    expected = [0.8 / 3, 0.8 / 3, 0.8 / 3, 0.2]
    assert all(abs(a - b) < 1e-9 for a, b in zip(w, expected))


def test_balance_degenerate_all_real_returns_uniform():
    items = [_item('imd2020', 'imd_real', f'r{i}', f'r{i}.jpg') for i in range(4)]
    assert splice_balance_weights(items) == [1.0, 1.0, 1.0, 1.0]


# ── helpers ────────────────────────────────────────────────────────────────


def test_is_splice_and_is_real():
    assert is_splice({'kind': 'imd_splice'})
    assert is_splice({'kind': 'casia_splice'})
    assert not is_splice({'kind': 'imd_real'})
    assert is_real({'kind': 'imd_real'})
    assert is_real({'kind': 'casia_real'})
    assert is_real({'kind': 'indoor_real'})
    assert not is_real({'kind': 'imd_splice'})


def test_stable_item_sort_key_deterministic():
    a = _item('s', 'k', 'c', 'i')
    b = _item('s', 'k', 'c', 'i')
    assert stable_item_sort_key(a) == stable_item_sort_key(b)
    assert len(stable_item_sort_key(a)) == 32


# ── build_quick_val_items ─────────────────────────────────────────────────


def test_quick_val_returns_full_when_under_cap():
    items = [_item('imd2020', 'imd_splice', f'c{i}', f's{i}.jpg') for i in range(5)]
    out = build_quick_val_items(items, cap=10)
    assert len(out) == 5


def test_quick_val_caps_and_is_deterministic():
    items = [_item('imd2020', 'imd_splice', f'c{i}', f's{i}.jpg') for i in range(40)]
    items += [_item('casia', 'casia_splice', f'cs{i}', f'cs{i}.jpg') for i in range(20)]
    out1 = build_quick_val_items(items, cap=20)
    out2 = build_quick_val_items(items, cap=20)
    assert len(out1) == 20
    assert [stable_item_sort_key(it) for it in out1] == \
           [stable_item_sort_key(it) for it in out2]


def test_quick_val_keeps_every_group_represented():
    items = [_item('imd2020', 'imd_splice', f'c{i}', f's{i}.jpg') for i in range(40)]
    items += [_item('casia', 'casia_splice', f'cs{i}', f'cs{i}.jpg') for i in range(2)]
    out = build_quick_val_items(items, cap=20)
    sources = {it['source'] for it in out}
    # The small CASIA group should not be erased.
    assert 'casia' in sources


# ── build_case_balanced_quick_val_items ───────────────────────────────────


def test_case_balanced_quick_val_pairs():
    items = []
    for i in range(3):
        cid = f'case{i}'
        items.append(_item('casia', 'imd_real',   cid, f'cr{i}.jpg'))
        items.append(_item('casia', 'imd_splice', cid, f'cs{i}.jpg'))
    for i in range(2):
        cid = f'imd{i}'
        items.append(_item('imd2020', 'imd_real',   cid, f'ir{i}.jpg'))
        items.append(_item('imd2020', 'imd_splice', cid, f'is{i}.jpg'))
    out = build_case_balanced_quick_val_items(items, imd_cases=2, casia_pairs=3)
    # 2 IMD pairs + 3 CASIA pairs = 10
    assert len(out) == 10


# ── deterministic_subsample + reals_subsample ─────────────────────────────


def test_deterministic_subsample_stable_across_calls():
    items = [{'img': f'p{i}.jpg'} for i in range(20)]
    a = deterministic_subsample(items, 5, seed='alpha')
    b = deterministic_subsample(items, 5, seed='alpha')
    c = deterministic_subsample(items, 5, seed='beta')
    assert a == b
    assert a != c
    assert len(a) == 5


def test_subsample_returns_full_when_under_target():
    items = [{'img': f'p{i}.jpg'} for i in range(3)]
    assert deterministic_subsample(items, 10, seed='x') == items


def test_reals_subsample_keeps_rate():
    items = [{'img': f'p{i}.jpg'} for i in range(20)]
    out = reals_subsample(items, 0.5, seed='r')
    assert len(out) == 10
    assert len(reals_subsample(items, 1.0, seed='r')) == 20


# ── val_mix / val_source counts ───────────────────────────────────────────


def test_val_mix_counts():
    items = [
        _item('casia', 'imd_real',   'a', 'a.jpg'),
        _item('casia', 'imd_real',   'b', 'b.jpg'),
        _item('casia', 'imd_splice', 'c', 'c.jpg'),
    ]
    assert val_mix_counts(items) == {
        ('casia', 'imd_real'): 2,
        ('casia', 'imd_splice'): 1,
    }
    assert val_source_counts(items) == {'casia': 3}


def test_items_for_source():
    items = [
        _item('casia', 'imd_real',   'a', 'a.jpg'),
        _item('imd2020', 'imd_real', 'b', 'b.jpg'),
    ]
    assert len(items_for_source(items, 'casia')) == 1
    assert len(items_for_source(items, 'unknown')) == 0


def test_build_shared_tau_calibration_items_per_source():
    items = []
    for i in range(5):
        items.append(_item('imd2020', 'imd_real',   f'r{i}', f'r{i}.jpg'))
        items.append(_item('imd2020', 'imd_splice', f's{i}', f's{i}.jpg'))
        items.append(_item('casia',   'imd_real',   f'cr{i}', f'cr{i}.jpg'))
        items.append(_item('casia',   'imd_splice', f'cs{i}', f'cs{i}.jpg'))
    out = build_shared_tau_calibration_items(
        items, singles_per_source=2, splices_per_source=2,
    )
    # 2 singles + 2 splices per source × 2 sources = 8
    assert len(out) == 8
