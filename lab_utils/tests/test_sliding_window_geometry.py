"""Tests for the careful sliding-window helpers in lab_utils.eval.sliding_window.

Covers the pieces that decide whether the sliding window recovers small-splice
recall without inflating false positives: the square scale-ladder geometry, the
top2 aggregator, and the calibrate-on-reals / apply-fixed-threshold flow.

No model, no GPU, no data — only numpy/python helpers (the module imports torch
at top, so torch must be importable, but nothing here touches a device).
"""

import numpy as np
import pytest

from lab_utils.eval.sliding_window import (
    _square_crop_boxes,
    _top2_logit,
    _area_bucket,
    _AREA_TIERS,
    calibrate_threshold_at_tnr,
    metrics_at_threshold,
)


# ── square scale-ladder geometry ──────────────────────────────────────────────

def test_square_boxes_are_square_and_in_bounds():
    H, W = 1000, 600
    boxes = _square_crop_boxes(H, W, scales=(1.0, 0.6, 0.4), stride_frac=0.5)
    assert boxes, 'expected sub-windows for scales < 1.0'
    short = min(H, W)
    for (t, l, side) in boxes:
        # within bounds
        assert 0 <= t and t + side <= H
        assert 0 <= l and l + side <= W
        # square side matches one of the requested scales of the short edge
        assert side in {round(short * 0.6), round(short * 0.4)}


def test_square_boxes_skip_full_scale():
    # scale >= 1.0 is the full image, added separately — no boxes for it.
    assert _square_crop_boxes(500, 500, scales=(1.0,), stride_frac=0.5) == []


def test_square_boxes_cover_far_edges():
    # A strided grid must explicitly include the bottom/right edge windows so
    # a splice in the corner is never missed.
    H, W = 900, 700
    boxes = _square_crop_boxes(H, W, scales=(0.4,), stride_frac=0.5)
    side = round(min(H, W) * 0.4)
    assert any(t + side == H for (t, l, s) in boxes), 'bottom edge not covered'
    assert any(l + side == W for (t, l, s) in boxes), 'right edge not covered'


# ── top2 aggregator ───────────────────────────────────────────────────────────

def test_top2_is_mean_of_two_highest():
    assert _top2_logit(np.array([-1.0, 5.0, 3.0, 0.0])) == pytest.approx(4.0)


def test_top2_single_window_falls_back():
    assert _top2_logit(np.array([2.5])) == pytest.approx(2.5)


def test_top2_is_below_max_when_windows_disagree():
    # The whole point: one suspicious window does not by itself max out top2.
    logits = np.array([-2.0, -1.5, 6.0])
    assert _top2_logit(logits) < float(logits.max())


# ── area tiers ────────────────────────────────────────────────────────────────

def test_area_bucket_edges():
    assert _area_bucket(0.0)   == 'tiny'
    assert _area_bucket(0.05)  == 'tiny'
    assert _area_bucket(0.051) == 'small'
    assert _area_bucket(0.149) == 'small'
    assert _area_bucket(0.15)  == 'medium'
    assert _area_bucket(0.29)  == 'medium'
    assert _area_bucket(0.30)  == 'large'
    assert _area_bucket(0.9)   == 'large'


# ── calibrate-then-apply ──────────────────────────────────────────────────────

def _mk_records(real_scores, splice_specs):
    """splice_specs: list of (score, area)."""
    recs = [{'is_real': True, 'area': 0.0, 's': float(x)} for x in real_scores]
    recs += [{'is_real': False, 'area': float(a), 's': float(x)} for (x, a) in splice_specs]
    return recs


def test_calibrate_threshold_hits_target_tnr():
    reals = np.linspace(-3.0, -0.5, 200)   # 200 real scores
    recs = _mk_records(reals, [(2.0, 0.03)])
    thr = calibrate_threshold_at_tnr(recs, 's', 0.95)
    # ~5% of reals should sit at/above the threshold → TNR ~= 0.95.
    achieved_tnr = float((reals < thr).mean())
    assert achieved_tnr == pytest.approx(0.95, abs=0.02)


def test_calibrate_threshold_monotonic_in_tnr():
    reals = np.random.RandomState(0).randn(500)
    recs = _mk_records(reals, [(1.0, 0.03)])
    t95 = calibrate_threshold_at_tnr(recs, 's', 0.95)
    t99 = calibrate_threshold_at_tnr(recs, 's', 0.99)
    assert t99 >= t95   # tighter TNR ⇒ higher threshold


def test_calibrate_no_reals_returns_inf():
    recs = [{'is_real': False, 'area': 0.03, 's': 1.0}]
    assert calibrate_threshold_at_tnr(recs, 's', 0.95) == float('inf')


def test_metrics_at_threshold_per_tier_tpr_and_tnr():
    # 10 reals below 0, one splice per tier well above the threshold=0.0.
    reals = list(np.linspace(-5.0, -0.5, 10))
    splices = [(3.0, 0.03),   # tiny
               (3.0, 0.10),   # small
               (3.0, 0.20),   # medium
               (3.0, 0.50)]   # large
    recs = _mk_records(reals, splices)
    m = metrics_at_threshold(recs, 's', threshold=0.0)
    assert m['tnr'] == pytest.approx(1.0)        # all reals below 0
    assert m['tpr'] == pytest.approx(1.0)        # all splices above 0
    for tier in _AREA_TIERS:
        assert m['tiers'][tier]['n'] == 1
        assert m['tiers'][tier]['tpr'] == pytest.approx(1.0)


def test_metrics_at_threshold_misses_below_threshold():
    # A tiny splice scoring below the threshold is a miss; tnr unaffected.
    recs = _mk_records([-1.0, -2.0], [(-0.5, 0.03)])
    m = metrics_at_threshold(recs, 's', threshold=0.0)
    assert m['tiers']['tiny']['tpr'] == pytest.approx(0.0)
    assert m['tnr'] == pytest.approx(1.0)
