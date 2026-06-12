"""Tests for lab_utils.eval.metrics.f1_iou / binary_metrics."""

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lab_utils.eval.metrics import binary_metrics, f1_iou


def test_f1_iou_perfect_agreement():
    a = np.array([1, 1, 0, 0], dtype=bool)
    assert f1_iou(a, a) == (1.0, 1.0)


def test_f1_iou_disjoint_returns_zero():
    a = np.array([1, 1, 0, 0], dtype=bool)
    b = np.array([0, 0, 1, 1], dtype=bool)
    f, i = f1_iou(a, b)
    assert f == 0.0
    assert i == 0.0


def test_f1_iou_partial_overlap():
    # pred {0, 1}, gt {0} -> inter=1, union=2, f1=2/3, iou=1/2
    p = np.array([1, 1, 0, 0], dtype=bool)
    g = np.array([1, 0, 0, 0], dtype=bool)
    f, i = f1_iou(p, g)
    assert round(f, 4) == round(2 / 3, 4)
    assert i == 0.5


def test_f1_iou_empty_both_principled_default():
    """Default empty_value=1.0: empty/empty is a perfect (degenerate) match."""
    z = np.zeros(4, dtype=bool)
    assert f1_iou(z, z) == (1.0, 1.0)


def test_f1_iou_empty_both_diagnose_convention():
    """empty_value=0.0 reproduces the diagnose-script convention."""
    z = np.zeros(4, dtype=bool)
    assert f1_iou(z, z, empty_value=0.0) == (0.0, 0.0)


def test_binary_metrics_returns_full_dict():
    p = np.array([1, 1, 0, 0], dtype=bool)
    g = np.array([1, 0, 0, 0], dtype=bool)
    m = binary_metrics(p, g)
    assert set(m.keys()) == {'f1', 'iou', 'prec', 'rec', 'pred_frac', 'gt_frac'}
    assert m['prec'] == 0.5    # 1 true pos of 2 pos predictions
    assert m['rec'] == 1.0     # caught the 1 GT
    assert m['pred_frac'] == 0.5
    assert m['gt_frac'] == 0.25


def test_binary_metrics_empty_returns_zeros():
    z = np.zeros(4, dtype=bool)
    m = binary_metrics(z, z)
    assert all(v == 0.0 for v in m.values())
