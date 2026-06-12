"""Tests for lab_utils.eval.window_geometry — pure geometry helpers."""

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lab_utils.eval.window_geometry import (
    axis_positions,
    capped_window_grid,
    centered_area_square,
    gt_bbox_and_centroid,
    inferred_bbox_from_patches,
    project_window_mask,
    square_expand_crop,
    window_footprint,
    window_grid,
    window_set_hash,
)


# ── axis_positions ────────────────────────────────────────────────────────


def test_axis_positions_full_stride():
    assert axis_positions(100, 50, 1.0) == [0, 50]


def test_axis_positions_half_stride():
    assert axis_positions(100, 50, 0.5) == [0, 25, 50]


def test_axis_positions_window_eq_length():
    assert axis_positions(50, 50, 1.0) == [0]


def test_axis_positions_includes_last_position():
    """The flush-with-edge position is always present even with awkward strides."""
    out = axis_positions(100, 50, 0.3)
    assert out[-1] == 50  # max_start


# ── window_grid / capped_window_grid ──────────────────────────────────────


def test_window_grid_uncapped():
    windows = window_grid((100, 100), scale=0.5, stride_frac=1.0, n_patch_per_side=4)
    assert len(windows) == 4
    for top, left, h, w in windows:
        assert h == w == 50
        assert 0 <= top and 0 <= left


def test_capped_window_grid_respects_max():
    out = capped_window_grid((400, 400), scale=0.25, stride_frac=0.25,
                              n_patch_per_side=4, max_windows=4)
    assert len(out) <= 4


def test_capped_window_grid_uncapped_when_max_large():
    out = capped_window_grid((400, 400), scale=0.25, stride_frac=0.25,
                              n_patch_per_side=4, max_windows=10_000)
    assert len(out) > 4  # uncapped count


# ── window_set_hash ───────────────────────────────────────────────────────


def test_window_set_hash_deterministic_and_short():
    windows = [(0, 0, 100, 100), (50, 50, 100, 100)]
    h1 = window_set_hash(windows)
    h2 = window_set_hash(windows)
    assert h1 == h2
    assert len(h1) == 12
    assert window_set_hash([]) == 'empty'


# ── square_expand_crop ────────────────────────────────────────────────────


def test_square_expand_crop_makes_square_in_bounds():
    top, left, side = square_expand_crop(10, 10, 20, 30, 100, 100)
    assert side == 30  # max of h, w
    assert top + side <= 100
    assert left + side <= 100


def test_square_expand_crop_clips_when_image_too_small():
    # h=80, w=80, but image is 50x50 — side should clip
    top, left, side = square_expand_crop(0, 0, 80, 80, 50, 50)
    assert side == 50


# ── centered_area_square ──────────────────────────────────────────────────


def test_centered_area_square_side_matches_area_frac():
    # area_frac=0.25, image 100x100 -> side² = 0.25 * 100 * 100 = 2500 -> side = 50
    top, left, side = centered_area_square((50.0, 50.0), area_frac=0.25,
                                            H_full=100, W_full=100)
    assert side == 50


def test_centered_area_square_clipped_inside_image():
    # Centroid near the edge -> shift inside, don't crop out
    top, left, side = centered_area_square((5.0, 5.0), area_frac=0.5,
                                            H_full=100, W_full=100)
    assert top >= 0 and left >= 0
    assert top + side <= 100 and left + side <= 100


# ── gt_bbox_and_centroid ──────────────────────────────────────────────────


def test_gt_bbox_and_centroid_basic():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True  # rows 2-4 (h=3), cols 3-6 (w=4)
    bbox, cent = gt_bbox_and_centroid(mask)
    assert bbox == (2, 3, 3, 4)
    # centroid is mean of True coords
    cy, cx = cent
    assert cy == 3.0  # mean of [2,2,2,2,3,3,3,3,4,4,4,4]
    assert cx == 4.5  # mean of [3,4,5,6] repeated


def test_gt_bbox_and_centroid_empty_raises():
    with pytest.raises(ValueError):
        gt_bbox_and_centroid(np.zeros((10, 10), dtype=bool))


# ── inferred_bbox_from_patches ────────────────────────────────────────────


def test_inferred_bbox_from_patches_empty_returns_none():
    g = np.zeros((8, 8), dtype=bool)
    assert inferred_bbox_from_patches(g, source_size=(800, 800),
                                       n_patch_per_side=8) is None


def test_inferred_bbox_from_patches_maps_to_source():
    g = np.zeros((8, 8), dtype=bool)
    g[1:3, 2:5] = True  # 2 rows, 3 cols on 8x8 grid
    out = inferred_bbox_from_patches(g, source_size=(800, 800), n_patch_per_side=8)
    assert out is not None
    top, left, h, w = out
    # Each patch is 100 source pixels (800 / 8). Rows 1-2 -> top ~100, h ~200.
    assert 80 < top < 120
    assert 180 < h < 220


# ── project_window_mask / window_footprint ────────────────────────────────


def test_window_footprint_full_window_covers_grid():
    meta = (0, 0, 448, 448)
    fp = window_footprint(meta, source_size=(448, 448), n_patch_per_side=14,
                          target_image_size=448)
    assert fp.shape == (14, 14)
    assert fp.all()


def test_project_window_mask_all_positive():
    """An all-true window mask projects to a non-empty target-grid region."""
    wm = np.ones((4, 4), dtype=bool)
    meta = (0, 0, 100, 100)
    out = project_window_mask(wm, meta, source_size=(100, 100),
                               n_patch_per_side=4, target_image_size=448)
    assert out.shape == (4, 4)
    assert out.any()


def test_project_window_mask_all_negative_stays_empty():
    wm = np.zeros((4, 4), dtype=bool)
    meta = (0, 0, 100, 100)
    out = project_window_mask(wm, meta, source_size=(100, 100),
                               n_patch_per_side=4, target_image_size=448)
    assert not out.any()
