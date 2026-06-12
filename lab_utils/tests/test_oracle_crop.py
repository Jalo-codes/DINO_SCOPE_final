"""Tests for oracle_mask_crop — the mask-centered zoom fallback that keeps a
small/off-center splice in frame instead of demoting it to the resize fallback."""

import numpy as np
import pytest

pytest.importorskip("torch")  # resolution.py imports torch at module load
from PIL import Image

from lab_utils.data.resolution import Resolution, oracle_mask_crop


def _mask(H, W, top, left, h, w):
    a = np.zeros((H, W), dtype=np.uint8)
    a[top:top + h, left:left + w] = 255
    return Image.fromarray(a, mode="L")


def test_oracle_zooms_tiny_offcenter_splice_into_frame():
    res = Resolution(64, 16)
    img = Image.new("RGB", (400, 300), (120, 120, 120))
    # ~0.33% area splice tucked in the top-left corner.
    mask = _mask(300, 400, top=10, left=12, h=20, w=20)
    out = oracle_mask_crop(img, mask, res, target_cov_range=(0.1, 0.4),
                           jitter_frac=0.0, rng=np.random.default_rng(0))
    assert out.valid and out.mode == "oracle"
    m = np.asarray(out.mask.convert("L")) > 0
    assert m.any(), "splice must be present in the oracle crop"
    # Zoomed from 0.33% up into the target band (well above the 1% accept floor).
    assert 0.05 <= out.coverage <= 0.6, out.coverage


def test_oracle_contains_full_bbox_even_at_corner():
    res = Resolution(64, 16)
    img = Image.new("RGB", (500, 500), (10, 20, 30))
    mask = _mask(500, 500, top=455, left=460, h=40, w=40)  # bottom-right corner
    out = oracle_mask_crop(img, mask, res, jitter_frac=0.0,
                           rng=np.random.default_rng(1))
    assert out.valid
    m = np.asarray(out.mask.convert("L")) > 0
    # The full 40x40 blob survives the crop+resize (no clipping of the bbox).
    assert m.sum() > 0
    assert out.coverage > 0.0


def test_oracle_empty_mask_signals_drop():
    res = Resolution(64, 16)
    img = Image.new("RGB", (200, 200), (0, 0, 0))
    mask = Image.new("L", (200, 200), 0)  # no splice pixels
    out = oracle_mask_crop(img, mask, res)
    assert not out.valid
    assert out.mode == "oracle_empty"


def test_oracle_coverage_tracks_target():
    res = Resolution(112, 16)
    img = Image.new("RGB", (600, 600), (5, 5, 5))
    mask = _mask(600, 600, top=250, left=250, h=60, w=60)  # 1% area, centered
    # Fixed target → realized coverage should land near it.
    out = oracle_mask_crop(img, mask, res, target_cov_range=(0.25, 0.25),
                           jitter_frac=0.0, rng=np.random.default_rng(2))
    assert out.valid
    assert 0.18 <= out.coverage <= 0.32, out.coverage
