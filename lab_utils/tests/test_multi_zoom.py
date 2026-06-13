"""Tests for multi-zoom bboxes computation in zoom.py."""

import os
import sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lab_utils.eval.zoom import multi_zoom_bboxes, get_padded_bbox


def test_get_padded_bbox():
    # Test that get_padded_bbox handles base padding and min crop patches constraints.
    bbox = get_padded_bbox(
        r0_tight=5, r1_tight=5, c0_tight=5, c1_tight=5,
        n=16, H=256, W=256,
        base_padding=2, pad_frac=0.15, min_crop_patches=8
    )
    assert bbox is not None
    x_min, y_min, x_max, y_max = bbox
    # Since n=16, H=256, W=256, each patch is 16x16 pixels.
    # Tight bbox of (5, 5, 5, 5) gets padded. 
    # Side is 1. pad = max(2, ceil(16 * 0.15 / 1)) = max(2, 3) = 3.
    # Padded bounds: r0 = 5 - 3 = 2, r1 = 5 + 3 = 8
    # Height = 8 - 2 + 1 = 7, which is < min_crop_patches (8).
    # Expanding symmetrically: cent_r = 5.
    # half = 8 // 2 = 4. r0 = 5 - 4 = 1. r1 = 1 + 8 - 1 = 8.
    # Same for cols: c0 = 1, c1 = 8.
    # Pixels:
    # x_min = 1 * 16 = 16
    # x_max = (8+1) * 16 = 144
    # y_min = 1 * 16 = 16
    # y_max = (8+1) * 16 = 144
    assert x_min == 16
    assert y_min == 16
    assert x_max == 144
    assert y_max == 144


def test_components_8_connected_and_overlap_merge():
    # Grid of size 16x16.
    # Create two components that are close and will overlap after padding.
    n = 16
    hot_mask = np.zeros((n, n), dtype=bool)
    
    # Component 1 (size >= 4)
    hot_mask[2:4, 2:4] = True
    # Component 2 (size >= 4)
    hot_mask[5:7, 5:7] = True
    
    att = np.zeros((n, n), dtype=float)
    att[hot_mask] = 1.0
    
    # 1. First run with high padding (causes overlap, so they merge)
    bboxes_merged = multi_zoom_bboxes(
        att, 256, 256,
        max_regions=3, theta_fill=0.45,
        base_padding=3, pad_frac=0.25, min_crop_patches=8,
        thresh_mode='otsu', hot_mask=hot_mask
    )
    # They should merge, producing 1 final bbox.
    assert len(bboxes_merged) == 1
    
    # 2. Run with zero padding and tiny min_crop_patches (no overlap, so they remain disjoint)
    bboxes_disjoint = multi_zoom_bboxes(
        att, 256, 256,
        max_regions=3, theta_fill=0.45,
        base_padding=0, pad_frac=0.0, min_crop_patches=2,
        thresh_mode='otsu', hot_mask=hot_mask
    )
    # They should not merge, producing 2 final bboxes.
    assert len(bboxes_disjoint) == 2


def test_capping_and_determinism():
    n = 16
    hot_mask = np.zeros((n, n), dtype=bool)
    
    # Create 4 disjoint components, each size 4.
    for i in range(4):
        r = 2 + i * 3
        c = 2
        hot_mask[r:r+2, c:c+2] = True
        
    att = np.zeros((n, n), dtype=float)
    # Set different masses (attention sums) for each component.
    # We want component 3 to have the highest mass, then 2, then 1, then 0.
    att[2:4, 2:4] = 1.0     # Mass = 4.0
    att[5:7, 2:4] = 2.0     # Mass = 8.0
    att[8:10, 2:4] = 3.0    # Mass = 12.0
    att[11:13, 2:4] = 4.0   # Mass = 16.0
    
    # Compute bboxes with max_regions=3.
    # Use zero padding so they don't merge.
    bboxes = multi_zoom_bboxes(
        att, 256, 256,
        max_regions=3, theta_fill=0.45,
        base_padding=0, pad_frac=0.0, min_crop_patches=2,
        thresh_mode='otsu', hot_mask=hot_mask
    )
    
    # Should cap at 3 regions
    assert len(bboxes) == 3
    
    # Determinism check: multiple runs should give identical results.
    bboxes_second = multi_zoom_bboxes(
        att, 256, 256,
        max_regions=3, theta_fill=0.45,
        base_padding=0, pad_frac=0.0, min_crop_patches=2,
        thresh_mode='otsu', hot_mask=hot_mask
    )
    assert bboxes == bboxes_second
