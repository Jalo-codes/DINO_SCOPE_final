"""lab_utils.eval.zoom — Attention-guided zoom bbox and mask projection utilities.

Provides the canonical attention_zoom_bbox() that replaces the ~50-line inline
bbox blocks previously duplicated across eval_zoom_localization.py and
explore_zoom_viz.py.  Also contains pixel-space helpers for projecting patch
masks to full-resolution images.

Dependencies: NumPy only (no torch, no PIL) except patches_to_pixels and
place_crop_in_full_frame which require PIL for the NN resize.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .gap_utils import compute_gap_threshold, compute_otsu_threshold


def compute_hysteresis_mask(
    att_nn: np.ndarray,
    *,
    weak_floor_mads: float = 1.0,
) -> np.ndarray:
    """Two-level (Canny-style) hot mask: seed at the Otsu threshold, then grow
    4-connected patches above a weak threshold.

    Fixes the single-global-threshold failure where moderately-hot patches
    (e.g. the cooler arm of an L-shaped splice border) fall below Otsu and get
    clipped out of the bbox: any patch above the weak threshold that touches a
    strong seed is kept, while isolated warm patches elsewhere are not.

    The weak threshold is Otsu re-run on the sub-strong values, floored at
    median + weak_floor_mads * MAD so it never drops into the background noise.

    Returns an (n, n) bool mask (all-False when no patch clears the strong
    threshold).
    """
    att = np.asarray(att_nn, dtype=np.float64)
    flat = att.reshape(-1)
    strong = compute_otsu_threshold(flat)
    seeds = att >= strong
    if not seeds.any():
        return seeds

    below = flat[flat < strong]
    weak = compute_otsu_threshold(below) if len(below) >= 3 else strong
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    weak = min(max(weak, med + weak_floor_mads * mad), strong)
    cand = att >= weak

    grown = seeds.copy()
    while True:
        nxt = grown.copy()
        nxt[1:, :] |= grown[:-1, :]
        nxt[:-1, :] |= grown[1:, :]
        nxt[:, 1:] |= grown[:, :-1]
        nxt[:, :-1] |= grown[:, 1:]
        nxt &= cand
        if (nxt == grown).all():
            return grown
        grown = nxt


def attention_zoom_bbox(
    att_nn: np.ndarray,
    H: int,
    W: int,
    *,
    base_padding: int = 2,
    pad_frac: float = 0.15,
    min_crop_patches: int = 8,
    thresh_mode: str = 'gap',
) -> Optional[Tuple[int, int, int, int]]:
    """Compute a zoom bbox from an (n, n) attention map.

    1. Threshold the attention to find the hot region ('gap' = largest single
       gap, biased toward the very-hot peak; 'otsu' = 2-means hot/cold
       partition, hot cluster includes hot AND very-hot patches).
    2. Take the tight bbox of all hot patches (guarantees every predicted hot
       patch is inside the returned crop).
    3. Add area-aware padding: pad = max(base_padding, ceil(n * pad_frac / detected_side))
       — small detections receive proportionally more context than large ones.
    4. Clamp to [0, n-1].
    5. If still smaller than min_crop_patches on either axis, expand symmetrically
       from the center of the detected region and clamp again.
    6. Map patch coordinates to source-image pixels.

    Args:
        att_nn:           (n, n) float attention map (any scale; only relative values matter)
        H, W:             source image height and width in pixels
        base_padding:     absolute minimum patches added on every side (default 2)
        pad_frac:         scales padding inversely with detected region size (default 0.15);
                          larger → more context around small detections
        min_crop_patches: minimum side length of the returned crop in patches (default 8)
        thresh_mode:      'gap' (default, back-compat), 'otsu', or 'hyst'
                          (hysteresis: Otsu seeds grown 4-connected over a weak
                          threshold — see compute_hysteresis_mask)

    Returns:
        (x_min, y_min, x_max, y_max) in source-pixel coordinates, or None if the
        threshold finds no clear hot region.
    """
    att_nn = np.asarray(att_nn, dtype=np.float64)
    n = att_nn.shape[0]
    att_flat = att_nn.reshape(-1)

    if thresh_mode == 'hyst':
        hot = compute_hysteresis_mask(att_nn)
    elif thresh_mode == 'otsu':
        hot = att_nn >= compute_otsu_threshold(att_flat)
    else:
        hot = att_nn >= compute_gap_threshold(att_flat)
    rows, cols = np.where(hot)
    if len(rows) == 0:
        return None

    # ── Tight bbox (includes ALL hot patches) ────────────────────────────────
    r0_tight, r1_tight = int(rows.min()), int(rows.max())
    c0_tight, c1_tight = int(cols.min()), int(cols.max())

    detected_h = r1_tight - r0_tight + 1
    detected_w = c1_tight - c0_tight + 1
    detected_side = max(detected_h, detected_w)

    # ── Area-aware padding ───────────────────────────────────────────────────
    pad = max(base_padding, math.ceil(n * pad_frac / max(detected_side, 1)))

    r0 = max(0, r0_tight - pad)
    r1 = min(n - 1, r1_tight + pad)
    c0 = max(0, c0_tight - pad)
    c1 = min(n - 1, c1_tight + pad)

    # ── Enforce minimum crop size ─────────────────────────────────────────────
    # Expand symmetrically from the center of the original detection.
    r_cent = 0.5 * (r0_tight + r1_tight)
    c_cent = 0.5 * (c0_tight + c1_tight)

    if (r1 - r0 + 1) < min_crop_patches:
        half = min_crop_patches // 2
        r0 = max(0, int(round(r_cent)) - half)
        r1 = min(n - 1, r0 + min_crop_patches - 1)
        if r1 == n - 1:
            r0 = max(0, r1 - min_crop_patches + 1)

    if (c1 - c0 + 1) < min_crop_patches:
        half = min_crop_patches // 2
        c0 = max(0, int(round(c_cent)) - half)
        c1 = min(n - 1, c0 + min_crop_patches - 1)
        if c1 == n - 1:
            c0 = max(0, c1 - min_crop_patches + 1)

    # ── Map patch coordinates to source pixels ────────────────────────────────
    x_min = max(0, int(c0 * W / n))
    x_max = min(W, int((c1 + 1) * W / n))
    y_min = max(0, int(r0 * H / n))
    y_max = min(H, int((r1 + 1) * H / n))

    if (x_max - x_min) < 2 or (y_max - y_min) < 2:
        return None

    return (x_min, y_min, x_max, y_max)


def patches_to_pixels(mask_nn: np.ndarray, H: int, W: int) -> np.ndarray:
    """NN-upsample an (n, n) bool patch mask to (H, W) pixel resolution."""
    img = Image.fromarray(mask_nn.astype(np.uint8) * 255, mode='L')
    return np.asarray(img.resize((W, H), Image.NEAREST)) > 127


def place_crop_in_full_frame(
    crop_mask_nn: np.ndarray,
    bbox: Tuple[int, int, int, int],
    H: int,
    W: int,
) -> np.ndarray:
    """Map a zoom-crop's (n, n) bool mask back to the full (H, W) image frame.

    Args:
        crop_mask_nn: (n, n) bool mask in zoom-crop patch coordinates
        bbox:         (x_min, y_min, x_max, y_max) pixel bbox of the zoom crop
                      in the full image
        H, W:         full image height and width in pixels

    Returns:
        (H, W) bool mask with the crop region filled in, zeros elsewhere.
    """
    x_min, y_min, x_max, y_max = bbox
    th = y_max - y_min
    tw = x_max - x_min
    crop_pil = Image.fromarray(crop_mask_nn.astype(np.uint8) * 255, mode='L')
    crop_px = np.asarray(crop_pil.resize((tw, th), Image.NEAREST)) > 127
    full = np.zeros((H, W), dtype=bool)
    full[y_min:y_max, x_min:x_max] = crop_px
    return full
