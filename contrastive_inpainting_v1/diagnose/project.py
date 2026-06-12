"""28x28 patch grid + source-pixel bbox  →  (H, W) original-resolution mask.

NN-expansion: each of the n*n patches becomes a contiguous pixel block inside
the bbox in the original image.  Pixels outside the bbox are always False.

This is the single function that bridges "model's patch-level prediction" and
"pixel-level F1 against original GT".  All passes call it on their predictions
before metrics.f1_pixel runs.

The implicit bias for crop-based passes is intentional and documented (see
the v2 design contract in `diagnose/__init__.py`): patches inside a small
crop project to small pixel blocks (fine granularity), patches outside the
crop are False at one big block (coarse zero).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def patch_grid_to_pixel_mask(
    patch_pred: np.ndarray,
    *,
    bbox: Tuple[int, int, int, int],
    full_size: Tuple[int, int],
) -> np.ndarray:
    """Project an (n, n) bool patch prediction to a (H, W) pixel mask.

    Args:
        patch_pred: (n, n) bool array. The model's per-patch decision.
        bbox: (top, left, h, w) of the region the patch grid corresponds to
              in the ORIGINAL image. For full pass this is (0, 0, H, W).
              For gtcrop pass this is the square crop's source-pixel coords.
              For swin pass this is a single window's source-pixel coords.
        full_size: (H, W) of the original image.

    Returns:
        (H, W) bool mask.  The bbox region is NN-expanded from patch_pred;
        pixels outside the bbox are False.
    """
    pp = np.asarray(patch_pred, dtype=bool)
    if pp.ndim != 2 or pp.shape[0] != pp.shape[1]:
        raise ValueError(f"patch_pred must be (n, n) bool; got shape {pp.shape}")
    n = pp.shape[0]
    top, left, h, w = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    H, W = int(full_size[0]), int(full_size[1])
    if h <= 0 or w <= 0:
        return np.zeros((H, W), dtype=bool)

    # Clip the bbox to the image — defensive; callers should already ensure this.
    if top < 0:
        h += top
        top = 0
    if left < 0:
        w += left
        left = 0
    if top + h > H:
        h = H - top
    if left + w > W:
        w = W - left
    if h <= 0 or w <= 0:
        return np.zeros((H, W), dtype=bool)

    # NN expansion: each pixel (i, j) inside the bbox maps to patch
    # (i * n // h, j * n // w).
    row_idx = np.minimum((np.arange(h) * n // max(h, 1)).astype(np.int64), n - 1)
    col_idx = np.minimum((np.arange(w) * n // max(w, 1)).astype(np.int64), n - 1)
    block = pp[row_idx[:, None], col_idx[None, :]]
    out = np.zeros((H, W), dtype=bool)
    out[top:top + h, left:left + w] = block
    return out


def patch_grid_or(
    masks: np.ndarray,
) -> np.ndarray:
    """Logical OR of a stack of (H, W) bool masks. Returns (H, W) bool."""
    if masks is None or len(masks) == 0:
        return None
    arr = np.asarray(masks, dtype=bool)
    if arr.ndim == 2:
        return arr.copy()
    return np.logical_or.reduce(arr, axis=0)


def gt_pixel_mask(mask_image: Image.Image, *, full_size: Tuple[int, int]) -> np.ndarray:
    """Load the source mask as a (H, W) bool pixel mask at original resolution.

    Caller passes the already-opened PIL mask; we just convert. Single source
    of truth for "what counts as a splice pixel" (anything >0 in the mask).
    """
    arr = np.asarray(mask_image, dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    H_expected, W_expected = int(full_size[0]), int(full_size[1])
    if arr.shape != (H_expected, W_expected):
        # Resize NN to match source if needed (some masks are stored at a
        # different resolution than the source image).
        m = mask_image
        if m.mode != "L":
            m = m.convert("L")
        from torchvision.transforms.functional import resize as _resize
        m = _resize(m, [H_expected, W_expected], interpolation=Image.NEAREST)
        arr = np.asarray(m, dtype=np.uint8)
    return (arr > 0).astype(bool)


def in_pixel_splice_frac(
    bbox: Tuple[int, int, int, int],
    *,
    gt_HW: np.ndarray,
) -> float:
    """Fraction of pixels inside `bbox` that are GT splice. Used as a per-pass
    diagnostic (in_crop_splice_frac, in_window_splice_frac).
    """
    top, left, h, w = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    H, W = gt_HW.shape
    top = max(0, min(top, H))
    left = max(0, min(left, W))
    bot = max(top, min(top + h, H))
    right = max(left, min(left + w, W))
    sub = gt_HW[top:bot, left:right]
    if sub.size == 0:
        return 0.0
    return float(sub.mean())
