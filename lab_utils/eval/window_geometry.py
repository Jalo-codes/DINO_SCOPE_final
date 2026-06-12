"""lab_utils.eval.window_geometry — pure geometry for sliding-window eval.

Source-pixel ↔ patch-grid coordinate transforms and bbox/crop helpers shared
by the diagnose / sweep scripts.  Pure NumPy + math: no model state, no
DataLoader, no logger.  ``window_gt_patches`` is the only function that
needs PIL/torchvision (for the mask resize) — everything else is plain
NumPy.

This module supersedes the local copies that previously lived in
:mod:`contrastive_inpainting_v1.scripts.diagnose` and the
shadow versions in :mod:`contrastive_inpainting_v1.diagnose.passes`.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Optional, Tuple

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Sliding-window grid
# ---------------------------------------------------------------------------


def axis_positions(length: int, side: int, stride_frac: float) -> List[int]:
    """Sliding-window start positions along one axis.

    Stride is ``max(1, round(side * stride_frac))``.  Always includes
    ``max_start = length - side`` so the last window is flush with the edge.
    """
    max_start = max(0, int(length) - int(side))
    if max_start == 0:
        return [0]
    stride = max(1, int(round(side * float(stride_frac))))
    pos = list(range(0, max_start + 1, stride))
    if pos[-1] != max_start:
        pos.append(max_start)
    return sorted(set(int(p) for p in pos))


def window_grid(
    source_size: Tuple[int, int],
    *,
    scale: float,
    stride_frac: float,
    n_patch_per_side: int,
) -> List[Tuple[int, int, int, int]]:
    """Generate ``(top, left, side, side)`` windows for the given source size.

    Returns every window the ``(scale, stride)`` combo produces — no cap.
    ``source_size`` is ``(W, H)`` (PIL convention).
    """
    W_src, H_src = int(source_size[0]), int(source_size[1])
    crop_side = max(int(n_patch_per_side),
                    int(round(min(H_src, W_src) * float(scale))))
    crop_side = min(crop_side, min(H_src, W_src))
    tops = axis_positions(H_src, crop_side, stride_frac)
    lefts = axis_positions(W_src, crop_side, stride_frac)
    return [(int(t), int(l), int(crop_side), int(crop_side))
            for t in tops for l in lefts]


def capped_window_grid(
    source_size: Tuple[int, int],
    *,
    scale: float,
    stride_frac: float,
    n_patch_per_side: int,
    max_windows: int,
) -> List[Tuple[int, int, int, int]]:
    """:func:`window_grid` post-hoc thinned to at most ``max_windows`` cells.

    Thins the longer axis first via linspace selection so the remaining
    positions stay roughly uniform across the image — used by the sweep
    scripts to bound forward-pass count per cell.
    """
    W_src, H_src = int(source_size[0]), int(source_size[1])
    crop_side = max(int(n_patch_per_side),
                    int(round(min(H_src, W_src) * float(scale))))
    crop_side = min(crop_side, min(H_src, W_src))
    tops = axis_positions(H_src, crop_side, stride_frac)
    lefts = axis_positions(W_src, crop_side, stride_frac)
    while len(tops) * len(lefts) > max_windows:
        if len(lefts) >= len(tops) and len(lefts) > 1:
            keep = max(1, int(max_windows) // len(tops))
            idx = np.linspace(0, len(lefts) - 1, num=keep)
            lefts = sorted(set(lefts[int(round(x))] for x in idx))
        elif len(tops) > 1:
            keep = max(1, int(max_windows) // len(lefts))
            idx = np.linspace(0, len(tops) - 1, num=keep)
            tops = sorted(set(tops[int(round(x))] for x in idx))
        else:
            break
    return [(int(t), int(l), int(crop_side), int(crop_side))
            for t in tops for l in lefts]


def window_set_hash(windows: List[Tuple[int, int, int, int]]) -> str:
    """Stable hash of a window-coord list (for scale-aliasing assertions)."""
    if not windows:
        return "empty"
    h = hashlib.md5()
    for t, l, s, _ in windows:
        h.update(f"{t},{l},{s};".encode("utf-8"))
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Square crops
# ---------------------------------------------------------------------------


def square_expand_crop(
    top: int, left: int, h: int, w: int, H_full: int, W_full: int,
) -> Tuple[int, int, int]:
    """Expand a rectangular crop to a square in source-pixel coords.

    Centers expansion on the shorter dim, clips to bounds, shifts (not
    truncates) when expansion would overflow.  Guarantees ``side > 0``,
    ``top + side <= H_full``, ``left + side <= W_full``.
    """
    target = max(int(h), int(w))
    target = min(target, int(H_full), int(W_full))
    extra_h = target - int(h)
    extra_w = target - int(w)
    new_top = int(top) - extra_h // 2
    new_left = int(left) - extra_w // 2
    if new_top < 0:
        new_top = 0
    if new_left < 0:
        new_left = 0
    if new_top + target > H_full:
        new_top = max(0, H_full - target)
    if new_left + target > W_full:
        new_left = max(0, W_full - target)
    return int(new_top), int(new_left), int(target)


def centered_area_square(
    centroid_yx: Tuple[float, float],
    *,
    area_frac: float,
    H_full: int,
    W_full: int,
) -> Tuple[int, int, int]:
    """Square crop of side ≈ ``sqrt(area_frac * H * W)`` centered on ``centroid_yx``.

    Side is interpreted as ``side² / (H*W) = area_frac`` — true image-area
    fraction, not min-dim-squared fraction.  Side capped at ``min(H, W)``.
    """
    a = float(np.clip(area_frac, 1e-6, 1.0))
    side = int(round(np.sqrt(a * float(H_full) * float(W_full))))
    side = max(8, min(side, int(H_full), int(W_full)))
    cy, cx = float(centroid_yx[0]), float(centroid_yx[1])
    top = int(round(cy - side / 2.0))
    left = int(round(cx - side / 2.0))
    if top < 0:
        top = 0
    if left < 0:
        left = 0
    if top + side > H_full:
        top = max(0, H_full - side)
    if left + side > W_full:
        left = max(0, W_full - side)
    return int(top), int(left), int(side)


# ---------------------------------------------------------------------------
# GT bbox / centroid (for gtcrop) and patch-grid → source bbox
# ---------------------------------------------------------------------------


def gt_bbox_and_centroid(
    gt_HW: np.ndarray,
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
    """Return ``((top, left, h, w), (cy, cx))`` for a ``(H, W)`` bool mask.

    Centroid is the mean of True-pixel coords (not bbox center) — more
    faithful to mass for irregular splice shapes.  Raises if mask is empty.
    """
    g = np.asarray(gt_HW, dtype=bool)
    if not g.any():
        raise ValueError("gt_bbox_and_centroid called on empty mask")
    rows = np.where(g.any(axis=1))[0]
    cols = np.where(g.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    ys, xs = np.where(g)
    cy = float(ys.mean())
    cx = float(xs.mean())
    return ((r0, c0, r1 - r0, c1 - c0), (cy, cx))


def inferred_bbox_from_patches(
    pred_grid: np.ndarray,
    *,
    source_size: Tuple[int, int],
    n_patch_per_side: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of True patches mapped from patch-grid → source pixels.

    ``pred_grid``: ``(n, n)`` bool.  ``source_size`` is ``(W, H)``.  Returns
    ``(top, left, h, w)`` in source coords or ``None`` if the grid is empty.
    """
    g = np.asarray(pred_grid, dtype=bool)
    if not g.any():
        return None
    W_src, H_src = source_size
    n = int(n_patch_per_side)
    rows = np.where(g.any(axis=1))[0]
    cols = np.where(g.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    top = int(round(r0 * H_src / n))
    bot = int(round(r1 * H_src / n))
    left = int(round(c0 * W_src / n))
    right = int(round(c1 * W_src / n))
    top = max(0, min(top, H_src - 1))
    bot = max(top + 1, min(bot, H_src))
    left = max(0, min(left, W_src - 1))
    right = max(left + 1, min(right, W_src))
    return (top, left, bot - top, right - left)


# ---------------------------------------------------------------------------
# Window-mask projection (patch-grid in window-coords → target-grid in
# image-coords).  Used by the swin sweep to OR per-window predictions.
# ---------------------------------------------------------------------------


def project_window_mask(
    wm: np.ndarray,
    meta: Tuple[int, int, int, int],
    *,
    source_size: Tuple[int, int],
    n_patch_per_side: int,
    target_image_size: int,
) -> np.ndarray:
    """Project a window's ``(n, n)`` patch-grid bool mask back to the
    target image's ``(n, n)`` patch grid.

    ``meta`` is ``(top, left, side, _)`` in source-pixel coords.
    ``source_size`` is ``(W, H)``.  Returns ``(n, n)`` bool array.
    """
    W_src, H_src = source_size
    top, left, s, _ = meta
    n = int(n_patch_per_side)
    T = int(target_image_size)
    target_patch_size = T // n
    win_patch_size_src = s / float(n)
    out = np.zeros((n, n), dtype=np.bool_)

    tgt_top_pix = top * T / float(H_src)
    tgt_bot_pix = (top + s) * T / float(H_src)
    tgt_left_pix = left * T / float(W_src)
    tgt_right_pix = (left + s) * T / float(W_src)
    tgt_top_patch = max(0, int(math.floor(tgt_top_pix / target_patch_size)))
    tgt_bot_patch = min(n, int(math.ceil(tgt_bot_pix / target_patch_size)))
    tgt_left_patch = max(0, int(math.floor(tgt_left_pix / target_patch_size)))
    tgt_right_patch = min(n, int(math.ceil(tgt_right_pix / target_patch_size)))

    for ti in range(tgt_top_patch, tgt_bot_patch):
        src_y = (ti + 0.5) * target_patch_size * H_src / float(T)
        wi = int((src_y - top) / win_patch_size_src)
        if wi < 0 or wi >= n:
            continue
        for tj in range(tgt_left_patch, tgt_right_patch):
            src_x = (tj + 0.5) * target_patch_size * W_src / float(T)
            wj = int((src_x - left) / win_patch_size_src)
            if 0 <= wj < n and wm[wi, wj]:
                out[ti, tj] = True
    return out


def window_footprint(
    meta: Tuple[int, int, int, int],
    *,
    source_size: Tuple[int, int],
    n_patch_per_side: int,
    target_image_size: int,
) -> np.ndarray:
    """``(n, n)`` bool mask of which target-grid patches lie inside the window."""
    all_positive = np.ones((n_patch_per_side, n_patch_per_side), dtype=np.bool_)
    return project_window_mask(
        all_positive,
        meta,
        source_size=source_size,
        n_patch_per_side=n_patch_per_side,
        target_image_size=target_image_size,
    )


def window_gt_patches(
    source_mask: Image.Image,
    meta: Tuple[int, int, int, int],
    *,
    target_image_size: int,
    n_patch_per_side: int,
    threshold: float = 0.06,
) -> np.ndarray:
    """Per-patch GT for a single window crop, in the window's own ``n×n`` grid.

    Crops the source mask at the window's source-pixel coords, resizes to
    ``T×T`` (NEAREST), then average-pools to ``n×n`` coverage and thresholds.
    ``threshold`` matches the harness-wide ``--gt_patch_threshold`` default.
    """
    top, left, s, _ = meta
    T = int(target_image_size)
    n = int(n_patch_per_side)
    crop = source_mask.crop((left, top, left + s, top + s))
    crop = TF.resize(crop, [T, T], interpolation=Image.NEAREST)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    patch = T // n
    avg = arr.reshape(n, patch, n, patch).mean(axis=(1, 3))
    return (avg > float(threshold)).astype(np.bool_)
