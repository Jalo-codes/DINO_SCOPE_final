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
from typing import List, Optional, Tuple

import numpy as np

from .gap_utils import compute_gap_threshold, compute_otsu_threshold
from .partition import _union_find_components


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


def get_padded_bbox(
    r0_tight: int,
    r1_tight: int,
    c0_tight: int,
    c1_tight: int,
    n: int,
    H: int,
    W: int,
    *,
    base_padding: int = 2,
    pad_frac: float = 0.15,
    min_crop_patches: int = 8,
) -> Optional[Tuple[int, int, int, int]]:
    """Factored-out padding, clamp, and expansion block of attention_zoom_bbox.
    Maps patch bounding box to pixel bounding box."""
    detected_h = r1_tight - r0_tight + 1
    detected_w = c1_tight - c0_tight + 1
    detected_side = max(detected_h, detected_w)

    pad = max(base_padding, math.ceil(n * pad_frac / max(detected_side, 1)))

    r0 = max(0, r0_tight - pad)
    r1 = min(n - 1, r1_tight + pad)
    c0 = max(0, c0_tight - pad)
    c1 = min(n - 1, c1_tight + pad)

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

    x_min = max(0, int(c0 * W / n))
    x_max = min(W, int((c1 + 1) * W / n))
    y_min = max(0, int(r0 * H / n))
    y_max = min(H, int((r1 + 1) * H / n))

    if (x_max - x_min) < 2 or (y_max - y_min) < 2:
        return None

    return (x_min, y_min, x_max, y_max)


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

    r0_tight, r1_tight = int(rows.min()), int(rows.max())
    c0_tight, c1_tight = int(cols.min()), int(cols.max())

    return get_padded_bbox(
        r0_tight, r1_tight, c0_tight, c1_tight, n, H, W,
        base_padding=base_padding, pad_frac=pad_frac,
        min_crop_patches=min_crop_patches
    )


def multi_zoom_bboxes(
    att_nn: np.ndarray,
    H_px: int,
    W_px: int,
    *,
    max_regions: int = 3,
    theta_fill: float = 0.45,
    base_padding: int = 2,
    pad_frac: float = 0.15,
    min_crop_patches: int = 8,
    thresh_mode: str = 'gap',
    hot_mask: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, int, int]]:
    """Determine multiple zoom bboxes for scenes with multiple disjoint hot regions."""
    from .partition import _union_find_components
    
    att_nn = np.asarray(att_nn, dtype=np.float64)
    n = att_nn.shape[0]
    
    if hot_mask is not None:
        H = np.asarray(hot_mask, dtype=bool)
    else:
        att_flat = att_nn.reshape(-1)
        if thresh_mode == 'hyst':
            H = compute_hysteresis_mask(att_nn)
        elif thresh_mode == 'otsu':
            H = att_nn >= compute_otsu_threshold(att_flat)
        else:
            H = att_nn >= compute_gap_threshold(att_flat)

    rows_hot, cols_hot = np.where(H)
    if len(rows_hot) == 0:
        return []

    # 1. Connected components (8-connected)
    N = n * n
    adj = np.zeros((N, N), dtype=bool)
    hot_indices = rows_hot * n + cols_hot
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr = rows_hot + dr
            nc = cols_hot + dc
            valid = (nr >= 0) & (nr < n) & (nc >= 0) & (nc < n)
            for idx, r_n, c_n in zip(hot_indices[valid], nr[valid], nc[valid]):
                if H[r_n, c_n]:
                    adj[idx, r_n * n + c_n] = True

    labels = _union_find_components(adj)
    
    # Identify unique components that belong to hot patches
    unique_labels = np.unique(labels[hot_indices])
    blobs = []
    for lbl in unique_labels:
        comp_idx = np.where((labels == lbl) & H.ravel())[0]
        # Drop blobs < m_min (4 patches)
        if len(comp_idx) >= 4:
            blobs.append(comp_idx)

    # 2. Valley split
    final_blobs = []
    for comp_idx in blobs:
        rs = comp_idx // n
        cs = comp_idx % n
        r0_t, r1_t = int(rs.min()), int(rs.max())
        c0_t, c1_t = int(cs.min()), int(cs.max())
        h = r1_t - r0_t + 1
        w = c1_t - c0_t + 1
        side = max(h, w)
        
        did_split = False
        if side > 0.5 * n:
            area = h * w
            fill_frac = len(comp_idx) / float(area) if area > 0 else 0.0
            if fill_frac < theta_fill:
                # Perpendicular to the longer axis:
                if w >= h:
                    best_c = -1
                    min_mass = len(comp_idx) + 1
                    best_dist = n
                    for c_val in range(c0_t + 1, c1_t):
                        mass = int(np.sum(cs == c_val))
                        dist = abs(c_val - (c0_t + c1_t) / 2.0)
                        if (mass < min_mass) or (mass == min_mass and dist < best_dist):
                            min_mass = mass
                            best_c = c_val
                            best_dist = dist
                    
                    if best_c != -1 and min_mass == 0:
                        sub1 = comp_idx[cs < best_c]
                        sub2 = comp_idx[cs > best_c]
                        if len(sub1) >= 4 and len(sub2) >= 4:
                            final_blobs.append(sub1)
                            final_blobs.append(sub2)
                            did_split = True
                else:
                    best_r = -1
                    min_mass = len(comp_idx) + 1
                    best_dist = n
                    for r_val in range(r0_t + 1, r1_t):
                        mass = int(np.sum(rs == r_val))
                        dist = abs(r_val - (r0_t + r1_t) / 2.0)
                        if (mass < min_mass) or (mass == min_mass and dist < best_dist):
                            min_mass = mass
                            best_r = r_val
                            best_dist = dist
                            
                    if best_r != -1 and min_mass == 0:
                        sub1 = comp_idx[rs < best_r]
                        sub2 = comp_idx[rs > best_r]
                        if len(sub1) >= 4 and len(sub2) >= 4:
                            final_blobs.append(sub1)
                            final_blobs.append(sub2)
                            did_split = True
        if not did_split:
            final_blobs.append(comp_idx)

    # 3. Merging: compute padded bboxes and merge if IoU > 0.3
    def calc_iou(b1, b2):
        x1, y1, x2, y2 = b1
        X1, Y1, X2, Y2 = b2
        inter_w = max(0, min(x2, X2) - max(x1, X1))
        inter_h = max(0, min(y2, Y2) - max(y1, Y1))
        inter = inter_w * inter_h
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (X2 - X1) * (Y2 - Y1)
        union = area1 + area2 - inter
        return inter / float(union) if union > 0 else 0.0

    while True:
        bboxes = []
        valid_indices = []
        for i, comp_idx in enumerate(final_blobs):
            rs = comp_idx // n
            cs = comp_idx % n
            r0_t, r1_t = int(rs.min()), int(rs.max())
            c0_t, c1_t = int(cs.min()), int(cs.max())
            bbox = get_padded_bbox(
                r0_t, r1_t, c0_t, c1_t, n, H_px, W_px,
                base_padding=base_padding, pad_frac=pad_frac,
                min_crop_patches=min_crop_patches
            )
            if bbox is not None:
                bboxes.append(bbox)
                valid_indices.append(i)
        
        final_blobs = [final_blobs[i] for i in valid_indices]
        
        merged_any = False
        num_blobs = len(final_blobs)
        for i in range(num_blobs):
            for j in range(i + 1, num_blobs):
                if calc_iou(bboxes[i], bboxes[j]) > 0.3:
                    merged_comp = np.union1d(final_blobs[i], final_blobs[j])
                    final_blobs[i] = merged_comp
                    final_blobs.pop(j)
                    merged_any = True
                    break
            if merged_any:
                break
        
        if not merged_any:
            break

    # 4. Fire condition
    if len(final_blobs) < 2:
        single_bbox = attention_zoom_bbox(
            att_nn, H_px, W_px,
            base_padding=base_padding, pad_frac=pad_frac,
            min_crop_patches=min_crop_patches, thresh_mode=thresh_mode
        )
        return [single_bbox] if single_bbox is not None else []

    # 5. Cap: keep top K=3 by hot-mass (attention sum)
    def blob_mass(comp_idx):
        return float(att_nn.ravel()[comp_idx].sum())

    final_blobs = sorted(final_blobs, key=blob_mass, reverse=True)[:max_regions]

    res_bboxes = []
    for comp_idx in final_blobs:
        rs = comp_idx // n
        cs = comp_idx % n
        r0_t, r1_t = int(rs.min()), int(rs.max())
        c0_t, c1_t = int(cs.min()), int(cs.max())
        bbox = get_padded_bbox(
            r0_t, r1_t, c0_t, c1_t, n, H_px, W_px,
            base_padding=base_padding, pad_frac=pad_frac,
            min_crop_patches=min_crop_patches
        )
        if bbox is not None:
            res_bboxes.append(bbox)
            
    return res_bboxes


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
