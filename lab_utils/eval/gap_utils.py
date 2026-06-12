"""lab_utils.eval.gap_utils — Gap-based thresholding and outlier scoring.

Pure NumPy utilities shared by zoom bbox selection, mask generation, and
outlier-score decoding.  No model state, no PIL, no torch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_gap_threshold(values: np.ndarray) -> float:
    """Find the largest gap in the sorted values (full range) and return its midpoint.

    Used for zoom bbox selection (find the gap in attention) and for outlier-score
    mask generation where you want to threshold by natural cluster separation rather
    than a fixed percentile.
    """
    flat = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if len(flat) < 3:
        return float(flat.max()) + 1.0
    diffs = np.diff(flat)
    gi = int(np.argmax(diffs))
    return float(0.5 * (flat[gi] + flat[gi + 1]))


def compute_otsu_threshold(values: np.ndarray) -> float:
    """Otsu's threshold (1D 2-means): split values into two clusters maximizing
    between-class variance, return the midpoint at the best split.

    Unlike compute_gap_threshold (largest single gap, which on attention maps
    often lands between 'hot' and 'very hot', keeping only the peak patches),
    Otsu weights both cluster masses — so the hot cluster includes hot AND
    very-hot patches and the threshold lands at the hot/cold partition.
    """
    flat = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    n = len(flat)
    if n < 3:
        return float(flat.max()) + 1.0
    csum = np.cumsum(flat)
    total = csum[-1]
    i = np.arange(1, n)                      # split: flat[:i] cold | flat[i:] hot
    m0 = csum[:-1] / i
    m1 = (total - csum[:-1]) / (n - i)
    var_between = i * (n - i) * (m0 - m1) ** 2
    gi = int(np.argmax(var_between))
    return float(0.5 * (flat[gi] + flat[gi + 1]))


def compute_gap_prediction(score: np.ndarray) -> np.ndarray:
    """Threshold score array by the largest gap in the upper half (above median).

    The upper-half restriction biases the threshold toward small, high-scoring
    regions — intentionally good for small splices where a few outlier patches
    sit well above the background.  For large splices (>50% area) the threshold
    tends to land inside the splice distribution; use k-means in that regime.

    Returns a bool array with the same shape as score.
    """
    flat = np.sort(np.asarray(score, dtype=np.float64).reshape(-1))
    med = np.median(flat)
    upper = flat[flat >= med]
    if len(upper) < 3:
        return np.zeros_like(score, dtype=bool)
    diffs = np.diff(upper)
    gi = int(np.argmax(diffs))
    gap_thr = float(0.5 * (upper[gi] + upper[gi + 1]))
    return score >= gap_thr


def compute_outlier_score(
    z_np: np.ndarray,
    att_flat: Optional[np.ndarray],
    mode: str,
) -> np.ndarray:
    """Prototype-based outlier score: 1 - cosine_similarity_to_background_mean.

    The background prototype is constructed from patches identified as
    background via the attention map:
      'median' — patches with attention <= median attention
      'gap'    — patches with attention below the full-range gap threshold

    If att_flat is None or mismatched in length, all patches form the prototype
    (degrades gracefully, though the score loses its background-vs-splice meaning).

    Args:
        z_np:     (N, D) float array of L2-normalized patch embeddings
        att_flat: (N,) float array of per-patch attention weights, or None
        mode:     'median' or 'gap'

    Returns:
        (N,) float array of outlier scores in [0, 2] (typically [0, ~1])
    """
    zz = z_np / (np.linalg.norm(z_np, axis=1, keepdims=True) + 1e-8)
    if att_flat is not None and len(att_flat) == len(zz):
        if mode == 'gap':
            thresh = compute_gap_threshold(att_flat)
            bg = zz[att_flat < thresh]
        else:  # 'median'
            bg = zz[att_flat <= np.median(att_flat)]
        if len(bg) == 0:
            bg = zz
    else:
        bg = zz
    proto = bg.mean(0)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    return 1.0 - (zz @ proto)
