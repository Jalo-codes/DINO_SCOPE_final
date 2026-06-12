"""K-means polarity rules: pick which of the two clusters is "the splice".

Two rules:
  attn  — cluster with HIGHER mean pool_attention is splice (NO GT used).
  ceil  — try both cluster assignments, pick whichever maximises F1 vs GT
          (uses GT to flip the label only; never relabels patches).  Diagnostic
          upper bound for the polarity decision.

Returns a `(n*n,)` flat bool mask + a `was_inverted` flag indicating whether
the picked cluster is the larger one (so callers can compute ceil_inverted_rate
without re-doing the comparison).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def polarity_attn(raw_labels: np.ndarray, attention: Optional[np.ndarray]) -> np.ndarray:
    """Cluster with higher mean attention is splice.

    If attention is None, falls back to the smaller cluster (legacy default).
    """
    raw = np.asarray(raw_labels).reshape(-1)
    n0 = int((raw == 0).sum())
    n1 = int((raw == 1).sum())
    if attention is None:
        chosen = 0 if n0 <= n1 else 1
        return (raw == chosen).astype(bool)
    att = np.asarray(attention).reshape(-1)
    mean0 = float(att[raw == 0].mean()) if n0 else float("-inf")
    mean1 = float(att[raw == 1].mean()) if n1 else float("-inf")
    chosen = 0 if mean0 >= mean1 else 1
    return (raw == chosen).astype(bool)


def polarity_overlap(raw_labels: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Cluster with the higher overlap *fraction* against a reference hot mask
    is splice (NO GT used).

    Intended for the zoom-crop pass: the reference is the FULL-image attention
    hot mask projected into crop patch coordinates, so the polarity decision
    keeps the evidence that targeted the crop instead of re-deciding from the
    crop's re-normalized attention. Fraction (not count) so the larger cluster
    gets no mechanical advantage.

    Falls back to the smaller cluster when the reference is empty or a cluster
    is empty (same legacy default as polarity_attn without attention).
    """
    raw = np.asarray(raw_labels).reshape(-1)
    ref = np.asarray(ref_mask, dtype=bool).reshape(-1)
    n0 = int((raw == 0).sum())
    n1 = int((raw == 1).sum())
    if not ref.any() or n0 == 0 or n1 == 0:
        chosen = 0 if n0 <= n1 else 1
        return (raw == chosen).astype(bool)
    frac0 = float(ref[raw == 0].mean())
    frac1 = float(ref[raw == 1].mean())
    chosen = 0 if frac0 >= frac1 else 1
    return (raw == chosen).astype(bool)


def polarity_ceil(
    raw_labels: np.ndarray, gt_flat: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """F1-max polarity: pick whichever cluster assignment scores higher F1 vs GT.

    Returns (pred_flat_bool, was_inverted). was_inverted=True iff the chosen
    cluster is the larger one — i.e., splice is the majority cluster in this
    partition. Track of inversion rate per area_tier is diagnostic for "polarity
    rule is wrong on large splices".
    """
    raw = np.asarray(raw_labels).reshape(-1)
    gt = np.asarray(gt_flat, dtype=bool).reshape(-1)
    pred_a = (raw == 0)
    pred_b = (raw == 1)
    # In-line F1 to avoid circular imports with metrics.f1_pixel (which is
    # pixel-level; this is patch-level). Identical formula.
    def _f1(p, g):
        inter = int((p & g).sum())
        denom = int(p.sum()) + int(g.sum())
        return (2.0 * inter / denom) if denom else 0.0
    f1_a = _f1(pred_a, gt)
    f1_b = _f1(pred_b, gt)
    if f1_b > f1_a:
        chosen = pred_b
    else:
        chosen = pred_a
    n_chosen = int(chosen.sum())
    n_other = int((~chosen).sum())
    was_inverted = bool(n_chosen > n_other)
    return chosen.astype(bool), was_inverted


def both_variants(
    raw_labels: np.ndarray,
    attention: Optional[np.ndarray],
    gt_flat: Optional[np.ndarray],
) -> Dict[str, Dict]:
    """Compute both polarity variants in one call.

    Returns:
        {
          'attn':  {'pred': (n*n,) bool, 'inverted': bool},
          'ceil':  {'pred': (n*n,) bool, 'inverted': bool}   # only if gt_flat
        }

    `inverted` for the attn rule means the rule chose the larger cluster
    (consistent with ceil's inverted flag — useful to compare).
    """
    out: Dict[str, Dict] = {}
    attn_pred = polarity_attn(raw_labels, attention)
    attn_inv = bool(int(attn_pred.sum()) > int((~attn_pred).sum()))
    out["attn"] = {"pred": attn_pred, "inverted": attn_inv}
    if gt_flat is not None:
        ceil_pred, ceil_inv = polarity_ceil(raw_labels, gt_flat)
        out["ceil"] = {"pred": ceil_pred, "inverted": ceil_inv}
    return out
