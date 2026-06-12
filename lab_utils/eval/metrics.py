"""lab_utils.eval.metrics — per-image and aggregate scoring functions."""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from lab_utils.eval.partition import (   # noqa: F401 (re-exported)
    calibrate_gate_tau,
    partition_image,
    silhouette_cosine,
    spherical_kmeans2,
)


def f1_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    empty_value: float = 1.0,
) -> Tuple[float, float]:
    """Binary patch-level F1 and IoU.

    Args:
        pred, gt:     Same-shape boolean / 0–1 arrays.
        empty_value:  Score returned when both masks are empty. Default 1.0
                      treats an empty prediction on empty ground truth as a
                      perfect (degenerate) match — appropriate for the
                      principled metric. Pass 0.0 for the "diagnose script"
                      convention where empty/empty is treated as no score.
    """
    pred  = pred.astype(bool)
    gt    = gt.astype(bool)
    inter = int((pred & gt).sum())
    p_sum = int(pred.sum())
    g_sum = int(gt.sum())
    union = p_sum + g_sum - inter
    if p_sum == 0 and g_sum == 0:
        return float(empty_value), float(empty_value)
    f1  = (2.0 * inter) / max(1, p_sum + g_sum)
    iou = inter / max(1, union)
    return float(f1), float(iou)


def binary_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, float]:
    """Full per-image dict: f1, iou, prec, rec, pred_frac, gt_frac.

    Empty unions yield 0.0 (the convention used by the diagnose scripts).
    For the principled (empty/empty -> 1.0) variant use :func:`f1_iou`.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int((pred & gt).sum())
    p_n = int(pred.sum())
    g_n = int(gt.sum())
    union = int((pred | gt).sum())
    n_total = int(pred.size)
    return {
        "f1":         (2.0 * inter / (p_n + g_n)) if (p_n + g_n) > 0 else 0.0,
        "iou":        (inter / union) if union > 0 else 0.0,
        "prec":       (inter / p_n) if p_n > 0 else 0.0,
        "rec":        (inter / g_n) if g_n > 0 else 0.0,
        "pred_frac":  (p_n / float(n_total)) if n_total > 0 else 0.0,
        "gt_frac":    (g_n / float(n_total)) if n_total > 0 else 0.0,
    }


def pixel_acc(pred: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of correctly classified patches."""
    return float((pred.astype(int) == gt.astype(int)).mean())


def threshold_sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 101,
) -> Tuple[float, float, float]:
    """Sweep a threshold over continuous scores to find best F1.

    Returns (best_f1, best_threshold, auprc).
    """
    thresholds = np.linspace(0.0, 1.0, int(n_thresholds))
    best_f1 = 0.0
    best_t  = 0.5
    precisions, recalls = [], []

    for t in thresholds:
        pred = (scores >= t).astype(int)
        f1, _ = f1_iou(pred, labels)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        precisions.append(prec)
        recalls.append(rec)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = float(t)

    # AUPRC via trapezoidal rule on (recall, precision) curve.
    rec_arr  = np.array(recalls)
    prec_arr = np.array(precisions)
    order    = np.argsort(rec_arr)
    auprc    = float(np.trapz(prec_arr[order], rec_arr[order]))
    return best_f1, best_t, auprc


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (binary)."""
    pos  = scores[labels.astype(bool)]
    neg  = scores[~labels.astype(bool)]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    # Mann-Whitney U statistic.
    u = float(np.sum(pos[:, None] > neg[None, :]))
    return u / (pos.size * neg.size)


def _stat(arr: Sequence[float]) -> dict:
    """Compute mean + median + std over a sequence; empty → NaN."""
    if len(arr) == 0:
        return {'mean': float('nan'), 'median': float('nan'), 'std': float('nan'), 'n': 0}
    a = np.array(arr, dtype=np.float64)
    return {
        'mean': float(np.mean(a)),
        'median': float(np.median(a)),
        'std': float(np.std(a)),
        'n': len(a),
    }
