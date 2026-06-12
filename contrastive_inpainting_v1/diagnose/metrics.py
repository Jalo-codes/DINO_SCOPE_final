"""Pixel-resolution F1/IoU/precision/recall. ONE definition used everywhere.

Contract:
  - pred and gt are (H, W) bool arrays of identical shape — original image
    resolution.  Pred outside the pass's operating region must be False
    (handled by the caller; project.patch_to_pixel_mask enforces this).
  - GT is the binary mask from the source mask file (>0 → True), not
    patch-averaged.  This is the only place pixel-resolution GT is consumed.
  - F1 = 2 * inter / (pred_n + gt_n).  When both are zero, F1 = 1.0 by
    convention (perfectly correct empty prediction on real image).  When
    only one is zero, F1 = 0.0 (typical FP-on-real or full-FN-on-splice).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def f1_pixel(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Pixel-level F1/IoU/prec/rec/pred_frac/gt_frac.

    Returns a dict so passes can grab whichever fields they need; summary
    only reads f1, iou, prec, rec, pred_frac.
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(
            f"f1_pixel shape mismatch: pred={pred.shape} gt={gt.shape}. "
            f"Caller must project both to the same (H, W) before calling."
        )
    inter = int(np.logical_and(pred, gt).sum())
    pred_n = int(pred.sum())
    gt_n = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())
    size = int(pred.size) if pred.size else 1
    # F1 conventions:
    #   pred_n==0 and gt_n==0   →   1.0  (perfect empty agreement; e.g., real
    #                                     image, pass predicts nothing → correct)
    #   pred_n==0 xor gt_n==0   →   0.0  (any disagreement when one side empty)
    if pred_n == 0 and gt_n == 0:
        f1 = 1.0
        iou = 1.0
    else:
        f1 = (2.0 * inter / (pred_n + gt_n)) if (pred_n + gt_n) else 0.0
        iou = (inter / union) if union else 0.0
    prec = (inter / pred_n) if pred_n else 0.0
    rec = (inter / gt_n) if gt_n else 0.0
    return {
        "f1": float(f1),
        "iou": float(iou),
        "prec": float(prec),
        "rec": float(rec),
        "pred_frac": float(pred_n / size),
        "gt_frac": float(gt_n / size),
        "_inter": int(inter),
        "_pred_n": int(pred_n),
        "_gt_n": int(gt_n),
        "_size": int(size),
    }


def f1_aggregate(rows_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-image f1_pixel results to a single F1 over the union.

    Useful for `f1_pixelwise_with_reals`: sum inter, pred_n, gt_n across
    images (treating each image's pixels as part of one big collection).
    Reports both the micro-aggregated F1 and the per-image median F1.
    """
    if not rows_metrics:
        return {"f1_micro": float("nan"), "f1_med": float("nan"), "n": 0}
    inter = sum(int(m.get("_inter", 0)) for m in rows_metrics)
    pred_n = sum(int(m.get("_pred_n", 0)) for m in rows_metrics)
    gt_n = sum(int(m.get("_gt_n", 0)) for m in rows_metrics)
    f1_micro = (2.0 * inter / (pred_n + gt_n)) if (pred_n + gt_n) else 0.0
    per_img = [float(m["f1"]) for m in rows_metrics if "f1" in m]
    f1_med = float(np.median(per_img)) if per_img else float("nan")
    return {
        "f1_micro": float(f1_micro),
        "f1_med": f1_med,
        "n": int(len(rows_metrics)),
    }


def stats(xs: Sequence[float]) -> Dict[str, float]:
    """Median / mean / std / p25 / p75 for a sequence; NaNs in input ignored."""
    arr = np.asarray([x for x in xs if not (x is None or np.isnan(float(x)))],
                     dtype=np.float64)
    if arr.size == 0:
        return {"med": float("nan"), "mean": float("nan"),
                "std": float("nan"), "p25": float("nan"), "p75": float("nan"),
                "n": 0}
    return {
        "med": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "n": int(arr.size),
    }
