"""Reusable image-level BCE detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

from lab_utils.data.area_tiers import AREA_TIER_LABELS, area_tier
from lab_utils.logging.text import log_line


REAL_KINDS = frozenset({"imd_real", "indoor_real", "casia_real"})


class BCEHeadAdapter(nn.Module):
    """Wrap a dict-output multi-head model as ``model(x) -> image_logit``."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            logit = out.get("image_logit")
            if logit is None:
                raise RuntimeError("BCEHeadAdapter: dict output has no image_logit")
            return logit
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


@dataclass(frozen=True)
class ImageBCEMetrics:
    n_total: int
    n_splice: int
    n_real: int
    auc: float
    threshold: float
    bal_acc: float
    tpr: float
    tnr: float
    opt_threshold: float
    opt_bal_acc: float
    opt_tpr: float
    opt_tnr: float
    tier_stats: Dict[str, Dict[str, float]]


def _auc_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    n_splice = int(labels.sum())
    n_real = int(len(labels) - n_splice)
    if n_splice <= 0 or n_real <= 0:
        return float("nan")
    order = np.argsort(-logits)
    sorted_labels = labels[order]
    tpr_pts = np.cumsum(sorted_labels) / max(n_splice, 1)
    fpr_pts = np.cumsum(1 - sorted_labels) / max(n_real, 1)
    auc = float(np.trapezoid(tpr_pts, fpr_pts))
    return 1.0 + auc if auc < 0 else auc


def image_bce_metrics(
    logits: Iterable[float],
    labels: Iterable[int],
    areas: Optional[Iterable[float]] = None,
    *,
    threshold: float = 0.5,
) -> ImageBCEMetrics:
    """Compute image-level BCE metrics from logits, labels, and optional areas."""

    logits_np = np.asarray(list(logits), dtype=np.float64)
    labels_np = np.asarray(list(labels), dtype=np.int32)
    if areas is None:
        areas_np = np.zeros_like(logits_np, dtype=np.float64)
    else:
        areas_np = np.asarray(list(areas), dtype=np.float64)
    if logits_np.shape[0] != labels_np.shape[0]:
        raise ValueError("image_bce_metrics: logits/labels length mismatch")
    if areas_np.shape[0] != labels_np.shape[0]:
        raise ValueError("image_bce_metrics: areas/labels length mismatch")

    n_total = int(labels_np.shape[0])
    n_splice = int(labels_np.sum())
    n_real = n_total - n_splice
    probs = 1.0 / (1.0 + np.exp(-logits_np)) if n_total else np.asarray([])
    preds = (probs >= float(threshold)).astype(np.int32)
    tp = int(((preds == 1) & (labels_np == 1)).sum())
    tn = int(((preds == 0) & (labels_np == 0)).sum())
    tpr = tp / n_splice if n_splice else float("nan")
    tnr = tn / n_real if n_real else float("nan")
    bal_acc = 0.5 * (tpr + tnr) if n_splice and n_real else float("nan")

    opt_threshold = 0.0
    opt_tpr, opt_tnr, opt_bal_acc = 0.0, 1.0, 0.5
    if n_splice and n_real:
        for t in np.unique(logits_np):
            pred = (logits_np >= t).astype(np.int32)
            ttpr = float(((pred == 1) & (labels_np == 1)).sum()) / n_splice
            ttnr = float(((pred == 0) & (labels_np == 0)).sum()) / n_real
            bacc = 0.5 * (ttpr + ttnr)
            if bacc > opt_bal_acc:
                opt_bal_acc = bacc
                opt_threshold = float(t)
                opt_tpr = ttpr
                opt_tnr = ttnr

    opt_preds = (logits_np >= opt_threshold).astype(np.int32) if n_total else np.asarray([])
    tier_stats: Dict[str, Dict[str, float]] = {}
    tiers = np.asarray([area_tier(a) for a in areas_np], dtype=object)
    for tier in AREA_TIER_LABELS:
        mask = (labels_np == 1) & (tiers == tier)
        n = int(mask.sum())
        tier_stats[tier] = {
            "n": n,
            "tpr": float(preds[mask].mean()) if n else float("nan"),
            "opt_tpr": float(opt_preds[mask].mean()) if n else float("nan"),
        }

    return ImageBCEMetrics(
        n_total=n_total,
        n_splice=n_splice,
        n_real=n_real,
        auc=_auc_from_logits(logits_np, labels_np),
        threshold=float(threshold),
        bal_acc=float(bal_acc),
        tpr=float(tpr),
        tnr=float(tnr),
        opt_threshold=float(opt_threshold),
        opt_bal_acc=float(opt_bal_acc),
        opt_tpr=float(opt_tpr),
        opt_tnr=float(opt_tnr),
        tier_stats=tier_stats,
    )


@torch.no_grad()
def collect_image_bce_logits(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[list[float], list[int], list[float]]:
    """Collect image logits, binary labels, and area fractions from a loader."""

    model.eval()
    logits: list[float] = []
    labels: list[int] = []
    areas: list[float] = []
    for batch in loader:
        if batch is None:
            continue
        img = batch["img"].to(device, non_blocking=True)
        raw = model(img).detach().cpu().float().numpy()
        meta_list = batch["meta"] if isinstance(batch["meta"], list) else [
            {k: v[i] for k, v in batch["meta"].items()}
            for i in range(img.shape[0])
        ]
        for i, value in enumerate(raw):
            meta = meta_list[i]
            kind = str(meta.get("kind", ""))
            logits.append(float(value))
            labels.append(0 if kind in REAL_KINDS else 1)
            areas.append(float(meta.get("blob_area_actual", 0.0)))
    return logits, labels, areas


def run_image_bce_eval(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    threshold: float = 0.5,
    log_tag: str = "[eval]",
    tag: str = "",
) -> dict:
    """Collect logits, compute metrics, log the standard image-BCE summary."""

    logits, labels, areas = collect_image_bce_logits(model, loader, device)
    metrics = image_bce_metrics(logits, labels, areas, threshold=threshold)
    suffix = f" {tag}" if tag else ""
    log_line(
        f"{log_tag}{suffix} "
        f"n_total={metrics.n_total} n_splice={metrics.n_splice} n_real={metrics.n_real} "
        f"auc={metrics.auc:.4f} "
        f"@ thresh={threshold:.3f}: bal_acc={metrics.bal_acc:.4f} "
        f"tpr={metrics.tpr:.4f} tnr={metrics.tnr:.4f} | "
        f"@ opt thresh={metrics.opt_threshold:.3f}: bal_acc={metrics.opt_bal_acc:.4f} "
        f"tpr={metrics.opt_tpr:.4f} tnr={metrics.opt_tnr:.4f}"
    )
    for tier in AREA_TIER_LABELS:
        stats = metrics.tier_stats[tier]
        log_line(
            f"{log_tag}{suffix}   area_tier={tier} n={int(stats['n'])} "
            f"tpr@{threshold:.3f}={stats['tpr']:.4f} "
            f"tpr@opt={stats['opt_tpr']:.4f}"
        )
    return {
        "auc": metrics.auc,
        "bal_acc": metrics.bal_acc,
        "tpr": metrics.tpr,
        "tnr": metrics.tnr,
        "opt_thresh": metrics.opt_threshold,
        "opt_bacc": metrics.opt_bal_acc,
        "n_total": metrics.n_total,
        "n_splice": metrics.n_splice,
        "n_real": metrics.n_real,
        "tier_stats": metrics.tier_stats,
        "bucket_stats": metrics.tier_stats,
        "logits": np.asarray(logits, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int32),
    }
