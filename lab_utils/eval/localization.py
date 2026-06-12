"""lab_utils.eval.localization — full-image and BCE-guided zoom localization
eval with a BCE-logit threshold sweep.

Per-image, we compute the raw partition (k-means + smaller-cluster mask) ONCE
on the full-image contrastive embeddings, optionally also via a small
source-resolution square-crop zoom pass where BCE selects the best crop. We
then report median F1/IoU per area_bucket at each threshold in a sweep, plus a
`no_gate` baseline that never suppresses at the image level.

This separates two questions cleanly:
  - "How good is the partition itself?"            → no_gate row
  - "What does the BCE-gated system actually do?"  → t=opt_thresh row
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from lab_utils.eval.partition import spherical_kmeans2
from lab_utils.eval.sliding_window import tile_window_contrastive_masks
from lab_utils.data.resolution import (
    Resolution, resize_only, resize_only_mask, oracle_mask_crop,
)
from lab_utils.data.augment.corruptions import CorruptionSpec, apply_corruption
from lab_utils.logging.text import log_line


_DEFAULT_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_NORMALIZE_STD  = (0.229, 0.224, 0.225)


_REAL_KINDS   = frozenset({'imd_real', 'indoor_real', 'casia_real'})
_SPLICE_KINDS = frozenset({'imd_splice', 'casia_splice'})


def _select_cluster(
    raw_labels: np.ndarray,
    attention: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pick which k-means cluster is the splice prediction.

    NOTE: this is the DEPLOYMENT-time polarity heuristic — used when there is no
    GT to oracle against (real inference, visualization). The eval metric uses
    ``_oracle_polarity`` instead, so localization scores never depend on this
    choice (or on the image head). Kept here as the reference deployment rule.

    When `attention` (per-patch BCE attention weights, shape (N,)) is provided,
    pick the cluster whose patches have HIGHER mean attention — this works
    regardless of cluster size and fixes the smaller-cluster-rule failure on
    splices that occupy >50% of the patch grid (oracle showed 78% inversion
    rate on CASIA large at full image with smaller-rule, vs 0% with attention).

    Falls back to the smaller-cluster-as-positive rule when attention is None
    (contrastive-only model with no BCE head).

    Args:
        raw_labels: (N,) int in {0, 1} from spherical_kmeans2.
        attention:  (N,) float per-patch BCE attention, or None.

    Returns:
        (N,) bool — True for predicted-splice patches.
    """
    raw_labels = np.asarray(raw_labels)
    if attention is not None:
        a = np.asarray(attention, dtype=np.float64)
        m0 = (raw_labels == 0)
        m1 = (raw_labels == 1)
        a0 = float(a[m0].mean()) if m0.any() else -np.inf
        a1 = float(a[m1].mean()) if m1.any() else -np.inf
        splice_label = 0 if a0 >= a1 else 1
    else:
        n0 = int((raw_labels == 0).sum())
        n1 = int((raw_labels == 1).sum())
        splice_label = 0 if n0 <= n1 else 1
    return (raw_labels == splice_label).astype(np.bool_)


def _oracle_polarity(
    raw_labels: np.ndarray, gt: np.ndarray
) -> Tuple[np.ndarray, float, float, float, float, float]:
    """Score a 2-cluster partition under ORACLE polarity.

    "Which cluster is the splice" is a trivial 2-way ambiguity, not a capability
    under test — so we DON'T predict it. We take the labeling (cluster==1 as
    splice, or its complement) that best matches GT, by F1 (IoU tie-break). This
    is the standard forgery-localization convention (max over mask and its
    complement) and fully decouples the metric from the image head: no attention,
    no smaller-cluster heuristic. What's measured is the partition itself.

    Returns:
        (chosen_pred_bool, f1, iou, prec, rec, pred_frac)
    """
    raw_labels = np.asarray(raw_labels)
    pred_a = (raw_labels == 1)
    pred_b = ~pred_a
    ma = _mask_metrics(pred_a, gt)   # (f1, iou, prec, rec, pred_frac)
    mb = _mask_metrics(pred_b, gt)
    if (ma[0], ma[1]) >= (mb[0], mb[1]):
        return (pred_a, *ma)
    return (pred_b, *mb)


def _bucket(a: float) -> str:
    if a < 0.15:
        return 'small'
    if a < 0.30:
        return 'medium'
    return 'large'


def _mask_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float, float]:
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = int((pred & gt).sum())
    pred_n = int(pred.sum()); gt_n = int(gt.sum())
    union = int((pred | gt).sum())
    f1  = (2 * inter / (pred_n + gt_n)) if (pred_n + gt_n) > 0 else 0.0
    iou = (inter / union) if union > 0 else 0.0
    prec = (inter / pred_n) if pred_n > 0 else 0.0
    rec = (inter / gt_n) if gt_n > 0 else 0.0
    pred_frac = pred_n / float(pred.size) if pred.size else 0.0
    return float(f1), float(iou), float(prec), float(rec), float(pred_frac)


def _f1_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    f1, iou, _, _, _ = _mask_metrics(pred, gt)
    return f1, iou


def _patches_to_pixels(
    pred_flat: np.ndarray, n_side: int, patch_size: int
) -> np.ndarray:
    """Upsample a flat (N,) patch-grid prediction to a pixel mask (S, S).

    Each patch cell expands to a patch_size × patch_size block (nearest), so
    S = n_side * patch_size = image_size. This is the model's own input frame —
    no information beyond the patch grid is invented, the comparison is just
    moved to pixel granularity.
    """
    grid = np.asarray(pred_flat, dtype=np.float64).reshape(n_side, n_side)
    px = np.kron(grid, np.ones((patch_size, patch_size), dtype=np.float64))
    return px > 0.5


def _load_gt_pixel_mask(meta: Dict, res: Resolution) -> Optional[np.ndarray]:
    """Load the GT mask from disk and rasterize to a pixel mask at the input
    frame (image_size × image_size), aligned with the eval crop (resize_only).

    Returns a boolean (image_size, image_size) array, or None if no mask path
    or the mask cannot be read / is empty.
    """
    mask_path = str(meta.get('mask_path', '') or '')
    if not mask_path:
        return None
    try:
        mask_pil = Image.open(mask_path).convert('L')
    except Exception:
        return None
    # Same resize the eval dataset applies (full image → square input frame), so
    # the patch grid and the pixel GT share one coordinate frame.
    mask_pil = resize_only_mask(mask_pil, res).convert('L')
    gt_px = np.asarray(mask_pil) > 127
    return gt_px if gt_px.any() else None


@dataclass
class _LocSample:
    kind: str
    area: float
    bucket: str
    bce_logit: Optional[float]    # None ⇒ no BCE head
    f1_full: float                # raw partition (no gate)
    iou_full: float
    prec_full: float
    rec_full: float
    pred_frac_full: float
    f1_swin: Optional[float]      # None ⇒ swin not run for this sample
    iou_swin: Optional[float]
    prec_swin: Optional[float]
    rec_swin: Optional[float]
    pred_frac_swin: Optional[float]
    swin_n_pass: int = 0
    swin_n_boundary: int = 0
    # PIXEL-level metrics for the full-image partition: the patch-grid
    # prediction upsampled to the input frame, scored against the GT mask at
    # that frame. None ⇒ pixel eval not requested (res not passed).
    f1_full_px: Optional[float] = None
    iou_full_px: Optional[float] = None
    prec_full_px: Optional[float] = None
    rec_full_px: Optional[float] = None
    # Deployed-polarity (attention/smaller-cluster) full-image scores + whether
    # the deployed choice agreed with the oracle's — for the oracle-tax report.
    f1_full_deployed: Optional[float] = None
    iou_full_deployed: Optional[float] = None
    polarity_agree: Optional[bool] = None

    # Outlier Gap Partition fields
    f1_gap: Optional[float] = None
    iou_gap: Optional[float] = None
    prec_gap: Optional[float] = None
    rec_gap: Optional[float] = None
    pred_frac_gap: Optional[float] = None

    # Clustering Diagnostics & Outlier Statistics
    pos_coh: Optional[float] = None
    bg_coh: Optional[float] = None
    cent_sep: Optional[float] = None
    out_gap: Optional[float] = None

    # Outlier patch fractions
    frac_outliers_mad: Optional[float] = None
    frac_outliers_03: Optional[float] = None
    frac_outliers_05: Optional[float] = None

    # Clustering fits (k=2, k=3, k=4)
    sil_k2: Optional[float] = None
    sil_k3: Optional[float] = None
    sil_k4: Optional[float] = None
    best_k: Optional[int] = None


def _mean_offdiag_sim(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return float('nan')
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = x @ x.T
    return float((sim.sum() - np.trace(sim)) / (x.shape[0] * (x.shape[0] - 1)))


def _cluster_shape_metrics(z: np.ndarray, mask: np.ndarray, score: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
    """Calculate pos_coh, bg_coh, cent_sep, out_gap."""
    z = np.asarray(z, dtype=np.float64)
    m = np.asarray(mask).reshape(-1).astype(bool)
    if z.ndim != 2 or z.shape[0] != m.size:
        return float('nan'), float('nan'), float('nan'), float('nan')

    if int(m.sum()) > 0:
        pos_coh = _mean_offdiag_sim(z[m])
    else:
        pos_coh = float('nan')

    if int((~m).sum()) > 0:
        bg_coh = _mean_offdiag_sim(z[~m])
    else:
        bg_coh = float('nan')

    if int(m.sum()) > 0 and int((~m).sum()) > 0:
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        cp = zn[m].mean(0)
        cn = zn[~m].mean(0)
        cp = cp / (np.linalg.norm(cp) + 1e-8)
        cn = cn / (np.linalg.norm(cn) + 1e-8)
        cent_sep = float(1.0 - np.dot(cp, cn))
    else:
        cent_sep = float('nan')

    out_gap = float('nan')
    if score is not None and int(m.sum()) > 0 and int((~m).sum()) > 0:
        s = np.asarray(score, dtype=np.float64).reshape(-1)
        if s.size == m.size:
            out_gap = float(np.median(s[m]) - np.median(s[~m]))

    return pos_coh, bg_coh, cent_sep, out_gap


def _mad_outlier_frac(s: np.ndarray, k: float = 3.0) -> float:
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826
    thr = med + k * max(mad, 1e-6)
    return float((s > thr).mean())


def spherical_kmeans_k(
    z: np.ndarray,
    k: int,
    n_init: int = 4,
    n_iters: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """Spherical k-means for arbitrary k on L2-normalised embeddings.
    Returns (labels, inertia).
    """
    z = np.ascontiguousarray(z, dtype=np.float32)
    n, d = z.shape
    best_labels = None
    best_inertia = np.inf

    for run in range(n_init):
        rng = np.random.default_rng(seed + run)
        centroids = []
        i0 = rng.integers(0, n)
        centroids.append(z[i0])
        for _ in range(1, k):
            c_stacked = np.stack(centroids, axis=0)
            sim = z @ c_stacked.T
            max_sim = sim.max(axis=1)
            dist = np.clip(1.0 - max_sim, a_min=0.0, a_max=None)
            if dist.sum() <= 0:
                idx = rng.integers(0, n)
            else:
                idx = rng.choice(n, p=dist / dist.sum())
            centroids.append(z[idx])
        centroids = np.stack(centroids, axis=0)
        labels = np.zeros(n, dtype=np.int64)

        for _ in range(n_iters):
            sim = z @ centroids.T
            new_labels = np.argmax(sim, axis=1)
            if (new_labels == labels).all():
                break
            labels = new_labels
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if mask.sum() == 0:
                    other = centroids[(j + 1) % k]
                    far = int(np.argmin(z @ other))
                    new_centroids[j] = z[far]
                else:
                    mean = z[mask].mean(axis=0)
                    n_norm = np.linalg.norm(mean) + 1e-12
                    new_centroids[j] = mean / n_norm
            centroids = new_centroids

        sim = z @ centroids.T
        labels = np.argmax(sim, axis=1)
        inertia = float((1.0 - sim[np.arange(n), labels]).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels

    return best_labels, best_inertia


def silhouette_cosine_k(z: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette score in cosine distance space for arbitrary k."""
    z = np.ascontiguousarray(z, dtype=np.float32)
    n = z.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2 or n_clusters >= n:
        return -1.0

    sim = z @ z.T
    dist = 1.0 - sim

    s = np.zeros(n)
    for i in range(n):
        l_i = labels[i]
        same_cluster_mask = (labels == l_i)
        same_cluster_mask[i] = False
        if same_cluster_mask.sum() > 0:
            a_i = dist[i, same_cluster_mask].mean()
        else:
            a_i = 0.0

        b_i = np.inf
        for l_j in unique_labels:
            if l_j == l_i:
                continue
            other_cluster_mask = (labels == l_j)
            d_ij = dist[i, other_cluster_mask].mean()
            if d_ij < b_i:
                b_i = d_ij

        denom = max(a_i, b_i)
        if denom > 1e-12:
            s[i] = (b_i - a_i) / denom
        else:
            s[i] = 0.0

    return float(s.mean())


# ── per-image collection ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_localization_samples(
    multi_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    n_patch_per_side: int,
    run_swin: bool,
    swin_scales: Sequence[float] = (1.0, 0.7, 0.5),
    swin_stride_frac: float = 0.5,
    swin_inner_batch: int = 8,
    swin_bce_gate_threshold: Optional[float] = None,
    swin_use_source_resolution: bool = True,
    swin_normalize_mean: Tuple[float, float, float] = _DEFAULT_NORMALIZE_MEAN,
    swin_normalize_std:  Tuple[float, float, float] = _DEFAULT_NORMALIZE_STD,
    res: Optional[Resolution] = None,
    log_tag: str = '[eval]',
    tag: str = '',
) -> List[_LocSample]:
    """One pass over loader → list of per-splice-image samples with raw F1/IoU.

    Reals are skipped (no GT mask). Items where the cropped GT mask is all-zero
    are also skipped.

    The zoom pass requires a BCE head because BCE selects the crop and its
    attention chooses contrastive cluster polarity. If the model has no BCE
    head, swin is skipped regardless of `run_swin`, and the caller should treat
    full-image partition as the authoritative measurement.

    Args:
        swin_use_source_resolution: Kept for caller compatibility. The current
            zoom-localization path always uses source-resolution PIL crops and
            skips the sample if the source image cannot be loaded.
    """
    import time
    multi_head.eval()
    samples: List[_LocSample] = []
    suffix = f' {tag}' if tag else ''
    t_start = time.time()
    n_splice_seen = 0
    log_every_n = 25  # progress every N splice items

    # Probe the first batch to detect head presence; if no BCE head, force
    # run_swin=False so we don't OR in noise predictions.
    bce_head_present = None  # set on first batch

    for batch in loader:
        if batch is None:
            continue
        img_batch = batch['img']
        meta_list = batch['meta'] if isinstance(batch['meta'], list) else [
            {k: v[i] for k, v in batch['meta'].items()}
            for i in range(img_batch.shape[0])
        ]
        gt_all = batch['splice_labels'].cpu().numpy()    # (B, N)

        # Forward whole batch ONCE for the full-image pass (cheap, batched).
        out = multi_head(img_batch.to(device, non_blocking=True))
        z_full = out['contrastive']
        if z_full is None:
            log_line(f'{log_tag}{suffix} no contrastive head; cannot localize')
            return []
        z_full_np = z_full.detach().cpu().float().numpy()    # (B, N, d)
        # bce_np feeds the threshold-GATED rows only (the deployed detect-then-
        # localize view). The partition metric itself is oracle-polarity, so it
        # no longer reads pool attention at all — localization quality is
        # measured independently of the image head.
        bce_np = (out['image_logit'].detach().cpu().float().numpy()
                  if out['image_logit'] is not None else None)
        # Per-patch pool attention — used ONLY to score the DEPLOYED-polarity
        # baseline (the oracle-tax), never the headline oracle metric.
        att_full = (out['pool_attention'].detach().cpu().float().numpy()
                    if out.get('pool_attention') is not None else None)

        if bce_head_present is None:
            bce_head_present = bce_np is not None
            if run_swin and not bce_head_present:
                log_line(
                    f'{log_tag}{suffix} swin requested but model has no BCE '
                    f'head; FORCING swin=off (BCE crop selection requires it)'
                )
                run_swin = False
        for i in range(img_batch.shape[0]):
            kind = meta_list[i].get('kind', '')
            if kind not in _SPLICE_KINDS:
                continue
            gt_i = gt_all[i]
            if int(gt_i.sum()) == 0:
                continue
            # Pixel-area bucketing when the GT mask is available (res passed):
            # consistent with the coarse→fine eval. Falls back to the patch-grid
            # area (blob_area_actual) when there's no pixel GT.
            gt_px = _load_gt_pixel_mask(meta_list[i], res) if res is not None else None
            if gt_px is not None:
                area = float(gt_px.mean())
            else:
                area = float(meta_list[i].get('blob_area_actual',
                                              gt_i.mean() if len(gt_i) else 0.0))

            # Full-image partition under ORACLE polarity. The k-means separation
            # is BCE-independent; we then take the cluster→splice labeling that
            # best matches GT (we do NOT predict polarity — it's the one thing
            # we legitimately oracle at eval). This fully decouples the
            # localization metric from the image head's attention.
            raw_labels, _ = spherical_kmeans2(z_full_np[i], n_init=4)
            full_pred, f1_f, iou_f, prec_f, rec_f, pred_frac_f = _oracle_polarity(
                raw_labels, gt_i
            )

            # Outlier Gap Partition + Cluster Shape Metrics + Clustering fit
            # 1) Compute outlier score
            zz = z_full_np[i] / (np.linalg.norm(z_full_np[i], axis=1, keepdims=True) + 1e-8)
            att_i = att_full[i] if att_full is not None else None
            if att_i is not None and len(np.asarray(att_i).reshape(-1)) == len(zz):
                a = np.asarray(att_i).reshape(-1)
                bg = zz[a <= np.median(a)]
                if len(bg) == 0:
                    bg = zz
            else:
                bg = zz
            proto = bg.mean(0)
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            score = 1.0 - (zz @ proto)

            # 2) Compute gap threshold
            s_sorted = np.sort(np.asarray(score, dtype=np.float64))
            med = np.median(s_sorted)
            upper = s_sorted[s_sorted >= med]
            if len(upper) < 3:
                gap_thr = float(s_sorted.max()) + 1.0
            else:
                diffs = np.diff(upper)
                gi = int(np.argmax(diffs))
                gap_thr = float(0.5 * (upper[gi] + upper[gi + 1]))

            gap_pred = score >= gap_thr
            f1_g, iou_g, prec_g, rec_g, pred_frac_g = _mask_metrics(gap_pred, gt_i)

            # 3) Compute cluster shape metrics on full_pred (k-means partition)
            pos_coh, bg_coh, cent_sep, out_gap = _cluster_shape_metrics(z_full_np[i], full_pred, score)

            # 4) Compute outlier patch fractions
            frac_outliers_mad = _mad_outlier_frac(score, k=3.0)
            frac_outliers_03 = float((score > 0.3).mean())
            frac_outliers_05 = float((score > 0.5).mean())

            # 5) Compute silhouette scores for k=2, k=3, k=4
            try:
                # k=2
                labels_k2 = raw_labels
                sil_k2 = silhouette_cosine_k(z_full_np[i], labels_k2)

                # k=3
                labels_k3, _ = spherical_kmeans_k(z_full_np[i], k=3)
                sil_k3 = silhouette_cosine_k(z_full_np[i], labels_k3)

                # k=4
                labels_k4, _ = spherical_kmeans_k(z_full_np[i], k=4)
                sil_k4 = silhouette_cosine_k(z_full_np[i], labels_k4)

                # Best fitting k
                best_k = 2 if sil_k2 >= max(sil_k3, sil_k4) else (3 if sil_k3 >= sil_k4 else 4)
            except Exception:
                sil_k2, sil_k3, sil_k4, best_k = float('nan'), float('nan'), float('nan'), None

            # DEPLOYED-polarity baseline (attention if present, else smaller-
            # cluster) → the 'oracle tax': how much the oracle inflates over what
            # the model would actually pick without peeking at GT.
            att_i = att_full[i] if att_full is not None else None
            dep_pred = _select_cluster(raw_labels, att_i)
            f1_dep, iou_dep, _, _, _ = _mask_metrics(dep_pred, gt_i)
            polarity_agree = bool(np.array_equal(dep_pred, full_pred))

            # PIXEL-level: upsample the SAME oracle-chosen prediction to the input
            # frame and score against the GT mask there. Polarity is one decision
            # per image (resolved above), applied to both granularities.
            f1_fpx, iou_fpx, prec_fpx, rec_fpx = None, None, None, None
            if gt_px is not None:
                pred_px = _patches_to_pixels(
                    full_pred, n_patch_per_side, res.patch_size
                )
                if pred_px.shape == gt_px.shape:
                    f1_fpx, iou_fpx, prec_fpx, rec_fpx, _ = _mask_metrics(
                        pred_px.reshape(-1), gt_px.reshape(-1)
                    )

            # Tile-window partition: SQUARE windows at native source resolution.
            # BCE picks the best light-zoom crop, then BCE attention sets
            # contrastive cluster polarity inside that crop. No full image, no
            # tensor-mode fallback, no aspect squish.
            f1_s, iou_s = None, None
            prec_s, rec_s, pred_frac_s = None, None, None
            swin_n_pass, swin_n_boundary = 0, 0
            if run_swin:
                src_path = str(meta_list[i].get('path', '') or '')
                if not src_path:
                    # No source path — can't do native-resolution tiling.
                    f1_s, iou_s, prec_s, rec_s, pred_frac_s = 0.0, 0.0, 0.0, 0.0, 0.0
                else:
                    try:
                        source_pil = Image.open(src_path).convert('RGB')
                    except Exception as exc:
                        log_line(
                            f'{log_tag}{suffix} loc WARN source load failed '
                            f'for {src_path!r}: {exc}; skipping swin for this item'
                        )
                        source_pil = None
                    if source_pil is not None:
                        # `swin_scales` first entry is the single tile scale.
                        tile_scale = float(swin_scales[0]) if swin_scales else 0.7
                        swin_pred, swin_n_pass, swin_n_boundary = tile_window_contrastive_masks(
                            multi_head, source_pil, device,
                            n_patch_per_side=n_patch_per_side,
                            scale=tile_scale,
                            stride_frac=float(swin_stride_frac),
                            inner_batch_size=int(swin_inner_batch),
                            bce_gate_threshold=(
                                float(swin_bce_gate_threshold)
                                if swin_bce_gate_threshold is not None else None
                            ),
                            normalize_mean=swin_normalize_mean,
                            normalize_std=swin_normalize_std,
                            target_image_size=img_batch.shape[-1],
                        )
                        swin_pred_flat = swin_pred.reshape(-1)
                        if swin_pred_flat.shape[0] != gt_i.shape[0]:
                            raise ValueError(
                                f'localization swin: pred shape {swin_pred_flat.shape} '
                                f'mismatches GT shape {gt_i.shape}'
                            )
                        f1_s, iou_s, prec_s, rec_s, pred_frac_s = _mask_metrics(swin_pred_flat, gt_i)
                    else:
                        f1_s, iou_s, prec_s, rec_s, pred_frac_s = 0.0, 0.0, 0.0, 0.0, 0.0

            samples.append(_LocSample(
                kind=kind, area=area, bucket=_bucket(area),
                bce_logit=(float(bce_np[i]) if bce_np is not None else None),
                f1_full=f1_f, iou_full=iou_f,
                prec_full=prec_f, rec_full=rec_f, pred_frac_full=pred_frac_f,
                f1_swin=f1_s, iou_swin=iou_s,
                prec_swin=prec_s, rec_swin=rec_s, pred_frac_swin=pred_frac_s,
                swin_n_pass=swin_n_pass, swin_n_boundary=swin_n_boundary,
                f1_full_px=f1_fpx, iou_full_px=iou_fpx,
                prec_full_px=prec_fpx, rec_full_px=rec_fpx,
                f1_full_deployed=f1_dep, iou_full_deployed=iou_dep,
                polarity_agree=polarity_agree,
                f1_gap=f1_g, iou_gap=iou_g,
                prec_gap=prec_g, rec_gap=rec_g, pred_frac_gap=pred_frac_g,
                pos_coh=pos_coh, bg_coh=bg_coh,
                cent_sep=cent_sep, out_gap=out_gap,
                frac_outliers_mad=frac_outliers_mad,
                frac_outliers_03=frac_outliers_03,
                frac_outliers_05=frac_outliers_05,
                sil_k2=sil_k2, sil_k3=sil_k3, sil_k4=sil_k4,
                best_k=best_k,
            ))
            n_splice_seen += 1
            if n_splice_seen % log_every_n == 0:
                elapsed = time.time() - t_start
                rate = n_splice_seen / elapsed if elapsed > 0 else 0.0
                log_line(
                    f'{log_tag}{suffix} loc progress n_splice={n_splice_seen} '
                    f'elapsed={elapsed:.0f}s rate={rate:.2f} items/s'
                )

    swin_label = 'on' if run_swin else 'off'
    gate_label = (f'crop_gate={swin_bce_gate_threshold:+.3f}'
                  if (run_swin and swin_bce_gate_threshold is not None)
                  else ('crop_gate=none' if run_swin else ''))
    scale_label = (f'tile_scale={float(swin_scales[0]):.2f} stride_frac={float(swin_stride_frac):.2f}'
                   if (run_swin and swin_scales) else '')
    log_line(f'{log_tag}{suffix} loc collected n_splice={len(samples)} '
             f'(swin={swin_label} {scale_label} {gate_label})')
    return samples


# ── threshold-sweep reporting ────────────────────────────────────────────────

def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {'n': 0, 'mean': float('nan'), 'median': float('nan'),
                'p1': float('nan'), 'p5': float('nan'), 'p25': float('nan'),
                'p75': float('nan'), 'p95': float('nan'), 'p99': float('nan'),
                'std': float('nan')}
    a = np.array(vals, dtype=np.float64)
    p1, p5, p25, p75, p95, p99 = np.percentile(a, [1, 5, 25, 75, 95, 99])
    return {
        'n':      int(len(a)),
        'mean':   float(np.mean(a)),
        'median': float(np.median(a)),
        'p1':     float(p1),
        'p5':     float(p5),
        'p25':    float(p25),
        'p75':    float(p75),
        'p95':    float(p95),
        'p99':    float(p99),
        'std':    float(np.std(a)),
    }


def _pct_line(st: Dict[str, float]) -> str:
    """Compact non-normal-aware distribution summary for a _stats dict."""
    if not st or st.get('n', 0) == 0:
        return 'n=0'
    return (f"n={st['n']} med={st['median']:.3f} "
            f"[p1={st['p1']:.3f} p5={st['p5']:.3f} p25={st['p25']:.3f} "
            f"p75={st['p75']:.3f} p95={st['p95']:.3f} p99={st['p99']:.3f}] "
            f"mean={st['mean']:.3f}")


def _report_one_threshold(
    samples: List[_LocSample],
    *,
    method: str,                # 'full' or 'swin'
    threshold: Optional[float], # None ⇒ no_gate
    threshold_label: str,
    log_tag: str,
    suffix: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Report median F1/IoU per area_bucket under a given gate threshold.

    Below-threshold items contribute (f1=0, iou=0) — the "you said it was real,
    so the predicted mask is empty" case. no_gate means every item passes.
    """
    by_bucket = {'small': [], 'medium': [], 'large': []}
    by_bucket_iou = {'small': [], 'medium': [], 'large': []}
    by_bucket_prec = {'small': [], 'medium': [], 'large': []}
    by_bucket_rec = {'small': [], 'medium': [], 'large': []}
    by_bucket_pred_frac = {'small': [], 'medium': [], 'large': []}
    by_bucket_swin_pass = {'small': [], 'medium': [], 'large': []}
    by_bucket_swin_boundary = {'small': [], 'medium': [], 'large': []}
    n_pass_per_bucket = {'small': 0, 'medium': 0, 'large': 0}
    n_total_per_bucket = {'small': 0, 'medium': 0, 'large': 0}

    for s in samples:
        if method == 'full':
            f1, iou = s.f1_full, s.iou_full
            prec, rec, pred_frac = s.prec_full, s.rec_full, s.pred_frac_full
        elif method == 'gap':
            f1, iou = s.f1_gap, s.iou_gap
            prec, rec, pred_frac = s.prec_gap, s.rec_gap, s.pred_frac_gap
        else:
            if s.f1_swin is None:
                continue
            f1, iou = s.f1_swin, s.iou_swin
            prec, rec, pred_frac = s.prec_swin, s.rec_swin, s.pred_frac_swin

        n_total_per_bucket[s.bucket] += 1
        if threshold is None or s.bce_logit is None:
            # no_gate, OR no BCE head exists ⇒ raw partition contributes
            by_bucket[s.bucket].append(f1)
            by_bucket_iou[s.bucket].append(iou)
            by_bucket_prec[s.bucket].append(prec)
            by_bucket_rec[s.bucket].append(rec)
            by_bucket_pred_frac[s.bucket].append(pred_frac)
            by_bucket_swin_pass[s.bucket].append(float(s.swin_n_pass))
            by_bucket_swin_boundary[s.bucket].append(float(s.swin_n_boundary > 0))
            n_pass_per_bucket[s.bucket] += 1
        elif s.bce_logit >= threshold:
            by_bucket[s.bucket].append(f1)
            by_bucket_iou[s.bucket].append(iou)
            by_bucket_prec[s.bucket].append(prec)
            by_bucket_rec[s.bucket].append(rec)
            by_bucket_pred_frac[s.bucket].append(pred_frac)
            by_bucket_swin_pass[s.bucket].append(float(s.swin_n_pass))
            by_bucket_swin_boundary[s.bucket].append(float(s.swin_n_boundary > 0))
            n_pass_per_bucket[s.bucket] += 1
        else:
            # Gated out: predicted empty mask ⇒ F1=IoU=0
            by_bucket[s.bucket].append(0.0)
            by_bucket_iou[s.bucket].append(0.0)
            by_bucket_prec[s.bucket].append(0.0)
            by_bucket_rec[s.bucket].append(0.0)
            by_bucket_pred_frac[s.bucket].append(0.0)
            by_bucket_swin_pass[s.bucket].append(0.0)
            by_bucket_swin_boundary[s.bucket].append(0.0)

    out = {}
    for b in ('small', 'medium', 'large'):
        f1_st  = _stats(by_bucket[b])
        iou_st = _stats(by_bucket_iou[b])
        prec_st = _stats(by_bucket_prec[b])
        rec_st = _stats(by_bucket_rec[b])
        pred_frac_st = _stats(by_bucket_pred_frac[b])
        swin_pass_st = _stats(by_bucket_swin_pass[b])
        swin_boundary_st = _stats(by_bucket_swin_boundary[b])
        out[b] = {'f1': f1_st, 'iou': iou_st,
                  'precision': prec_st, 'recall': rec_st,
                  'pred_frac': pred_frac_st,
                  'swin_n_pass': swin_pass_st,
                  'swin_boundary_rate': swin_boundary_st,
                  'n_pass': n_pass_per_bucket[b],
                  'n_total': n_total_per_bucket[b]}
        if f1_st['n'] == 0:
            continue
        log_line(
            f'{log_tag}{suffix} loc {method:<4} {threshold_label:<14} '
            f'bucket={b} pass={n_pass_per_bucket[b]}/{n_total_per_bucket[b]} '
            f'f1[{_pct_line(f1_st)} sd={f1_st["std"]:.3f}] '
            f'iou[{_pct_line(iou_st)} sd={iou_st["std"]:.3f}] | '
            f'prec_med={prec_st["median"]:.4f} prec_mean={prec_st["mean"]:.4f} '
            f'rec_med={rec_st["median"]:.4f} rec_mean={rec_st["mean"]:.4f} '
            f'pred_frac_med={pred_frac_st["median"]:.4f}'
            + (f' swin_pos_win_med={swin_pass_st["median"]:.1f} '
               f'swin_boundary_rate={swin_boundary_st["median"]:.2f}'
               if method == 'swin' else '')
        )
    return out


def _report_pixel_summary(
    samples: List[_LocSample],
    *,
    log_tag: str,
    suffix: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per-bucket median PIXEL F1/IoU/prec/rec for the full-image partition.

    Reported ungated — raw partition quality at pixel granularity — as a
    companion to the patch-level threshold sweep. Samples without pixel data
    (no mask on disk, or `res` not passed to collection) are skipped.
    """
    f1   = {'small': [], 'medium': [], 'large': []}
    iou  = {'small': [], 'medium': [], 'large': []}
    prec = {'small': [], 'medium': [], 'large': []}
    rec  = {'small': [], 'medium': [], 'large': []}
    for s in samples:
        if s.f1_full_px is None:
            continue
        f1[s.bucket].append(s.f1_full_px)
        iou[s.bucket].append(s.iou_full_px)
        prec[s.bucket].append(s.prec_full_px)
        rec[s.bucket].append(s.rec_full_px)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for b in ('small', 'medium', 'large'):
        f1_st = _stats(f1[b])
        out[b] = {'f1': f1_st, 'iou': _stats(iou[b]),
                  'precision': _stats(prec[b]), 'recall': _stats(rec[b])}
        if f1_st['n'] == 0:
            continue
        log_line(
            f'{log_tag}{suffix} loc full PIXEL          '
            f'bucket={b} '
            f'f1[{_pct_line(f1_st)} sd={f1_st["std"]:.3f}] '
            f'iou[{_pct_line(out[b]["iou"])} sd={out[b]["iou"]["std"]:.3f}] | '
            f'prec_med={out[b]["precision"]["median"]:.4f} '
            f'prec_mean={out[b]["precision"]["mean"]:.4f} '
            f'rec_med={out[b]["recall"]["median"]:.4f} '
            f'rec_mean={out[b]["recall"]["mean"]:.4f}'
        )
    return out


def report_localization_threshold_sweep(
    samples: List[_LocSample],
    *,
    methods: Sequence[str] = ('full', 'swin'),
    threshold_grid: Sequence[float] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    opt_thresh: Optional[float] = None,
    log_tag: str = '[eval]',
    tag: str = '',
) -> Dict:
    """Print a grouped report covering: no_gate, fixed-grid thresholds, opt.

    Args:
        samples: per-image localization data from `collect_localization_samples`.
        methods: which partition methods to report — 'full' (always available),
                 'swin' (only if collect_localization_samples ran with run_swin=True).
        threshold_grid: BCE logits at which to gate. Skipped if no BCE head.
        opt_thresh: calibrated threshold (e.g., from image-BCE eval). Reported
                    as its own row labeled `opt(xxx)`. If None, omitted.

    Returns:
        Dict[method][threshold_label][bucket] = { f1: stats, iou: stats, ...}
    """
    suffix = f' {tag}' if tag else ''
    if not samples:
        log_line(f'{log_tag}{suffix} loc no samples')
        return {}

    def _q_str(vals: List[float]) -> str:
        a = np.array([v for v in vals if v == v and not np.isnan(v)], dtype=np.float64)
        if a.size == 0:
            return "nan/nan/nan"
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        return f"{q1:.4f}/{med:.4f}/{q3:.4f}(m={float(np.mean(a)):.4f})"

    # Print cluster shape metrics (embedding diagnostics) for k-means partitions
    log_line(f'{log_tag}{suffix} === representation diagnostics (k-means partition) ===')
    for b in ('small', 'medium', 'large'):
        bs = [s for s in samples if s.bucket == b]
        if not bs:
            continue

        pos_cohs = [s.pos_coh for s in bs if s.pos_coh is not None]
        bg_cohs = [s.bg_coh for s in bs if s.bg_coh is not None]
        cent_seps = [s.cent_sep for s in bs if s.cent_sep is not None]
        out_gaps = [s.out_gap for s in bs if s.out_gap is not None]

        mads = [s.frac_outliers_mad for s in bs if s.frac_outliers_mad is not None]
        th03s = [s.frac_outliers_03 for s in bs if s.frac_outliers_03 is not None]
        th05s = [s.frac_outliers_05 for s in bs if s.frac_outliers_05 is not None]

        sil_k2s = [s.sil_k2 for s in bs if s.sil_k2 is not None]
        sil_k3s = [s.sil_k3 for s in bs if s.sil_k3 is not None]
        sil_k4s = [s.sil_k4 for s in bs if s.sil_k4 is not None]

        # Best fitting k percentages
        k2_count = sum(1 for s in bs if s.best_k == 2)
        k3_count = sum(1 for s in bs if s.best_k == 3)
        k4_count = sum(1 for s in bs if s.best_k == 4)
        total = max(1, len(bs))
        k2_pct = (k2_count / total) * 100
        k3_pct = (k3_count / total) * 100
        k4_pct = (k4_count / total) * 100

        log_line(
            f'{log_tag}{suffix} embed_diag bucket={b} n={len(bs)} | '
            f'pos_coh[q1/med/q3]={_q_str(pos_cohs)} | '
            f'bg_coh[q1/med/q3]={_q_str(bg_cohs)} | '
            f'cent_sep[q1/med/q3]={_q_str(cent_seps)} | '
            f'out_gap[q1/med/q3]={_q_str(out_gaps)}'
        )
        log_line(
            f'{log_tag}{suffix} outlier_frac bucket={b} n={len(bs)} | '
            f'mad3.0[q1/med/q3]={_q_str(mads)} | '
            f'th0.3[q1/med/q3]={_q_str(th03s)} | '
            f'th0.5[q1/med/q3]={_q_str(th05s)}'
        )
        log_line(
            f'{log_tag}{suffix} cluster_fit bucket={b} n={len(bs)} | '
            f'sil_k2[q1/med/q3]={_q_str(sil_k2s)} | '
            f'sil_k3[q1/med/q3]={_q_str(sil_k3s)} | '
            f'sil_k4[q1/med/q3]={_q_str(sil_k4s)} | '
            f'best_k_counts(k2/k3/k4)={k2_pct:.1f}%/{k3_pct:.1f}%/{k4_pct:.1f}%'
        )

    has_bce = any(s.bce_logit is not None for s in samples)
    has_swin = any(s.f1_swin is not None for s in samples)

    out: Dict = {m: {} for m in methods}
    for method in methods:
        if method == 'swin' and not has_swin:
            continue

        # 1) no_gate baseline (always)
        out[method]['no_gate'] = _report_one_threshold(
            samples, method=method, threshold=None,
            threshold_label='no_gate',
            log_tag=log_tag, suffix=suffix,
        )

        # 2) Fixed-grid sweep (skip if no BCE head)
        if has_bce:
            for t in threshold_grid:
                label = f't={float(t):+.3f}'
                out[method][label] = _report_one_threshold(
                    samples, method=method, threshold=float(t),
                    threshold_label=label,
                    log_tag=log_tag, suffix=suffix,
                )

        # 3) Calibrated opt threshold (if provided)
        if has_bce and opt_thresh is not None:
            label = f'opt({opt_thresh:+.3f})'
            out[method][label] = _report_one_threshold(
                samples, method=method, threshold=float(opt_thresh),
                threshold_label=label,
                log_tag=log_tag, suffix=suffix,
            )

    # PIXEL companion: raw full-image partition quality at pixel granularity,
    # ungated (gate-independent), printed alongside the patch-level sweep so the
    # two granularities sit side by side.
    if 'full' in methods and any(s.f1_full_px is not None for s in samples):
        out.setdefault('full', {})['pixel'] = _report_pixel_summary(
            samples, log_tag=log_tag, suffix=suffix,
        )

    return out


# ── per-bucket summary (for the corruption sweep) ────────────────────────────

def summarize_localization(
    samples: List[_LocSample],
) -> Dict[str, Dict[str, float]]:
    """Per-bucket median patch + pixel F1/IoU/prec/rec for a _LocSample list.

    A compact, gate-free summary (raw oracle-polarity partition quality) the
    corruption sweep tabulates per condition with Δ-vs-clean. Buckets with no
    samples report {'n': 0}.
    """
    out: Dict[str, Dict[str, float]] = {}
    for b in ('small', 'medium', 'large'):
        bs = [s for s in samples if s.bucket == b]
        if not bs:
            out[b] = {'n': 0}
            continue
        d: Dict[str, float] = {
            'n':         len(bs),
            'f1_patch':  float(np.median([s.f1_full for s in bs])),
            'iou_patch': float(np.median([s.iou_full for s in bs])),
            'prec_patch': float(np.median([s.prec_full for s in bs])),
            'rec_patch': float(np.median([s.rec_full for s in bs])),
        }
        px = [s for s in bs if s.f1_full_px is not None]
        if px:
            d.update({
                'f1_px':  float(np.median([s.f1_full_px for s in px])),
                'iou_px': float(np.median([s.iou_full_px for s in px])),
                'prec_px': float(np.median([s.prec_full_px for s in px])),
                'rec_px': float(np.median([s.rec_full_px for s in px])),
            })
        out[b] = d
    return out


# ── coarse→fine prediction-guided zoom refine ────────────────────────────────
#
# Pass 1: full-image partition flags the rough splice region (high recall, low
# precision — it over-covers). Pass 2: crop the SOURCE to that region's bbox +
# padding, re-segment at higher effective resolution, map the refined mask back
# into the input frame. One window per pass → no cross-window polarity problem;
# needs no BCE head. Scored at PIXEL granularity (the refine's gain is sub-patch,
# so patch-level on the full grid would just reproduce the coarse pass).

@dataclass
class _CFSample:
    kind: str
    area: float
    bucket: str
    coarse_f1: float
    coarse_iou: float
    coarse_prec: float
    coarse_rec: float
    refine_f1: float
    refine_iou: float
    refine_prec: float
    refine_rec: float
    refined: bool        # True iff pass 2 actually ran (region small enough)


@torch.no_grad()
def _embed_pil(
    model: nn.Module, pil: Image.Image, res: Resolution, device: torch.device,
    mean: Tuple[float, float, float], std: Tuple[float, float, float],
) -> Optional[np.ndarray]:
    """Resize a PIL image to the input frame, normalize, forward → (N, d) embeds.

    Uses the same `resize_only` squash the dataset/GT use, so the patch grid and
    the GT pixel mask share one coordinate frame.
    """
    img = resize_only(pil.convert('RGB'), res)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # (S, S, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1)               # (3, S, S)
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    t = ((t - m) / s).unsqueeze(0).to(device)
    out = model(t)
    z = out['contrastive']
    if z is None:
        return None
    return z[0].detach().cpu().float().numpy()               # (N, d)


def _minority_bbox(
    raw_labels: np.ndarray, P: int, pad_frac: float,
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """Patch-grid bbox (r0, r1, c0, c1) of the MINORITY cluster + padding.

    The minority cluster is the splice candidate (a splice is typically the
    smaller region). Returns (None, frac) if degenerate. `frac` is the minority
    cluster's patch fraction — the caller skips the zoom when it's too large
    (large splices already localize well, and minority=background would invert).
    This is a pure geometry heuristic — no attention, no GT.
    """
    grid = np.asarray(raw_labels).reshape(P, P)
    n1 = int((grid == 1).sum())
    n0 = grid.size - n1
    minority = 1 if n1 <= n0 else 0
    m = (grid == minority)
    if not m.any():
        return None, 0.0
    minority_frac = float(m.sum()) / m.size
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    dr = max(1, int(round((r1 - r0) * pad_frac)))
    dc = max(1, int(round((c1 - c0) * pad_frac)))
    r0 = max(0, r0 - dr); r1 = min(P, r1 + dr)
    c0 = max(0, c0 - dc); c1 = min(P, c1 + dc)
    return (r0, r1, c0, c1), minority_frac


def _place_fine_in_pixel_frame(
    fine_pred_flat: np.ndarray,
    grid_bbox: Tuple[int, int, int, int],
    res: Resolution,
) -> np.ndarray:
    """Map a pass-2 (N,) partition onto the input frame at its bbox location.

    The crop's own (S, S) prediction is downsampled (nearest) to the bbox's size
    in input-frame pixels and pasted there; everything outside the bbox is
    background. Returns an (S, S) bool mask.
    """
    P = res.num_patches_per_side
    ps = res.patch_size
    S = res.image_size
    r0, r1, c0, c1 = grid_bbox
    fine_px_crop = _patches_to_pixels(fine_pred_flat, P, ps)      # (S, S) bool, crop frame
    th = (r1 - r0) * ps
    tw = (c1 - c0) * ps
    pil = Image.fromarray((fine_px_crop.astype(np.uint8) * 255), mode='L').resize(
        (tw, th), Image.NEAREST
    )
    placed = np.asarray(pil) > 127                                # (th, tw) bool
    full = np.zeros((S, S), dtype=bool)
    full[r0 * ps:r1 * ps, c0 * ps:c1 * ps] = placed
    return full


def _oracle_pixel(
    pred_a: np.ndarray, pred_b: np.ndarray, gt: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Oracle-polarity pixel metrics: better of the two labelings by F1/IoU."""
    ma = _mask_metrics(pred_a.reshape(-1), gt.reshape(-1))
    mb = _mask_metrics(pred_b.reshape(-1), gt.reshape(-1))
    chosen = ma if (ma[0], ma[1]) >= (mb[0], mb[1]) else mb
    return chosen[0], chosen[1], chosen[2], chosen[3]


@torch.no_grad()
def collect_coarse_to_fine_samples(
    model: nn.Module,
    items: List[Dict],
    device: torch.device,
    *,
    res: Resolution,
    normalize_mean: Tuple[float, float, float] = _DEFAULT_NORMALIZE_MEAN,
    normalize_std:  Tuple[float, float, float] = _DEFAULT_NORMALIZE_STD,
    pad_frac: float = 0.25,
    refine_max_frac: float = 0.40,
    corruption_spec: Optional[CorruptionSpec] = None,
    log_tag: str = '[eval]',
    tag: str = '',
) -> List[_CFSample]:
    """Two-pass prediction-guided zoom refine, scored at pixel granularity.

    Iterates splice ITEMS (loads source + mask from disk so pass 2 zooms at full
    source resolution). Reals are skipped (no segmentation target). The refine is
    reported AS-IS (no best-of-coarse/refine selection) so the coarse→refine Δ is
    an honest measure of whether zooming helps. When the flagged region is larger
    than `refine_max_frac`, the zoom is skipped and refined := coarse (a
    deployable policy: don't zoom when the region is already big).
    """
    model.eval()
    P = res.num_patches_per_side
    suffix = f' {tag}' if tag else ''
    samples: List[_CFSample] = []
    n_seen = 0
    n_refined = 0
    for it in items:
        if it.get('kind', '') not in _SPLICE_KINDS:
            continue
        img_path = str(it.get('img') or it.get('path') or '')
        mask_path = str(it.get('mask') or it.get('mask_path') or '')
        if not img_path or not mask_path:
            continue
        try:
            src = Image.open(img_path).convert('RGB')
        except Exception:
            continue
        gt_px = _load_gt_pixel_mask({'mask_path': mask_path}, res)
        if gt_px is None:
            continue
        area = float(gt_px.mean())   # clean-mask area (corruption never touches GT)
        bucket = _bucket(area)

        # Optional WHOLE-IMAGE corruption applied to the source before BOTH
        # passes — the deployed pipeline receives a degraded image and has no
        # clean copy to zoom into. Lets us ask: does the zoom recover precision
        # the corruption cost? (Global-region only; GT stays clean.)
        if corruption_spec is not None:
            src = apply_corruption(src, corruption_spec).image

        # PASS 1 — coarse full-image partition.
        z1 = _embed_pil(model, src, res, device, normalize_mean, normalize_std)
        if z1 is None:
            continue
        raw1, _ = spherical_kmeans2(z1, n_init=4)
        cpx_a = _patches_to_pixels((raw1 == 1), P, res.patch_size)
        cpx_b = ~cpx_a
        c_f1, c_iou, c_prec, c_rec = _oracle_pixel(cpx_a, cpx_b, gt_px)

        # PASS 2 — refine, only when the flagged region is compact enough.
        bbox, minority_frac = _minority_bbox(raw1, P, pad_frac)
        refined = False
        r_f1, r_iou, r_prec, r_rec = c_f1, c_iou, c_prec, c_rec
        if bbox is not None and minority_frac <= refine_max_frac:
            r0, r1, c0, c1 = bbox
            W, H = src.size
            x0 = int(round(c0 / P * W)); x1 = max(x0 + 1, int(round(c1 / P * W)))
            y0 = int(round(r0 / P * H)); y1 = max(y0 + 1, int(round(r1 / P * H)))
            z2 = _embed_pil(model, src.crop((x0, y0, x1, y1)), res, device,
                            normalize_mean, normalize_std)
            if z2 is not None:
                raw2, _ = spherical_kmeans2(z2, n_init=4)
                ref_a = _place_fine_in_pixel_frame((raw2 == 1), bbox, res)
                ref_b = _place_fine_in_pixel_frame((raw2 == 0), bbox, res)
                r_f1, r_iou, r_prec, r_rec = _oracle_pixel(ref_a, ref_b, gt_px)
                refined = True
                n_refined += 1

        samples.append(_CFSample(
            kind=it.get('kind', ''), area=area, bucket=bucket,
            coarse_f1=c_f1, coarse_iou=c_iou, coarse_prec=c_prec, coarse_rec=c_rec,
            refine_f1=r_f1, refine_iou=r_iou, refine_prec=r_prec, refine_rec=r_rec,
            refined=refined,
        ))
        n_seen += 1

    log_line(f'{log_tag}{suffix} coarse2fine collected n_splice={n_seen} '
             f'refined={n_refined} (pad_frac={pad_frac} refine_max_frac={refine_max_frac})')
    return samples


def report_coarse_to_fine(
    samples: List[_CFSample], *, log_tag: str = '[eval]', tag: str = '',
) -> Dict[str, Dict[str, float]]:
    """Per-bucket coarse→refine pixel F1/IoU/prec/rec with Δ. Refine should lift
    precision (it shrinks the over-coverage) at little recall cost."""
    suffix = f' {tag}' if tag else ''
    if not samples:
        log_line(f'{log_tag}{suffix} coarse2fine no samples')
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for b in ('small', 'medium', 'large'):
        bs = [s for s in samples if s.bucket == b]
        if not bs:
            continue
        cf1_st = _stats([s.coarse_f1 for s in bs])
        ci_st  = _stats([s.coarse_iou for s in bs])
        cpr_st = _stats([s.coarse_prec for s in bs])
        crc_st = _stats([s.coarse_rec for s in bs])
        rf1_st = _stats([s.refine_f1 for s in bs])
        ri_st  = _stats([s.refine_iou for s in bs])
        rpr_st = _stats([s.refine_prec for s in bs])
        rrc_st = _stats([s.refine_rec for s in bs])
        cf1, rf1 = cf1_st['median'], rf1_st['median']
        cpr, rpr = cpr_st['median'], rpr_st['median']
        n_ref = sum(1 for s in bs if s.refined)
        out[b] = {'n': len(bs), 'n_refined': n_ref,
                  'coarse_f1': cf1, 'refine_f1': rf1,
                  'coarse_f1_mean': cf1_st['mean'], 'refine_f1_mean': rf1_st['mean'],
                  'coarse_prec': cpr, 'refine_prec': rpr}
        log_line(
            f'{log_tag}{suffix} c2f bucket={b} n={len(bs)} refined={n_ref} | '
            f'coarse f1={cf1:.4f}(m={cf1_st["mean"]:.4f}) '
            f'iou={ci_st["median"]:.4f}(m={ci_st["mean"]:.4f}) '
            f'prec={cpr:.4f}(m={cpr_st["mean"]:.4f}) '
            f'rec={crc_st["median"]:.4f}(m={crc_st["mean"]:.4f}) -> '
            f'refine f1={rf1:.4f}(m={rf1_st["mean"]:.4f}) '
            f'iou={ri_st["median"]:.4f}(m={ri_st["mean"]:.4f}) '
            f'prec={rpr:.4f}(m={rpr_st["mean"]:.4f}) '
            f'rec={rrc_st["median"]:.4f}(m={rrc_st["mean"]:.4f}) | '
            f'Δf1={rf1 - cf1:+.4f} Δprec={rpr - cpr:+.4f} '
            f'Δrec={rrc_st["median"] - crc_st["median"]:+.4f}'
        )
        log_line(f'{log_tag}{suffix} c2f bucket={b} coarse f1 {_pct_line(cf1_st)}')
        log_line(f'{log_tag}{suffix} c2f bucket={b} refine f1 {_pct_line(rf1_st)}')
    return out


# ── detector-grounded localization (deployed operating point) ────────────────

def report_grounded_localization(
    loc_samples: List[_LocSample],
    real_logits: Sequence[float],
    *,
    tnr_target: float = 0.95,
    log_tag: str = '[eval]',
    tag: str = '',
) -> Dict:
    """Localization grounded in the deployed detector at a fixed operating point.

    Picks the image-logit threshold ``t`` where TNR == ``tnr_target`` (so a
    ``1 - tnr_target`` fraction of reals are FALSE POSITIVES), then scores
    localization as the pipeline actually behaves:

      - splice ABOVE t (TP): its oracle-polarity loc IoU/F1.
      - splice BELOW t (FN): 0 — detector said "real", no mask emitted.
      - real   ABOVE t (FP): 0 — a spurious mask on a clean image.
      - real   BELOW t (TN): excluded — correctly silent, no prediction/target.

    The FN/FP zeros are what make this lower (and honest) than the ungated
    partition quality. Reports patch AND pixel; medians can saturate to 0 when
    the detector misses a lot, so an ``ALL+FP`` mean is also printed.

    Requires loc_samples to carry ``bce_logit`` (a BCE-having model).
    """
    suffix = f' {tag}' if tag else ''
    if not loc_samples:
        log_line(f'{log_tag}{suffix} grounded-loc no splice samples')
        return {}
    if any(s.bce_logit is None for s in loc_samples):
        log_line(f'{log_tag}{suffix} grounded-loc needs an image-BCE head (no logits); skipping')
        return {}
    real = np.asarray(list(real_logits), dtype=np.float64)
    if real.size == 0:
        log_line(f'{log_tag}{suffix} grounded-loc no real logits; skipping')
        return {}

    splice_logits = np.array([s.bce_logit for s in loc_samples], dtype=np.float64)
    n_splice = len(loc_samples)
    n_real = int(real.size)

    # Threshold at TNR = tnr_target: keep tnr_target of reals below t.
    t = float(np.quantile(real, tnr_target))
    n_fp = int((real >= t).sum())
    fpr = n_fp / max(n_real, 1)
    n_tp = int((splice_logits >= t).sum())
    n_fn = n_splice - n_tp
    tpr = n_tp / max(n_splice, 1)

    # Combined detection AUC (splice=1, real=0).
    all_logits = np.concatenate([splice_logits, real])
    all_labels = np.concatenate([np.ones(n_splice), np.zeros(n_real)]).astype(np.int32)
    try:
        order = np.argsort(-all_logits)
        sl = all_labels[order]
        tpr_pts = np.cumsum(sl) / max(n_splice, 1)
        fpr_pts = np.cumsum(1 - sl) / max(n_real, 1)
        auc = float(np.trapezoid(tpr_pts, fpr_pts))
        if auc < 0:
            auc = 1.0 + auc
    except Exception:
        auc = float('nan')

    log_line(
        f'{log_tag}{suffix} GROUNDED @ TNR={tnr_target:.2f} thresh={t:+.3f} '
        f'det_auc={auc:.4f} tpr={tpr:.4f} fpr={fpr:.4f} '
        f'n_splice={n_splice} n_TP={n_tp} n_FN={n_fn} n_real={n_real} n_FP={n_fp}'
    )

    def _gated(samples, attr):
        return [(getattr(s, attr) if (s.bce_logit >= t and getattr(s, attr) is not None)
                 else 0.0) for s in samples]

    out: Dict = {'thresh': t, 'auc': auc, 'tpr': tpr, 'fpr': fpr,
                 'n_tp': n_tp, 'n_fn': n_fn, 'n_fp': n_fp}

    for b in ('small', 'medium', 'large'):
        bs = [s for s in loc_samples if s.bucket == b]
        if not bs:
            continue
        f1p = _gated(bs, 'f1_full'); ioup = _gated(bs, 'iou_full')
        f1x = _gated(bs, 'f1_full_px'); ioux = _gated(bs, 'iou_full_px')
        n_det = sum(1 for s in bs if s.bce_logit >= t)
        log_line(
            f'{log_tag}{suffix} grounded-loc bucket={b} n={len(bs)} detected={n_det}/{len(bs)} | '
            f'f1_patch={float(np.median(f1p)):.4f}(m={float(np.mean(f1p)):.4f}) '
            f'iou_patch={float(np.median(ioup)):.4f}(m={float(np.mean(ioup)):.4f}) | '
            f'f1_px={float(np.median(f1x)):.4f}(m={float(np.mean(f1x)):.4f}) '
            f'iou_px={float(np.median(ioux)):.4f}(m={float(np.mean(ioux)):.4f}) (FN=0)'
        )
        out[b] = {'n': len(bs), 'detected': n_det,
                  'f1_patch': float(np.median(f1p)), 'iou_patch': float(np.median(ioup)),
                  'f1_px': float(np.median(f1x)), 'iou_px': float(np.median(ioux))}

    # ALL splices (gated) + FP reals as zeros → the single grounded number.
    all_f1x = _gated(loc_samples, 'f1_full_px') + [0.0] * n_fp
    all_ioux = _gated(loc_samples, 'iou_full_px') + [0.0] * n_fp
    all_f1p = _gated(loc_samples, 'f1_full') + [0.0] * n_fp
    all_ioup = _gated(loc_samples, 'iou_full') + [0.0] * n_fp
    log_line(
        f'{log_tag}{suffix} grounded-loc ALL+FP n={len(all_ioux)} '
        f'(splice={n_splice} +FP={n_fp}) | '
        f'f1_patch={float(np.median(all_f1p)):.4f}(m={float(np.mean(all_f1p)):.4f}) '
        f'iou_patch={float(np.median(all_ioup)):.4f}(m={float(np.mean(all_ioup)):.4f}) | '
        f'f1_px={float(np.median(all_f1x)):.4f}(m={float(np.mean(all_f1x)):.4f}) '
        f'iou_px={float(np.median(all_ioux)):.4f}(m={float(np.mean(all_ioux)):.4f})'
    )
    out['all_fp'] = {'n': len(all_ioux), 'f1_px': float(np.median(all_f1x)),
                     'iou_px': float(np.median(all_ioux)),
                     'mean_iou_px': float(np.mean(all_ioux))}
    return out


def report_oracle_tax(
    loc_samples: List['_LocSample'], *,
    log_tag: str = '[eval]', tag: str = '',
) -> Dict:
    """The 'oracle tax': how much oracle polarity inflates the headline IoU over
    the DEPLOYED polarity (attention / smaller-cluster), per size bucket.

    A large tax or low ``polarity_agree`` means the headline oracle IoU is not
    reproducible at deployment — the partition is near coin-flip and the oracle
    is just picking the lucky side. Small tax + high agreement ⇒ the oracle is
    harmless and the number is trustworthy.
    """
    suffix = f' {tag}' if tag else ''
    samples = [s for s in loc_samples if s.iou_full_deployed is not None]
    if not samples:
        return {}
    out: Dict = {}
    for b in ('small', 'medium', 'large'):
        bs = [s for s in samples if s.bucket == b]
        if not bs:
            continue
        orc = _stats([s.iou_full for s in bs])
        dep = _stats([s.iou_full_deployed for s in bs])
        tax = _stats([s.iou_full - s.iou_full_deployed for s in bs])
        agree = float(np.mean([1.0 if s.polarity_agree else 0.0 for s in bs]))
        out[b] = {'oracle': orc, 'deployed': dep, 'tax': tax, 'agree_rate': agree}
        log_line(f'{log_tag}{suffix} oracle-tax size={b:<6} n={orc["n"]:<3} '
                 f"oracle_iou_med={orc['median']:.3f}(m={orc['mean']:.3f}) "
                 f"deployed_iou_med={dep['median']:.3f}(m={dep['mean']:.3f}) "
                 f"tax_med={tax['median']:+.3f}(m={tax['mean']:+.3f}) "
                 f"polarity_agree={agree:.2f}")
    return out


def report_loc_by_confidence(
    loc_samples: List['_LocSample'], *,
    t_op: Optional[float] = None,
    log_tag: str = '[eval]', tag: str = '',
) -> Dict:
    """FREE re-slice of full-image localization by detector confidence × size.

    Splits splices into missed (logit < t_op) / less-confident / confident
    (>= median logit of the detected) and reports full-image IoU per tier per
    size bucket — the detection↔localization entanglement, at zero extra cost
    (reuses the bce_logit + f1_full/iou_full already on each _LocSample).
    """
    suffix = f' {tag}' if tag else ''
    samples = [s for s in loc_samples if s.bce_logit is not None]
    if not samples:
        return {}
    thr = 0.0 if t_op is None else float(t_op)
    det = [s.bce_logit for s in samples if s.bce_logit >= thr]
    conf_cut = float(np.median(det)) if det else thr
    log_line(f'{log_tag}{suffix} loc-by-conf (full-image; '
             f'missed<{thr:+.2f} <=less< {conf_cut:+.2f} <=confident) '
             f'— detection/localization entanglement')
    out: Dict = {}
    for b in ('small', 'medium', 'large'):
        bs = [s for s in samples if s.bucket == b]
        if not bs:
            continue
        tiers: Dict[str, List[float]] = {'confident': [], 'less': [], 'missed': []}
        for s in bs:
            lg = float(s.bce_logit)
            key = 'missed' if lg < thr else ('confident' if lg >= conf_cut else 'less')
            tiers[key].append(s.iou_full)
        row = {}
        for k in ('confident', 'less', 'missed'):
            st = _stats(tiers[k])
            row[k] = st
            if st['n']:
                log_line(f'{log_tag}{suffix} loc-by-conf size={b:<6} det={k:<9} '
                         f"n={st['n']:<3} iou_med={st['median']:.3f} "
                         f"iou_p25={st['p25']:.3f} iou_mean={st['mean']:.3f}")
        out[b] = row
    return out


# ── seeded zoom-eval: paired full / natural-zoom / oracle-zoom localization ───
#
# Per val splice we synthesize an EVEN spread of in-frame splice sizes by
# assigning each item a deterministic target coverage (seeded by image path, so
# the val set is byte-identical every epoch and across runs). We then score, all
# oracle-polarity at PIXEL granularity, all non-destructive (geometric crops, no
# photometric corruption):
#   - full   : whole-frame resize_only (the deployment-natural baseline)
#   - zoom    : a RANDOM-position crop sized to the target coverage — a real
#               sliding window does NOT know where the splice is (off-center,
#               may clip the splice)
#   - oracle  : a mask-centered crop into the object (the targeting upper bound)
# Plus the image-BCE logit at full and at zoom (free — same forward) so we can
# slice localization by detector confidence at no extra cost.

@dataclass
class _ZoomSample:
    kind: str
    area_full: float           # native splice area fraction (full frame)
    target_cov: float          # assigned in-frame coverage target (even spread)
    realized_cov: float        # actual in-frame coverage of the natural-zoom crop
    bucket_zoom: str           # size bucket of the natural-zoom in-frame coverage
    f1_full: float
    iou_full: float
    f1_zoom: float
    iou_zoom: float
    f1_oracle: float
    iou_oracle: float
    bce_logit_full: Optional[float]
    bce_logit_zoom: Optional[float]


@torch.no_grad()
def _embed_logit_pil(
    model: nn.Module, pil: Image.Image, res: Resolution, device: torch.device,
    mean: Tuple[float, float, float], std: Tuple[float, float, float],
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """One forward of a PIL crop → (contrastive z (N, d) or None, image_logit or None)."""
    img = resize_only(pil.convert('RGB'), res)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    t = ((t - m) / s).unsqueeze(0).to(device)
    out = model(t)
    z = out['contrastive']
    z_np = z[0].detach().cpu().float().numpy() if z is not None else None
    logit = (float(out['image_logit'][0].detach().cpu().float())
             if out.get('image_logit') is not None else None)
    return z_np, logit


def _oracle_pixel_from_z(
    z_np: Optional[np.ndarray], gt_px: Optional[np.ndarray], res: Resolution,
) -> Tuple[float, float]:
    """k-means(2) → oracle-polarity PIXEL (f1, iou) vs gt_px (bool, frame-sized)."""
    if z_np is None or gt_px is None or not gt_px.any():
        return 0.0, 0.0
    raw, _ = spherical_kmeans2(z_np, n_init=4)
    P, ps = res.num_patches_per_side, res.patch_size
    px_a = _patches_to_pixels((raw == 1), P, ps)
    f1, iou, _, _ = _oracle_pixel(px_a, ~px_a, gt_px)
    return f1, iou


def _seeded_natural_crop(src, mask, res, target_cov, rng):
    """Random-POSITION square crop sized so a fully-contained splice ≈ target_cov.

    Off-center allowed (the splice may be partially clipped) — the honest
    'sliding window does not know where the splice is' view, unlike the oracle.
    Returns (crop_img PIL, crop_mask PIL 'L', realized_cov) or (None, None, 0).
    """
    m = np.asarray(mask.convert('L'), dtype=np.uint8) > 0
    H, W = m.shape
    if not m.any():
        return None, None, 0.0
    ys, xs = np.where(m)
    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    splice_px = float(m.sum())
    side = int(round((splice_px / max(float(target_cov), 1e-6)) ** 0.5))
    side = max(8, min(side, H, W))
    max_top, max_left = H - side, W - side
    # Range of top/left that keeps the crop OVERLAPPING the splice bbox.
    top_lo  = max(0, min(max_top,  r0 - side + 1))
    top_hi  = max(0, min(max_top,  r1 - 1))
    left_lo = max(0, min(max_left, c0 - side + 1))
    left_hi = max(0, min(max_left, c1 - 1))
    top  = int(rng.integers(min(top_lo, top_hi),   max(top_lo, top_hi)  + 1))
    left = int(rng.integers(min(left_lo, left_hi), max(left_lo, left_hi) + 1))
    crop_img  = src.crop((left, top, left + side, top + side))
    crop_mask = mask.crop((left, top, left + side, top + side))
    cov = float(m[top:top + side, left:left + side].sum()) / float(side * side)
    return crop_img, crop_mask, cov


@torch.no_grad()
def collect_zoom_eval_samples(
    model: nn.Module,
    items: List[Dict],
    device: torch.device,
    *,
    res: Resolution,
    cov_range: Tuple[float, float] = (0.05, 0.55),
    seed: str = 'zoomval',
    normalize_mean: Tuple[float, float, float] = _DEFAULT_NORMALIZE_MEAN,
    normalize_std:  Tuple[float, float, float] = _DEFAULT_NORMALIZE_STD,
    oracle_jitter: float = 0.15,
    skip_oracle: bool = False,
    log_tag: str = '[eval]',
    tag: str = '',
) -> List[_ZoomSample]:
    """Paired full / seeded natural-zoom / oracle-zoom localization per splice.

    Deterministic: each item's target coverage + crop placement are seeded by
    its path, so the val zoom set is identical every epoch. Non-destructive
    (geometric crops only). Reals are skipped.
    """
    model.eval()
    suffix = f' {tag}' if tag else ''
    lo_c, hi_c = float(cov_range[0]), float(cov_range[1])
    samples: List[_ZoomSample] = []
    for it in items:
        if it.get('kind', '') not in _SPLICE_KINDS:
            continue
        img_path  = str(it.get('img') or it.get('path') or '')
        mask_path = str(it.get('mask') or it.get('mask_path') or '')
        if not img_path or not mask_path:
            continue
        try:
            src = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        except Exception:
            continue
        if int(np.asarray(mask).max()) == 0:
            continue
        gt_full = _load_gt_pixel_mask({'mask_path': mask_path}, res)
        if gt_full is None:
            continue
        area_full = float(gt_full.mean())

        h = int(hashlib.md5(f'{seed}|{img_path}'.encode('utf-8')).hexdigest()[:16], 16)
        rng = np.random.default_rng(h)
        target_cov = float(rng.uniform(lo_c, hi_c))

        # FULL frame.
        z_f, logit_f = _embed_logit_pil(model, src, res, device, normalize_mean, normalize_std)
        f1_f, iou_f = _oracle_pixel_from_z(z_f, gt_full, res)

        # NATURAL zoom (random position).
        cz, mz, realized = _seeded_natural_crop(src, mask, res, target_cov, rng)
        if cz is None:
            continue
        gt_z = np.asarray(resize_only_mask(mz, res).convert('L')) > 127
        z_z, logit_z = _embed_logit_pil(model, cz, res, device, normalize_mean, normalize_std)
        f1_z, iou_z = _oracle_pixel_from_z(z_z, gt_z, res)

        # ORACLE zoom (mask-centered into the object).
        if skip_oracle:
            f1_o, iou_o = 0.0, 0.0
        else:
            oc = oracle_mask_crop(
                src, mask, res,
                target_cov_range=(max(0.02, target_cov * 0.85), min(0.95, target_cov * 1.15)),
                jitter_frac=float(oracle_jitter), rng=rng,
            )
            if oc.valid and oc.mask is not None:
                gt_o = np.asarray(oc.mask.convert('L')) > 127
                z_o, _ = _embed_logit_pil(model, oc.image, res, device, normalize_mean, normalize_std)
                f1_o, iou_o = _oracle_pixel_from_z(z_o, gt_o, res)
            else:
                f1_o, iou_o = f1_z, iou_z   # couldn't crop (empty mask) — fall back to zoom

        samples.append(_ZoomSample(
            kind=it.get('kind', ''), area_full=area_full,
            target_cov=target_cov, realized_cov=realized, bucket_zoom=_bucket(realized),
            f1_full=f1_f, iou_full=iou_f, f1_zoom=f1_z, iou_zoom=iou_z,
            f1_oracle=f1_o, iou_oracle=iou_o,
            bce_logit_full=logit_f, bce_logit_zoom=logit_z,
        ))

    log_line(f'{log_tag}{suffix} zoom-eval collected n_splice={len(samples)} '
             f'cov_range=({lo_c:.2f},{hi_c:.2f}) seed={seed!r}')
    return samples


def report_zoom_eval(
    samples: List[_ZoomSample], *,
    t_op: Optional[float] = None,
    condensed: bool = False,
    log_tag: str = '[eval]', tag: str = '',
) -> Dict:
    """Report the three zoom-eval views requested for the localization pulse.

    A. Per in-frame size bucket: full / zoom / oracle IoU with a non-normal-aware
       distribution (p1/p5/p25/median/p75/p95/p99 + mean) — exposes the bimodal
       lukewarm-vs-catastrophic shape.
    B. Detector-confidence × size: zoom IoU split by the zoom-crop BCE logit
       (missed / less-confident / confident) — are detection and localization
       entangled? (Free re-slice; no extra forward.)
    C. Improvement by ORIGINAL score: Δ(zoom-full) and Δ(oracle-full) stratified
       by full-IoU quartile — did zoom rescue catastrophic originals or only
       nudge already-good ones?
    """
    suffix = f' {tag}' if tag else ''
    if not samples:
        log_line(f'{log_tag}{suffix} zoom-eval no samples')
        return {}
    buckets = ('small', 'medium', 'large')
    out: Dict = {}

    # ── Condensed mode: one line per size + aggregate, FULL + ZOOM only ──
    if condensed:
        out['by_size'] = {}
        all_full_iou: List[float] = []
        all_zoom_iou: List[float] = []
        all_full_f1: List[float] = []
        all_zoom_f1: List[float] = []
        for b in buckets:
            bs = [s for s in samples if s.bucket_zoom == b]
            if not bs:
                continue
            full_iou = _stats([s.iou_full for s in bs])
            zoom_iou = _stats([s.iou_zoom for s in bs])
            full_f1 = _stats([s.f1_full for s in bs])
            zoom_f1 = _stats([s.f1_zoom for s in bs])
            log_line(f'{log_tag}{suffix} zoom size={b:<6} '
                     f'FULL iou {_pct_line(full_iou)} f1 {_pct_line(full_f1)}  '
                     f'ZOOM iou {_pct_line(zoom_iou)} f1 {_pct_line(zoom_f1)}')
            out['by_size'][b] = {'full': full_iou, 'zoom': zoom_iou,
                                 'full_f1': full_f1, 'zoom_f1': zoom_f1}
            all_full_iou.extend(s.iou_full for s in bs)
            all_zoom_iou.extend(s.iou_zoom for s in bs)
            all_full_f1.extend(s.f1_full for s in bs)
            all_zoom_f1.extend(s.f1_zoom for s in bs)
        if all_full_iou:
            agg_full = _stats(all_full_iou)
            agg_zoom = _stats(all_zoom_iou)
            agg_full_f1 = _stats(all_full_f1)
            agg_zoom_f1 = _stats(all_zoom_f1)
            log_line(f'{log_tag}{suffix} zoom aggregate  '
                     f'FULL iou {_pct_line(agg_full)} f1 {_pct_line(agg_full_f1)}  '
                     f'ZOOM iou {_pct_line(agg_zoom)} f1 {_pct_line(agg_zoom_f1)}')
            out['aggregate'] = {'full': agg_full, 'zoom': agg_zoom,
                                'full_f1': agg_full_f1, 'zoom_f1': agg_zoom_f1}
        return out

    # ── Full mode: per-size FULL / ZOOM / ORACLE + confidence + delta ──
    out['by_size'] = {}
    for b in buckets:
        bs = [s for s in samples if s.bucket_zoom == b]
        if not bs:
            continue
        full_iou = _stats([s.iou_full for s in bs])
        zoom_iou = _stats([s.iou_zoom for s in bs])
        orac_iou = _stats([s.iou_oracle for s in bs])
        full_f1 = _stats([s.f1_full for s in bs])
        zoom_f1 = _stats([s.f1_zoom for s in bs])
        orac_f1 = _stats([s.f1_oracle for s in bs])
        cov_med = float(np.median([s.realized_cov for s in bs]))
        log_line(f'{log_tag}{suffix} zoom size={b:<6} cov_med={cov_med:.2f} '
                 f'FULL   iou {_pct_line(full_iou)} f1 {_pct_line(full_f1)}')
        log_line(f'{log_tag}{suffix} zoom size={b:<6} cov_med={cov_med:.2f} '
                 f'ZOOM   iou {_pct_line(zoom_iou)} f1 {_pct_line(zoom_f1)}')
        log_line(f'{log_tag}{suffix} zoom size={b:<6} cov_med={cov_med:.2f} '
                 f'ORACLE iou {_pct_line(orac_iou)} f1 {_pct_line(orac_f1)}')
        out['by_size'][b] = {'full': full_iou, 'zoom': zoom_iou,
                             'oracle': orac_iou, 'realized_cov_med': cov_med,
                             'full_f1': full_f1, 'zoom_f1': zoom_f1,
                             'oracle_f1': orac_f1}

    # ── B. detector-confidence × size (view-matched zoom logit) ──
    if all(s.bce_logit_zoom is not None for s in samples):
        thr = 0.0 if t_op is None else float(t_op)
        det = [s.bce_logit_zoom for s in samples if s.bce_logit_zoom >= thr]
        conf_cut = float(np.median(det)) if det else thr
        log_line(f'{log_tag}{suffix} zoom CONFIDENCE×SIZE (zoom-logit: '
                 f'missed<{thr:+.2f} <=less< {conf_cut:+.2f} <=confident) '
                 f'— detection/localization entanglement')
        out['confidence'] = {}
        for b in buckets:
            bs = [s for s in samples if s.bucket_zoom == b]
            if not bs:
                continue
            tiers: Dict[str, List[float]] = {'confident': [], 'less': [], 'missed': []}
            for s in bs:
                lg = float(s.bce_logit_zoom)
                key = 'missed' if lg < thr else ('confident' if lg >= conf_cut else 'less')
                tiers[key].append(s.iou_zoom)
            row = {}
            for k in ('confident', 'less', 'missed'):
                st = _stats(tiers[k])
                row[k] = st
                if st['n']:
                    log_line(f'{log_tag}{suffix} zoom size={b:<6} det={k:<9} '
                             f"n={st['n']:<3} iou_med={st['median']:.3f} "
                             f"iou_p25={st['p25']:.3f} iou_mean={st['mean']:.3f}")
            out['confidence'][b] = row

    # ── C. improvement Δ by ORIGINAL (full) IoU quartile ──
    ious_full = np.array([s.iou_full for s in samples], dtype=np.float64)
    q25, q50, q75 = (float(x) for x in np.quantile(ious_full, [0.25, 0.5, 0.75]))

    def _qidx(v: float) -> int:
        return 0 if v <= q25 else (1 if v <= q50 else (2 if v <= q75 else 3))

    qlabels = ['Q1_worst', 'Q2', 'Q3', 'Q4_best']
    log_line(f'{log_tag}{suffix} zoom Δ-BY-ORIGINAL (full-iou quartiles '
             f'q25={q25:.3f} q50={q50:.3f} q75={q75:.3f}) '
             f'— rescue (worst) vs nudge (best)')
    out['delta_by_original'] = {}
    for qi in range(4):
        sel = [s for s in samples if _qidx(s.iou_full) == qi]
        if not sel:
            continue
        dz = _stats([s.iou_zoom - s.iou_full for s in sel])
        do = _stats([s.iou_oracle - s.iou_full for s in sel])
        f0 = _stats([s.iou_full for s in sel])
        log_line(f'{log_tag}{suffix} zoom {qlabels[qi]:<8} n={len(sel):<3} '
                 f"full_med={f0['median']:.3f} | "
                 f"Δzoom med={dz['median']:+.3f} mean={dz['mean']:+.3f} | "
                 f"Δoracle med={do['median']:+.3f} mean={do['mean']:+.3f}")
        out['delta_by_original'][qlabels[qi]] = {
            'n': len(sel), 'full_median': f0['median'],
            'delta_zoom': dz, 'delta_oracle': do,
        }
    return out
