"""contrastive_inpainting_v1.scripts.train_multi_head — joint trainer for the
two-head splice detector (image-level BCE + contrastive localization).

Forked from train_image_bce so it inherits the heavy-aug pipeline, biased
crop mixture, sliding-window eval, and robustness sweep verbatim. Adds:
  - MultiHeadDetector backbone with optional contrastive + BCE heads
  - Selective contrastive loss gated to splice items only
  - Per-step training log line partitioned: loss=total (bce=... contr=...)
  - Optional localization eval (F1/IoU per area_tier) when contrastive
    head is present, gated by the BCE head when both heads are present.

Three configurations via flags (single script):
    BCE-only:         --contrastive_dim 0 --pool_hidden 256 --lambda_contrastive 0
    Contrastive-only: --contrastive_dim 128 --pool_hidden 0 --lambda_image_bce 0
    Joint (default):  (no overrides)
"""

import argparse
import atexit
import hashlib
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from lab_utils.logging.text import install_log, log_line
from lab_utils.data.dataset import LabDataset, lab_collate_fn
from lab_utils.data.loaders import LoaderConfig, build_eval_loader
from lab_utils.train.checkpoint import load as ckpt_load, save as ckpt_save
from lab_utils.train.distributed import (
    DistributedContext,
    barrier,
    cleanup as ddp_cleanup,
    setup as ddp_setup,
    unwrap_model,
    wrap_model,
)
from lab_utils.data.resolution import Resolution
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.model.losses.contrastive import (
    selective_contrastive_loss,
    selective_symmetric_contrastive_loss,
)
from lab_utils.model.losses.bce import selective_patch_bce_loss

from lab_utils.eval.robustness import (
    run_robustness_sweep,
    metrics_from_logits,
)
from lab_utils.eval.localization import (
    collect_zoom_eval_samples,
    report_zoom_eval,
    _patches_to_pixels as _loc_patches_to_pixels,
    _load_gt_pixel_mask as _loc_load_gt_pixel_mask,
    _mask_metrics as _loc_mask_metrics,
    _f1_iou as _loc_f1_iou,
    _bucket as _loc_bucket,
    _stats as _loc_stats,
    _pct_line as _loc_pct_line,
)
from lab_utils.eval.metrics import auroc
from lab_utils.eval.zoom import attention_zoom_bbox
from lab_utils.eval.gap_utils import compute_gap_prediction
from lab_utils.eval.partition import spherical_kmeans2
from contrastive_inpainting_v1.diagnose.polarity import polarity_attn

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.experiments.tgif2_flux import (
    build_tgif2_items, split_tgif2_coco_ids)
from contrastive_inpainting_v1.configs.augment import (
    build_aug_kwargs,
    build_degradation_kwargs,
    build_heavy_aug_kwargs,
    build_light_aug_kwargs,
    eval_aug_settings,
    DEFAULT_EVAL_AUG_CONDITIONS,
)
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec


# Default crop-scale mixture: throw in uncropped images, mild crops, and crops down to 60% area.
DEFAULT_CROP_MIX = [
    ((1.00, 1.00), 0.30),
    ((0.80, 1.00), 0.40),
    ((0.60, 0.80), 0.30),
]

_REAL_KINDS  = frozenset({'imd_real', 'indoor_real', 'casia_real'})
_SPLICE_KINDS = frozenset({'imd_splice', 'casia_splice'})


def _is_real(item: dict) -> bool:
    return item.get('kind', '') in _REAL_KINDS


def _kind_is_splice(kind: str) -> bool:
    return kind in _SPLICE_KINDS


# ── BCE-head adapter for sliding-window / robustness callables ───────────────

class _BCEHeadAdapter(nn.Module):
    """Wraps MultiHeadDetector so callers that expect `model(img) → (B,)`
    (robustness sweep) keep working when the BCE head exists.
    """
    def __init__(self, multi_head: nn.Module):
        super().__init__()
        self.multi_head = multi_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.multi_head(x)
        if out['image_logit'] is None:
            raise RuntimeError(
                '_BCEHeadAdapter: model has no image-BCE head; cannot produce image logit'
            )
        return out['image_logit']


# ── Eval: image-level BCE ────────────────────────────────────────────────────

def _run_image_bce_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    log_tag: str = '[eval]',
    tag: str = '',
) -> Dict:
    """Image-level BCE eval — same metric set as train_image_bce.

    Expects `model(img)` to return `(B,)` (use _BCEHeadAdapter).
    """
    model.eval()
    logits_all, labels_all, areas_all = [], [], []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            img = batch['img'].to(device, non_blocking=True)
            meta_list = batch['meta'] if isinstance(batch['meta'], list) else [
                {k: v[i] for k, v in batch['meta'].items()}
                for i in range(img.shape[0])
            ]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=(device.type == 'cuda')):
                logit = model(img)
            logit = logit.detach().cpu().float().numpy()
            for i in range(len(logit)):
                logits_all.append(float(logit[i]))
                labels_all.append(0 if _is_real({'kind': meta_list[i].get('kind', '')}) else 1)
                areas_all.append(float(meta_list[i].get('blob_area_actual', 0.0)))

    logits = np.array(logits_all, dtype=np.float64)
    labels = np.array(labels_all, dtype=np.int32)
    n_total  = len(labels)
    n_splice = int(labels.sum())
    n_real   = n_total - n_splice
    if n_total == 0:
        log_line(f'{log_tag} {tag} EMPTY')
        return {}

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    tpr     = tp / n_splice if n_splice > 0 else float('nan')
    tnr     = tn / n_real   if n_real   > 0 else float('nan')
    bal_acc = 0.5 * (tpr + tnr) if n_splice > 0 and n_real > 0 else float('nan')

    def _f1(_tp, _fp, _fn):
        denom = 2 * _tp + _fp + _fn
        return (2 * _tp / denom) if denom > 0 else float('nan')

    f1 = _f1(tp, fp, fn)

    auc = float(auroc(logits, labels)) if (n_splice > 0 and n_real > 0) else float('nan')

    opt_thresh, opt_tpr, opt_tnr, opt_bacc = 0.0, 0.0, 1.0, 0.5
    if n_splice > 0 and n_real > 0:
        for t in np.unique(logits):
            p = (logits >= t).astype(np.int32)
            _tpr = float(((p == 1) & (labels == 1)).sum()) / n_splice
            _tnr = float(((p == 0) & (labels == 0)).sum()) / n_real
            _ba  = 0.5 * (_tpr + _tnr)
            if _ba > opt_bacc:
                opt_bacc, opt_thresh, opt_tpr, opt_tnr = _ba, float(t), _tpr, _tnr
    opt_preds = (logits >= opt_thresh).astype(np.int32)
    opt_tp = int(((opt_preds == 1) & (labels == 1)).sum())
    opt_fp = int(((opt_preds == 1) & (labels == 0)).sum())
    opt_fn = int(((opt_preds == 0) & (labels == 1)).sum())
    opt_f1 = _f1(opt_tp, opt_fp, opt_fn)

    def _bucket(a):
        if a < 0.15: return 'small'
        if a < 0.30: return 'medium'
        return 'large'

    real_logits = logits[labels == 0]
    bucket_stats = {}
    for bname in ('small', 'medium', 'large'):
        mask = np.array([
            labels_all[i] == 1 and _bucket(areas_all[i]) == bname
            for i in range(n_total)
        ])
        if mask.sum() == 0:
            bucket_stats[bname] = {'n': 0, 'tpr': float('nan'),
                                   'opt_tpr': float('nan'), 'auc': float('nan')}
        else:
            # Size-stratified AUROC: this bucket's splices vs ALL reals.
            if real_logits.size > 0:
                b_scores = np.concatenate([logits[mask], real_logits])
                b_labels = np.concatenate([
                    np.ones(int(mask.sum())), np.zeros(real_logits.size)]).astype(np.int32)
                b_auc = float(auroc(b_scores, b_labels))
            else:
                b_auc = float('nan')
            bucket_stats[bname] = {
                'n':       int(mask.sum()),
                'tpr':     float(preds[mask].mean()),
                'opt_tpr': float(opt_preds[mask].mean()),
                'auc':     b_auc,
            }

    suffix = f' {tag}' if tag else ''
    log_line(
        f'{log_tag}{suffix} '
        f'n_total={n_total} n_splice={n_splice} n_real={n_real} '
        f'auc={auc:.4f} '
        f'@ thresh=0.5: bal_acc={bal_acc:.4f} f1={f1:.4f} tpr={tpr:.4f} tnr={tnr:.4f} | '
        f'@ opt thresh={opt_thresh:.3f}: bal_acc={opt_bacc:.4f} '
        f'f1={opt_f1:.4f} tpr={opt_tpr:.4f} tnr={opt_tnr:.4f}'
    )
    for bname in ('small', 'medium', 'large'):
        bs = bucket_stats[bname]
        log_line(
            f'{log_tag}{suffix}   area_tier={bname} n={bs["n"]} '
            f'auc={bs["auc"]:.4f} tpr@0.5={bs["tpr"]:.4f} tpr@opt={bs["opt_tpr"]:.4f}'
        )

    return dict(
        auc=auc, bal_acc=bal_acc, f1=f1, tpr=tpr, tnr=tnr,
        opt_thresh=opt_thresh, opt_bacc=opt_bacc, opt_f1=opt_f1,
        n_total=n_total, n_splice=n_splice, n_real=n_real,
        bucket_stats=bucket_stats, logits=logits, labels=labels,
    )



# ── Eval: contrastive localization (delegated to lab_utils/eval/localization) ─

@torch.no_grad()
def _run_localization_eval(
    multi_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    cfg: Config,
    log_tag: str = '[eval]',
    tag: str = '',
) -> Dict:
    """Lean per-tier localization for the contrastive head (kmeans only).

    Spherical k-means(2) with attention-polarity, scored at pixel granularity
    per area_tier (small/medium/large) plus an aggregate row.
    """
    multi_head.eval()
    n_side = cfg.resolution.num_patches_per_side
    psz = cfg.resolution.patch_size
    buckets = ('small', 'medium', 'large')

    acc = {b: {'f1': [], 'iou': []} for b in buckets}

    for batch in loader:
        if batch is None:
            continue
        img = batch['img'].to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=(device.type == 'cuda')):
            out = multi_head(img)
        z_b = out.get('contrastive')
        if z_b is None:
            log_line(f'{log_tag} {tag} loc: model has no contrastive head; skip')
            return {}
        z_b = z_b.detach().cpu().float().numpy()                    # (B, N, D)
        att_b = out.get('pool_attention')
        att_b = att_b.detach().cpu().float().numpy() if att_b is not None else None
        meta_list = batch['meta'] if isinstance(batch['meta'], list) else [
            {k: v[i] for k, v in batch['meta'].items()} for i in range(img.shape[0])
        ]
        for i in range(len(z_b)):
            meta = meta_list[i]
            if _is_real(meta):
                continue
            gt_px = _loc_load_gt_pixel_mask(meta, cfg.resolution)
            if gt_px is None:
                continue
            b = _loc_bucket(float(gt_px.mean()))
            z = z_b[i]                                              # (N, D)
            att = att_b[i].reshape(n_side, n_side) if att_b is not None else None

            raw, _ = spherical_kmeans2(z, n_init=4)
            km = polarity_attn(raw, att).reshape(n_side, n_side)

            pred_px = _loc_patches_to_pixels(
                km.reshape(-1).astype(np.float64), n_side, psz)
            f1, iou, _, _, _ = _loc_mask_metrics(pred_px, gt_px)
            acc[b]['f1'].append(f1)
            acc[b]['iou'].append(iou)

    suffix = f' {tag}' if tag else ''

    all_f1: list = []
    all_iou: list = []
    for b in buckets:
        d = acc[b]
        if not d['iou']:
            continue
        f1_st = _loc_stats(d['f1'])
        iou_st = _loc_stats(d['iou'])
        log_line(
            f'{log_tag}{suffix} loc area_tier={b:<6} '
            f'f1[{_loc_pct_line(f1_st)}] '
            f'iou[{_loc_pct_line(iou_st)}]'
        )
        all_f1.extend(d['f1'])
        all_iou.extend(d['iou'])

    if all_f1:
        agg_f1 = _loc_stats(all_f1)
        agg_iou = _loc_stats(all_iou)
        log_line(
            f'{log_tag}{suffix} loc aggregate       '
            f'f1[{_loc_pct_line(agg_f1)}] '
            f'iou[{_loc_pct_line(agg_iou)}]'
        )

    return acc


@torch.no_grad()
def _run_patch_bce_loc_eval(
    eval_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    res: Resolution,
    log_tag: str = '[eval]',
    tag: str = '',
) -> Dict:
    """Per-epoch FULL-FRAME localization for the patch-BCE head.

    Scores on the SAME pixel ruler as the contrastive localization eval
    (``_patches_to_pixels`` + ``_load_gt_pixel_mask`` at the model input frame),
    so the two methods are comparable epoch-by-epoch. Decodes the deployed
    sigmoid(patch_logit) >= 0.5 mask and also reports the best-threshold IoU
    ceiling. Splice items only; reals skipped (no GT mask). No swin (full-frame
    is the cheap per-epoch signal; the offline sweep adds windows).
    """
    eval_model.eval()
    n_side = res.num_patches_per_side
    psz = res.patch_size
    buckets = ('small', 'medium', 'large')
    acc = {b: {'iou': [], 'f1': [], 'prec': [], 'rec': [], 'iou_ceil': []} for b in buckets}
    thr_grid = np.linspace(-4.0, 4.0, 17)
    for batch in loader:
        if batch is None:
            continue
        img = batch['img'].to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=(device.type == 'cuda')):
            out = eval_model(img)
        pl = out.get('patch_logit')
        if pl is None:
            log_line(f'{log_tag} {tag} patch_loc: model has no patch head; skip')
            return {}
        pl = pl.detach().cpu().float().numpy()                     # (B, N)
        meta_list = batch['meta'] if isinstance(batch['meta'], list) else [
            {k: v[i] for k, v in batch['meta'].items()} for i in range(img.shape[0])
        ]
        for i in range(len(pl)):
            meta = meta_list[i]
            if _is_real(meta):
                continue
            gt_px = _loc_load_gt_pixel_mask(meta, res)
            if gt_px is None:
                continue
            b = _loc_bucket(float(gt_px.mean()))
            score = pl[i].reshape(-1)                              # (N,)
            pred_px = _loc_patches_to_pixels((score >= 0.0).astype(np.float64), n_side, psz)
            f1, iou, prec, rec, _ = _loc_mask_metrics(pred_px, gt_px)
            acc[b]['iou'].append(iou);   acc[b]['f1'].append(f1)
            acc[b]['prec'].append(prec); acc[b]['rec'].append(rec)
            best = 0.0
            for t in thr_grid:
                _, iou_t = _loc_f1_iou(
                    _loc_patches_to_pixels((score >= t).astype(np.float64), n_side, psz), gt_px)
                if iou_t > best:
                    best = iou_t
            acc[b]['iou_ceil'].append(best)
    suffix = f' {tag}' if tag else ''
    for b in buckets:
        if not acc[b]['iou']:
            continue
        iou_st = _loc_stats(acc[b]['iou'])
        f1_st = _loc_stats(acc[b]['f1'])
        prec_st = _loc_stats(acc[b]['prec'])
        rec_st = _loc_stats(acc[b]['rec'])
        ceil_st = _loc_stats(acc[b]['iou_ceil'])
        log_line(
            f'{log_tag}{suffix} patch_loc area_tier={b} '
            f'iou[{_loc_pct_line(iou_st)}] '
            f'f1[{_loc_pct_line(f1_st)}] | '
            f'prec_med={prec_st["median"]:.4f}(m={prec_st["mean"]:.4f}) '
            f'rec_med={rec_st["median"]:.4f}(m={rec_st["mean"]:.4f}) '
            f'iou_ceil_med={ceil_st["median"]:.4f}(m={ceil_st["mean"]:.4f})'
        )
    return acc


# ── helpers ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_image_logits(model_callable, loader, device):
    """Run model over loader; return (logits, labels) numpy arrays."""
    if hasattr(model_callable, 'eval'):
        model_callable.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        if batch is None:
            continue
        img = batch['img'].to(device, non_blocking=True)
        meta_list = batch['meta'] if isinstance(batch['meta'], list) else [
            {k: v[i] for k, v in batch['meta'].items()} for i in range(img.shape[0])
        ]
        logit = model_callable(img).detach().cpu().float().numpy()
        for i in range(len(logit)):
            logits_all.append(float(logit[i]))
            labels_all.append(0 if _is_real({'kind': meta_list[i].get('kind', '')}) else 1)
    return np.array(logits_all, dtype=np.float64), np.array(labels_all, dtype=np.int32)


def _seed_worker(worker_id: int) -> None:
    """Per-worker RNG reseed. PyTorch reseeds Python's ``random`` per worker per
    epoch, but NOT numpy — forked workers inherit the parent's numpy state, so
    the np.random-based noise augmentations (corruptions/light) would replay the
    SAME noise realizations across workers and across epochs. Derive a fresh
    numpy seed from the per-worker torch seed to break that correlation."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def _subsample_items(items, n, *, seed='swin') -> list:
    """Deterministic md5 subsample. Stable across runs."""
    if not items or len(items) <= n:
        return list(items)
    def _key(it):
        path = it.get('img') or it.get('path') or ''
        return hashlib.md5(f'{seed}|{path}'.encode('utf-8')).hexdigest()
    return sorted(items, key=_key)[:n]


_TGIF_FILE_CACHE: Dict[str, Optional[Tuple[int, int]]] = {}  # path → size; None = corrupt


def _tgif_file_size(path: str) -> Optional[Tuple[int, int]]:
    """Full-decode a file once, cache (W, H); None = unreadable/truncated."""
    if path in _TGIF_FILE_CACHE:
        return _TGIF_FILE_CACHE[path]
    try:
        with Image.open(path) as im:
            im.load()   # header-only open() does not catch truncated payloads
            size = im.size
    except Exception as exc:
        log_line(f'[data] tgif corrupt file → drop: {path} ({exc})')
        size = None
    _TGIF_FILE_CACHE[path] = size
    return size


def _prep_tgif_items(items, *, mask_cache_dir, log_tag='[data]', tag='tgif'):
    """Drop TGIF items with unreadable files; fix mask/img size mismatches.

    Truncated PNGs raise DataError mid-epoch inside LabDataset and kill the
    run. Worse, every cropping path (train random crops, zoom-eval windows)
    applies IMAGE-pixel crop coords to the mask — a mask at a different
    resolution yields silently misaligned supervision, not an error. So:
    corrupt img/mask → drop (loud); mask.size != img.size → write a
    NEAREST-resized mask copy under mask_cache_dir and repoint the item.
    Decode results are cached per path, so per-epoch re-calls are free.
    """
    kept, n_corrupt, n_fixed = [], 0, 0
    for it in items:
        img_size = _tgif_file_size(it['img'])
        if img_size is None:
            n_corrupt += 1
            continue
        mask_path = it.get('mask')
        if not mask_path:
            kept.append(it)
            continue
        mask_size = _tgif_file_size(mask_path)
        if mask_size is None:
            n_corrupt += 1
            continue
        if mask_size != img_size:
            key = hashlib.md5(
                f'{mask_path}|{img_size[0]}x{img_size[1]}'.encode('utf-8')
            ).hexdigest()[:16]
            fixed = os.path.join(mask_cache_dir, f'{key}.png')
            if not os.path.exists(fixed):
                os.makedirs(mask_cache_dir, exist_ok=True)
                tmp = f'{fixed}.tmp{os.getpid()}'
                with Image.open(mask_path) as m:
                    m.convert('L').resize(img_size, Image.NEAREST).save(tmp, format='PNG')
                os.replace(tmp, fixed)
            it = {**it, 'mask': fixed}
            n_fixed += 1
        kept.append(it)
    if n_corrupt or n_fixed:
        log_line(f'{log_tag} {tag}: kept={len(kept)}/{len(items)} '
                 f'dropped_corrupt={n_corrupt} mask_size_fixed={n_fixed}')
    return kept


def _interp_aug_kwargs(lo: Dict, hi: Dict, s: float) -> Dict:
    """Elementwise lerp between two aug-kwargs presets at strength s ∈ [0, 1].

    int-typed knobs (the JPEG quality bounds) stay ints; everything else
    becomes float. Keys must match — both presets come from configs.augment.
    """
    out: Dict = {}
    for k, a in lo.items():
        b = hi[k]
        v = a + (b - a) * float(s)
        out[k] = int(round(v)) if isinstance(a, int) and isinstance(b, int) else float(v)
    return out


def _tgif_model_filter(fakes, model_filter, *, log_tag='[eval]', tag='tgif2'):
    """Filter TGIF fakes to one inpainting model (normalized substring match,
    so 'flux-dev' matches 'flux_dev'/'FLUX.1-dev'/...). Falls back to ALL
    models with a loud warning when nothing matches — a naming mismatch in the
    index should degrade visibly, not silently empty the eval."""
    def _norm(s):
        return ''.join(ch for ch in str(s).lower() if ch.isalpha())
    want = _norm(model_filter)
    if not want:
        return fakes
    kept = [it for it in fakes if want in _norm(it.get('tgif_model', ''))]
    if not kept:
        models = sorted({str(it.get('tgif_model', '')) for it in fakes})
        log_line(f'{log_tag} {tag}: model filter {model_filter!r} matched 0/'
                 f'{len(fakes)} fakes (models present: {models}); using ALL')
        return fakes
    return kept


def _tgif_partition_cells(fakes) -> Dict[Tuple[str, str], List[dict]]:
    """Group fakes by the four headline cells: (type sp/fr × mask family)."""
    cells: Dict[Tuple[str, str], List[dict]] = {}
    for it in fakes:
        k = (str(it.get('tgif_type', 'NA')), str(it.get('tgif_mask_family', 'NA')))
        cells.setdefault(k, []).append(it)
    return cells



def _attention_zoom_second_pass(
    model, img, pool_attention, splice_labels, is_splice,
    *, res, gt_patch_threshold, min_mask_patch_frac, max_frame_frac,
    thresh_mode='otsu',
):
    """Attention-guided zoom second pass for TRAINING (detect-then-zoom view).

    For each sample — splice AND real — crop the input tensor to the bbox of
    the hottest pool-attention region (same attention_zoom_bbox the c2f eval
    uses) and resize back to full resolution, then forward the zoom batch.
    Reals are the FP-pressure half: the model must keep calling a tight crop
    of its own most-suspicious real region REAL. Labels are re-derived from
    the patch-label grid rasterized into the crop frame; a splice whose
    in-frame fraction drops below min_mask_patch_frac keeps its fake image
    label but is IGNORED (the first pass's no-penalty policy). Samples with
    no hot region or a bbox covering > max_frame_frac of the frame are
    skipped (zoom would be a no-op).

    Returns (out, zoom_labels, zoom_is_splice, zoom_miss, kept_idx) or None
    when nothing is zoomable.
    """
    B = img.shape[0]
    S = res.image_size
    P = res.num_patches_per_side
    att = pool_attention.detach().float().cpu().numpy()
    crops, lbls, keep_idx = [], [], []
    for i in range(B):
        bbox = attention_zoom_bbox(
            att[i].reshape(P, P), S, S, thresh_mode=thresh_mode)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        if (x1 - x0) * (y1 - y0) > max_frame_frac * S * S:
            continue
        crop = img[i:i + 1, :, y0:y1, x0:x1]
        crops.append(F.interpolate(
            crop, size=(S, S), mode='bilinear', align_corners=False))
        # Patch labels for the crop: NN-upsample the (P, P) grid to pixels,
        # crop the same box, average per new patch → density, threshold with
        # the same gt_patch_threshold the dataset uses. Sub-patch precision
        # of the original mask is lost, but the grid is the only GT in-batch.
        g = splice_labels[i].view(1, 1, P, P).float()
        g_px = F.interpolate(g, size=(S, S), mode='nearest')[:, :, y0:y1, x0:x1]
        dens = F.adaptive_avg_pool2d(g_px, (P, P)).view(-1)
        lbls.append((dens > gt_patch_threshold).long())
        keep_idx.append(i)
    if not keep_idx:
        return None
    zoom_img = torch.cat(crops, dim=0)
    zoom_labels = torch.stack(lbls, dim=0)
    idx = torch.tensor(keep_idx, device=img.device, dtype=torch.long)
    zoom_is_splice = is_splice[idx]
    frac = zoom_labels.float().mean(dim=1)
    zoom_miss = zoom_is_splice & (frac < min_mask_patch_frac)
    out = model(zoom_img)
    return out, zoom_labels, zoom_is_splice, zoom_miss, idx


def _make_bce_eval_callable(model_callable, items, cfg, device,
                            *, batch_size, num_workers,
                            gt_patch_threshold: float = 0.06):
    """Build closure: aug_kwargs → metrics dict, for the robustness sweep."""
    def _eval_under_aug(aug_kwargs, *, tag):
        ds = LabDataset(
            items, cfg.resolution,
            cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
            augment=False,
            use_degradation=False,
            use_invariance=False,
            use_splice_degradation=False,
            gt_patch_threshold=float(gt_patch_threshold),
            **aug_kwargs,
        )
        loader = build_eval_loader(
            ds,
            LoaderConfig(
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda'),
            ),
        )
        logits, labels = _collect_image_logits(model_callable, loader, device)
        return metrics_from_logits(logits, labels)
    return _eval_under_aug


def _splice_balance_weights(items: list) -> list:
    """Shim — see :func:`lab_utils.data.sampling.splice_balance_weights`."""
    from lab_utils.data.sampling import splice_balance_weights
    return splice_balance_weights(items, target_splice_frac=0.5)


# ── argparse ─────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Two-head splice forensics: image-BCE + contrastive localization'
    )
    p.add_argument('--imd2020_root',     type=str, default=None)
    p.add_argument('--casia_root',       type=str, default=None)
    p.add_argument('--indoor_root',      type=str, default=None)
    p.add_argument('--coco_inpaint_root', type=str, default=None,
                   help='COCO-inpaint root (dir holding images/{modified,original,mask}). '
                        'Joins the training mix; held-out val slice used for loc eval.')
    p.add_argument('--sagid_root',        type=str, default=None,
                   help='SAGID root (dir holding {modified,original,mask}). '
                        'Joins the training mix; held-out val slice used for loc eval.')
    p.add_argument('--bfree_root',        type=str, default=None,
                   help='BFree root (dir holding COCO_real_512, '
                        'SD2.1_inpainted_{diffcat,samecat}, masks|mask, bbox; or a '
                        'parent containing it). Joins the training mix; held-out '
                        'val slice used for loc eval.')
    # Train/val split control — mirrors train_image_bce. The flip (train on
    # CASIA, validate on IMD) puts CASIA's tiny-heavy splice distribution into
    # TRAIN and holds IMD out as the OOD check.
    p.add_argument('--casia_train',      action='store_true', default=False,
                   help='Include CASIA in the TRAIN split (default: CASIA val-only). '
                        'Pair with --imd_val_only for train-on-CASIA / val-on-IMD.')
    p.add_argument('--imd_val_only',     action='store_true', default=False,
                   help='Hold IMD2020 out entirely as OOD validation (no IMD in train).')
    p.add_argument('--checkpoint_root', '--run_root', dest='checkpoint_root', type=str, default=None)
    p.add_argument('--resume',           type=str, default=None)
    p.add_argument('--num_epochs',       type=int, default=10)
    p.add_argument('--batch_size',       type=int, default=8)
    p.add_argument('--grad_accum',       type=int, default=4)
    p.add_argument('--lr',               type=float, default=2e-4)
    p.add_argument('--weight_decay',     type=float, default=1e-4)
    p.add_argument('--train_samples',    type=int, default=2000)
    p.add_argument('--splice_mix', nargs='+', default=None, metavar='SRC=FRAC',
                   help='Per-source target fractions of the splice-positive draw '
                        'budget, e.g. --splice_mix imd2020=0.3 coco_inpaint=0.4 '
                        'sagid=0.3 (normalized to sum 1). Sources present in the '
                        'train mix but omitted here are EXCLUDED from splice draws. '
                        'Unset = uniform over all splice items (legacy behavior). '
                        'Reals stay source-agnostic and class-balanced.')
    p.add_argument('--num_workers',      type=int, default=0,
                   help='0 avoids the OMP_NUM_THREADS + fork SIGABRT seen on the '
                        'training box. Crop telemetry is also only reliable at 0.')
    p.add_argument('--device',           type=str, default='cuda')
    p.add_argument('--seed',             type=int, default=42)
    p.add_argument('--log_every',        type=int, default=20)
    p.add_argument('--no_grad_ckpt', action='store_true',
                   help='Disable gradient checkpointing (faster on high-VRAM GPUs '
                        'like A100-80GB; uses more memory but ~2x throughput).')
    p.add_argument('--bf16', action='store_true',
                   help='Enable bf16 mixed-precision training via torch.cuda.amp '
                        '(~2x throughput on A100/H100 tensor cores).')
    # Heads
    p.add_argument('--contrastive_dim',  type=int, default=128,
                   help='Output dim of contrastive projector (0 disables).')
    p.add_argument('--pool_hidden',      type=int, default=256,
                   help='Hidden dim of BCE attention pool (0 disables).')
    # Loss weights
    p.add_argument('--lambda_image_bce',  type=float, default=1.0)
    p.add_argument('--lambda_contrastive', type=float, default=2.0,
                   help='Weight on contrastive loss. Bumped to 2.0 default '
                        'because BCE saturates early on bce_acc and otherwise '
                        'dominates the joint backbone gradients.')
    # Per-patch BCE localization head (the supervised splice-flagging baseline).
    # Reuses the SAME splice_labels / splice_patch_weights the contrastive head
    # trains on, so the two methods' supervision masks are identical and the
    # side-by-side isolates the objective, not the labels.
    p.add_argument('--patch_bce', action='store_true', default=False,
                   help='Enable the dense per-patch BCE localization head.')
    p.add_argument('--lambda_patch_bce', type=float, default=1.0,
                   help='Weight on the per-patch BCE loss (active when --patch_bce).')
    p.add_argument('--patch_pos_weight', type=float, default=10.0,
                   help='BCE pos_weight for the rare positive (splice) patches. '
                        'Higher = more recall on small splices at some precision '
                        'cost. 10.0 is a reasonable default for ~1-10%% splice area.')
    # Swin localization (only fires for joint models with both heads).
    # Single-scale light zoom: a small capped set of source-resolution SQUARE
    # crops is scored by the BCE head, the best passing crop is selected, and
    # BCE attention sets contrastive cluster polarity inside that crop.
    p.add_argument('--swin_scale',       type=float, default=0.7,
                   help='Square window side as fraction of min(H_src, W_src). '
                        '0.7 ≈ 50% area, the median sweet spot from the '
                        'oracle-crop sweep. Windows are SQUARE in source pixels '
                        'so resize to model input is square→square (no squish).')
    p.add_argument('--swin_stride_frac', type=float, default=1.0,
                   help='Stride as fraction of window side. 1.0 = non-overlapping '
                        'tiles (~4 windows for square source, more for elongated). '
                        '0.5 = half-overlap (~4× more windows).')
    p.add_argument('--swin_inner_batch', type=int, default=8,
                   help='Inner batch size for per-window forward passes.')
    p.add_argument('--swin_every',       type=int, default=1,
                   help='Run swin localization every N epochs. 0 disables.')
    # Localization threshold sweep (BCE-logit gate)
    p.add_argument('--loc_threshold_grid', type=float, nargs='+',
                   default=[-2.0, -1.0, 0.0, 1.0, 2.0],
                   help='BCE-logit thresholds at which to report gated F1/IoU.')
    # GT patch rasterization (boundary-aware label policy)
    p.add_argument('--gt_patch_threshold', type=float, default=0.06,
                   help='Hard-mode density threshold: a 16x16 patch is GT-positive '
                        'when its mask density exceeds this. Used when --gt_soft_label=0. '
                        'Default 0.06 (was 0.15) — credits edge patches as positives.')
    p.add_argument('--gt_soft_label',     type=int, default=1, choices=(0, 1),
                   help='1 = soft per-patch labels with an "ignore band" near the '
                        'splice boundary; edge patches get reduced supervision '
                        'weight so the contrastive head is not penalized for being '
                        'wrong on barely-overlapping patches. 0 = legacy hard label.')
    p.add_argument('--gt_soft_low',       type=float, default=0.03,
                   help='Soft-label LOW band: patches with 0 < density < low get '
                        'weight 0 (ignore — neither supervised as splice nor as bg).')
    p.add_argument('--gt_soft_high',      type=float, default=0.20,
                   help='Soft-label HIGH band: patches with density >= high get '
                        'full positive supervision (weight=1).')
    # Crop
    p.add_argument('--use_crop_mix',     dest='use_crop_mix', action='store_true', default=True)
    p.add_argument('--no_crop_mix',      dest='use_crop_mix', action='store_false',
                   help='Disable the multi-scale crop mixture and use a single '
                        '[train_crop_min, 1.0] scale range. Set train_crop_min=0.80 '
                        'to reproduce the pre-phase-2 near-full-frame crops.')
    p.add_argument('--train_crop_min',   type=float, default=0.60)
    # Degradation harness (off by default; the multi-head trainer otherwise
    # leaves it disabled). NEWLY wired into this trainer — smoke-test before a
    # long run. When on, splice-region degradation + whole-image corruption are
    # applied to TRAIN, with the heavy-aug down-weight on over-destroyed samples.
    p.add_argument('--use_splice_degradation', action='store_true', default=False,
                   help='Enable splice-region degradation on TRAIN (heavy aug).')
    p.add_argument('--use_real_degradation', dest='use_real_degradation',
                   action='store_true', default=None,
                   help='Apply MATCHED local degradation to real negatives so '
                        '"local artifact present" is not a fake/real shortcut. '
                        'Default: follows --use_splice_degradation (on when it is).')
    p.add_argument('--no_real_degradation', dest='use_real_degradation',
                   action='store_false',
                   help='Force real-negative degradation OFF even when splice '
                        'degradation is on (re-introduces the artifact shortcut).')
    p.add_argument('--whole_corrupt_prob', type=float, default=0.0,
                   help='Prob of whole-image corruption on TRAIN (0 = off).')
    # Full-AE positives: fraction of inpaint items (those with real_path) that
    # get the pristine-background paste. <1.0 leaves a slice as raw full-VAE
    # frames — full-AE positives whose background carries generator noise but is
    # labeled REAL per-patch: the hard negative teaching "VAE noise != forgery".
    # Default 0.15: a small sp-mimic slice keeps the paste-back framing covered
    # while the bulk trains on the harder full-VAE frames.
    p.add_argument('--paste_frac', type=float, default=0.15,
                   help='Fraction of inpaint items pasted onto pristine bg '
                        '(1.0 = all regional splices; default 0.15 keeps most '
                        'as full-AE positives). TRAIN only; eval always pastes.')
    # Train-time gaussian-noise override (laundering robustness). None = leave
    # the aug_intensity preset untouched. Forces the semantic pathway to carry
    # the load instead of leaning on the high-frequency VAE fingerprint.
    p.add_argument('--noise_prob', type=float, default=None,
                   help='Override light-aug gaussian-noise probability on TRAIN.')
    p.add_argument('--noise_std_max', type=float, default=None,
                   help='Override light-aug gaussian-noise max std on TRAIN.')
    # Localization eval diagnostics: adds oracle / routed / s_i-selector rows on
    # top of the default kmeans + outlier_med. Off by default to keep output lean.
    # Splice crop policy (the small-splice fix, ported from train_image_bce).
    # oracle_fallback zooms tiny/off-center CASIA splices into frame instead of
    # demoting them to a whole-image resize that relabels them REAL. Critical
    # for the CASIA-train flip: without it the model collapses to 0.50 acc.
    p.add_argument('--min_mask_patch_frac', type=float, default=0.01,
                   help='Min splice patch-fraction to ACCEPT a random crop (flat 1%%).')
    p.add_argument('--crop_max_tries', type=int, default=48,
                   help='Random-crop attempts before the oracle fallback fires.')
    p.add_argument('--splice_crop_mode', type=str, default='oracle_fallback',
                   choices=('random', 'oracle_fallback'),
                   help="random = legacy center/resize fallback (misses tiny "
                        "splices, relabels them real); oracle_fallback = "
                        "mask-centered zoom crop guaranteeing the splice is in "
                        "frame, drop only on empty mask.")
    p.add_argument('--oracle_cov', type=float, nargs=2, default=(0.10, 0.40),
                   metavar=('LO', 'HI'),
                   help='Target splice coverage range for the oracle zoom crop.')
    # Natural zoom-coverage TRAIN crop: size-biased RANDOM-position crops that
    # target this in-frame splice coverage band (jittered oracle fallback). When
    # set it supersedes the lower-bound-only min_mask_patch_frac acceptance.
    p.add_argument('--splice_cov_band', type=float, nargs=2, default=None,
                   metavar=('LO', 'HI'),
                   help='Natural zoom-coverage band for TRAIN splice crops '
                        '(e.g. 0.35 0.50). Unset = legacy crop behavior.')
    # Seeded, non-destructive zoom EVAL on val splices (paired full/zoom/oracle
    # localization with confidence×size + Δ-by-original breakdowns). Off by
    # default so it never disturbs the headline full-frame val.
    p.add_argument('--val_zoom', action='store_true', default=False,
                   help='Run the seeded paired zoom-eval on val splices each '
                        'full-eval (extra forwards; localization-only).')
    p.add_argument('--viz_every', type=int, default=1,
                   help='Run viz_attention composite on a fixed validation subset every N epochs (0 disables).')
    p.add_argument('--val_zoom_cov', type=float, nargs=2, default=(0.05, 0.55),
                   metavar=('LO', 'HI'),
                   help='Per-item target in-frame coverage is drawn uniformly '
                        'from this range (seeded) → even spread of splice sizes.')
    # Attention-zoom TRAINING pass: after warmup epochs, a fraction of steps
    # gets a second forward on attention-bbox crops of the SAME batch (splices
    # AND reals), explicitly training the resolutions/framings the
    # detect-then-zoom deploy path produces. Reals in the zoom pass are the
    # FP-pressure half: a tight crop of the model's own most-suspicious real
    # region must still score REAL. Needs the BCE pool (pool_hidden > 0) for
    # the attention map.
    p.add_argument('--zoom_pass_prob', type=float, default=0.35,
                   help='Per-step probability of the attention-zoom second '
                        'pass (whole batch when it fires). 0 disables.')
    p.add_argument('--zoom_pass_start_epoch', type=int, default=2,
                   help='First epoch (0-based) the zoom pass runs — earlier '
                        'epochs have uninformative attention.')
    p.add_argument('--lambda_zoom', type=float, default=0.5,
                   help='Weight on the zoom-pass losses relative to the '
                        'first-pass losses (per-head lambdas apply on top).')
    p.add_argument('--zoom_pass_max_frac', type=float, default=0.80,
                   help='Skip a sample when its attention bbox covers more '
                        'than this fraction of the frame (zoom ≈ no-op).')
    # TGIF2 (diffusion inpainting) OOD detection slice, per epoch (BCE head).
    p.add_argument('--tgif_root', type=str, default=None,
                   help='TGIF2 FLUX root (dir holding tgif2_index.json). Set to '
                        'run a per-epoch OOD detection slice (AUROC + bal-acc).')
    p.add_argument('--tgif_index', type=str, default=None,
                   help='Path to tgif2_index.json (default <tgif_root>/tgif2_index.json).')
    p.add_argument('--tgif_n_fake', type=int, default=150)
    p.add_argument('--tgif_n_real', type=int, default=50)
    p.add_argument('--tgif_model', type=str, default='flux-dev',
                   help='Restrict the TGIF eval slice to one inpainting model '
                        '(normalized substring match). Empty string = all.')
    p.add_argument('--tgif_loc_every', type=int, default=3,
                   help='Run TGIF localization (kmeans + outlier_med, + patch head) '
                        'every N epochs. Detection slice still runs every epoch. '
                        '0 disables TGIF localization.')
    # TGIF FR half-split fine-tune (in-domain comparison from the TGIF paper):
    # train on one deterministic half of the coco_ids (FR fakes + their reals),
    # evaluate ONLY on the held-out half (also FR only). Split is at coco_id
    # level so a source image's fakes and pristine original never straddle the
    # train/eval boundary. The two id lists are written into checkpoint_root.
    p.add_argument('--tgif_train_half', action='store_true',
                   help='Add the train-half TGIF FR fakes+reals to TRAIN '
                        '(source=tgif2) and restrict every TGIF eval slice to '
                        'the held-out half, FR only. Requires --tgif_root.')
    p.add_argument('--tgif_half_seed', type=str, default='tgif_fr_half',
                   help='Seed string for the deterministic coco_id half-split.')
    p.add_argument('--tgif_half_frac', type=float, default=0.5,
                   help='Fraction of coco_ids assigned to the TRAIN half.')
    # Augmentation intensity. 'medium' matches the proven casia_bce_swin_v2 flip
    # recipe; 'heavy' overlaps eval-probe severities.
    # Aug curriculum: ramp invariance pressure instead of applying it from
    # step one. Epoch 0 trains at the LIGHT preset (clear discriminative
    # signal), then each epoch lerps toward HEAVY, capped at
    # aug_curriculum_max of the light→heavy span so the final regime stays
    # below the destructive end. Schedule is a function of the ABSOLUTE epoch
    # index, so --resume continues the ramp where it left off.
    p.add_argument('--aug_curriculum', action='store_true', default=False,
                   help='Per-epoch TRAIN aug schedule light→heavy (overrides '
                        '--aug_intensity for train; eval aug untouched).')
    p.add_argument('--aug_curriculum_max', type=float, default=0.75,
                   help='Final curriculum strength (1.0 = the full heavy preset).')
    p.add_argument('--aug_intensity',    type=str, default='medium',
                   choices=('none', 'light', 'medium', 'heavy'))
    # Robustness sweep
    p.add_argument('--robust_every',     type=int, default=3)
    p.add_argument('--robust_conditions', type=str, nargs='+',
                   default=list(DEFAULT_EVAL_AUG_CONDITIONS))
    p.add_argument('--eval_max_items',   type=int, default=200)
    # Contrastive loss (symmetric is the v2 default; legacy kept for parity).
    # None ⇒ keep the cfg default.
    p.add_argument('--contrastive_loss_mode', type=str, default=None,
                   choices=('symmetric', 'legacy'),
                   help='override cfg.CONTRASTIVE_LOSS_MODE')
    p.add_argument('--tau_pos', type=float, default=None,
                   help='symmetric: same-label cohesion floor (override cfg.TAU_POS)')
    p.add_argument('--tau_neg', type=float, default=None,
                   help='symmetric: diff-label separation ceiling (override cfg.TAU_NEG)')
    p.add_argument('--area_balance_power', type=float, default=None,
                   help='symmetric: region area-balance power (0=full,1=raw)')
    p.add_argument('--contrastive_norm_power', type=float, default=None,
                   help='symmetric: hinge-mean violation-count sensitivity '
                        '(1=active-only/plateau, 0=all-pairs, 0.5=blend; '
                        'override cfg.CONTRASTIVE_NORM_POWER)')
    p.add_argument('--contrastive_single_class_weight', type=float, default=None,
                   help='symmetric: very-low weight for full real images')
    return p


@torch.no_grad()
def _run_epoch_viz(model: nn.Module, items: list, epoch: int, out_dir: str, device: torch.device, cfg: Config):
    import torchvision.transforms.functional as TF
    from torchvision import transforms
    from lab_utils.viz import heatmap_rgb, overlay_blend, mask_tint, save_composite
    from lab_utils.eval.partition import spherical_kmeans2
    from contrastive_inpainting_v1.diagnose.polarity import polarity_attn
    from contrastive_inpainting_v1.scripts.swin_outlier_decode import _outlier_score, _gap_thr, _otsu_thr

    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    n = cfg.resolution.num_patches_per_side
    T = cfg.resolution.image_size
    normalize = transforms.Normalize(list(cfg.IMAGENET_MEAN), list(cfg.IMAGENET_STD))

    log_line(f'[eval] viz epoch={epoch} running composite viz on {len(items)} items → {out_dir}/')

    for idx, it in enumerate(items):
        img_path = str(it.get('img', ''))
        try:
            source = Image.open(img_path).convert('RGB')
        except Exception as e:
            log_line(f'[eval] viz WARN failed to load {img_path}: {e}')
            continue

        W, H = source.size
        src_np = np.asarray(source, dtype=np.uint8)
        viz_hw = (H, W)

        gt = None
        mask_path = it.get('mask')
        if mask_path:
            try:
                gt_img = Image.open(str(mask_path)).convert('L')
                if gt_img.size != (W, H):
                    gt_img = gt_img.resize((W, H), Image.NEAREST)
                gt = np.asarray(gt_img, dtype=np.uint8) > 0
            except Exception:
                pass

        inp = normalize(TF.to_tensor(TF.resize(source, [T, T], Image.BILINEAR))).unsqueeze(0).to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=(device.type == 'cuda')):
            out = model(inp)

        panels = [('Original', src_np)]

        att = out.get('pool_attention')
        att_np = None
        if att is not None:
            att_np = att[0].detach().cpu().float().numpy()
            att_heat = heatmap_rgb(att_np.reshape(n, n), viz_hw)
            panels.append(('BCE Attention', overlay_blend(src_np, att_heat)))

        z = out.get('contrastive')
        if z is not None:
            z_np = z[0].detach().cpu().float().numpy()
            score = _outlier_score(z_np, att_np)
            out_heat = heatmap_rgb(score.reshape(n, n), viz_hw)
            panels.append(('Outlier Score', overlay_blend(src_np, out_heat)))

            if gt is not None:
                panels.append(('GT Mask', mask_tint(src_np, gt, viz_hw, (0, 255, 0))))

            raw_labels, _ = spherical_kmeans2(z_np, n_init=4)
            if att_np is not None:
                km_mask = polarity_attn(raw_labels, att_np)
            else:
                km_mask = raw_labels if raw_labels.sum() <= len(raw_labels) / 2 else 1 - raw_labels
            panels.append(('K-means', mask_tint(src_np, km_mask.reshape(n, n).astype(bool), viz_hw, (0, 140, 255))))

            score_flat = score.reshape(-1)
            gap_thr = _gap_thr(score_flat)
            panels.append(('Gap', mask_tint(src_np, (score.reshape(n, n) >= gap_thr), viz_hw, (255, 120, 0))))

            otsu_thr = _otsu_thr(score_flat)
            panels.append(('Otsu', mask_tint(src_np, (score.reshape(n, n) >= otsu_thr), viz_hw, (255, 255, 0))))

        pl = out.get('patch_logit')
        if pl is not None:
            pl_np = pl[0].detach().cpu().float().numpy()
            panels.append(('Patch BCE', mask_tint(src_np, (pl_np >= 0).reshape(n, n), viz_hw, (200, 0, 255))))

        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        save_path = os.path.join(out_dir, f'{idx:03d}_{stem}.png')
        save_composite(panels, save_path, panel_size=280)

    log_line(f'[eval] viz epoch={epoch} saved {len(items)} composites')


# ── training ─────────────────────────────────────────────────────────────────

def main():
    args = _build_parser().parse_args()
    from contrastive_inpainting_v1.pipeline.cli import apply_path_defaults
    apply_path_defaults(args)

    # ── DDP setup ─────────────────────────────────────────────────────────────
    # torchrun sets RANK/LOCAL_RANK/WORLD_SIZE; single-process falls back cleanly.
    ctx: DistributedContext = ddp_setup()
    # Tear the process group down on ANY exit (normal completion OR crash) so a
    # back-to-back sweep loop isn't blocked by a leaked NCCL communicator /
    # lingering rendezvous from the previous run. No-op when single-process.
    atexit.register(ddp_cleanup)
    torch.manual_seed(args.seed + ctx.rank)
    np.random.seed(args.seed + ctx.rank)

    # Rank 0 owns all logging + the run dir. Other ranks silence log_line so
    # torchrun --tee does not N×-duplicate every line. Eval / checkpoint are
    # also rank-0 only (below).
    if ctx.is_main:
        from lab_utils.logging.run_dir import build_run_dir as _build_run_dir
        _ckpt_root_abs = os.path.abspath(args.checkpoint_root)
        _run_dir = _build_run_dir(
            os.path.dirname(_ckpt_root_abs),
            os.path.basename(_ckpt_root_abs),
            role='train',
        )
        install_log(str(_run_dir.log_path))
    else:
        global log_line
        log_line = lambda *_a, **_k: None  # noqa: E731  rank>0 stays quiet

    device = torch.device(
        f'cuda:{ctx.local_rank}'
        if (args.device == 'cuda' and torch.cuda.is_available())
        else args.device
    )
    log_line(f'[dist] rank={ctx.rank}/{ctx.world_size} local_rank={ctx.local_rank} '
             f'is_main={ctx.is_main} device={device}')

    # Sanity: at least one head must be active (per multi_head_detector ctor).
    if args.contrastive_dim <= 0 and args.pool_hidden <= 0:
        raise ValueError('Both heads disabled (contrastive_dim=0 and pool_hidden=0).')
    bce_active  = args.pool_hidden     > 0 and args.lambda_image_bce  > 0.0
    cont_active = args.contrastive_dim > 0 and args.lambda_contrastive > 0.0
    patch_active = args.patch_bce and args.lambda_patch_bce > 0.0
    log_line(
        f'[cfg] device={device} '
        f'heads: contrastive_dim={args.contrastive_dim} pool_hidden={args.pool_hidden} '
        f'patch_bce={args.patch_bce} | '
        f'lambdas: image_bce={args.lambda_image_bce} contrastive={args.lambda_contrastive} '
        f'patch_bce={args.lambda_patch_bce} (pos_weight={args.patch_pos_weight}) | '
        f'active: bce={bce_active} contrastive={cont_active} patch={patch_active}'
    )

    cfg = Config()

    # CLI overrides for the contrastive loss knobs (None ⇒ keep cfg default).
    if args.contrastive_loss_mode is not None:
        cfg.CONTRASTIVE_LOSS_MODE = args.contrastive_loss_mode
    if args.tau_pos is not None:
        cfg.TAU_POS = args.tau_pos
    if args.tau_neg is not None:
        cfg.TAU_NEG = args.tau_neg
    if args.area_balance_power is not None:
        cfg.AREA_BALANCE_POWER = args.area_balance_power
    if args.contrastive_norm_power is not None:
        cfg.CONTRASTIVE_NORM_POWER = args.contrastive_norm_power
    if args.contrastive_single_class_weight is not None:
        cfg.CONTRASTIVE_SINGLE_CLASS_WEIGHT = args.contrastive_single_class_weight
    if cont_active:
        if cfg.CONTRASTIVE_LOSS_MODE == 'symmetric':
            log_line(
                f'[cfg] contrastive=symmetric tau_pos={cfg.TAU_POS} '
                f'tau_neg={cfg.TAU_NEG} area_balance_power={cfg.AREA_BALANCE_POWER} '
                f'norm_power={cfg.CONTRASTIVE_NORM_POWER} '
                f'single_class_weight={cfg.CONTRASTIVE_SINGLE_CLASS_WEIGHT}'
            )
        else:
            log_line(
                f'[cfg] contrastive=legacy neg_margin={cfg.NEG_MARGIN} '
                f'attract_margin={cfg.ATTRACT_MARGIN}'
            )

    # ── data ────────────────────────────────────────────────────────────────
    spec = IMD2020BCESpec(
        imd2020_root=args.imd2020_root,
        casia_root=args.casia_root,
        indoor_root=args.indoor_root,
        coco_inpaint_root=args.coco_inpaint_root,
        sagid_root=args.sagid_root,
        bfree_root=args.bfree_root,
        imd_train=not args.imd_val_only,
        casia_train=args.casia_train,
    )
    train_items, val_items = spec.build_items(cfg)
    log_line(f'[cfg] train={len(train_items)} val={len(val_items)}')

    # Loud, non-silent guard: a requested root that resolved to nothing means
    # that source is absent from BOTH train and eval (the classic "no auroc
    # because the loader was empty" failure). Warn per source.
    from collections import Counter as _Counter
    _all_src = _Counter(str(it.get('source', '')) for it in (train_items + val_items))
    for _root, _src, _flag in (
        (args.imd2020_root, 'imd2020', '--imd2020_root'),
        (args.casia_root, 'casia', '--casia_root'),
        (args.coco_inpaint_root, 'coco_inpaint', '--coco_inpaint_root'),
        (args.sagid_root, 'sagid', '--sagid_root'),
        (args.bfree_root, 'bfree', '--bfree_root'),
    ):
        if _root and _all_src.get(_src, 0) == 0:
            log_line(f'[cfg] WARN: {_flag}={_root!r} resolved but indexed 0 items '
                     f'(source={_src!r}) — check the folder layout/subdir names.')
        elif not _root:
            log_line(f'[cfg] note: {_flag} not set — source {_src!r} absent from '
                     f'train AND eval.')

    # ── TGIF FR half-split fine-tune ──────────────────────────────────────
    # tgif_eval_ids is the contract with the eval/robust blocks below: when it
    # is non-None, every build_tgif2_items call there filters to these held-out
    # coco_ids and to FR-type fakes, so eval never sees a trained-on source
    # image (or its pristine original).
    tgif_eval_ids: Optional[set] = None
    if args.tgif_train_half:
        if not args.tgif_root:
            raise ValueError('--tgif_train_half requires --tgif_root')
        _tg_index = args.tgif_index or os.path.join(args.tgif_root, 'tgif2_index.json')
        _tr_ids, _ev_ids = split_tgif2_coco_ids(
            _tg_index, train_frac=args.tgif_half_frac, seed=args.tgif_half_seed)
        tgif_eval_ids = set(_ev_ids)
        ft_fakes, ft_reals = build_tgif2_items(
            args.tgif_root, args.tgif_index, include_reals=True,
            coco_ids=set(_tr_ids), types={'fr'})
        if args.tgif_model:
            ft_fakes = _tgif_model_filter(
                ft_fakes, args.tgif_model, log_tag='[cfg]', tag='tgif_train_half')
        _tgif_mask_cache = os.path.join(args.checkpoint_root, 'tgif_mask_cache')
        ft_fakes = _prep_tgif_items(
            ft_fakes, mask_cache_dir=_tgif_mask_cache,
            log_tag='[cfg]', tag='tgif_train_half fakes')
        ft_reals = _prep_tgif_items(
            ft_reals, mask_cache_dir=_tgif_mask_cache,
            log_tag='[cfg]', tag='tgif_train_half reals')
        if not ft_fakes or not ft_reals:
            raise ValueError(
                f'--tgif_train_half resolved an empty train half '
                f'(fakes={len(ft_fakes)} reals={len(ft_reals)}) — check '
                f'--tgif_root/--tgif_index/--tgif_model.')
        train_items = list(train_items) + ft_fakes + ft_reals
        log_line(f'[cfg] tgif_train_half: seed={args.tgif_half_seed!r} '
                 f'frac={args.tgif_half_frac} coco_ids: '
                 f'train={len(_tr_ids)} eval={len(_ev_ids)} | '
                 f'train half (fr): fakes={len(ft_fakes)} reals={len(ft_reals)} '
                 f'→ train_items={len(train_items)}')
        if ctx.is_main:
            os.makedirs(args.checkpoint_root, exist_ok=True)
            for _fname, _ids in (('tgif_train_coco_ids.txt', _tr_ids),
                                 ('tgif_eval_coco_ids.txt', _ev_ids)):
                with open(os.path.join(args.checkpoint_root, _fname), 'w') as _f:
                    _f.write('\n'.join(_ids) + '\n')
            log_line(f'[cfg] tgif_train_half: wrote tgif_train_coco_ids.txt + '
                     f'tgif_eval_coco_ids.txt → {args.checkpoint_root}')

    aug_kwargs = build_aug_kwargs(cfg, args.aug_intensity)
    # Optional gaussian-noise override (laundering robustness): bump prob / max
    # std above the preset so the semantic pathway must carry the decision.
    if args.noise_prob is not None:
        aug_kwargs['noise_prob'] = float(args.noise_prob)
    if args.noise_std_max is not None:
        aug_kwargs['noise_std_max'] = float(args.noise_std_max)
        aug_kwargs['noise_std_min'] = min(
            float(aug_kwargs.get('noise_std_min', 0.0)), float(args.noise_std_max))
    if args.noise_prob is not None or args.noise_std_max is not None:
        log_line(f'[cfg] noise override: prob={aug_kwargs["noise_prob"]:.2f} '
                 f'std=[{aug_kwargs["noise_std_min"]:.4f},{aug_kwargs["noise_std_max"]:.4f}]')
    log_line(f'[cfg] aug_intensity={args.aug_intensity} aug_kwargs={aug_kwargs}')
    if args.aug_curriculum:
        curr_lo = build_light_aug_kwargs(cfg)
        curr_hi = build_heavy_aug_kwargs(cfg)
        log_line(f'[cfg] aug_curriculum=ON max_strength={args.aug_curriculum_max:.2f} '
                 f'(per-epoch light→heavy lerp; overrides aug_intensity for TRAIN)')

    if args.use_crop_mix:
        crop_mix = DEFAULT_CROP_MIX
        crop_scale_fallback = (DEFAULT_CROP_MIX[0][0][0], DEFAULT_CROP_MIX[-1][0][1])
    else:
        crop_mix = None
        crop_scale_fallback = (float(args.train_crop_min), 1.00)
    log_line(f'[cfg] crop_mix={"on" if crop_mix else "off"} '
             f'fallback_range={crop_scale_fallback}')
    log_line(f'[cfg] crop ratio unified for reals+splices: {cfg.CROP_RATIO}')

    # Degradation harness pass-through (default OFF ⇒ identical to before). When
    # either knob is on, splice-region degradation / whole-image corruption are
    # applied to TRAIN with the heavy-aug down-weight on over-destroyed samples.
    # MATCHED real degradation: when splice images get local degradation, real
    # negatives MUST get a comparable rate of local artifacts or the BCE/
    # contrastive heads learn "local noise/compression blob present => fake" — a
    # shortcut that inflates in-domain AUROC and collapses OOD (exactly like the
    # zoom shortcut). Default: track --use_splice_degradation. The degrade head
    # supervises these real artifacts (target_clean_prob keeps a clean fraction),
    # so the splice head sees matched artifact marginals.
    use_real_degradation = (
        bool(args.use_splice_degradation)
        if args.use_real_degradation is None
        else bool(args.use_real_degradation)
    )
    degrade_kw = dict(
        use_degradation=use_real_degradation,
        use_invariance=False,
        use_splice_degradation=bool(args.use_splice_degradation),
        splice_degradation_prob=cfg.SPLICE_DEGRADATION_PROB,
        splice_mask_corrupt_prob=cfg.SPLICE_MASK_CORRUPT_PROB,
        splice_mask_loss_weight=cfg.SPLICE_MASK_LOSS_WEIGHT,
        whole_image_corrupt_prob=float(args.whole_corrupt_prob),
        heavy_whole_aug_severity_thresh=cfg.HEAVY_WHOLE_AUG_SEVERITY_THRESH,
        heavy_aug_degrade_loss_weight=cfg.HEAVY_AUG_DEGRADE_LOSS_WEIGHT,
        degradation_kwargs=build_degradation_kwargs(cfg),
    )
    if args.use_splice_degradation or args.whole_corrupt_prob > 0:
        log_line(f'[cfg] degradation ON: splice_degrade={args.use_splice_degradation} '
                 f'real_degrade={use_real_degradation} (matched-artifact balance) '
                 f'whole_corrupt_prob={args.whole_corrupt_prob} '
                 f'(NEWLY wired into multi-head trainer — verify a smoke epoch)')
        if args.use_splice_degradation and not use_real_degradation:
            log_line('[cfg] WARN: splice degradation ON but real degradation OFF — '
                     'the "local artifact = fake" shortcut is ACTIVE. Pass '
                     '--use_real_degradation to match (recommended).')

    train_ds = LabDataset(
        train_items, cfg.resolution,
        cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
        augment=True,
        light_aug_kwargs=aug_kwargs,
        crop_scale=crop_scale_fallback,
        crop_ratio=cfg.CROP_RATIO,
        imd_crop_scale=crop_scale_fallback,
        imd_crop_ratio=cfg.CROP_RATIO,
        crop_max_tries=int(args.crop_max_tries),
        min_mask_patch_frac=float(args.min_mask_patch_frac),
        splice_crop_mode=str(args.splice_crop_mode),
        oracle_target_cov=tuple(args.oracle_cov),
        splice_cov_band=(tuple(args.splice_cov_band) if args.splice_cov_band else None),
        crop_scale_mix=crop_mix,
        imd_crop_scale_mix=crop_mix,
        **degrade_kw,
        gt_patch_threshold=float(args.gt_patch_threshold),
        gt_soft_label=bool(args.gt_soft_label),
        gt_soft_low=float(args.gt_soft_low),
        gt_soft_high=float(args.gt_soft_high),
        paste_frac=float(args.paste_frac),
    )
    log_line(f'[cfg] splice_crop_mode={args.splice_crop_mode} '
             f'min_mask_patch_frac={args.min_mask_patch_frac} '
             f'crop_max_tries={args.crop_max_tries} '
             f'oracle_cov={tuple(args.oracle_cov)}')
    swin_status = 'OFF'
    if (args.contrastive_dim > 0 and args.pool_hidden > 0 and args.swin_every > 0):
        swin_status = (
            f'every {args.swin_every} ep, scale={args.swin_scale}, '
            f'stride_frac={args.swin_stride_frac}, source-square crops, '
            f'OR over BCE-positive windows, max_windows=8'
        )
    log_line(f'[cfg] robust_every={args.robust_every} '
             f'eval_max_items={args.eval_max_items} swin={swin_status}')
    log_line(f'[cfg] gt_patch_threshold={args.gt_patch_threshold:.3f} '
             f'gt_soft_label={bool(args.gt_soft_label)} '
             f'gt_soft_low={args.gt_soft_low:.3f} '
             f'gt_soft_high={args.gt_soft_high:.3f}')

    if args.splice_mix:
        # Parse SRC=FRAC tokens → {source: fraction}.
        source_fracs: Dict[str, float] = {}
        for tok in args.splice_mix:
            if '=' not in tok:
                raise ValueError(f'--splice_mix expects SRC=FRAC tokens, got {tok!r}')
            src, frac = tok.split('=', 1)
            source_fracs[src.strip()] = float(frac)
        from lab_utils.data.sampling import source_splice_balance_weights
        weights, mix_stats = source_splice_balance_weights(
            train_items, source_fracs, target_splice_frac=0.5)
        log_line(f'[cfg] splice_mix (per-source splice budget): {source_fracs}')
        log_line(f'[cfg] splice_mix realized: {mix_stats}')
        if 'excluded_sources' in mix_stats:
            log_line(f'[cfg] WARN: splice sources present but EXCLUDED from draws '
                     f'(no fraction given): {mix_stats["excluded_sources"]}')
    else:
        weights = _splice_balance_weights(train_items)
    # DDP: each rank draws its own shard of the per-epoch sample budget from the
    # full weighted distribution (per-rank generator seed → different draws).
    per_rank_samples = max(1, args.train_samples // max(1, ctx.world_size))
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=per_rank_samples, replacement=True,
        generator=torch.Generator().manual_seed(args.seed * 100 + ctx.rank),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=lab_collate_fn,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
    )

    # Val splits — capped per source for the per-epoch full eval
    imd_val_items   = [it for it in val_items if it.get('source', '') == 'imd2020']
    casia_val_items = [it for it in val_items if it.get('source', '') == 'casia']
    coco_val_items  = [it for it in val_items if it.get('source', '') == 'coco_inpaint']
    sagid_val_items = [it for it in val_items if it.get('source', '') == 'sagid']
    bfree_val_items = [it for it in val_items if it.get('source', '') == 'bfree']
    log_line(f'[cfg] imd_val={len(imd_val_items)} casia_val={len(casia_val_items)} '
             f'coco_inpaint_val={len(coco_val_items)} sagid_val={len(sagid_val_items)} '
             f'bfree_val={len(bfree_val_items)}')

    imd_eval_items   = _subsample_items(imd_val_items,   args.eval_max_items, seed='full_eval')
    casia_eval_items = _subsample_items(casia_val_items, args.eval_max_items, seed='full_eval')
    coco_eval_items  = _subsample_items(coco_val_items,  args.eval_max_items, seed='full_eval')
    sagid_eval_items = _subsample_items(sagid_val_items, args.eval_max_items, seed='full_eval')
    bfree_eval_items = _subsample_items(bfree_val_items, args.eval_max_items, seed='full_eval')
    log_line(f'[cfg] full_eval cap: imd={len(imd_eval_items)}/{len(imd_val_items)} '
             f'casia={len(casia_eval_items)}/{len(casia_val_items)} '
             f'coco_inpaint={len(coco_eval_items)}/{len(coco_val_items)} '
             f'sagid={len(sagid_eval_items)}/{len(sagid_val_items)} '
             f'bfree={len(bfree_eval_items)}/{len(bfree_val_items)}')

    def _sub_loader(items):
        if not items:
            return None
        ds = LabDataset(
            items, cfg.resolution,
            cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
            augment=False,
            use_degradation=False, use_invariance=False, use_splice_degradation=False,
            gt_patch_threshold=float(args.gt_patch_threshold),
        )
        return build_eval_loader(ds, LoaderConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
        ))

    imd_val_loader   = _sub_loader(imd_eval_items)
    casia_val_loader = _sub_loader(casia_eval_items)
    coco_val_loader  = _sub_loader(coco_eval_items)
    sagid_val_loader = _sub_loader(sagid_eval_items)
    bfree_val_loader = _sub_loader(bfree_eval_items)

    viz_items = []
    if args.viz_every > 0:
        # Pool the viz composites ACROSS all val sources so the panel reflects
        # every distribution we care about (esp. the OOD ones that are dropping),
        # not just IMD. Even per-source splice budget; reals from the two sources
        # that have them.
        _viz_srcs = [
            ('imd',   imd_val_items),
            ('casia', casia_val_items),
            ('coco',  coco_val_items),
            ('sagid', sagid_val_items),
            ('bfree', bfree_val_items),
        ]
        _vs, _vr = [], []
        for _name, _items in _viz_srcs:
            _s = [it for it in _items if it.get('kind') in _SPLICE_KINDS and it.get('mask')]
            _r = [it for it in _items if 'real' in str(it.get('kind', ''))]
            _vs += _subsample_items(_s, 2, seed=f'viz_{_name}')
            _vr += _subsample_items(_r, 1, seed=f'viz_{_name}')
        viz_items = _vs + _subsample_items(_vr, 2, seed='viz_real')

    # ── model ────────────────────────────────────────────────────────────────
    model = build_multi_head_detector(
        model_name=cfg.MODEL_NAME,
        resolution=cfg.resolution,
        lora_rank=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        lora_targets=cfg.LORA_TARGETS,
        contrastive_dim=args.contrastive_dim,
        pool_hidden=args.pool_hidden,
        patch_bce=args.patch_bce,
        device=device,
    )
    if args.no_grad_ckpt:
        model.backbone.gradient_checkpointing_disable()
        log_line('[cfg] gradient_checkpointing DISABLED (--no_grad_ckpt)')
    has_bce_head = args.pool_hidden > 0

    start_epoch = 0
    if args.resume:
        ckpt = ckpt_load(args.resume)
        model.load_state_dict(ckpt['model'])
        start_epoch = int(ckpt.get('epoch', 0))
        log_line(f'[ckpt] resumed epoch={start_epoch} ← {args.resume}')

    # DDP-wrap after loading any resume weights (load into the bare module).
    # find_unused_parameters=False: every head config exercises all of its
    # params each step (the contrastive proj is touched even on splice-free
    # batches via z.sum()*0), matching the working BCE trainer and avoiding the
    # gradient-checkpointing "marked ready twice" DDP error.
    model = wrap_model(model, ctx, device=device, find_unused_parameters=False)

    # ── optimizer / schedule ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, math.ceil(per_rank_samples / args.batch_size / args.grad_accum))
    total_steps = args.num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.05,
    )

    pos_weight = torch.ones(1, dtype=torch.float32, device=device)
    log_line('[cfg] pos_weight=1.0 (sampler-balanced, no class reweight)')

    # ── mixed precision ──────────────────────────────────────────────────────
    use_amp = args.bf16 and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp else None
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        log_line('[cfg] bf16 mixed-precision ENABLED via torch.cuda.amp')

    # ── training loop ────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(start_epoch, args.num_epochs):
        # ── aug curriculum: set this epoch's TRAIN aug before the loader
        # iterator is created (workers fork per epoch and inherit the change;
        # num_workers=0 reads it directly). CLI noise overrides stay on top.
        if args.aug_curriculum:
            s = min(1.0, epoch / max(args.num_epochs - 1, 1)) * args.aug_curriculum_max
            ck = _interp_aug_kwargs(curr_lo, curr_hi, s)
            if args.noise_prob is not None:
                ck['noise_prob'] = float(args.noise_prob)
            if args.noise_std_max is not None:
                ck['noise_std_max'] = float(args.noise_std_max)
                ck['noise_std_min'] = min(ck['noise_std_min'], float(args.noise_std_max))
            train_ds.light_aug_kwargs = ck
            log_line(f'[cfg] aug_curriculum epoch={epoch} strength={s:.2f} '
                     f'jpeg(p={ck["jpeg_prob"]:.2f} q>={ck["jpeg_q_min"]}) '
                     f'noise(p={ck["noise_prob"]:.2f} s<={ck["noise_std_max"]:.3f}) '
                     f'resize(p={ck["resize_prob"]:.2f} >={ck["resize_scale_min"]:.2f}) '
                     f'blur(p={ck["blur_prob"]:.2f} s<={ck["blur_sigma_max"]:.2f})')
        model.train()
        optimizer.zero_grad()
        loss_acc_total = 0.0
        loss_acc_bce   = 0.0
        loss_acc_cont  = 0.0
        contr_n_active_seen = 0
        contr_n_total_seen  = 0
        contr_ignored       = 0      # splice images below 1% in-frame → no contrastive penalty
        ignore_band_acc     = 0.0    # patch-weight ignore-band frac, splice items
        soft_pos_band_acc   = 0.0    # patch-weight soft-pos-band frac, splice items
        diag_steps_observed = 0      # steps where cont_diag was produced
        sim_pos_acc = sim_neg_acc = real_sim_acc = 0.0  # symmetric-mode sim diag
        repel_af_acc = 0.0  # symmetric-mode repel active-pair fraction (median)
        sim_diag_steps = 0
        bce_correct = bce_total = 0
        bce_ignored = 0              # splice images below 1% in-frame → no penalty
        loss_acc_patch = 0.0
        patch_pos_acc = patch_pred_acc = 0.0   # patch-BCE diag (target/pred pos frac)
        patch_diag_steps = 0
        n_steps_observed = 0
        # Attention-zoom second-pass telemetry + gate (needs the BCE pool's
        # attention map; earlier epochs have uninformative attention).
        zoom_pass_active = (
            args.zoom_pass_prob > 0
            and epoch >= args.zoom_pass_start_epoch
            and has_bce_head
        )
        zoom_steps_fired = 0
        zoom_n_samples = zoom_n_real = zoom_n_ignored = 0
        zoom_loss_acc = 0.0
        if zoom_pass_active and epoch == args.zoom_pass_start_epoch:
            log_line(f'[train] attention-zoom pass ACTIVE from epoch={epoch} '
                     f'(prob={args.zoom_pass_prob} lambda_zoom={args.lambda_zoom} '
                     f'max_frame_frac={args.zoom_pass_max_frac})')
        t0 = time.time()
        _n_batches = max(1, math.ceil(per_rank_samples / args.batch_size))
        _pbar = tqdm(enumerate(train_loader), total=_n_batches,
                     desc=f'Epoch {epoch}/{args.num_epochs-1}',
                     dynamic_ncols=True, leave=True)

        for step, batch in _pbar:
            if batch is None:
                continue

            img  = batch['img'].to(device, non_blocking=True)
            meta_list = batch['meta'] if isinstance(batch['meta'], list) else [
                {k: v[i] for k, v in batch['meta'].items()}
                for i in range(img.shape[0])
            ]
            kinds = [m.get('kind', '') for m in meta_list]
            label = torch.tensor(
                [0.0 if k in _REAL_KINDS else 1.0 for k in kinds],
                dtype=torch.float32, device=device,
            )                                              # (B,) {0,1}
            is_splice = (label > 0.5)                       # (B,) bool

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                out = model(img)
                image_logit = out['image_logit']
                z_contrastive = out['contrastive']

            # ── BCE branch ────────────────────────────────────────────────
            if bce_active and image_logit is not None:
                # Image-level IGNORE: a splice whose in-frame coverage fell below
                # the 1% TP threshold comes back is_single. Give it NO penalty —
                # neither forced-positive nor relabeled-real — by zeroing its BCE
                # weight. Reals and supervised splices keep weight 1. (Under
                # oracle_fallback this is a rare safety net, not a behavior change.)
                is_single = batch['is_single'].to(device, non_blocking=True).bool()
                img_bce_w = torch.ones_like(label)
                ignore_mask = is_splice & is_single
                img_bce_w[ignore_mask] = 0.0
                bce_per = F.binary_cross_entropy_with_logits(
                    image_logit, label, pos_weight=pos_weight.expand_as(label),
                    reduction='none',
                )
                bce_loss = (bce_per * img_bce_w).sum() / img_bce_w.sum().clamp(min=1.0)
                with torch.no_grad():
                    sup = img_bce_w > 0
                    bce_correct += int((((image_logit.detach() >= 0).float() == label) & sup).sum().item())
                    bce_total   += int(sup.sum().item())
                    bce_ignored += int(ignore_mask.sum().item())
            else:
                bce_loss = torch.tensor(0.0, device=device)

            # ── Contrastive branch (splice items only) ────────────────────
            cont_diag = None
            if cont_active and z_contrastive is not None:
                splice_labels        = batch['splice_labels'].to(device, non_blocking=True)
                splice_loss_weight   = batch['splice_loss_weight'].to(device, non_blocking=True)
                splice_patch_weights = batch['splice_patch_weights'].to(device, non_blocking=True)
                is_single = batch['is_single'].to(device, non_blocking=True).bool()
                miss_splice = is_splice & is_single
                contrastive_splice_active = is_splice & ~is_single
                contr_ignored += int(miss_splice.sum().item())
                if cfg.CONTRASTIVE_LOSS_MODE == 'symmetric':
                    # Symmetric path: reals ARE included but weighted very low
                    # (single_class_weight). Missed/sub-threshold splice crops
                    # are ignored so the localization head never receives a
                    # weak contradictory signal from crops the image head ignores.
                    symmetric_active = ~miss_splice
                    cont_loss, cont_diag = selective_symmetric_contrastive_loss(
                        z_contrastive, splice_labels,
                        is_single_class=is_single,
                        active_mask=symmetric_active,
                        tau_pos=cfg.TAU_POS, tau_neg=cfg.TAU_NEG,
                        lambda_repel=cfg.LAMBDA_REPEL,
                        single_class_weight=cfg.CONTRASTIVE_SINGLE_CLASS_WEIGHT,
                        area_balance_power=cfg.AREA_BALANCE_POWER,
                        norm_power=cfg.CONTRASTIVE_NORM_POWER,
                        sample_weights=splice_loss_weight,
                        patch_weights=splice_patch_weights,
                        diversity_weight=cfg.CONTRASTIVE_DIVERSITY_WEIGHT,
                        diversity_tau=cfg.CONTRASTIVE_DIVERSITY_TAU,
                        topk=cfg.CONTRASTIVE_TOPK,
                    )
                else:
                    cont_loss, cont_diag = selective_contrastive_loss(
                        z_contrastive, splice_labels,
                        is_single_class=~is_splice,
                        active_mask=contrastive_splice_active,
                        neg_margin=cfg.NEG_MARGIN,
                        lambda_repel=cfg.LAMBDA_REPEL,
                        single_class_weight=0.0,       # reals get exactly 0 supervision
                        sample_weights=splice_loss_weight,
                        patch_weights=splice_patch_weights,
                        attract_margin=cfg.ATTRACT_MARGIN,
                        single_class_attract_margin=cfg.SINGLE_CLASS_ATTRACT_MARGIN,
                        single_class_attract_squared=cfg.SINGLE_CLASS_ATTRACT_SQUARED,
                        single_class_topk=cfg.SINGLE_CLASS_TOPK,
                    )
                contr_n_active_seen += int(contrastive_splice_active.sum().item())
                contr_n_total_seen  += int(is_splice.sum().item())
                if cont_diag is not None:
                    ignore_band_acc   += float(cont_diag.get('ignore_band_frac', 0.0))
                    soft_pos_band_acc += float(cont_diag.get('soft_pos_band_frac', 0.0))
                    diag_steps_observed += 1
                    spm = cont_diag.get('sim_pos_median', float('nan'))
                    if spm == spm:  # symmetric-mode diag present (not NaN)
                        sim_pos_acc  += float(spm)
                        sim_neg_acc  += float(cont_diag.get('sim_neg_median', 0.0))
                        rsm = cont_diag.get('real_sim_median', float('nan'))
                        real_sim_acc += float(rsm) if rsm == rsm else 0.0
                        raf = cont_diag.get('repel_active_frac_median', float('nan'))
                        repel_af_acc += float(raf) if raf == raf else 0.0
                        sim_diag_steps += 1
            else:
                cont_loss = torch.tensor(0.0, device=device)

            # ── Patch-BCE branch (dense supervised splice flagging) ───────
            patch_diag = None
            patch_logit = out.get('patch_logit')
            if patch_active and patch_logit is not None:
                splice_labels        = batch['splice_labels'].to(device, non_blocking=True)
                splice_loss_weight   = batch['splice_loss_weight'].to(device, non_blocking=True)
                splice_patch_weights = batch['splice_patch_weights'].to(device, non_blocking=True)
                is_single = batch['is_single'].to(device, non_blocking=True).bool()
                # Reals (all-zero labels) supervise EVERY patch as negative →
                # trains specificity; supervised splices use their boundary band;
                # missed-splice crops (splice present but sub-threshold) ignored.
                pw = torch.where(
                    is_splice.view(-1, 1),
                    splice_patch_weights.clamp(0.0, 1.0),
                    torch.ones_like(splice_patch_weights),
                )
                active = ~(is_splice & is_single)
                patch_loss, patch_diag = selective_patch_bce_loss(
                    patch_logit, splice_labels,
                    active_mask=active,
                    pos_weight=args.patch_pos_weight,
                    sample_weights=splice_loss_weight,
                    patch_weights=pw,
                )
            else:
                patch_loss = torch.tensor(0.0, device=device)

            total_loss = (
                args.lambda_image_bce   * bce_loss +
                args.lambda_contrastive * cont_loss +
                args.lambda_patch_bce   * patch_loss
            )
            scaler.scale(total_loss / args.grad_accum).backward()

            # ── Attention-zoom second pass (detect-then-zoom training) ────
            # Same batch, zoomed to each sample's hottest attention region —
            # splices train the zoomed resolutions/framings, reals train FP
            # resistance on the model's own most-suspicious crops. Runs as its
            # OWN forward+backward after the main backward: the backbone uses
            # gradient checkpointing, and summing two forwards into one DDP
            # backward marks each checkpointed param ready twice. Two sync'd
            # backwards per step is the standard-safe pattern; grads accumulate
            # until the grad_accum optimizer step.
            if (zoom_pass_active and out['pool_attention'] is not None
                    and random.random() < args.zoom_pass_prob):
                zres = _attention_zoom_second_pass(
                    model, img, out['pool_attention'],
                    batch['splice_labels'].to(device, non_blocking=True),
                    is_splice,
                    res=cfg.resolution,
                    gt_patch_threshold=float(args.gt_patch_threshold),
                    min_mask_patch_frac=float(args.min_mask_patch_frac),
                    max_frame_frac=float(args.zoom_pass_max_frac),
                )
                if zres is not None:
                    z_out, z_lbl, z_is_splice, z_miss, _z_idx = zres
                    z_label = z_is_splice.float()
                    z_bce = torch.tensor(0.0, device=device)
                    z_cont = torch.tensor(0.0, device=device)
                    z_patch = torch.tensor(0.0, device=device)
                    if bce_active and z_out['image_logit'] is not None:
                        z_w = torch.ones_like(z_label)
                        z_w[z_miss] = 0.0   # splice fell out of frame → no penalty
                        z_per = F.binary_cross_entropy_with_logits(
                            z_out['image_logit'], z_label,
                            pos_weight=pos_weight.expand_as(z_label),
                            reduction='none',
                        )
                        z_bce = (z_per * z_w).sum() / z_w.sum().clamp(min=1.0)
                    if cont_active and z_out['contrastive'] is not None:
                        if cfg.CONTRASTIVE_LOSS_MODE == 'symmetric':
                            z_cont, _ = selective_symmetric_contrastive_loss(
                                z_out['contrastive'], z_lbl,
                                is_single_class=(~z_is_splice) | z_miss,
                                active_mask=~z_miss,
                                tau_pos=cfg.TAU_POS, tau_neg=cfg.TAU_NEG,
                                lambda_repel=cfg.LAMBDA_REPEL,
                                single_class_weight=cfg.CONTRASTIVE_SINGLE_CLASS_WEIGHT,
                                area_balance_power=cfg.AREA_BALANCE_POWER,
                                norm_power=cfg.CONTRASTIVE_NORM_POWER,
                                diversity_weight=cfg.CONTRASTIVE_DIVERSITY_WEIGHT,
                                diversity_tau=cfg.CONTRASTIVE_DIVERSITY_TAU,
                                topk=cfg.CONTRASTIVE_TOPK,
                            )
                        else:
                            z_cont, _ = selective_contrastive_loss(
                                z_out['contrastive'], z_lbl,
                                is_single_class=~z_is_splice,
                                active_mask=z_is_splice & ~z_miss,
                                neg_margin=cfg.NEG_MARGIN,
                                lambda_repel=cfg.LAMBDA_REPEL,
                                single_class_weight=0.0,
                                attract_margin=cfg.ATTRACT_MARGIN,
                                single_class_attract_margin=cfg.SINGLE_CLASS_ATTRACT_MARGIN,
                                single_class_attract_squared=cfg.SINGLE_CLASS_ATTRACT_SQUARED,
                                single_class_topk=cfg.SINGLE_CLASS_TOPK,
                            )
                    if patch_active and z_out.get('patch_logit') is not None:
                        z_patch, _ = selective_patch_bce_loss(
                            z_out['patch_logit'], z_lbl,
                            active_mask=~(z_is_splice & z_miss),
                            pos_weight=args.patch_pos_weight,
                        )
                    zoom_loss = (
                        args.lambda_image_bce   * z_bce +
                        args.lambda_contrastive * z_cont +
                        args.lambda_patch_bce   * z_patch
                    )
                    (args.lambda_zoom * zoom_loss / args.grad_accum).backward()
                    zoom_steps_fired += 1
                    zoom_n_samples += int(z_label.numel())
                    zoom_n_real    += int((~z_is_splice).sum().item())
                    zoom_n_ignored += int(z_miss.sum().item())
                    zoom_loss_acc  += float(zoom_loss.item())

            n_steps_observed += 1
            loss_acc_total += float(total_loss.item())
            loss_acc_patch += float(patch_loss.item())
            if patch_diag is not None:
                patch_pos_acc    += float(patch_diag.get('patch_pos_frac', 0.0))
                patch_pred_acc   += float(patch_diag.get('pred_pos_frac', 0.0))
                patch_diag_steps += 1
            loss_acc_bce   += float(bce_loss.item())
            loss_acc_cont  += float(cont_loss.item())

            # ── live tqdm postfix ──
            if n_steps_observed > 0:
                _pbar.set_postfix({
                    'bce': f'{loss_acc_bce / n_steps_observed:.4f}',
                    'ctr': f'{loss_acc_cont / n_steps_observed:.4f}',
                    'total': f'{loss_acc_total / n_steps_observed:.4f}',
                }, refresh=False)

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    elapsed = time.time() - t0
                    bce_str = f'{loss_acc_bce / n_steps_observed:.4f}' if bce_active else '--'
                    cont_str = f'{loss_acc_cont / n_steps_observed:.4f}' if cont_active else '--'
                    patch_str = f'{loss_acc_patch / n_steps_observed:.4f}' if patch_active else '--'
                    if patch_active and patch_diag_steps > 0:
                        patch_str += (f' pos={patch_pos_acc / patch_diag_steps:.3f}'
                                      f' pred={patch_pred_acc / patch_diag_steps:.3f}')
                    bce_acc_str = f'{bce_correct / max(bce_total, 1):.4f}' if bce_active else '--'
                    contr_active_frac = (contr_n_active_seen / max(contr_n_total_seen, 1)) \
                        if cont_active else 0.0
                    if cont_active and diag_steps_observed > 0:
                        ig_str = f'{ignore_band_acc / diag_steps_observed:.3f}'
                        sp_str = f'{soft_pos_band_acc / diag_steps_observed:.3f}'
                        soft_extra = f' ignore={ig_str} soft_pos={sp_str}'
                    else:
                        soft_extra = ''
                    if cont_active and sim_diag_steps > 0:
                        soft_extra += (
                            f' simpos={sim_pos_acc / sim_diag_steps:.3f}'
                            f' simneg={sim_neg_acc / sim_diag_steps:.3f}'
                            f' realsim={real_sim_acc / sim_diag_steps:.3f}'
                            f' repel_af={repel_af_acc / sim_diag_steps:.3f}'
                        )
                    log_line(
                        f'[train] epoch={epoch} step={global_step} '
                        f'loss={loss_acc_total / n_steps_observed:.4f} '
                        f'(bce={bce_str} contr={cont_str} patch={patch_str}) '
                        f'bce_acc={bce_acc_str} '
                        f'contr_active={contr_active_frac:.2f}{soft_extra} '
                        f'lr={scheduler.get_last_lr()[0]:.2e} '
                        f'elapsed={elapsed:.0f}s'
                    )

        # End-of-epoch flush
        if n_steps_observed % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        bce_str  = f'{loss_acc_bce  / max(n_steps_observed, 1):.4f}' if bce_active else '--'
        cont_str = f'{loss_acc_cont / max(n_steps_observed, 1):.4f}' if cont_active else '--'
        patch_str = f'{loss_acc_patch / max(n_steps_observed, 1):.4f}' if patch_active else '--'
        bce_acc_str = f'{bce_correct / max(bce_total, 1):.4f}' if bce_active else '--'
        _pbar.close()
        log_line(
            f'[train] epoch={epoch} DONE '
            f'loss={loss_acc_total / max(n_steps_observed, 1):.4f} '
            f'(bce={bce_str} contr={cont_str} patch={patch_str}) bce_acc={bce_acc_str} '
            f'bce_ignored={bce_ignored} contr_ignored={contr_ignored}'
        )
        if zoom_pass_active:
            log_line(
                f'[zoom] epoch={epoch} steps_fired={zoom_steps_fired}/{n_steps_observed} '
                f'samples={zoom_n_samples} (real={zoom_n_real} '
                f'splice_ignored={zoom_n_ignored}) '
                f'loss={zoom_loss_acc / max(zoom_steps_fired, 1):.4f}'
            )
        # Crop telemetry: what the splice cropping actually did this epoch.
        # oracle_rate climbing means many CASIA splices needed the mask-centered
        # zoom; dropped>0 means empty-mask samples were correctly discarded.
        cs = train_ds.drain_crop_stats()
        n_acc = cs['random'] + cs['oracle'] + cs['fallback']
        log_line(
            f'[crop] epoch={epoch} random={cs["random"]} oracle={cs["oracle"]} '
            f'fallback={cs["fallback"]} dropped={cs["dropped"]} '
            f'oracle_rate={cs["oracle"] / max(n_acc, 1):.3f} '
            f'splice_cov_mean={cs["cov_mean"]:.3f} (n={cs["cov_n"]})'
        )

        # ── eval + checkpoint: rank 0 only (other ranks wait at the barrier) ──
        if ctx.is_main:
            eval_model = unwrap_model(model)   # bare module for inference
            bce_adapter = _BCEHeadAdapter(eval_model) if has_bce_head else None

            # Build the unified eval sources list
            eval_sources = [
                (imd_val_loader,   imd_eval_items,   'imd_val'),
                (casia_val_loader, casia_eval_items, 'casia_val'),
                (coco_val_loader,  None,             'coco_inpaint_val'),
                (sagid_val_loader, None,             'sagid_val'),
                (bfree_val_loader, None,             'bfree_val'),
            ]

            # Treat TGIF 4 cells as first-class, equal eval sources
            if args.tgif_root:
                tg_fakes, tg_reals = build_tgif2_items(
                    args.tgif_root, args.tgif_index, include_reals=True,
                    coco_ids=tgif_eval_ids,
                    types=({'fr'} if tgif_eval_ids is not None else None))
                if tg_fakes and tg_reals:
                    tg_fakes = _tgif_model_filter(
                        tg_fakes, args.tgif_model, log_tag='[eval]', tag='tgif')
                    cells = _tgif_partition_cells(tg_fakes)
                    per_cell = max(1, int(args.tgif_n_fake) // max(len(cells), 1))
                    _tg_cache = os.path.join(args.checkpoint_root, 'tgif_mask_cache')
                    tg_reals_sub = _subsample_items(tg_reals, int(args.tgif_n_real), seed='tgif_real')
                    tg_reals_sub = _prep_tgif_items(
                        tg_reals_sub, mask_cache_dir=_tg_cache, log_tag='[eval]', tag='tgif reals')
                    for (t_, mf) in sorted(cells):
                        c_fakes = _subsample_items(cells[(t_, mf)], per_cell, seed=f'tgif_fake|{t_}|{mf}')
                        c_fakes = _prep_tgif_items(
                            c_fakes, mask_cache_dir=_tg_cache,
                            log_tag='[eval]', tag=f'tgif fakes {t_}/{mf}')
                        c_items = c_fakes + tg_reals_sub
                        c_ds = LabDataset(
                            c_items, cfg.resolution, cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
                            augment=False, use_degradation=False, use_invariance=False,
                            use_splice_degradation=False, gt_patch_threshold=float(args.gt_patch_threshold),
                        )
                        c_loader = build_eval_loader(c_ds, LoaderConfig(
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=(device.type == 'cuda'),
                        ))
                        # Only provide fakes for zoom eval
                        eval_sources.append((c_loader, c_fakes, f'tgif_val/{t_}/{mf}'))

            imd_opt_thresh = None

            # 1. Detection (BCE head)
            if bce_adapter is not None:
                for loader, _, tag in eval_sources:
                    if loader is None: continue
                    metrics = _run_image_bce_eval(bce_adapter, loader, device, log_tag='[eval]', tag=tag)
                    if tag == 'imd_val':
                        imd_opt_thresh = metrics.get('opt_thresh')

            # 2. Localization (Contrastive head, lean kmeans)
            if args.contrastive_dim > 0:
                for loader, _, tag in eval_sources:
                    if loader is None: continue
                    _run_localization_eval(eval_model, loader, device, cfg=cfg, log_tag='[eval]', tag=tag)

            # 3. Dense Localization (Patch BCE head)
            if patch_active:
                for loader, _, tag in eval_sources:
                    if loader is None: continue
                    _run_patch_bce_loc_eval(eval_model, loader, device, res=cfg.resolution, log_tag='[eval]', tag=tag)

            # 4. Zoom Eval (Contrastive head) - Condensed report, skip oracle, skip confidence gating
            if args.val_zoom and args.contrastive_dim > 0:
                for _, zoom_items, tag in eval_sources:
                    if not zoom_items: continue
                    zsamples = collect_zoom_eval_samples(
                        eval_model, zoom_items, device, res=cfg.resolution,
                        cov_range=tuple(args.val_zoom_cov), seed=f'zoomval|{tag}',
                        normalize_mean=cfg.IMAGENET_MEAN, normalize_std=cfg.IMAGENET_STD,
                        skip_oracle=True, log_tag='[eval]', tag=tag,
                    )
                    report_zoom_eval(
                        zsamples, condensed=True, log_tag='[eval]', tag=tag,
                    )

            # ── robustness sweep (BCE head) ──────────────────────────────────
            do_robust = (args.pool_hidden > 0 and args.robust_every > 0
                         and (epoch + 1) % args.robust_every == 0)
            if do_robust and bce_adapter is not None:
                t_rob = time.time()
                log_line(f'[robust] starting robustness sweep epoch={epoch} '
                         f'cap_per_source={args.eval_max_items} '
                         f'conditions={args.robust_conditions}')
                aug_conditions = [
                    (name, eval_aug_settings(name, cfg))
                    for name in args.robust_conditions
                ]
                # The augmentation sweep covers ALL datasets — the in-domain
                # sources plus the TGIF OOD hold (built on the fly since it
                # lives outside val_items).
                robust_sources = [
                    (imd_val_items,   'imd_val'),
                    (casia_val_items, 'casia_val'),
                    (coco_val_items,  'coco_inpaint_val'),
                    (sagid_val_items, 'sagid_val'),
                    (bfree_val_items, 'bfree_val'),
                ]
                if args.tgif_root:
                    tg_fakes, tg_reals = build_tgif2_items(
                        args.tgif_root, args.tgif_index, include_reals=True,
                        coco_ids=tgif_eval_ids,
                        types=({'fr'} if tgif_eval_ids is not None else None))
                    tg_fakes = _subsample_items(tg_fakes, int(args.tgif_n_fake), seed='tgif_fake')
                    tg_reals = _subsample_items(tg_reals, int(args.tgif_n_real), seed='tgif_real')
                    _tg_cache = os.path.join(args.checkpoint_root, 'tgif_mask_cache')
                    tg_fakes = _prep_tgif_items(
                        tg_fakes, mask_cache_dir=_tg_cache, log_tag='[robust]', tag='tgif fakes')
                    tg_reals = _prep_tgif_items(
                        tg_reals, mask_cache_dir=_tg_cache, log_tag='[robust]', tag='tgif reals')
                    if tg_fakes and tg_reals:
                        robust_sources.append((tg_fakes + tg_reals, 'tgif2'))
                for sub_items, sub_tag in robust_sources:
                    if not sub_items:
                        continue
                    cap_items = _subsample_items(sub_items, args.eval_max_items, seed='robust')
                    log_line(f'[robust] {sub_tag} cap n={len(cap_items)}/{len(sub_items)}')
                    eval_fn = _make_bce_eval_callable(
                        bce_adapter, cap_items, cfg, device,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        gt_patch_threshold=float(args.gt_patch_threshold),
                    )
                    run_robustness_sweep(
                        eval_fn, aug_conditions,
                        metrics_to_show=('auc', 'bal_acc', 'f1', 'tpr', 'tnr',
                                         'tpr_at_tnr_95', 'tpr_at_tnr_99'),
                        baseline_name='none',
                        log_tag='[robust]', tag=sub_tag,
                    )
                log_line(f'[robust] robustness sweep done '
                         f'elapsed={time.time() - t_rob:.0f}s')

            # 5. Epoch Viz
            if args.viz_every > 0 and (epoch + 1) % args.viz_every == 0 and viz_items:
                viz_dir = os.path.join(args.checkpoint_root, 'viz', f'epoch_{epoch:03d}')
                _run_epoch_viz(eval_model, viz_items, epoch, viz_dir, device, cfg)

            # ── checkpoint ────────────────────────────────────────────────────
            ckpt_path = os.path.join(args.checkpoint_root, f'epoch_{epoch:03d}.pt')
            ckpt_save({'model': unwrap_model(model).state_dict(), 'epoch': epoch + 1,
                       'optimizer': optimizer.state_dict()}, ckpt_path)

        # Keep ranks in lockstep: workers wait here while rank 0 evals + saves.
        barrier(ctx)

    log_line('[train] training complete')
    ddp_cleanup()   # explicit, deterministic teardown (atexit is the crash net)


if __name__ == '__main__':
    main()
