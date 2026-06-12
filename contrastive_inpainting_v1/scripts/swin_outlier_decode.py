"""swin_outlier_decode.py — outlier decode vs k-means, the HONEST version.

Image-level (real swin, not per-window-local), all sizes, all scales, with
precision/recall and full quartile distributions. The question: can a SINGLE
blind outlier decode replace k-means across every splice size, since at
inference we don't know the size?

The outlier score for a window/frame:
    score[i] = 1 − cos(patch_i, background_prototype)
    background_prototype = mean of the detector's low-attention (≤median) patches
Higher score = more unlike the background = more splice-like.

Decodes compared (each turns the score into a patch mask):
    kmeans   : current k-means(2)+attention decode            [REFERENCE]
    oracle   : best threshold on the score vs GT              [CEILING, uses GT]
    otsu     : 1-D Otsu on the score histogram                [blind, always-splits]
    mad2.5   : score > median + 2.5·1.4826·MAD                [blind, conservative]
    mad3.0   : score > median + 3.0·1.4826·MAD                [blind, conservative]
    gap      : largest gap in the sorted upper-half scores    [blind, always-splits]

Views (ALL pixel-level vs the WHOLE splice GT — the honest, comparable ruler):
    FULL_FRAME : decode on the whole image, no windows.
    SWIN_IMAGE : OR the per-window decode over BCE-gated windows = the real
                 deployed sliding window.
    BEST_CAP_WIN : the single best-capture window (oracle selection) — a ceiling
                 for "if we picked the right window."

Per (scale, split, view, area_tier, strategy) it reports IoU quartiles (q1/med/q3)
plus median precision and recall.

Usage:
    python -m contrastive_inpainting_v1.scripts.swin_outlier_decode \\
        --ckpt /media/ssd/runs/casia_mh_symmetric_v2_nocrop/epoch_004.pt \\
        --imd2020_root /media/ssd/DINO_SCOPE_DATA/IMD2020 \\
        --casia_root   /media/ssd/DINO_SCOPE_DATA/casia \\
        --casia_train --imd_val_only \\
        --scales 0.35 0.5 0.7 --stride 0.5 --localized \\
        --eval_max_items 300
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from lab_utils.data.sampling import deterministic_subsample
from lab_utils.eval.partition import spherical_kmeans2
from lab_utils.eval.window_geometry import window_grid
from lab_utils.logging.text import install_log, log_line
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.train.checkpoint import load as ckpt_load

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec
from contrastive_inpainting_v1.diagnose import polarity as _polarity
from contrastive_inpainting_v1.diagnose import project as _project


_SPLICE_KINDS = frozenset({'imd_splice', 'casia_splice'})
_BUCKETS = ('tiny', 'small', 'medium', 'large')


def _bucket(a: float) -> str:
    if a <= 0.05:
        return 'tiny'
    if a < 0.15:
        return 'small'
    if a < 0.30:
        return 'medium'
    return 'large'


def _load_gt(mask_path: str, H: int, W: int) -> Optional[np.ndarray]:
    try:
        img = Image.open(mask_path).convert('L')
        if img.size != (W, H):
            img = img.resize((W, H), Image.NEAREST)
        return (np.asarray(img, dtype=np.uint8) > 0)
    except Exception:
        return None


def _gt_patches(gt_HW, top, left, side, n, T, thr=0.06):
    H, W = gt_HW.shape
    r0 = max(0, top); r1 = min(H, top + side)
    c0 = max(0, left); c1 = min(W, left + side)
    crop = gt_HW[r0:r1, c0:c1]
    if crop.size == 0:
        return np.zeros((n, n), dtype=bool)
    img = Image.fromarray(crop.astype(np.uint8) * 255, mode='L')
    img = TF.resize(img, [T, T], interpolation=Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    p = T // n
    return (arr.reshape(n, p, n, p).mean(axis=(1, 3)) > thr)


def _localized(windows, gt_HW, crop_side):
    if not gt_HW.any():
        return windows
    H, W = gt_HW.shape
    rows = np.where(gt_HW.any(axis=1))[0]; cols = np.where(gt_HW.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    pad = int(crop_side)
    br0, br1 = max(0, r0 - pad), min(H, r1 + pad)
    bc0, bc1 = max(0, c0 - pad), min(W, c1 + pad)
    local = [(t, l, s, s) for (t, l, s, _) in windows
             if t + s > br0 and t < br1 and l + s > bc0 and l < bc1]
    return local or windows[:1]


def _capture(gt_HW, t, l, s):
    sp = float(gt_HW.sum())
    if sp == 0:
        return 0.0
    H, W = gt_HW.shape
    return float(gt_HW[max(0, t):min(H, t + s), max(0, l):min(W, l + s)].sum()) / sp


# ── outlier score + threshold strategies ────────────────────────────────────

def _outlier_score(z, att):
    zz = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    if att is not None and len(np.asarray(att).reshape(-1)) == len(zz):
        a = np.asarray(att).reshape(-1)
        bg = zz[a <= np.median(a)]
        if len(bg) == 0:
            bg = zz
    else:
        bg = zz
    proto = bg.mean(0); proto = proto / (np.linalg.norm(proto) + 1e-8)
    return 1.0 - (zz @ proto)


def _otsu_thr(s, bins=64):
    s = np.asarray(s, dtype=np.float64); lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return hi + 1.0
    hist, edges = np.histogram(s, bins=bins, range=(lo, hi))
    p = hist.astype(np.float64) / (hist.sum() + 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p); mu = np.cumsum(p * centers); mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / (denom + 1e-12), 0.0)
    return float(centers[int(np.argmax(sigma_b))])


def _mad_thr(s, k):
    med = float(np.median(s)); mad = float(np.median(np.abs(s - med))) * 1.4826
    return med + k * max(mad, 1e-6)


def _gap_thr(s):
    s = np.sort(np.asarray(s, dtype=np.float64)); med = np.median(s)
    upper = s[s >= med]
    if len(upper) < 3:
        return float(s.max()) + 1.0
    diffs = np.diff(upper); gi = int(np.argmax(diffs))
    return float(0.5 * (upper[gi] + upper[gi + 1]))


def _oracle_thr(score, gt_flat, grid=40):
    """Best threshold vs GT. If the window has no splice GT, return a threshold
    that predicts NOTHING (the true oracle: no splice → no flag)."""
    gt = np.asarray(gt_flat).reshape(-1).astype(bool)
    lo, hi = float(score.min()), float(score.max())
    if int(gt.sum()) == 0 or hi <= lo:
        return hi + 1.0
    best_thr, best_iou = hi + 1.0, -1.0
    for thr in np.linspace(lo, hi, grid):
        pred = score >= thr
        inter = int((pred & gt).sum()); union = int((pred | gt).sum())
        iou = (inter / union) if union else 0.0
        if iou > best_iou:
            best_iou, best_thr = iou, float(thr)
    return best_thr


_BLIND = {
    'otsu':   _otsu_thr,
    'mad2.5': lambda s: _mad_thr(s, 2.5),
    'mad3.0': lambda s: _mad_thr(s, 3.0),
    'gap':    _gap_thr,
}
# Contrastive-embedding decodes (need the 'contrastive' head).
#   graph : calibrated connected-components decode (committed foreground) — the
#           comparison target for k-means; bleed-resistant, abstains naturally.
_CONTRASTIVE_STRATS = ['kmeans', 'graph', 'oracle'] + list(_BLIND.keys())
# Supervised patch-BCE decodes (need the 'patch_logit' head).
#   patchbce        : sigmoid(logit) >= 0.5  (== logit >= 0)   [deployed]
#   patchbce_oracle : best logit threshold vs GT               [CEILING]
_PATCH_STRATS = ['patchbce', 'patchbce_oracle']


def _model_strategies(model) -> List[str]:
    """Which decode strategies a checkpoint supports, from its heads."""
    strats: List[str] = []
    if getattr(model, 'contrastive_proj', None) is not None:
        strats += _CONTRASTIVE_STRATS
    if getattr(model, 'patch_head', None) is not None:
        strats += _PATCH_STRATS
    return strats


def _decode_masks(z, att, pl, gt_win_flat, n, strategies, kmeans_init=4,
                  graph_spec=None):
    """strategy → (n,n) bool mask. gt_win_flat used only by the oracle decodes.

    Contrastive decodes read ``z``/``att``; patch decodes read ``pl`` (per-patch
    logits). Only the requested ``strategies`` are computed.
    """
    out = {}
    gt = np.asarray(gt_win_flat).reshape(-1)
    want_contrastive = any(s in strategies for s in _CONTRASTIVE_STRATS)
    if want_contrastive and z is not None:
        if 'kmeans' in strategies:
            raw, _ = spherical_kmeans2(z, n_init=kmeans_init)
            out['kmeans'] = _polarity.polarity_attn(raw, att).reshape(n, n)
        if 'graph' in strategies and graph_spec is not None:
            from lab_utils.eval.partition import decode_deploy_mask
            fg, _ = decode_deploy_mask(z, graph_spec, attention=att, grid_hw=(n, n))
            out['graph'] = fg.astype(bool).reshape(n, n)
        score = _outlier_score(z, att)
        if 'oracle' in strategies:
            out['oracle'] = (score >= _oracle_thr(score, gt)).reshape(n, n)
        for name, fn in _BLIND.items():
            if name in strategies:
                out[name] = (score >= fn(score)).reshape(n, n)
    if pl is not None:
        plf = np.asarray(pl).reshape(-1)
        if 'patchbce' in strategies:
            out['patchbce'] = (plf >= 0.0).reshape(n, n)          # sigmoid >= 0.5
        if 'patchbce_oracle' in strategies:
            out['patchbce_oracle'] = (plf >= _oracle_thr(plf, gt)).reshape(n, n)
    return out


# ── metrics ──────────────────────────────────────────────────────────────

def _pr_iou(pred_HW, gt_HW):
    p = pred_HW.reshape(-1).astype(bool); g = gt_HW.reshape(-1).astype(bool)
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    union = tp + fp + fn
    iou = (tp / union) if union else 0.0
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else float('nan')   # nan = flagged nothing
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else float('nan')
    return float(prec), float(rec), float(iou)


@torch.no_grad()
def _forward(model, crops_t, device, inner=8):
    z_l, lg_l, at_l, pl_l = [], [], [], []
    for i in range(0, len(crops_t), inner):
        out = model(crops_t[i:i + inner].to(device, non_blocking=True))
        z_l.append(out['contrastive'].detach().cpu().float().numpy()
                   if out.get('contrastive') is not None else None)
        lg_l.append(out['image_logit'].detach().cpu().float().numpy()
                    if out.get('image_logit') is not None else None)
        at_l.append(out['pool_attention'].detach().cpu().float().numpy()
                    if out.get('pool_attention') is not None else None)
        pl_l.append(out['patch_logit'].detach().cpu().float().numpy()
                    if out.get('patch_logit') is not None else None)
    z  = (np.concatenate([x for x in z_l if x is not None], 0)
          if any(x is not None for x in z_l) else None)
    lg = (np.concatenate([x for x in lg_l if x is not None], 0).reshape(-1)
          if any(x is not None for x in lg_l) else None)
    at = (np.concatenate([x for x in at_l if x is not None], 0)
          if any(x is not None for x in at_l) else None)
    pl = (np.concatenate([x for x in pl_l if x is not None], 0)
          if any(x is not None for x in pl_l) else None)
    return z, lg, at, pl


# ── per-image processing for one scale ──────────────────────────────────────

def _process(model, source, gt_HW, *, scale, stride, gate, localized,
             n, T, normalize, device, inner, kmeans_init, strategies,
             graph_spec=None):
    W_src, H_src = source.size
    out: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    # FULL FRAME
    full_t = normalize(TF.to_tensor(
        TF.resize(source, [T, T], interpolation=Image.BILINEAR))).unsqueeze(0)
    zf, _lf, af, plf = _forward(model, full_t, device, inner)
    gt_full_nn = (np.asarray(TF.resize(
        Image.fromarray(gt_HW.astype(np.uint8) * 255), [T, T],
        interpolation=Image.NEAREST), dtype=np.float32) / 255.0
    ).reshape(n, T // n, n, T // n).mean(axis=(1, 3)) > 0.06
    fmasks = _decode_masks(
        zf[0] if zf is not None else None,
        af[0] if af is not None else None,
        plf[0] if plf is not None else None,
        gt_full_nn.reshape(-1), n, strategies, kmeans_init, graph_spec)
    out['FULL_FRAME'] = {
        s: _pr_iou(_project.patch_grid_to_pixel_mask(
            fmasks[s], bbox=(0, 0, H_src, W_src), full_size=(H_src, W_src)), gt_HW)
        for s in strategies}

    # WINDOWS
    crop_side = max(n, int(round(min(H_src, W_src) * scale)))
    crop_side = min(crop_side, min(H_src, W_src))
    windows = window_grid((W_src, H_src), scale=scale, stride_frac=stride,
                          n_patch_per_side=n)
    if localized:
        windows = _localized(windows, gt_HW, crop_side)
    crops = []
    for (t, l, s, _) in windows:
        c = TF.resize(source.crop((l, t, l + s, t + s)), [T, T], interpolation=Image.BILINEAR)
        crops.append(normalize(TF.to_tensor(c)).unsqueeze(0))
    zw, lw, aw, plw = _forward(model, torch.cat(crops, 0), device, inner)

    gated = [k for k in range(len(windows))
             if (lw is None or float(lw[k]) >= gate)]
    # per-window masks per strategy + best-capture tracking
    win_masks = {s: [] for s in strategies}     # list of (mask_nn, meta)
    best_cap, best_k, best_decode = -1.0, None, None
    for k in gated:
        t, l, s, _s = windows[k]
        gt_win = _gt_patches(gt_HW, t, l, s, n, T).reshape(-1)
        d = _decode_masks(
            zw[k] if zw is not None else None,
            aw[k] if aw is not None else None,
            plw[k] if plw is not None else None,
            gt_win, n, strategies, kmeans_init, graph_spec)
        for strat in strategies:
            win_masks[strat].append((d[strat], (t, l, s, s)))
        cap = _capture(gt_HW, t, l, s)
        if cap > best_cap:
            best_cap, best_k, best_decode = cap, k, d

    # SWIN_IMAGE: OR projected window masks per strategy
    out['SWIN_IMAGE'] = {}
    for strat in strategies:
        acc = np.zeros((H_src, W_src), dtype=bool)
        for m, meta in win_masks[strat]:
            acc |= _project.patch_grid_to_pixel_mask(
                m, bbox=meta, full_size=(H_src, W_src))
        out['SWIN_IMAGE'][strat] = _pr_iou(acc, gt_HW)

    # BEST_CAP_WIN
    if best_decode is not None:
        t, l, s, _s = windows[best_k]
        out['BEST_CAP_WIN'] = {
            strat: _pr_iou(_project.patch_grid_to_pixel_mask(
                best_decode[strat], bbox=(t, l, s, s), full_size=(H_src, W_src)), gt_HW)
            for strat in strategies}
    return out


def _q(vals):
    a = np.asarray([v for v in vals if v == v], dtype=np.float64)   # drop nan
    if a.size == 0:
        return float('nan'), float('nan'), float('nan')
    return (float(np.percentile(a, 25)), float(np.median(a)), float(np.percentile(a, 75)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--imd2020_root', default=None)
    p.add_argument('--casia_root', default=None)
    p.add_argument('--indoor_root', default=None)
    p.add_argument('--casia_train', action='store_true', default=False)
    p.add_argument('--imd_val_only', action='store_true', default=False)
    p.add_argument('--scales', type=float, nargs='+', default=[0.35, 0.5, 0.7])
    p.add_argument('--stride', type=float, default=0.5)
    p.add_argument('--localized', action='store_true', default=True)
    p.add_argument('--no_localized', dest='localized', action='store_false')
    p.add_argument('--bce_gate_threshold', type=float, default=0.0)
    p.add_argument('--eval_max_items', type=int, default=300)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--pool_hidden', type=int, default=256)
    p.add_argument('--kmeans_init', type=int, default=4)
    p.add_argument('--inner_batch_size', type=int, default=8)
    p.add_argument('--output_log', type=str, default=None)
    # Graph decode is always compared alongside k-means (strategy 'graph');
    # these tune it. tau_pos/tau_neg should match the trained margins.
    p.add_argument('--tau_pos', type=float, default=0.55,
                   help='graph decode: same-region cohesion floor (match trained value).')
    p.add_argument('--tau_neg', type=float, default=0.20,
                   help='graph decode: cross-region separation ceiling.')
    p.add_argument('--graph_s_edge', type=float, default=None,
                   help='graph decode: absolute edge similarity (default mid-band).')
    p.add_argument('--graph_knn', type=int, default=10,
                   help='graph decode: mutual-kNN k.')
    p.add_argument('--graph_spatial', type=int, default=None,
                   help='graph decode: Chebyshev grid radius for spatial-gated edges.')
    p.add_argument('--no_graph', action='store_true', default=False,
                   help='Disable the graph strategy (k-means + score decodes only).')
    args = p.parse_args()
    from contrastive_inpainting_v1.pipeline.cli import apply_path_defaults
    apply_path_defaults(args)

    if args.output_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_log)), exist_ok=True)
        install_log(args.output_log)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    cfg = Config()
    n = cfg.resolution.num_patches_per_side
    T = cfg.resolution.image_size
    normalize = transforms.Normalize(list(cfg.IMAGENET_MEAN), list(cfg.IMAGENET_STD))

    log_line(f'[loc] outlier-decode FULL ckpt={args.ckpt} scales={args.scales} '
             f'stride={args.stride} gate={args.bce_gate_threshold} localized={args.localized}')

    spec = IMD2020BCESpec(
        imd2020_root=args.imd2020_root, casia_root=args.casia_root,
        indoor_root=args.indoor_root,
        imd_train=not args.imd_val_only, casia_train=args.casia_train)
    _, val_items = spec.build_items(cfg)

    by_split = {
        'imd_val':   [it for it in val_items if it.get('source') == 'imd2020' and it.get('kind') in _SPLICE_KINDS and it.get('mask')],
        'casia_val': [it for it in val_items if it.get('source') == 'casia' and it.get('kind') in _SPLICE_KINDS and it.get('mask')],
    }
    for k in by_split:
        by_split[k] = deterministic_subsample(by_split[k], args.eval_max_items, seed='outlier')
    log_line(f'[loc] items: imd={len(by_split["imd_val"])} casia={len(by_split["casia_val"])}')

    ckpt = ckpt_load(args.ckpt, map_location=str(device))
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    c_dim = int(sd['contrastive_proj.weight'].shape[0]) if 'contrastive_proj.weight' in sd else 0
    p_hidden = int(sd['pool.V.weight'].shape[0]) if 'pool.V.weight' in sd else 0
    has_patch = 'patch_head.weight' in sd
    model = build_multi_head_detector(
        model_name=cfg.MODEL_NAME, resolution=cfg.resolution,
        lora_rank=cfg.LORA_RANK, lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT, lora_targets=cfg.LORA_TARGETS,
        contrastive_dim=c_dim, pool_hidden=p_hidden, patch_bce=has_patch, device=device)
    model.load_state_dict(sd); model.eval()
    strategies = _model_strategies(model)
    from lab_utils.eval.partition import DecodeSpec
    graph_spec = DecodeSpec(
        method='graph', tau_pos=float(args.tau_pos), tau_neg=float(args.tau_neg),
        s_edge=args.graph_s_edge, mutual_knn_k=int(args.graph_knn),
        r_spatial=args.graph_spatial)
    if args.no_graph and 'graph' in strategies:
        strategies = [s for s in strategies if s != 'graph']
    if not strategies:
        log_line('[loc] ERROR: checkpoint exposes no localization head; abort')
        return
    log_line(f'[ckpt] loaded epoch={ckpt.get("epoch","?")} c_dim={c_dim} '
             f'pool_hidden={p_hidden} patch_bce={has_patch} strategies={strategies}')

    views = ('FULL_FRAME', 'SWIN_IMAGE', 'BEST_CAP_WIN')
    for scale in args.scales:
        # acc[split][view][bucket][strategy] = list of (prec, rec, iou)
        acc: Dict = {}
        for split, items in by_split.items():
            if not items:
                continue
            acc[split] = {v: {b: {s: [] for s in strategies} for b in _BUCKETS} for v in views}
            for i, it in enumerate(items):
                try:
                    source = Image.open(str(it.get('img'))).convert('RGB')
                except Exception:
                    continue
                W_src, H_src = source.size
                gt_HW = _load_gt(str(it.get('mask')), H_src, W_src)
                if gt_HW is None or not gt_HW.any():
                    continue
                area_tier = _bucket(float(gt_HW.mean()))
                res = _process(model, source, gt_HW, scale=scale, stride=args.stride,
                               gate=args.bce_gate_threshold, localized=args.localized,
                               n=n, T=T, normalize=normalize, device=device,
                               inner=args.inner_batch_size, kmeans_init=args.kmeans_init,
                               strategies=strategies, graph_spec=graph_spec)
                for v in views:
                    if v not in res:
                        continue
                    for s in strategies:
                        acc[split][v][bucket][s].append(res[v][s])
                if (i + 1) % 50 == 0:
                    log_line(f'[loc] scale={scale:.2f} {split} {i+1}/{len(items)}')

        # report
        for split in acc:
            for v in views:
                for b in _BUCKETS:
                    cell = acc[split][v][b]
                    ncount = len(cell[strategies[0]])
                    if ncount == 0:
                        continue
                    log_line(f'[loc] ===== scale={scale:.2f} {split} view={v} area_tier={b} n={ncount} =====')
                    for s in strategies:
                        precs = [x[0] for x in cell[s]]
                        recs  = [x[1] for x in cell[s]]
                        ious  = [x[2] for x in cell[s]]
                        q1, md, q3 = _q(ious)
                        _, pm, _ = _q(precs)
                        _, rm, _ = _q(recs)
                        tag = ('REF ' if s == 'kmeans' else
                               'CEIL' if (s == 'oracle' or s.endswith('_oracle')) else
                               'SUP ' if s == 'patchbce' else 'blnd')
                        log_line(
                            f'[loc]   {tag} {s:<15} iou[q1/med/q3]='
                            f'{q1:.3f}/{md:.3f}/{q3:.3f}  prec_med={pm:.3f} rec_med={rm:.3f}')
    log_line('[loc] outlier-decode FULL DONE')


if __name__ == '__main__':
    main()
