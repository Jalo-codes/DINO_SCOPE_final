"""eval_patch_localization.py — localization eval for the supervised
patch-BCE ("BCE localization") model, no training.

Sibling of eval_localization.py (which serves the contrastive model). The
patch-BCE checkpoint localizes via its dense per-patch head, decoded as
sigmoid(patch_logit) >= 0.5 — there is no k-means/partition step. This script
reports, per area_tier, the deployed decode and its best-threshold ceiling at
three views, all on the source-resolution pixel ruler vs the WHOLE splice GT:

    FULL_FRAME   : decode the whole image (no windows)
    SWIN_IMAGE   : OR the per-window decode over BCE-gated windows (deployed swin)
    BEST_CAP_WIN : the single best-capture window (oracle selection) — a ceiling

    DEPLOY = patchbce         (sigmoid >= 0.5)
    CEIL   = patchbce_oracle  (best logit threshold vs GT)

It reuses swin_outlier_decode._process (same windows, projection, ruler) and the
trainer's image-BCE eval (for AUROC + the calibrated swin gate), so numbers line
up cell-for-cell with the contrastive model's swin_outlier_decode run.

Usage:
    python -m contrastive_inpainting_v1.scripts.eval_patch_localization \\
        --ckpt /media/ssd/runs/casia_patchbce_v1/epoch_XXX.pt \\
        --imd2020_root /media/ssd/DINO_SCOPE_DATA/IMD2020 \\
        --casia_root   /media/ssd/DINO_SCOPE_DATA/casia \\
        --casia_train --imd_val_only \\
        --scales 0.7 --stride 0.5 --localized --eval_max_items 300 \\
        --output_log contrastive_inpainting_v1/logs/eval_patch_localization.log
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
from torchvision import transforms
from PIL import Image

from lab_utils.data.dataset import LabDataset
from lab_utils.data.loaders import LoaderConfig, build_eval_loader
from lab_utils.data.sampling import deterministic_subsample
from lab_utils.eval.image_bce import BCEHeadAdapter, run_image_bce_eval
from lab_utils.logging.text import install_log, log_line
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.train.checkpoint import load as ckpt_load

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec
# Reuse the tested swin machinery (same windows / projection / pixel ruler).
from contrastive_inpainting_v1.scripts.swin_outlier_decode import (
    _process, _load_gt, _bucket, _q, _BUCKETS, _SPLICE_KINDS,
)
_PATCH_STRATS = ['patchbce', 'patchbce_oracle']
_VIEWS = ('FULL_FRAME', 'SWIN_IMAGE', 'BEST_CAP_WIN')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Patch-BCE localization eval (no training).')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--imd2020_root', default=None)
    p.add_argument('--casia_root', default=None)
    p.add_argument('--indoor_root', default=None)
    p.add_argument('--casia_train', action='store_true', default=False)
    p.add_argument('--imd_val_only', action='store_true', default=False)
    p.add_argument('--scales', type=float, nargs='+', default=[0.7])
    p.add_argument('--stride', type=float, default=0.5)
    p.add_argument('--localized', action='store_true', default=True)
    p.add_argument('--no_localized', dest='localized', action='store_false')
    p.add_argument('--bce_gate_threshold', type=float, default=None,
                   help='Window gate on image-logit. Default: the imd_val '
                        'balanced-accuracy-optimal threshold (calibrated); '
                        'pass a value to override, or 0.0 for sigmoid>=0.5.')
    p.add_argument('--eval_max_items', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--gt_patch_threshold', type=float, default=0.06)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--inner_batch_size', type=int, default=8)
    p.add_argument('--output_log', type=str, default=None)
    return p


def main():
    args = _build_parser().parse_args()
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

    log_line(f'[loc] patch-localization ckpt={args.ckpt} scales={args.scales} '
             f'stride={args.stride} localized={args.localized}')

    spec = IMD2020BCESpec(
        imd2020_root=args.imd2020_root, casia_root=args.casia_root,
        indoor_root=args.indoor_root,
        imd_train=not args.imd_val_only, casia_train=args.casia_train)
    _, val_items = spec.build_items(cfg)

    # All-kind slices (reals + splices) for detection; splice-only for localization.
    det_split = {
        'imd_val':   [it for it in val_items if it.get('source') == 'imd2020'],
        'casia_val': [it for it in val_items if it.get('source') == 'casia'],
    }
    loc_split = {
        k: [it for it in v if it.get('kind') in _SPLICE_KINDS and it.get('mask')]
        for k, v in det_split.items()
    }
    for k in det_split:
        det_split[k] = deterministic_subsample(det_split[k], args.eval_max_items, seed='full_eval')
        loc_split[k] = deterministic_subsample(loc_split[k], args.eval_max_items, seed='outlier')
    log_line(f'[loc] det items: imd={len(det_split["imd_val"])} casia={len(det_split["casia_val"])} | '
             f'loc items: imd={len(loc_split["imd_val"])} casia={len(loc_split["casia_val"])}')

    # ── model ────────────────────────────────────────────────────────────────
    ckpt = ckpt_load(args.ckpt, map_location=str(device))
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    c_dim = int(sd['contrastive_proj.weight'].shape[0]) if 'contrastive_proj.weight' in sd else 0
    p_hidden = int(sd['pool.V.weight'].shape[0]) if 'pool.V.weight' in sd else 0
    has_patch = 'patch_head.weight' in sd
    if not has_patch:
        log_line('[loc] ERROR: checkpoint has no patch_head. This script evaluates the '
                 'patch-BCE model; for the contrastive model use eval_localization.py.')
        return
    model = build_multi_head_detector(
        model_name=cfg.MODEL_NAME, resolution=cfg.resolution,
        lora_rank=cfg.LORA_RANK, lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT, lora_targets=cfg.LORA_TARGETS,
        contrastive_dim=c_dim, pool_hidden=p_hidden, patch_bce=has_patch, device=device)
    model.load_state_dict(sd); model.eval()
    log_line(f'[ckpt] loaded epoch={ckpt.get("epoch","?")} c_dim={c_dim} '
             f'pool_hidden={p_hidden} patch_bce={has_patch}')

    has_bce = p_hidden > 0

    # ── detection (image-BCE) → calibrated swin gate from imd_val ────────────
    imd_opt_thresh = None
    if has_bce:
        adapter = BCEHeadAdapter(model)
        def _det_loader(items):
            if not items:
                return None
            ds = LabDataset(items, cfg.resolution, cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
                            augment=False, use_degradation=False, use_invariance=False,
                            use_splice_degradation=False,
                            gt_patch_threshold=float(args.gt_patch_threshold))
            return build_eval_loader(ds, LoaderConfig(
                batch_size=args.batch_size, num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda')))
        il = _det_loader(det_split['imd_val'])
        if il is not None:
            m = run_image_bce_eval(adapter, il, device, log_tag='[eval]', tag='imd_val')
            imd_opt_thresh = m.get('opt_thresh')
        cl = _det_loader(det_split['casia_val'])
        if cl is not None:
            run_image_bce_eval(adapter, cl, device, log_tag='[eval]', tag='casia_val')

    gate = (args.bce_gate_threshold if args.bce_gate_threshold is not None
            else (imd_opt_thresh if (imd_opt_thresh is not None and has_bce) else 0.0))
    log_line(f'[loc] swin window gate (image-logit >= {gate:+.3f})')

    # ── localization (patch head) ────────────────────────────────────────────
    for scale in args.scales:
        acc = {}
        for split, items in loc_split.items():
            if not items:
                continue
            acc[split] = {v: {b: {s: [] for s in _PATCH_STRATS} for b in _BUCKETS} for v in _VIEWS}
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
                               gate=gate, localized=args.localized,
                               n=n, T=T, normalize=normalize, device=device,
                               inner=args.inner_batch_size, kmeans_init=4,
                               strategies=_PATCH_STRATS)
                for v in _VIEWS:
                    if v not in res:
                        continue
                    for s in _PATCH_STRATS:
                        acc[split][v][bucket][s].append(res[v][s])
                if (i + 1) % 50 == 0:
                    log_line(f'[loc] scale={scale:.2f} {split} {i+1}/{len(items)}')

        for split in acc:
            for v in _VIEWS:
                for b in _BUCKETS:
                    cell = acc[split][v][b]
                    ncount = len(cell['patchbce'])
                    if ncount == 0:
                        continue
                    log_line(f'[loc] ===== scale={scale:.2f} {split} view={v} area_tier={b} n={ncount} =====')
                    for s in _PATCH_STRATS:
                        q1, md, q3 = _q([x[2] for x in cell[s]])
                        _, pm, _ = _q([x[0] for x in cell[s]])
                        _, rm, _ = _q([x[1] for x in cell[s]])
                        tag = 'CEIL ' if s.endswith('_oracle') else 'DEPLOY'
                        log_line(
                            f'[loc]   {tag} {s:<15} iou[q1/med/q3]='
                            f'{q1:.3f}/{md:.3f}/{q3:.3f}  prec_med={pm:.3f} rec_med={rm:.3f}')
    log_line('[loc] patch-localization DONE')


if __name__ == '__main__':
    main()
