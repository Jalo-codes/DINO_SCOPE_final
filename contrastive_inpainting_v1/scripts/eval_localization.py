"""eval_localization.py — canonical localization eval on an already-trained
checkpoint, with NO training.

Reuses the *exact* functions the trainer runs at epoch end
(``_run_image_bce_eval`` + ``_run_localization_eval``), so the numbers match
training 1:1. For each split it reports:

    detection (BCE head)  : auroc, balanced-acc, tpr/tnr, per-area-tier tpr
    localization (full)   : patch + pixel F1/IoU per area_tier, no_gate +
                            BCE-logit threshold sweep + opt(calibrated) gate
    localization (swin)   : same, via source-resolution sliding windows
    + the free re-slices   : loc-by-confidence, oracle-vs-deployed polarity tax

The calibrated gate (``opt_thresh``) is taken from the imd_val BCE eval and
reused for CASIA — honest about deployment (no peeking at CASIA labels).

This is the contrastive head's deployed localization (k-means partition +
attention polarity). For a patch-BCE checkpoint, use swin_outlier_decode.py
instead (its 'patchbce' strategy decodes the dense head).

Usage:
    python -m contrastive_inpainting_v1.scripts.eval_localization \\
        --ckpt /media/ssd/runs/casia_mh_symmetric_v2_nocrop/epoch_004.pt \\
        --imd2020_root /media/ssd/DINO_SCOPE_DATA/IMD2020 \\
        --casia_root   /media/ssd/DINO_SCOPE_DATA/casia \\
        --casia_train --imd_val_only \\
        --eval_max_items 300 --swin --swin_scale 0.7 \\
        --output_log contrastive_inpainting_v1/logs/eval_localization.log
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch

from lab_utils.data.dataset import LabDataset
from lab_utils.data.loaders import LoaderConfig, build_eval_loader
from lab_utils.data.sampling import deterministic_subsample
from lab_utils.eval.image_bce import BCEHeadAdapter, run_image_bce_eval
from lab_utils.logging.text import install_log, log_line
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.train.checkpoint import load as ckpt_load

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec
# Reuse the trainer's eval functions verbatim so results match epoch-end eval.
from contrastive_inpainting_v1.scripts.train_multi_head import (
    _run_localization_eval,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Localization eval on a trained checkpoint (no training).')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--imd2020_root', default=None)
    p.add_argument('--casia_root', default=None)
    p.add_argument('--indoor_root', default=None)
    p.add_argument('--casia_train', action='store_true', default=False)
    p.add_argument('--imd_val_only', action='store_true', default=False)
    p.add_argument('--eval_max_items', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--gt_patch_threshold', type=float, default=0.06)
    # swin (deployed sliding window). ON by default — it's a localization test.
    p.add_argument('--swin', dest='swin', action='store_true', default=True)
    p.add_argument('--no_swin', dest='swin', action='store_false')
    p.add_argument('--swin_scale', type=float, default=0.7)
    p.add_argument('--swin_stride_frac', type=float, default=1.0)
    p.add_argument('--swin_inner_batch', type=int, default=8)
    p.add_argument('--loc_threshold_grid', type=float, nargs='+',
                   default=[-2.0, -1.0, 0.0, 1.0, 2.0])
    p.add_argument('--device', type=str, default='cuda')
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

    log_line(f'[eval] localization ckpt={args.ckpt} swin={args.swin} '
             f'swin_scale={args.swin_scale} eval_max_items={args.eval_max_items}')

    # ── items + loaders (mirrors the trainer's full_eval slice exactly) ──────
    spec = IMD2020BCESpec(
        imd2020_root=args.imd2020_root, casia_root=args.casia_root,
        indoor_root=args.indoor_root,
        imd_train=not args.imd_val_only, casia_train=args.casia_train)
    _, val_items = spec.build_items(cfg)
    imd_val_items   = [it for it in val_items if it.get('source', '') == 'imd2020']
    casia_val_items = [it for it in val_items if it.get('source', '') == 'casia']
    imd_eval_items   = deterministic_subsample(imd_val_items,   args.eval_max_items, seed='full_eval')
    casia_eval_items = deterministic_subsample(casia_val_items, args.eval_max_items, seed='full_eval')
    log_line(f'[eval] items: imd={len(imd_eval_items)}/{len(imd_val_items)} '
             f'casia={len(casia_eval_items)}/{len(casia_val_items)}')

    def _loader(items):
        if not items:
            return None
        ds = LabDataset(
            items, cfg.resolution, cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
            augment=False, use_degradation=False, use_invariance=False,
            use_splice_degradation=False, gt_patch_threshold=float(args.gt_patch_threshold))
        return build_eval_loader(ds, LoaderConfig(
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda')))

    imd_loader   = _loader(imd_eval_items)
    casia_loader = _loader(casia_eval_items)

    # ── model ────────────────────────────────────────────────────────────────
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
    log_line(f'[ckpt] loaded epoch={ckpt.get("epoch","?")} c_dim={c_dim} '
             f'pool_hidden={p_hidden} patch_bce={has_patch}')

    has_bce = p_hidden > 0
    bce_adapter = BCEHeadAdapter(model) if has_bce else None

    # ── detection (BCE head) → calibrated gate from imd_val ──────────────────
    imd_opt_thresh = None
    if bce_adapter is not None:
        if imd_loader is not None:
            imd_metrics = run_image_bce_eval(bce_adapter, imd_loader, device,
                                             log_tag='[eval]', tag='imd_val')
            imd_opt_thresh = imd_metrics.get('opt_thresh')
        if casia_loader is not None:
            run_image_bce_eval(bce_adapter, casia_loader, device,
                               log_tag='[eval]', tag='casia_val')

    # ── localization (contrastive head) ──────────────────────────────────────
    if c_dim <= 0:
        log_line('[eval] model has no contrastive head; localization via this '
                 'path needs it. For a patch-BCE checkpoint use '
                 'swin_outlier_decode.py (strategy=patchbce).')
        log_line('[eval] localization DONE')
        return

    do_swin = bool(args.swin) and has_bce   # swin needs the BCE gate
    for loader, tag in ((imd_loader, 'imd_val'), (casia_loader, 'casia_val')):
        if loader is None:
            continue
        _run_localization_eval(
            model, loader, device,
            cfg=cfg,
            run_swin=do_swin,
            threshold_grid=tuple(args.loc_threshold_grid),
            opt_thresh=imd_opt_thresh,
            swin_scales=(float(args.swin_scale),),
            swin_stride_frac=float(args.swin_stride_frac),
            swin_inner_batch=int(args.swin_inner_batch),
            swin_use_source_resolution=True,
            log_tag='[eval]', tag=tag,
        )
    log_line('[eval] localization DONE')


if __name__ == '__main__':
    main()
