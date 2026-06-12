"""contrastive_inpainting_v1.scripts.eval_swin_bce — calibrated sliding-window eval
for an image-level BCE checkpoint.

The headline question for the train-on-CASIA / val-on-IMD splice detector:
**does the sliding window recover small-splice recall WITHOUT inflating false
positives on real images?** Raw max-over-windows is a multiple-comparison FP
trap, so this script never reads each split's own optimal threshold. Instead it:

  1. Runs square, source-resolution sliding-window inference on CASIA-val (the
     in-domain headline split) and IMD-val (the OOD generalization split).
  2. CALIBRATES one decision threshold on CASIA-val *reals* at a fixed TNR
     (0.95 / 0.99) — the honest deployment operating point.
  3. APPLIES that single threshold to both splits and reports, per splice-size
     tier (tiny/small/medium/large), full-image vs swin(max) vs swin(top2) TPR
     plus the real-image TNR actually achieved.

Pass the SAME split flags you trained with so eval never touches train images:
for the flip that means ``--casia_train --imd_val_only`` (CASIA-val held out,
all IMD held out).

Usage:
    python -m contrastive_inpainting_v1.scripts.eval_swin_bce \\
        --ckpt /media/ssd/runs/casia_bce_swin_v1/epoch_009.pt \\
        --imd2020_root /media/ssd/DINO_SCOPE_DATA/IMD2020 \\
        --casia_root   /media/ssd/DINO_SCOPE_DATA/casia \\
        --casia_train --imd_val_only
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
import torch.nn as nn

from lab_utils.logging.text import install_log, log_line
from lab_utils.data.dataset import LabDataset
from lab_utils.data.loaders import LoaderConfig, build_eval_loader
from lab_utils.data.sampling import deterministic_subsample
from lab_utils.train.checkpoint import load as ckpt_load
from lab_utils.model.image_bce_detector import build_image_bce_detector
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.eval.sliding_window import (
    run_sliding_window_eval,
    format_sliding_window_report,
    calibrate_threshold_at_tnr,
    metrics_at_threshold,
)

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec


# Aggregators reported, in order. Keys must exist on every swin record.
_AGGREGATORS = (('full_logit', 'FULL'), ('max_logit', 'SWIN'), ('top2_logit', 'SWIN2'))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Calibrated sliding-window BCE eval')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--imd2020_root', type=str, default=None)
    p.add_argument('--casia_root',   type=str, default=None)
    p.add_argument('--indoor_root',  type=str, default=None)
    # Mirror the trainer split flags so eval reconstructs the SAME val split.
    p.add_argument('--casia_train',  action='store_true', default=False,
                   help='CASIA was in train (only the CASIA val split is held out '
                        'for eval). Pass this for the flipped run.')
    p.add_argument('--imd_val_only', action='store_true', default=False,
                   help='IMD held out entirely as OOD val. Pass this for the flip.')
    # Sliding window
    p.add_argument('--swin_scales',  type=float, nargs='+', default=[1.0, 0.6, 0.4],
                   help='Window scales (fraction of short edge). 1.0 = full image.')
    p.add_argument('--swin_square',  action='store_true', default=True)
    p.add_argument('--no_swin_square', dest='swin_square', action='store_false')
    p.add_argument('--swin_stride_frac', type=float, default=0.5)
    p.add_argument('--swin_inner_batch', type=int, default=8)
    p.add_argument('--tnr_targets',  type=float, nargs='+', default=[0.95, 0.99],
                   help='Calibration TNR operating points (on CASIA-val reals).')
    # Runtime
    p.add_argument('--eval_max_items', type=int, default=400,
                   help='Cap per-source val items (deterministic subsample).')
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--device',      type=str, default='cuda')
    p.add_argument('--pool_hidden', type=int, default=256)
    p.add_argument('--output_log',  type=str, default=None)
    return p


def _mk_loader(items, cfg, *, batch_size, num_workers, pin):
    if not items:
        return None
    ds = LabDataset(
        items, cfg.resolution,
        cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
        augment=False,
        use_degradation=False, use_invariance=False, use_splice_degradation=False,
    )
    return build_eval_loader(ds, LoaderConfig(
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin,
    ))


def _log_score_dist(records, score_key, *, split_tag):
    """Median+SD of the aggregator, reals pooled separately from splices."""
    scores = np.array([r[score_key] for r in records], dtype=np.float64)
    is_real = np.array([bool(r['is_real']) for r in records])
    rl, sp = scores[is_real], scores[~is_real]
    def _ms(a):
        return (float(np.median(a)), float(np.std(a))) if a.size else (float('nan'), float('nan'))
    r_med, r_sd = _ms(rl)
    s_med, s_sd = _ms(sp)
    log_line(
        f'[swin-cal] split={split_tag} {score_key} '
        f'reals med={r_med:+.3f} sd={r_sd:.3f} (n={int(rl.size)})  |  '
        f'splices med={s_med:+.3f} sd={s_sd:.3f} (n={int(sp.size)})'
    )


def main():
    args = _build_parser().parse_args()
    from contrastive_inpainting_v1.pipeline.cli import apply_path_defaults
    apply_path_defaults(args)

    if args.output_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_log)), exist_ok=True)
        install_log(args.output_log)
    else:
        from lab_utils.logging.run_dir import build_run_dir as _build_run_dir
        _ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        _rd = _build_run_dir(os.path.dirname(_ckpt_dir), os.path.basename(_ckpt_dir),
                             role='eval-swin-bce')
        install_log(str(_rd.log_path))

    device = torch.device(
        args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    )
    cfg = Config()
    log_line(f'[eval] ckpt={args.ckpt} device={device}')
    log_line(f'[cfg] swin_scales={list(args.swin_scales)} square={args.swin_square} '
             f'stride_frac={args.swin_stride_frac} tnr_targets={list(args.tnr_targets)}')

    # ── data: reconstruct the val split (CASIA-val + IMD) ─────────────────────
    spec = IMD2020BCESpec(
        imd2020_root=args.imd2020_root,
        casia_root=args.casia_root,
        indoor_root=args.indoor_root,
        imd_train=not args.imd_val_only,
        casia_train=args.casia_train,
    )
    _, val_items = spec.build_items(cfg)
    imd_items   = [it for it in val_items if it.get('source', '') == 'imd2020']
    casia_items = [it for it in val_items if it.get('source', '') == 'casia']
    imd_items   = deterministic_subsample(imd_items,   args.eval_max_items, seed='swin_eval')
    casia_items = deterministic_subsample(casia_items, args.eval_max_items, seed='swin_eval')
    log_line(f'[cfg] eval items: casia_val={len(casia_items)} imd_val={len(imd_items)}')
    if not casia_items:
        log_line('[eval] ERROR: no CASIA-val items — cannot calibrate. '
                 'Pass --casia_root (and --casia_train for the flipped run).')
        return

    # ── model (auto-detect ImageBCE vs MultiHead checkpoint) ──────────────────
    ckpt = ckpt_load(args.ckpt, map_location=str(device))
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    if 'contrastive_proj.weight' in sd:
        # MultiHeadDetector checkpoint — build it and expose ONLY the image-BCE
        # logit so the swin detection path (model(img) -> (B,)) is unchanged.
        c_dim = int(sd['contrastive_proj.weight'].shape[0])
        p_hidden = int(sd['pool.V.weight'].shape[0]) if 'pool.V.weight' in sd else args.pool_hidden
        mh = build_multi_head_detector(
            model_name=cfg.MODEL_NAME, resolution=cfg.resolution,
            lora_rank=cfg.LORA_RANK, lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT, lora_targets=cfg.LORA_TARGETS,
            contrastive_dim=c_dim, pool_hidden=p_hidden, device=device,
        )
        mh.load_state_dict(sd)

        class _ImageLogitAdapter(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x)['image_logit']

        model = _ImageLogitAdapter(mh).to(device)
        log_line(f'[ckpt] MultiHeadDetector head (contrastive_dim={c_dim} '
                 f'pool_hidden={p_hidden}) — swin detection on its image-BCE logit')
    else:
        model = build_image_bce_detector(
            model_name=cfg.MODEL_NAME, resolution=cfg.resolution,
            lora_rank=cfg.LORA_RANK, lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT, lora_targets=cfg.LORA_TARGETS,
            pool_hidden=args.pool_hidden, device=device,
        )
        model.load_state_dict(sd)
    model.eval()
    log_line(f'[ckpt] loaded epoch={ckpt.get("epoch", "?")}')

    # ── sliding-window inference per split ────────────────────────────────────
    def _records(items, tag):
        loader = _mk_loader(items, cfg, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin=(device.type == 'cuda'))
        if loader is None:
            return []
        recs = run_sliding_window_eval(
            model, loader, device,
            scales=tuple(args.swin_scales),
            stride_frac=float(args.swin_stride_frac),
            inner_batch_size=int(args.swin_inner_batch),
            square=bool(args.swin_square),
            use_source_resolution=True,
            log_tag='[swin]', tag=tag,
        )
        # The raw full-vs-swin diagnostic report (per-method opt threshold).
        format_sliding_window_report(recs, log_tag='[swin]', tag=tag)
        return recs

    casia_recs = _records(casia_items, 'casia_val')
    imd_recs   = _records(imd_items,   'imd_val')
    split_recs = [('casia_val', casia_recs)] + ([('imd_val', imd_recs)] if imd_recs else [])

    # ── score distributions (median+SD, reals pooled separately) ──────────────
    for score_key, _name in _AGGREGATORS:
        for split_tag, recs in split_recs:
            _log_score_dist(recs, score_key, split_tag=split_tag)

    # ── CALIBRATE on CASIA-val reals, APPLY to both splits ────────────────────
    log_line('[swin-cal] ==== calibrated operating points '
             '(threshold fit on CASIA-val reals, applied to all splits) ====')
    for tnr_target in args.tnr_targets:
        for score_key, name in _AGGREGATORS:
            thr = calibrate_threshold_at_tnr(casia_recs, score_key, tnr_target)
            log_line(f'[swin-cal] --- tnr_target={tnr_target:.2f} agg={name} '
                     f'thr={thr:+.3f} (calib=casia_val reals) ---')
            for split_tag, recs in split_recs:
                m = metrics_at_threshold(recs, score_key, thr)
                tiers = ' '.join(
                    f'{t}={m["tiers"][t]["tpr"]:.3f}(n{m["tiers"][t]["n"]})'
                    for t in ('tiny', 'small', 'medium', 'large')
                )
                log_line(
                    f'[swin-cal]   split={split_tag} {name} '
                    f'tnr={m["tnr"]:.3f} tpr={m["tpr"]:.3f}  tiers: {tiers}'
                )


if __name__ == '__main__':
    main()
