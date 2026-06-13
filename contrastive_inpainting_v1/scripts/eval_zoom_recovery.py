"""contrastive_inpainting_v1.scripts.eval_zoom_recovery — does zoom rescue small
splices, and is the bottleneck PLACEMENT or CAPABILITY?

Standalone (checkpoint-level) version of the per-epoch ``--val_zoom`` eval. For
each val splice it scores, all oracle-polarity at PIXEL granularity and all
NON-DESTRUCTIVE (geometric crops only):

  - full        : whole frame (the deployment-natural baseline)
  - natural-zoom: a seeded RANDOM-position crop sized to a target coverage
                  (what a blind sliding window actually sees)
  - oracle-zoom : a mask-centered crop into the object (the targeting ceiling)

The split that matters for "will swin fix this":
  - natural ≈ oracle  ⇒ blind window placement is easy; swin fully recovers it.
  - natural ≪ oracle  ⇒ the model localizes well WHEN the window lands (oracle),
                        so the residual bottleneck is window placement/density,
                        not the model — a cheaper fix than retraining.

Also reports (optional) the detect-then-zoom coarse→fine refine, the
non-tile-OR alternative that does not degrade medium/large.

Usage:
    python -m contrastive_inpainting_v1.scripts.eval_zoom_recovery \\
        --ckpt /media/ssd/runs/casia_mh_symmetric_v1/checkpoints/epoch_011.pt \\
        --imd2020_root /media/ssd/DINO_SCOPE_DATA/IMD2020 \\
        --casia_root /media/ssd/DINO_SCOPE_DATA/casia \\
        --casia_train --imd_val_only --coarse2fine
"""

import argparse
import hashlib
import os
import sys
from typing import Dict, List, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch

from lab_utils.logging.text import install_log, log_line
from lab_utils.train.checkpoint import load as ckpt_load
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.data.dataset import LabDataset
from lab_utils.data.loaders import LoaderConfig, build_eval_loader
from lab_utils.eval.localization import (
    collect_localization_samples,
    report_localization_threshold_sweep,
    report_loc_by_confidence,
    report_oracle_tax,
    collect_zoom_eval_samples,
    report_zoom_eval,
    collect_coarse_to_fine_samples,
    report_coarse_to_fine,
)

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec


def _infer_head_dims(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    contrastive_dim = int(sd['contrastive_proj.weight'].shape[0]) if 'contrastive_proj.weight' in sd else 0
    pool_hidden     = int(sd['pool.V.weight'].shape[0]) if 'pool.V.weight' in sd else 0
    return contrastive_dim, pool_hidden


def _subsample(items: List[Dict], n: int, *, seed: str) -> List[Dict]:
    """Deterministic md5 subsample — stable across runs (matches the trainer)."""
    if not items or len(items) <= n:
        return list(items)
    def _key(it):
        path = it.get('img') or it.get('path') or ''
        return hashlib.md5(f'{seed}|{path}'.encode('utf-8')).hexdigest()
    return sorted(items, key=_key)[:n]


def _loader(items: List[Dict], cfg, args, device):
    """Full-frame eval loader (clean) over a list of items."""
    ds = LabDataset(
        items, cfg.resolution, cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
        augment=False, use_degradation=False, use_invariance=False,
        use_splice_degradation=False, gt_patch_threshold=float(args.gt_patch_threshold),
    )
    return build_eval_loader(ds, LoaderConfig(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    ))


def main():
    p = argparse.ArgumentParser(description='Zoom-recovery decomposition (placement vs capability)')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--imd2020_root', type=str, default=None)
    p.add_argument('--casia_root',   type=str, default=None)
    p.add_argument('--indoor_root',  type=str, default=None)
    p.add_argument('--casia_train',  action='store_true', default=False)
    p.add_argument('--imd_val_only', action='store_true', default=False)
    p.add_argument('--eval_max_items', type=int, default=300)
    p.add_argument('--cov_range', type=float, nargs=2, default=(0.05, 0.55),
                   metavar=('LO', 'HI'),
                   help='Per-item target in-frame coverage drawn uniformly here '
                        '(seeded) → even spread of splice sizes.')
    p.add_argument('--coarse2fine', action='store_true', default=False,
                   help='Also run the detect-then-zoom coarse→fine refine.')
    p.add_argument('--pad_frac', type=float, default=0.25)
    p.add_argument('--refine_max_frac', type=float, default=0.40)
    p.add_argument('--zoom_mode', type=str, choices=['single', 'multi'], default='single')
    # Tile-OR sliding-window localization (the actual "swin"). On by default.
    p.add_argument('--run_swin', dest='run_swin', action='store_true', default=True,
                   help='Run the tile-OR sliding-window localization (default on).')
    p.add_argument('--no_swin', dest='run_swin', action='store_false',
                   help='Disable the tile-OR swin pass (zoom decomposition only).')
    p.add_argument('--swin_scale', type=float, default=0.7,
                   help='Window side as a fraction of the source short edge.')
    p.add_argument('--swin_stride_frac', type=float, default=1.0)
    p.add_argument('--swin_inner_batch', type=int, default=8)
    p.add_argument('--gt_patch_threshold', type=float, default=0.06)
    p.add_argument('--loc_threshold_grid', type=float, nargs='+',
                   default=[-2.0, -1.0, 0.0, 1.0, 2.0])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--output_log', type=str, default=None)
    from lab_utils.eval.decode_cli import add_decode_args
    add_decode_args(p)
    args = p.parse_args()
    from contrastive_inpainting_v1.pipeline.cli import apply_path_defaults
    apply_path_defaults(args)

    if args.output_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_log)), exist_ok=True)
        install_log(args.output_log)
    else:
        from lab_utils.logging.run_dir import build_run_dir as _build_run_dir
        _ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        _rd = _build_run_dir(os.path.dirname(_ckpt_dir), os.path.basename(_ckpt_dir),
                             role='eval-zoom-recovery')
        install_log(str(_rd.log_path))

    device = torch.device(
        args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    )
    cfg = Config()
    log_line(f'[zoom] ckpt={args.ckpt} device={device} cov_range={tuple(args.cov_range)}')

    # ── data: same spec/split as the trainer ──────────────────────────────────
    spec = IMD2020BCESpec(
        imd2020_root=args.imd2020_root, casia_root=args.casia_root,
        indoor_root=args.indoor_root,
        imd_train=not args.imd_val_only, casia_train=args.casia_train,
    )
    _, val_items = spec.build_items(cfg)
    imd_val   = [it for it in val_items if it.get('source', '') == 'imd2020']
    casia_val = [it for it in val_items if it.get('source', '') == 'casia']
    log_line(f'[zoom] imd_val={len(imd_val)} casia_val={len(casia_val)}')

    # ── model (auto-detect heads) ──────────────────────────────────────────────
    ckpt = ckpt_load(args.ckpt)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    contrastive_dim, pool_hidden = _infer_head_dims(sd)
    if contrastive_dim == 0:
        log_line('[zoom] ERROR: checkpoint has no contrastive head — nothing to localize.')
        return
    model = build_multi_head_detector(
        model_name=cfg.MODEL_NAME, resolution=cfg.resolution,
        lora_rank=cfg.LORA_RANK, lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT, lora_targets=cfg.LORA_TARGETS,
        contrastive_dim=contrastive_dim, pool_hidden=pool_hidden, device=device,
    )
    model.load_state_dict(sd)
    model.eval()
    log_line(f'[zoom] loaded epoch={ckpt.get("epoch", "?")} '
             f'contrastive_dim={contrastive_dim} pool_hidden={pool_hidden}')

    from lab_utils.eval.decode_cli import decode_spec_from_args, decode_label
    decode_spec = decode_spec_from_args(args)
    log_line(f'[zoom] decode={decode_label(decode_spec)}')

    for items, tag in ((imd_val, 'imd_val'), (casia_val, 'casia_val')):
        if not items:
            continue
        items = _subsample(items, args.eval_max_items, seed=f'zoomrec|{tag}')
        log_line(f'[zoom] ── {tag} (n_items={len(items)}) ──')

        # ── tile-OR SLIDING-WINDOW localization (the actual "swin") ──────────
        if args.run_swin:
            loader = _loader(items, cfg, args, device)
            lsamples = collect_localization_samples(
                model, loader, device,
                n_patch_per_side=cfg.resolution.num_patches_per_side,
                run_swin=True,
                swin_scales=(float(args.swin_scale),),
                swin_stride_frac=float(args.swin_stride_frac),
                swin_inner_batch=int(args.swin_inner_batch),
                swin_bce_gate_threshold=0.0,
                swin_use_source_resolution=True,
                swin_normalize_mean=cfg.IMAGENET_MEAN, swin_normalize_std=cfg.IMAGENET_STD,
                res=cfg.resolution, decode_spec=decode_spec, log_tag='[swin]', tag=tag,
            )
            if lsamples:
                report_localization_threshold_sweep(
                    lsamples, methods=('full', 'swin'),
                    threshold_grid=tuple(args.loc_threshold_grid),
                    opt_thresh=None, log_tag='[swin]', tag=tag,
                )
                report_loc_by_confidence(lsamples, t_op=None, log_tag='[swin]', tag=tag)
                report_oracle_tax(lsamples, log_tag='[swin]', tag=tag)

        zsamples = collect_zoom_eval_samples(
            model, items, device,
            res=cfg.resolution, cov_range=tuple(args.cov_range),
            seed=f'zoomrec|{tag}',
            normalize_mean=cfg.IMAGENET_MEAN, normalize_std=cfg.IMAGENET_STD,
            decode_spec=decode_spec, log_tag='[zoom]', tag=tag,
        )
        report_zoom_eval(zsamples, log_tag='[zoom]', tag=tag)

        if args.coarse2fine:
            cf = collect_coarse_to_fine_samples(
                model, items, device, res=cfg.resolution,
                normalize_mean=cfg.IMAGENET_MEAN, normalize_std=cfg.IMAGENET_STD,
                pad_frac=float(args.pad_frac), refine_max_frac=float(args.refine_max_frac),
                decode_spec=decode_spec, zoom_mode=args.zoom_mode, log_tag='[zoom]', tag=tag,
            )
            report_coarse_to_fine(cf, log_tag='[zoom]', tag=tag)

    log_line('[zoom] eval complete')


if __name__ == '__main__':
    main()
