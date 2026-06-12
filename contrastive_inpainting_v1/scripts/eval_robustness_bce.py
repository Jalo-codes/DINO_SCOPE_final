"""contrastive_inpainting_v1.scripts.eval_robustness_bce — standalone augmentation
robustness sweep for an image-level BCE checkpoint.

Loads a checkpoint, runs the full augmentation robustness sweep across all
supported conditions (full-image global corruptions + splice-region-only
corruptions), reported per split (imd_val / casia_val).

**Pass the same split flags you used at training time** so the eval never
touches training data.  For the CASIA-train / IMD-val flip:
    --casia_train --imd_val_only

Usage:
    # flip run (train-on-CASIA, val-on-IMD):
    python -m contrastive_inpainting_v1.scripts.eval_robustness_bce \\
        --ckpt /media/ssd/runs/casia_bce_swin_v2/epoch_007.pt \\
        --imd2020_root /media/ssd/DINO_SCOPE_DATA/IMD2020 \\
        --casia_root   /media/ssd/DINO_SCOPE_DATA/casia \\
        --casia_train --imd_val_only

    # symmetric run (train-on-IMD):
    python -m contrastive_inpainting_v1.scripts.eval_robustness_bce \\
        --ckpt /media/ssd/runs/image_bce_prime/epoch_010.pt \\
        --imd2020_root /media/ssd/DINO_SCOPE_DATA/IMD2020 \\
        --casia_root   /media/ssd/DINO_SCOPE_DATA/casia
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
from torch.utils.data import DataLoader

from lab_utils.logging.text import install_log, log_line
from lab_utils.data.dataset import LabDataset
from lab_utils.data.loaders import LoaderConfig, build_eval_loader
from lab_utils.train.checkpoint import load as ckpt_load
from lab_utils.model.image_bce_detector import build_image_bce_detector
from lab_utils.eval.robustness import run_robustness_sweep, metrics_from_logits

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.configs.augment import (
    eval_aug_settings,
    EVAL_AUG_CHOICES,
    DEFAULT_EVAL_AUG_CONDITIONS,
)
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec


_REAL_KINDS = frozenset({'imd_real', 'indoor_real', 'casia_real'})


def _is_real(item):
    return item.get('kind', '') in _REAL_KINDS


@torch.no_grad()
def _collect_logits(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        if batch is None:
            continue
        img = batch['img'].to(device, non_blocking=True)
        meta_list = batch['meta'] if isinstance(batch['meta'], list) else [
            {k: v[i] for k, v in batch['meta'].items()} for i in range(img.shape[0])
        ]
        logit = model(img).detach().cpu().float().numpy()
        for i in range(len(logit)):
            logits_all.append(float(logit[i]))
            labels_all.append(0 if _is_real({'kind': meta_list[i].get('kind', '')}) else 1)
    return np.array(logits_all, dtype=np.float64), np.array(labels_all, dtype=np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--imd2020_root', type=str, default=None)
    p.add_argument('--casia_root',   type=str, default=None)
    p.add_argument('--indoor_root',  type=str, default=None)
    # Split flags — MUST match the flags used at training time.
    p.add_argument('--casia_train',  action='store_true', default=False,
                   help='CASIA was in the training set (flip run). Only '
                        'the CASIA val split is used for eval.')
    p.add_argument('--imd_val_only', action='store_true', default=False,
                   help='IMD was held out entirely (flip run). All IMD '
                        'items are safe to use for eval.')
    p.add_argument('--conditions', type=str, nargs='+',
                   default=list(EVAL_AUG_CHOICES),
                   choices=EVAL_AUG_CHOICES,
                   help='Augmentation conditions to sweep (default: ALL).')
    p.add_argument('--eval_max_items', type=int, default=400,
                   help='Cap per-source val items (deterministic subsample).')
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device',      type=str, default='cuda')
    p.add_argument('--pool_hidden', type=int, default=256)
    p.add_argument('--output_log',  type=str, default=None)
    args = p.parse_args()
    from contrastive_inpainting_v1.pipeline.cli import apply_path_defaults
    apply_path_defaults(args)

    if args.output_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_log)), exist_ok=True)
        install_log(args.output_log)
    else:
        from lab_utils.logging.run_dir import build_run_dir as _build_run_dir
        _ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        _rd = _build_run_dir(
            os.path.dirname(_ckpt_dir),
            os.path.basename(_ckpt_dir),
            role='eval-robust-bce',
        )
        install_log(str(_rd.log_path))

    device = torch.device(
        args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    )
    cfg = Config()

    log_line(f'[eval] ckpt={args.ckpt} device={device}')
    log_line(f'[cfg] casia_train={args.casia_train} imd_val_only={args.imd_val_only} '
             f'conditions={args.conditions}')

    # ── data ─────────────────────────────────────────────────────────────────
    # Use IMD2020BCESpec with the SAME split flags as training so no training
    # items leak into the eval.  (The old code used IMD2020ContrastiveSpec with
    # casia_train=False, which for the flip run would include CASIA train items
    # in the eval set — a data-contamination bug.)
    from lab_utils.data.sampling import deterministic_subsample
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
    imd_items   = deterministic_subsample(imd_items,   args.eval_max_items, seed='robust_eval')
    casia_items = deterministic_subsample(casia_items, args.eval_max_items, seed='robust_eval')
    log_line(f'[cfg] imd_val={len(imd_items)} casia_val={len(casia_items)}')

    # ── model ────────────────────────────────────────────────────────────────
    model = build_image_bce_detector(
        model_name=cfg.MODEL_NAME,
        resolution=cfg.resolution,
        lora_rank=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        lora_targets=cfg.LORA_TARGETS,
        pool_hidden=args.pool_hidden,
        device=device,
    )
    ckpt = ckpt_load(args.ckpt)
    model.load_state_dict(ckpt['model'])
    model.eval()
    log_line(f'[ckpt] loaded epoch={ckpt.get("epoch", "?")}')

    # ── sweep ────────────────────────────────────────────────────────────────
    aug_conditions = [(name, eval_aug_settings(name, cfg)) for name in args.conditions]

    def _make_eval(items):
        def _eval(aug_kwargs, *, tag):
            ds = LabDataset(
                items, cfg.resolution,
                cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
                augment=False,
                use_degradation=False,
                use_invariance=False,
                use_splice_degradation=False,
                **aug_kwargs,
            )
            loader = build_eval_loader(
                ds,
                LoaderConfig(
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == 'cuda'),
                ),
            )
            logits, labels = _collect_logits(model, loader, device)
            return metrics_from_logits(logits, labels)
        return _eval

    # Separate: global corruptions (whole image), then splice-region-only.
    global_conds = [(n, k) for n, k in aug_conditions if not n.startswith('mask_')]
    mask_conds   = [(n, k) for n, k in aug_conditions if n.startswith('mask_')]

    for items, sub_tag in ((imd_items, 'imd_val'), (casia_items, 'casia_val')):
        if not items:
            continue
        eval_fn = _make_eval(items)
        for group_tag, group in (('global', global_conds), ('mask_region', mask_conds)):
            if not group:
                continue
            run_robustness_sweep(
                eval_fn,
                group,
                metrics_to_show=('auc', 'bal_acc', 'tpr', 'tnr',
                                 'tpr_at_tnr_95', 'tpr_at_tnr_99'),
                baseline_name='none',
                log_tag='[robust]', tag=f'{sub_tag} {group_tag}',
            )


if __name__ == '__main__':
    main()
