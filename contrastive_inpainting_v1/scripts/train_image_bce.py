"""contrastive_inpainting_v1.scripts.train_image_bce — multi-scale BCE training
with built-in sliding-window eval each epoch.

Differences from train_image_bce:
  - Wider training crop range (0.40, 1.00) so the model sees splices at scales
    1×–2.5× their natural area, matching what 0.5× sliding-window crops produce
    at inference. One model handles both zoomed-in and zoomed-out views.
  - Each epoch's eval runs BOTH full-image AND sliding-window inference and
    reports them side-by-side: per-area-tier TPR lift, FP risk on real images,
    TPR at fixed-FPR operating points.

Usage:
    python -m contrastive_inpainting_v1.scripts.train_image_bce \\
        --imd2020_root /media/ssd/DINO_SCOPE_DATA/IMD2020 \\
        --casia_root   /media/ssd/DINO_SCOPE_DATA/casia \\
        --checkpoint_root /media/ssd/runs/image_bce_v2
"""

import argparse
import dataclasses
import hashlib
import math
import os
import sys
import time
from typing import Dict, List, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from lab_utils.logging.text import install_log, log_line
from lab_utils.data.dataset import LabDataset, lab_collate_fn
from lab_utils.data.loaders import LoaderConfig, build_eval_loader
from lab_utils.data.resolution import Resolution
from lab_utils.train.checkpoint import load as ckpt_load, save as ckpt_save  # noqa: F401
from lab_utils.train.distributed import (
    DistributedContext,
    barrier,
    cleanup as ddp_cleanup,
    setup as ddp_setup,
    unwrap_model,
    wrap_model,
)
from lab_utils.model.image_bce_detector import build_image_bce_detector

from lab_utils.eval.sliding_window import (
    run_sliding_window_eval,
    format_sliding_window_report,
)
from lab_utils.eval.robustness import (
    run_robustness_sweep,
    metrics_from_logits,
)

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.configs.augment import (
    build_aug_kwargs,
    eval_aug_settings,
    DEFAULT_EVAL_AUG_CONDITIONS,
)
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec


# Default crop-scale mixture: heavy weight on near-natural scales, tight zoom rare.
# Splice-area inflation factor by tier:  ~1.0×, ~1.4×, ~2.0×, ~2.5×.
DEFAULT_CROP_MIX = [
    ((0.85, 1.00), 0.50),   # 50%: near-natural — splices at their natural area
    ((0.65, 0.85), 0.30),   # 30%: slight crop  — splice area inflates ~1.4×
    ((0.45, 0.65), 0.15),   # 15%: medium zoom  — ~2.0×
    ((0.30, 0.45), 0.05),   # 5%:  tight zoom   — ~2.5× (model needs SOME of these)
]


# ── Eval ─────────────────────────────────────────────────────────────────────

def _run_image_bce_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    log_tag: str = '[eval]',
    tag: str = '',
) -> Dict:
    """Binary image-level eval. Returns dict of scalars."""
    model.eval()
    logits_all, labels_all, kinds_all, areas_all = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            img       = batch['img'].to(device, non_blocking=True)
            is_single = batch['is_single'].cpu().numpy().astype(bool)
            meta_list = batch['meta'] if isinstance(batch['meta'], list) else [
                {k: v[i] for k, v in batch['meta'].items()}
                for i in range(img.shape[0])
            ]

            logit = model(img)   # (B,)
            logit_np = logit.detach().cpu().float().numpy()

            for i in range(len(logit_np)):
                logits_all.append(float(logit_np[i]))
                labels_all.append(0 if bool(is_single[i]) else 1)   # 1 = splice
                kinds_all.append(str(meta_list[i].get('kind', '')))
                areas_all.append(float(meta_list[i].get('blob_area_actual', 0.0)))

    logits = np.array(logits_all, dtype=np.float64)
    labels = np.array(labels_all, dtype=np.int32)
    probs  = 1.0 / (1.0 + np.exp(-logits))
    preds  = (probs >= threshold).astype(np.int32)

    n_total  = len(labels)
    n_splice = int(labels.sum())
    n_real   = n_total - n_splice

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    tpr = tp / n_splice if n_splice > 0 else float('nan')
    tnr = tn / n_real   if n_real   > 0 else float('nan')
    bal_acc = 0.5 * (tpr + tnr) if n_splice > 0 and n_real > 0 else float('nan')
    prec    = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    f1      = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')
    acc     = (tp + tn) / n_total if n_total > 0 else float('nan')

    # AUC via trapezoidal rule (sort by descending logit)
    try:
        order = np.argsort(-logits)
        sorted_labels = labels[order]
        tpr_pts = np.cumsum(sorted_labels) / n_splice
        fpr_pts = np.cumsum(1 - sorted_labels) / n_real
        auc = float(np.trapezoid(tpr_pts, fpr_pts))
        if auc < 0:
            auc = 1.0 + auc
    except Exception:
        auc = float('nan')

    # Optimal threshold (maximises balanced accuracy on this eval set)
    opt_thresh, opt_tpr, opt_tnr, opt_bacc = logits[0], 0.0, 1.0, 0.5
    if n_splice > 0 and n_real > 0:
        grid = np.unique(logits)
        for t in grid:
            p = (logits >= t).astype(np.int32)
            _tpr = float(((p == 1) & (labels == 1)).sum()) / n_splice
            _tnr = float(((p == 0) & (labels == 0)).sum()) / n_real
            _ba  = 0.5 * (_tpr + _tnr)
            if _ba > opt_bacc:
                opt_bacc, opt_thresh, opt_tpr, opt_tnr = _ba, float(t), _tpr, _tnr
    opt_preds = (logits >= opt_thresh).astype(np.int32)

    # Per-area-bucket detection rate (at both fixed and optimal threshold)
    def area_tier(a):
        if a < 0.15:
            return 'small'
        if a < 0.30:
            return 'medium'
        return 'large'

    bucket_stats = {}
    for bname in ('small', 'medium', 'large'):
        mask = np.array([
            labels_all[i] == 1 and area_tier(areas_all[i]) == bname
            for i in range(n_total)
        ])
        if mask.sum() == 0:
            bucket_stats[bname] = {'n': 0, 'tpr': float('nan'), 'opt_tpr': float('nan')}
        else:
            bucket_stats[bname] = {
                'n':       int(mask.sum()),
                'tpr':     float(preds[mask].mean()),
                'opt_tpr': float(opt_preds[mask].mean()),
            }

    suffix = f' {tag}' if tag else ''
    log_line(
        f'{log_tag}{suffix} '
        f'n_total={n_total} n_splice={n_splice} n_real={n_real} '
        f'auc={auc:.4f} '
        f'@ thresh=0.5: bal_acc={bal_acc:.4f} tpr={tpr:.4f} tnr={tnr:.4f} | '
        f'@ opt thresh={opt_thresh:.3f}: bal_acc={opt_bacc:.4f} tpr={opt_tpr:.4f} tnr={opt_tnr:.4f}'
    )
    for bname in ('small', 'medium', 'large'):
        bs = bucket_stats[bname]
        log_line(
            f'{log_tag}{suffix}   area_tier={bname} n={bs["n"]} '
            f'tpr@0.5={bs["tpr"]:.4f} tpr@opt={bs["opt_tpr"]:.4f}'
        )

    return dict(
        acc=acc, bal_acc=bal_acc, tpr=tpr, tnr=tnr, prec=prec, f1=f1, auc=auc,
        n_total=n_total, n_splice=n_splice, n_real=n_real,
        bucket_stats=bucket_stats,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Phase 2 prime: image-level BCE splice detector')
    p.add_argument('--imd2020_root',     type=str, default=None)
    p.add_argument('--casia_root',       type=str, default=None)
    p.add_argument('--indoor_root',      type=str, default=None)
    # Train/val split control. The flip — train on CASIA, validate on IMD —
    # puts CASIA's tiny-heavy splice distribution into TRAIN (so the classifier
    # actually sees small splices) and holds IMD out as the OOD check.
    p.add_argument('--casia_train',      action='store_true', default=False,
                   help='Include CASIA in the TRAIN split (default: CASIA val-only). '
                        'Pair with --imd_val_only for train-on-CASIA / val-on-IMD.')
    p.add_argument('--imd_val_only',     action='store_true', default=False,
                   help='Hold IMD2020 out entirely as OOD validation (no IMD in train).')
    p.add_argument('--checkpoint_root', '--run_root', dest='checkpoint_root', type=str, default=None)
    p.add_argument('--resume',           type=str, default=None, help='Checkpoint to resume from')
    p.add_argument('--num_epochs',       type=int, default=20)
    p.add_argument('--batch_size',       type=int, default=8)
    p.add_argument('--grad_accum',       type=int, default=4)
    p.add_argument('--lr',               type=float, default=2e-4)
    p.add_argument('--weight_decay',     type=float, default=1e-4)
    p.add_argument('--train_samples',    type=int, default=2000,
                   help='Items per epoch (balanced by class weight)')
    p.add_argument('--num_workers',      type=int, default=4)
    p.add_argument('--device',           type=str, default='cuda')
    p.add_argument('--seed',             type=int, default=42)
    p.add_argument('--log_every',        type=int, default=20,  help='Steps between train log lines')
    p.add_argument('--pool_hidden',      type=int, default=256)
    # Splice crop policy (the small-splice fix). Default = oracle_fallback so
    # tiny/off-center CASIA splices get zoomed into frame instead of being
    # demoted to the whole-image fallback (which relabels them REAL).
    p.add_argument('--min_mask_patch_frac', type=float, default=0.01,
                   help='Min splice patch-fraction to ACCEPT a random crop '
                        '(flat 1%%). Lower = fewer fallbacks.')
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
    # Sliding-window eval
    p.add_argument('--swin_scales',      type=float, nargs='+', default=[1.0, 0.6, 0.4],
                   help='Window scales (fraction of short edge). 1.0 = full image. '
                        'Tighter scales (0.4) zoom small splices up to detectable size.')
    p.add_argument('--swin_square',      action='store_true', default=True,
                   help='Square source-resolution sub-windows (no aspect squish; '
                        'matches training crops). On by default.')
    p.add_argument('--no_swin_square',   dest='swin_square', action='store_false',
                   help='Disable square windows (legacy aspect-ratio windows).')
    p.add_argument('--swin_stride_frac', type=float, default=0.5,
                   help='Stride as a fraction of window size (0.5 = 50%% overlap)')
    p.add_argument('--swin_inner_batch', type=int, default=8,
                   help='Forward-pass batch size inside sliding-window inference')
    p.add_argument('--swin_every',       type=int, default=3,
                   help='Run sliding-window eval every N epochs (0 = never)')
    # Training crop scale — biased mixture (most samples near-natural)
    p.add_argument('--train_crop_min',   type=float, default=0.40,
                   help='(Legacy) min crop scale; ignored if --crop_mix is used.')
    p.add_argument('--use_crop_mix',     action='store_true', default=True,
                   help='Use biased mixture distribution over crop scale (default ON)')
    # Augmentation intensity
    p.add_argument('--aug_intensity',    type=str, default='medium',
                   choices=('none', 'light', 'medium', 'heavy'),
                   help="Whole-image augmentation intensity. 'medium' "
                        "applies moderate JPEG/noise/resize on a minority "
                        "of samples; 'heavy' overlaps eval-probe severities.")
    # Robustness sweep
    p.add_argument('--robust_every',     type=int, default=3,
                   help='Run robustness sweep every N epochs (0 = never)')
    p.add_argument('--robust_conditions', type=str, nargs='+',
                   default=list(DEFAULT_EVAL_AUG_CONDITIONS),
                   help='Augmentation conditions to sweep')
    # Cap val sets used by swin and robust evals (full-image eval still uses all)
    p.add_argument('--eval_max_items',   type=int, default=200,
                   help='Cap per-source val items for swin and robust evals')
    return p


_REAL_KINDS = frozenset({'imd_real', 'indoor_real', 'casia_real'})


def _is_real(item: dict) -> bool:
    return item.get('kind', '') in _REAL_KINDS


@torch.no_grad()
def _collect_logits(model, loader, device):
    """Run model over loader; return (logits, labels) numpy arrays."""
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


def _subsample_items(items, n, *, seed='swin') -> list:
    """Deterministic subsample by md5(seed | path). Stable across runs."""
    if not items or len(items) <= n:
        return list(items)
    def _key(it):
        path = it.get('img') or it.get('path') or ''
        return hashlib.md5(f'{seed}|{path}'.encode('utf-8')).hexdigest()
    return sorted(items, key=_key)[:n]


def _make_bce_eval_callable(model, items, cfg, device, *, batch_size, num_workers):
    """Build a closure: aug_kwargs → metrics dict, for the robustness sweep."""
    def _eval_under_aug(aug_kwargs, *, tag):
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
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda'),
            ),
        )
        logits, labels = _collect_logits(model, loader, device)
        return metrics_from_logits(logits, labels)
    return _eval_under_aug


def _splice_balance_weights(items: list) -> list:
    """Shim — see :func:`lab_utils.data.sampling.splice_balance_weights`."""
    from lab_utils.data.sampling import splice_balance_weights
    return splice_balance_weights(items, target_splice_frac=0.5)


def main():
    args = _build_parser().parse_args()
    from contrastive_inpainting_v1.pipeline.cli import apply_path_defaults
    apply_path_defaults(args)

    # ── DDP setup ─────────────────────────────────────────────────────────────
    # torchrun sets RANK/LOCAL_RANK/WORLD_SIZE; single-process falls back cleanly.
    ctx: DistributedContext = ddp_setup()
    import atexit
    atexit.register(ddp_cleanup)   # clean NCCL teardown on any exit (sweep-safe)
    torch.manual_seed(args.seed + ctx.rank)
    np.random.seed(args.seed + ctx.rank)

    # Rank 0 owns all logging + the run dir. Silence the other ranks: log_line
    # echoes to stdout on EVERY rank otherwise, so torchrun --tee would N×-dupe
    # every line. Eval / checkpoint are also rank-0 only (below).
    if ctx.is_main:
        # New run-dir layout: split args.checkpoint_root into (parent, name) so
        # build_run_dir creates the new logs/<ts>_<git>_train/ subdir.
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

    cfg = Config()

    # ── data ─────────────────────────────────────────────────────────────────
    spec = IMD2020BCESpec(
        imd2020_root=args.imd2020_root,
        casia_root=args.casia_root,
        indoor_root=args.indoor_root,
        imd_train=not args.imd_val_only,
        casia_train=args.casia_train,
    )
    train_items, val_items = spec.build_items(cfg)
    log_line(f'[cfg] train={len(train_items)} val={len(val_items)}')

    # Training dataset:
    #   - Same crop_scale and crop_ratio for IMD and non-IMD items so the
    #     crop treatment is identical across reals and splices (no extra
    #     variable confounded with kind).
    #   - Biased mixture distribution over scale: most samples near-natural
    #     scale, decreasing weight toward tight zoom. See DEFAULT_CROP_MIX.
    #   - Whole-image augmentation intensity selectable via --aug_intensity.
    aug_kwargs = build_aug_kwargs(cfg, args.aug_intensity)
    log_line(f'[cfg] aug_intensity={args.aug_intensity} aug_kwargs={aug_kwargs}')

    if args.use_crop_mix:
        crop_mix = DEFAULT_CROP_MIX
        crop_scale_fallback = (DEFAULT_CROP_MIX[0][0][0], DEFAULT_CROP_MIX[-1][0][1])
    else:
        crop_mix = None
        crop_scale_fallback = (float(args.train_crop_min), 1.00)
    log_line(f'[cfg] crop_mix={"on" if crop_mix else "off"} '
             f'fallback_range={crop_scale_fallback}')
    log_line(f'[cfg] crop ratio unified for reals+splices: {cfg.CROP_RATIO}')

    train_ds = LabDataset(
        train_items, cfg.resolution,
        cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
        augment=True,
        light_aug_kwargs=aug_kwargs,
        crop_scale=crop_scale_fallback,
        crop_ratio=cfg.CROP_RATIO,
        imd_crop_scale=crop_scale_fallback,   # same as crop_scale: unified
        imd_crop_ratio=cfg.CROP_RATIO,        # same as crop_ratio: unified
        crop_max_tries=int(args.crop_max_tries),
        min_mask_patch_frac=float(args.min_mask_patch_frac),
        splice_crop_mode=str(args.splice_crop_mode),
        oracle_target_cov=tuple(args.oracle_cov),
        crop_scale_mix=crop_mix,
        imd_crop_scale_mix=crop_mix,          # same mix for IMD: unified
        use_degradation=False,
        use_invariance=False,
        use_splice_degradation=False,
    )
    log_line(f'[cfg] splice_crop_mode={args.splice_crop_mode} '
             f'min_mask_patch_frac={args.min_mask_patch_frac} '
             f'crop_max_tries={args.crop_max_tries} '
             f'oracle_cov={tuple(args.oracle_cov)}')
    log_line(f'[cfg] swin_scales={list(args.swin_scales)} '
             f'square={args.swin_square} '
             f'stride_frac={args.swin_stride_frac} '
             f'every={args.swin_every}')
    log_line(f'[cfg] robust_every={args.robust_every} '
             f'eval_max_items={args.eval_max_items}')

    weights = _splice_balance_weights(train_items)
    # DDP: each rank draws its own shard of the per-epoch sample budget from the
    # full weighted distribution (per-rank generator seed → different draws).
    per_rank_samples = max(1, args.train_samples // max(1, ctx.world_size))
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=per_rank_samples,
        replacement=True,
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
    )

    # Val datasets — no augmentation
    val_ds = LabDataset(
        val_items, cfg.resolution,
        cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
        augment=False,
        use_degradation=False,
        use_invariance=False,
        use_splice_degradation=False,
    )
    val_loader = build_eval_loader(
        val_ds,
        LoaderConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
        ),
    )

    # Separate IMD2020 vs CASIA val for comparison
    imd_val_items  = [it for it in val_items if it.get('source', '') == 'imd2020']
    casia_val_items = [it for it in val_items if it.get('source', '') == 'casia']
    log_line(f'[cfg] imd_val={len(imd_val_items)} casia_val={len(casia_val_items)}')

    def _sub_loader(items):
        if not items:
            return None
        ds = LabDataset(
            items, cfg.resolution,
            cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
            augment=False,
            use_degradation=False, use_invariance=False, use_splice_degradation=False,
        )
        return build_eval_loader(ds, LoaderConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
        ))

    # Cap per-epoch full-image eval to eval_max_items per source (deterministic
    # md5 subsample, same subset every epoch so trends are comparable).
    imd_eval_items   = _subsample_items(imd_val_items,   args.eval_max_items, seed='full_eval')
    casia_eval_items = _subsample_items(casia_val_items, args.eval_max_items, seed='full_eval')
    log_line(f'[cfg] full_eval cap: imd={len(imd_eval_items)}/{len(imd_val_items)} '
             f'casia={len(casia_eval_items)}/{len(casia_val_items)}')
    imd_val_loader   = _sub_loader(imd_eval_items)
    casia_val_loader = _sub_loader(casia_eval_items)

    # ── model ─────────────────────────────────────────────────────────────────
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

    start_epoch = 0
    if args.resume:
        ckpt = ckpt_load(args.resume, map_location=str(device))
        model.load_state_dict(ckpt['model'])
        start_epoch = int(ckpt.get('epoch', 0))
        log_line(f'[ckpt] resumed epoch={start_epoch} ← {args.resume}')

    # DDP-wrap after loading any resume weights (load into the bare module).
    model = wrap_model(model, ctx, device=device)

    # ── optimizer / schedule ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, math.ceil(per_rank_samples / args.batch_size / args.grad_accum))
    total_steps = args.num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.05,
    )

    # pos_weight=1.0: WeightedRandomSampler already balances batches 50/50,
    # so no additional per-class weighting is needed in the loss.
    n_splice_total = sum(1 for it in train_items if not _is_real(it))
    n_real_total   = sum(1 for it in train_items if _is_real(it))
    pos_weight = torch.ones(1, dtype=torch.float32, device=device)
    log_line(f'[cfg] pos_weight=1.0 (sampler-balanced) n_splice={n_splice_total} n_real={n_real_total}')

    # ── training loop ─────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        optimizer.zero_grad()
        loss_acc = correct = total = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue

            img  = batch['img'].to(device, non_blocking=True)
            # Use item kind as label — not batch['is_single'] which flips to
            # "real" when a small splice gets cropped below min_mask_patch_frac.
            meta_list = batch['meta'] if isinstance(batch['meta'], list) else [
                {k: v[i] for k, v in batch['meta'].items()}
                for i in range(img.shape[0])
            ]
            label = torch.tensor(
                [0.0 if _is_real({'kind': m.get('kind', '')}) else 1.0
                 for m in meta_list],
                dtype=torch.float32, device=device,
            )

            logit = model(img)   # (B,)
            loss  = F.binary_cross_entropy_with_logits(
                logit, label, pos_weight=pos_weight.expand_as(label)
            ) / args.grad_accum
            loss.backward()

            loss_acc += loss.item() * args.grad_accum
            pred     = (logit.detach() >= 0).float()
            correct  += (pred == label).sum().item()
            total    += label.numel()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    elapsed = time.time() - t0
                    log_line(
                        f'[train] epoch={epoch} step={global_step} '
                        f'loss={loss_acc / (step + 1):.4f} '
                        f'acc={correct / total:.4f} '
                        f'lr={scheduler.get_last_lr()[0]:.2e} '
                        f'elapsed={elapsed:.0f}s'
                    )

        # end-of-epoch flush
        if total % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        log_line(
            f'[train] epoch={epoch} DONE '
            f'loss={loss_acc / max(step + 1, 1):.4f} '
            f'acc={correct / max(total, 1):.4f}'
        )
        # Crop telemetry: what the splice cropping actually did this epoch.
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

            # ── eval (full image) ────────────────────────────────────────────
            if imd_val_loader is not None:
                _run_image_bce_eval(eval_model, imd_val_loader, device,
                                    log_tag='[eval]', tag='imd_val')
            if casia_val_loader is not None:
                _run_image_bce_eval(eval_model, casia_val_loader, device,
                                    log_tag='[eval]', tag='casia_val')

            # ── eval (sliding window — subsampled) ────────────────────────────
            do_swin = (args.swin_every > 0) and ((epoch + 1) % args.swin_every == 0)
            if do_swin:
                t_swin = time.time()
                log_line(f'[swin] starting sliding-window eval epoch={epoch} '
                         f'cap_per_source={args.eval_max_items}')
                for sub_items, sub_tag in (
                    (imd_val_items,   'imd_val'),
                    (casia_val_items, 'casia_val'),
                ):
                    if not sub_items:
                        continue
                    cap_items = _subsample_items(sub_items, args.eval_max_items, seed='swin')
                    cap_ds = LabDataset(
                        cap_items, cfg.resolution,
                        cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
                        augment=False,
                        use_degradation=False, use_invariance=False, use_splice_degradation=False,
                    )
                    cap_loader = build_eval_loader(cap_ds, LoaderConfig(
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == 'cuda'),
                    ))
                    log_line(f'[swin] {sub_tag} cap n={len(cap_items)}/{len(sub_items)}')
                    records = run_sliding_window_eval(
                        eval_model, cap_loader, device,
                        scales=tuple(args.swin_scales),
                        stride_frac=float(args.swin_stride_frac),
                        inner_batch_size=int(args.swin_inner_batch),
                        square=bool(args.swin_square),
                        log_tag='[swin]', tag=sub_tag,
                    )
                    format_sliding_window_report(
                        records, log_tag='[swin]', tag=sub_tag,
                    )
                log_line(f'[swin] sliding-window eval done '
                         f'elapsed={time.time() - t_swin:.0f}s')

            # ── eval (augmentation robustness sweep — subsampled) ─────────────
            do_robust = (args.robust_every > 0) and ((epoch + 1) % args.robust_every == 0)
            if do_robust:
                t_rob = time.time()
                log_line(f'[robust] starting robustness sweep epoch={epoch} '
                         f'cap_per_source={args.eval_max_items} '
                         f'conditions={args.robust_conditions}')
                aug_conditions = [
                    (name, eval_aug_settings(name, cfg))
                    for name in args.robust_conditions
                ]
                for sub_items, sub_tag in (
                    (imd_val_items,   'imd_val'),
                    (casia_val_items, 'casia_val'),
                ):
                    if not sub_items:
                        continue
                    cap_items = _subsample_items(sub_items, args.eval_max_items, seed='robust')
                    log_line(f'[robust] {sub_tag} cap n={len(cap_items)}/{len(sub_items)}')
                    eval_fn = _make_bce_eval_callable(
                        eval_model, cap_items, cfg, device,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
                    run_robustness_sweep(
                        eval_fn, aug_conditions,
                        metrics_to_show=('auc', 'bal_acc', 'tpr', 'tnr',
                                         'tpr_at_tnr_95', 'tpr_at_tnr_99'),
                        baseline_name='none',
                        log_tag='[robust]', tag=sub_tag,
                    )
                log_line(f'[robust] robustness sweep done '
                         f'elapsed={time.time() - t_rob:.0f}s')

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
