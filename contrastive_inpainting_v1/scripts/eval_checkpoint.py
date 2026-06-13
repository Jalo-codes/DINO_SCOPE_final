"""contrastive_inpainting_v1.scripts.eval_checkpoint — standalone checkpoint
report for IMD + TGIF: detection, localization, dense localization, zoom, and
upgraded zoom-aware viz, pointed at a saved checkpoint (e.g. on Drive).

Sources: IMD2020 (one source) + each TGIF category as its OWN source/cell
(type sp/fr × mask family random/semantic). Per-source image counts are
passable: --imd_n caps IMD, --tgif_per_cell sets fakes per TGIF category,
--tgif_n_real sets the shared real negatives per category.

Detection + localization + dense-localization share a SINGLE backbone forward
per source (the model returns image_logit / contrastive / patch_logit from one
forward; a replay cache fans that one pass out to all three metric functions
instead of re-forwarding the DINO backbone three times). Zoom and the
robustness sweep necessarily re-forward (they crop / corrupt the input).

Viz: two rows per item — FULL and attention-guided ZOOM — each showing the
image-head (pool-attention) weighting with p=… and the k-means/graph partition.

Arch (contrastive_dim / pool_hidden / patch_bce) is inferred from the
checkpoint state dict.

Colab A100 40GB:
    !python -m contrastive_inpainting_v1.scripts.eval_checkpoint \
        --ckpt /content/drive/MyDrive/DINO_SCOPE_RUNS/<run>/epoch_007.pt \
        --imd2020_root /content/IMD2020 \
        --tgif_root /content/dataset_root/content/flux_originals \
        --tgif_index /content/dataset_root/content/flux_originals/tgif2_index.json \
        --imd_n 200 --tgif_per_cell 75 --tgif_n_real 50 --val_zoom
"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from lab_utils.logging.text import install_log, log_line
from lab_utils.data.dataset import LabDataset
from lab_utils.data.loaders import LoaderConfig, build_eval_loader
from lab_utils.train.checkpoint import load as ckpt_load
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.eval.localization import collect_zoom_eval_samples, report_zoom_eval
from lab_utils.eval.robustness import run_robustness_sweep
from lab_utils.eval.zoom import attention_zoom_bbox
from lab_utils.eval.partition import spherical_kmeans2, decode_deploy_mask, polarity_attn
from lab_utils.eval.decode_cli import add_decode_args, decode_spec_from_args, decode_label

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.configs.augment import (
    eval_aug_settings,
    DEFAULT_EVAL_AUG_CONDITIONS,
    EVAL_AUG_CHOICES,
)
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec
from contrastive_inpainting_v1.experiments.tgif2_flux import (
    build_tgif2_items, split_tgif2_coco_ids)


# Reuse the trainer's eval machinery verbatim — same metrics, same log format,
# so checkpoint reports are directly comparable with in-train eval lines.
from contrastive_inpainting_v1.scripts.train_multi_head import (
    _SPLICE_KINDS,
    _BCEHeadAdapter,
    _make_bce_eval_callable,
    _prep_tgif_items,
    _run_image_bce_eval,
    _run_localization_eval,
    _run_patch_bce_loc_eval,
    _subsample_items,
    _tgif_model_filter,
)


# ── A100-friendly wrapper: bf16 autocast forward, fp32 outputs ───────────────

class _AutocastModel(nn.Module):
    """Run the wrapped model under bf16 autocast and upcast every floating
    output back to fp32, so downstream numpy conversions never see bfloat16."""

    def __init__(self, model: nn.Module, enabled: bool):
        super().__init__()
        self.model = model
        self.enabled = bool(enabled)

    def forward(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=self.enabled):
            out = self.model(x)
        if isinstance(out, dict):
            return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point()
                        else v)
                    for k, v in out.items()}
        return out.float() if torch.is_tensor(out) and out.is_floating_point() else out


class _ReplayModel(nn.Module):
    """Returns precomputed forward outputs in call order so detection,
    localization, and dense localization share ONE real backbone forward
    instead of re-forwarding the DINO backbone once per metric. Populate via
    :func:`_precompute_outputs`; call :meth:`rewind` before each consumer.

    Alignment relies on every consumer iterating the SAME deterministic eval
    loader and skipping None batches identically (they all do `if batch is None:
    continue` before the forward), so the i-th forward call maps to the i-th
    cached non-None batch.
    """

    def __init__(self):
        super().__init__()
        self._cache = []
        self._ptr = 0

    def add(self, out_dict):
        self._cache.append(out_dict)

    def rewind(self):
        self._ptr = 0

    @property
    def n_batches(self):
        return len(self._cache)

    def forward(self, x):
        out = self._cache[self._ptr]
        self._ptr += 1
        return out


@torch.no_grad()
def _precompute_outputs(model, loader, device):
    """Iterate the loader ONCE, run one forward per batch, cache CPU outputs.
    Returns a populated _ReplayModel (or None if the loader is empty)."""
    replay = _ReplayModel()
    for batch in loader:
        if batch is None:
            continue
        img = batch['img'].to(device, non_blocking=True)
        out = model(img)
        replay.add({k: (v.detach().cpu() if torch.is_tensor(v) else v)
                    for k, v in out.items()})
    return replay if replay.n_batches > 0 else None


def _infer_arch(state_dict) -> dict:
    """Read head dims straight from checkpoint tensor shapes."""
    arch = {'contrastive_dim': 0, 'pool_hidden': 0, 'patch_bce': False}
    w = state_dict.get('contrastive_proj.weight')
    if w is not None:
        arch['contrastive_dim'] = int(w.shape[0])
    w = state_dict.get('pool.V.weight')
    if w is not None:
        arch['pool_hidden'] = int(w.shape[0])
    if 'patch_head.weight' in state_dict:
        arch['patch_bce'] = True
    return arch


# ── upgraded viz: FULL row + attention-guided ZOOM row, kmeans/graph ──────────

def _kmeans_panel(z_np: np.ndarray, att_np, n: int):
    """(N, D) embeddings → boolean (n, n) splice mask via spherical kmeans,
    polarity set by the image-head attention (hot cluster = splice)."""
    raw_labels, _ = spherical_kmeans2(z_np, n_init=4)
    if att_np is not None:
        km = polarity_attn(raw_labels, att_np)
    else:
        km = raw_labels if raw_labels.sum() <= len(raw_labels) / 2 else 1 - raw_labels
    return km.reshape(n, n).astype(bool)


@torch.no_grad()
def run_checkpoint_viz(model, viz_items, out_dir, device, cfg, decode_spec,
                       *, zoom_thresh_mode='otsu', max_frame_frac=0.85):
    """Two-row composite per item.

    Row FULL:  Original | image-head attention (p=…) | kmeans/graph | GT
    Row ZOOM:  attention-guided crop | attention on crop (p=…) | kmeans/graph | GT crop

    The zoom bbox comes from the FULL pass's pool attention via
    attention_zoom_bbox — the same detect-then-zoom geometry as training's
    second pass and the c2f eval. Partition is kmeans or graph.
    """
    import torchvision.transforms.functional as TF
    from torchvision import transforms
    from lab_utils.viz import heatmap_rgb, overlay_blend, mask_tint, save_composite

    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    n = cfg.resolution.num_patches_per_side
    T = cfg.resolution.image_size
    normalize = transforms.Normalize(list(cfg.IMAGENET_MEAN), list(cfg.IMAGENET_STD))

    log_line(f'[eval] viz rendering {len(viz_items)} two-row composites → {out_dir}/')

    def _forward_panels(src_np, inp, tag):
        out = model(inp)
        viz_hw = src_np.shape[:2]
        panels = []
        att = out.get('pool_attention')
        att_np = None
        prob = None
        logit = out.get('image_logit')
        if logit is not None:
            prob = float(torch.sigmoid(logit[0]).item())
        if att is not None:
            att_np = att[0].detach().cpu().float().numpy()
            att_heat = heatmap_rgb(att_np.reshape(n, n), viz_hw)
            p_str = f' p={prob:.3f}' if prob is not None else ''
            panels.append((f'BCE Attn ({tag}){p_str}',
                           overlay_blend(src_np, att_heat)))
        z = out.get('contrastive')
        if z is not None:
            z_np = z[0].detach().cpu().float().numpy()
            if decode_spec.method == 'graph':
                fg, info = decode_deploy_mask(
                    z_np, decode_spec,
                    attention=(att_np.reshape(-1) if att_np is not None else None),
                    grid_hw=(n, n)
                )
                lbl = f'Graph ({tag})'
                if info.get('abstained'):
                    lbl += '\nABSTAIN'
                panels.append((lbl, mask_tint(src_np, fg.reshape(n, n), viz_hw, (255, 90, 0))))
            else:
                km = _kmeans_panel(z_np, att_np, n)
                panels.append((f'K-means ({tag})',
                               mask_tint(src_np, km, viz_hw, (0, 140, 255))))
        return panels, att_np

    n_saved = 0
    for idx, it in enumerate(viz_items):
        img_path = str(it.get('img', ''))
        try:
            source = Image.open(img_path).convert('RGB')
        except Exception as e:
            log_line(f'[eval] viz WARN failed to load {img_path}: {e}')
            continue

        disp = source.resize((T, T), Image.BILINEAR)
        disp_np = np.asarray(disp, dtype=np.uint8)

        gt = None
        mask_path = it.get('mask')
        if mask_path:
            try:
                gt_img = Image.open(str(mask_path)).convert('L').resize(
                    (T, T), Image.NEAREST)
                gt = np.asarray(gt_img, dtype=np.uint8) > 0
            except Exception:
                pass

        inp = normalize(TF.to_tensor(disp)).unsqueeze(0).to(device, non_blocking=True)

        # ── FULL row ──
        panels = [('Original', disp_np)]
        full_panels, att_np = _forward_panels(disp_np, inp, 'full')
        panels += full_panels
        if gt is not None:
            panels.append(('GT', mask_tint(disp_np, gt, (T, T), (0, 255, 0))))
        while len(panels) < 4:                       # keep rows aligned
            panels.append(('', np.zeros_like(disp_np)))

        # ── ZOOM row (attention-guided crop of the SAME image) ──
        bbox = None
        if att_np is not None:
            bbox = attention_zoom_bbox(att_np.reshape(n, n), T, T,
                                       thresh_mode=zoom_thresh_mode)
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                if (x1 - x0) * (y1 - y0) > max_frame_frac * T * T:
                    bbox = None                       # zoom would be a no-op
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            crop_np = np.asarray(
                Image.fromarray(disp_np[y0:y1, x0:x1]).resize((T, T), Image.BILINEAR),
                dtype=np.uint8)
            crop_inp = normalize(TF.to_tensor(Image.fromarray(crop_np))
                                 ).unsqueeze(0).to(device, non_blocking=True)
            panels.append((f'Zoom [{x0},{y0},{x1},{y1}]', crop_np))
            zoom_panels, _ = _forward_panels(crop_np, crop_inp, 'zoom')
            panels += zoom_panels
            if gt is not None:
                gt_crop = np.asarray(
                    Image.fromarray(gt[y0:y1, x0:x1].astype(np.uint8) * 255)
                    .resize((T, T), Image.NEAREST), dtype=np.uint8) > 0
                panels.append(('GT (zoom)',
                               mask_tint(crop_np, gt_crop, (T, T), (0, 255, 0))))

        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        src_tag = str(it.get('source', 'unk'))
        save_path = os.path.join(out_dir, f'{idx:03d}_{src_tag}_{stem}.png')
        save_composite(panels, save_path, panel_size=280, cols=4)
        n_saved += 1

    log_line(f'[eval] viz saved {n_saved} composites → {out_dir}/')


# ── one-forward eval for a single source ─────────────────────────────────────

def _eval_source(model, loader, tag, *, cfg, device, arch, bce_adapter_cls, decode_spec):
    """Detection + localization + dense-localization off ONE backbone forward.

    Returns the optimal image-threshold from the BCE pass (or None) — kept for
    parity with the trainer, which calibrates on imd_val.
    """
    replay = _precompute_outputs(model, loader, device)
    if replay is None:
        log_line(f'[eval] {tag} EMPTY (no batches)')
        return None

    opt_thresh = None
    if arch['pool_hidden'] > 0:                       # detection (image-BCE)
        replay.rewind()
        metrics = _run_image_bce_eval(
            bce_adapter_cls(replay), loader, device, log_tag='[eval]', tag=tag)
        opt_thresh = metrics.get('opt_thresh')
    if arch['contrastive_dim'] > 0:                   # localization (kmeans/graph)
        replay.rewind()
        _run_localization_eval(replay, loader, device, cfg=cfg,
                               decode_spec=decode_spec,
                               log_tag='[eval]', tag=tag)
    if arch['patch_bce']:                             # dense localization
        replay.rewind()
        _run_patch_bce_loc_eval(replay, loader, device, res=cfg.resolution,
                                log_tag='[eval]', tag=tag)
    return opt_thresh


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Standalone IMD + TGIF checkpoint eval (single-forward).')
    p.add_argument('--ckpt', type=str, required=True,
                   help='Checkpoint path (e.g. on Drive). Arch is inferred.')
    p.add_argument('--out_dir', type=str, default=None,
                   help='Report root (default: <ckpt_dir>/report_<ckpt_stem>)')

    # Data roots — IMD + TGIF only.
    p.add_argument('--imd2020_root', type=str, default=None)
    p.add_argument('--tgif_root',    type=str, default=None)
    p.add_argument('--tgif_index',   type=str, default=None)
    p.add_argument('--imd_val_only', action='store_true', default=True,
                   help='IMD held out entirely as val (default for eval).')

    # Passable per-source image counts.
    p.add_argument('--imd_n', type=int, default=200,
                   help='Max IMD images (real+splice) in the eval slice.')
    p.add_argument('--tgif_per_cell', type=int, default=75,
                   help='Fake images per TGIF category (cell).')
    p.add_argument('--tgif_n_real', type=int, default=50,
                   help='Shared real negatives per TGIF category.')
    p.add_argument('--tgif_model', type=str, default=None,
                   help='Optional: restrict to a single TGIF model substring. '
                        'Default None = keep ALL models (all 12 cells).')

    # TGIF eval-half contract (ids file wins; trainer writes it next to ckpts).
    p.add_argument('--tgif_train_half', action='store_true', default=False)
    p.add_argument('--tgif_half_frac', type=float, default=0.5)
    p.add_argument('--tgif_half_seed', type=str, default='tgif_fr_half')
    p.add_argument('--tgif_eval_ids_file', type=str, default=None,
                   help='coco_id-per-line file. Default: auto-detect '
                        'tgif_eval_coco_ids.txt next to --ckpt.')

    # Eval knobs.
    p.add_argument('--gt_patch_threshold', type=float, default=0.06)
    p.add_argument('--val_zoom', action='store_true', default=False)
    p.add_argument('--val_zoom_cov', type=float, nargs=2, default=(0.05, 0.55))
    p.add_argument('--robust', action='store_true', default=False,
                   help='Also run the eval-aug robustness sweep per source '
                        '(re-forwards per condition — the slow block).')
    p.add_argument('--robust_conditions', type=str, nargs='+',
                   default=list(DEFAULT_EVAL_AUG_CONDITIONS),
                   choices=list(EVAL_AUG_CHOICES))

    # Viz.
    p.add_argument('--viz_imd', type=int, default=20,
                   help='IMD splices in the composite set.')
    p.add_argument('--viz_per_cell', type=int, default=5,
                   help='Fakes per TGIF category in the composite set.')
    p.add_argument('--viz_reals', type=int, default=10)
    p.add_argument('--zoom_thresh_mode', type=str, default='otsu',
                   choices=('gap', 'otsu'))

    # Runtime — eval-only, so the batch can run far hotter than training.
    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device',      type=str, default='cuda')
    p.add_argument('--no_bf16', action='store_true', default=False)
    add_decode_args(p)
    return p


def main():
    args = _build_parser().parse_args()
    decode_spec = decode_spec_from_args(args)
    device = torch.device(
        args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_dir = args.out_dir or os.path.join(ckpt_dir, f'report_{stem}')
    os.makedirs(out_dir, exist_ok=True)
    install_log(os.path.join(out_dir, 'eval.log'))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_bf16 = (not args.no_bf16) and device.type == 'cuda'
    log_line(f'[cfg] ckpt={args.ckpt} out={out_dir} device={device} bf16={use_bf16} decode={decode_label(decode_spec)}')

    cfg = Config()

    # ── model: arch inferred from the checkpoint ──
    ckpt = ckpt_load(args.ckpt)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    arch = _infer_arch(sd)
    log_line(f"[cfg] inferred arch: contrastive_dim={arch['contrastive_dim']} "
             f"pool_hidden={arch['pool_hidden']} patch_bce={arch['patch_bce']} "
             f"epoch={ckpt.get('epoch', '?')}")
    model = build_multi_head_detector(
        model_name=cfg.MODEL_NAME,
        resolution=cfg.resolution,
        lora_rank=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        lora_targets=cfg.LORA_TARGETS,
        device=device,
        **arch,
    )
    model.load_state_dict(sd)
    model.backbone.gradient_checkpointing_disable()
    model.eval()
    eval_model = _AutocastModel(model, enabled=use_bf16)

    def _sub_loader(items):
        if not items:
            return None
        ds = LabDataset(
            items, cfg.resolution,
            cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
            augment=False,
            use_degradation=False, use_invariance=False,
            use_splice_degradation=False,
            gt_patch_threshold=float(args.gt_patch_threshold),
        )
        return build_eval_loader(ds, LoaderConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
        ))

    # ── source 1: IMD ──
    eval_sources = []        # (loader, items, tag)
    viz_items = []
    spec = IMD2020BCESpec(
        imd2020_root=args.imd2020_root,
        imd_train=not args.imd_val_only,
    )
    _, imd_val = spec.build_items(cfg)
    imd_val = [it for it in imd_val if it.get('source', '') == 'imd2020']
    if imd_val:
        imd_cap = _subsample_items(imd_val, args.imd_n, seed='full_eval')
        eval_sources.append((_sub_loader(imd_cap), imd_cap, 'imd_val'))
        log_line(f'[cfg] imd_val={len(imd_cap)}/{len(imd_val)}')
        _imd_sp = [it for it in imd_val
                   if it.get('kind') in _SPLICE_KINDS and it.get('mask')]
        _imd_re = [it for it in imd_val if 'real' in str(it.get('kind', ''))]
        viz_items += _subsample_items(_imd_sp, args.viz_imd, seed='viz_imd')
        viz_items += _subsample_items(_imd_re, args.viz_reals, seed='viz_imd_real')

    # ── source 2..N: each TGIF category as its own cell ──
    tgif_eval_ids = None
    ids_file = args.tgif_eval_ids_file or os.path.join(ckpt_dir, 'tgif_eval_coco_ids.txt')
    if os.path.isfile(ids_file):
        with open(ids_file) as f:
            tgif_eval_ids = {ln.strip() for ln in f if ln.strip()}
        log_line(f'[cfg] tgif eval ids: {len(tgif_eval_ids)} from {ids_file}')
    elif args.tgif_train_half and args.tgif_root:
        _tg_index = args.tgif_index or os.path.join(args.tgif_root, 'tgif2_index.json')
        _, _ev_ids = split_tgif2_coco_ids(
            _tg_index, train_frac=args.tgif_half_frac, seed=args.tgif_half_seed)
        tgif_eval_ids = set(_ev_ids)
        log_line(f'[cfg] tgif eval ids: {len(tgif_eval_ids)} recomputed '
                 f'(seed={args.tgif_half_seed!r} frac={args.tgif_half_frac})')

    if args.tgif_root:
        # All 12 cells = model × type(sp/fr) × mask_type(bbox/segm/random). No
        # model filter and no FR-only restriction here — we want every folder.
        # The held-out coco_id filter (if any) still applies to ALL types, so a
        # COCO image trained on as an FR fake never leaks back in via its sp
        # variant or its real.
        tg_fakes, tg_reals = build_tgif2_items(
            args.tgif_root, args.tgif_index, include_reals=True,
            coco_ids=tgif_eval_ids, types=None)
        if args.tgif_model:                          # optional single-model restrict
            tg_fakes = _tgif_model_filter(
                tg_fakes, args.tgif_model, log_tag='[eval]', tag='tgif')
        if tg_fakes and tg_reals:
            # Partition by mask FAMILY (semantic = bbox+segm merged, vs random)
            # rather than the raw mask_type, so bbox and segm share one cell —
            # the extra bbox/segm split made cells hard to compare.
            cells = {}
            for fk in tg_fakes:
                key = (str(fk.get('tgif_model', 'NA')),
                       str(fk.get('tgif_type', 'NA')),
                       str(fk.get('tgif_mask_family', 'NA')))
                cells.setdefault(key, []).append(fk)
            log_line(f'[cfg] tgif cells={len(cells)}: {sorted(cells)}')
            _tg_cache = os.path.join(out_dir, 'tgif_mask_cache')
            tg_reals_sub = _subsample_items(tg_reals, int(args.tgif_n_real), seed='tgif_real')
            tg_reals_sub = _prep_tgif_items(
                tg_reals_sub, mask_cache_dir=_tg_cache, log_tag='[eval]', tag='tgif reals')
            for (md, t_, mf) in sorted(cells):
                cell_seed = f'{md}|{t_}|{mf}'
                c_fakes = _subsample_items(
                    cells[(md, t_, mf)], int(args.tgif_per_cell), seed=f'tgif_fake|{cell_seed}')
                c_fakes = _prep_tgif_items(
                    c_fakes, mask_cache_dir=_tg_cache,
                    log_tag='[eval]', tag=f'tgif fakes {cell_seed}')
                tag = f'tgif_val/{md}/{t_}/{mf}'
                eval_sources.append((_sub_loader(c_fakes + tg_reals_sub), c_fakes, tag))
                log_line(f'[cfg] {tag}: fakes={len(c_fakes)} reals={len(tg_reals_sub)}')
                viz_items += _subsample_items(c_fakes, args.viz_per_cell, seed=f'viz|{cell_seed}')

    if not eval_sources:
        log_line('[eval] no sources — pass --imd2020_root and/or --tgif_root')
        return

    # ── detection + localization + dense-loc: ONE forward per source ──
    for loader, _, tag in eval_sources:
        _eval_source(eval_model, loader, tag, cfg=cfg, device=device,
                     arch=arch, bce_adapter_cls=_BCEHeadAdapter, decode_spec=decode_spec)

    # ── zoom eval (own forwards: it crops the input) ──
    if args.val_zoom and arch['contrastive_dim'] > 0:
        for _, zoom_items, tag in eval_sources:
            if not zoom_items:
                continue
            zsamples = collect_zoom_eval_samples(
                eval_model, zoom_items, device, res=cfg.resolution,
                cov_range=tuple(args.val_zoom_cov), seed=f'zoomval|{tag}',
                normalize_mean=cfg.IMAGENET_MEAN, normalize_std=cfg.IMAGENET_STD,
                skip_oracle=True, log_tag='[eval]', tag=tag,
            )
            report_zoom_eval(zsamples, condensed=True, log_tag='[eval]', tag=tag)

    # ── robustness sweep (opt-in; re-forwards per aug condition) ──
    if args.robust and arch['pool_hidden'] > 0:
        bce_adapter = _BCEHeadAdapter(eval_model)
        aug_conditions = [(name, eval_aug_settings(name, cfg))
                          for name in args.robust_conditions]
        for _, items, tag in eval_sources:
            if not items:
                continue
            eval_fn = _make_bce_eval_callable(
                bce_adapter, items, cfg, device,
                batch_size=args.batch_size, num_workers=args.num_workers,
                gt_patch_threshold=float(args.gt_patch_threshold),
            )
            run_robustness_sweep(
                eval_fn, aug_conditions,
                metrics_to_show=('auc', 'bal_acc', 'f1', 'tpr', 'tnr',
                                 'tpr_at_tnr_95', 'tpr_at_tnr_99'),
                baseline_name='none', log_tag='[robust]', tag=tag,
            )

    # ── viz ──
    run_checkpoint_viz(
        eval_model, viz_items, os.path.join(out_dir, 'viz'), device, cfg, decode_spec,
        zoom_thresh_mode=args.zoom_thresh_mode,
    )

    log_line(f'[eval] checkpoint report complete → {out_dir}')


if __name__ == '__main__':
    main()
