"""graph_lab.dump_embeddings — one-time cache of REAL embeddings for the sandbox.

Runs the model ONCE over a handful of val splices and freezes everything the
graph decode needs into a single ``.npz``:

    z        (K, N, D)  L2-normalized contrastive embeddings  (the decode input)
    att      (K, N)     per-patch BCE attention
    thumb    (K, P, P, 3) uint8  square thumbnail of the source image
    gt       (K, P, P)  bool      ground-truth splice mask at thumbnail res
    split    (K,)       str       'imd_val' | 'casia_val'
    stem     (K,)       str       image stem (for filenames)
    grid_n   ()         int       patches per side (N == grid_n**2)
    tau_pos / tau_neg   ()        float  the run's trained margins

After this runs once you never touch the GPU/model again — ``sandbox.py`` reloads
the npz instantly and you sweep decode knobs on this REAL geometry.

Usage (mirrors viz_decode's loading flags):
    python -m graph_lab.dump_embeddings \\
        --ckpt /content/drive/MyDrive/DINO_SCOPE_RUNS/<run>/epoch_006.pt \\
        --imd2020_root /content/IMD2020 --casia_root /content/casia \\
        --casia_train --imd_val_only \\
        --tau_pos 0.55 --tau_neg 0.20 \\
        --n_items 20 --out graph_lab/cache/e006.npz
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from lab_utils.data.sampling import deterministic_subsample
from lab_utils.train.checkpoint import load as ckpt_load
from lab_utils.train.amp import resolve_amp
from lab_utils.model.multi_head_detector import build_multi_head_detector

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec

_SPLICE_KINDS = ('imd_splice', 'casia_splice')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--imd2020_root', default=None)
    p.add_argument('--casia_root', default=None)
    p.add_argument('--indoor_root', default=None)
    p.add_argument('--casia_train', action='store_true', default=False)
    p.add_argument('--imd_val_only', action='store_true', default=False)
    p.add_argument('--n_items', type=int, default=20,
                   help='Number of splices to cache PER split.')
    p.add_argument('--out', required=True, help='Output .npz path.')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--panel', type=int, default=320,
                   help='Square thumbnail size (P) stored in the cache.')
    # Stored alongside the embeddings so the sandbox defaults match the run.
    p.add_argument('--tau_pos', type=float, default=0.55)
    p.add_argument('--tau_neg', type=float, default=0.20)
    return p


def main():
    args = _build_parser().parse_args()
    from contrastive_inpainting_v1.pipeline.cli import apply_path_defaults
    apply_path_defaults(args)

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    use_amp, amp_dtype = resolve_amp(device, want_amp=True)
    cfg = Config()
    n = cfg.resolution.num_patches_per_side
    T = cfg.resolution.image_size
    P = int(args.panel)
    normalize = transforms.Normalize(list(cfg.IMAGENET_MEAN), list(cfg.IMAGENET_STD))

    # ── items ──────────────────────────────────────────────────────────────────
    spec = IMD2020BCESpec(
        imd2020_root=args.imd2020_root, casia_root=args.casia_root,
        indoor_root=args.indoor_root,
        imd_train=not args.imd_val_only, casia_train=args.casia_train)
    _, val_items = spec.build_items(cfg)
    by_split = {
        'imd_val':   [it for it in val_items if it.get('source') == 'imd2020'
                      and it.get('kind') in _SPLICE_KINDS and it.get('mask')],
        'casia_val': [it for it in val_items if it.get('source') == 'casia'
                      and it.get('kind') in _SPLICE_KINDS and it.get('mask')],
    }
    for k in by_split:
        by_split[k] = deterministic_subsample(by_split[k], args.n_items, seed='graphlab')
    print(f'[dump] items: imd={len(by_split["imd_val"])} casia={len(by_split["casia_val"])}')

    # ── model ──────────────────────────────────────────────────────────────────
    ckpt = ckpt_load(args.ckpt, map_location=str(device))
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    c_dim = int(sd['contrastive_proj.weight'].shape[0]) if 'contrastive_proj.weight' in sd else 0
    p_hidden = int(sd['pool.V.weight'].shape[0]) if 'pool.V.weight' in sd else 0
    has_patch = 'patch_head.weight' in sd
    if c_dim <= 0:
        print('[dump] ERROR: checkpoint has no contrastive head — nothing to cache.')
        return
    model = build_multi_head_detector(
        model_name=cfg.MODEL_NAME, resolution=cfg.resolution,
        lora_rank=cfg.LORA_RANK, lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT, lora_targets=cfg.LORA_TARGETS,
        contrastive_dim=c_dim, pool_hidden=p_hidden, patch_bce=has_patch, device=device)
    model.load_state_dict(sd); model.eval()
    print(f'[dump] loaded epoch={ckpt.get("epoch","?")} c_dim={c_dim} pool_hidden={p_hidden}')

    z_all, att_all, thumb_all, gt_all, split_all, stem_all = [], [], [], [], [], []

    for split, items in by_split.items():
        for it in items:
            img_path = str(it.get('img', ''))
            stem = os.path.splitext(os.path.basename(img_path))[0]
            try:
                source = Image.open(img_path).convert('RGB')
            except Exception as exc:
                print(f'[dump] WARN load failed {img_path}: {exc}')
                continue

            thumb = np.asarray(source.resize((P, P), Image.BILINEAR), dtype=np.uint8)

            gt = np.zeros((P, P), dtype=bool)
            try:
                gt_img = Image.open(str(it.get('mask'))).convert('L').resize((P, P), Image.NEAREST)
                gt = np.asarray(gt_img, dtype=np.uint8) > 0
            except Exception:
                pass

            inp = normalize(TF.to_tensor(
                TF.resize(source, [T, T], Image.BILINEAR))).unsqueeze(0).to(device)
            with torch.autocast(device_type='cuda', dtype=(amp_dtype or torch.float32),
                                enabled=use_amp):
                out = model(inp)
            z = out.get('contrastive')
            if z is None:
                continue
            z_np = z[0].detach().cpu().float().numpy()
            # Renormalize once here so the sandbox's strict assert never trips on
            # amp rounding — the cache is the canonical decode input.
            z_np = z_np / (np.linalg.norm(z_np, axis=1, keepdims=True) + 1e-12)
            att = out.get('pool_attention')
            att_np = (att[0].detach().cpu().float().numpy() if att is not None
                      else np.zeros(z_np.shape[0], dtype=np.float32))

            z_all.append(z_np.astype(np.float32))
            att_all.append(att_np.astype(np.float32))
            thumb_all.append(thumb)
            gt_all.append(gt)
            split_all.append(split)
            stem_all.append(stem)

    if not z_all:
        print('[dump] ERROR: cached nothing.')
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        z=np.stack(z_all), att=np.stack(att_all),
        thumb=np.stack(thumb_all), gt=np.stack(gt_all),
        split=np.array(split_all), stem=np.array(stem_all),
        grid_n=np.int64(n),
        tau_pos=np.float32(args.tau_pos), tau_neg=np.float32(args.tau_neg),
    )
    print(f'[dump] wrote {len(z_all)} items → {args.out}  '
          f'(grid_n={n}, tau_pos={args.tau_pos}, tau_neg={args.tau_neg})')


if __name__ == '__main__':
    main()
