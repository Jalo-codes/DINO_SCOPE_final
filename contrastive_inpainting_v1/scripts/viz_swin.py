"""contrastive_inpainting_v1.scripts.viz_swin — visualize sliding-window (swin) decode.

Saves composite PNGs showing full-frame vs sliding-window (swin) decodes for
both K-means and Graph-components decodes, so you can see the effect of
window aggregation and BCE gating side by side.
"""

import argparse
import os
import sys
from typing import Dict, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from lab_utils.data.sampling import deterministic_subsample
from lab_utils.train.checkpoint import load as ckpt_load
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.eval.partition import DecodeSpec
from lab_utils.eval.sliding_window import sliding_window_contrastive_masks
from lab_utils.viz import heatmap_rgb, overlay_blend, mask_tint, save_composite

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
    p.add_argument('--n_items', type=int, default=10,
                   help='Number of splices to render PER split.')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--panel_size', type=int, default=280)
    p.add_argument('--scales', type=float, nargs='+', default=[1.0, 0.7, 0.5])
    p.add_argument('--stride', type=float, default=0.5)
    p.add_argument('--bce_gate_threshold', type=float, default=None,
                   help='BCE gating threshold (None to disable, typically 0.0 to enable).')
    # Graph decode params
    p.add_argument('--tau_pos', type=float, default=0.55)
    p.add_argument('--tau_neg', type=float, default=0.20)
    p.add_argument('--graph_s_edge', type=float, default=None)
    p.add_argument('--graph_knn', type=int, default=10)
    p.add_argument('--graph_spatial', type=int, default=None)
    p.add_argument('--graph_theta_w', type=float, default=None)
    p.add_argument('--graph_theta_x', type=float, default=None)
    p.add_argument('--graph_m_min', type=int, default=4)
    return p


def main():
    args = _build_parser().parse_args()
    from contrastive_inpainting_v1.pipeline.cli import apply_path_defaults
    apply_path_defaults(args)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    cfg = Config()
    n = cfg.resolution.num_patches_per_side
    T = cfg.resolution.image_size
    normalize = transforms.Normalize(list(cfg.IMAGENET_MEAN), list(cfg.IMAGENET_STD))

    kmeans_spec = DecodeSpec(method='kmeans')
    graph_spec = DecodeSpec(
        method='graph',
        tau_pos=float(args.tau_pos),
        tau_neg=float(args.tau_neg),
        s_edge=args.graph_s_edge,
        mutual_knn_k=int(args.graph_knn),
        r_spatial=args.graph_spatial,
        m_min=int(args.graph_m_min),
        theta_w=args.graph_theta_w,
        theta_x=args.graph_theta_x
    )

    print(f'[viz_swin] ckpt={args.ckpt} scales={args.scales} stride={args.stride} gate={args.bce_gate_threshold}')

    # ── items ────────────────────────────────────────────────────────────────
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
        by_split[k] = deterministic_subsample(by_split[k], args.n_items, seed='vizswin')
    print(f'[viz_swin] items: imd={len(by_split["imd_val"])} casia={len(by_split["casia_val"])}')

    # ── model ────────────────────────────────────────────────────────────────
    ckpt = ckpt_load(args.ckpt, map_location=str(device))
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    c_dim = int(sd['contrastive_proj.weight'].shape[0]) if 'contrastive_proj.weight' in sd else 0
    p_hidden = int(sd['pool.V.weight'].shape[0]) if 'pool.V.weight' in sd else 0
    has_patch = 'patch_head.weight' in sd
    if c_dim <= 0:
        print('[viz_swin] ERROR: checkpoint has no contrastive head.')
        return
    model = build_multi_head_detector(
        model_name=cfg.MODEL_NAME, resolution=cfg.resolution,
        lora_rank=cfg.LORA_RANK, lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT, lora_targets=cfg.LORA_TARGETS,
        contrastive_dim=c_dim, pool_hidden=p_hidden, patch_bce=has_patch, device=device)
    model.load_state_dict(sd); model.eval()

    n_saved = 0
    for split, items in by_split.items():
        for idx, it in enumerate(items):
            img_path = str(it.get('img', ''))
            stem = os.path.splitext(os.path.basename(img_path))[0]
            try:
                source = Image.open(img_path).convert('RGB')
            except Exception as exc:
                print(f'[viz_swin] WARN load failed {img_path}: {exc}')
                continue
            W, H = source.size
            src_np = np.asarray(source, dtype=np.uint8)
            viz_hw = (H, W)

            gt = None
            try:
                gt_img = Image.open(str(it.get('mask'))).convert('L')
                if gt_img.size != (W, H):
                    gt_img = gt_img.resize((W, H), Image.NEAREST)
                gt = np.asarray(gt_img, dtype=np.uint8) > 0
            except Exception:
                pass

            inp = normalize(TF.to_tensor(TF.resize(source, [T, T], Image.BILINEAR))).to(device)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                    # 1. K-means swin decode
                    km_full, km_swin = sliding_window_contrastive_masks(
                        model, inp, device, n_patch_per_side=n,
                        scales=args.scales, stride_frac=args.stride,
                        decode_spec=kmeans_spec, bce_gate_threshold=args.bce_gate_threshold,
                        source_image=source
                    )

                    # 2. Graph swin decode
                    g_full, g_swin = sliding_window_contrastive_masks(
                        model, inp, device, n_patch_per_side=n,
                        scales=args.scales, stride_frac=args.stride,
                        decode_spec=graph_spec, bce_gate_threshold=args.bce_gate_threshold,
                        source_image=source
                    )

            panels = [
                ('Original', src_np),
                ('GT Mask', mask_tint(src_np, gt, viz_hw, (0, 255, 0)) if gt is not None else src_np),
                ('K-means (Full)', mask_tint(src_np, km_full.reshape(n, n), viz_hw, (0, 140, 255))),
                ('K-means (Swin)', mask_tint(src_np, km_swin.reshape(n, n), viz_hw, (0, 70, 255))),
                ('Graph (Full)', mask_tint(src_np, g_full.reshape(n, n), viz_hw, (255, 90, 0))),
                ('Graph (Swin)', mask_tint(src_np, g_swin.reshape(n, n), viz_hw, (255, 0, 100)))
            ]

            save_path = os.path.join(args.out_dir, f'{split}_{idx:03d}_{stem}.png')
            save_composite(panels, save_path, panel_size=int(args.panel_size), cols=6)
            n_saved += 1
            print(f'[viz_swin] saved composite {n_saved}: {save_path}')

    print(f'[viz_swin] saved {n_saved} composites to {args.out_dir}/')


if __name__ == '__main__':
    main()
