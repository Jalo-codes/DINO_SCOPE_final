"""contrastive_inpainting_v1.scripts.viz_decode — eyeball the graph decode.

Saves one labelled composite PNG per val splice showing, side by side on the
SAME image:

    Original | BCE Attention | GT Mask | K-means | Graph | Graph+spatial

so you can SEE what the calibrated graph-components decode actually produces
versus k-means and the ground truth — no aggregate numbers, just masks. The
Graph panels are labelled with the decode's own reasoning (#components,
#accepted, abstain) so a blank panel reads as "abstained", not "broken".

Full-frame only (the cleanest view of the decode itself; no windows/zoom).

Usage:
    python -m contrastive_inpainting_v1.scripts.viz_decode \\
        --ckpt /content/drive/MyDrive/DINO_SCOPE_RUNS/<run>/epoch_006.pt \\
        --imd2020_root /content/IMD2020 --casia_root /content/casia \\
        --casia_train --imd_val_only \\
        --tau_pos 0.55 --tau_neg 0.20 \\
        --n_items 24 --out_dir /content/viz_decode_e006
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from lab_utils.data.sampling import deterministic_subsample
from lab_utils.logging.text import install_log, log_line
from lab_utils.train.checkpoint import load as ckpt_load
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.eval.partition import DecodeSpec, decode_deploy_mask
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
    p.add_argument('--n_items', type=int, default=24,
                   help='Number of splices to render PER split.')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--panel_size', type=int, default=280)
    # Graph decode params (tau_pos/tau_neg should match the trained margins).
    p.add_argument('--tau_pos', type=float, default=0.55)
    p.add_argument('--tau_neg', type=float, default=0.20)
    p.add_argument('--graph_s_edge', type=float, default=None)
    p.add_argument('--graph_knn', type=int, default=10)
    p.add_argument('--graph_spatial', type=int, default=2,
                   help='Chebyshev radius for the Graph+spatial panel (0/neg to skip).')
    p.add_argument('--output_log', type=str, default=None)
    return p


def _glabel(name: str, info: dict) -> str:
    """Panel label with the graph decode's own reasoning."""
    if info.get('abstained'):
        return f'{name}\nABSTAIN ({info.get("n_components", 0)} comp)'
    return (f'{name}\n{info.get("n_accepted", 0)}/{info.get("n_components", 0)} '
            f'comp accepted')


def main():
    args = _build_parser().parse_args()
    from contrastive_inpainting_v1.pipeline.cli import apply_path_defaults
    apply_path_defaults(args)
    if args.output_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_log)), exist_ok=True)
        install_log(args.output_log)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    cfg = Config()
    n = cfg.resolution.num_patches_per_side
    T = cfg.resolution.image_size
    normalize = transforms.Normalize(list(cfg.IMAGENET_MEAN), list(cfg.IMAGENET_STD))

    kmeans_spec = DecodeSpec()                     # default = k-means
    graph_spec = DecodeSpec(method='graph', tau_pos=float(args.tau_pos),
                            tau_neg=float(args.tau_neg), s_edge=args.graph_s_edge,
                            mutual_knn_k=int(args.graph_knn))
    do_spatial = args.graph_spatial and int(args.graph_spatial) > 0
    graph_sp_spec = (DecodeSpec(method='graph', tau_pos=float(args.tau_pos),
                                tau_neg=float(args.tau_neg), s_edge=args.graph_s_edge,
                                mutual_knn_k=int(args.graph_knn),
                                r_spatial=int(args.graph_spatial))
                     if do_spatial else None)

    log_line(f'[viz] ckpt={args.ckpt} tau_pos={args.tau_pos} tau_neg={args.tau_neg} '
             f's_edge={args.graph_s_edge} knn={args.graph_knn} '
             f'spatial={args.graph_spatial if do_spatial else "off"} → {args.out_dir}/')

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
        by_split[k] = deterministic_subsample(by_split[k], args.n_items, seed='vizdecode')
    log_line(f'[viz] items: imd={len(by_split["imd_val"])} casia={len(by_split["casia_val"])}')

    # ── model ────────────────────────────────────────────────────────────────
    ckpt = ckpt_load(args.ckpt, map_location=str(device))
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    c_dim = int(sd['contrastive_proj.weight'].shape[0]) if 'contrastive_proj.weight' in sd else 0
    p_hidden = int(sd['pool.V.weight'].shape[0]) if 'pool.V.weight' in sd else 0
    has_patch = 'patch_head.weight' in sd
    if c_dim <= 0:
        log_line('[viz] ERROR: checkpoint has no contrastive head — nothing to decode.')
        return
    model = build_multi_head_detector(
        model_name=cfg.MODEL_NAME, resolution=cfg.resolution,
        lora_rank=cfg.LORA_RANK, lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT, lora_targets=cfg.LORA_TARGETS,
        contrastive_dim=c_dim, pool_hidden=p_hidden, patch_bce=has_patch, device=device)
    model.load_state_dict(sd); model.eval()
    log_line(f'[viz] loaded epoch={ckpt.get("epoch","?")} c_dim={c_dim} pool_hidden={p_hidden}')

    n_saved = 0
    for split, items in by_split.items():
        for idx, it in enumerate(items):
            img_path = str(it.get('img', ''))
            try:
                source = Image.open(img_path).convert('RGB')
            except Exception as exc:
                log_line(f'[viz] WARN load failed {img_path}: {exc}')
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

            inp = normalize(TF.to_tensor(
                TF.resize(source, [T, T], Image.BILINEAR))).unsqueeze(0).to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=(device.type == 'cuda')):
                out = model(inp)
            z = out.get('contrastive')
            if z is None:
                continue
            z_np = z[0].detach().cpu().float().numpy()
            att = out.get('pool_attention')
            att_np = att[0].detach().cpu().float().numpy() if att is not None else None

            panels = [('Original', src_np)]
            if att_np is not None:
                panels.append(('BCE Attention',
                               overlay_blend(src_np, heatmap_rgb(att_np.reshape(n, n), viz_hw))))
            if gt is not None:
                panels.append(('GT Mask', mask_tint(src_np, gt, viz_hw, (0, 255, 0))))

            km_fg, _ = decode_deploy_mask(z_np, kmeans_spec, attention=att_np, grid_hw=(n, n))
            panels.append(('K-means',
                           mask_tint(src_np, km_fg.reshape(n, n), viz_hw, (0, 140, 255))))

            g_fg, g_info = decode_deploy_mask(z_np, graph_spec, attention=att_np, grid_hw=(n, n))
            panels.append((_glabel('Graph', g_info),
                           mask_tint(src_np, g_fg.reshape(n, n), viz_hw, (255, 90, 0))))

            if graph_sp_spec is not None:
                gs_fg, gs_info = decode_deploy_mask(z_np, graph_sp_spec, attention=att_np, grid_hw=(n, n))
                panels.append((_glabel(f'Graph+sp{args.graph_spatial}', gs_info),
                               mask_tint(src_np, gs_fg.reshape(n, n), viz_hw, (255, 0, 200))))

            stem = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(args.out_dir, f'{split}_{idx:03d}_{stem}.png')
            save_composite(panels, save_path, panel_size=int(args.panel_size), cols=6)
            n_saved += 1

    log_line(f'[viz] saved {n_saved} composites to {args.out_dir}/')


if __name__ == '__main__':
    main()
