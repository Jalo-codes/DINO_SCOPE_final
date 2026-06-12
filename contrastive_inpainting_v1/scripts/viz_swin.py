"""contrastive_inpainting_v1.scripts.viz_swin — visualize sliding-window (swin) decode.

Saves composite PNGs showing:
1. A summary comparing full-frame vs aggregated sliding-window (swin) decodes for
   both K-means and Graph-components at different scales.
2. For each scale, a detailed composite showing each individual sub-window's
   crop, attention map, K-means mask, and Graph components.
"""

import argparse
import os
import sys
import math
from typing import Dict, Tuple, List

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
from lab_utils.eval.partition import DecodeSpec, decode_deploy_mask, spherical_kmeans2
from lab_utils.eval.sliding_window import _square_crop_boxes
from lab_utils.viz import heatmap_rgb, overlay_blend, mask_tint, save_composite
from contrastive_inpainting_v1.diagnose.polarity import polarity_attn

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


def _kmeans_panel(z_np: np.ndarray, att_np, n: int):
    """(N, D) embeddings → boolean (n, n) splice mask via spherical kmeans,
    polarity set by the image-head attention (hot cluster = splice)."""
    raw_labels, _ = spherical_kmeans2(z_np, n_init=4)
    if att_np is not None:
        km = polarity_attn(raw_labels, att_np)
    else:
        km = raw_labels if raw_labels.sum() <= len(raw_labels) / 2 else 1 - raw_labels
    return km.reshape(n, n).astype(bool)


def multi_mask_tint(
    base: np.ndarray,
    labels_2d: np.ndarray,
    size_hw: Tuple[int, int],
    color_map: Dict[int, Tuple[int, int, int]],
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a multi-label 2-D mask as colored tints on the original image."""
    labels_up = np.round(np.asarray(
        Image.fromarray(labels_2d.astype(np.float32)).resize(
            (size_hw[1], size_hw[0]), Image.NEAREST
        )
    )).astype(np.int32)
    out = base.copy()
    for label, color in color_map.items():
        mask = (labels_up == label)
        if not mask.any():
            continue
        c = np.array(color, dtype=np.float32)
        out[mask] = np.clip(
            (1 - alpha) * base[mask].astype(np.float32) + alpha * c, 0, 255
        ).astype(np.uint8)
    return out


def project_win_to_tgt(wm, top, left, side, H_src, W_src, T, n):
    """Project (n, n) window mask back to the (n, n) target grid."""
    agg = np.zeros((n, n), dtype=bool)
    target_patch_size = T // n
    win_patch_size_src = side / float(n)
    
    tgt_top_pix    = top  * T / float(H_src)
    tgt_bot_pix    = (top  + side) * T / float(H_src)
    tgt_left_pix   = left * T / float(W_src)
    tgt_right_pix  = (left + side) * T / float(W_src)
    
    tgt_top_patch    = max(0, int(math.floor(tgt_top_pix    / target_patch_size)))
    tgt_bot_patch    = min(n, int(math.ceil (tgt_bot_pix    / target_patch_size)))
    tgt_left_patch   = max(0, int(math.floor(tgt_left_pix   / target_patch_size)))
    tgt_right_patch  = min(n, int(math.ceil (tgt_right_pix  / target_patch_size)))

    for ti in range(tgt_top_patch, tgt_bot_patch):
        src_y_pix = (ti + 0.5) * target_patch_size * H_src / float(T)
        wi = int((src_y_pix - top) / win_patch_size_src)
        if wi < 0 or wi >= n:
            continue
        for tj in range(tgt_left_patch, tgt_right_patch):
            src_x_pix = (tj + 0.5) * target_patch_size * W_src / float(T)
            wj = int((src_x_pix - left) / win_patch_size_src)
            if wj < 0 or wj >= n:
                continue
            if wm[wi, wj]:
                agg[ti, tj] = True
    return agg


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

            # Full Image Forward
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                    out = model(inp)
                    
            z_full = out.get('contrastive')[0].detach().cpu().float().numpy()
            att_full = out.get('pool_attention')
            att_full_np = att_full[0].detach().cpu().float().numpy() if att_full is not None else None
            
            # Full image decodes
            km_full = _kmeans_panel(z_full, att_full_np, n)
            g_full, g_full_info = decode_deploy_mask(z_full, graph_spec, attention=att_full_np, grid_hw=(n, n))
            g_full = g_full.reshape(n, n)
            
            # We will accumulate aggregated masks per scale
            swin_results = {}
            
            # For each scale (skipping full-frame 1.0 which is analyzed directly)
            sub_scales = [s for s in args.scales if s < 1.0]
            if not sub_scales:
                sub_scales = [0.7] # default fallback
                
            for scale in sub_scales:
                # Get square crop boxes in source resolution space
                crop_boxes = _square_crop_boxes(H, W, [scale], args.stride)
                
                km_agg = np.zeros((n, n), dtype=bool)
                g_agg = np.zeros((n, n), dtype=bool)
                
                win_panels = []
                
                for k, (top, left, side) in enumerate(crop_boxes):
                    # 1. Crop sub-image
                    crop_pil = source.crop((left, top, left + side, top + side))
                    crop_resized = crop_pil.resize((T, T), Image.BILINEAR)
                    crop_np = np.asarray(crop_resized, dtype=np.uint8)
                    
                    # Forward pass
                    crop_t = normalize(TF.to_tensor(crop_resized)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                            enabled=(device.type == 'cuda')):
                            out_win = model(crop_t)
                            
                    z_win = out_win.get('contrastive')[0].detach().cpu().float().numpy()
                    att_win = out_win.get('pool_attention')
                    att_win_np = att_win[0].detach().cpu().float().numpy() if att_win is not None else None
                    logit_win = out_win.get('image_logit')
                    logit_val = float(torch.sigmoid(logit_win[0]).item()) if logit_win is not None else 1.0
                    
                    # Decodes
                    km_win = _kmeans_panel(z_win, att_win_np, n)
                    g_win, g_win_info = decode_deploy_mask(z_win, graph_spec, attention=att_win_np, grid_hw=(n, n))
                    g_win = g_win.reshape(n, n)
                    
                    # Check gating
                    is_gated = (args.bce_gate_threshold is not None and logit_win is not None and float(logit_win[0].item()) <= args.bce_gate_threshold)
                    
                    if not is_gated:
                        # Project back to full image coords
                        km_proj = project_win_to_tgt(km_win, top, left, side, H, W, T, n)
                        g_proj = project_win_to_tgt(g_win, top, left, side, H, W, T, n)
                        km_agg |= km_proj
                        g_agg |= g_proj
                        
                    # Panels for this window:
                    # 1. Crop original
                    win_panels.append((f'Win {k} Crop\n[y={top}, x={left}]', crop_np))
                    # 2. Attention
                    if att_win_np is not None:
                        heat = heatmap_rgb(att_win_np.reshape(n, n), (T, T))
                        attn_overlay = overlay_blend(crop_np, heat)
                        p_str = f' p={logit_val:.2f}' if logit_win is not None else ''
                        gate_status = 'GATED OUT' if is_gated else 'ACTIVE'
                        win_panels.append((f'Win {k} Attn{p_str}\n({gate_status})', attn_overlay))
                    else:
                        win_panels.append(('', np.zeros_like(crop_np)))
                        
                    # 3. K-means
                    color_km = (255, 0, 0) if is_gated else (0, 140, 255)
                    km_overlay = mask_tint(crop_np, km_win, (T, T), color_km)
                    win_panels.append((f'Win {k} K-means\n{"Gated" if is_gated else "Active"}', km_overlay))
                    
                    # 4. Graph
                    if is_gated:
                        g_overlay = mask_tint(crop_np, g_win, (T, T), (255, 0, 0))
                    else:
                        color_map = {}
                        for comp in g_win_info.get('components', []):
                            cid = comp['comp_id']
                            color_map[cid] = (0, 255, 0) if comp['accepted'] else (255, 0, 0)
                        
                        labels_np = g_win_info.get('labels')
                        bg_id = g_win_info.get('background_id')
                        m_min = g_win_info.get('m_min', 4)
                        
                        comp_ids, comp_sizes = np.unique(labels_np, return_counts=True)
                        for cid, sz in zip(comp_ids, comp_sizes):
                            if cid != bg_id and sz < m_min:
                                color_map[int(cid)] = (120, 120, 120)
                                
                        g_overlay = multi_mask_tint(crop_np, labels_np.reshape(n, n), (T, T), color_map)
                    
                    n_ac = g_win_info.get('n_accepted', 0)
                    n_co = g_win_info.get('n_components', 0)
                    lbl_g = f'Win {k} Graph\n{n_ac}/{n_co} accepted'
                    if is_gated:
                        lbl_g += ' (Gated)'
                    win_panels.append((lbl_g, g_overlay))
                    
                # Save detailed composite for this scale
                detail_path = os.path.join(args.out_dir, f'{split}_{idx:03d}_{stem}_scale_{scale:.2f}.png')
                save_composite(win_panels, detail_path, panel_size=int(args.panel_size), cols=4)
                print(f'[viz_swin] saved details for scale {scale:.2f} to {detail_path}')
                
                swin_results[scale] = (km_agg, g_agg)
                
            # Now build summary composite
            summary_panels = [
                ('Original', src_np),
                ('GT Mask', mask_tint(src_np, gt, viz_hw, (0, 255, 0)) if gt is not None else src_np),
                ('K-means (Full)', mask_tint(src_np, km_full, viz_hw, (0, 140, 255))),
                ('Graph (Full)', mask_tint(src_np, g_full, viz_hw, (255, 90, 0)))
            ]
            
            for scale in sub_scales:
                km_agg, g_agg = swin_results[scale]
                summary_panels.append((f'K-means Swin ({scale:.2f})', mask_tint(src_np, km_agg, viz_hw, (0, 70, 255))))
                summary_panels.append((f'Graph Swin ({scale:.2f})', mask_tint(src_np, g_agg, viz_hw, (255, 0, 100))))
                
            summary_path = os.path.join(args.out_dir, f'{split}_{idx:03d}_{stem}_summary.png')
            save_composite(summary_panels, summary_path, panel_size=int(args.panel_size), cols=4)
            n_saved += 1
            print(f'[viz_swin] saved summary composite {n_saved}: {summary_path}')

    print(f'[viz_swin] saved {n_saved} composites to {args.out_dir}/')


if __name__ == '__main__':
    main()
