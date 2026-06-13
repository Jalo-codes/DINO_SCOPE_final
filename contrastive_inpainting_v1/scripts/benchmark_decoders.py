"""contrastive_inpainting_v1.scripts.benchmark_decoders — numerical benchmark comparing decoders on IMD2020.

Runs K-means, Graph (plain), and Graph+sp2 decoders on the IMD2020 validation splices,
computing Mean, Median, and Q1/Q3 Quartiles for F1 and IoU metrics.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

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
from lab_utils.eval.partition import DecodeSpec, decode_deploy_mask, spherical_kmeans2
from lab_utils.eval.localization import (
    _patches_to_pixels,
    _load_gt_pixel_mask,
    _mask_metrics,
)
from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec
from contrastive_inpainting_v1.diagnose.polarity import polarity_attn
from contrastive_inpainting_v1.scripts.train_multi_head import _SPLICE_KINDS, _subsample_items


def _infer_arch(state_dict) -> dict:
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


def compute_metrics_stats(vals: List[float]) -> Tuple[float, float, float, float, float]:
    """Returns (mean, median, q1, q3, std)"""
    if not vals:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    arr = np.array(vals, dtype=np.float64)
    mean = float(np.mean(arr))
    med = float(np.median(arr))
    q1, q3 = np.percentile(arr, [25, 75])
    std = float(np.std(arr))
    return mean, med, float(q1), float(q3), std


def main():
    parser = argparse.ArgumentParser(description='Benchmark decoders on IMD2020.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint.')
    parser.add_argument('--imd2020_root', type=str, required=True, help='Path to IMD2020.')
    parser.add_argument('--imd_n', type=int, default=200, help='Max IMD fakes to evaluate.')
    parser.add_argument('--tau_pos', type=float, default=0.45)
    parser.add_argument('--tau_neg', type=float, default=0.40)
    parser.add_argument('--graph_s_edge', type=float, default=0.30)
    parser.add_argument('--graph_knn', type=int, default=10)
    parser.add_argument('--graph_m_min', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    cfg = Config()
    n = cfg.resolution.num_patches_per_side
    psz = cfg.resolution.patch_size

    # Load model
    ckpt = ckpt_load(args.ckpt, map_location=str(device))
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    arch = _infer_arch(sd)
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
    model.eval()

    # Load data
    spec = IMD2020BCESpec(imd2020_root=args.imd2020_root, imd_train=False)
    _, imd_val = spec.build_items(cfg)
    imd_val = [it for it in imd_val if it.get('source', '') == 'imd2020']
    
    # We only benchmark on splice fakes (negatives/reals are excluded)
    imd_splices = [it for it in imd_val if it.get('kind') in _SPLICE_KINDS and it.get('mask')]
    if not imd_splices:
        print("[benchmark] Error: no IMD2020 splices found.")
        return
        
    imd_cap = _subsample_items(imd_splices, args.imd_n, seed='benchmark')
    print(f"[benchmark] Benchmarking on {len(imd_cap)} splice items from IMD2020.")

    ds = LabDataset(
        imd_cap, cfg.resolution,
        cfg.IMAGENET_MEAN, cfg.IMAGENET_STD,
        augment=False,
        use_degradation=False, use_invariance=False,
        use_splice_degradation=False,
        gt_patch_threshold=0.06,
    )
    loader = build_eval_loader(ds, LoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    ))

    # Setup decoders
    kmeans_spec = DecodeSpec(method='kmeans')
    graph_spec = DecodeSpec(
        method='graph',
        tau_pos=args.tau_pos,
        tau_neg=args.tau_neg,
        s_edge=args.graph_s_edge,
        mutual_knn_k=args.graph_knn,
        m_min=args.graph_m_min,
    )
    graph_sp_spec = DecodeSpec(
        method='graph',
        tau_pos=args.tau_pos,
        tau_neg=args.tau_neg,
        s_edge=args.graph_s_edge,
        mutual_knn_k=args.graph_knn,
        r_spatial=2,
        m_min=args.graph_m_min,
    )

    results = {
        'K-means': {'f1': [], 'iou': []},
        'Graph': {'f1': [], 'iou': []},
        'Graph+sp2': {'f1': [], 'iou': []},
    }

    n_processed = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            img = batch['img'].to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                out = model(img)
            
            z_b = out.get('contrastive')
            if z_b is None:
                print("[benchmark] Error: checkpoint has no contrastive head.")
                return
            z_b = z_b.detach().cpu().float().numpy()
            
            att_b = out.get('pool_attention')
            att_b = att_b.detach().cpu().float().numpy() if att_b is not None else None
            
            meta_list = batch['meta'] if isinstance(batch['meta'], list) else [
                {k: v[i] for k, v in batch['meta'].items()} for i in range(img.shape[0])
            ]
            
            for i in range(len(z_b)):
                meta = meta_list[i]
                gt_px = _load_gt_pixel_mask(meta, cfg.resolution)
                if gt_px is None:
                    continue
                
                z = z_b[i]
                att = att_b[i] if att_b is not None else None
                
                # 1. K-Means
                km_fg, _ = decode_deploy_mask(z, kmeans_spec, attention=att, grid_hw=(n, n))
                pred_px_km = _patches_to_pixels(km_fg.reshape(-1).astype(np.float64), n, psz)
                f1_km, iou_km, _, _, _ = _mask_metrics(pred_px_km, gt_px)
                results['K-means']['f1'].append(f1_km)
                results['K-means']['iou'].append(iou_km)
                
                # 2. Graph (plain)
                g_fg, _ = decode_deploy_mask(z, graph_spec, attention=att, grid_hw=(n, n))
                pred_px_g = _patches_to_pixels(g_fg.reshape(-1).astype(np.float64), n, psz)
                f1_g, iou_g, _, _, _ = _mask_metrics(pred_px_g, gt_px)
                results['Graph']['f1'].append(f1_g)
                results['Graph']['iou'].append(iou_g)
                
                # 3. Graph+sp2
                g_sp2_fg, _ = decode_deploy_mask(z, graph_sp_spec, attention=att, grid_hw=(n, n))
                pred_px_gsp2 = _patches_to_pixels(g_sp2_fg.reshape(-1).astype(np.float64), n, psz)
                f1_gsp2, iou_gsp2, _, _, _ = _mask_metrics(pred_px_gsp2, gt_px)
                results['Graph+sp2']['f1'].append(f1_gsp2)
                results['Graph+sp2']['iou'].append(iou_gsp2)
                
                n_processed += 1

    print(f"\n[benchmark] Completed evaluation on {n_processed} items.")
    print("=" * 70)
    print(f"{'Decoder':<12} | {'Metric':<6} | {'Mean':<8} | {'Median':<8} | {'Q1 (25%)':<8} | {'Q3 (75%)':<8} | {'SD':<8}")
    print("-" * 70)
    
    for decoder in ('K-means', 'Graph', 'Graph+sp2'):
        for metric in ('f1', 'iou'):
            mean, med, q1, q3, std = compute_metrics_stats(results[decoder][metric])
            dec_label = decoder if metric == 'f1' else ""
            print(f"{dec_label:<12} | {metric.upper():<6} | {mean:.4f} | {med:.4f} | {q1:.4f} | {q3:.4f} | {std:.4f}")
        print("-" * 70)


if __name__ == '__main__':
    main()
