"""graph_lab.sandbox — twist the graph-decode knobs on CACHED real embeddings.

No model, no GPU, no dataset. Loads a ``.npz`` produced by ``dump_embeddings.py``
and re-runs ``graph_components_decode`` under whatever parameters you give it,
writing labelled PNG composites so you can SEE where the graph connection helps.

Two modes:

  Single  — one parameter set, one composite per image:
              Original | GT | K-means | Graph | Graph+spatial
            with each Graph panel coloured by component
            (green=accepted, red=rejected, gray=sub-m_min) and labelled with
            its own reasoning (#accepted/#components or ABSTAIN) and IoU vs GT.

  Sweep   — vary ONE knob across a range (--sweep s_edge --sweep_vals 0.30 0.35 0.40):
            one composite per image with Original | GT | <a panel per value>,
            plus a stdout table of median IoU / abstain-rate at each setting so
            the knee is obvious.

Examples:
    # single setting, all the knobs explicit
    python -m graph_lab.sandbox --cache graph_lab/cache/e006.npz \\
        --out graph_lab/out/baseline --s_edge 0.375 --knn 10 --spatial 2

    # sweep the edge threshold and find where precision turns on
    python -m graph_lab.sandbox --cache graph_lab/cache/e006.npz \\
        --out graph_lab/out/sweep_sedge --sweep s_edge \\
        --sweep_vals 0.30 0.34 0.38 0.42 0.46
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
from PIL import Image

from lab_utils.eval.partition import DecodeSpec, decode_deploy_mask
from lab_utils.viz import heatmap_rgb, overlay_blend, mask_tint, save_composite

# Knobs the sweep mode can vary (name → DecodeSpec field).
_SWEEPABLE = ('s_edge', 'mutual_knn_k', 'r_spatial', 'm_min', 'theta_w', 'theta_x',
              'tau_pos', 'tau_neg')


# ── small viz helpers (shared shape with scripts/viz_decode) ───────────────────

def _multi_mask_tint(base, labels_2d, size_hw, color_map, alpha=0.45):
    labels_up = np.round(np.asarray(
        Image.fromarray(labels_2d.astype(np.float32)).resize(
            (size_hw[1], size_hw[0]), Image.NEAREST))).astype(np.int32)
    out = base.copy()
    for label, color in color_map.items():
        m = (labels_up == label)
        if not m.any():
            continue
        c = np.array(color, dtype=np.float32)
        out[m] = np.clip((1 - alpha) * base[m].astype(np.float32) + alpha * c, 0, 255).astype(np.uint8)
    return out


def _component_overlay(base, info, grid_n, size_hw):
    """Tint every component: green=accepted, red=rejected, gray=sub-m_min."""
    labels = info.get('labels')
    if labels is None:
        return base
    bg_id = info.get('background_id')
    m_min = info.get('m_min', 4)
    color_map = {}
    for comp in info.get('components', []):
        color_map[comp['comp_id']] = (0, 255, 0) if comp['accepted'] else (255, 0, 0)
    ids, sizes = np.unique(labels, return_counts=True)
    for cid, sz in zip(ids.tolist(), sizes.tolist()):
        if cid != bg_id and sz < m_min:
            color_map[int(cid)] = (120, 120, 120)
    return _multi_mask_tint(base, labels.reshape(grid_n, grid_n), size_hw, color_map)


def _iou(mask_flat, grid_n, gt_2d):
    """IoU of a flat patch mask (upsampled to GT res) against GT bool mask."""
    pred = np.asarray(
        Image.fromarray((mask_flat.reshape(grid_n, grid_n).astype(np.uint8) * 255)).resize(
            (gt_2d.shape[1], gt_2d.shape[0]), Image.NEAREST), dtype=np.uint8) > 127
    inter = int((pred & gt_2d).sum())
    union = int((pred | gt_2d).sum())
    return (inter / union) if union > 0 else (1.0 if pred.sum() == 0 else 0.0)


def _glabel(name, info, iou):
    if info.get('abstained'):
        return f'{name}\nABSTAIN  IoU={iou:.2f}'
    return (f'{name}\n{info.get("n_accepted", 0)}/{info.get("n_components", 0)} comp'
            f'  IoU={iou:.2f}')


def _spec(args, **overrides) -> DecodeSpec:
    base = dict(
        method='graph',
        tau_pos=float(args.tau_pos), tau_neg=float(args.tau_neg),
        s_edge=args.s_edge, mutual_knn_k=int(args.knn),
        m_min=int(args.m_min), theta_w=args.theta_w, theta_x=args.theta_x,
    )
    base.update(overrides)
    return DecodeSpec(**base)


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True, help='.npz from dump_embeddings.py')
    p.add_argument('--out', required=True, help='Output dir for composites.')
    p.add_argument('--panel_size', type=int, default=300)
    # Decode knobs (None → graph_components_decode's calibrated defaults).
    p.add_argument('--tau_pos', type=float, default=None, help='Default: from cache.')
    p.add_argument('--tau_neg', type=float, default=None, help='Default: from cache.')
    p.add_argument('--s_edge', type=float, default=None, help='Edge sim bar. Default mid-band.')
    p.add_argument('--knn', type=int, default=10, help='Mutual-kNN k.')
    p.add_argument('--spatial', type=int, default=2,
                   help='Chebyshev radius for the Graph+spatial panel (single mode; 0=skip).')
    p.add_argument('--m_min', type=int, default=4)
    p.add_argument('--theta_w', type=float, default=None, help='Cohesion floor. Default tau_pos-0.05.')
    p.add_argument('--theta_x', type=float, default=None, help='Sim-to-bg ceiling. Default mid-band.')
    # Sweep mode.
    p.add_argument('--sweep', choices=_SWEEPABLE, default=None,
                   help='Vary ONE knob across --sweep_vals (overrides single mode).')
    p.add_argument('--sweep_vals', type=float, nargs='+', default=None)
    return p


def main():
    args = _build_parser().parse_args()
    d = np.load(args.cache, allow_pickle=True)
    grid_n = int(d['grid_n'])
    if args.tau_pos is None:
        args.tau_pos = float(d['tau_pos'])
    if args.tau_neg is None:
        args.tau_neg = float(d['tau_neg'])
    os.makedirs(args.out, exist_ok=True)

    z, att, thumb, gt = d['z'], d['att'], d['thumb'], d['gt']
    split, stem = d['split'], d['stem']
    K = z.shape[0]
    P = thumb.shape[1]
    viz_hw = (P, P)
    km_spec = DecodeSpec()  # default = k-means reference

    print(f'[sandbox] {args.cache}: {K} items grid_n={grid_n} '
          f'tau_pos={args.tau_pos} tau_neg={args.tau_neg}')

    # ── SWEEP mode ─────────────────────────────────────────────────────────────
    if args.sweep is not None:
        if not args.sweep_vals:
            print('[sandbox] ERROR: --sweep requires --sweep_vals.')
            return
        vals = args.sweep_vals
        cast = int if args.sweep in ('mutual_knn_k', 'r_spatial', 'm_min') else float
        specs = [(v, _spec(args, **{args.sweep: cast(v)})) for v in vals]
        print(f'[sandbox] sweep {args.sweep} over {[cast(v) for v in vals]}')

        # IoU table accumulators: per value → list of IoUs, abstain count.
        agg: Dict[float, List[float]] = {v: [] for v in vals}
        abst: Dict[float, int] = {v: 0 for v in vals}

        for i in range(K):
            panels = [('Original', thumb[i]),
                      ('GT', mask_tint(thumb[i], gt[i], viz_hw, (0, 255, 0)))]
            for v, sp in specs:
                fg, info = decode_deploy_mask(z[i], sp, attention=att[i], grid_hw=(grid_n, grid_n))
                iou = _iou(fg, grid_n, gt[i])
                agg[v].append(iou)
                abst[v] += int(bool(info.get('abstained')))
                panels.append((_glabel(f'{args.sweep}={cast(v)}', info, iou),
                               _component_overlay(thumb[i], info, grid_n, viz_hw)))
            save_composite(panels, os.path.join(args.out, f'{split[i]}_{i:03d}_{stem[i]}.png'),
                           panel_size=int(args.panel_size), cols=len(panels))

        print(f'\n[sandbox] sweep summary ({K} items)')
        print(f'  {args.sweep:>12} | median IoU | mean IoU | abstain')
        print('  ' + '-' * 48)
        for v in vals:
            arr = np.array(agg[v])
            print(f'  {cast(v)!s:>12} |    {np.median(arr):.3f}  |  {arr.mean():.3f} |'
                  f'  {abst[v]}/{K}')
        print(f'\n[sandbox] wrote {K} composites → {args.out}/')
        return

    # ── SINGLE mode ────────────────────────────────────────────────────────────
    g_spec = _spec(args)
    gs_spec = _spec(args, r_spatial=int(args.spatial)) if args.spatial and args.spatial > 0 else None

    km_ious, g_ious = [], []
    for i in range(K):
        panels = [('Original', thumb[i]),
                  ('GT', mask_tint(thumb[i], gt[i], viz_hw, (0, 255, 0)))]

        km_fg, _ = decode_deploy_mask(z[i], km_spec, attention=att[i], grid_hw=(grid_n, grid_n))
        km_iou = _iou(km_fg, grid_n, gt[i]); km_ious.append(km_iou)
        panels.append((f'K-means\nIoU={km_iou:.2f}',
                       mask_tint(thumb[i], km_fg.reshape(grid_n, grid_n), viz_hw, (0, 140, 255))))

        g_fg, g_info = decode_deploy_mask(z[i], g_spec, attention=att[i], grid_hw=(grid_n, grid_n))
        g_iou = _iou(g_fg, grid_n, gt[i]); g_ious.append(g_iou)
        panels.append((_glabel('Graph', g_info, g_iou),
                       _component_overlay(thumb[i], g_info, grid_n, viz_hw)))

        if gs_spec is not None:
            gs_fg, gs_info = decode_deploy_mask(z[i], gs_spec, attention=att[i], grid_hw=(grid_n, grid_n))
            gs_iou = _iou(gs_fg, grid_n, gt[i])
            panels.append((_glabel(f'Graph+sp{args.spatial}', gs_info, gs_iou),
                           _component_overlay(thumb[i], gs_info, grid_n, viz_hw)))

        save_composite(panels, os.path.join(args.out, f'{split[i]}_{i:03d}_{stem[i]}.png'),
                       panel_size=int(args.panel_size), cols=len(panels))

    km, g = np.array(km_ious), np.array(g_ious)
    print(f'\n[sandbox] {K} items — median IoU:  k-means={np.median(km):.3f}  '
          f'graph={np.median(g):.3f}   (mean {km.mean():.3f} vs {g.mean():.3f})')
    print(f'[sandbox] graph wins on {int((g > km).sum())}/{K}, ties {int((g == km).sum())}')
    print(f'[sandbox] wrote {K} composites → {args.out}/')


if __name__ == '__main__':
    main()
