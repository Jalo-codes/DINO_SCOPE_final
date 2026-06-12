"""swin: every (scale, stride) combo produces an UNCAPPED set of windows.

For each window:
  - Forward the model on the 448x448 resized crop.
  - Get image-level BCE logit + per-patch contrastive + pool_attention.
  - Compute splice_frac inside the window (from pixel GT — diagnostic only).
  - Categorize:
      clean_pos   : bce_logit >= tau_win AND splice_frac >= 0.5
      mixed_pos   : bce_logit >= tau_win AND 0 <  splice_frac < 0.5
      false_pos   : bce_logit >= tau_win AND splice_frac == 0
      missed_pos  : bce_logit <  tau_win AND splice_frac > 0
      clean_neg   : bce_logit <  tau_win AND splice_frac == 0
  - For BCE-positive windows: spherical_kmeans2 on per-patch contrastive,
    apply attn polarity, project to (H, W). OR across BCE-positive windows
    yields the swin prediction.
  - polarity_agreement: across overlapping BCE-positive windows, fraction of
    source-pixel area where their projected predictions agree (TRUE/FALSE
    consistent). Stride bug pin: drops at stride=0.5 means OR is noisy.

Returns a dict keyed by `swin_s{SS}_t{TT}`:
  {
    'swin_s055_t10_pure_mask':       (H, W) bool   (OR over BCE-positive windows)
    'swin_s055_t10_n_windows':       int
    'swin_s055_t10_n_bce_pos':       int
    'swin_s055_t10_window_set_hash': str
    'swin_s055_t10_polarity_agreement': float
    'swin_s055_t10_bce_logit_max':   float
    'swin_s055_t10_bce_logit_mean':  float
    'swin_s055_t10_per_window':      list[dict]  (one entry per window; for [bce_win] summary)
  }

Each per_window entry has:
  {
    'top','left','side','bce_logit','splice_frac','category','is_bce_pos'
  }
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from lab_utils.eval.partition import spherical_kmeans2

from .. import polarity, project
from . import common


def swin_key(scale: float, stride_frac: float) -> str:
    return f"s{int(round(scale * 100)):03d}_t{int(round(stride_frac * 10)):02d}"


def _category(bce_logit: float, splice_frac: float, *, tau_win: float) -> str:
    is_pos = float(bce_logit) >= float(tau_win)
    if is_pos and splice_frac >= 0.5:
        return "clean_pos"
    if is_pos and splice_frac > 0.0:
        return "mixed_pos"
    if is_pos and splice_frac == 0.0:
        return "false_pos"
    if (not is_pos) and splice_frac > 0.0:
        return "missed_pos"
    return "clean_neg"


@torch.no_grad()
def run_swin(
    model,
    source_image: Image.Image,
    device: torch.device,
    *,
    scale: float,
    stride_frac: float,
    image_size: int,
    n_patch_per_side: int,
    imagenet_mean,
    imagenet_std,
    tau_win: float,
    gt_HW: np.ndarray = None,
    decode_spec=None,
) -> Dict:
    """Run swin for one (scale, stride) combo."""
    W_src, H_src = source_image.size
    H, W = int(H_src), int(W_src)
    n = int(n_patch_per_side)

    windows = common.window_grid(
        source_size=(W_src, H_src),
        scale=scale, stride_frac=stride_frac, n_patch_per_side=n,
    )
    set_hash = common.window_set_hash(windows)

    per_window: List[Dict] = []
    bce_pos_pred_HW: List[np.ndarray] = []
    bce_pos_indices: List[int] = []

    for k, (top, left, side, _) in enumerate(windows):
        # Forward on the crop.
        t = common.crop_resize_to_tensor(
            source_image, bbox_top=top, bbox_left=left, bbox_side=side,
            target_size=image_size,
            imagenet_mean=imagenet_mean, imagenet_std=imagenet_std,
        ).to(device, non_blocking=True)
        out = model(t)
        bce_logit = (float(out["image_logit"][0].detach().cpu().item())
                     if out.get("image_logit") is not None else float("nan"))

        # Compute splice_frac inside the window (pixel-level, from gt_HW).
        if gt_HW is not None:
            sub = gt_HW[top:top + side, left:left + side]
            splice_frac = float(sub.mean()) if sub.size else 0.0
        else:
            splice_frac = 0.0

        category = _category(bce_logit, splice_frac, tau_win=tau_win)
        is_pos = (float(bce_logit) >= float(tau_win))

        entry = {
            "k": int(k),
            "top": int(top), "left": int(left), "side": int(side),
            "bce_logit": float(bce_logit),
            "splice_frac": float(splice_frac),
            "category": str(category),
            "is_bce_pos": bool(is_pos),
        }

        # Only partition + project for BCE-positive windows (saves compute).
        if is_pos and out.get("contrastive") is not None:
            z = out["contrastive"][0].detach().cpu().float().numpy()
            att = (out["pool_attention"][0].detach().cpu().float().numpy()
                   if out.get("pool_attention") is not None else None)
            if decode_spec is not None and getattr(decode_spec, 'method', 'kmeans') == 'graph':
                from lab_utils.eval.partition import decode_deploy_mask
                attn_pred, _ = decode_deploy_mask(z, decode_spec, attention=att,
                                                  grid_hw=(n, n))
            else:
                raw, _ = spherical_kmeans2(z, n_init=4)
                attn_pred = polarity.polarity_attn(raw, att)
            pred_grid = np.asarray(attn_pred).reshape(n, n)
            pred_HW = project.patch_grid_to_pixel_mask(
                pred_grid, bbox=(top, left, side, side), full_size=(H, W),
            )
            bce_pos_pred_HW.append(pred_HW)
            bce_pos_indices.append(int(k))

        per_window.append(entry)

    # OR across BCE-positive windows.
    if bce_pos_pred_HW:
        pure_mask = np.logical_or.reduce(np.asarray(bce_pos_pred_HW, dtype=bool),
                                          axis=0)
    else:
        pure_mask = np.zeros((H, W), dtype=bool)

    # polarity_agreement: among overlapping pairs of BCE-positive windows,
    # fraction of overlap area where their predictions match.
    pol_agree = _polarity_agreement(bce_pos_pred_HW, windows, bce_pos_indices,
                                     H=H, W=W)

    logits = [w["bce_logit"] for w in per_window]
    pos_logits = [w["bce_logit"] for w in per_window if w["is_bce_pos"]]

    key = swin_key(scale, stride_frac)
    return {
        f"swin_{key}_pure_mask": pure_mask,
        f"swin_{key}_n_windows": int(len(windows)),
        f"swin_{key}_n_bce_pos": int(len(bce_pos_pred_HW)),
        f"swin_{key}_window_set_hash": str(set_hash),
        f"swin_{key}_polarity_agreement": float(pol_agree),
        f"swin_{key}_bce_logit_max": float(max(logits)) if logits else float("nan"),
        f"swin_{key}_bce_logit_mean": float(np.mean(logits)) if logits else float("nan"),
        f"swin_{key}_bce_logit_max_pos": float(max(pos_logits)) if pos_logits else float("nan"),
        f"swin_{key}_bce_logit_mean_pos": float(np.mean(pos_logits)) if pos_logits else float("nan"),
        f"swin_{key}_per_window": per_window,
        f"swin_{key}_scale": float(scale),
        f"swin_{key}_stride_frac": float(stride_frac),
    }


def _polarity_agreement(
    pred_HWs: List[np.ndarray],
    windows: List[Tuple[int, int, int, int]],
    bce_pos_indices: List[int],
    *,
    H: int,
    W: int,
) -> float:
    """Across all overlapping BCE-positive window PAIRS, fraction of overlap-
    pixels where both windows' predictions agree.

    If there are no overlapping BCE-positive window pairs, returns 1.0
    (trivially consistent — no contradiction possible).
    """
    if len(pred_HWs) < 2:
        return 1.0
    # For each pair, find overlap region in source pixels.
    total_overlap = 0
    total_agree = 0
    for i in range(len(pred_HWs)):
        ti, li, si, _ = windows[bce_pos_indices[i]]
        for j in range(i + 1, len(pred_HWs)):
            tj, lj, sj, _ = windows[bce_pos_indices[j]]
            top = max(ti, tj)
            left = max(li, lj)
            bot = min(ti + si, tj + sj)
            right = min(li + si, lj + sj)
            if bot <= top or right <= left:
                continue
            a = pred_HWs[i][top:bot, left:right]
            b = pred_HWs[j][top:bot, left:right]
            n_pixels = int(a.size)
            n_agree = int((a == b).sum())
            total_overlap += n_pixels
            total_agree += n_agree
    if total_overlap == 0:
        return 1.0
    return float(total_agree) / float(total_overlap)
