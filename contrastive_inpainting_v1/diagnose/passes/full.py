"""full_pure + full_ceil: one forward at 448x448 on the whole image.

Both rules share the same forward + same k-means partition; only the cluster-
identity selection differs.

Returns a dict the caller drops into the row:
  {
    'full_pure_mask':      (H, W) bool   pixel mask (NN-expanded patch grid)
    'full_pure_inverted':  bool          attn rule picked the larger cluster?

    # ceil only if gt_HW is not None:
    'full_ceil_mask':      (H, W) bool
    'full_ceil_inverted':  bool

    'full_bce_logit':      float         image-level BCE logit
    'full_pool_attention_mean':  float   mean pool_attention (diagnostic)
  }
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from lab_utils.eval.partition import DecodeSpec, decode_oracle_labels

from .. import polarity, project
from . import common


@torch.no_grad()
def run_full(
    model,
    source_image: Image.Image,
    device: torch.device,
    *,
    image_size: int,
    n_patch_per_side: int,
    imagenet_mean,
    imagenet_std,
    gt_HW: Optional[np.ndarray] = None,
    decode_spec=None,
) -> Dict:
    """One forward at 448x448, partition, project both polarities to pixels."""
    W_src, H_src = source_image.size  # PIL is (W, H)
    H, W = int(H_src), int(W_src)

    t = common.full_image_to_tensor(
        source_image,
        target_size=image_size,
        imagenet_mean=imagenet_mean,
        imagenet_std=imagenet_std,
    ).to(device, non_blocking=True)
    out = model(t)
    if out.get("contrastive") is None:
        raise RuntimeError("full pass requires contrastive head in model output")

    z = out["contrastive"][0].detach().cpu().float().numpy()
    att = (out["pool_attention"][0].detach().cpu().float().numpy()
           if out.get("pool_attention") is not None else None)
    bce_logit = (float(out["image_logit"][0].detach().cpu().item())
                 if out.get("image_logit") is not None else float("nan"))

    raw = decode_oracle_labels(z, decode_spec or DecodeSpec())
    n = int(n_patch_per_side)

    # Build patch-grid GT (used only by ceil — no leakage to the row's
    # canonical GT, which lives at pixel resolution in gt_HW).
    gt_flat: Optional[np.ndarray] = None
    if gt_HW is not None:
        # Average-pool pixel GT to patch grid for the polarity oracle.
        # This is patch-grid GT, not pixel — only used for cluster-flip decision.
        patch_h = max(1, H // n)
        patch_w = max(1, W // n)
        # Build a coarse (n, n) by sampling: faster than full reshape on odd
        # sizes. Use the centre pixel of each patch cell.
        ys = np.minimum((np.arange(n) * H // n) + patch_h // 2, H - 1)
        xs = np.minimum((np.arange(n) * W // n) + patch_w // 2, W - 1)
        gt_grid = gt_HW[ys[:, None], xs[None, :]]
        gt_flat = gt_grid.reshape(-1).astype(bool)

    variants = polarity.both_variants(raw, att, gt_flat)

    result: Dict = {
        "full_bce_logit": float(bce_logit),
        "full_pool_attention_mean": float(att.mean()) if att is not None else float("nan"),
    }

    # Project each variant's (n, n) prediction to (H, W) pixels.
    bbox_full = (0, 0, H, W)
    for rule, info in variants.items():
        pred_grid = info["pred"].reshape(n, n)
        pred_HW = project.patch_grid_to_pixel_mask(
            pred_grid, bbox=bbox_full, full_size=(H, W),
        )
        result[f"full_{rule}_mask"] = pred_HW
        result[f"full_{rule}_inverted"] = bool(info["inverted"])

    return result
