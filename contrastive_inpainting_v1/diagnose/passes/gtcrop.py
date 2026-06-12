"""gtcrop: square crop of fixed image-area fraction, centered on GT centroid.

NO pad knob.  ONE knob: `area_frac` = (crop_side² / (H * W)).

Per-bucket area sweeps (from v2 design):
  small  : [0.30, 0.45, 0.60, 0.75]
  medium : [0.50, 0.65, 0.80, 0.90]
  large  : skipped

Crop is always square in source pixels, then resized to (T, T) — pure scale,
no aspect distortion. The crop's 28x28 partition is projected back to the
original (H, W) pixel mask at the crop's source coords; pixels outside the
crop are False (so any GT splice outside is FN under f1_pixel).

Returns (per area_frac):
  {
    'gtcrop_a{XX}_pure_mask':      (H, W) bool
    'gtcrop_a{XX}_pure_inverted':  bool
    'gtcrop_a{XX}_ceil_mask':      (H, W) bool
    'gtcrop_a{XX}_ceil_inverted':  bool
    'gtcrop_a{XX}_bce_logit':      float  (BCE on the crop)
    'gtcrop_a{XX}_bbox':           (top, left, side, side)
    'gtcrop_a{XX}_in_crop_splice_frac': float  (fraction of crop pixels that are GT)
    'gtcrop_a{XX}_oncrop_pixel_share': float  (crop area / total image area)
  }
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from lab_utils.eval.partition import DecodeSpec, decode_oracle_labels

from .. import polarity, project
from . import common


def area_key(area_frac: float) -> str:
    """Stable string id for a given area fraction. e.g., 0.30 -> 'a30'."""
    return f"a{int(round(float(area_frac) * 100)):02d}"


@torch.no_grad()
def run_gtcrop(
    model,
    source_image: Image.Image,
    device: torch.device,
    *,
    area_frac: float,
    gt_HW: np.ndarray,
    image_size: int,
    n_patch_per_side: int,
    imagenet_mean,
    imagenet_std,
    decode_spec=None,
) -> Dict:
    """One forward at 448x448 on a square area-frac crop centered on GT centroid."""
    W_src, H_src = source_image.size
    H, W = int(H_src), int(W_src)
    n = int(n_patch_per_side)

    # GT centroid (mean of True pixel coords; falls back to bbox center if empty).
    try:
        _, centroid = common.gt_bbox_and_centroid(gt_HW)
    except ValueError:
        # Caller should not reach here on reals; defensive — center crop.
        centroid = (H / 2.0, W / 2.0)

    top, left, side = common.centered_area_square(
        centroid, area_frac=area_frac, H_full=H, W_full=W,
    )
    bbox_src = (top, left, side, side)

    # Crop & forward.
    t = common.crop_resize_to_tensor(
        source_image, bbox_top=top, bbox_left=left, bbox_side=side,
        target_size=image_size,
        imagenet_mean=imagenet_mean, imagenet_std=imagenet_std,
    ).to(device, non_blocking=True)
    out = model(t)
    if out.get("contrastive") is None:
        raise RuntimeError("gtcrop pass requires contrastive head")
    z = out["contrastive"][0].detach().cpu().float().numpy()
    att = (out["pool_attention"][0].detach().cpu().float().numpy()
           if out.get("pool_attention") is not None else None)
    bce_logit = (float(out["image_logit"][0].detach().cpu().item())
                 if out.get("image_logit") is not None else float("nan"))

    raw = decode_oracle_labels(z, decode_spec or DecodeSpec())

    # Patch-grid GT inside the crop (for ceil polarity only).
    # Sample the centre pixel of each patch cell within the crop.
    ys = np.minimum((np.arange(n) * side // n) + max(1, side // n) // 2,
                    side - 1) + top
    xs = np.minimum((np.arange(n) * side // n) + max(1, side // n) // 2,
                    side - 1) + left
    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)
    gt_in_crop_grid = gt_HW[ys[:, None], xs[None, :]]
    gt_flat = gt_in_crop_grid.reshape(-1).astype(bool)

    variants = polarity.both_variants(raw, att, gt_flat)

    in_crop_splice_frac = project.in_pixel_splice_frac(bbox_src, gt_HW=gt_HW)
    oncrop_pixel_share = float(side * side) / float(H * W) if (H * W) else 0.0

    key = area_key(area_frac)
    result: Dict = {
        f"gtcrop_{key}_bce_logit": float(bce_logit),
        f"gtcrop_{key}_bbox": (int(top), int(left), int(side), int(side)),
        f"gtcrop_{key}_in_crop_splice_frac": float(in_crop_splice_frac),
        f"gtcrop_{key}_oncrop_pixel_share": float(oncrop_pixel_share),
        f"gtcrop_{key}_area_frac": float(area_frac),
        f"gtcrop_{key}_crop_side_px": int(side),
    }
    for rule, info in variants.items():
        pred_grid = info["pred"].reshape(n, n)
        pred_HW = project.patch_grid_to_pixel_mask(
            pred_grid, bbox=bbox_src, full_size=(H, W),
        )
        result[f"gtcrop_{key}_{rule}_mask"] = pred_HW
        result[f"gtcrop_{key}_{rule}_inverted"] = bool(info["inverted"])
    return result


def areas_for_bucket(bucket: str) -> List[float]:
    """Per-bucket area-fraction sweep (v2 spec)."""
    if area_tier == "small":
        return [0.30, 0.45, 0.60, 0.75]
    if area_tier == "medium":
        return [0.50, 0.65, 0.80, 0.90]
    return []
