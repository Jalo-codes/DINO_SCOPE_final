"""Shared pre-processing helpers for diagnose passes.

Pure-geometry helpers (window grid, square crops, bbox projection, etc.)
have moved to :mod:`lab_utils.eval.window_geometry`.  This module re-exports
them so existing imports inside the ``diagnose.passes`` package keep
working, and adds the torchvision-flavored preprocessing helpers needed for
``crop → tensor`` ingestion.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from lab_utils.eval.window_geometry import (
    axis_positions,
    centered_area_square,
    gt_bbox_and_centroid,
    square_expand_crop,
    window_grid,
    window_set_hash,
)


__all__ = [
    'axis_positions',
    'centered_area_square',
    'gt_bbox_and_centroid',
    'square_expand_crop',
    'window_grid',
    'window_set_hash',
    'crop_resize_to_tensor',
    'full_image_to_tensor',
]


# ---------------------------------------------------------------------------
# Preprocessing: PIL crop → square tensor with ImageNet norm
# (kept local because torchvision is not a hard dep of lab_utils.eval)
# ---------------------------------------------------------------------------


def crop_resize_to_tensor(
    source_image: Image.Image,
    *,
    bbox_top: int,
    bbox_left: int,
    bbox_side: int,
    target_size: int,
    imagenet_mean: Tuple[float, float, float],
    imagenet_std: Tuple[float, float, float],
) -> torch.Tensor:
    """Crop square region, resize, normalize. Returns ``(1, 3, T, T)``."""
    cropped = TF.crop(source_image, int(bbox_top), int(bbox_left),
                      int(bbox_side), int(bbox_side))
    resized = TF.resize(cropped, [int(target_size), int(target_size)],
                        interpolation=Image.BILINEAR)
    norm = transforms.Normalize(list(imagenet_mean), list(imagenet_std))
    return norm(TF.to_tensor(resized)).unsqueeze(0)


def full_image_to_tensor(
    source_image: Image.Image,
    *,
    target_size: int,
    imagenet_mean: Tuple[float, float, float],
    imagenet_std: Tuple[float, float, float],
) -> torch.Tensor:
    """Resize PIL image to ``(T, T)``, normalize, return ``(1, 3, T, T)``.

    Introduces aspect-ratio squish for non-square images — matches the
    model's training-time preprocessing.
    """
    resized = TF.resize(source_image, [int(target_size), int(target_size)],
                        interpolation=Image.BILINEAR)
    norm = transforms.Normalize(list(imagenet_mean), list(imagenet_std))
    return norm(TF.to_tensor(resized)).unsqueeze(0)
