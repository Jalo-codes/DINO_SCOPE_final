"""lab_utils.data.resolution — single source of truth for image/patch geometry.

`Resolution` is the one place where image_size, patch_size, and num_patches
live.  Pass a Resolution everywhere instead of individual size integers.
Mismatches (e.g. AE cache built at 224 but config says 448) raise ConfigError
at construction time with a clear remediation message.

Crop helpers are all gathered here so that no other module needs to import
torchvision crop ops directly.  All accept and return PIL Images; callers
apply normalization separately.
"""

import dataclasses
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from lab_utils.errors import ConfigError, DataError


@dataclasses.dataclass(frozen=True)
class Resolution:
    """Immutable description of a ViT-compatible image resolution.

    Args:
        image_size: Square image side length in pixels.
        patch_size: ViT patch side length in pixels.

    Raises:
        ConfigError: If image_size is not divisible by patch_size.
    """
    image_size: int
    patch_size: int

    def __post_init__(self):
        if self.image_size <= 0:
            raise ConfigError(f"Resolution.image_size must be > 0, got {self.image_size}")
        if self.patch_size <= 0:
            raise ConfigError(f"Resolution.patch_size must be > 0, got {self.patch_size}")
        if self.image_size % self.patch_size != 0:
            raise ConfigError(
                f"image_size={self.image_size} must be divisible by "
                f"patch_size={self.patch_size}.  "
                f"Adjust one of these values so they divide evenly."
            )

    @property
    def num_patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        n = self.num_patches_per_side
        return n * n

    def __str__(self) -> str:
        return (f"Resolution(image_size={self.image_size}, "
                f"patch_size={self.patch_size}, "
                f"num_patches={self.num_patches})")


# ── Crop helpers ────────────────────────────────────────────────────────────

@dataclasses.dataclass
class CropResult:
    """Paired (image, mask) after a crop operation.

    mask is None when the source had no mask.

    ``mode`` records HOW the crop was produced ('random' | 'oracle' |
    'oracle_empty' | 'resize') for training telemetry; ``coverage`` is the
    realized splice pixel-fraction inside the crop (0 for maskless / empty).
    """
    image: Image.Image
    mask: Optional[Image.Image]
    valid: bool = True
    fallback_used: bool = False
    chosen_params: Optional[Tuple[int, int, int, int]] = None
    mode: str = 'random'
    coverage: float = 0.0


def resize_only(img: Image.Image, res: Resolution) -> Image.Image:
    """Resize img to (res.image_size, res.image_size) without cropping."""
    return TF.resize(img, [res.image_size, res.image_size], interpolation=Image.BILINEAR)


def resize_only_mask(mask: Image.Image, res: Resolution) -> Image.Image:
    """Resize a mask to resolution using NEAREST to preserve hard edges."""
    return TF.resize(mask, [res.image_size, res.image_size], interpolation=Image.NEAREST)


def center_crop_resize(img: Image.Image, res: Resolution) -> Image.Image:
    """Center-square-crop then resize to resolution."""
    s = min(img.size)
    return TF.resize(TF.center_crop(img, s), [res.image_size, res.image_size])


def random_resized_crop(
    img: Image.Image,
    res: Resolution,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
) -> Image.Image:
    """Random-resized-crop an image to resolution (no paired mask)."""
    i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=scale, ratio=ratio)
    return TF.resize(TF.crop(img, i, j, h, w), [res.image_size, res.image_size])


def random_resized_crop_pair(
    img: Image.Image,
    mask: Optional[Image.Image],
    res: Resolution,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    max_tries: int = 24,
    min_mask_patch_frac: float = 0.0,
) -> CropResult:
    """Random-resized-crop with the same params applied to both img and mask.

    Falls back to center_crop_resize if all tries produce an all-zero mask.
    """
    if mask is None:
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=scale, ratio=ratio)
        cropped_img = TF.resize(TF.crop(img, i, j, h, w), [res.image_size, res.image_size])
        return CropResult(
            image=cropped_img,
            mask=None,
            valid=True,
            fallback_used=False,
            chosen_params=(i, j, h, w),
        )

    for _ in range(max_tries):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=scale, ratio=ratio)
        cropped_img  = TF.resize(TF.crop(img,  i, j, h, w), [res.image_size, res.image_size])
        cropped_mask = TF.resize(
            TF.crop(mask, i, j, h, w),
            [res.image_size, res.image_size],
            interpolation=Image.NEAREST,
        )
        labels = mask_to_patch_labels(cropped_mask, res)
        frac = float(labels.float().mean().item())
        if labels.sum().item() > 0 and frac >= float(min_mask_patch_frac):
            return CropResult(
                image=cropped_img,
                mask=cropped_mask,
                valid=True,
                fallback_used=False,
                chosen_params=(i, j, h, w),
            )

    # Fallback: center crop keeps spatial coherence and typically covers the mask.
    s = min(img.size)
    fb_img  = TF.resize(TF.center_crop(img,  s), [res.image_size, res.image_size])
    fb_mask = TF.resize(
        TF.center_crop(mask, s),
        [res.image_size, res.image_size],
        interpolation=Image.NEAREST,
    )
    return CropResult(
        image=fb_img,
        mask=fb_mask,
        valid=False,
        fallback_used=True,
        chosen_params=None,
    )


def oracle_mask_crop(
    img: Image.Image,
    mask: Image.Image,
    res: Resolution,
    *,
    target_cov_range: Tuple[float, float] = (0.10, 0.40),
    jitter_frac: float = 0.25,
    rng=None,
) -> CropResult:
    """Mask-centered zoom crop that *guarantees* the splice stays in frame.

    The fix for small/off-center splices that random-resized crops can't
    surface (the case that otherwise demotes a tiny splice to the center-crop
    fallback — and, image-level, relabels it as *real*). Picks a SQUARE window
    centered on the mask centroid, sized so the splice covers roughly
    ``target_cov_range`` of the window area — zooming a tiny splice up to a
    learnable scale — always containing the full mask bounding box, with
    optional center jitter for variety.

    Returns ``valid=True`` with ``mode='oracle'`` on success.  Returns
    ``valid=False`` with ``mode='oracle_empty'`` ONLY when the mask is empty
    (no splice pixels) — the caller should then DROP the sample rather than
    feed a splice-free crop with a fake label.
    """
    import numpy as np
    if rng is None:
        rng = np.random

    m = np.asarray(mask.convert('L'), dtype=np.uint8) > 0
    H, W = m.shape
    if not m.any():
        return CropResult(
            image=resize_only(img, res), mask=None,
            valid=False, fallback_used=True, mode='oracle_empty', coverage=0.0,
        )

    ys, xs = np.where(m)
    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    bbox_h, bbox_w = r1 - r0, c1 - c0
    cy, cx = float(ys.mean()), float(xs.mean())
    splice_px = float(m.sum())

    lo, hi = float(target_cov_range[0]), float(target_cov_range[1])
    tcov = float(rng.uniform(lo, hi))
    side_for_cov = (splice_px / max(tcov, 1e-6)) ** 0.5
    side = int(round(max(side_for_cov, float(bbox_h), float(bbox_w))))
    side = max(8, min(side, H, W))

    jit = int(round(float(jitter_frac) * side))
    dy = int(round(rng.uniform(-jit, jit))) if jit > 0 else 0
    dx = int(round(rng.uniform(-jit, jit))) if jit > 0 else 0
    top  = int(round(cy - side / 2.0)) + dy
    left = int(round(cx - side / 2.0)) + dx
    # Clamp so the whole bbox stays in frame AND the crop stays within bounds.
    top  = max(max(0, r1 - side), min(top,  min(r0, H - side)))
    left = max(max(0, c1 - side), min(left, min(c0, W - side)))

    coverage = float(m[top:top + side, left:left + side].sum()) / float(side * side)

    crop_img  = img.crop((left, top, left + side, top + side)).resize(
        (res.image_size, res.image_size), Image.BILINEAR)
    crop_mask = mask.crop((left, top, left + side, top + side)).resize(
        (res.image_size, res.image_size), Image.NEAREST)
    return CropResult(
        image=crop_img, mask=crop_mask,
        valid=True, fallback_used=True, chosen_params=(top, left, side, side),
        mode='oracle', coverage=coverage,
    )


# ── Patch-label helper ──────────────────────────────────────────────────────

def mask_to_patch_labels(
    mask: Image.Image,
    res: Resolution,
    threshold: float = 0.15,
) -> torch.Tensor:
    """Convert a PIL 'L' mask (image_size × image_size) to per-patch binary labels.

    A patch is labelled 1 if the mean foreground density inside it exceeds
    `threshold` (default 0.15, matching the existing contrastive_test code).

    Args:
        mask:      PIL 'L' image at exactly (res.image_size, res.image_size).
        res:       Resolution for patch_size.
        threshold: Per-patch mean density threshold.

    Returns:
        1D LongTensor of shape (res.num_patches,).

    Raises:
        DataError: If mask size does not match resolution.
    """
    w, h = mask.size
    if w != res.image_size or h != res.image_size:
        raise DataError(
            f"mask_to_patch_labels: mask size ({w}×{h}) does not match "
            f"resolution.image_size={res.image_size}.  "
            f"Resize the mask before calling this function."
        )
    mask_t  = TF.to_tensor(mask)                             # (1, H, W) in [0,1]
    density = F.avg_pool2d(mask_t, res.patch_size, res.patch_size).flatten()
    return (density > threshold).long()


def mask_to_patch_labels_soft(
    mask: Image.Image,
    res: Resolution,
    low: float = 0.02,
    high: float = 0.06,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Soft per-patch labels with an "ignore band" near the splice boundary.

    Returns (labels, weights) of shape (res.num_patches,) each:

        density == 0.0           →  label=0, weight=1.0   (confident background)
        0 < density < low        →  label=0, weight=0.0   (IGNORE — too small to
                                                           supervise either way;
                                                           edge patches grazing
                                                           the splice mask)
        low <= density < high    →  label=1, weight=ramp  (linear 0→1 over band)
        density >= high          →  label=1, weight=1.0   (confident splice)

    Rationale: with the legacy hard threshold of 0.15, edge patches (where the
    splice mask covers a small fraction of the patch) were forced into the
    background class with full weight, injecting wrong supervision into the
    contrastive head. The ramp gives gradual credit at the boundary; the
    ignore band withholds any supervision for nearly-clean patches that still
    have a hint of splice in them.

    Args:
        mask: PIL 'L' image at exactly (res.image_size, res.image_size).
        res:  Resolution for patch_size.
        low:  Density below which (and >0) patches are IGNORED. 0.02 by default.
        high: Density at/above which patches are FULLY POSITIVE. 0.06 by default.
              Must satisfy 0 < low < high <= 1.

    Returns:
        labels:  LongTensor (num_patches,) — binary class assignment.
        weights: FloatTensor (num_patches,) in [0, 1] — per-patch contribution
                 multiplier; 0 means this patch is excluded from supervision.

    Raises:
        DataError: If mask size does not match resolution.
        ValueError: If band thresholds are invalid.
    """
    if not (0.0 < float(low) < float(high) <= 1.0):
        raise ValueError(
            f'mask_to_patch_labels_soft: must have 0 < low < high <= 1, '
            f'got low={low}, high={high}'
        )
    w, h = mask.size
    if w != res.image_size or h != res.image_size:
        raise DataError(
            f"mask_to_patch_labels_soft: mask size ({w}×{h}) does not match "
            f"resolution.image_size={res.image_size}."
        )
    mask_t  = TF.to_tensor(mask)
    density = F.avg_pool2d(mask_t, res.patch_size, res.patch_size).flatten()

    low_t  = float(low)
    high_t = float(high)

    # Labels: positive if density crossed the lower band (i.e., we're not
    # treating these as background even if their weight ramps).
    labels = (density >= low_t).long()

    # Weights: piecewise.
    weights = torch.zeros_like(density)
    weights[density == 0.0] = 1.0                          # confident bg
    weights[density >= high_t] = 1.0                       # confident splice
    ramp_mask = (density >= low_t) & (density < high_t)
    if bool(ramp_mask.any()):
        weights[ramp_mask] = (density[ramp_mask] - low_t) / (high_t - low_t)
    return labels, weights
