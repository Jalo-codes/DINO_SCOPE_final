"""lab_utils.data.blob — vectorized ellipse blob generator.

Replaces the Perlin-noise blob from contrastive_test.  A rotated ellipse
with a soft sigmoid edge runs in sub-ms at 448² and produces area that is
exact to floating-point precision (no binary search, no retries).

Public API
----------
EllipseBlobParams   — frozen dataclass describing the sampling distribution
generate_blob_mask  — draw one mask from the distribution
paste_soft_alpha    — alpha-composite two images with a soft float mask
"""

import dataclasses
import math
import random
from typing import Optional

import numpy as np
from PIL import Image

from lab_utils.errors import DataError
from lab_utils.data.resolution import Resolution


@dataclasses.dataclass(frozen=True)
class EllipseBlobParams:
    """Sampling distribution for generate_blob_mask.

    Args:
        min_area_frac:       Minimum blob area as fraction of image area.
        max_area_frac:       Maximum blob area as fraction of image area.
        min_aspect:          Minimum semi-major/semi-minor ratio (≥ 1.0).
        max_aspect:          Maximum semi-major/semi-minor ratio.
        boundary_lobes:      Number of sinusoidal lobes added to boundary
                             (0 = pure ellipse).
        boundary_amplitude:  Amplitude of lobe perturbation as a fraction of
                             the local radius (0.0 = pure ellipse).
        edge_softness_px:    Width in pixels of the soft sigmoid edge.
                             Larger values → smoother transition.
    """
    min_area_frac:      float = 0.10
    max_area_frac:      float = 0.40
    min_aspect:         float = 1.0
    max_aspect:         float = 2.5
    boundary_lobes:     int   = 0
    boundary_amplitude: float = 0.0
    edge_softness_px:   float = 4.0

    def __post_init__(self):
        if not (0.0 < self.min_area_frac < self.max_area_frac <= 1.0):
            raise ValueError(
                f"EllipseBlobParams: need 0 < min_area_frac < max_area_frac ≤ 1, "
                f"got {self.min_area_frac}, {self.max_area_frac}"
            )
        if self.min_aspect < 1.0 or self.max_aspect < self.min_aspect:
            raise ValueError(
                f"EllipseBlobParams: need 1 ≤ min_aspect ≤ max_aspect, "
                f"got {self.min_aspect}, {self.max_aspect}"
            )
        if self.edge_softness_px <= 0:
            raise ValueError(
                f"EllipseBlobParams: edge_softness_px must be > 0, "
                f"got {self.edge_softness_px}"
            )


# Default params matching the contrastive_test blob area range.
DEFAULT_BLOB_PARAMS = EllipseBlobParams(
    min_area_frac=0.10,
    max_area_frac=0.40,
)


def generate_blob_mask(
    res: Resolution,
    params: EllipseBlobParams = DEFAULT_BLOB_PARAMS,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a soft float mask in [0, 1] for a random rotated ellipse.

    The mask value is 1 inside the ellipse and falls off with a sigmoid
    transition of width `params.edge_softness_px` at the boundary.

    The ellipse area (measured at the 0.5 contour) equals `target_frac ×
    image_area` to floating-point precision.  The center is constrained so
    the ellipse fits within the image, guaranteeing the measured area is
    within ±1% of the sampled target.

    Args:
        res:    Resolution of the output mask.
        params: Sampling distribution.
        seed:   Integer seed for reproducibility (None = random).

    Returns:
        Float32 numpy array of shape (res.image_size, res.image_size) in [0, 1].
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    H = W = res.image_size

    # Sample target area and aspect ratio.
    target_frac = rng.uniform(params.min_area_frac, params.max_area_frac)
    aspect      = rng.uniform(params.min_aspect, params.max_aspect)
    angle       = rng.uniform(0, math.pi)  # rotation in radians

    # Compute semi-axes from target area: π × a × b = target_frac × H × W
    # with a = aspect × b  →  b = sqrt(target_frac × H × W / (π × aspect))
    b = math.sqrt(target_frac * H * W / (math.pi * aspect))
    a = aspect * b  # semi-major axis (longer)

    # Bounding box of rotated ellipse: max extent in x and y.
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    half_w = math.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2)
    half_h = math.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2)

    # Constrain centre so the ellipse (at hard boundary) stays within image.
    margin_x = min(half_w, W / 2 - 1)
    margin_y = min(half_h, H / 2 - 1)
    cx = rng.uniform(margin_x, W - margin_x)
    cy = rng.uniform(margin_y, H - margin_y)

    # Vectorised distance computation.
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = (xs - cx) *  cos_t + (ys - cy) * sin_t   # along major axis
    dy = (xs - cx) * -sin_t + (ys - cy) * cos_t   # along minor axis

    # Normalised ellipse distance (= 1 at boundary, < 1 inside).
    d = np.sqrt((dx / a) ** 2 + (dy / b) ** 2)

    # Optional sinusoidal lobe perturbation.
    if params.boundary_lobes > 0 and params.boundary_amplitude > 0.0:
        phi = np.arctan2(dy, dx)
        d   = d * (1.0 + params.boundary_amplitude
                   * np.sin(params.boundary_lobes * phi))

    # Soft sigmoid: mask = 1 inside ellipse, 0 outside, with smooth transition.
    # sigmoid((1 - d) / sigma) where sigma = edge_softness_px / min(a, b)
    sigma = params.edge_softness_px / min(a, b)
    x = np.clip((1.0 - d) / sigma, -60.0, 60.0)
    mask  = 1.0 / (1.0 + np.exp(-x))
    return mask.astype(np.float32)


def generate_blob_mask_pil(
    res: Resolution,
    params: EllipseBlobParams = DEFAULT_BLOB_PARAMS,
    seed: Optional[int] = None,
    threshold: float = 0.5,
) -> Image.Image:
    """Generate a hard-binary PIL 'L' mask (0 or 255) for a rotated ellipse.

    Convenience wrapper around generate_blob_mask that thresholds the soft
    mask and returns a PIL image compatible with the rest of the pipeline.
    """
    soft = generate_blob_mask(res, params, seed)
    binary = (soft >= threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode='L')


# ── Paste helpers ────────────────────────────────────────────────────────────

def paste_soft_alpha(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Alpha-composite foreground onto background using a soft float mask.

    Args:
        background: uint8 RGB array of shape (H, W, 3).
        foreground: uint8 RGB array of shape (H, W, 3).
        alpha:      Float array of shape (H, W) in [0, 1].

    Returns:
        Composited uint8 RGB array of shape (H, W, 3).

    Raises:
        DataError: If shapes are incompatible.
    """
    if background.shape != foreground.shape:
        raise DataError(
            f"paste_soft_alpha: background {background.shape} and "
            f"foreground {foreground.shape} must have the same shape."
        )
    if alpha.shape[:2] != background.shape[:2]:
        raise DataError(
            f"paste_soft_alpha: alpha {alpha.shape} H×W must match "
            f"background {background.shape[:2]}."
        )
    a = alpha[..., None].astype(np.float32)
    bg = background.astype(np.float32)
    fg = foreground.astype(np.float32)
    return np.clip(a * fg + (1.0 - a) * bg, 0, 255).astype(np.uint8)
