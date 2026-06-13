"""lab_utils.data.augment.light — lightweight full-image augmentations.

Each function:
  - Accepts explicit parameters (no hidden Config reads).
  - Returns AugmentResult with the applied op and a severity in [0, 1].
  - Passes the mask through unchanged (these are full-image, label-preserving ops).

Functions
---------
apply_jpeg           — JPEG compression at a given quality
apply_gaussian_noise — additive Gaussian noise at a given std
apply_resize_jitter  — downscale + upscale to simulate resampling artefacts
apply_flip_h         — horizontal flip (also flips mask)
"""

import io
import random
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from lab_utils.data.augment import AppliedOp, AugmentResult


# ── primitive ops (PIL → PIL, no AugmentResult) ─────────────────────────────

def _jpeg_encode(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def _add_gaussian(img: Image.Image, std: float) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.clip(arr + np.random.normal(0.0, float(std), arr.shape), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _add_poisson(img: Image.Image, peak: float) -> Image.Image:
    """Signal-dependent Poisson (shot) noise. Lower `peak` → heavier noise.

    Photon shot noise scales with intensity, so it destroys the camera's
    fixed-pattern / PRNU noise fingerprint (the cheap shortcut) without
    moving the underlying semantics — exactly the perturbation we want when
    training a *semantic*-displacement detector rather than a noise one.
    """
    arr  = np.array(img).astype(np.float32) / 255.0
    peak = max(1.0, float(peak))
    out  = np.clip(np.random.poisson(arr * peak) / peak, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


def _resize_jitter(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    sw = max(8, int(round(w * float(scale))))
    sh = max(8, int(round(h * float(scale))))
    small = img.resize((sw, sh), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


# ── public AugmentResult API ─────────────────────────────────────────────────

def apply_jpeg(
    img: Image.Image,
    mask: Optional[Image.Image] = None,
    *,
    quality: int,
    quality_range: tuple = (35, 100),
) -> AugmentResult:
    """JPEG-compress at the given quality.

    Args:
        img:           Input PIL RGB image.
        mask:          Optional mask (passed through unchanged).
        quality:       JPEG quality in [1, 100].  Lower → more compression.
        quality_range: (min, max) range used to normalize severity.

    Returns:
        AugmentResult with name='jpeg', params={'quality': quality},
        severity = (q_max - quality) / (q_max - q_min), clamped to [0, 1].
    """
    q_min, q_max = quality_range
    severity = float(np.clip((q_max - quality) / max(1, q_max - q_min), 0.0, 1.0))
    out = _jpeg_encode(img, quality)
    op  = AppliedOp(name='jpeg', params={'quality': int(quality)}, severity=severity)
    return AugmentResult(image=out, mask=mask, applied=(op,))


def apply_gaussian_noise(
    img: Image.Image,
    mask: Optional[Image.Image] = None,
    *,
    std: float,
    std_range: tuple = (0.0, 0.30),
) -> AugmentResult:
    """Add zero-mean Gaussian noise with the given std.

    Args:
        img:       Input PIL RGB image.
        mask:      Optional mask (passed through unchanged).
        std:       Noise standard deviation in pixel-value space [0, 1].
        std_range: (min, max) used to normalize severity.

    Returns:
        AugmentResult with name='gaussian_noise', severity = std / std_max.
    """
    std_min, std_max = std_range
    severity = float(np.clip((std - std_min) / max(1e-8, std_max - std_min), 0.0, 1.0))
    out = _add_gaussian(img, std)
    op  = AppliedOp(name='gaussian_noise', params={'std': float(std)}, severity=severity)
    return AugmentResult(image=out, mask=mask, applied=(op,))


def apply_poisson_noise(
    img: Image.Image,
    mask: Optional[Image.Image] = None,
    *,
    peak: float,
    peak_range: tuple = (8.0, 128.0),
) -> AugmentResult:
    """Add signal-dependent Poisson (shot) noise.

    Args:
        img:        Input PIL RGB image.
        mask:       Optional mask (passed through unchanged; label-preserving).
        peak:       Photon-count peak. LOWER → heavier noise.
        peak_range: (min, max) used to normalize severity; severity rises as
                    peak falls toward the min.

    Returns:
        AugmentResult with name='poisson_noise', params={'peak': peak}.
    """
    p_min, p_max = peak_range
    severity = float(np.clip((p_max - peak) / max(1e-8, p_max - p_min), 0.0, 1.0))
    out = _add_poisson(img, peak)
    op  = AppliedOp(name='poisson_noise', params={'peak': float(peak)}, severity=severity)
    return AugmentResult(image=out, mask=mask, applied=(op,))


def apply_resize_jitter(
    img: Image.Image,
    mask: Optional[Image.Image] = None,
    *,
    scale: float,
    scale_range: tuple = (0.80, 1.0),
) -> AugmentResult:
    """Downscale then upscale to simulate resampling artefacts.

    Args:
        img:         Input PIL RGB image.
        mask:        Optional mask (passed through unchanged; jitter is label-preserving).
        scale:       Intermediate downscale factor in (0, 1].  Smaller → more artefact.
        scale_range: (min, max) used to normalize severity.

    Returns:
        AugmentResult with name='resize_jitter', severity = (1 - scale) normalised.
    """
    s_min, s_max = scale_range
    severity = float(np.clip((s_max - scale) / max(1e-8, s_max - s_min), 0.0, 1.0))
    out = _resize_jitter(img, scale)
    op  = AppliedOp(name='resize_jitter', params={'scale': float(scale)}, severity=severity)
    return AugmentResult(image=out, mask=mask, applied=(op,))


def apply_blur(
    img: Image.Image,
    mask: Optional[Image.Image] = None,
    *,
    sigma: float,
    sigma_range: tuple = (0.0, 3.0),
) -> AugmentResult:
    """Apply Gaussian blur with the given standard deviation (radius)."""
    s_min, s_max = sigma_range
    severity = float(np.clip((sigma - s_min) / max(1e-8, s_max - s_min), 0.0, 1.0))
    out = img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    op  = AppliedOp(name='blur', params={'sigma': float(sigma)}, severity=severity)
    return AugmentResult(image=out, mask=mask, applied=(op,))


def apply_flip_h(
    img: Image.Image,
    mask: Optional[Image.Image] = None,
) -> AugmentResult:
    """Horizontal flip.  Flips both image and mask (if provided).

    Returns:
        AugmentResult with name='flip_h', severity=0.5 (constant — the flip
        either happens or it doesn't; there is no severity axis).
    """
    out      = img.transpose(Image.FLIP_LEFT_RIGHT)
    out_mask = mask.transpose(Image.FLIP_LEFT_RIGHT) if mask is not None else None
    op       = AppliedOp(name='flip_h', params={}, severity=0.5)
    return AugmentResult(image=out, mask=out_mask, applied=(op,))


# ── compound: apply several light augs stochastically ───────────────────────

def apply_light_augmentations(
    img: Image.Image,
    mask: Optional[Image.Image] = None,
    *,
    jpeg_prob: float = 0.25,
    jpeg_q_min: int = 88,
    jpeg_q_max: int = 98,
    noise_prob: float = 0.15,
    noise_std_min: float = 0.002,
    noise_std_max: float = 0.015,
    poisson_prob: float = 0.0,
    poisson_peak_min: float = 16.0,
    poisson_peak_max: float = 64.0,
    resize_prob: float = 0.20,
    resize_scale_min: float = 0.80,
    resize_scale_max: float = 0.98,
    flip_prob: float = 0.50,
    blur_prob: float = 0.0,
    blur_sigma_min: float = 0.0,
    blur_sigma_max: float = 1.0,
) -> AugmentResult:
    """Apply a random subset of light augmentations.

    All probabilities and ranges are explicit arguments — experiments pass
    their own values; no Config is read inside this function.

    Returns:
        AugmentResult accumulating all AppliedOps that fired.
    """
    out      = img
    out_mask = mask
    applied  = []

    if random.random() < jpeg_prob:
        q   = random.randint(int(jpeg_q_min), int(jpeg_q_max))
        res = apply_jpeg(out, out_mask, quality=q,
                         quality_range=(jpeg_q_min, jpeg_q_max))
        out = res.image
        applied.extend(res.applied)

    if random.random() < noise_prob:
        std = random.uniform(noise_std_min, noise_std_max)
        res = apply_gaussian_noise(out, out_mask, std=std,
                                   std_range=(noise_std_min, noise_std_max))
        out = res.image
        applied.extend(res.applied)

    if random.random() < poisson_prob:
        # peak sampled in [min, max]; lower peak = heavier shot noise
        peak = random.uniform(poisson_peak_min, poisson_peak_max)
        res  = apply_poisson_noise(out, out_mask, peak=peak,
                                   peak_range=(poisson_peak_min, poisson_peak_max))
        out = res.image
        applied.extend(res.applied)

    if random.random() < resize_prob:
        scale = random.uniform(resize_scale_min, resize_scale_max)
        res   = apply_resize_jitter(out, out_mask, scale=scale,
                                    scale_range=(resize_scale_min, resize_scale_max))
        out = res.image
        applied.extend(res.applied)

    if random.random() < blur_prob:
        sigma = random.uniform(blur_sigma_min, blur_sigma_max)
        res   = apply_blur(out, out_mask, sigma=sigma,
                           sigma_range=(blur_sigma_min, blur_sigma_max))
        out = res.image
        applied.extend(res.applied)

    if random.random() < flip_prob:
        res      = apply_flip_h(out, out_mask)
        out      = res.image
        out_mask = res.mask
        applied.extend(res.applied)

    return AugmentResult(image=out, mask=out_mask, applied=tuple(applied))
