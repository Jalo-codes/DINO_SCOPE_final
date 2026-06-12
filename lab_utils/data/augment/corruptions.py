"""lab_utils.data.augment.corruptions — corruption families for splice harness.

Rewritten from contrastive_test/data/harness_corruptions.py.

Key changes from the original:
  - sample_corruption_spec / sample_distinct_corruption_specs take **explicit
    param ranges** instead of a cfg object — no hidden Config reads inside.
  - apply_corruption returns AugmentResult (image, mask=None, applied=(...)).
  - apply_light_augmentations (thin wrapper) also returns AugmentResult.
  - make_invariance_pair returns a pair of AugmentResults.

Curriculum (what q-range / std-range to use at epoch N) is entirely the
experiment's job.  Pass the right ranges to sample_corruption_spec.
"""

import io
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from lab_utils.data.augment import AppliedOp, AugmentResult


# ── primitive PIL ops ────────────────────────────────────────────────────────

def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def _additive_noise(img: Image.Image, std: float) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.clip(arr + np.random.normal(0.0, float(std), arr.shape), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _poisson(img: Image.Image, peak: float) -> Image.Image:
    arr  = np.array(img).astype(np.float32) / 255.0
    peak = max(8.0, float(peak))
    arr  = np.clip(np.random.poisson(arr * peak) / peak, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _downscale(img: Image.Image, scale: float) -> Image.Image:
    w, h  = img.size
    sw    = max(1, int(w * float(scale)))
    sh    = max(1, int(h * float(scale)))
    small = img.resize((sw, sh), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _resize_jitter(img: Image.Image, scale_min: float, scale_max: float) -> Image.Image:
    w, h  = img.size
    scale = random.uniform(float(scale_min), float(scale_max))
    sw    = max(8, int(round(w * scale)))
    sh    = max(8, int(round(h * scale)))
    small = img.resize((sw, sh), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


# ── CorruptionSpec ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CorruptionSpec:
    """Fully-specified corruption to apply.  No Config references."""
    family: str
    params: Dict[str, Any] = field(default_factory=dict)
    severity_tag: str = 'train'


# ── apply_corruption ─────────────────────────────────────────────────────────

def _severity_jpeg(quality: int, q_min: int = 35, q_max: int = 100) -> float:
    return float(np.clip((q_max - quality) / max(1, q_max - q_min), 0.0, 1.0))


def _severity_std(std: float, std_min: float = 0.0, std_max: float = 0.30) -> float:
    return float(np.clip((std - std_min) / max(1e-8, std_max - std_min), 0.0, 1.0))


def apply_corruption(
    img: Image.Image,
    spec: CorruptionSpec,
    mask: Optional[Image.Image] = None,
) -> AugmentResult:
    """Apply spec to img and return AugmentResult.

    Args:
        img:  Input PIL RGB image.
        spec: Fully-specified corruption (family + params).
        mask: Optional mask (passed through unchanged).

    Returns:
        AugmentResult with a single AppliedOp (or zero ops for 'clean').

    Raises:
        KeyError: If spec.family is not supported.
    """
    if spec.family == 'clean':
        return AugmentResult(image=img.copy(), mask=mask, applied=())

    if spec.family == 'jpeg':
        q   = int(spec.params['quality'])
        out = _jpeg(img, q)
        op  = AppliedOp(name='jpeg', params={'quality': q},
                        severity=_severity_jpeg(q))
        return AugmentResult(image=out, mask=mask, applied=(op,))

    if spec.family == 'double_jpeg':
        q1  = int(spec.params['q1'])
        q2  = int(spec.params['q2'])
        out = _jpeg(_jpeg(img, q1), q2)
        op  = AppliedOp(name='double_jpeg', params={'q1': q1, 'q2': q2},
                        severity=_severity_jpeg(min(q1, q2)))
        return AugmentResult(image=out, mask=mask, applied=(op,))

    if spec.family == 'gaussian':
        std = float(spec.params['std'])
        out = _additive_noise(img, std)
        op  = AppliedOp(name='gaussian', params={'std': std},
                        severity=_severity_std(std))
        return AugmentResult(image=out, mask=mask, applied=(op,))

    if spec.family == 'poisson':
        peak = float(spec.params['peak'])
        out  = _poisson(img, peak)
        # Severity: lower peak → more noise.  Reference range: [8, 96].
        severity = float(np.clip((96.0 - peak) / 88.0, 0.0, 1.0))
        op   = AppliedOp(name='poisson', params={'peak': peak}, severity=severity)
        return AugmentResult(image=out, mask=mask, applied=(op,))

    if spec.family == 'resize':
        scale = float(spec.params['scale'])
        out   = _downscale(img, scale)
        # Severity: smaller scale → more degradation.
        severity = float(np.clip(1.0 - scale, 0.0, 1.0))
        op    = AppliedOp(name='resize', params={'scale': scale}, severity=severity)
        return AugmentResult(image=out, mask=mask, applied=(op,))

    raise KeyError(f"apply_corruption: unsupported family {spec.family!r}")


# ── samplers (explicit params, no cfg) ──────────────────────────────────────

def sample_corruption_spec(
    families: Sequence[str] = ('jpeg', 'gaussian'),
    *,
    allow_clean: bool = False,
    severity_tag: str = 'train',
    jpeg_q_min: int = 35,
    jpeg_q_max: int = 65,
    double_jpeg_q1_min: int = 82,
    double_jpeg_q1_max: int = 96,
    double_jpeg_q2_min: int = 45,
    double_jpeg_q2_max: int = 82,
    gaussian_std_min: float = 0.10,
    gaussian_std_max: float = 0.30,
    poisson_peak_choices: Tuple[int, ...] = (16, 24, 32, 48, 64, 96),
    resize_min: float = 0.55,
    resize_max: float = 0.90,
) -> CorruptionSpec:
    """Sample one CorruptionSpec from the given family pool.

    All ranges are explicit arguments — no Config is read inside.

    Args:
        families:    Pool of corruption families to choose from.
        allow_clean: If True, 'clean' is prepended to the pool.
        severity_tag: Attached to the returned spec for downstream logging.
        *_min / *_max / *_choices: Family-specific parameter ranges.

    Returns:
        CorruptionSpec with sampled family and params.
    """
    pool   = ['clean'] + list(families) if allow_clean else list(families)
    family = random.choice(pool)

    if family == 'clean':
        return CorruptionSpec(family='clean', params={}, severity_tag=severity_tag)

    if family == 'jpeg':
        q = random.randint(int(jpeg_q_min), int(jpeg_q_max))
        return CorruptionSpec(family='jpeg', params={'quality': q},
                              severity_tag=severity_tag)

    if family == 'double_jpeg':
        q1    = random.randint(int(double_jpeg_q1_min), int(double_jpeg_q1_max))
        q2_hi = min(int(double_jpeg_q2_max), q1)
        q2    = random.randint(int(double_jpeg_q2_min),
                               max(int(double_jpeg_q2_min), q2_hi))
        return CorruptionSpec(family='double_jpeg', params={'q1': q1, 'q2': q2},
                              severity_tag=severity_tag)

    if family == 'gaussian':
        std = random.uniform(float(gaussian_std_min), float(gaussian_std_max))
        return CorruptionSpec(family='gaussian', params={'std': std},
                              severity_tag=severity_tag)

    if family == 'poisson':
        peak = float(random.choice(list(poisson_peak_choices)))
        return CorruptionSpec(family='poisson', params={'peak': peak},
                              severity_tag=severity_tag)

    if family == 'resize':
        scale = random.uniform(float(resize_min), float(resize_max))
        return CorruptionSpec(family='resize', params={'scale': scale},
                              severity_tag=severity_tag)

    raise KeyError(f"sample_corruption_spec: unsupported family {family!r}")


def sample_distinct_corruption_specs(
    n: int,
    families: Sequence[str] = ('jpeg', 'gaussian'),
    *,
    allow_clean_first: bool = False,
    allow_clean_other: bool = False,
    severity_tag: str = 'train',
    **sampler_kwargs,
) -> Tuple[CorruptionSpec, ...]:
    """Sample n CorruptionSpecs that are pairwise distinct by (family, params).

    Falls back to allowing repeats if the family pool is too small.

    Args:
        n:                  Number of specs to sample.
        families:           Pool passed to sample_corruption_spec.
        allow_clean_first:  Allow 'clean' for the first spec.
        allow_clean_other:  Allow 'clean' for subsequent specs.
        severity_tag:       Forwarded to each spec.
        **sampler_kwargs:   Forwarded to sample_corruption_spec.

    Returns:
        Tuple of CorruptionSpecs.
    """
    specs: list = []
    used: set   = set()
    for idx in range(n):
        allow_clean = allow_clean_first if idx == 0 else allow_clean_other
        for _ in range(16):
            spec = sample_corruption_spec(
                families,
                allow_clean=allow_clean,
                severity_tag=severity_tag,
                **sampler_kwargs,
            )
            key = (spec.family, tuple(sorted(spec.params.items())))
            if spec.family == 'clean' or key not in used:
                specs.append(spec)
                used.add(key)
                break
        else:
            specs.append(spec)  # noqa: F821  (defined in loop body above)
    return tuple(specs)


def make_invariance_pair(
    img: Image.Image,
    *,
    jpeg_prob: float = 0.70,
    jpeg_q_min: int = 90,
    jpeg_q_max: int = 98,
    noise_prob: float = 0.50,
    noise_std_min: float = 0.002,
    noise_std_max: float = 0.01,
    resize_prob: float = 0.50,
    resize_scale_min: float = 0.88,
    resize_scale_max: float = 0.98,
) -> Tuple[AugmentResult, AugmentResult]:
    """Build a (clean, lightly-augmented) pair for invariance regularization.

    Returns:
        (clean_result, augmented_result) — both AugmentResult with mask=None.
    """
    clean = AugmentResult(image=img.copy(), mask=None, applied=())

    out     = img.copy()
    applied = []

    if random.random() < jpeg_prob:
        q   = random.randint(jpeg_q_min, jpeg_q_max)
        out = _jpeg(out, q)
        applied.append(AppliedOp(
            name='jpeg', params={'quality': q},
            severity=_severity_jpeg(q, jpeg_q_min, jpeg_q_max),
        ))

    if random.random() < noise_prob:
        std = random.uniform(noise_std_min, noise_std_max)
        out = _additive_noise(out, std)
        applied.append(AppliedOp(
            name='gaussian_noise', params={'std': std},
            severity=_severity_std(std, noise_std_min, noise_std_max),
        ))

    if random.random() < resize_prob:
        out = _resize_jitter(out, resize_scale_min, resize_scale_max)
        applied.append(AppliedOp(
            name='resize_jitter',
            params={'scale_min': resize_scale_min, 'scale_max': resize_scale_max},
            severity=(1.0 - (resize_scale_min + resize_scale_max) / 2.0),
        ))

    augmented = AugmentResult(image=out, mask=None, applied=tuple(applied))
    return clean, augmented
