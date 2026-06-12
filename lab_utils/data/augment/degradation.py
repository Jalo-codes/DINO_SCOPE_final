"""lab_utils.data.augment.degradation — multi-region degradation harness.

Rewritten from contrastive_test/data/harness_degradation.py.

Key changes:
  - build_degradation_example takes explicit param args instead of cfg.
  - Uses lab_utils blob (ellipse, not Perlin) for mask generation.
  - Returns DegradationExample with an `applied` tuple of AppliedOps so
    downstream code can log what was done.
  - No Config object is read inside any function in this module.
"""

import dataclasses
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from lab_utils.data.augment import AppliedOp, AugmentResult
from lab_utils.data.augment.corruptions import (
    CorruptionSpec,
    apply_corruption,
    sample_corruption_spec,
)
from lab_utils.data.blob import EllipseBlobParams, generate_blob_mask_pil
from lab_utils.data.resolution import Resolution, mask_to_patch_labels


# ── internal helpers ─────────────────────────────────────────────────────────

def _full_mask(size: int) -> Image.Image:
    return Image.new('L', (size, size), 255)


def _mask_area_frac(mask: Image.Image) -> float:
    arr = np.asarray(mask, dtype=np.uint8) > 127
    return float(arr.mean())


def _intersection_frac(a: Image.Image, b: Image.Image) -> float:
    aa = np.asarray(a, dtype=np.uint8) > 127
    bb = np.asarray(b, dtype=np.uint8) > 127
    inter = float(np.logical_and(aa, bb).sum())
    denom = float(max(1, min(aa.sum(), bb.sum())))
    return inter / denom


def _apply_masked_corruption(
    base: Image.Image,
    mask: Image.Image,
    spec: CorruptionSpec,
) -> Image.Image:
    result = apply_corruption(base, spec)
    return Image.composite(result.image, base, mask)


def _sample_blob(res: Resolution, area_range: Tuple[float, float],
                 blob_params: EllipseBlobParams) -> Image.Image:
    params = dataclasses.replace(
        blob_params,
        min_area_frac=float(area_range[0]),
        max_area_frac=float(area_range[1]),
    )
    return generate_blob_mask_pil(res, params)


def _sample_disjoint_blob(
    res: Resolution,
    area_range: Tuple[float, float],
    blob_params: EllipseBlobParams,
    forbidden: Sequence[Image.Image],
    max_overlap: float,
    max_tries: int,
) -> Image.Image:
    last = None
    for _ in range(max_tries):
        cand = _sample_blob(res, area_range, blob_params)
        last = cand
        if all(_intersection_frac(cand, other) <= float(max_overlap)
               for other in forbidden):
            return cand
    return last if last is not None else _sample_blob(res, area_range, blob_params)


def _choose_variant(
    variants: Tuple[str, ...],
    probs: Tuple[float, ...],
) -> str:
    if len(variants) != len(probs):
        raise ValueError(
            f"variants and probs must have the same length, "
            f"got {len(variants)} vs {len(probs)}"
        )
    total = float(sum(probs))
    if total <= 0:
        return str(variants[0])
    r   = random.random() * total
    acc = 0.0
    for name, p in zip(variants, probs):
        acc += float(p)
        if r <= acc:
            return str(name)
    return str(variants[-1])


def _sample_target_spec(
    dominant: CorruptionSpec,
    families: Sequence[str],
    target_clean_prob: float,
    **sampler_kwargs,
) -> CorruptionSpec:
    if dominant.family != 'clean' and random.random() < float(target_clean_prob):
        return CorruptionSpec(family='clean', params={}, severity_tag='train')
    for _ in range(16):
        spec = sample_corruption_spec(families, allow_clean=False, **sampler_kwargs)
        if spec.family != dominant.family:
            return spec
    return spec  # noqa: F821


# ── public result types ──────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class RegionRegime:
    """Description of one spatial region and its corruption assignment."""
    mask:       Image.Image
    area_frac:  float
    corruption: CorruptionSpec
    regime_id:  int
    role:       str  # 'dominant' | 'target' | 'auxiliary'


@dataclasses.dataclass
class DegradationExample:
    """Output of build_degradation_example.

    Extends AugmentResult semantics: image, mask (None for this type), applied.
    Also carries patch-level labels and supervision metadata.
    """
    image:      Image.Image
    labels:     torch.Tensor   # shape (num_patches,) long
    supervised: bool
    is_single:  bool
    meta:       Dict[str, Any]
    applied:    Tuple[AppliedOp, ...] = dataclasses.field(default_factory=tuple)


# ── main entry point ─────────────────────────────────────────────────────────

def build_degradation_example(
    img: Image.Image,
    res: Resolution,
    *,
    families: Tuple[str, ...] = ('jpeg', 'double_jpeg', 'gaussian', 'poisson', 'resize'),
    variants: Tuple[str, ...] = (
        'clean_plus_small', 'global_plus_small', 'large_plus_small', 'global_negative'
    ),
    variant_probs: Tuple[float, ...] = (0.35, 0.25, 0.25, 0.15),
    small_area: Tuple[float, float] = (0.05, 0.25),
    large_area: Tuple[float, float] = (0.45, 0.80),
    target_soft_max_area: float = 0.45,
    target_clean_prob: float = 0.20,
    max_mask_overlap: float = 0.08,
    mask_tries: int = 24,
    blob_params: Optional[EllipseBlobParams] = None,
    label_threshold: float = 0.15,
    **sampler_kwargs,
) -> DegradationExample:
    """Build one multi-region degradation training example.

    All configuration is explicit — no Config object is read inside.

    Args:
        img:                 Source PIL RGB image (already at res.image_size).
        res:                 Resolution for blob generation and label extraction.
        families:            Corruption families available for sampling.
        variants:            Degradation variant names.
        variant_probs:       Sampling weights for variants (must match len).
        small_area:          (min, max) area fraction for small blobs.
        large_area:          (min, max) area fraction for large/dominant blobs.
        target_soft_max_area: Cap on small_area[1] used for target placement.
        target_clean_prob:   Probability of assigning 'clean' as target spec.
        max_mask_overlap:    Max overlap fraction allowed between disjoint blobs.
        mask_tries:          How many sampling attempts for disjoint blobs.
        blob_params:         EllipseBlobParams (defaults to standard ellipse).
        label_threshold:     Per-patch density threshold for mask_to_patch_labels.
        **sampler_kwargs:    Forwarded to sample_corruption_spec (q ranges, etc.)

    Returns:
        DegradationExample.
    """
    if blob_params is None:
        blob_params = EllipseBlobParams(
            min_area_frac=min(small_area[0], large_area[0]),
            max_area_frac=max(small_area[1], large_area[1]),
        )

    small_area_eff = (float(small_area[0]),
                      min(float(small_area[1]), float(target_soft_max_area)))
    size    = res.image_size
    n_patches = res.num_patches
    variant = _choose_variant(variants, variant_probs)
    base    = img.copy()
    applied: List[AppliedOp] = []

    # ── global_negative ──────────────────────────────────────────────────────
    if variant == 'global_negative':
        dominant = sample_corruption_spec(families, allow_clean=False, **sampler_kwargs)
        res_aug  = apply_corruption(base, dominant)
        out      = res_aug.image
        applied.extend(res_aug.applied)
        labels   = torch.zeros(n_patches, dtype=torch.long)
        meta     = {
            'variant': 'global_negative',
            'degrade_type': dominant.family,
            'treatment_side': 'global',
            'dominant_family': dominant.family,
            'target_family': 'none',
            'num_regimes': 1,
            'num_treated_regions': 0,
            'target_area_frac': 0.0,
            'dominant_area_frac': 1.0,
            'typed_target_regime_id': -1,
            'dominant_regime_id': 0,
            'is_global_negative': 1,
            'is_multi_region': 0,
            'has_large_context': 1,
        }
        return DegradationExample(
            image=out, labels=labels, supervised=True,
            is_single=True, meta=meta, applied=tuple(applied),
        )

    # ── clean_plus_small ─────────────────────────────────────────────────────
    if variant == 'clean_plus_small':
        target      = sample_corruption_spec(families, allow_clean=False, **sampler_kwargs)
        target_mask = _sample_blob(res, small_area_eff, blob_params)
        out         = _apply_masked_corruption(base, target_mask, target)
        res_aug     = apply_corruption(base, target)
        applied.extend(res_aug.applied)
        area        = _mask_area_frac(target_mask)
        labels      = mask_to_patch_labels(target_mask, res, label_threshold)
        meta        = {
            'variant': variant,
            'degrade_type': target.family,
            'treatment_side': 'local_target',
            'dominant_family': 'clean',
            'target_family': target.family,
            'num_regimes': 2,
            'num_treated_regions': 1,
            'target_area_frac': area,
            'dominant_area_frac': 1.0 - area,
            'typed_target_regime_id': 1,
            'dominant_regime_id': 0,
            'is_global_negative': 0,
            'is_multi_region': 0,
            'has_large_context': int((1.0 - area) > 0.5),
        }
        return DegradationExample(
            image=out, labels=labels, supervised=True,
            is_single=False, meta=meta, applied=tuple(applied),
        )

    # ── global_plus_small ────────────────────────────────────────────────────
    if variant == 'global_plus_small':
        dominant    = sample_corruption_spec(families, allow_clean=False, **sampler_kwargs)
        target      = _sample_target_spec(dominant, families, target_clean_prob, **sampler_kwargs)
        out         = apply_corruption(base, dominant).image
        target_mask = _sample_blob(res, small_area_eff, blob_params)
        target_patch = apply_corruption(base, target).image
        out          = Image.composite(target_patch, out, target_mask)
        for spec in (dominant, target):
            applied.extend(apply_corruption(base, spec).applied)
        area   = _mask_area_frac(target_mask)
        labels = mask_to_patch_labels(target_mask, res, label_threshold)
        meta   = {
            'variant': variant,
            'degrade_type': target.family if target.family != 'clean' else dominant.family,
            'treatment_side': 'local_target_on_global',
            'dominant_family': dominant.family,
            'target_family': target.family,
            'num_regimes': 2,
            'num_treated_regions': 2 if target.family != 'clean' else 1,
            'target_area_frac': area,
            'dominant_area_frac': 1.0,
            'typed_target_regime_id': 1,
            'dominant_regime_id': 0,
            'is_global_negative': 0,
            'is_multi_region': int(target.family != 'clean'),
            'has_large_context': 1,
        }
        return DegradationExample(
            image=out, labels=labels, supervised=True,
            is_single=False, meta=meta, applied=tuple(applied),
        )

    # ── large_plus_small ─────────────────────────────────────────────────────
    if variant == 'large_plus_small':
        dominant = sample_corruption_spec(families, allow_clean=False, **sampler_kwargs)
        target   = sample_corruption_spec(families, allow_clean=False, **sampler_kwargs)
        for _ in range(16):
            if target.family != dominant.family:
                break
            target = sample_corruption_spec(families, allow_clean=False, **sampler_kwargs)
        dominant_mask = _sample_blob(res, large_area, blob_params)
        target_mask   = _sample_disjoint_blob(
            res, small_area_eff, blob_params,
            forbidden=[dominant_mask],
            max_overlap=max_mask_overlap,
            max_tries=mask_tries,
        )
        out          = _apply_masked_corruption(base, dominant_mask, dominant)
        target_patch = apply_corruption(base, target).image
        out          = Image.composite(target_patch, out, target_mask)
        for spec in (dominant, target):
            applied.extend(apply_corruption(base, spec).applied)
        area_t = _mask_area_frac(target_mask)
        area_d = _mask_area_frac(dominant_mask)
        labels = mask_to_patch_labels(target_mask, res, label_threshold)
        meta   = {
            'variant': variant,
            'degrade_type': target.family if target.family != 'clean' else dominant.family,
            'treatment_side': 'local_target_on_large',
            'dominant_family': dominant.family,
            'target_family': target.family,
            'num_regimes': 3,
            'num_treated_regions': 2,
            'target_area_frac': area_t,
            'dominant_area_frac': area_d,
            'typed_target_regime_id': 2,
            'dominant_regime_id': 1,
            'is_global_negative': 0,
            'is_multi_region': 1,
            'has_large_context': int(area_d > 0.5),
        }
        return DegradationExample(
            image=out, labels=labels, supervised=True,
            is_single=False, meta=meta, applied=tuple(applied),
        )

    raise KeyError(f"build_degradation_example: unsupported variant {variant!r}")
