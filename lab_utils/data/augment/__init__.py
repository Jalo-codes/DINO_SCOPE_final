"""lab_utils.data.augment — standardized augmentation building blocks.

Boundary contract:

* ``lab_utils.data.augment`` (this package) ships **raw augmentation
  primitives** that take explicit numerical kwargs.  No knowledge of
  experiment configs.
* Experiment-side cfg-to-kwargs adapters (e.g.
  ``contrastive_inpainting_v1.configs.augment``) live in the experiment package
  so this library does not take a dependency on any one experiment's
  ``Config`` shape.

Every public augmentation function returns :class:`AugmentResult` so
experiments can log, stratify, or ignore what was applied without changing
the call site.  Curriculum (which severity to request at epoch N) is the
experiment's job.
"""

import dataclasses
from typing import Optional, Tuple

from PIL import Image


@dataclasses.dataclass(frozen=True)
class AppliedOp:
    """Record of a single applied augmentation step.

    Attributes:
        name:     Short identifier, e.g. 'jpeg', 'gaussian_noise', 'flip_h'.
        params:   Exact parameters used (quality=70, std=0.05, …).
        severity: Normalized severity in [0, 1].  0 = no effect, 1 = maximum
                  degradation within the family.
    """
    name: str
    params: dict
    severity: float


@dataclasses.dataclass
class AugmentResult:
    """Return type for every public augmentation function.

    Attributes:
        image:   Augmented PIL RGB image.
        mask:    Augmented PIL 'L' mask if applicable, else None.
        applied: Tuple of AppliedOp records for every step that ran.
    """
    image:   Image.Image
    mask:    Optional[Image.Image]
    applied: Tuple[AppliedOp, ...]


# Top-level re-exports for the common cases.  Sub-modules are still
# importable directly if a caller wants to be explicit.
def __getattr__(name: str):
    # Lazy: avoid eager import cycles (corruptions / light / degradation
    # all import from this module's dataclasses above).
    if name in ("CorruptionSpec", "apply_corruption", "sample_corruption_spec",
                "sample_distinct_corruption_specs", "make_invariance_pair"):
        from lab_utils.data.augment import corruptions
        return getattr(corruptions, name)
    if name in ("apply_jpeg", "apply_gaussian_noise", "apply_resize_jitter",
                "apply_flip_h", "apply_light_augmentations"):
        from lab_utils.data.augment import light
        return getattr(light, name)
    if name in ("build_degradation_example", "RegionRegime",
                "DegradationExample"):
        from lab_utils.data.augment import degradation
        return getattr(degradation, name)
    raise AttributeError(f"module 'lab_utils.data.augment' has no attribute {name!r}")


__all__ = [
    'AppliedOp', 'AugmentResult',
    # Lazy-loaded; the names also resolve via __getattr__:
    'CorruptionSpec', 'apply_corruption', 'sample_corruption_spec',
    'sample_distinct_corruption_specs', 'make_invariance_pair',
    'apply_jpeg', 'apply_gaussian_noise', 'apply_resize_jitter',
    'apply_flip_h', 'apply_light_augmentations',
    'build_degradation_example', 'RegionRegime', 'DegradationExample',
]
