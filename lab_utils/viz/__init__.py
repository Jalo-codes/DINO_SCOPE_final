"""lab_utils.viz — Visualization utilities (PIL-only, no matplotlib)."""

from lab_utils.viz.composite import (
    cmap_jet,
    heatmap_rgb,
    overlay_blend,
    mask_tint,
    save_composite,
)

__all__ = [
    'cmap_jet',
    'heatmap_rgb',
    'overlay_blend',
    'mask_tint',
    'save_composite',
]
