"""lab_utils.eval — evaluation utilities for forensics experiments."""

from lab_utils.eval.metrics import f1_iou, binary_metrics, calibrate_gate_tau
from lab_utils.eval.image_bce import (
    BCEHeadAdapter,
    ImageBCEMetrics,
    collect_image_bce_logits,
    image_bce_metrics,
    run_image_bce_eval,
)
from lab_utils.eval.window_geometry import (
    axis_positions,
    capped_window_grid,
    centered_area_square,
    gt_bbox_and_centroid,
    inferred_bbox_from_patches,
    project_window_mask,
    square_expand_crop,
    window_footprint,
    window_grid,
    window_gt_patches,
    window_set_hash,
)

__all__ = [
    'f1_iou', 'binary_metrics', 'calibrate_gate_tau',
    'BCEHeadAdapter', 'ImageBCEMetrics',
    'collect_image_bce_logits', 'image_bce_metrics', 'run_image_bce_eval',
    'axis_positions', 'capped_window_grid', 'centered_area_square',
    'gt_bbox_and_centroid', 'inferred_bbox_from_patches',
    'project_window_mask', 'square_expand_crop', 'window_footprint',
    'window_grid', 'window_gt_patches', 'window_set_hash',
]
