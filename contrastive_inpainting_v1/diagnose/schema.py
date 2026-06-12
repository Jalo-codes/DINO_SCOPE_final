"""ROW_KEYS contract: every row must have every key.

The point of this module is to make "schema drift" (a pass silently failing
and leaving keys absent) a hard error before any summary runs. Builds the
ROW_KEYS list from the active CLI args (swin combos, gtcrop areas, etc.) so
that the contract is dynamic but enforced.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

from .passes.gtcrop import area_key
from .passes.swin import swin_key


# Universal keys present on every row regardless of args.
_UNIVERSAL_KEYS = [
    # Identity / bookkeeping.
    "path", "split", "source", "kind",
    "is_real", "bucket", "gt_frac",
    # full pass — _pure always; _ceil only on splices (NaN-stored on reals).
    "full_pure_f1", "full_pure_iou", "full_pure_prec", "full_pure_rec",
    "full_pure_pred_frac", "full_pure_inverted",
    "full_ceil_f1", "full_ceil_iou", "full_ceil_prec", "full_ceil_rec",
    "full_ceil_pred_frac", "full_ceil_inverted",
    "full_bce_logit", "full_pool_attention_mean",
]


def build_row_keys(
    *,
    swin_combos: Iterable[Tuple[float, float]],
    gtcrop_buckets_to_areas: Dict[str, List[float]],
) -> List[str]:
    """Build the full list of expected keys for a row.

    Args:
        swin_combos: iterable of (scale, stride_frac) tuples.
        gtcrop_buckets_to_areas: dict area_tier → list of area_frac to sweep.
            Used to enumerate area suffixes; rows for a area_tier without an area
            simply have NaN in that area's columns.
    """
    keys: List[str] = list(_UNIVERSAL_KEYS)

    # gtcrop keys (per area_frac that appears in ANY bucket).
    all_areas = sorted({a for areas in gtcrop_buckets_to_areas.values() for a in areas})
    for area in all_areas:
        k = area_key(area)
        for rule in ("pure", "ceil"):
            keys.extend([
                f"gtcrop_{k}_{rule}_f1",
                f"gtcrop_{k}_{rule}_iou",
                f"gtcrop_{k}_{rule}_prec",
                f"gtcrop_{k}_{rule}_rec",
                f"gtcrop_{k}_{rule}_pred_frac",
                f"gtcrop_{k}_{rule}_inverted",
            ])
        keys.extend([
            f"gtcrop_{k}_bce_logit",
            f"gtcrop_{k}_in_crop_splice_frac",
            f"gtcrop_{k}_oncrop_pixel_share",
            f"gtcrop_{k}_crop_side_px",
            f"gtcrop_{k}_area_frac",
        ])

    # swin keys (per (scale, stride) combo).
    for scale, stride in swin_combos:
        k = swin_key(float(scale), float(stride))
        keys.extend([
            f"swin_{k}_pure_f1",
            f"swin_{k}_pure_iou",
            f"swin_{k}_pure_prec",
            f"swin_{k}_pure_rec",
            f"swin_{k}_pure_pred_frac",
            f"swin_{k}_n_windows",
            f"swin_{k}_n_bce_pos",
            f"swin_{k}_window_set_hash",
            f"swin_{k}_polarity_agreement",
            f"swin_{k}_bce_logit_max",
            f"swin_{k}_bce_logit_mean",
            f"swin_{k}_bce_logit_max_pos",
            f"swin_{k}_bce_logit_mean_pos",
            f"swin_{k}_scale",
            f"swin_{k}_stride_frac",
            # per-window stratification counts (per category, per row).
            f"swin_{k}_n_clean_pos",
            f"swin_{k}_n_mixed_pos",
            f"swin_{k}_n_false_pos",
            f"swin_{k}_n_missed_pos",
            f"swin_{k}_n_clean_neg",
            # per-category bce logit aggregates (mean over windows in cat).
            f"swin_{k}_logit_mean_clean_pos",
            f"swin_{k}_logit_mean_mixed_pos",
            f"swin_{k}_logit_mean_false_pos",
            f"swin_{k}_logit_mean_missed_pos",
            f"swin_{k}_logit_mean_clean_neg",
        ])

    return keys


def validate_row(row: Dict[str, Any], expected_keys: List[str]) -> None:
    """Hard-fail with the missing key list if any expected key is absent.

    NaN values are allowed (they explicitly mean "this pass didn't apply to
    this image" — e.g. ceil F1 on reals); absent keys are not.
    """
    missing = [k for k in expected_keys if k not in row]
    if missing:
        raise KeyError(
            f"Row schema violation: missing {len(missing)} keys (first 8): "
            f"{missing[:8]}\nPath={row.get('path')!r}"
        )


def nan_init_row(expected_keys: List[str]) -> Dict[str, Any]:
    """Initialize a row with NaN/None for every expected key.

    Passes overwrite the keys they own; anything left NaN at validate-time is
    a deliberate "didn't apply" (e.g. ceil_* on reals, gtcrop_* on large bucket).
    """
    row: Dict[str, Any] = {}
    for k in expected_keys:
        if k.endswith("_inverted") or k.endswith("_set_hash") or k in {
            "path", "split", "source", "kind"
        }:
            row[k] = None
        elif k == "is_real":
            row[k] = False
        elif k == "bucket":
            row[k] = ""
        else:
            row[k] = float("nan")
    return row
