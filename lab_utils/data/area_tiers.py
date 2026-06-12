"""Canonical area-tier labels for reporting splice-size behavior."""

from __future__ import annotations

import math
from typing import Iterable, Tuple


AREA_TIER_LABELS: Tuple[str, ...] = ("tiny", "small", "medium", "large")
AREA_TIER_EDGES: Tuple[float, float, float] = (0.05, 0.15, 0.30)


def area_tier(area_frac: float) -> str:
    """Map a splice area fraction to a reporting tier.

    This is for evaluation/reporting only. It is intentionally separate from
    the bucket-size detector helpers in :mod:`lab_utils.data.sampling`.
    """

    try:
        a = float(area_frac)
    except (TypeError, ValueError):
        a = 0.0
    if not math.isfinite(a) or a <= 0.0:
        return "tiny"
    if a <= AREA_TIER_EDGES[0]:
        return "tiny"
    if a < AREA_TIER_EDGES[1]:
        return "small"
    if a < AREA_TIER_EDGES[2]:
        return "medium"
    return "large"


def area_tier_labels(*, include_real: bool = False) -> Tuple[str, ...]:
    """Return tier labels in stable reporting order."""

    return (("real",) if include_real else ()) + AREA_TIER_LABELS


def with_area_tier(items: Iterable[dict], *, key: str = "area_tier") -> list[dict]:
    """Return shallow item copies annotated with ``key`` from area metadata."""

    out = []
    for item in items:
        copied = dict(item)
        copied[key] = area_tier(copied.get("blob_area_actual", 0.0))
        out.append(copied)
    return out
