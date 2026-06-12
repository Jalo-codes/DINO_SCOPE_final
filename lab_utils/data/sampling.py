"""lab_utils.data.sampling — deterministic subsets and per-item sample weights.

These helpers are pure (no torch, no I/O) and operate on the standard
``items`` list-of-dicts produced by the indexers.  Each item dict carries at
least:

    {
        'source':   'imd2020' | 'casia' | 'indoor' | ...,
        'kind':     'imd_real' | 'imd_splice' | 'casia_real' | 'casia_splice' | ...,
        'case_id':  str,
        'img':      str (path),
    }

The module exists because the trainers (``train.py``,
``train_multi_head_v2.py``, ``train_image_bce_v2.py``, ``train_image_bce.py``)
each ship a copy of the same balance-weight / quick-val plumbing with
subtle drift.  Keep all the variants here and let callers pass the flavor
they want via kwargs.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple



_SPLICE_KINDS: FrozenSet[str] = frozenset({"imd_splice", "casia_splice"})
_REAL_KINDS:   FrozenSet[str] = frozenset({"imd_real", "casia_real", "indoor_real"})




def is_splice(item: dict, splice_kinds: FrozenSet[str] = _SPLICE_KINDS) -> bool:
    """True if ``item['kind']`` is in the splice set."""
    return str(item.get("kind", "")) in splice_kinds


def is_real(item: dict, real_kinds: FrozenSet[str] = _REAL_KINDS) -> bool:
    """True if ``item['kind']`` is in the real set."""
    return str(item.get("kind", "")) in real_kinds


def stable_item_sort_key(item: dict) -> str:
    """Deterministic sort key from ``(source, kind, case_id, img)``.

    Used everywhere a script needs a *stable* item ordering across runs;
    do not change the field set without breaking val subsets across runs.
    """
    raw = "|".join([
        str(item.get("source", "")),
        str(item.get("kind", "")),
        str(item.get("case_id", "")),
        str(item.get("img", "")),
    ])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def splice_balance_weights(
    items: Sequence[dict],
    *,
    target_splice_frac: float = 0.5,
    splice_kinds: FrozenSet[str] = _SPLICE_KINDS,
    return_stats: bool = False,
) -> List[float] | Tuple[List[float], Dict[str, float]]:
    """Per-item sampler weights balancing splice vs single-region items.

    The default (``target_splice_frac=0.5``) reproduces the simple
    "equal total weight for splice and real" behavior used by the
    image-BCE / multi-head trainers.  Pass ``target_splice_frac=0.6`` to
    bias toward splice positives (the regime used by ``train.py``).

    Args:
        items:               List of item dicts.
        target_splice_frac:  Desired splice-positive fraction (clamped to [0, 1]).
        splice_kinds:        Kinds that count as splice positives.
        return_stats:        When True, also return a Counter-style dict
                             (matches the legacy ``train.py`` interface).

    Returns:
        Either ``weights`` (a list aligned with ``items``) or
        ``(weights, stats)`` when ``return_stats`` is True.
    """
    target = float(min(1.0, max(0.0, target_splice_frac)))
    single_frac = 1.0 - target

    buckets: List[str] = []
    for item in items:
        bucket = "splice_pos" if is_splice(item, splice_kinds) else "single_region"
        buckets.append(bucket)

    counts = Counter(buckets)
    n_splice = counts.get("splice_pos", 0)
    n_single = counts.get("single_region", 0)

    # Degenerate cases: if one side is empty, fall back to uniform weighting
    # so the sampler does not crash.  Matches the legacy behavior.
    if n_splice == 0 or n_single == 0:
        weights = [1.0] * len(items)
    else:
        class_mass = {"splice_pos": target, "single_region": single_frac}
        weights = [class_mass[b] / max(1, counts[b]) for b in buckets]

    if not return_stats:
        return weights

    stats: Dict[str, float] = dict(sorted(counts.items()))
    stats["target_splice_frac"] = target
    stats["target_single_frac"] = single_frac
    return weights, stats


def source_splice_balance_weights(
    items: Sequence[dict],
    source_fracs: Dict[str, float],
    *,
    target_splice_frac: float = 0.5,
    splice_kinds: FrozenSet[str] = _SPLICE_KINDS,
) -> Tuple[List[float], Dict[str, float]]:
    """Sampler weights that control the splice mix *by source*.

    Two-level allocation:
      1. Class balance — splice positives receive ``target_splice_frac`` of the
         total draw mass; real / single-region items share ``1 - target``.
      2. Within the splice class, the splice mass is split across sources by
         ``source_fracs`` (a ``{source: fraction}`` map, normalized to sum 1).
         Each source's share is spread uniformly over its own splice items, so
         the per-epoch draw distribution matches the requested fractions
         regardless of how many files each source contributes.

    Real items keep uniform weight (source-agnostic) summing to ``1 - target``.

    A splice source that is present in ``items`` but absent from
    ``source_fracs`` gets **zero** weight — it is excluded from training. The
    returned stats dict reports realized per-source splice item counts plus any
    such excluded sources, so the caller can warn.

    Returns ``(weights, stats)``.
    """
    target = float(min(1.0, max(0.0, target_splice_frac)))

    total_frac = sum(max(0.0, f) for f in source_fracs.values())
    fracs = ({s: max(0.0, f) / total_frac for s, f in source_fracs.items()}
             if total_frac > 0 else {})

    splice_idx_by_src: Dict[str, List[int]] = {}
    real_idx: List[int] = []
    for i, item in enumerate(items):
        if is_splice(item, splice_kinds):
            src = str(item.get("source", "unknown"))
            splice_idx_by_src.setdefault(src, []).append(i)
        else:
            real_idx.append(i)

    weights = [0.0] * len(items)

    # Splice mass, allocated per source by fraction.
    for src, idxs in splice_idx_by_src.items():
        f = fracs.get(src, 0.0)
        if f <= 0.0 or not idxs:
            continue
        w = (target * f) / len(idxs)
        for i in idxs:
            weights[i] = w

    # Real mass, uniform across all reals.
    if real_idx:
        wr = (1.0 - target) / len(real_idx)
        for i in real_idx:
            weights[i] = wr

    # Degenerate guard: if every weight collapsed to 0 (e.g. no listed source
    # had items), fall back to uniform so the sampler does not crash.
    if not any(w > 0.0 for w in weights):
        weights = [1.0] * len(items)

    excluded = sorted(s for s in splice_idx_by_src if fracs.get(s, 0.0) <= 0.0)
    stats: Dict[str, float] = {
        "target_splice_frac": target,
        "n_real": float(len(real_idx)),
    }
    for src in sorted(splice_idx_by_src):
        stats[f"n_splice[{src}]"] = float(len(splice_idx_by_src[src]))
        stats[f"frac[{src}]"] = float(fracs.get(src, 0.0))
    if excluded:
        stats["excluded_sources"] = ",".join(excluded)  # type: ignore[assignment]
    return weights, stats


def val_mix_counts(items: Sequence[dict]) -> Dict[Tuple[str, str], int]:
    """Counter over ``(source, kind)`` pairs, sorted for stable logging."""
    return dict(sorted(
        Counter(
            (str(i.get("source", "unknown")), str(i.get("kind", "unknown")))
            for i in items
        ).items(),
        key=lambda kv: str(kv[0]),
    ))


def val_source_counts(items: Sequence[dict]) -> Dict[str, int]:
    """Counter over ``source``, sorted for stable logging."""
    return dict(sorted(
        Counter(str(i.get("source", "unknown")) for i in items).items(),
        key=lambda kv: str(kv[0]),
    ))


def items_for_source(items: Iterable[dict], source: str) -> List[dict]:
    """Filter to items whose ``source`` matches."""
    return [item for item in items if str(item.get("source", "")) == str(source)]


def build_quick_val_items(
    items: Sequence[dict],
    cap: int,
) -> List[dict]:
    """Deterministic stratified val subset over ``(source, kind)``.

    Groups ``items`` by ``(source, kind)``, allocates per-group quotas
    proportional to group size, then deterministically picks within each
    group via :func:`stable_item_sort_key`.  Pass ``cap <= 0`` to disable
    capping.
    """
    if cap <= 0 or len(items) <= cap:
        return list(items)

    groups: Dict[Tuple[str, str], List[dict]] = {}
    for item in items:
        key = (str(item.get("source", "unknown")), str(item.get("kind", "unknown")))
        groups.setdefault(key, []).append(item)

    total = len(items)
    targets: Dict[Tuple[str, str], int] = {}
    remainders: List[Tuple[float, Tuple[str, str]]] = []
    assigned = 0
    for key, group_items in groups.items():
        exact = float(cap) * float(len(group_items)) / float(total)
        take = min(len(group_items), int(exact))
        if take == 0 and len(group_items) > 0:
            take = 1
        targets[key] = take
        assigned += take
        remainders.append((exact - int(exact), key))

    if assigned > cap:
        for _frac, key in sorted(remainders, key=lambda x: (x[0], str(x[1]))):
            if assigned <= cap:
                break
            if targets[key] > 1:
                targets[key] -= 1
                assigned -= 1

    if assigned < cap:
        for _frac, key in sorted(remainders, key=lambda x: (-x[0], str(x[1]))):
            if assigned >= cap:
                break
            room = len(groups[key]) - targets[key]
            if room <= 0:
                continue
            add = min(room, cap - assigned)
            targets[key] += add
            assigned += add

    chosen: List[dict] = []
    for key, group_items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ordered = sorted(group_items, key=stable_item_sort_key)
        chosen.extend(ordered[:targets[key]])

    return sorted(chosen, key=stable_item_sort_key)[:cap]


def build_case_balanced_quick_val_items(
    items: Sequence[dict],
    *,
    imd_cases: int,
    casia_pairs: int,
) -> List[dict]:
    """Quick val as N IMD cases + M CASIA pairs.

    IMD2020 contributes one real + one deterministic splice per case;
    CASIA contributes its native real/fake pair per case.  Indoor holdout
    is intentionally excluded to keep the source mix simple.
    """
    by_source_case: Dict[Tuple[str, str], List[dict]] = {}
    for item in items:
        source = str(item.get("source", "unknown"))
        case_id = str(item.get("case_id", ""))
        by_source_case.setdefault((source, case_id), []).append(item)

    chosen: List[dict] = []

    # CASIA: one real + one splice per case by construction.
    casia_cases = sorted(
        [case for (source, case), _g in by_source_case.items() if source == "casia"],
        key=lambda c: hashlib.md5(f"casia|{c}".encode("utf-8")).hexdigest(),
    )[:max(0, int(casia_pairs))]
    for case_id in casia_cases:
        group = sorted(by_source_case[("casia", case_id)], key=stable_item_sort_key)
        real = next((it for it in group if str(it.get("kind")) == "imd_real"), None)
        splice = next((it for it in group if str(it.get("kind")) == "imd_splice"), None)
        if real is not None:
            chosen.append(real)
        if splice is not None:
            chosen.append(splice)

    # IMD2020: one real + one deterministic splice representative per case.
    imd_cases_all = sorted(
        [case for (source, case), _g in by_source_case.items() if source == "imd2020"],
        key=lambda c: hashlib.md5(f"imd2020|{c}".encode("utf-8")).hexdigest(),
    )[:max(0, int(imd_cases))]
    for case_id in imd_cases_all:
        group = sorted(by_source_case[("imd2020", case_id)], key=stable_item_sort_key)
        real = next((it for it in group if str(it.get("kind")) == "imd_real"), None)
        splice = next((it for it in group if str(it.get("kind")) == "imd_splice"), None)
        if real is not None:
            chosen.append(real)
        if splice is not None:
            chosen.append(splice)

    return sorted(chosen, key=stable_item_sort_key)


def build_shared_tau_calibration_items(
    items: Sequence[dict],
    *,
    singles_per_source: int,
    splices_per_source: int,
    sources: Sequence[str] = ("imd2020", "casia"),
    single_kinds: FrozenSet[str] = frozenset({"imd_real", "indoor_real"}),
    splice_kind: str = "imd_splice",
) -> List[dict]:
    """Per-source N singles + M splices for shared-tau calibration."""
    chosen: List[dict] = []
    for source in sources:
        source_items = items_for_source(items, source)
        singles = sorted(
            [it for it in source_items if str(it.get("kind", "")) in single_kinds],
            key=stable_item_sort_key,
        )[:max(0, int(singles_per_source))]
        splices = sorted(
            [it for it in source_items if str(it.get("kind", "")) == splice_kind],
            key=stable_item_sort_key,
        )[:max(0, int(splices_per_source))]
        chosen.extend(singles)
        chosen.extend(splices)
    return sorted(chosen, key=stable_item_sort_key)


def deterministic_subsample(
    items: Sequence[dict],
    n: int,
    *,
    seed: str,
) -> List[dict]:
    """Deterministic subsample by hashing ``(seed, img_or_path)``.

    Returns a copy of ``items`` truncated to ``n``.  When ``len(items) <= n``,
    returns the full list unchanged.  Stable across runs with the same seed
    string.  Used by diagnose scripts that need a fixed-cap eval subset
    that survives reshuffling of the upstream item list.
    """
    if not items or len(items) <= n:
        return list(items)

    def _key(it: dict) -> str:
        path = it.get("img") or it.get("path") or ""
        return hashlib.md5(f"{seed}|{path}".encode("utf-8")).hexdigest()

    return sorted(items, key=_key)[:n]


def reals_subsample(
    items: Sequence[dict],
    rate: float,
    *,
    seed: str,
) -> List[dict]:
    """Keep approximately ``rate`` fraction of items deterministically.

    Wraps :func:`deterministic_subsample`.  ``rate >= 1.0`` returns the
    full list unchanged; ``rate <= 0.0`` returns an empty list (well, one
    item — kept consistent with the legacy behavior).
    """
    if not items or rate >= 1.0:
        return list(items)
    target = max(1, int(round(len(items) * float(rate))))
    return deterministic_subsample(items, target, seed=seed + "|reals")
