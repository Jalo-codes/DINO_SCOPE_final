"""Summary emitters: [loc], [bce_win], [deploy], [fp], [zoom].

Every line is self-describing: explicit pass name, oracle suffix, area_tier, n,
and the metric. No abbreviations. Column order is fixed per tag.

Reader contract: someone reading the log alone should be able to reproduce
the experiment from the header + summary lines.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

from lab_utils.logging.text import log_line

from . import metrics
from .passes.gtcrop import area_key
from .passes.swin import swin_key


# ----------------------------------------------------------------------------
# [loc] — per-pass localization F1, per area_tier, splices only
# ----------------------------------------------------------------------------


def emit_loc(
    rows: List[Dict],
    *,
    split: str,
    pass_id: str,
    suffix: str = "",
    fire_filter_key: str = None,
) -> None:
    """One [loc] line per area_tier for a single pass id.

    `pass_id` is the prefix in row keys, e.g. 'full_pure', 'gtcrop_a30_pure',
    'swin_s055_t10_pure'. The row keys must be `{pass_id}_f1` etc.

    If `fire_filter_key` is given (e.g. 'swin_s055_t10_n_bce_pos'), only rows
    where that key is > 0 are included. Use this for swin passes to get
    localization F1 *conditional on at least one BCE-positive window*, i.e.,
    the pure-localization quality with BCE-recall confound removed. Tag is
    [loc_when_fired] in that case.
    """
    tag = "loc_when_fired" if fire_filter_key else "loc"
    splices = [r for r in rows if not r.get("is_real", False)]
    for area_tier in ("small", "medium", "large"):
        sub = [r for r in splices if r.get("bucket") == area_tier]
        sub = [r for r in sub if _has(r, f"{pass_id}_f1")]
        n_total_in_bucket = len(sub)
        if fire_filter_key:
            sub = [r for r in sub
                   if _has(r, fire_filter_key)
                   and float(r[fire_filter_key]) > 0]
        if not sub:
            continue
        f1 = metrics.stats([r[f"{pass_id}_f1"] for r in sub])
        iou = metrics.stats([r[f"{pass_id}_iou"] for r in sub])
        prec = metrics.stats([r[f"{pass_id}_prec"] for r in sub])
        rec = metrics.stats([r[f"{pass_id}_rec"] for r in sub])
        pf = metrics.stats([r[f"{pass_id}_pred_frac"] for r in sub])
        # In-crop splice fraction for gtcrop passes (post-zoom splice density).
        in_crop_key = None
        if pass_id.startswith("gtcrop_"):
            # pass_id is 'gtcrop_aXX_pure' or 'gtcrop_aXX_ceil'; the in_crop
            # splice frac is stored under 'gtcrop_aXX_in_crop_splice_frac'.
            crop_prefix = "_".join(pass_id.split("_")[:2])  # 'gtcrop_aXX'
            in_crop_key = f"{crop_prefix}_in_crop_splice_frac"
        extras = ""
        if in_crop_key and all(_has(r, in_crop_key) for r in sub):
            icf = metrics.stats([r[in_crop_key] for r in sub])
            extras = f" in_crop_splice_frac_med={icf['med']:.4f}"
        filter_note = ""
        if fire_filter_key:
            filter_note = (f" filter={fire_filter_key}>0 "
                           f"({len(sub)}/{n_total_in_bucket} kept)")
        line = (
            f"[{tag}]{suffix} split={split} pass={pass_id:<28} area_tier={area_tier:<6} "
            f"n={f1['n']:3d}{filter_note} "
            f"f1_pixel_med={f1['med']:.4f} f1_pixel_mean={f1['mean']:.4f} f1_pixel_std={f1['std']:.4f} "
            f"iou_med={iou['med']:.4f} prec_med={prec['med']:.4f} rec_med={rec['med']:.4f} "
            f"pred_frac_med={pf['med']:.4f}{extras}"
        )
        log_line(line)


def emit_loc_polarity_compare(
    rows: List[Dict],
    *,
    split: str,
    pass_prefix: str,  # e.g., 'full' or 'gtcrop_a30'
    suffix: str = "",
) -> None:
    """Per-bucket [loc] comparison of {prefix}_pure vs {prefix}_ceil, with
    ceil_inverted_rate and attn_vs_ceil_gap.

    Splices only (ceil is undefined on reals).
    """
    splices = [r for r in rows if not r.get("is_real", False)]
    for area_tier in ("small", "medium", "large"):
        sub = [r for r in splices if r.get("bucket") == area_tier
               and _has(r, f"{pass_prefix}_pure_f1")
               and _has(r, f"{pass_prefix}_ceil_f1")]
        if not sub:
            continue
        pure_f1 = [r[f"{pass_prefix}_pure_f1"] for r in sub]
        ceil_f1 = [r[f"{pass_prefix}_ceil_f1"] for r in sub]
        inv = [bool(r.get(f"{pass_prefix}_ceil_inverted", False)) for r in sub]
        gap = [c - p for p, c in zip(pure_f1, ceil_f1)]
        s_p = metrics.stats(pure_f1)
        s_c = metrics.stats(ceil_f1)
        s_g = metrics.stats(gap)
        log_line(
            f"[loc]{suffix} split={split} pass={pass_prefix}_polarity_compare area_tier={area_tier:<6} "
            f"n={s_p['n']:3d} "
            f"pure_f1_med={s_p['med']:.4f} ceil_f1_med={s_c['med']:.4f} "
            f"pure_vs_ceil_gap_med={s_g['med']:+.4f} "
            f"ceil_inverted_rate={float(np.mean(inv)):.3f}"
        )


# ----------------------------------------------------------------------------
# [bce_win] — per-category BCE window stats (per swin combo, per area_tier)
# ----------------------------------------------------------------------------


_CATEGORIES = ("clean_pos", "mixed_pos", "false_pos", "missed_pos", "clean_neg")


def emit_windows_dataset(
    rows: List[Dict],
    *,
    split: str,
    swin_combo: tuple,
    suffix: str = "",
) -> None:
    """One [windows] dataset line per swin combo, summarising actual per-image
    window counts across the split (splices + reals).

    Reports:
      - n_windows distribution (median, mean, p25, p75) — how many windows the
        (scale, stride) combo produced per image, NOT a single probe figure.
      - n_bce_pos distribution — how many of those fired BCE at tau_win.
      - polarity_agreement distribution — among BCE-positive overlapping
        windows, fraction of overlap pixels where projected predictions match.
    """
    scale, stride = swin_combo
    key = swin_key(scale, stride)
    sub = [r for r in rows if _has(r, f"swin_{key}_n_windows")]
    if not sub:
        return
    n_wins = metrics.stats([int(r[f"swin_{key}_n_windows"]) for r in sub])
    n_bce = metrics.stats([int(r[f"swin_{key}_n_bce_pos"]) for r in sub])
    pol = metrics.stats([float(r[f"swin_{key}_polarity_agreement"]) for r in sub])
    hashes = sorted({str(r.get(f"swin_{key}_window_set_hash"))
                     for r in sub if r.get(f"swin_{key}_window_set_hash")})
    log_line(
        f"[windows]{suffix} dataset split={split} pass=swin_{key} "
        f"n_images={n_wins['n']:3d} "
        f"n_windows_med={n_wins['med']:.1f} mean={n_wins['mean']:.2f} "
        f"p25={n_wins['p25']:.1f} p75={n_wins['p75']:.1f} "
        f"n_bce_pos_med={n_bce['med']:.1f} mean={n_bce['mean']:.2f} "
        f"polarity_agreement_med={pol['med']:.3f} mean={pol['mean']:.3f} "
        f"distinct_window_set_hashes={len(hashes)}"
    )


def emit_bce_win(
    rows: List[Dict],
    *,
    split: str,
    swin_combo: tuple,  # (scale, stride_frac)
    suffix: str = "",
) -> None:
    """One [bce_win] line per (bucket × category) for the given swin combo."""
    scale, stride = swin_combo
    key = swin_key(scale, stride)
    splices = [r for r in rows if not r.get("is_real", False)]
    for area_tier in ("small", "medium", "large"):
        sub = [r for r in splices if r.get("bucket") == area_tier
               and _has(r, f"swin_{key}_n_windows")]
        if not sub:
            continue
        for cat in _CATEGORIES:
            counts = [int(r.get(f"swin_{key}_n_{cat}", 0)) for r in sub]
            logit_means = [r.get(f"swin_{key}_logit_mean_{cat}", float("nan"))
                           for r in sub]
            total_wins = int(sum(counts))
            s_c = metrics.stats(counts)
            s_l = metrics.stats(logit_means)
            log_line(
                f"[bce_win]{suffix} split={split} pass=swin_{key} area_tier={area_tier:<6} "
                f"cat={cat:<10} n_images={s_c['n']:3d} n_windows_total={total_wins:5d} "
                f"count_per_image_med={s_c['med']:.2f} count_per_image_mean={s_c['mean']:.2f} "
                f"per_image_logit_mean_med={s_l['med']:+.3f}"
            )


# ----------------------------------------------------------------------------
# [deploy] — F1 with image-level BCE gate baked in
# ----------------------------------------------------------------------------


def emit_deploy(
    rows: List[Dict],
    *,
    split: str,
    pass_id: str,
    bce_logit_key: str,  # row key holding the image-level BCE logit for this pass's input
    taus: Sequence[float],
    suffix: str = "",
) -> None:
    """For each tau, recompute F1 with the gate: if bce_logit < tau, prediction
    is forced empty (image-level FN; all GT becomes FN).

    Logs deploy F1 + image-level FNR (splices missed by the gate at tau).
    """
    splices = [r for r in rows if not r.get("is_real", False)]
    for tau in taus:
        for area_tier in ("small", "medium", "large"):
            sub = [r for r in splices if r.get("bucket") == area_tier
                   and _has(r, f"{pass_id}_f1")
                   and _has(r, bce_logit_key)]
            if not sub:
                continue
            gated_f1 = []
            n_missed = 0
            for r in sub:
                logit = float(r[bce_logit_key])
                f1 = float(r[f"{pass_id}_f1"])
                if logit >= float(tau):
                    gated_f1.append(f1)
                else:
                    gated_f1.append(0.0)  # forced-empty pred → F1=0 (gt is non-empty for splice)
                    n_missed += 1
            s = metrics.stats(gated_f1)
            fnr = float(n_missed) / float(len(sub)) if sub else 0.0
            log_line(
                f"[deploy]{suffix} split={split} pass={pass_id:<28} area_tier={area_tier:<6} "
                f"tau={tau:+.2f} n={s['n']:3d} "
                f"f1_pixel_deploy_med={s['med']:.4f} image_fnr={fnr:.3f}"
            )


# ----------------------------------------------------------------------------
# [fp] — reals: pred_frac distribution + flag rate per pass
# ----------------------------------------------------------------------------


def emit_fp(
    rows: List[Dict],
    *,
    split: str,
    pass_id: str,
    bce_logit_key: str,
    taus: Sequence[float],
    reals_subsample_rate: float = 1.0,
    suffix: str = "",
) -> None:
    """One [fp] line per pass over reals.

    Reports pred_frac distribution (any pred on real = FP), and flag_rate at
    each tau (fraction of real images BCE-flagged on this pass's input).
    """
    reals = [r for r in rows if r.get("is_real", False)
             and _has(r, f"{pass_id}_pred_frac")]
    if not reals:
        return
    n_sampled = len(reals)
    n_full = int(round(n_sampled / max(reals_subsample_rate, 1e-9)))
    pred_fracs = [r[f"{pass_id}_pred_frac"] for r in reals]
    s = metrics.stats(pred_fracs)
    parts = [
        f"[fp]{suffix} split={split} pass={pass_id:<28} "
        f"n_reals_sampled={n_sampled} (of ~{n_full}, rate={reals_subsample_rate:.2f}) "
        f"pred_frac_med={s['med']:.4f} pred_frac_mean={s['mean']:.4f} pred_frac_p75={s['p75']:.4f}"
    ]
    log_line("".join(parts))
    for tau in taus:
        if not _has(reals[0], bce_logit_key):
            continue
        flagged = sum(1 for r in reals if float(r[bce_logit_key]) >= float(tau))
        rate = float(flagged) / float(n_sampled) if n_sampled else 0.0
        log_line(
            f"[fp]{suffix} split={split} pass={pass_id:<28} "
            f"tau={tau:+.2f} flag_rate={rate:.3f} flagged={flagged}/{n_sampled}"
        )


# ----------------------------------------------------------------------------
# [zoom] — per (bucket, area) BCE flag rate on splice crops
# ----------------------------------------------------------------------------


def emit_zoom(
    rows: List[Dict],
    *,
    split: str,
    gtcrop_buckets_to_areas: Dict[str, List[float]],
    taus: Sequence[float],
    suffix: str = "",
) -> None:
    """Answer the question "when we zoom in by ratio X, what % of crops are flagged?"

    For each (bucket, area_frac), report fraction of splice images whose
    gtcrop_aXX bce_logit >= tau. Bigger area = less zoom; smaller area = more
    zoom. Splices only — reals don't get gtcrop.
    """
    splices = [r for r in rows if not r.get("is_real", False)]
    for area_tier in ("small", "medium", "large"):
        areas = gtcrop_buckets_to_areas.get(area_tier, [])
        for area in areas:
            k = area_key(area)
            sub = [r for r in splices if r.get("bucket") == area_tier
                   and _has(r, f"gtcrop_{k}_bce_logit")]
            if not sub:
                continue
            n = len(sub)
            for tau in taus:
                flagged = sum(1 for r in sub
                              if float(r[f"gtcrop_{k}_bce_logit"]) >= float(tau))
                rate = float(flagged) / float(n) if n else 0.0
                log_line(
                    f"[zoom]{suffix} split={split} area_tier={area_tier:<6} pass=gtcrop_{k} "
                    f"tau={tau:+.2f} n={n:3d} bce_flag_rate={rate:.3f} flagged={flagged}/{n}"
                )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _has(row: Dict, key: str) -> bool:
    """Row has a non-NaN, non-None value for this key."""
    if key not in row:
        return False
    v = row[key]
    if v is None:
        return False
    try:
        if isinstance(v, float) and (v != v):  # NaN check
            return False
    except Exception:
        pass
    return True
