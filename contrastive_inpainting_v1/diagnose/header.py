"""Self-describing log header.

Emits enough information at the top of every run for someone reading the log
alone to understand the experiment:
  - script identity, git sha, seed
  - checkpoint path, epoch, model dims
  - image preprocessing (size, patch grid, IM-net norm)
  - area_tier boundaries
  - gtcrop area sweep per area_tier
  - swin (scale, stride) combos + window-set hash for a probe image
  - WINDOW HASH UNIQUENESS ASSERTION (catches the scale=0.7≡0.5 bug class)
  - bce thresholds (main + alts) for [deploy]/[fp]/[zoom]
  - reals subsampling info
  - pass id → oracle-knowledge description, one line per pass
  - metric definition lines (f1_pixel, f1_deploy, ...)
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

from lab_utils.logging.text import log_line

from .passes import common
from .passes.gtcrop import area_key
from .passes.swin import swin_key


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def emit_header(
    *,
    script_name: str,
    seed: int,
    ckpt_path: str,
    ckpt_epoch,
    contrastive_dim: int,
    pool_hidden: int,
    image_size: int,
    n_patch_per_side: int,
    gt_patch_threshold: float,
    bucket_thresholds: Tuple[float, float],
    gtcrop_buckets_to_areas: Dict[str, List[float]],
    swin_combos: List[Tuple[float, float]],
    tau_win: float,
    taus_image_level: List[float],
    reals_subsample_rate: float,
    reals_seed: int,
    splices_n: Dict[str, int],
    reals_full_n: Dict[str, int],
    reals_sampled_n: Dict[str, int],
    probe_image_size: Tuple[int, int],
    cmd: str,
) -> None:
    """Emit the full self-describing header block."""
    log_line(
        f"[cfg] script={script_name} git={_git_sha()} seed={seed} "
        f"cmd={cmd}"
    )
    log_line(
        f"[cfg] checkpoint={ckpt_path} epoch={ckpt_epoch} "
        f"contrastive_dim={contrastive_dim} pool_hidden={pool_hidden}"
    )
    log_line(
        f"[cfg] image_size={image_size} patches_per_side={n_patch_per_side} "
        f"patch_size_px={image_size // n_patch_per_side} "
        f"gt_patch_threshold={gt_patch_threshold:.3f} "
        f"f1_at_pixel_resolution=true"
    )
    s_lo, s_hi = bucket_thresholds
    log_line(
        f"[buckets] small=<{s_lo:.2f} medium=[{s_lo:.2f},{s_hi:.2f}) large=>={s_hi:.2f} "
        f"(by GT splice area fraction of full image)"
    )

    # gtcrop area sweep per area_tier.
    for area_tier in ("small", "medium", "large"):
        areas = gtcrop_buckets_to_areas.get(area_tier, [])
        if areas:
            ks = [area_key(a) for a in areas]
            log_line(
                f"[cfg] gtcrop area_tier={area_tier} areas={[round(a, 3) for a in areas]} "
                f"area_keys={ks} convention=area_frac=side**2/(H*W) crop=square_centered_on_gt_centroid"
            )
        else:
            log_line(f"[cfg] gtcrop area_tier={area_tier} areas=[] (skipped by design)")

    # swin combos + window-set hash uniqueness.
    log_line(
        f"[cfg] swin tau_win={tau_win:+.3f} max_windows=uncapped n_combos={len(swin_combos)}"
    )
    log_line(
        f"[windows] NOTE: per-image n_windows depends on the image's resolution and "
        f"aspect ratio. The lines below are a SINGLE FIXED PROBE used only to (a) hash "
        f"each (scale, stride) window-coord set and (b) assert no two combos collide on "
        f"the same probe. Actual per-image distributions are emitted as [windows] dataset "
        f"lines after each split runs."
    )

    # Generate windows on a probe image for each combo and hash.
    Wp, Hp = int(probe_image_size[0]), int(probe_image_size[1])
    hashes: Dict[str, str] = {}
    for scale, stride in swin_combos:
        wins = common.window_grid(
            source_size=(Wp, Hp),
            scale=float(scale), stride_frac=float(stride),
            n_patch_per_side=n_patch_per_side,
        )
        h = common.window_set_hash(wins)
        side = wins[0][2] if wins else 0
        hashes[f"s={scale:.2f},t={stride:.2f}"] = h
        log_line(
            f"[windows] probe scale={scale:.2f} stride={stride:.2f} probe_size={Wp}x{Hp} "
            f"probe_n_windows={len(wins)} probe_side_px={side} hash={h}"
        )

    distinct = len(set(hashes.values())) == len(hashes)
    log_line(
        f"[windows] ASSERT all (scale, stride) probe hashes distinct: "
        f"{'OK' if distinct else 'FAIL'}  "
        f"({len(set(hashes.values()))} unique / {len(hashes)} combos)"
    )
    if not distinct:
        # Group identical hashes for diagnosis.
        from collections import defaultdict
        groups = defaultdict(list)
        for k, v in hashes.items():
            groups[v].append(k)
        for h, ks in groups.items():
            if len(ks) > 1:
                log_line(f"[windows] FAIL  hash={h}  collides_for={ks}")

    # BCE thresholds.
    log_line(
        f"[cfg] bce_taus_image_level={taus_image_level} "
        f"(used by [deploy], [fp], [zoom]) "
        f"bce_tau_win={tau_win:+.3f} (used for swin window selection)"
    )

    # Subsampling.
    log_line(
        f"[cfg] reals_subsample_rate={reals_subsample_rate:.2f} reals_seed={reals_seed}"
    )
    for split in sorted(set(list(splices_n.keys()) + list(reals_full_n.keys()))):
        log_line(
            f"[cfg] split={split} n_splices={splices_n.get(split, 0)} "
            f"n_reals_full={reals_full_n.get(split, 0)} "
            f"n_reals_sampled={reals_sampled_n.get(split, 0)}"
        )

    # Pass enumeration (oracle scoping).
    log_line("[passes] full_pure               gt_used=none           applies_to=all_images")
    log_line("[passes] full_ceil               gt_used=polarity_only  applies_to=splice_only")
    for area_tier, areas in gtcrop_buckets_to_areas.items():
        for area in areas:
            k = area_key(area)
            log_line(f"[passes] gtcrop_{k}_pure          gt_used=crop_region       applies_to=splice_area_tier={area_tier}")
            log_line(f"[passes] gtcrop_{k}_ceil          gt_used=region+polarity   applies_to=splice_area_tier={area_tier}")
    for scale, stride in swin_combos:
        k = swin_key(scale, stride)
        log_line(f"[passes] swin_{k}_pure            gt_used=none           applies_to=all_images")

    # Metric definitions.
    log_line(
        "[metric_defs] f1_pixel = 2*|pred ∩ gt|/(|pred|+|gt|) at original-image pixel resolution. "
        "pred = patch_grid_to_pixel_mask(28x28 partition, bbox=pass_operating_region). "
        "pixels outside bbox forced False — splice missed by a crop = FN, no inflation."
    )
    log_line(
        "[metric_defs] f1_deploy = f1_pixel with gate: if image-level bce_logit < tau, "
        "pred is forced empty (all GT becomes FN). Reported per (pass, tau, bucket)."
    )
    log_line(
        "[metric_defs] polarity_inverted = the chosen cluster is the LARGER one "
        "(so splice is the majority cluster in this partition; the smaller-cluster legacy rule would have been wrong)."
    )
    log_line(
        "[metric_defs] ceil_inverted_rate = rate at which the ceil polarity (F1-max vs GT) "
        "needed to pick the larger cluster. High rate on a area_tier = polarity rule unusable there."
    )
    log_line(
        "[metric_defs] swin polarity_agreement = fraction of source-pixel area where, "
        "across overlapping BCE-positive window pairs, both windows' projected predictions match. "
        "Drops at stride<1 indicate OR-aggregation of disagreeing partitions (stride-bug pin)."
    )
    log_line(
        "[metric_defs] bce_win categories: clean_pos (logit>=tau & splice_frac>=0.5), "
        "mixed_pos (logit>=tau & 0<splice_frac<0.5 — sliver), "
        "false_pos (logit>=tau & splice_frac==0), "
        "missed_pos (logit<tau & splice_frac>0 — BCE missed splice), "
        "clean_neg (logit<tau & splice_frac==0)."
    )
    log_line(
        "[metric_defs] [loc] = localization F1 over all splices in bucket; "
        "for swin passes this INCLUDES BCE-miss cases as F1=0 (production behavior). "
        "[loc_when_fired] = same but filtered to rows where at least one window "
        "fired BCE — separates localization quality from image-level BCE recall. "
        "Compare the two to see how much of swin's F1 deficit is BCE recall vs "
        "actual localization quality."
    )
    log_line(
        "[metric_defs] gtcrop in_crop_splice_frac = fraction of pixels inside the crop "
        "that are GT splice. For apples-to-apples comparison across buckets, the gtcrop "
        "area that matches splice density between small and medium is the meaningful one — "
        "if F1 still diverges at matched in_crop_splice_frac, the divergence is "
        "model/quantization/annotation, not crop choice."
    )
