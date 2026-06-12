"""diagnose — clean rewrite of splice-localization diagnostics.

Read `contrastive_inpainting_v1/diagnose/__init__.py` for the design contract.

Usage:
    python -m contrastive_inpainting_v1.scripts.diagnose \
        --imd2020_root /media/ssd/DINO_SCOPE_DATA/IMD2020 \
        --casia_root /media/ssd/DINO_SCOPE_DATA/casia \
        --checkpoint_dir /media/ssd/runs/multi_head_v2/joint_swin_lambda2_bce_zoom \
        --eval_max_items 500 \
        --reals_subsample_rate 0.25 \
        --swin_scales 0.55 0.70 0.85 \
        --swin_stride_fracs 1.0 0.5 \
        --tau_win 0.5 \
        --taus_image_level 0.0 0.5 \
        --output_log /media/ssd/runs/.../diagnose.log
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
from PIL import Image

from lab_utils.data.dataset import LabDataset
from lab_utils.data.loaders import LoaderConfig, build_eval_loader
from lab_utils.data.sampling import deterministic_subsample, reals_subsample
from lab_utils.logging.run_dir import build_run_dir as _build_run_dir
from lab_utils.logging.text import install_log, log_line
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.train.checkpoint import find_latest_checkpoint, load as ckpt_load

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.experiments.imd2020_contrastive import IMD2020ContrastiveSpec

from contrastive_inpainting_v1.diagnose import metrics as _metrics
from contrastive_inpainting_v1.diagnose import project as _project
from contrastive_inpainting_v1.diagnose import schema as _schema
from contrastive_inpainting_v1.diagnose import summarize as _summarize
from contrastive_inpainting_v1.diagnose import header as _header
from contrastive_inpainting_v1.diagnose.passes import full as pass_full
from contrastive_inpainting_v1.diagnose.passes import gtcrop as pass_gtcrop
from contrastive_inpainting_v1.diagnose.passes import swin as pass_swin


_SPLICE_KINDS = frozenset({"imd_splice", "casia_splice"})
_REAL_KINDS = frozenset({"imd_real", "casia_real"})
_BUCKET_LO = 0.15
_BUCKET_HI = 0.30


def _bucket(a: float) -> str:
    if a < _BUCKET_LO:
        return "small"
    if a < _BUCKET_HI:
        return "medium"
    return "large"


def _resolve_latest_checkpoint(checkpoint_dir: str) -> str:
    path = find_latest_checkpoint(checkpoint_dir)
    if path is None:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir!r}")
    return path


def _infer_head_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    contrastive_dim = 0
    pool_hidden = 0
    if "contrastive_proj.weight" in state_dict:
        contrastive_dim = int(state_dict["contrastive_proj.weight"].shape[0])
    if "pool.V.weight" in state_dict:
        pool_hidden = int(state_dict["pool.V.weight"].shape[0])
    return contrastive_dim, pool_hidden


_subsample_items = deterministic_subsample
_reals_subsample = reals_subsample


def _load_pixel_gt(mask_path: str, *, H: int, W: int) -> Optional[np.ndarray]:
    """Open a mask path and return (H, W) bool, NN-resized if needed."""
    if not mask_path or not os.path.isfile(mask_path):
        return None
    try:
        m = Image.open(mask_path).convert("L")
        return _project.gt_pixel_mask(m, full_size=(H, W))
    except Exception as exc:
        log_line(f"[loc] WARN mask load failed for {mask_path!r}: {exc}")
        return None


# ----------------------------------------------------------------------------
# Per-image dispatch: run all passes, build the row
# ----------------------------------------------------------------------------


def _process_one_image(
    *,
    model,
    device: torch.device,
    cfg: Config,
    args,
    meta: Dict,
    split: str,
    source_image: Image.Image,
    gt_HW: Optional[np.ndarray],
    is_real: bool,
    bucket: str,
    gt_frac: float,
    swin_combos: List[Tuple[float, float]],
    gtcrop_areas_for_this_bucket: List[float],
    row_keys: List[str],
) -> Dict:
    """Run every applicable pass on one image, build the row, validate it."""
    row = _schema.nan_init_row(row_keys)
    row["path"] = str(meta.get("path", "") or meta.get("img", "") or "")
    row["split"] = split
    row["source"] = str(meta.get("source", "") or "")
    row["kind"] = str(meta.get("kind", "") or "")
    row["is_real"] = bool(is_real)
    row["bucket"] = str(bucket)
    row["gt_frac"] = float(gt_frac)

    image_size = int(cfg.resolution.image_size)
    n_patch = int(cfg.resolution.num_patches_per_side)
    W_src, H_src = source_image.size
    H, W = int(H_src), int(W_src)

    from lab_utils.eval.decode_cli import decode_spec_from_args
    decode_spec = decode_spec_from_args(args)

    # ------- full pass (all images; ceil only when gt_HW present) -------
    full_out = pass_full.run_full(
        model, source_image, device,
        image_size=image_size, n_patch_per_side=n_patch,
        imagenet_mean=cfg.IMAGENET_MEAN, imagenet_std=cfg.IMAGENET_STD,
        gt_HW=gt_HW, decode_spec=decode_spec,
    )
    row["full_bce_logit"] = full_out["full_bce_logit"]
    row["full_pool_attention_mean"] = full_out["full_pool_attention_mean"]
    for rule in ("pure", "ceil"):
        mask_key = f"full_{rule}_mask"
        if mask_key not in full_out:
            continue
        pred_HW = full_out[mask_key]
        if gt_HW is not None:
            m = _metrics.f1_pixel(pred_HW, gt_HW)
        else:
            # Real image — pred_HW vs all-False GT.
            m = _metrics.f1_pixel(pred_HW, np.zeros_like(pred_HW, dtype=bool))
        row[f"full_{rule}_f1"] = m["f1"]
        row[f"full_{rule}_iou"] = m["iou"]
        row[f"full_{rule}_prec"] = m["prec"]
        row[f"full_{rule}_rec"] = m["rec"]
        row[f"full_{rule}_pred_frac"] = m["pred_frac"]
        row[f"full_{rule}_inverted"] = bool(full_out[f"full_{rule}_inverted"])

    # ------- gtcrop passes (splices only; area_tier dispatches areas) -------
    if (not is_real) and (gt_HW is not None) and gt_HW.any():
        for area in gtcrop_areas_for_this_bucket:
            try:
                gc_out = pass_gtcrop.run_gtcrop(
                    model, source_image, device,
                    area_frac=float(area), gt_HW=gt_HW,
                    image_size=image_size, n_patch_per_side=n_patch,
                    imagenet_mean=cfg.IMAGENET_MEAN, imagenet_std=cfg.IMAGENET_STD,
                    decode_spec=decode_spec,
                )
            except Exception as exc:
                log_line(
                    f"[loc] WARN gtcrop area={area} failed for {row['path']!r}: {exc}"
                )
                continue
            k = pass_gtcrop.area_key(area)
            row[f"gtcrop_{k}_bce_logit"] = gc_out[f"gtcrop_{k}_bce_logit"]
            row[f"gtcrop_{k}_in_crop_splice_frac"] = gc_out[f"gtcrop_{k}_in_crop_splice_frac"]
            row[f"gtcrop_{k}_oncrop_pixel_share"] = gc_out[f"gtcrop_{k}_oncrop_pixel_share"]
            row[f"gtcrop_{k}_crop_side_px"] = gc_out[f"gtcrop_{k}_crop_side_px"]
            row[f"gtcrop_{k}_area_frac"] = gc_out[f"gtcrop_{k}_area_frac"]
            for rule in ("pure", "ceil"):
                mask_key = f"gtcrop_{k}_{rule}_mask"
                if mask_key not in gc_out:
                    continue
                pred_HW = gc_out[mask_key]
                m = _metrics.f1_pixel(pred_HW, gt_HW)
                row[f"gtcrop_{k}_{rule}_f1"] = m["f1"]
                row[f"gtcrop_{k}_{rule}_iou"] = m["iou"]
                row[f"gtcrop_{k}_{rule}_prec"] = m["prec"]
                row[f"gtcrop_{k}_{rule}_rec"] = m["rec"]
                row[f"gtcrop_{k}_{rule}_pred_frac"] = m["pred_frac"]
                row[f"gtcrop_{k}_{rule}_inverted"] = bool(gc_out[f"gtcrop_{k}_{rule}_inverted"])

    # ------- swin passes (all images) -------
    for scale, stride in swin_combos:
        try:
            sw_out = pass_swin.run_swin(
                model, source_image, device,
                scale=float(scale), stride_frac=float(stride),
                image_size=image_size, n_patch_per_side=n_patch,
                imagenet_mean=cfg.IMAGENET_MEAN, imagenet_std=cfg.IMAGENET_STD,
                tau_win=float(args.tau_win),
                gt_HW=gt_HW if gt_HW is not None else np.zeros((H, W), dtype=bool),
                decode_spec=decode_spec,
            )
        except Exception as exc:
            log_line(
                f"[loc] WARN swin scale={scale} stride={stride} failed for {row['path']!r}: {exc}"
            )
            continue
        k = pass_swin.swin_key(float(scale), float(stride))
        pred_HW = sw_out[f"swin_{k}_pure_mask"]
        if gt_HW is not None:
            m = _metrics.f1_pixel(pred_HW, gt_HW)
        else:
            m = _metrics.f1_pixel(pred_HW, np.zeros_like(pred_HW, dtype=bool))
        row[f"swin_{k}_pure_f1"] = m["f1"]
        row[f"swin_{k}_pure_iou"] = m["iou"]
        row[f"swin_{k}_pure_prec"] = m["prec"]
        row[f"swin_{k}_pure_rec"] = m["rec"]
        row[f"swin_{k}_pure_pred_frac"] = m["pred_frac"]
        row[f"swin_{k}_n_windows"] = int(sw_out[f"swin_{k}_n_windows"])
        row[f"swin_{k}_n_bce_pos"] = int(sw_out[f"swin_{k}_n_bce_pos"])
        row[f"swin_{k}_window_set_hash"] = str(sw_out[f"swin_{k}_window_set_hash"])
        row[f"swin_{k}_polarity_agreement"] = float(sw_out[f"swin_{k}_polarity_agreement"])
        row[f"swin_{k}_bce_logit_max"] = float(sw_out[f"swin_{k}_bce_logit_max"])
        row[f"swin_{k}_bce_logit_mean"] = float(sw_out[f"swin_{k}_bce_logit_mean"])
        row[f"swin_{k}_bce_logit_max_pos"] = float(sw_out[f"swin_{k}_bce_logit_max_pos"])
        row[f"swin_{k}_bce_logit_mean_pos"] = float(sw_out[f"swin_{k}_bce_logit_mean_pos"])
        row[f"swin_{k}_scale"] = float(sw_out[f"swin_{k}_scale"])
        row[f"swin_{k}_stride_frac"] = float(sw_out[f"swin_{k}_stride_frac"])
        # Per-category counts + per-image-mean logit per category.
        per_win = sw_out[f"swin_{k}_per_window"]
        for cat in ("clean_pos", "mixed_pos", "false_pos", "missed_pos", "clean_neg"):
            cat_entries = [e for e in per_win if e["category"] == cat]
            row[f"swin_{k}_n_{cat}"] = int(len(cat_entries))
            if cat_entries:
                row[f"swin_{k}_logit_mean_{cat}"] = float(
                    np.mean([e["bce_logit"] for e in cat_entries])
                )

    _schema.validate_row(row, row_keys)
    return row


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imd2020_root", type=str, default=None)
    p.add_argument("--casia_root", type=str, default=None)
    p.add_argument("--indoor_root", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--output_log", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--eval_max_items", type=int, default=99999)
    p.add_argument("--reals_subsample_rate", type=float, default=0.25)
    p.add_argument("--subsample_seed", type=str, default="diag_v2")
    p.add_argument("--swin_scales", nargs="*", type=float, default=[0.55, 0.70, 0.85])
    p.add_argument("--swin_stride_fracs", nargs="*", type=float, default=[1.0, 0.5])
    p.add_argument("--tau_win", type=float, default=0.5,
                   help="Per-window BCE threshold for swin window selection (default 0.5).")
    p.add_argument("--taus_image_level", nargs="*", type=float, default=[0.0, 0.5],
                   help="Image-level BCE thresholds used by [deploy]/[fp]/[zoom].")
    p.add_argument("--gt_patch_threshold", type=float, default=0.06,
                   help="Patch-grid GT threshold (legacy compat; not used by pixel F1 directly).")
    p.add_argument("--probe_image_w", type=int, default=384)
    p.add_argument("--probe_image_h", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    from lab_utils.eval.decode_cli import add_decode_args
    add_decode_args(p)
    args = p.parse_args()
    from contrastive_inpainting_v1.pipeline.cli import apply_path_defaults
    apply_path_defaults(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Checkpoint.
    ckpt_path = args.checkpoint or _resolve_latest_checkpoint(args.checkpoint_dir)
    # Logging: when the user passes --output_log, honor it (back-compat).
    # Otherwise land under <checkpoint_dir>/logs/<ts>_<git>_diagnose-v2/run.log
    # so successive runs do not stomp on each other.
    if args.output_log:
        install_log(args.output_log)
    else:
        _ckpt_dir_abs = os.path.abspath(args.checkpoint_dir)
        _rd = _build_run_dir(
            os.path.dirname(_ckpt_dir_abs),
            os.path.basename(_ckpt_dir_abs),
            role='diagnose-v2',
        )
        install_log(str(_rd.log_path))

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    cfg = Config()

    # Build items.
    spec = IMD2020ContrastiveSpec(
        imd2020_root=args.imd2020_root,
        casia_root=args.casia_root,
        indoor_root=args.indoor_root,
    )
    _, val_items = spec.build_items(cfg)
    by_split = {
        "imd_val": [it for it in val_items if it.get("source", "") == "imd2020"],
        "casia_val": [it for it in val_items if it.get("source", "") == "casia"],
    }
    splices_n: Dict[str, int] = {}
    reals_full_n: Dict[str, int] = {}
    reals_sampled_n: Dict[str, int] = {}
    items_per_split: Dict[str, list] = {}
    for split, items in by_split.items():
        splices = [it for it in items if it.get("kind") in _SPLICE_KINDS]
        reals = [it for it in items if it.get("kind") in _REAL_KINDS]
        splices = _subsample_items(splices, int(args.eval_max_items), seed=args.subsample_seed)
        reals_sub = _reals_subsample(reals, float(args.reals_subsample_rate),
                                      seed=args.subsample_seed)
        splices_n[split] = len(splices)
        reals_full_n[split] = len(reals)
        reals_sampled_n[split] = len(reals_sub)
        items_per_split[split] = splices + reals_sub

    # Model.
    ckpt = ckpt_load(ckpt_path)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    contrastive_dim, pool_hidden = _infer_head_dims(state_dict)
    if contrastive_dim <= 0 or pool_hidden <= 0:
        raise RuntimeError(
            f"diagnostic requires joint checkpoint; got contrastive_dim={contrastive_dim}, "
            f"pool_hidden={pool_hidden}"
        )
    model = build_multi_head_detector(
        model_name=cfg.MODEL_NAME,
        resolution=cfg.resolution,
        lora_rank=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        lora_targets=cfg.LORA_TARGETS,
        contrastive_dim=contrastive_dim,
        pool_hidden=pool_hidden,
        device=device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Compose configuration objects used by header/passes/summary.
    swin_combos = [(float(s), float(t))
                   for s in args.swin_scales for t in args.swin_stride_fracs]
    # gtcrop areas per area_tier (v2 spec; not args yet — fixed).
    gtcrop_buckets_to_areas: Dict[str, List[float]] = {
        "small": [0.30, 0.45, 0.60, 0.75],
        "medium": [0.50, 0.65, 0.80, 0.90],
        "large": [],
    }

    # Row schema.
    row_keys = _schema.build_row_keys(
        swin_combos=swin_combos,
        gtcrop_buckets_to_areas=gtcrop_buckets_to_areas,
    )

    # Header.
    cmd = " ".join(sys.argv)
    _header.emit_header(
        script_name="diagnose.py",
        seed=int(args.seed),
        ckpt_path=ckpt_path,
        ckpt_epoch=ckpt.get("epoch", "?"),
        contrastive_dim=int(contrastive_dim),
        pool_hidden=int(pool_hidden),
        image_size=int(cfg.resolution.image_size),
        n_patch_per_side=int(cfg.resolution.num_patches_per_side),
        gt_patch_threshold=float(args.gt_patch_threshold),
        bucket_thresholds=(_BUCKET_LO, _BUCKET_HI),
        gtcrop_buckets_to_areas=gtcrop_buckets_to_areas,
        swin_combos=swin_combos,
        tau_win=float(args.tau_win),
        taus_image_level=list(args.taus_image_level),
        reals_subsample_rate=float(args.reals_subsample_rate),
        reals_seed=int(args.seed),
        splices_n=splices_n,
        reals_full_n=reals_full_n,
        reals_sampled_n=reals_sampled_n,
        probe_image_size=(int(args.probe_image_w), int(args.probe_image_h)),
        cmd=cmd,
    )

    # Run per split.
    for split, items in items_per_split.items():
        if not items:
            continue
        log_line(f"[loc] split={split} START n_items={len(items)}")
        rows: List[Dict] = []
        for idx, meta in enumerate(items):
            kind = str(meta.get("kind", ""))
            is_real = kind in _REAL_KINDS
            path = str(meta.get("img", "") or meta.get("path", "") or "")
            if not path:
                continue
            try:
                source = Image.open(path).convert("RGB")
            except Exception as exc:
                log_line(f"[loc] WARN source load failed {path!r}: {exc}")
                continue
            W_src, H_src = source.size
            H, W = int(H_src), int(W_src)

            # GT.
            if is_real:
                gt_HW = None
                gt_frac = 0.0
                area_tier = "real"
            else:
                mask_path = str(meta.get("mask", "") or meta.get("mask_path", "") or "")
                gt_HW = _load_pixel_gt(mask_path, H=H, W=W)
                if gt_HW is None or (not gt_HW.any()):
                    # No usable mask — treat as skipped splice.
                    log_line(f"[loc] WARN no usable mask for splice {path!r}, skipping")
                    continue
                gt_frac = float(gt_HW.mean())
                area_tier = _bucket(gt_frac)

            gtcrop_areas_for_bucket = gtcrop_buckets_to_areas.get(bucket, [])
            try:
                row = _process_one_image(
                    model=model, device=device, cfg=cfg, args=args,
                    meta=dict(meta), split=split,
                    source_image=source,
                    gt_HW=gt_HW, is_real=is_real,
                    area_tier=bucket, gt_frac=gt_frac,
                    swin_combos=swin_combos,
                    gtcrop_areas_for_this_area_tier=gtcrop_areas_for_bucket,
                    row_keys=row_keys,
                )
                rows.append(row)
            except Exception as exc:
                log_line(f"[loc] ERROR row failed {path!r}: {exc}")
                raise

            if (idx + 1) % 25 == 0:
                log_line(f"[loc] split={split} progress {idx+1}/{len(items)}")

        log_line(f"[loc] split={split} END n_rows={len(rows)}")

        _emit_summaries(
            rows=rows, split=split,
            swin_combos=swin_combos,
            gtcrop_buckets_to_areas=gtcrop_buckets_to_areas,
            taus_image_level=list(args.taus_image_level),
            reals_subsample_rate=float(args.reals_subsample_rate),
        )

    log_line("[loc] DONE")


def _emit_summaries(
    *,
    rows: List[Dict],
    split: str,
    swin_combos: List[Tuple[float, float]],
    gtcrop_buckets_to_areas: Dict[str, List[float]],
    taus_image_level: List[float],
    reals_subsample_rate: float,
) -> None:
    """Emit [loc] / [bce_win] / [deploy] / [fp] / [zoom] lines for one split."""
    # full pass.
    _summarize.emit_loc(rows, split=split, pass_id="full_pure")
    _summarize.emit_loc(rows, split=split, pass_id="full_ceil")
    _summarize.emit_loc_polarity_compare(rows, split=split, pass_prefix="full")
    _summarize.emit_deploy(
        rows, split=split, pass_id="full_pure",
        bce_logit_key="full_bce_logit", taus=taus_image_level,
    )
    _summarize.emit_fp(
        rows, split=split, pass_id="full_pure",
        bce_logit_key="full_bce_logit", taus=taus_image_level,
        reals_subsample_rate=reals_subsample_rate,
    )

    # gtcrop passes.
    from contrastive_inpainting_v1.diagnose.passes.gtcrop import area_key
    for area_tier, areas in gtcrop_buckets_to_areas.items():
        for area in areas:
            k = area_key(area)
            _summarize.emit_loc(rows, split=split, pass_id=f"gtcrop_{k}_pure")
            _summarize.emit_loc(rows, split=split, pass_id=f"gtcrop_{k}_ceil")
            _summarize.emit_loc_polarity_compare(rows, split=split, pass_prefix=f"gtcrop_{k}")
    _summarize.emit_zoom(
        rows, split=split,
        gtcrop_buckets_to_areas=gtcrop_buckets_to_areas,
        taus=taus_image_level,
    )

    # swin passes.
    from contrastive_inpainting_v1.diagnose.passes.swin import swin_key
    for combo in swin_combos:
        k = swin_key(combo[0], combo[1])
        # Actual per-image distribution (counts vary per image with resolution).
        _summarize.emit_windows_dataset(rows, split=split, swin_combo=combo)
        # F1 including BCE-miss as F1=0 (deployable production behavior).
        _summarize.emit_loc(rows, split=split, pass_id=f"swin_{k}_pure")
        # F1 conditional on at least one window firing BCE — separates
        # localization quality from image-level BCE recall.
        _summarize.emit_loc(
            rows, split=split, pass_id=f"swin_{k}_pure",
            fire_filter_key=f"swin_{k}_n_bce_pos",
        )
        _summarize.emit_bce_win(rows, split=split, swin_combo=combo)
        _summarize.emit_deploy(
            rows, split=split, pass_id=f"swin_{k}_pure",
            bce_logit_key=f"swin_{k}_bce_logit_max", taus=taus_image_level,
        )
        _summarize.emit_fp(
            rows, split=split, pass_id=f"swin_{k}_pure",
            bce_logit_key=f"swin_{k}_bce_logit_max", taus=taus_image_level,
            reals_subsample_rate=reals_subsample_rate,
        )


if __name__ == "__main__":
    main()
