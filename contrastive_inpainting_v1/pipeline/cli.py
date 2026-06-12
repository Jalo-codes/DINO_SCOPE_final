"""Shared v3 argparse helpers."""

from __future__ import annotations

import argparse

from lab_utils.paths import resolve_data_paths


def add_data_path_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--imd2020_root", default=None)
    parser.add_argument("--casia_root", default=None)
    parser.add_argument("--tgif2_root", "--tgif_root", dest="tgif2_root", default=None)
    parser.add_argument("--run_root", "--checkpoint_root", dest="run_root", default=None)
    return parser


def resolved_paths(args):
    return resolve_data_paths(args)


def apply_path_defaults(args):
    """Mutate an argparse namespace with shared env-backed path defaults."""

    paths = resolve_data_paths(args)
    if hasattr(args, "imd2020_root") and not getattr(args, "imd2020_root"):
        args.imd2020_root = paths.imd2020_root
    if hasattr(args, "casia_root") and not getattr(args, "casia_root"):
        args.casia_root = paths.casia_root
    if hasattr(args, "tgif2_root") and not getattr(args, "tgif2_root"):
        args.tgif2_root = paths.tgif2_root
    if hasattr(args, "tgif_root") and not getattr(args, "tgif_root"):
        args.tgif_root = paths.tgif2_root
    if hasattr(args, "checkpoint_root") and not getattr(args, "checkpoint_root"):
        args.checkpoint_root = paths.run_root or "/media/ssd/runs/contrastive_v3"
    if hasattr(args, "run_root") and not getattr(args, "run_root"):
        args.run_root = paths.run_root or "/media/ssd/runs/contrastive_v3"
    return args
