"""lab_utils.train.checkpoint — save and load training state.

No EMA in lab_utils.  The experiment owns its own weight-averaging strategy
if needed.

Checkpoint layout (all in one .pt file):
    {
        'epoch':        int,
        'model':        state_dict,
        'optimizer':    state_dict,
        'scheduler':    state_dict | None,
        'scaler':       state_dict | None,
        'best_metric':  float,
        'cfg':          dict,            # serialised experiment config
        'meta':         dict,            # freeform experiment metadata
    }
"""

import glob
import os
from typing import Any, Dict, Iterable, Optional

import torch

from lab_utils.errors import DataError
from lab_utils.logging.text import log_line


_DEFAULT_CKPT_PATTERNS = ("epoch_*.pt", "last.pt", "best.pt")


def find_latest_checkpoint(
    run_dir: str,
    *,
    patterns: Iterable[str] = _DEFAULT_CKPT_PATTERNS,
) -> Optional[str]:
    """Find the most recent checkpoint in ``run_dir`` by mtime.

    Searches each pattern (default: ``epoch_*.pt``, ``last.pt``, ``best.pt``)
    under ``run_dir`` (non-recursive) and returns the path with the latest
    mtime across all matches, or ``None`` if no file matches.

    Use this whenever a script needs to resume / probe a run dir; it replaces
    the ad-hoc glob+max(getmtime) snippets that have proliferated across the
    diagnose / eval scripts.
    """
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(run_dir, pat)))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def save(
    state: Dict[str, Any],
    path: str,
    *,
    is_main: bool = True,
) -> None:
    """Save checkpoint to `path`.  Only rank-0 writes.

    Args:
        state:   Dict containing at minimum 'epoch' and 'model'.
        path:    Absolute path for the .pt file.
        is_main: Set False on non-zero ranks to skip writing.
    """
    if not is_main:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    epoch = state.get('epoch', '?')
    log_line(f'[ckpt] saved epoch={epoch} → {path}')


def load(path: str, *, map_location: str = 'cpu') -> Dict[str, Any]:
    """Load a checkpoint from `path`.

    Args:
        path:         Path to .pt file.
        map_location: Device to map tensors to on load.

    Returns:
        The state dict as saved by `save`.

    Raises:
        DataError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise DataError(f"checkpoint.load: file not found: {path}")
    state = torch.load(path, map_location=map_location)
    epoch = state.get('epoch', '?')
    log_line(f'[ckpt] loaded epoch={epoch} ← {path}')
    return state


def save_best(
    state: Dict[str, Any],
    run_dir: str,
    *,
    is_main: bool = True,
) -> None:
    """Overwrite best.pt in run_dir."""
    save(state, os.path.join(run_dir, 'best.pt'), is_main=is_main)


def save_last(
    state: Dict[str, Any],
    run_dir: str,
    *,
    is_main: bool = True,
) -> None:
    """Overwrite last.pt in run_dir."""
    save(state, os.path.join(run_dir, 'last.pt'), is_main=is_main)
