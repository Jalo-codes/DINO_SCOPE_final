"""lab_utils.logging.run_dir — canonical run-directory builder.

Layout (resume-safe; *every* script lands here)::

    <root>/                              # e.g. /media/ssd/runs/
      <name>/                            # stable; resume key
        checkpoints/                     # epoch_*.pt, last.pt
        config.json                      # frozen experiment config
        logs/
          <YYYYMMDD-HHMMSS>_<git7>_<role>/   # one per invocation
            manifest.json                # this invocation's argv, role, env
            run.log                      # tagged log_line() output
            metrics.csv                  # CSVLogger output (if any)
            artifacts/                   # plots, JSON dumps
        latest_<role> -> logs/<latest>/  # best-effort convenience symlink

The stable ``<root>/<name>/`` survives across invocations — checkpoints
write/read from a known path, so ``--resume`` Just Works.  Each invocation
gets a fresh timestamped subdir under ``logs/``, so prior logs are never
overwritten.

API::

    rd = build_run_dir(root, name, role)        # returns RunDir
    install_log(str(rd.log_path))
    ckpt_path = rd.checkpoints / 'epoch_005.pt'

``role`` is a short identifier: ``'train'`` | ``'eval'`` | ``'diagnose-swin'``
etc.  Choose one when there's no ambiguity in the script's purpose.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional


_SAFE_RE = re.compile(r'[^A-Za-z0-9_.-]+')


def _safe(s: str) -> str:
    """Coerce a free-form string to a filesystem-safe role name."""
    s = _SAFE_RE.sub('_', str(s)).strip('_')
    return s or 'run'


def _git_sha_short() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out or 'unknown'
    except Exception:
        return 'unknown'


def _gpu_info() -> dict:
    info = {'count': 0, 'names': []}
    try:
        import torch as _torch  # local import — lab_utils.logging must work without torch
        if _torch.cuda.is_available():
            info['count'] = int(_torch.cuda.device_count())
            info['names'] = [_torch.cuda.get_device_name(i)
                             for i in range(info['count'])]
    except Exception:
        pass
    return info


@dataclasses.dataclass(frozen=True)
class RunDir:
    """Resolved paths for one invocation of one experiment.

    ``root`` is the stable per-experiment dir; ``log_dir`` is the
    timestamped per-invocation dir.  Callers should always read/write
    checkpoints via ``checkpoints``, and the active log via ``log_path``.
    """
    root:          Path  # <run_root>/<name>/
    name:          str
    role:          str
    git_sha:       str
    checkpoints:   Path  # <root>/checkpoints/
    config_path:   Path  # <root>/config.json
    log_dir:       Path  # <root>/logs/<ts>_<git>_<role>/
    log_path:      Path  # <log_dir>/run.log
    metrics_path:  Path  # <log_dir>/metrics.csv
    artifacts_dir: Path  # <log_dir>/artifacts/
    manifest_path: Path  # <log_dir>/manifest.json

    def __fspath__(self) -> str:
        # Treat the RunDir as the stable root for os.path / Path operations.
        # Callers that want the timestamped log dir should use ``.log_dir`` explicitly.
        return str(self.root)

    def __str__(self) -> str:
        # f-string interpolation prints the stable root path, matching the
        # behavior of the legacy ``build_run_dir`` return value (a path string).
        return str(self.root)


def build_run_dir(
    root: str,
    name: str,
    role: str = 'run',
    *,
    timestamp: Optional[datetime.datetime] = None,
    write_symlink: bool = True,
) -> RunDir:
    """Create the directory layout and write per-invocation manifest.

    The stable ``<root>/<name>/`` and its ``checkpoints/`` subdir are created
    if missing (resume-safe).  A fresh timestamped log subdir is created for
    *every* invocation so prior logs are preserved.

    Args:
        root:           Parent dir (e.g. ``/media/ssd/runs``).
        name:           Experiment / run name (stable across invocations).
        role:           Short tag for this invocation: ``'train'`` | ``'eval'``
                        | ``'diagnose-swin'`` etc.  Recorded in the log-dir name
                        and manifest so the role is grep-able from the path.
        timestamp:      Override the timestamp used in the log dir name.
                        Default: ``datetime.now()``.  Useful for tests.
        write_symlink:  When True (default), best-effort update
                        ``<root>/<name>/latest_<role>`` to point at the new
                        log dir.  Symlink failure is non-fatal.

    Returns:
        :class:`RunDir`.
    """
    root_dir = Path(root).expanduser().resolve() / name
    checkpoints = root_dir / 'checkpoints'
    config_path = root_dir / 'config.json'
    logs_root = root_dir / 'logs'

    root_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    git_sha = _git_sha_short()
    ts = (timestamp or datetime.datetime.now()).strftime('%Y%m%d-%H%M%S')
    role_safe = _safe(role)
    log_dir_name = f'{ts}_{git_sha}_{role_safe}'
    log_dir = logs_root / log_dir_name
    # Race-tolerant mkdir: under DDP, multiple ranks call build_run_dir
    # simultaneously and we want them to SHARE the same per-invocation
    # log dir.  ``exist_ok=True`` makes the directory creation idempotent.
    # The cost: two sequential invocations within the same second + git
    # sha + role share a log dir (edge case; manifest will be overwritten
    # by the later caller).  Worth it to avoid the DDP race.
    log_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = log_dir / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rd = RunDir(
        root=root_dir,
        name=str(name),
        role=role_safe,
        git_sha=git_sha,
        checkpoints=checkpoints,
        config_path=config_path,
        log_dir=log_dir,
        log_path=log_dir / 'run.log',
        metrics_path=log_dir / 'metrics.csv',
        artifacts_dir=artifacts_dir,
        manifest_path=log_dir / 'manifest.json',
    )

    # Write per-invocation manifest.
    manifest = {
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'pid': os.getpid(),
        'hostname': socket.gethostname(),
        'git_sha': git_sha,
        'name': str(name),
        'role': role_safe,
        'argv': list(sys.argv),
        'cwd': os.getcwd(),
        'python': platform.python_version(),
        'gpu': _gpu_info(),
        'paths': {
            'root': str(rd.root),
            'checkpoints': str(rd.checkpoints),
            'log_dir': str(rd.log_dir),
            'log_path': str(rd.log_path),
            'metrics_path': str(rd.metrics_path),
            'artifacts_dir': str(rd.artifacts_dir),
        },
    }
    with open(rd.manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Best-effort symlink ``latest_<role>`` for convenience.
    if write_symlink:
        try:
            link = root_dir / f'latest_{role_safe}'
            if link.exists() or link.is_symlink():
                link.unlink()
            # Relative target so the link survives a move of the parent dir.
            link.symlink_to(Path('logs') / log_dir.name, target_is_directory=True)
        except Exception:
            pass

    return rd


# Backward-compat shim: a few callers still pass two positional args and rely
# on the return value being os.path.join()-able.  The new RunDir is
# os.fspath-compatible, so both patterns keep working.
def build_run_dir_legacy(root: str, name: str) -> str:
    """Deprecated: prefer :func:`build_run_dir` with an explicit ``role``."""
    return str(build_run_dir(root, name, role='run').root)


def build_run_dir_from_checkpoint_root(
    checkpoint_root: str,
    role: str,
    *,
    timestamp: Optional[datetime.datetime] = None,
    write_symlink: bool = True,
) -> RunDir:
    """Convenience: split ``<parent>/<name>`` and call :func:`build_run_dir`.

    The trainers historically take a flat ``--checkpoint_root`` argument
    that already includes the experiment name (e.g.
    ``/media/ssd/runs/multi_head_v2/joint_swin_lambda2_bce_zoom``).  This
    helper does the basename split so callers do not repeat the same
    7 lines of ``os.path.abspath`` / ``dirname`` / ``basename`` boilerplate.
    """
    abs_root = os.path.abspath(checkpoint_root)
    return build_run_dir(
        os.path.dirname(abs_root),
        os.path.basename(abs_root),
        role,
        timestamp=timestamp,
        write_symlink=write_symlink,
    )
