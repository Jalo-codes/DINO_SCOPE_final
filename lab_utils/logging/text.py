"""lab_utils.logging.text — tagged, thread-safe line-oriented text logging.

Every public log call requires a tag from ALLOWED_TAGS.  Passing an unknown
tag raises ValueError immediately so mis-tagged lines are caught at development
time rather than cluttering production logs silently.

Allowed tags (enforced):
    [data]    [augment]  [blob]    [train]   [eval]
    [memory]  [ckpt]     [dist]    [lora]    [suite]  [cfg]
    [swin]    [swin-cal] [robust]  [polarity] [oracle] [inferred] [trace]
    [cap]     [cap2]     [straddle] [gate]   (swin test plan phases 1–4)
"""

import datetime
import os
import platform
import subprocess
import sys
import threading
from typing import Optional

_LOG_PATH: Optional[str] = None
_LOG_LOCK = threading.Lock()

ALLOWED_TAGS = frozenset([
    '[data]', '[augment]', '[blob]', '[train]', '[eval]',
    '[memory]', '[ckpt]', '[dist]', '[lora]', '[suite]', '[cfg]',
    '[swin]', '[swin-cal]', '[robust]', '[polarity]', '[oracle]', '[inferred]', '[trace]',
    '[gtcrop]', '[predcrop]', '[crop]', '[tgif]',
    # diagnose_v2 tags
    '[loc]', '[loc_when_fired]', '[bce_win]', '[deploy]', '[fp]', '[zoom]',
    '[passes]', '[metric_defs]', '[windows]', '[buckets]',
    # swin test plan tags (Phase 1–4)
    '[cap]', '[cap2]', '[straddle]', '[gate]',
])


def _check_tag(message: str) -> None:
    """Raise ValueError if message does not start with an allowed tag."""
    for tag in ALLOWED_TAGS:
        if message.startswith(tag):
            return
    raise ValueError(
        f"log_line message must start with one of {sorted(ALLOWED_TAGS)}.\n"
        f"Got: {message!r}"
    )


def install_log(log_path: str) -> str:
    """Open (or append to) a text log file and write a self-describing header.

    Must be called once per process before any log_line/log_warn/log_error
    calls.  Safe to call again — subsequent calls append a new header block
    so restarts are visible in the same file.
    """
    global _LOG_PATH
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _LOG_PATH = log_path

    try:
        git_sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = 'unknown'

    py_ver = platform.python_version()
    try:
        import torch as _torch
        torch_ver = _torch.__version__
        if _torch.cuda.is_available():
            _props = _torch.cuda.get_device_properties(0)
            gpu_info = (
                f'{_torch.cuda.get_device_name(0)} '
                f'{_props.total_memory / 1024 ** 3:.1f}GB'
            )
            gpu_count = _torch.cuda.device_count()
        else:
            gpu_info = 'cpu'
            gpu_count = 0
    except Exception:
        torch_ver = 'unknown'
        gpu_info = 'unknown'
        gpu_count = 0

    ts = datetime.datetime.now().isoformat(timespec='seconds')
    cmd = ' '.join(sys.argv)
    header = (
        f"# ============================================================\n"
        f"# log opened  pid={os.getpid()}  ts={ts}  git={git_sha}\n"
        f"# python={py_ver}  torch={torch_ver}"
        f"  device={gpu_info}  n_gpu={gpu_count}\n"
        f"# cmd: {cmd}\n"
        f"# ============================================================\n"
    )
    with _LOG_LOCK:
        with open(_LOG_PATH, 'a', buffering=1) as f:
            f.write(header)
    return log_path


def log_line(message: str, *, echo: bool = True, flush: bool = False) -> None:
    """Write one tagged line to the log and optionally to stdout.

    The message must begin with an allowed tag (see ALLOWED_TAGS).
    Raises ValueError on unknown tag so mis-tagged lines surface immediately.
    """
    _check_tag(message)
    if echo:
        print(message, flush=flush)
    if _LOG_PATH is None:
        return
    with _LOG_LOCK:
        with open(_LOG_PATH, 'a', buffering=1) as f:
            f.write(f"{message}\n")


def log_warn(message: str, *, echo: bool = True) -> None:
    """Like log_line but prefixes WARN to the tagged message."""
    _check_tag(message)
    warn_msg = message.replace(message.split(']')[0] + ']',
                                message.split(']')[0] + '] WARN:', 1)
    if echo:
        print(warn_msg, flush=False)
    if _LOG_PATH is None:
        return
    with _LOG_LOCK:
        with open(_LOG_PATH, 'a', buffering=1) as f:
            f.write(f"{warn_msg}\n")


def log_error(message: str, *, echo: bool = True) -> None:
    """Like log_line but prefixes ERROR to the tagged message."""
    _check_tag(message)
    err_msg = message.replace(message.split(']')[0] + ']',
                               message.split(']')[0] + '] ERROR:', 1)
    if echo:
        print(err_msg, file=sys.stderr, flush=True)
    if _LOG_PATH is None:
        return
    with _LOG_LOCK:
        with open(_LOG_PATH, 'a', buffering=1) as f:
            f.write(f"{err_msg}\n")


# ---------------------------------------------------------------------------
# log_metric_row — fixed-precision tabular metric line
# ---------------------------------------------------------------------------

# Default decimal places per metric "kind".  Matches across sweep variants so
# numbers line up vertically when grep'd.  Keys are matched as suffixes of
# the metric name (e.g. ``f1_med`` ends in ``_med`` AND in ``f1_med`` — the
# longer match wins).
_DECIMALS_BY_SUFFIX = {
    '_med':        4,
    '_iou':        4,
    '_f1':         4,
    '_prec':       4,
    '_rec':        4,
    '_acc':        4,
    '_auc':        4,
    '_auprc':      4,
    '_frac':       4,
    '_tau':        4,
    '_delta':      4,
    '_rate':       3,
    '_thresh':     4,
    '_threshold':  4,
    '_mean':       4,
    '_std':        4,
    '_p25':        4,
    '_p75':        4,
}


def _format_number(name: str, value, default_decimals: int = 4) -> str:
    """Render ``value`` with consistent decimal precision based on metric name.

    Integers stay integer; NaN renders as ``nan``; strings pass through.
    """
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return 'nan'
        # Pick the longest matching suffix.
        decimals = default_decimals
        for suffix, d in sorted(_DECIMALS_BY_SUFFIX.items(),
                                key=lambda kv: -len(kv[0])):
            if name.endswith(suffix):
                decimals = d
                break
        return f'{value:.{decimals}f}'
    return str(value)


def log_metric_row(
    tag: str,
    *,
    prefix: str = '',
    **fields,
) -> None:
    """Emit a fixed-precision ``key=value`` row under ``tag``.

    ``tag`` is one of the allowed log tags (e.g. ``'[swin]'``).
    ``prefix`` is an optional inline label between the tag and the fields
    (e.g. a sweep cell id or a split name).
    Field order in the output matches kwargs insertion order.

    Float fields use a per-name decimal precision (see ``_DECIMALS_BY_SUFFIX``)
    so the same metric column lines up vertically across rows.  Integers
    stay integer; ``nan`` renders explicitly.

    Example::

        log_metric_row('[swin]', prefix='cell=07 imd_val',
                       f1_med=0.7748, iou_med=0.6324, n=130)
        # -> [swin] cell=07 imd_val f1_med=0.7748 iou_med=0.6324 n=130
    """
    if not tag.startswith('[') or not tag.endswith(']') or tag not in ALLOWED_TAGS:
        raise ValueError(
            f"log_metric_row: tag must be one of {sorted(ALLOWED_TAGS)}; got {tag!r}"
        )
    parts = [_format_number(k, v) for k, v in fields.items()]
    body = ' '.join(f'{k}={p}' for (k, _), p in zip(fields.items(), parts))
    line = f'{tag} {prefix} {body}'.strip() if prefix else f'{tag} {body}'
    log_line(line)
