"""Sweep orchestrator — run a queue of training/eval jobs sequentially.

Standalone, stdlib-only, torch-free: it only shells out to child processes
(``python -m <module> --flag value ...``), so it runs fine in a venv without
torch (e.g. driving jobs on the 2080 Ti box or inside a Colab cell).

Usage::

    python -m contrastive_inpainting_v1.scripts.orchestrate \
        --queue sweeps/example_queue.json --run_root /content/drive/runs/sweep1 \
        [--dry_run] [--force NAME ...] [--stop_on_fail] [--cwd DIR]

Queue JSON::

    {
      "base_args": {"imd2020_root": "/content/IMD2020"},
      "runs": [
        {"name": "tau60", "module": "contrastive_inpainting_v1.scripts.train_multi_head",
         "args": {"tau_pos": 0.60, "num_epochs": 8}},
        ...
      ]
    }

Arg conversion: ``base_args`` then entry ``args`` (entry wins); JSON ``true``
becomes a bare ``--flag``, ``false`` is omitted, lists become repeated values
after the flag, everything else becomes ``--key str(value)``.  The literal
``{run_root}`` placeholder in string values is substituted with the resolved
absolute ``--run_root``.  Entries without an explicit ``checkpoint_root`` get
``--checkpoint_root <run_root>/<name>`` injected.

Resume-safe: each completed entry writes ``<run_root>/<name>/ORCH_DONE.json``;
entries whose marker has ``exit_code == 0`` are skipped on rerun unless named
via ``--force``.  Failures are recorded and the queue continues (unless
``--stop_on_fail``).  A ``sweep_summary.csv`` (status + last metrics.csv row
per run) is written at the end and on interrupt.

NOTE: this script intentionally uses plain ``print()`` rather than
``lab_utils.logging.text.log_line`` (the tag whitelist there does not cover
orchestrator tags, and the orchestrator must stay importable everywhere).
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

TRAINER_MODULE = 'contrastive_inpainting_v1.scripts.train_multi_head'
DONE_MARKER = 'ORCH_DONE.json'
TAIL_LINES = 30

# Summary columns that always come first (metric columns are appended after).
_FIXED_COLS = ['name', 'status', 'exit_code', 'wall_seconds']


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec='seconds')


# --------------------------------------------------------------------------- #
# Queue parsing / validation
# --------------------------------------------------------------------------- #

def load_queue(path: str) -> dict:
    try:
        with open(path) as f:
            queue = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f'[orchestrate] ERROR: cannot read queue {path!r}: {e}')

    runs = queue.get('runs')
    if not isinstance(runs, list) or not runs:
        sys.exit('[orchestrate] ERROR: queue must contain a non-empty "runs" list')

    errors = []
    seen = set()
    for i, entry in enumerate(runs):
        if not isinstance(entry, dict):
            errors.append(f'runs[{i}]: entry is not an object')
            continue
        name = entry.get('name')
        module = entry.get('module')
        if not name or not isinstance(name, str):
            errors.append(f'runs[{i}]: missing/invalid "name"')
        elif name in seen:
            errors.append(f'runs[{i}]: duplicate name {name!r}')
        else:
            seen.add(name)
        if not module or not isinstance(module, str):
            errors.append(f'runs[{i}] ({name!r}): missing/invalid "module"')
        args = entry.get('args', {})
        if not isinstance(args, dict):
            errors.append(f'runs[{i}] ({name!r}): "args" must be an object')

    base_args = queue.get('base_args', {})
    if not isinstance(base_args, dict):
        errors.append('"base_args" must be an object')

    if errors:
        for e in errors:
            print(f'[orchestrate] queue error: {e}', file=sys.stderr)
        sys.exit(f'[orchestrate] ERROR: {len(errors)} queue validation error(s); nothing was run')

    return queue


def _substitute(value, run_root: str):
    if isinstance(value, str):
        return value.replace('{run_root}', run_root)
    if isinstance(value, list):
        return [_substitute(v, run_root) for v in value]
    return value


def build_argv(entry: dict, base_args: dict, run_root: str) -> list[str]:
    """Resolve one queue entry into a full child argv."""
    name = entry['name']
    module = entry['module']
    merged = dict(base_args)
    merged.update(entry.get('args', {}))

    # Inject checkpoint_root for the trainer / any entry that doesn't set it.
    if 'checkpoint_root' not in merged:
        merged['checkpoint_root'] = os.path.join(run_root, name)

    argv = [sys.executable, '-m', module]
    for key, value in merged.items():
        value = _substitute(value, run_root)
        flag = f'--{key}'
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            argv.append(flag)
            argv.extend(str(v) for v in value)
        else:
            argv.append(flag)
            argv.append(str(value))
    return argv


# --------------------------------------------------------------------------- #
# Markers
# --------------------------------------------------------------------------- #

def marker_path(run_root: str, name: str) -> str:
    return os.path.join(run_root, name, DONE_MARKER)


def read_marker(run_root: str, name: str):
    try:
        with open(marker_path(run_root, name)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def write_marker(run_root: str, name: str, payload: dict) -> None:
    path = marker_path(run_root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Child execution
# --------------------------------------------------------------------------- #

def run_entry(name: str, argv: list[str], run_root: str, child_cwd: str):
    """Run one entry, streaming output to console + orchestrator.log.

    Returns (exit_code, started_at, finished_at, wall_seconds).
    Raises KeyboardInterrupt after cleanly stopping the child.
    """
    entry_dir = os.path.join(run_root, name)
    os.makedirs(entry_dir, exist_ok=True)
    log_path = os.path.join(entry_dir, 'orchestrator.log')

    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'

    started_at = _now_iso()
    t0 = time.monotonic()
    tail = collections.deque(maxlen=TAIL_LINES)

    with open(log_path, 'a') as log_f:
        header = f'===== [{started_at}] orchestrate launch: {shlex.join(argv)} =====\n'
        log_f.write(header)
        log_f.flush()

        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=child_cwd,
            env=env,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()
                tail.append(line)
            exit_code = proc.wait()
        except KeyboardInterrupt:
            print(f'\n[orchestrate] interrupt — terminating {name!r} ...')
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f'[orchestrate] {name!r} did not exit in 10s — killing')
                proc.kill()
                proc.wait()
            log_f.write(f'===== [{_now_iso()}] interrupted by user =====\n')
            raise

    wall = time.monotonic() - t0
    if exit_code != 0:
        print(f'[orchestrate] {name!r} FAILED (exit {exit_code}). Last {len(tail)} output lines:')
        for line in tail:
            sys.stdout.write('    | ' + line)
        sys.stdout.flush()
    return exit_code, started_at, _now_iso(), round(wall, 3)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

def last_metrics_row(run_root: str, name: str) -> dict:
    """Last row of the newest metrics.csv for a run; {} if none found."""
    entry_dir = os.path.join(run_root, name)
    candidates = []
    latest = os.path.join(entry_dir, 'latest_train', 'metrics.csv')
    if os.path.isfile(latest):
        candidates.append(latest)
    else:
        found = glob.glob(os.path.join(entry_dir, 'logs', '*', 'metrics.csv'))
        if found:
            candidates.append(max(found, key=os.path.getmtime))
    for path in candidates:
        try:
            with open(path, newline='') as f:
                rows = list(csv.DictReader(f))
            if rows:
                return {k: v for k, v in rows[-1].items() if k}
        except (OSError, csv.Error):
            continue
    return {}


def write_summary(run_root: str, results: dict, order: list[str]) -> None:
    """Write sweep_summary.csv and print an aligned table."""
    metric_cols: list[str] = []
    rows = []
    for name in order:
        res = results.get(name, {})
        metrics = {}
        if res.get('status') in ('done', 'failed', 'skipped'):
            metrics = last_metrics_row(run_root, name)
        for k in metrics:
            if k not in metric_cols and k not in _FIXED_COLS:
                metric_cols.append(k)
        row = {
            'name': name,
            'status': res.get('status', 'pending'),
            'exit_code': res.get('exit_code', ''),
            'wall_seconds': res.get('wall_seconds', ''),
        }
        row.update({k: v for k, v in metrics.items() if k not in _FIXED_COLS})
        rows.append(row)

    columns = _FIXED_COLS + metric_cols
    os.makedirs(run_root, exist_ok=True)
    out_path = os.path.join(run_root, 'sweep_summary.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in columns})

    # Aligned console table.
    str_rows = [[str(row.get(c, '')) for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in str_rows)) if str_rows else len(c)
              for i, c in enumerate(columns)]
    print()
    print('[orchestrate] sweep summary (' + out_path + '):')
    print('  ' + '  '.join(c.ljust(w) for c, w in zip(columns, widths)))
    print('  ' + '  '.join('-' * w for w in widths))
    for r in str_rows:
        print('  ' + '  '.join(v.ljust(w) for v, w in zip(r, widths)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        prog='orchestrate',
        description='Run a JSON queue of training/eval jobs sequentially (resume-safe).')
    p.add_argument('--queue', required=True, help='Path to queue JSON.')
    p.add_argument('--run_root', required=True,
                   help='Root dir for all runs in this sweep.')
    p.add_argument('--dry_run', action='store_true',
                   help='Print resolved commands and skip decisions; touch nothing.')
    p.add_argument('--force', action='append', default=[], metavar='NAME',
                   help='Rerun this entry even if its done-marker exists (repeatable).')
    p.add_argument('--stop_on_fail', action='store_true',
                   help='Stop the queue at the first nonzero child exit.')
    p.add_argument('--cwd', default=None,
                   help='Working directory for child processes (default: current dir).')
    args = p.parse_args()

    run_root = os.path.abspath(os.path.expanduser(args.run_root))
    child_cwd = os.path.abspath(os.path.expanduser(args.cwd)) if args.cwd else os.getcwd()
    if not os.path.isdir(child_cwd):
        sys.exit(f'[orchestrate] ERROR: --cwd {child_cwd!r} is not a directory')

    queue = load_queue(args.queue)
    base_args = queue.get('base_args', {})
    runs = queue['runs']
    names = [e['name'] for e in runs]

    unknown_force = [n for n in args.force if n not in names]
    if unknown_force:
        sys.exit(f'[orchestrate] ERROR: --force names not in queue: {unknown_force}')

    # Decide skip/run per entry up front.
    plan = []  # (entry, argv, skip)
    for entry in runs:
        name = entry['name']
        argv = build_argv(entry, base_args, run_root)
        marker = read_marker(run_root, name)
        done = marker is not None and marker.get('exit_code') == 0
        skip = done and name not in args.force
        plan.append((entry, argv, skip))

    if args.dry_run:
        print(f'[orchestrate] DRY RUN — queue {args.queue!r}, run_root {run_root!r}, '
              f'child cwd {child_cwd!r}')
        for entry, argv, skip in plan:
            if skip:
                print(f'[skip] {entry["name"]} (done)')
            else:
                print(shlex.join(argv))
        return

    os.makedirs(run_root, exist_ok=True)
    results: dict[str, dict] = {}
    interrupted = False
    overall_rc = 0

    try:
        for entry, argv, skip in plan:
            name = entry['name']
            if skip:
                print(f'[skip] {name} (done)')
                marker = read_marker(run_root, name) or {}
                results[name] = {
                    'status': 'skipped',
                    'exit_code': marker.get('exit_code', 0),
                    'wall_seconds': marker.get('wall_seconds', ''),
                }
                continue

            if name in args.force:
                stale = marker_path(run_root, name)
                if os.path.exists(stale):
                    os.remove(stale)
                    print(f'[force] {name}: removed stale {DONE_MARKER}')

            print(f'\n[run] {name}: {shlex.join(argv)}')
            exit_code, started_at, finished_at, wall = run_entry(
                name, argv, run_root, child_cwd)

            write_marker(run_root, name, {
                'name': name,
                'module': entry['module'],
                'argv': argv,
                'exit_code': exit_code,
                'started_at': started_at,
                'finished_at': finished_at,
                'wall_seconds': wall,
            })
            status = 'done' if exit_code == 0 else 'failed'
            results[name] = {'status': status, 'exit_code': exit_code,
                             'wall_seconds': wall}
            print(f'[{status}] {name} (exit {exit_code}, {wall:.1f}s)')

            if exit_code != 0:
                overall_rc = 1
                if args.stop_on_fail:
                    print('[orchestrate] --stop_on_fail: stopping queue')
                    break
    except KeyboardInterrupt:
        interrupted = True
        overall_rc = 130

    write_summary(run_root, results, names)
    if interrupted:
        print('[orchestrate] interrupted — summary written; rerun to resume')
    sys.exit(overall_rc)


if __name__ == '__main__':
    main()
