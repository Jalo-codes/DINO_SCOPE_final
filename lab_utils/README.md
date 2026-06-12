# `lab_utils` — shared library for DINO_SCOPE experiments

Single source of truth for **logging, run layout, eval, sampling, geometry,
and training helpers** used by every script under `contrastive_test_v2/`.

> Older `v2/`–`v6/` and `contrastive_test/` directories are out of scope —
> they are kept for reference and do not import `lab_utils`.

## Canonical top-level surface

Most callers should import from the package root:

```python
from lab_utils import (
    # logging / runs
    install_log, log_line, log_warn, log_error, log_metric_row,
    CSVLogger, RunDir, build_run_dir,

    # errors
    DataError, ConfigError, EvalError,

    # eval
    f1_iou, binary_metrics, dispatch_eval, EvalRecord, EvalResult,

    # checkpoints
    find_latest_checkpoint, ckpt_load, ckpt_save,

    # data / sampling
    splice_balance_weights, build_quick_val_items,
    build_case_balanced_quick_val_items, build_shared_tau_calibration_items,
    deterministic_subsample, reals_subsample,
    is_real, is_splice, stable_item_sort_key,
)
```

Sub-modules are also importable directly when you need something off the
main surface (e.g. `from lab_utils.eval.window_geometry import window_grid`).

## Run-output layout

`build_run_dir(root, name, role)` returns a [`RunDir`](logging/run_dir.py)
that resolves to this layout:

```
<root>/                              # e.g. /media/ssd/runs/
  <name>/                            # stable; survives resume
    checkpoints/                     # epoch_*.pt, last.pt, best.pt
    config.json                      # frozen experiment config
    logs/
      <YYYYMMDD-HHMMSS>_<git7>_<role>/   # one per invocation
        manifest.json                # argv, env, gpu info, role
        run.log                      # tagged log_line() output
        metrics.csv                  # CSVLogger output
        artifacts/                   # plots, JSON dumps
    latest_<role> -> logs/<latest>/  # convenience symlink
```

- `<root>/<name>/` is **stable** — `--resume` reads `checkpoints/` here.
- Each invocation creates a **fresh** `logs/<ts>_<git>_<role>/` so prior
  logs are never overwritten.
- `role` is one of `train`, `eval`, `diagnose-<mode>`, `suite`, …

### Resuming

Pass the same `--checkpoint_root` (or `--run_root` if you've migrated to the
new CLI) and the same experiment name. Checkpoint discovery uses
[`find_latest_checkpoint`](train/checkpoint.py) (replaces the duplicated
`_latest_checkpoint` helpers that used to live in `diagnose_v2.py` and
`diagnose_swin_windows.py`).

## Log conventions

Every line starts with an **allowed tag** (`[data]`, `[train]`, `[eval]`,
`[swin]`, `[ckpt]`, `[dist]`, `[cfg]`, …). Unknown tags raise at the call
site so misformatted lines are caught immediately. See
[`ALLOWED_TAGS` in `logging/text.py`](logging/text.py).

- **`log_line(msg)`** — free-form tagged line.
- **`log_warn / log_error`** — same shape with `WARN:` / `ERROR:` prefix.
- **`log_metric_row(tag, prefix=..., **fields)`** — tabular metric line
  with fixed decimal precision per suffix (`_f1`, `_iou`, `_med`,
  `_pred_frac`, etc.). Use this for sweep tables and per-bucket summaries —
  it makes columns line up vertically across rows.

```
[swin] cell=07 imd_val n=130 f1_med=0.7748 iou_med=0.6324 prec_med=0.6466
```

## CLI vocabulary

`lab_utils.cli.add_common_args(parser)` registers the canonical surface:

```
--run_root        parent dir holding runs   (alias: --checkpoint_root)
--name            stable run identifier
--resume          load latest checkpoint
--seed            --split_seed
--device          --num_workers
--batch_size      --eval_batch_size
```

Adoption is opt-in — existing scripts keep their current arg names until
rewritten. `lab_utils.cli.resolve_run_paths(args, role=...)` returns a
ready-made `RunDir` from these args.

## Tests

```
pytest                       # all of lab_utils/tests/
pytest lab_utils/tests/test_find_latest_checkpoint.py -v
```

The `contrastive_test_v2/tests/test_parity_imd2020.py` script is an
integration parity check (lab_utils vs legacy contrastive_test losses /
indexer) and is invoked directly:

```
python -m contrastive_test_v2.tests.test_parity_imd2020
```

## Sub-package map

| Sub-package | What lives here |
| --- | --- |
| `logging` | `install_log`, `log_line`, `log_metric_row`, `CSVLogger`, `RunDir`, `build_run_dir` |
| `errors` | `DataError`, `ConfigError`, `EvalError` |
| `data` | `LabDataset`, `LoaderConfig`, `build_train_loader`, `build_eval_loader`, indexers, **sampling** (balance weights / quick-val builders / deterministic subsamples), augment building blocks |
| `eval` | `run_eval`, `run_bce_eval`, `dispatch_eval`, `f1_iou`, `binary_metrics`, sliding-window, partition, **window_geometry** (pure geometric helpers) |
| `model` | `ContrastiveDetector`, `MultiHeadDetector`, `ImageBCEDetector`, loss registry |
| `train` | DDP setup, precision, memory, `loop`, `checkpoint` (`find_latest_checkpoint`), LoRA, **distributed work-sharding helpers** (`shard_iterable`, `gather_dicts`) |
| `suite` | `TestSpec`, `run_suite` — multi-experiment eval harness |
| `cli` | `add_common_args`, `resolve_run_paths` — shared argparse surface |

## Design notes

- **No experiment configs in `lab_utils`.** The shared layer takes explicit
  numerical kwargs. Experiment-side cfg-to-kwargs adapters belong in
  `contrastive_test_v2.configs.*`.
- **Pure geometry vs preprocessing.** `lab_utils.eval.window_geometry` is
  pure NumPy + PIL (no torch). The torchvision-flavored crop/normalize
  helpers stay near their callers in `contrastive_test_v2.diagnose_v2.passes.common`.
- **Empty metrics.** `f1_iou(p, g)` returns `(1.0, 1.0)` on empty/empty by
  default (principled). Pass `empty_value=0.0` to reproduce the legacy
  diagnose-script convention.
