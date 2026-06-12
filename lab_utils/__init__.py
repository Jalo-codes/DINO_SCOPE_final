"""lab_utils — shared building blocks for DINO_SCOPE experiments.

Canonical surface — everything most callers need is available at the top
level::

    from lab_utils import (
        # logging / runs
        install_log, log_line, log_warn, log_error, build_run_dir,
        # errors
        DataError, ConfigError, EvalError,
        # eval
        f1_iou, binary_metrics,
        # checkpoints
        find_latest_checkpoint, ckpt_load, ckpt_save,
        # data / sampling
        splice_balance_weights, build_quick_val_items,
        deterministic_subsample, reals_subsample,
    )

Sub-packages stay importable for everything else.
"""

from lab_utils.errors import DataError, ConfigError, EvalError
from lab_utils.logging.text import (
    install_log, log_line, log_warn, log_error, log_metric_row,
)
from lab_utils.logging.run_dir import RunDir, build_run_dir
from lab_utils.paths import DataPaths, resolve_data_paths

from lab_utils.eval.metrics import f1_iou, binary_metrics

from lab_utils.train.checkpoint import (
    find_latest_checkpoint,
    load as ckpt_load,
    save as ckpt_save,
)

from lab_utils.data.sampling import (
    build_case_balanced_quick_val_items,
    build_quick_val_items,
    build_shared_tau_calibration_items,
    deterministic_subsample,
    is_real,
    is_splice,
    reals_subsample,
    splice_balance_weights,
    stable_item_sort_key,
)


__all__ = [
    # errors
    'DataError', 'ConfigError', 'EvalError',
    # logging / runs
    'install_log', 'log_line', 'log_warn', 'log_error', 'log_metric_row',
    'RunDir', 'build_run_dir',
    'DataPaths', 'resolve_data_paths',
    # eval
    'f1_iou', 'binary_metrics',
    # checkpoints
    'find_latest_checkpoint', 'ckpt_load', 'ckpt_save',
    # data / sampling
    'build_case_balanced_quick_val_items',
    'build_quick_val_items',
    'build_shared_tau_calibration_items',
    'deterministic_subsample',
    'is_real', 'is_splice',
    'reals_subsample',
    'splice_balance_weights',
    'stable_item_sort_key',
]
