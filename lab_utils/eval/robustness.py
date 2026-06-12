"""lab_utils.eval.robustness — model-agnostic augmentation robustness sweep.

Provides one orchestration function that takes a model-specific eval callable
and a list of augmentation conditions, runs eval under each, and prints a
side-by-side robustness table.

The eval callable owns the model and dataset — the sweep just iterates over
augmentation conditions and collects metrics. This keeps the utility usable
across BCE models, contrastive models, or anything else with a
"clean → corrupted" robustness question.

Expected eval_callable signature:
    eval_callable(aug_kwargs: Dict, *, tag: str) -> Dict[str, Any]

Where aug_kwargs has the keys consumed by LabDataset:
    'eval_aug_mode', 'eval_corruption_spec', 'eval_corruption_region'
(or whatever your dataset accepts as augmentation kwargs).

The returned dict should be JSON-friendly (scalars for the metrics in
metrics_to_show, anything else ok). Recommended keys:
    'auc', 'bal_acc', 'tpr', 'tnr', 'tpr_at_tnr_95', 'n_total'
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lab_utils.logging.text import log_line


def run_robustness_sweep(
    eval_callable: Callable[..., Dict[str, Any]],
    aug_conditions: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    metrics_to_show: Sequence[str] = ('auc', 'bal_acc', 'tpr', 'tnr', 'tpr_at_tnr_95'),
    baseline_name: Optional[str] = 'none',
    log_tag: str = '[robust]',
    tag: str = '',
) -> Dict[str, Dict[str, Any]]:
    """Run eval_callable under each augmentation condition; print a table.

    Args:
        eval_callable: takes (aug_kwargs, *, tag) → metrics dict.
        aug_conditions: list of (name, aug_kwargs) tuples, in display order.
        metrics_to_show: keys to extract from each metrics dict for the table.
        baseline_name: if present in results, show Δ vs this condition.
        log_tag: log line tag (must be in ALLOWED_TAGS).
        tag: optional sub-tag e.g. 'imd_val' / 'casia_val' to disambiguate.

    Returns:
        {condition_name: metrics_dict} for further programmatic use.
    """
    suffix = f' {tag}' if tag else ''
    log_line(
        f'{log_tag}{suffix} starting sweep over {len(aug_conditions)} '
        f'aug conditions: {[n for n, _ in aug_conditions]}'
    )

    results: Dict[str, Dict[str, Any]] = {}
    for cond_name, aug_kwargs in aug_conditions:
        log_line(f'{log_tag}{suffix} aug={cond_name} running...')
        metrics = eval_callable(aug_kwargs, tag=f'{tag}/{cond_name}' if tag else cond_name)
        if not isinstance(metrics, dict):
            raise TypeError(
                f'eval_callable returned {type(metrics).__name__}, expected dict'
            )
        results[cond_name] = metrics

    format_robustness_table(
        results,
        metrics_to_show=metrics_to_show,
        baseline_name=baseline_name,
        log_tag=log_tag,
        tag=tag,
    )
    return results


def format_robustness_table(
    results: Dict[str, Dict[str, Any]],
    *,
    metrics_to_show: Sequence[str] = ('auc', 'bal_acc', 'tpr', 'tnr', 'tpr_at_tnr_95'),
    baseline_name: Optional[str] = 'none',
    log_tag: str = '[robust]',
    tag: str = '',
) -> None:
    """Pretty-print a results dict as a robustness table."""
    if not results:
        log_line(f'{log_tag} no results to format')
        return
    suffix = f' {tag}' if tag else ''

    baseline = results.get(baseline_name) if baseline_name else None

    # Header
    name_w = max(8, max(len(n) for n in results))
    header_parts = [f'condition'.ljust(name_w)]
    for m in metrics_to_show:
        header_parts.append(f'{m:>14}')
    if baseline is not None:
        header_parts.append(f'   {"Δauc":>8}')
    log_line(f'{log_tag}{suffix} ' + '  '.join(header_parts))

    # Rows
    for name, m in results.items():
        row = [name.ljust(name_w)]
        for key in metrics_to_show:
            val = m.get(key)
            if val is None:
                row.append(f'{"-":>14}')
            elif isinstance(val, (int,)):
                row.append(f'{val:>14d}')
            else:
                try:
                    row.append(f'{float(val):>14.4f}')
                except (TypeError, ValueError):
                    row.append(f'{str(val):>14}')
        if baseline is not None and 'auc' in m and 'auc' in baseline:
            try:
                delta = float(m['auc']) - float(baseline['auc'])
                row.append(f'   {delta:+8.4f}')
            except (TypeError, ValueError):
                row.append(f'   {"-":>8}')
        log_line(f'{log_tag}{suffix} ' + '  '.join(row))


def metrics_from_logits(logits, labels, *, fixed_tnr_targets=(0.95, 0.99)) -> Dict[str, Any]:
    """Convenience: derive a standard metrics dict from per-image logits + labels.

    Most BCE-style evals already produce (logit, label) per image; pass them
    here to get a consistently-shaped metrics dict for the robustness sweep.

    Args:
        logits: 1D numpy array of model output logits (higher = positive class).
        labels: 1D numpy array of {0, 1} labels (1 = positive class).
        fixed_tnr_targets: list of TNR targets to report TPR at.

    Returns:
        dict with keys: auc, bal_acc, tpr, tnr, opt_threshold, tpr_at_tnr_<X>,
        n_total, n_pos, n_neg.
    """
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n_total = len(labels)
    n_pos = int(labels.sum())
    n_neg = n_total - n_pos

    out: Dict[str, Any] = {
        'n_total': n_total, 'n_pos': n_pos, 'n_neg': n_neg,
        'auc': float('nan'),
        'bal_acc': float('nan'), 'tpr': float('nan'), 'tnr': float('nan'),
        'opt_threshold': float('nan'),
    }
    if n_pos == 0 or n_neg == 0:
        return out

    # AUC
    order = np.argsort(-logits)
    sl = labels[order]
    tpr_pts = np.cumsum(sl) / n_pos
    fpr_pts = np.cumsum(1 - sl) / n_neg
    auc = float(np.trapezoid(tpr_pts, fpr_pts))
    if auc < 0:
        auc = 1.0 + auc
    out['auc'] = auc

    # Optimal balanced-accuracy threshold
    best_t, best_b, best_tpr, best_tnr = float(logits[0]), 0.5, 0.0, 1.0
    for t in np.unique(logits):
        p = (logits >= t).astype(np.int32)
        tpr = float(((p == 1) & (labels == 1)).sum()) / n_pos
        tnr = float(((p == 0) & (labels == 0)).sum()) / n_neg
        b = 0.5 * (tpr + tnr)
        if b > best_b:
            best_b, best_t, best_tpr, best_tnr = b, float(t), tpr, tnr
    # Classification F1 at that same bal-acc-optimal threshold.
    _p = (logits >= best_t).astype(np.int32)
    _tp = int(((_p == 1) & (labels == 1)).sum())
    _fp = int(((_p == 1) & (labels == 0)).sum())
    _fn = int(((_p == 0) & (labels == 1)).sum())
    _denom = 2 * _tp + _fp + _fn
    f1 = (2 * _tp / _denom) if _denom > 0 else float('nan')
    out.update(dict(opt_threshold=best_t, bal_acc=best_b, f1=f1,
                    tpr=best_tpr, tnr=best_tnr))

    # TPR at fixed TNR targets
    real_logits = logits[labels == 0]
    for target in fixed_tnr_targets:
        t = float(np.quantile(real_logits, target))
        tpr_at = float(((logits >= t) & (labels == 1)).sum()) / n_pos
        key = f'tpr_at_tnr_{int(round(target * 100)):02d}'
        out[key] = tpr_at

    return out
