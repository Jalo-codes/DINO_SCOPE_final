# `contrastive_test_v3` — cleaned modern experiment surface

`contrastive_test_v3` is an additive, cleaned copy of the recent
`contrastive_test_v2` BCE / contrastive / Swin / zoom work.

Existing v2 and legacy folders remain untouched.  Bucket-size prediction is
intentionally not part of v3; v3 may still report splice-size `area_tier`
(`tiny`, `small`, `medium`, `large`) as an evaluation breakdown.

## Active Scripts

- `scripts/train_multi_head.py` — joint image-BCE + contrastive localization.
- `scripts/train_image_bce.py` — image-level BCE baseline with Swin eval.
- `scripts/eval_swin_bce.py` — calibrated full-vs-window detection readout.
- `scripts/eval_zoom_recovery.py` — natural/oracle zoom recovery analysis.
- `scripts/eval_localization.py` — contrastive localization readout.
- `scripts/eval_patch_localization.py` — patch-BCE localization readout.
- `scripts/eval_robustness_bce.py` — image-BCE robustness sweep.
- `scripts/diagnose.py` — structured full / GT-crop / Swin diagnostics.

## Shared Utilities

Use `lab_utils.paths.resolve_data_paths()` for common data/run roots and
`lab_utils.data.area_tiers.area_tier()` for size reporting.  Reusable mechanics
should move to `lab_utils`; experiment names, split policy, and config presets
stay in this package.

## Excluded

The following v2 surfaces are deliberately excluded from v3:

- bucket-size detector training/eval
- bucket-size experiment specs
- bucket-size cost matrices
- bucket-size model/loss/eval imports
