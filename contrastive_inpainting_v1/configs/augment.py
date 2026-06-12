"""Experiment-owned augmentation parameter builders for contrastive_inpainting_v1.

The shared `lab_utils` layer receives explicit values.  This module is where
v2 decides which values correspond to a named eval mode or training regime.
"""

from typing import Dict, Tuple

from lab_utils.data.augment.corruptions import CorruptionSpec
from lab_utils.errors import ConfigError


def build_light_aug_kwargs(cfg) -> Dict:
    return dict(
        jpeg_prob=cfg.LIGHT_JPEG_PROB,
        jpeg_q_min=cfg.LIGHT_JPEG_Q_MIN,
        jpeg_q_max=cfg.LIGHT_JPEG_Q_MAX,
        noise_prob=cfg.LIGHT_NOISE_PROB,
        noise_std_min=cfg.LIGHT_NOISE_STD_MIN,
        noise_std_max=cfg.LIGHT_NOISE_STD_MAX,
        resize_prob=cfg.LIGHT_RESIZE_PROB,
        resize_scale_min=cfg.LIGHT_RESIZE_SCALE_MIN,
        resize_scale_max=cfg.LIGHT_RESIZE_SCALE_MAX,
        flip_prob=cfg.LIGHT_FLIP_PROB,
        blur_prob=cfg.LIGHT_BLUR_PROB,
        blur_sigma_min=cfg.LIGHT_BLUR_SIGMA_MIN,
        blur_sigma_max=cfg.LIGHT_BLUR_SIGMA_MAX,
    )


def build_medium_aug_kwargs(cfg) -> Dict:
    """Stronger than light — meant for robustness training, not destructive.

    Probabilities are bumped, ranges widened, but no individual augmentation
    is heavy enough to destroy the splice signal on its own. Designed so a
    minority of training samples receive each perturbation; the majority
    still see clean (or lightly-augmented) inputs.
    """
    return dict(
        jpeg_prob=0.45,
        jpeg_q_min=60,
        jpeg_q_max=90,
        noise_prob=0.30,
        noise_std_min=0.002,
        noise_std_max=0.025,
        resize_prob=0.25,
        resize_scale_min=0.65,
        resize_scale_max=0.95,
        flip_prob=cfg.LIGHT_FLIP_PROB,
        blur_prob=0.25,
        blur_sigma_min=0.2,
        blur_sigma_max=1.2,
    )


def build_heavy_aug_kwargs(cfg) -> Dict:
    """Aggressive — designed to overlap with eval-probe severities so the
    model trains on the same distributions it's tested against, while
    stopping short of destroying the splice signal.

    Severity floors are chosen to avoid signal destruction:
      - JPEG Q ≥ 40: below this the splice's JPEG history gets drowned out.
      - Gaussian σ ≤ 0.075: above this the splice-boundary stats wash out.
      - Resize ≥ 0.45: below this the 16-pixel patch grid becomes degenerate.

    Per-family probabilities are bumped so most samples see at least one
    perturbation; combinations are common (each family rolls independently).
    """
    return dict(
        jpeg_prob=0.55,
        jpeg_q_min=40,
        jpeg_q_max=95,
        noise_prob=0.45,
        noise_std_min=0.005,
        noise_std_max=0.075,
        resize_prob=0.40,
        resize_scale_min=0.45,
        resize_scale_max=0.95,
        flip_prob=cfg.LIGHT_FLIP_PROB,
        blur_prob=0.30,
        blur_sigma_min=0.2,
        blur_sigma_max=2.5,
    )


def build_aug_kwargs(cfg, intensity: str) -> Dict:
    """Dispatch on a string intensity name. 'none' returns no-op kwargs."""
    intensity = (intensity or 'light').lower()
    if intensity == 'none':
        return dict(
            jpeg_prob=0.0, jpeg_q_min=88, jpeg_q_max=98,
            noise_prob=0.0, noise_std_min=0.0, noise_std_max=0.0,
            resize_prob=0.0, resize_scale_min=0.95, resize_scale_max=1.0,
            flip_prob=cfg.LIGHT_FLIP_PROB,
            blur_prob=0.0, blur_sigma_min=0.0, blur_sigma_max=1.0,
        )
    if intensity == 'light':
        return build_light_aug_kwargs(cfg)
    if intensity == 'medium':
        return build_medium_aug_kwargs(cfg)
    if intensity == 'heavy':
        return build_heavy_aug_kwargs(cfg)
    raise ValueError(f"unknown aug intensity {intensity!r}; "
                     f"expected 'none' | 'light' | 'medium' | 'heavy'")


def build_degradation_kwargs(cfg) -> Dict:
    return dict(
        families=cfg.HARNESS_FAMILIES,
        variants=cfg.HARNESS_VARIANTS,
        variant_probs=cfg.HARNESS_VARIANT_PROBS,
        small_area=cfg.HARNESS_SMALL_AREA,
        large_area=cfg.HARNESS_LARGE_AREA,
        target_soft_max_area=cfg.HARNESS_TARGET_SOFT_MAX_AREA,
        target_clean_prob=cfg.HARNESS_TARGET_CLEAN_PROB,
        max_mask_overlap=cfg.HARNESS_MAX_MASK_OVERLAP,
        mask_tries=cfg.HARNESS_MASK_TRIES,
        jpeg_q_min=cfg.JPEG_Q_MIN,
        jpeg_q_max=cfg.JPEG_Q_MAX,
        double_jpeg_q1_min=cfg.DOUBLE_JPEG_Q1_MIN,
        double_jpeg_q1_max=cfg.DOUBLE_JPEG_Q1_MAX,
        double_jpeg_q2_min=cfg.DOUBLE_JPEG_Q2_MIN,
        double_jpeg_q2_max=cfg.DOUBLE_JPEG_Q2_MAX,
        gaussian_std_min=cfg.GAUSSIAN_STD_MIN,
        gaussian_std_max=cfg.GAUSSIAN_STD_MAX,
        poisson_peak_choices=cfg.POISSON_PEAK_CHOICES,
        resize_min=cfg.RESIZE_MIN,
        resize_max=cfg.RESIZE_MAX,
    )


def dataset_noise_kwargs(cfg) -> Dict:
    """Return v2 noise-supervision knobs for LabDataset."""
    return dict(
        use_degradation=cfg.USE_DEGRADATION,
        use_invariance=cfg.USE_INVARIANCE,
        use_splice_degradation=cfg.USE_SPLICE_DEGRADATION,
        splice_degradation_prob=cfg.SPLICE_DEGRADATION_PROB,
        splice_mask_corrupt_prob=cfg.SPLICE_MASK_CORRUPT_PROB,
        splice_mask_loss_weight=cfg.SPLICE_MASK_LOSS_WEIGHT,
        noise_head_splice_fp_weight=cfg.NOISE_HEAD_SPLICE_FP_WEIGHT,
        whole_image_corrupt_prob=cfg.WHOLE_IMAGE_CORRUPT_PROB,
        heavy_whole_aug_severity_thresh=cfg.HEAVY_WHOLE_AUG_SEVERITY_THRESH,
        heavy_aug_degrade_loss_weight=cfg.HEAVY_AUG_DEGRADE_LOSS_WEIGHT,
    )


def eval_aug_settings(mode: str, cfg) -> Dict:
    """Translate a v2 eval mode into explicit LabDataset kwargs."""
    mode = str(mode or 'none')
    if mode == 'none':
        return {
            'eval_aug_mode': 'none',
            'eval_corruption_spec': None,
            'eval_corruption_region': 'global',
        }

    region = 'mask' if mode.startswith('mask_') else 'global'
    family = mode.replace('global_', '').replace('mask_', '')
    if family == 'aggressive_mix':
        family = 'gaussian'

    specs = {
        'jpeg': CorruptionSpec('jpeg', {'quality': int(cfg.JPEG_Q_MIN)}, severity_tag='eval'),
        'double_jpeg': CorruptionSpec(
            'double_jpeg',
            {'q1': int(cfg.DOUBLE_JPEG_Q1_MIN), 'q2': int(cfg.DOUBLE_JPEG_Q2_MIN)},
            severity_tag='eval',
        ),
        'gaussian': CorruptionSpec('gaussian', {'std': float(cfg.GAUSSIAN_STD_MAX)}, severity_tag='eval'),
        'poisson': CorruptionSpec('poisson', {'peak': float(min(cfg.POISSON_PEAK_CHOICES))}, severity_tag='eval'),
        'resize': CorruptionSpec('resize', {'scale': float(cfg.RESIZE_MIN)}, severity_tag='eval'),
    }
    if family not in specs:
        raise ConfigError(f"unknown eval_aug mode {mode!r}")

    return {
        'eval_aug_mode': mode,
        'eval_corruption_spec': specs[family],
        'eval_corruption_region': region,
    }


EVAL_AUG_CHOICES: Tuple[str, ...] = (
    'none',
    'global_jpeg', 'global_double_jpeg', 'global_gaussian',
    'global_poisson', 'global_resize', 'aggressive_mix',
    'mask_jpeg', 'mask_double_jpeg', 'mask_gaussian',
    'mask_poisson', 'mask_resize',
)

DEFAULT_EVAL_AUG_CONDITIONS: Tuple[str, ...] = (
    'none',
    'global_jpeg',
    'global_double_jpeg',
    'global_gaussian',
    'global_poisson',
    'global_resize',
)


__all__ = [
    'EVAL_AUG_CHOICES',
    'DEFAULT_EVAL_AUG_CONDITIONS',
    'build_aug_kwargs',
    'build_degradation_kwargs',
    'build_light_aug_kwargs',
    'build_medium_aug_kwargs',
    'build_heavy_aug_kwargs',
    'dataset_noise_kwargs',
    'eval_aug_settings',
]
