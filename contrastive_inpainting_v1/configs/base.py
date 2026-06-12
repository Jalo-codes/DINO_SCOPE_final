"""contrastive_inpainting_v1.configs.base — base Config for v2 experiments.

Key difference from contrastive_test/configs/experiment_config.py:
  - Holds ONE Resolution instance instead of separate IMG_SIZE / NUM_PATCHES fields.
  - Resolution mismatch between config and AE cache raises ConfigError immediately.
  - No EMA fields.
"""

import dataclasses
from typing import Optional, Tuple

from lab_utils.data.resolution import Resolution


@dataclasses.dataclass
class Config:
    # ── Resolution (single source of truth) ──────────────────────────────
    resolution: Resolution = dataclasses.field(
        default_factory=lambda: Resolution(image_size=448, patch_size=16)
    )

    # ── Backbone ──────────────────────────────────────────────────────────
    MODEL_NAME: str = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'

    # ── LoRA ──────────────────────────────────────────────────────────────
    LORA_RANK: int = 32
    LORA_ALPHA: int = 64
    LORA_DROPOUT: float = 0.1
    LORA_TARGETS: tuple = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                           'up_proj', 'down_proj')

    # ── Projector ─────────────────────────────────────────────────────────
    PROJ_DIM: int = 128
    NUM_HEADS: int = 2
    HEAD_DIM: int = 128

    # ── Optimization ──────────────────────────────────────────────────────
    LEARNING_RATE: float = 2e-4
    LR_MIN_FRAC: float = 0.05
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 20
    TRAIN_BATCH_SIZE: int = 4
    GRAD_ACCUM_STEPS: int = 4
    TRAIN_SAMPLES_PER_EPOCH: Optional[int] = 2000

    # ── Contrastive loss: symmetric (default) vs legacy ───────────────────
    # 'symmetric' — mirrored dead-point hinges (similar≥TAU_POS, diff≤TAU_NEG),
    #               √-balanced region terms, active-pair denominator. The v2
    #               default; localization-only (BCE owns detection + polarity).
    # 'legacy'    — the original point-attract/hinge loss (NEG_MARGIN +
    #               ATTRACT_MARGIN + SINGLE_CLASS_ATTRACT_*); kept for parity.
    CONTRASTIVE_LOSS_MODE: str = 'symmetric'
    TAU_POS: float = 0.55           # same-label cohesion floor (no upper bound)
    TAU_NEG: float = 0.20           # diff-label separation ceiling
    AREA_BALANCE_POWER: float = 0.5  # sqrt-tempered region balance (0=full, 1=raw)
    # Violation-count sensitivity of the per-region hinge means. 1.0 = mean over
    # active (violating) pairs only — count-insensitive, plateaus (legacy). 0.0 =
    # mean over all pairs — "more wrong = more loss" but late gradient starves.
    # 0.5 = geometric blend: count moves the scalar, few violations still boosted.
    CONTRASTIVE_NORM_POWER: float = 1.0
    # Full real images contribute only a *very low* cohesion signal — reals are
    # localization-irrelevant; the BCE head owns detection.
    CONTRASTIVE_SINGLE_CLASS_WEIGHT: float = 0.05
    # Deferred anti-collapse / mining hooks (OFF by default; gate on measurement).
    CONTRASTIVE_DIVERSITY_WEIGHT: float = 0.0
    CONTRASTIVE_DIVERSITY_TAU: float = 0.90
    CONTRASTIVE_TOPK: int = 0

    # ── Loss weights ──────────────────────────────────────────────────────
    NEG_MARGIN: float = 0.3
    ATTRACT_MARGIN: float = 0.4          # legacy loss only
    SINGLE_CLASS_ATTRACT_MARGIN: float = 1.0   # legacy loss only
    SINGLE_CLASS_ATTRACT_SQUARED: bool = True  # legacy loss only
    SINGLE_CLASS_TOPK: int = 0           # legacy loss only
    LAMBDA_REPEL: float = 1.0
    SINGLE_CLASS_WEIGHT: float = 1.0     # legacy loss only (other trainers)
    TRAIN_SPLICE_SAMPLE_FRAC: float = 0.50
    INVARIANCE_WEIGHT: float = 0.20
    DEGRADE_WEIGHT: float = 1.00
    DEGRADE_SINGLE_CLASS_WEIGHT: float = 0.35
    ORTHO_WEIGHT: float = 0.05
    BCE_POS_WEIGHT: float = 3.0
    BCE_WEIGHT: float = 0.35

    # ── Blob / splice geometry ────────────────────────────────────────────
    BLOB_MIN_AREA: float = 0.10
    BLOB_MAX_AREA: float = 0.40
    IMD2020_MIN_MASK_PATCH_FRAC: float = 0.03

    # ── Crop params ───────────────────────────────────────────────────────
    CROP_SCALE: Tuple[float, float] = (0.60, 1.00)
    CROP_RATIO: Tuple[float, float] = (0.75, 1.33)
    IMD_CROP_SCALE: Tuple[float, float] = (0.60, 1.00)
    IMD_CROP_RATIO: Tuple[float, float] = (0.75, 1.33)
    CROP_MAX_TRIES: int = 24

    # ── Light augmentations ───────────────────────────────────────────────
    LIGHT_JPEG_PROB: float = 0.25
    LIGHT_JPEG_Q_MIN: int = 88
    LIGHT_JPEG_Q_MAX: int = 98
    LIGHT_NOISE_PROB: float = 0.15
    LIGHT_NOISE_STD_MIN: float = 0.002
    LIGHT_NOISE_STD_MAX: float = 0.015
    LIGHT_RESIZE_PROB: float = 0.20
    LIGHT_RESIZE_SCALE_MIN: float = 0.80
    LIGHT_RESIZE_SCALE_MAX: float = 0.98
    LIGHT_BLUR_PROB: float = 0.0
    LIGHT_BLUR_SIGMA_MIN: float = 0.0
    LIGHT_BLUR_SIGMA_MAX: float = 1.0
    LIGHT_FLIP_PROB: float = 0.50

    # ── Corruption severity ranges ────────────────────────────────────────
    JPEG_Q_MIN: int = 35
    JPEG_Q_MAX: int = 65
    DOUBLE_JPEG_Q1_MIN: int = 82
    DOUBLE_JPEG_Q1_MAX: int = 96
    DOUBLE_JPEG_Q2_MIN: int = 45
    DOUBLE_JPEG_Q2_MAX: int = 82
    GAUSSIAN_STD_MIN: float = 0.10
    GAUSSIAN_STD_MAX: float = 0.30
    POISSON_PEAK_CHOICES: tuple = (16, 24, 32, 48, 64, 96)
    RESIZE_MIN: float = 0.55
    RESIZE_MAX: float = 0.90

    # ── Degradation harness ───────────────────────────────────────────────
    HARNESS_FAMILIES: tuple = ('jpeg', 'double_jpeg', 'gaussian', 'poisson', 'resize')
    HARNESS_VARIANTS: tuple = (
        'clean_plus_small', 'global_plus_small', 'large_plus_small', 'global_negative'
    )
    HARNESS_VARIANT_PROBS: tuple = (0.35, 0.25, 0.25, 0.15)
    HARNESS_SMALL_AREA: Tuple[float, float] = (0.05, 0.25)
    HARNESS_LARGE_AREA: Tuple[float, float] = (0.45, 0.80)
    HARNESS_TARGET_SOFT_MAX_AREA: float = 0.45
    HARNESS_TARGET_CLEAN_PROB: float = 0.20
    HARNESS_MAX_MASK_OVERLAP: float = 0.08
    HARNESS_MASK_TRIES: int = 24
    USE_DEGRADATION: bool = True
    USE_INVARIANCE: bool = True
    USE_SPLICE_DEGRADATION: bool = True
    SPLICE_DEGRADATION_PROB: float = 0.70
    SPLICE_MASK_CORRUPT_PROB: float = 0.35
    SPLICE_MASK_LOSS_WEIGHT: float = 0.20
    NOISE_HEAD_SPLICE_FP_WEIGHT: float = 0.10
    WHOLE_IMAGE_CORRUPT_PROB: float = 0.10
    HEAVY_WHOLE_AUG_SEVERITY_THRESH: float = 0.65
    HEAVY_AUG_DEGRADE_LOSS_WEIGHT: float = 0.50

    # ── Eval ──────────────────────────────────────────────────────────────
    GATE_INIT_TAU: float = 0.10
    BCE_THRESHOLD: float = 0.50
    CALIBRATION_FRAC: float = 0.50
    CALIBRATION_MIN_ITEMS: int = 32
    CALIBRATION_SINGLE_ITEMS_PER_SOURCE: int = 25
    CALIBRATION_SPLICE_ITEMS_PER_SOURCE: int = 25
    EVAL_TAU_OFFSETS: tuple = (-0.08, -0.04, -0.02, 0.00, 0.02, 0.04, 0.08)
    # Gate mode for the silhouette/null gate at eval time.
    #   'silhouette'      — global tau on raw silhouette (legacy default)
    #   'silhouette_null' — per-image z-score against random-partition null;
    #                       content-prior-agnostic, expected to transfer better
    #                       across datasets (e.g. IMD2020 → CASIA)
    GATE_MODE: str = 'silhouette'
    # Random partitions per image used to build the silhouette null
    # (only consulted when GATE_MODE='silhouette_null').
    EVAL_NULL_SHUFFLES: int = 32
    # Tau-sweep offsets used in null mode (z-score units, not silhouette units).
    # Override only when GATE_MODE='silhouette_null'; the silhouette-mode
    # offsets above are kept distinct so legacy runs are unaffected.
    EVAL_TAU_Z_OFFSETS: tuple = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
    EVAL_EVERY: int = 1
    FULL_EVAL_EVERY: int = 5
    FULL_EVAL_CASIA_MAX_ITEMS: int = 500
    QUICK_VAL_ITEMS: int = 512
    QUICK_VAL_IMD_CASES: int = 300
    QUICK_VAL_CASIA_PAIRS: int = 300
    EVAL_FAMILY_EVERY: int = 0
    LOG_EVERY: int = 20
    CKPT_EVERY: int = 1
    EARLY_STOP_WARMUP: int = 4
    EARLY_STOP_PATIENCE: int = 4
    EARLY_STOP_MIN_DELTA: float = 0.002

    # ── Dataset paths ─────────────────────────────────────────────────────
    INDOOR_DATASET_ROOT: str = '/content/indoor_dataset'
    INDOOR_HOLDOUT_SUBDIR: str = 'unclassified'
    IMD2020_ROOT: str = '/content/IMD2020'
    IMD2020_VAL_SPLIT: float = 0.10
    IMD2020_SPLIT_SEED: int = 42
    CASIA_ROOT: str = '/content/casia'
    CASIA_VAL_SPLIT: float = 0.15
    CASIA_SPLIT_SEED: int = 42
    INDOOR_REAL_CAP: int = 512

    # ── ImageNet normalisation ────────────────────────────────────────────
    IMAGENET_MEAN: tuple = (0.485, 0.456, 0.406)
    IMAGENET_STD:  tuple = (0.229, 0.224, 0.225)

    # ── Runtime ──────────────────────────────────────────────────────────
    CHECKPOINT_ROOT: str = '/content/checkpoints_v2'
    TRAIN_NUM_WORKERS: int = 4
    VAL_NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    SEED: int = 42

    valid_exts: tuple = ('.jpg', '.png', '.jpeg', '.webp', '.bmp')

    # ── Convenience properties (replace old IMG_SIZE / NUM_PATCHES direct access) ──

    @property
    def IMG_SIZE(self) -> int:
        return self.resolution.image_size

    @property
    def PATCH_SIZE(self) -> int:
        return self.resolution.patch_size

    @property
    def NUM_PATCHES(self) -> int:
        return self.resolution.num_patches
