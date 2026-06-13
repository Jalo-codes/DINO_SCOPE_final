"""lab_utils.data.dataset — LabDataset, the single dataset class for all experiments.

Dispatches by item['kind']:
    imd_splice   — real-world splice from IMD2020 or CASIA with a GT mask
    casia_splice — alias for imd_splice (same logic)
    imd_real     — authentic (negative) image from IMD2020 / CASIA
    indoor_real  — authentic indoor image (negative); may receive degradation
    ae_splice    — AE-reconstructed composite (requires ae_cache)

Standard sample dict (all __getitem__ returns):
    img           : Tensor (3, H, W) — normalized
    splice_labels : Tensor (num_patches,) long
    supervised    : Tensor bool
    is_single     : Tensor bool
    degrade_labels      : Tensor (num_patches,) long
    degrade_supervised  : Tensor bool
    is_single_degrade   : Tensor bool
    degrade_meta_*      : various scalar tensors
    invariance_clean    : Tensor (3, H, W)
    invariance_aug      : Tensor (3, H, W)
    invariance_active   : Tensor bool
    meta          : dict — path, kind, case_id, source, applied_ops, …

Shape contract: img tensor is asserted to be (3, res.image_size, res.image_size)
before return.  DataError is raised (not a silent None) on shape mismatch.
"""

import hashlib
import io
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from lab_utils.errors import DataError
from lab_utils.logging.text import log_warn
from lab_utils.data.resolution import (
    Resolution,
    mask_to_patch_labels,
    mask_to_patch_labels_soft,
    oracle_mask_crop,
    random_resized_crop_pair,
    resize_only,
    resize_only_mask,
)
from lab_utils.data.blob import EllipseBlobParams, generate_blob_mask_pil
from lab_utils.data.paste import paste_regional_ae
from lab_utils.data.augment.light import apply_light_augmentations
from lab_utils.data.augment.corruptions import (
    CorruptionSpec,
    apply_corruption,
    make_invariance_pair,
    sample_corruption_spec,
)
from lab_utils.data.augment.degradation import build_degradation_example


# ── helpers ──────────────────────────────────────────────────────────────────

def _stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)


def _zero_degrade_meta() -> Dict[str, Any]:
    return {
        'degrade_type': 'none',
        'treatment_side': 'none',
        'variant': 'none',
        'dominant_family': 'clean',
        'target_family': 'none',
        'num_regimes': 0,
        'num_treated_regions': 0,
        'target_area_frac': 0.0,
        'dominant_area_frac': 0.0,
        'typed_target_regime_id': -1,
        'dominant_regime_id': -1,
        'is_global_negative': 0,
        'is_multi_region': 0,
        'has_large_context': 0,
    }


def _applied_to_dicts(applied) -> List[Dict[str, Any]]:
    return [
        {'name': op.name, 'params': op.params, 'severity': op.severity}
        for op in applied
    ]


def _full_mask(res: Resolution) -> Image.Image:
    return Image.new('L', (res.image_size, res.image_size), 255)


_CORRUPTION_SAMPLER_KEYS = {
    'jpeg_q_min',
    'jpeg_q_max',
    'double_jpeg_q1_min',
    'double_jpeg_q1_max',
    'double_jpeg_q2_min',
    'double_jpeg_q2_max',
    'gaussian_std_min',
    'gaussian_std_max',
    'poisson_peak_choices',
    'resize_min',
    'resize_max',
}


def _corruption_sampler_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in _CORRUPTION_SAMPLER_KEYS}


def _composite_corruption(
    img: Image.Image,
    mask: Image.Image,
    spec: CorruptionSpec,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    aug = apply_corruption(img, spec)
    return Image.composite(aug.image, img, mask), _applied_to_dicts(aug.applied)


# ── LabDataset ───────────────────────────────────────────────────────────────

class LabDataset(Dataset):
    """General-purpose dataset for all lab_utils experiments.

    Args:
        items:          List of item dicts (see indexer.py for format).
        res:            Resolution — image_size and patch_size.
        normalize_mean: ImageNet-style channel means.
        normalize_std:  ImageNet-style channel stds.
        augment:        True during training (random crops + light augs).
        use_degradation: Build degradation examples for real negatives.
        use_invariance:  Build invariance pairs for real negatives.
        ae_cache:       AECache instance required for 'ae_splice' items.
        blob_params:    EllipseBlobParams for AE splice blob generation.
        imd_crop_scale: Random crop scale range for imd_splice items.
        imd_crop_ratio: Random crop ratio range for imd_splice items.
        crop_scale:     Random crop scale range for other items.
        crop_ratio:     Random crop ratio range for other items.
        crop_max_tries: Max random crop attempts before centre-crop fallback.
        min_mask_patch_frac: Minimum patch fraction to accept as supervised.
        light_aug_kwargs: Forwarded to apply_light_augmentations.
        degradation_kwargs: Forwarded to build_degradation_example.
        use_splice_degradation: Add noise/compression regions to splice images.
        splice_degradation_prob: Chance to add a local/mask noise target on a
            splice image; otherwise the noise head receives a light all-clean
            negative if noise_head_splice_fp_weight > 0.
        splice_mask_corrupt_prob: Fraction of splice degradation examples that
            target the GT splice mask itself.
        splice_mask_loss_weight: Supervision weight when the splice mask is
            intentionally corrupted; default 0.2 implements the 1/5 penalty.
        whole_image_corrupt_prob: Training-time chance to corrupt the whole image.
        heavy_whole_aug_severity_thresh: If a severe full-image corruption is
            layered after a local noise target, lower only the degradation-head
            sample weight.
        heavy_aug_degrade_loss_weight: Degradation-head sample weight after
            severe full-image augmentation.
        eval_aug_mode: Metadata label for deterministic eval augmentation.
        eval_corruption_spec: Explicit corruption to apply at eval time.
        eval_corruption_region: 'global' or 'mask'. Mask falls back to global
            if a sample has no mask.
        deterministic_seed: Seed for reproducible val evaluation.
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        res: Resolution,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std:  Tuple[float, float, float] = (0.229, 0.224, 0.225),
        *,
        augment: bool = True,
        use_degradation: bool = False,
        use_invariance: bool = False,
        ae_cache=None,
        blob_params: Optional[EllipseBlobParams] = None,
        imd_crop_scale: Tuple[float, float] = (0.18, 1.00),
        imd_crop_ratio: Tuple[float, float] = (0.60, 1.70),
        crop_scale: Tuple[float, float] = (0.40, 1.00),
        crop_ratio: Tuple[float, float] = (0.75, 1.33),
        crop_max_tries: int = 24,
        # Optional mixture distributions over crop scale: list of
        # ((lo, hi), weight) tuples. When provided, overrides crop_scale /
        # imd_crop_scale and samples a (lo, hi) range per item according to
        # the weights, then uses uniform sampling within that range.
        crop_scale_mix: Optional[List[Tuple[Tuple[float, float], float]]] = None,
        imd_crop_scale_mix: Optional[List[Tuple[Tuple[float, float], float]]] = None,
        min_mask_patch_frac: float = 0.03,
        # Splice crop fallback policy when the random retries can't surface the
        # splice at >= min_mask_patch_frac coverage:
        #   'random'         — legacy: center/resize fallback. The crop misses
        #                      the (small) splice, so image-level the sample is
        #                      relabeled REAL (is_single) — silently throwing
        #                      away tiny positives. Fine for big-splice data.
        #   'oracle_fallback'— mask-centered ZOOM crop that guarantees the
        #                      splice is in frame at oracle_target_cov coverage
        #                      (a real positive), and DROPS the sample only if
        #                      the mask is empty. Use for tiny-splice data (CASIA).
        splice_crop_mode: str = 'random',
        oracle_target_cov: Tuple[float, float] = (0.10, 0.40),
        light_aug_kwargs: Optional[Dict] = None,
        degradation_kwargs: Optional[Dict] = None,
        use_splice_degradation: bool = False,
        splice_degradation_prob: float = 1.0,
        splice_mask_corrupt_prob: float = 0.0,
        splice_mask_loss_weight: float = 1.0,
        noise_head_splice_fp_weight: float = 0.0,
        whole_image_corrupt_prob: float = 0.0,
        heavy_whole_aug_severity_thresh: float = 1.01,
        heavy_aug_degrade_loss_weight: float = 1.0,
        eval_aug_mode: str = 'none',
        eval_corruption_spec: Optional[CorruptionSpec] = None,
        eval_corruption_region: str = 'global',
        deterministic_seed: int = 0,
        gt_patch_threshold: float = 0.15,
        gt_soft_label: bool = False,
        gt_soft_low: float = 0.02,
        gt_soft_high: float = 0.06,
        # Natural zoom-coverage crop: when set (lo, hi), splice crops target an
        # IN-FRAME coverage band [lo, hi] via size-biased RANDOM-position crops
        # (off-center allowed), with the jittered oracle as fallback. None ⇒
        # legacy behavior (min_mask_patch_frac lower bound only).
        splice_cov_band: Optional[Tuple[float, float]] = None,
        # Crop even when augment=False (geometric crop without photometric aug).
        # Enables a "clean-but-cropped" zoom val alongside the full-frame val.
        force_crop: bool = False,
        # Fraction of inpaint items (those carrying real_path) that get the
        # pristine-background paste. 1.0 = always paste (regional splice). With
        # paste_frac < 1, a (1 - paste_frac) fraction is left as the raw
        # full-VAE frame — a full-AE positive whose background carries generator
        # noise but is still labeled REAL per-patch: the hard negative that
        # teaches "VAE noise != forgery". Only fires when augment=True so eval
        # sets stay deterministic (always pasted).
        paste_frac: float = 1.0,
    ):
        self.items               = items
        self.res                 = res
        self.augment             = bool(augment)
        self.use_degradation     = bool(use_degradation)
        self.use_invariance      = bool(use_invariance)
        self.ae_cache            = ae_cache
        self.blob_params         = blob_params or EllipseBlobParams()
        self.crop_scale_mix      = self._validate_mix(crop_scale_mix)
        self.imd_crop_scale_mix  = self._validate_mix(imd_crop_scale_mix)
        self.imd_crop_scale      = imd_crop_scale
        self.imd_crop_ratio      = imd_crop_ratio
        self.crop_scale          = crop_scale
        self.crop_ratio          = crop_ratio
        self.crop_max_tries      = int(crop_max_tries)
        self.min_mask_patch_frac = float(min_mask_patch_frac)
        self.splice_crop_mode    = str(splice_crop_mode)
        self.oracle_target_cov   = tuple(oracle_target_cov)
        # Per-epoch crop telemetry (drain via drain_crop_stats()). Reliable
        # with num_workers=0; with workers each copy tracks its own shard.
        self._crop_tally: Dict[str, int] = {
            'random': 0, 'oracle': 0, 'fallback': 0, 'dropped': 0,
        }
        self._crop_cov_sum = 0.0
        self._crop_cov_n   = 0
        self._last_crop_mode = 'random'
        self._last_crop_cov  = 0.0
        self.light_aug_kwargs    = light_aug_kwargs or {}
        self.degradation_kwargs  = degradation_kwargs or {}
        self.use_splice_degradation = bool(use_splice_degradation)
        self.splice_degradation_prob = float(splice_degradation_prob)
        self.splice_mask_corrupt_prob = float(splice_mask_corrupt_prob)
        self.splice_mask_loss_weight = float(splice_mask_loss_weight)
        self.noise_head_splice_fp_weight = float(noise_head_splice_fp_weight)
        self.whole_image_corrupt_prob = float(whole_image_corrupt_prob)
        self.heavy_whole_aug_severity_thresh = float(heavy_whole_aug_severity_thresh)
        self.heavy_aug_degrade_loss_weight = float(heavy_aug_degrade_loss_weight)
        self.eval_aug_mode = str(eval_aug_mode or 'none')
        self.eval_corruption_spec = eval_corruption_spec
        self.eval_corruption_region = str(eval_corruption_region or 'global')
        self.deterministic_seed  = int(deterministic_seed)
        self.gt_patch_threshold  = float(gt_patch_threshold)
        self.gt_soft_label       = bool(gt_soft_label)
        self.gt_soft_low         = float(gt_soft_low)
        self.gt_soft_high        = float(gt_soft_high)
        if self.gt_soft_label and not (0.0 < self.gt_soft_low < self.gt_soft_high <= 1.0):
            raise ValueError(
                f'LabDataset: gt_soft_low/gt_soft_high must satisfy '
                f'0 < low < high <= 1, got low={self.gt_soft_low}, '
                f'high={self.gt_soft_high}'
            )
        self.splice_cov_band = (
            (float(splice_cov_band[0]), float(splice_cov_band[1]))
            if splice_cov_band is not None else None
        )
        if self.splice_cov_band is not None:
            _lo, _hi = self.splice_cov_band
            if not (0.0 < _lo < _hi <= 1.0):
                raise ValueError(
                    f'LabDataset: splice_cov_band must satisfy 0 < lo < hi <= 1, '
                    f'got {self.splice_cov_band}'
                )
        self.force_crop = bool(force_crop)
        self.paste_frac = float(paste_frac)
        # Matched-zoom support: when splice_cov_band is active, reals must be
        # cropped to the SAME distribution of zoom levels (crop-area fractions)
        # that splices realize, or the BCE head learns "tight crop = fake" as a
        # shortcut. We sample a splice mask area on the fly per real and apply the
        # identical scale formula (crop_area ≈ a_src / coverage). Cache mask areas
        # to bound the extra I/O to one read per distinct mask.
        self._splice_items = [
            it for it in items
            if it.get('kind') in ('imd_splice', 'casia_splice') and it.get('mask')
        ]
        self._splice_area_cache: Dict[str, Optional[float]] = {}
        self.normalize           = transforms.Normalize(
            list(normalize_mean), list(normalize_std)
        )
        self._log_config()

    def _log_config(self) -> None:
        """Emit [data] log lines describing this dataset's full configuration.

        Called once at construction.  In distributed runs this fires on every
        rank, but only rank-0 has install_log() called so only rank-0 writes
        to the file.  Other ranks print to their own stdout (harmless).
        """
        from collections import Counter
        from lab_utils.logging.text import log_line, _LOG_PATH  # noqa: PLC0415

        # Only log when a log file is active (rank-0 in distributed; tests that
        # haven't called install_log won't spam stdout).
        if _LOG_PATH is None:
            return

        kind_counts = dict(sorted(Counter(
            str(i.get('kind', 'unknown')) for i in self.items
        ).items()))

        # 1 ── Summary
        log_line(
            f'[data] LabDataset: n={len(self.items)} augment={self.augment} '
            f'use_degradation={self.use_degradation} '
            f'use_invariance={self.use_invariance} '
            f'use_splice_degradation={self.use_splice_degradation}'
        )
        # 2 ── Kind breakdown
        log_line(f'[data] LabDataset kinds: {kind_counts}')
        if self.augment and self.paste_frac < 1.0:
            log_line(
                f'[data] LabDataset paste_frac={self.paste_frac:.2f} '
                f'(~{1.0 - self.paste_frac:.0%} of inpaint items kept as full-AE positives)'
            )

        # 3 ── Light augmentation config
        lak = self.light_aug_kwargs
        log_line(
            f'[data] LabDataset light_aug: '
            f'jpeg={lak.get("jpeg_prob", 0.0):.2f} '
            f'jpeg_q=[{lak.get("jpeg_q_min", 88)},{lak.get("jpeg_q_max", 98)}] '
            f'noise={lak.get("noise_prob", 0.0):.2f} '
            f'noise_std=[{lak.get("noise_std_min", 0.0):.4f},'
            f'{lak.get("noise_std_max", 0.0):.4f}] '
            f'poisson={lak.get("poisson_prob", 0.0):.2f} '
            f'poisson_peak=[{lak.get("poisson_peak_min", 0.0):.0f},'
            f'{lak.get("poisson_peak_max", 0.0):.0f}] '
            f'resize={lak.get("resize_prob", 0.0):.2f} '
            f'resize_scale=[{lak.get("resize_scale_min", 0.0):.2f},'
            f'{lak.get("resize_scale_max", 0.0):.2f}] '
            f'flip={lak.get("flip_prob", 0.0):.2f}'
        )

        # 4 ── Degradation families (only when active)
        if self.use_degradation:
            dk = self.degradation_kwargs
            log_line(
                f'[data] LabDataset degradation: '
                f'families={dk.get("families", ())} '
                f'variants={dk.get("variants", ())} '
                f'variant_probs={dk.get("variant_probs", ())} '
                f'small_area={dk.get("small_area", ())} '
                f'large_area={dk.get("large_area", ())}'
            )

        # 5 ── Splice-region degradation (only when active)
        if self.use_splice_degradation:
            log_line(
                f'[data] LabDataset splice_degrade: '
                f'prob={self.splice_degradation_prob:.2f} '
                f'mask_corrupt_prob={self.splice_mask_corrupt_prob:.2f} '
                f'mask_loss_weight={self.splice_mask_loss_weight:.2f} '
                f'fp_weight={self.noise_head_splice_fp_weight:.2f}'
            )

        # 6 ── Whole-image corruption (only when non-zero)
        if self.whole_image_corrupt_prob > 0.0:
            log_line(
                f'[data] LabDataset whole_corrupt: '
                f'prob={self.whole_image_corrupt_prob:.2f} '
                f'heavy_thresh={self.heavy_whole_aug_severity_thresh:.2f} '
                f'heavy_loss_weight={self.heavy_aug_degrade_loss_weight:.2f}'
            )

        # 7 ── Crop parameters
        log_line(
            f'[data] LabDataset crop: '
            f'scale={self.crop_scale} ratio={self.crop_ratio} '
            f'imd_scale={self.imd_crop_scale} imd_ratio={self.imd_crop_ratio} '
            f'max_tries={self.crop_max_tries}'
        )
        if self.crop_scale_mix is not None:
            log_line(f'[data] LabDataset crop_scale_mix={self.crop_scale_mix}')
        if self.imd_crop_scale_mix is not None:
            log_line(f'[data] LabDataset imd_crop_scale_mix={self.imd_crop_scale_mix}')
        if self.splice_cov_band is not None or self.force_crop:
            _real_match = (self.splice_cov_band is not None and len(self._splice_items) > 0)
            log_line(
                f'[data] LabDataset zoom-coverage: '
                f'splice_cov_band={self.splice_cov_band} force_crop={self.force_crop} '
                f'real_matched_zoom={_real_match} (n_splice_pool={len(self._splice_items)}) '
                f'(splice crops target this in-frame coverage w/ jittered oracle '
                f'fallback; reals cropped to the SAME zoom distribution so crop '
                f'tightness is not a fake/real cue)'
            )

        # 8 ── Supervision threshold + eval mode
        log_line(
            f'[data] LabDataset supervision: '
            f'min_mask_patch_frac={self.min_mask_patch_frac:.4f} '
            f'gt_patch_threshold={self.gt_patch_threshold:.3f} '
            f'gt_soft_label={self.gt_soft_label} '
            f'gt_soft_low={self.gt_soft_low:.3f} '
            f'gt_soft_high={self.gt_soft_high:.3f} '
            f'eval_aug_mode={self.eval_aug_mode!r}'
        )

    def __len__(self) -> int:
        return len(self.items)

    # ── crop helpers ─────────────────────────────────────────────────────────

    def _is_imd_kind(self, kind: str) -> bool:
        return kind in ('imd_splice', 'imd_real', 'casia_splice')

    def _patch_labels_and_weights(
        self, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rasterize a mask to per-patch labels + per-patch supervision weights.

        - Hard mode (gt_soft_label=False): weights are all 1.0, labels are
          binary (density > gt_patch_threshold).
        - Soft mode (gt_soft_label=True): density-band logic per
          mask_to_patch_labels_soft (ignore band, ramp, etc.).
        """
        if self.gt_soft_label:
            return mask_to_patch_labels_soft(
                mask, self.res, low=self.gt_soft_low, high=self.gt_soft_high
            )
        labels = mask_to_patch_labels(
            mask, self.res, threshold=self.gt_patch_threshold
        )
        weights = torch.ones(self.res.num_patches, dtype=torch.float32)
        return labels, weights

    @staticmethod
    def _validate_mix(mix):
        if mix is None:
            return None
        if not isinstance(mix, (list, tuple)) or not mix:
            raise ValueError(f'crop_scale_mix must be a non-empty list of '
                             f'((lo, hi), weight) tuples, got {mix!r}')
        for entry in mix:
            try:
                rng, w = entry
                lo, hi = rng
            except Exception:
                raise ValueError(f'crop_scale_mix entry must be ((lo, hi), weight), '
                                 f'got {entry!r}')
            if not (0.0 < float(lo) <= float(hi) <= 1.0):
                raise ValueError(f'crop_scale_mix range must satisfy 0 < lo <= hi <= 1, '
                                 f'got ({lo}, {hi})')
            if float(w) <= 0:
                raise ValueError(f'crop_scale_mix weight must be > 0, got {w}')
        return list(mix)

    def _sample_scale(self, mix, default):
        """Pick a (lo, hi) range either from a mixture or fall back to default."""
        if mix is None:
            return default
        weights = np.array([float(w) for _, w in mix], dtype=np.float64)
        weights = weights / weights.sum()
        idx = int(np.random.choice(len(mix), p=weights))
        return mix[idx][0]

    def _splice_area_frac(self, item: Dict) -> Optional[float]:
        """Source-image area fraction of a splice mask (cached by mask path)."""
        p = item.get('mask')
        if not p:
            return None
        if p in self._splice_area_cache:
            return self._splice_area_cache[p]
        a: Optional[float]
        try:
            m = Image.open(p).convert('L')
            a = float((np.asarray(m, dtype=np.uint8) > 0).mean())
        except Exception:
            a = None
        self._splice_area_cache[p] = a
        return a

    def _sample_matched_real_scale(
        self, cov_lo: float, cov_hi: float
    ) -> Optional[float]:
        """Draw a crop-area fraction for a REAL that matches the splice zoom
        distribution: pick a random splice's source mask area a_src and a target
        coverage ~U(cov_lo, cov_hi), then crop_area ≈ a_src / coverage — the same
        formula splice crops use. Returns None if no splice pool is available
        (caller then keeps the legacy scale)."""
        if not self._splice_items:
            return None
        for _ in range(4):  # a few tries in case of unreadable masks
            it = self._splice_items[random.randrange(len(self._splice_items))]
            a = self._splice_area_frac(it)
            if a is not None and a > 0.0:
                cov = random.uniform(cov_lo, cov_hi)
                return min(1.0, max(0.02, a / cov))
        return None

    def _record_crop(self, mode: str, coverage: float) -> None:
        """Tally a crop outcome for per-epoch telemetry + remember it for meta."""
        self._last_crop_mode = mode
        self._last_crop_cov  = float(coverage)
        if mode in self._crop_tally:
            self._crop_tally[mode] += 1
        if coverage > 0.0:
            self._crop_cov_sum += float(coverage)
            self._crop_cov_n   += 1

    def drain_crop_stats(self) -> Dict[str, Any]:
        """Return + reset the per-epoch crop telemetry.

        Counts by mode ('random'/'oracle'/'fallback'/'dropped') plus the mean
        realized splice coverage. Reliable with ``num_workers=0`` (single
        process); with workers each copy reports only its own shard.
        """
        stats: Dict[str, Any] = dict(self._crop_tally)
        stats['cov_mean'] = (self._crop_cov_sum / self._crop_cov_n) if self._crop_cov_n else 0.0
        stats['cov_n']    = int(self._crop_cov_n)
        self._crop_tally  = {k: 0 for k in self._crop_tally}
        self._crop_cov_sum = 0.0
        self._crop_cov_n   = 0
        return stats

    def _crop_item(
        self, item: Dict, img: Image.Image, mask: Optional[Image.Image]
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image], bool]:
        """Return (cropped_img, cropped_mask, crop_valid).

        ``cropped_img is None`` signals the caller should DROP this sample —
        only happens for ``splice_crop_mode='oracle_fallback'`` when the splice
        mask is empty (a splice-free crop must never carry a fake label).
        """
        kind = item.get('kind', 'indoor_real')
        is_splice = kind in ('imd_splice', 'casia_splice')

        # Cropping is GEOMETRIC and gated separately from photometric aug:
        # force_crop=True lets an eval (augment=False) loader still zoom-crop
        # (clean-but-cropped val) with no photometric augmentation applied.
        do_crop = self.augment or self.force_crop
        if not do_crop:
            out_img  = resize_only(img, self.res)
            out_mask = resize_only_mask(mask, self.res).convert('L') if mask is not None else None
            self._record_crop('resize', 0.0)   # full-frame eval path
            return out_img, out_mask, True

        band = self.splice_cov_band
        if self._is_imd_kind(kind):
            scale = self._sample_scale(self.imd_crop_scale_mix, self.imd_crop_scale)
            ratio = self.imd_crop_ratio
        else:
            scale = self._sample_scale(self.crop_scale_mix, self.crop_scale)
            ratio = self.crop_ratio

        # Coverage-band mode (natural zoom): size the crop so a contained splice
        # lands at ~[lo, hi] IN-FRAME coverage, but keep the position RANDOM
        # (off-center allowed) — unlike the mask-centered oracle. Acceptance =
        # realized in-frame patch fraction inside the band; misses fall back to
        # the jittered oracle below.
        cov_lo = cov_hi = None
        if band is not None:
            cov_lo, cov_hi = band
            if is_splice and mask is not None:
                a_src = float((np.asarray(mask.convert('L'), dtype=np.uint8) > 0).mean())
                if a_src > 0.0:
                    # coverage ≈ a_src / crop_area_frac ⇒ crop_area ∈ [a/hi, a/lo].
                    lo_scale = min(1.0, max(0.02, a_src / cov_hi))
                    hi_scale = min(1.0, max(lo_scale + 1e-3, a_src / cov_lo))
                    scale = (lo_scale, hi_scale)
            elif not is_splice:
                # MATCHED ZOOM for reals: sample a crop-area fraction from the
                # splice zoom distribution so the BCE head can't read crop
                # tightness as a fake/real cue. A tight band around the sampled
                # target keeps per-crop variety while matching the marginal.
                f = self._sample_matched_real_scale(cov_lo, cov_hi)
                if f is not None:
                    scale = (max(0.02, f * 0.85), min(1.0, max(0.02 + 1e-3, f * 1.15)))

        # When the band drives acceptance we let random_resized_crop_pair return
        # any masked crop and band-check the realized coverage here; otherwise
        # keep the legacy inner lower-bound so band=None is byte-identical.
        inner_min = 0.0 if band is not None else (self.min_mask_patch_frac if is_splice else 0.0)
        for _ in range(self.crop_max_tries):
            result = random_resized_crop_pair(
                img, mask, self.res, scale, ratio,
                max_tries=1,
                min_mask_patch_frac=inner_min,
            )
            cropped_img  = result.image
            cropped_mask = result.mask
            if cropped_mask is None:                       # real / maskless
                self._record_crop('random', 0.0)
                return cropped_img, None, True
            labels = mask_to_patch_labels(cropped_mask, self.res, threshold=self.gt_patch_threshold)
            frac   = float(labels.float().mean().item())
            if labels.sum().item() == 0:
                continue
            if band is not None and is_splice:
                if not (cov_lo <= frac <= cov_hi):
                    continue
            elif is_splice and frac < self.min_mask_patch_frac:
                continue
            self._record_crop('random', frac)
            return cropped_img, cropped_mask, True

        # Retries exhausted. Mask-centered JITTERED-oracle zoom (center + coverage
        # jitter live inside oracle_mask_crop) guarantees the splice is in frame.
        # A coverage band auto-enables this — a tight band is hard to hit by
        # random position alone.
        want_oracle = (self.splice_crop_mode == 'oracle_fallback') or (band is not None)
        if is_splice and mask is not None and want_oracle:
            oc = oracle_mask_crop(
                img, mask, self.res,
                target_cov_range=(band if band is not None else self.oracle_target_cov),
            )
            if oc.valid:
                self._record_crop('oracle', oc.coverage)
                return oc.image, oc.mask.convert('L'), True
            # Empty/broken mask — DROP, never relabel a splice-free crop as real.
            self._record_crop('dropped', 0.0)
            return None, None, False

        # Legacy fallback: whole-image resize. Image-level this relabels a
        # missed splice as REAL (is_single) — fine only for big-splice data.
        out_img  = resize_only(img, self.res)
        out_mask = resize_only_mask(mask, self.res).convert('L') if mask is not None else None
        self._record_crop('fallback', 0.0)
        return out_img, out_mask, False

    def _apply_eval_aug(
        self,
        img: Image.Image,
        mask: Optional[Image.Image],
        applied_ops: List[Dict[str, Any]],
        meta_extra: Dict[str, Any],
    ) -> Tuple[Image.Image, Optional[torch.Tensor], bool, bool]:
        """Apply deterministic eval corruption requested by eval_aug_mode."""
        spec = self.eval_corruption_spec
        if spec is None:
            return img, None, False, True
        labels = None
        supervised = False
        is_single = True
        if self.eval_corruption_region == 'mask' and mask is not None:
            out, ops = _composite_corruption(img, mask, spec)
            region = 'mask'
            labels = mask_to_patch_labels(mask, self.res, threshold=self.gt_patch_threshold)
            supervised = bool(labels.sum().item() > 0)
            is_single = not supervised
        else:
            aug = apply_corruption(img, spec)
            out, ops = aug.image, _applied_to_dicts(aug.applied)
            region = 'global'
        applied_ops.extend(ops)
        severity = max([float(op.get('severity', 0.0)) for op in ops] or [0.0])
        meta_extra.update({
            'noise_family': spec.family,
            'noise_region': region,
            'noise_severity': severity,
            'whole_image_aug': int(region == 'global'),
            'eval_aug_mode': self.eval_aug_mode,
        })
        return out, labels, supervised, is_single

    def _maybe_whole_image_corrupt(
        self,
        img: Image.Image,
        applied_ops: List[Dict[str, Any]],
        meta_extra: Dict[str, Any],
    ) -> Image.Image:
        """Training-time whole-image corruption for invariance/FP resistance."""
        if (not self.augment) or self.whole_image_corrupt_prob <= 0:
            return img
        if random.random() >= self.whole_image_corrupt_prob:
            return img
        spec = sample_corruption_spec(
            self.degradation_kwargs.get('families', ('jpeg', 'double_jpeg', 'gaussian', 'poisson', 'resize')),
            allow_clean=False,
            **_corruption_sampler_kwargs(self.degradation_kwargs),
        )
        aug = apply_corruption(img, spec)
        ops = _applied_to_dicts(aug.applied)
        applied_ops.extend(ops)
        severity = max([float(op.get('severity', 0.0)) for op in ops] or [0.0])
        meta_extra.update({
            'whole_image_noise_family': spec.family,
            'whole_image_noise_severity': severity,
            'whole_image_aug': 1,
        })
        if meta_extra.get('noise_family', 'none') == 'none':
            meta_extra.update({
                'noise_family': spec.family,
                'noise_region': 'global',
                'noise_severity': severity,
            })
        return aug.image

    def _adjust_degrade_weight_for_whole_aug(
        self,
        current_weight: float,
        meta_extra: Dict[str, Any],
        active: bool,
    ) -> float:
        if not active:
            return float(current_weight)
        severity = float(meta_extra.get('whole_image_noise_severity', 0.0))
        if severity >= self.heavy_whole_aug_severity_thresh:
            return float(self.heavy_aug_degrade_loss_weight)
        return float(current_weight)

    def _maybe_splice_degradation(
        self,
        img: Image.Image,
        mask: Optional[Image.Image],
        splice_labels: torch.Tensor,
        applied_ops: List[Dict[str, Any]],
    ) -> Tuple[Image.Image, torch.Tensor, bool, bool, float, float, Dict[str, Any]]:
        """Optionally add a second degradation target to splice images."""
        dm = _zero_degrade_meta()
        splice_weight = 1.0
        degrade_weight = 1.0
        if (not self.augment) or (not self.use_splice_degradation) or mask is None:
            return img, torch.zeros(self.res.num_patches, dtype=torch.long), False, True, splice_weight, degrade_weight, dm

        if random.random() >= self.splice_degradation_prob:
            dm.update({
                'variant': 'splice_fp_negative',
                'degrade_type': 'none',
                'treatment_side': 'splice_only_negative',
                'dominant_family': 'clean',
                'target_family': 'none',
                'num_regimes': 1,
                'num_treated_regions': 0,
                'target_area_frac': float(splice_labels.float().mean()),
                'dominant_area_frac': 1.0,
                'is_multi_region': 0,
                'noise_family': 'none',
                'noise_region': 'splice_fp_negative',
                'noise_severity': 0.0,
            })
            return (
                img,
                torch.zeros(self.res.num_patches, dtype=torch.long),
                self.noise_head_splice_fp_weight > 0,
                True,
                splice_weight,
                self.noise_head_splice_fp_weight,
                dm,
            )

        if random.random() < self.splice_mask_corrupt_prob:
            spec = sample_corruption_spec(
                self.degradation_kwargs.get('families', ('jpeg', 'double_jpeg', 'gaussian', 'poisson', 'resize')),
                allow_clean=False,
                **_corruption_sampler_kwargs(self.degradation_kwargs),
            )
            out, ops = _composite_corruption(img, mask, spec)
            applied_ops.extend(ops)
            severity = max([float(op.get('severity', 0.0)) for op in ops] or [0.0])
            dm.update({
                'variant': 'splice_mask_corrupt',
                'degrade_type': spec.family,
                'treatment_side': 'splice_mask',
                'dominant_family': 'clean',
                'target_family': spec.family,
                'num_regimes': 2,
                'num_treated_regions': 1,
                'target_area_frac': float(splice_labels.float().mean()),
                'dominant_area_frac': 1.0,
                'is_multi_region': 1,
                'noise_family': spec.family,
                'noise_region': 'splice_mask',
                'noise_severity': severity,
            })
            return out, splice_labels.clone(), True, False, self.splice_mask_loss_weight, degrade_weight, dm

        noise_ex = build_degradation_example(img, self.res, **self.degradation_kwargs)
        applied_ops.extend(_applied_to_dicts(noise_ex.applied))
        dm.update(noise_ex.meta)
        dm.update({
            'noise_family': dm.get('target_family', dm.get('degrade_type', 'unknown')),
            'noise_region': dm.get('treatment_side', 'local_target'),
            'noise_severity': max([float(op.severity) for op in noise_ex.applied] or [0.0]),
        })
        # Reduce weight when the noise blob is local but doesn't intersect the splice
        # mask — we can't be sure whether the splice region looks different under noise,
        # so treat it as an uncertain signal rather than a clean positive.
        if bool(noise_ex.supervised) and not bool(noise_ex.is_single):
            noise_in_splice = bool((noise_ex.labels & splice_labels).sum() > 0)
            if not noise_in_splice:
                degrade_weight = self.splice_mask_loss_weight  # ~0.2 by default
        return noise_ex.image, noise_ex.labels, bool(noise_ex.supervised), bool(noise_ex.is_single), splice_weight, degrade_weight, dm

    # ── sample builders ──────────────────────────────────────────────────────

    def _make_zero_sample(
        self,
        img_t: torch.Tensor,
        item: Dict,
        kind: str,
        *,
        applied_ops: Optional[List[Dict[str, Any]]] = None,
        meta_extra: Optional[Dict[str, Any]] = None,
        invariance_clean: Optional[torch.Tensor] = None,
        invariance_aug: Optional[torch.Tensor] = None,
        invariance_active: bool = False,
    ) -> Dict:
        zero_img = torch.zeros_like(img_t)
        zero_lbl = torch.zeros(self.res.num_patches, dtype=torch.long)
        dm = _zero_degrade_meta()
        extra = dict(meta_extra or {})
        return {
            'img': img_t,
            'splice_labels': zero_lbl,
            'splice_patch_weights': torch.ones(self.res.num_patches, dtype=torch.float32),
            'supervised': torch.tensor(False, dtype=torch.bool),
            'is_single': torch.tensor(True, dtype=torch.bool),
            'splice_loss_weight': torch.tensor(1.0, dtype=torch.float32),
            'degrade_labels': zero_lbl.clone(),
            'degrade_supervised': torch.tensor(False, dtype=torch.bool),
            'is_single_degrade': torch.tensor(True, dtype=torch.bool),
            'degrade_loss_weight': torch.tensor(1.0, dtype=torch.float32),
            'degrade_is_global_negative': torch.tensor(bool(dm.get('is_global_negative', 0)), dtype=torch.bool),
            'degrade_is_multi_region': torch.tensor(False, dtype=torch.bool),
            'degrade_has_large_context': torch.tensor(False, dtype=torch.bool),
            'degrade_num_regimes': torch.tensor(0, dtype=torch.long),
            'degrade_num_treated_regions': torch.tensor(0, dtype=torch.long),
            'degrade_target_area_frac': torch.tensor(0.0, dtype=torch.float32),
            'degrade_dominant_area_frac': torch.tensor(0.0, dtype=torch.float32),
            'invariance_clean': invariance_clean if invariance_clean is not None else zero_img,
            'invariance_aug': invariance_aug if invariance_aug is not None else zero_img.clone(),
            'invariance_active': torch.tensor(bool(invariance_active), dtype=torch.bool),
            'meta': {
                'path': item.get('img', ''),
                'mask_path': item.get('mask') or '',
                'kind': kind,
                'case_id': item.get('case_id', ''),
                'source': item.get('source', ''),
                'applied_ops': applied_ops or [],
                'noise_family': 'none',
                'noise_region': 'none',
                'noise_severity': 0.0,
                'whole_image_aug': 0,
                'eval_aug_mode': self.eval_aug_mode,
                **extra,
                'blob_area_actual': 0.0,
                'splice_supervision_active': 0,
                **{f'degrade_{k}': v for k, v in dm.items()},
            },
        }

    def _build_splice_sample(
        self, item: Dict, img: Image.Image, mask: Optional[Image.Image]
    ) -> Optional[Dict]:
        """imd_splice / casia_splice path."""
        kind = item.get('kind', 'imd_splice')
        cropped_img, cropped_mask, crop_valid = self._crop_item(item, img, mask)
        if cropped_img is None:
            # Oracle fallback could not surface a splice (empty/broken mask).
            # Drop rather than feed a splice-free crop with a fake label.
            return None

        supervision_valid = crop_valid
        crop_mode = self._last_crop_mode
        crop_cov  = float(self._last_crop_cov)
        splice_labels        = torch.zeros(self.res.num_patches, dtype=torch.long)
        splice_patch_weights = torch.ones(self.res.num_patches, dtype=torch.float32)
        if cropped_mask is not None:
            splice_labels, splice_patch_weights = self._patch_labels_and_weights(cropped_mask)
            frac = float(splice_labels.float().mean())
            if frac < self.min_mask_patch_frac:
                supervision_valid = False

        if self.augment:
            aug_res    = apply_light_augmentations(cropped_img, cropped_mask, **self.light_aug_kwargs)
            cropped_img  = aug_res.image
            cropped_mask = aug_res.mask
            applied_ops  = _applied_to_dicts(aug_res.applied)
            # Re-derive labels after flip
            if cropped_mask is not None:
                splice_labels, splice_patch_weights = self._patch_labels_and_weights(cropped_mask)
                frac = float(splice_labels.float().mean())
                if frac < self.min_mask_patch_frac:
                    supervision_valid = False
        else:
            applied_ops = []

        dm = _zero_degrade_meta()
        meta_extra = {
            'noise_family': 'none',
            'noise_region': 'none',
            'noise_severity': 0.0,
            'whole_image_aug': 0,
            'eval_aug_mode': self.eval_aug_mode,
        }
        degrade_labels = torch.zeros(self.res.num_patches, dtype=torch.long)
        degrade_supervised = False
        is_single_degrade = True
        splice_weight = 1.0
        degrade_weight = 1.0
        if self.augment:
            (
                cropped_img,
                degrade_labels,
                degrade_supervised,
                is_single_degrade,
                splice_weight,
                degrade_weight,
                dm,
            ) = self._maybe_splice_degradation(
                cropped_img, cropped_mask, splice_labels, applied_ops
            )
            # Bake the degradation into the image with a light JPEG pass so the
            # backbone can't exploit uncompressed block boundaries as a cue.
            if degrade_supervised and not is_single_degrade:
                _q = random.randint(88, 98)
                _buf = io.BytesIO()
                cropped_img.save(_buf, format='JPEG', quality=_q)
                _buf.seek(0)
                cropped_img = Image.open(_buf).convert('RGB')
            meta_extra.update({
                'noise_family': dm.get('noise_family', dm.get('target_family', 'none')),
                'noise_region': dm.get('noise_region', dm.get('treatment_side', 'none')),
                'noise_severity': float(dm.get('noise_severity', 0.0)),
            })
            cropped_img = self._maybe_whole_image_corrupt(cropped_img, applied_ops, meta_extra)
            degrade_weight = self._adjust_degrade_weight_for_whole_aug(
                degrade_weight, meta_extra, degrade_supervised
            )
        else:
            cropped_img, eval_d_labels, eval_d_active, eval_d_single = self._apply_eval_aug(
                cropped_img, cropped_mask, applied_ops, meta_extra
            )
            if eval_d_labels is not None:
                degrade_labels = eval_d_labels
                degrade_supervised = eval_d_active
                is_single_degrade = eval_d_single

        splice_supervised = bool(kind in ('imd_splice', 'casia_splice') and
                                 (not self.augment or supervision_valid))
        img_t = self.normalize(TF.to_tensor(cropped_img))
        zero_img = torch.zeros_like(img_t)

        return {
            'img': img_t,
            'splice_labels': splice_labels,
            'splice_patch_weights': splice_patch_weights,
            'supervised': torch.tensor(splice_supervised, dtype=torch.bool),
            'is_single': torch.tensor(not splice_supervised, dtype=torch.bool),
            'splice_loss_weight': torch.tensor(float(splice_weight), dtype=torch.float32),
            'degrade_labels': degrade_labels,
            'degrade_supervised': torch.tensor(degrade_supervised, dtype=torch.bool),
            'is_single_degrade': torch.tensor(is_single_degrade, dtype=torch.bool),
            'degrade_loss_weight': torch.tensor(float(degrade_weight), dtype=torch.float32),
            'degrade_is_global_negative': torch.tensor(False, dtype=torch.bool),
            'degrade_is_multi_region': torch.tensor(bool(dm.get('is_multi_region', 0)), dtype=torch.bool),
            'degrade_has_large_context': torch.tensor(bool(dm.get('has_large_context', 0)), dtype=torch.bool),
            'degrade_num_regimes': torch.tensor(int(dm.get('num_regimes', 0)), dtype=torch.long),
            'degrade_num_treated_regions': torch.tensor(int(dm.get('num_treated_regions', 0)), dtype=torch.long),
            'degrade_target_area_frac': torch.tensor(float(dm.get('target_area_frac', 0.0)), dtype=torch.float32),
            'degrade_dominant_area_frac': torch.tensor(float(dm.get('dominant_area_frac', 0.0)), dtype=torch.float32),
            'invariance_clean': zero_img,
            'invariance_aug': zero_img.clone(),
            'invariance_active': torch.tensor(False, dtype=torch.bool),
            'meta': {
                'path': item.get('img', ''),
                'mask_path': item.get('mask') or '',
                'kind': kind,
                'case_id': item.get('case_id', ''),
                'source': item.get('source', ''),
                'applied_ops': applied_ops,
                **meta_extra,
                'blob_area_actual': float(splice_labels.float().mean()),
                'splice_supervision_active': int(splice_supervised),
                'crop_mode': crop_mode,
                'crop_cov': crop_cov,
                'splice_loss_weight': float(splice_weight),
                'degrade_loss_weight': float(degrade_weight),
                **{f'degrade_{k}': v for k, v in dm.items()},
            },
        }

    def _build_real_sample(
        self, item: Dict, img: Image.Image
    ) -> Optional[Dict]:
        """imd_real / indoor_real path — may apply degradation / invariance."""
        kind = item.get('kind', 'indoor_real')
        cropped_img, _, _ = self._crop_item(item, img, None)

        if self.augment:
            aug_res    = apply_light_augmentations(cropped_img, None, **self.light_aug_kwargs)
            cropped_img  = aug_res.image
            applied_ops  = _applied_to_dicts(aug_res.applied)
        else:
            applied_ops = []

        # Degradation head
        img_for_model = cropped_img
        degrade_ex    = None
        dm            = _zero_degrade_meta()
        degrade_supervised = False
        is_single_degrade  = True
        degrade_weight = 1.0
        meta_extra = {
            'noise_family': 'none',
            'noise_region': 'none',
            'noise_severity': 0.0,
            'whole_image_aug': 0,
            'eval_aug_mode': self.eval_aug_mode,
        }

        if (self.augment or not self.augment) and self.use_degradation and kind in ('imd_real', 'indoor_real'):
            degrade_ex = build_degradation_example(
                cropped_img, self.res, **self.degradation_kwargs
            )
            img_for_model      = degrade_ex.image
            dm                 = dict(degrade_ex.meta)
            degrade_supervised = bool(degrade_ex.supervised)
            is_single_degrade  = bool(degrade_ex.is_single)
            applied_ops.extend(
                {'name': op.name, 'params': op.params, 'severity': op.severity}
                for op in degrade_ex.applied
            )
            meta_extra.update({
                'noise_family': dm.get('target_family', dm.get('degrade_type', 'unknown')),
                'noise_region': dm.get('treatment_side', 'local_target'),
                'noise_severity': max([float(op.severity) for op in degrade_ex.applied] or [0.0]),
            })
            # Matched JPEG bake (mirrors the splice path): when a LOCAL degradation
            # blob was composited, recompress so the backbone can't read the
            # uncompressed block boundary as a cue. Without this, only splices
            # carry the bake => a real/fake shortcut.
            if self.augment and degrade_supervised and not is_single_degrade:
                _q = random.randint(88, 98)
                _buf = io.BytesIO()
                img_for_model.save(_buf, format='JPEG', quality=_q)
                _buf.seek(0)
                img_for_model = Image.open(_buf).convert('RGB')

        if self.augment:
            img_for_model = self._maybe_whole_image_corrupt(img_for_model, applied_ops, meta_extra)
            degrade_weight = self._adjust_degrade_weight_for_whole_aug(
                degrade_weight, meta_extra, degrade_supervised
            )
        else:
            img_for_model, eval_d_labels, eval_d_active, eval_d_single = self._apply_eval_aug(
                img_for_model, None, applied_ops, meta_extra
            )
            if eval_d_labels is not None:
                degrade_ex = None
                degrade_supervised = eval_d_active
                is_single_degrade = eval_d_single

        img_t = self.normalize(TF.to_tensor(img_for_model))

        # Invariance pair
        zero_img         = torch.zeros_like(img_t)
        invariance_clean = zero_img
        invariance_aug   = zero_img.clone()
        invariance_active = False
        if self.augment and self.use_invariance and kind in ('imd_real', 'indoor_real'):
            clean_res, aug_res2 = make_invariance_pair(cropped_img)
            invariance_clean  = self.normalize(TF.to_tensor(clean_res.image))
            invariance_aug    = self.normalize(TF.to_tensor(aug_res2.image))
            invariance_active = True

        return {
            'img': img_t,
            'splice_labels': torch.zeros(self.res.num_patches, dtype=torch.long),
            'splice_patch_weights': torch.ones(self.res.num_patches, dtype=torch.float32),
            'supervised': torch.tensor(False, dtype=torch.bool),
            'is_single': torch.tensor(True, dtype=torch.bool),
            'splice_loss_weight': torch.tensor(1.0, dtype=torch.float32),
            'degrade_labels': degrade_ex.labels if degrade_ex else
                              torch.zeros(self.res.num_patches, dtype=torch.long),
            'degrade_supervised': torch.tensor(degrade_supervised, dtype=torch.bool),
            'is_single_degrade': torch.tensor(is_single_degrade, dtype=torch.bool),
            'degrade_loss_weight': torch.tensor(float(degrade_weight), dtype=torch.float32),
            'degrade_is_global_negative': torch.tensor(bool(dm.get('is_global_negative', 0)), dtype=torch.bool),
            'degrade_is_multi_region': torch.tensor(bool(dm.get('is_multi_region', 0)), dtype=torch.bool),
            'degrade_has_large_context': torch.tensor(bool(dm.get('has_large_context', 0)), dtype=torch.bool),
            'degrade_num_regimes': torch.tensor(int(dm.get('num_regimes', 0)), dtype=torch.long),
            'degrade_num_treated_regions': torch.tensor(int(dm.get('num_treated_regions', 0)), dtype=torch.long),
            'degrade_target_area_frac': torch.tensor(float(dm.get('target_area_frac', 0.0)), dtype=torch.float32),
            'degrade_dominant_area_frac': torch.tensor(float(dm.get('dominant_area_frac', 0.0)), dtype=torch.float32),
            'invariance_clean': invariance_clean,
            'invariance_aug': invariance_aug,
            'invariance_active': torch.tensor(invariance_active, dtype=torch.bool),
            'meta': {
                'path': item.get('img', ''),
                'mask_path': '',
                'kind': kind,
                'case_id': item.get('case_id', ''),
                'source': item.get('source', ''),
                'applied_ops': applied_ops,
                **meta_extra,
                'blob_area_actual': 0.0,
                'splice_supervision_active': 0,
                'degrade_loss_weight': float(degrade_weight),
                **{f'degrade_{k}': v for k, v in dm.items()},
            },
        }

    def _build_ae_splice_sample(self, item: Dict) -> Optional[Dict]:
        """ae_splice path — composites AE recon into clean image."""
        size = self.res.image_size

        # Clean negative — no ae_recon path in item
        if not item.get('ae_recon'):
            try:
                img_pil = Image.open(item['img']).convert('RGB').resize(
                    (size, size), Image.BICUBIC
                )
            except Exception as exc:
                raise DataError(
                    f"ae_splice: failed to load clean negative {item['img']!r}: {exc}"
                ) from exc
            applied_ops: List[Dict[str, Any]] = []
            meta_extra = {
                'noise_family': 'none',
                'noise_region': 'none',
                'noise_severity': 0.0,
                'whole_image_aug': 0,
                'eval_aug_mode': self.eval_aug_mode,
            }
            img_for_model = img_pil
            if self.augment:
                img_for_model = self._maybe_whole_image_corrupt(
                    img_for_model, applied_ops, meta_extra
                )
            else:
                img_for_model, _, _, _ = self._apply_eval_aug(
                    img_for_model, None, applied_ops, meta_extra
                )

            img_t = self.normalize(TF.to_tensor(img_for_model))
            invariance_clean = None
            invariance_aug = None
            invariance_active = False
            if self.augment and self.use_invariance:
                clean_res, aug_res = make_invariance_pair(img_pil)
                invariance_clean = self.normalize(TF.to_tensor(clean_res.image))
                invariance_aug = self.normalize(TF.to_tensor(aug_res.image))
                invariance_active = True
            return self._make_zero_sample(
                img_t,
                item,
                item.get('kind', 'ae_splice'),
                applied_ops=applied_ops,
                meta_extra=meta_extra,
                invariance_clean=invariance_clean,
                invariance_aug=invariance_aug,
                invariance_active=invariance_active,
            )

        # AE splice positive
        try:
            orig_pil  = Image.open(item['img']).convert('RGB').resize((size, size), Image.BICUBIC)
            recon_pil = Image.open(item['ae_recon']).convert('RGB').resize((size, size), Image.BICUBIC)
        except Exception as exc:
            raise DataError(
                f"ae_splice: failed to load images for {item['img']!r}: {exc}"
            ) from exc

        # Sample blob mask
        min_frac = self.min_mask_patch_frac
        max_frac = min(0.50, self.blob_params.max_area_frac + 0.05)
        best_mask, best_labels, best_frac = None, None, None
        best_dist = float('inf')
        for attempt in range(12):
            if self.augment:
                blob_seed = random.randint(0, 2**31 - 1)
            else:
                base = _stable_seed(f"{item['img']}|ae|{self.deterministic_seed}")
                blob_seed = base + attempt

            blob_mask_pil = generate_blob_mask_pil(self.res, self.blob_params, blob_seed)
            labels = mask_to_patch_labels(blob_mask_pil, self.res, threshold=self.gt_patch_threshold)
            frac   = float(labels.float().mean())

            if min_frac <= frac <= max_frac:
                best_mask, best_labels, best_frac = blob_mask_pil, labels, frac
                break
            dist = min_frac - frac if frac < min_frac else frac - max_frac
            if dist < best_dist:
                best_mask, best_labels, best_frac = blob_mask_pil, labels, frac
                best_dist = dist

        composited_pil = paste_regional_ae(orig_pil, recon_pil, best_mask)
        ae_supervised  = best_frac >= min_frac   # blob large enough to supervise degrade head
        applied_ops: List[Dict[str, Any]] = []
        dm = _zero_degrade_meta()

        # ── AE items are NEVER splice positives ───────────────────────────────
        # The AE reconstruction blob is local visual inconsistency → degrade head only.
        # The splice head always sees these as clean single-region images (is_single=True).
        zero_lbl       = torch.zeros(self.res.num_patches, dtype=torch.long)
        degrade_labels     = best_labels.clone()   # AE blob → degrade head
        degrade_supervised = bool(ae_supervised)
        is_single_degrade  = not bool(ae_supervised)
        degrade_weight = 1.0
        meta_extra = {
            'noise_family': 'none',
            'noise_region': 'none',
            'noise_severity': 0.0,
            'whole_image_aug': 0,
            'eval_aug_mode': self.eval_aug_mode,
        }
        if self.augment:
            # Optional extra corruption — modifies image appearance only.
            # degrade_labels stays pinned to the AE blob; we do NOT let an
            # independently-placed noise blob overwrite the primary AE target.
            if self.use_splice_degradation and random.random() < self.splice_degradation_prob:
                if random.random() < self.splice_mask_corrupt_prob:
                    # Noise composited inside the AE blob — reinforces the target
                    spec = sample_corruption_spec(
                        self.degradation_kwargs.get(
                            'families',
                            ('jpeg', 'double_jpeg', 'gaussian', 'poisson', 'resize'),
                        ),
                        allow_clean=False,
                        **_corruption_sampler_kwargs(self.degradation_kwargs),
                    )
                    composited_pil, ops = _composite_corruption(composited_pil, best_mask, spec)
                    applied_ops.extend(ops)
                    severity = max([float(op.get('severity', 0.0)) for op in ops] or [0.0])
                    dm.update({
                        'variant': 'ae_mask_corrupt',
                        'degrade_type': spec.family,
                        'treatment_side': 'ae_blob',
                        'dominant_family': 'clean',
                        'target_family': spec.family,
                        'num_regimes': 2,
                        'num_treated_regions': 1,
                        'target_area_frac': best_frac,
                        'dominant_area_frac': 1.0,
                        'is_multi_region': 1,
                        'noise_family': spec.family,
                        'noise_region': 'ae_blob',
                        'noise_severity': severity,
                    })
                    meta_extra.update({
                        'noise_family': spec.family,
                        'noise_region': 'ae_blob',
                        'noise_severity': severity,
                    })
                else:
                    # Independent noise blob — image gets harder, degrade label stays AE blob
                    noise_ex = build_degradation_example(
                        composited_pil, self.res, **self.degradation_kwargs
                    )
                    composited_pil = noise_ex.image   # image only — do NOT take noise_ex.labels
                    applied_ops.extend(_applied_to_dicts(noise_ex.applied))
            composited_pil = self._maybe_whole_image_corrupt(composited_pil, applied_ops, meta_extra)
            degrade_weight = self._adjust_degrade_weight_for_whole_aug(
                degrade_weight, meta_extra, degrade_supervised
            )
        else:
            composited_pil, eval_d_labels, eval_d_active, eval_d_single = self._apply_eval_aug(
                composited_pil, best_mask, applied_ops, meta_extra
            )
            if eval_d_labels is not None:
                degrade_labels     = eval_d_labels
                degrade_supervised = eval_d_active
                is_single_degrade  = eval_d_single

        img_t    = self.normalize(TF.to_tensor(composited_pil))
        zero_img = torch.zeros_like(img_t)

        return {
            'img': img_t,
            # Splice head: AE items are always clean negatives — no GT, no supervision
            'splice_labels': zero_lbl,
            'splice_patch_weights': torch.ones(self.res.num_patches, dtype=torch.float32),
            'supervised': torch.tensor(False, dtype=torch.bool),
            'is_single': torch.tensor(True, dtype=torch.bool),
            'splice_loss_weight': torch.tensor(1.0, dtype=torch.float32),
            'degrade_labels': degrade_labels,
            'degrade_supervised': torch.tensor(degrade_supervised, dtype=torch.bool),
            'is_single_degrade': torch.tensor(is_single_degrade, dtype=torch.bool),
            'degrade_loss_weight': torch.tensor(float(degrade_weight), dtype=torch.float32),
            'degrade_is_global_negative': torch.tensor(bool(dm.get('is_global_negative', 0)), dtype=torch.bool),
            'degrade_is_multi_region': torch.tensor(bool(dm.get('is_multi_region', 0)), dtype=torch.bool),
            'degrade_has_large_context': torch.tensor(bool(dm.get('has_large_context', 0)), dtype=torch.bool),
            'degrade_num_regimes': torch.tensor(int(dm.get('num_regimes', 0)), dtype=torch.long),
            'degrade_num_treated_regions': torch.tensor(int(dm.get('num_treated_regions', 0)), dtype=torch.long),
            'degrade_target_area_frac': torch.tensor(float(dm.get('target_area_frac', 0.0)), dtype=torch.float32),
            'degrade_dominant_area_frac': torch.tensor(float(dm.get('dominant_area_frac', 0.0)), dtype=torch.float32),
            'invariance_clean': zero_img,
            'invariance_aug': zero_img.clone(),
            'invariance_active': torch.tensor(False, dtype=torch.bool),
            'meta': {
                'path': item.get('img', ''),
                'mask_path': '',
                'kind': 'ae_splice',
                'case_id': item.get('case_id', ''),
                'source': item.get('source', 'ae_splice'),
                'ae_vae': item.get('ae_vae', ''),
                'ae_recon': item.get('ae_recon', ''),
                'applied_ops': applied_ops,
                **meta_extra,
                'blob_area_actual': best_frac,
                'splice_supervision_active': 0,   # AE items never supervise the splice head
                'splice_loss_weight': 1.0,
                'degrade_loss_weight': float(degrade_weight),
                **{f'degrade_{k}': v for k, v in dm.items()},
            },
        }

    # ── __getitem__ ──────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        item = self.items[idx]
        kind = item.get('kind', 'indoor_real')

        # Deterministic seed for val
        if not self.augment:
            seed = _stable_seed(f"{item.get('img', str(idx))}|{self.deterministic_seed}")
            py_state = random.getstate()
            np_state = np.random.get_state()
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)

        try:
            if kind == 'ae_splice':
                sample = self._build_ae_splice_sample(item)
            elif kind in ('imd_splice', 'casia_splice'):
                img  = Image.open(item['img']).convert('RGB')
                mask = Image.open(item['mask']).convert('L') if item.get('mask') else None
                # Inpaint sources (SD-inpaint family) run the WHOLE image through
                # the VAE, so the un-edited background carries generator artifacts
                # everywhere. Paste the pristine original over the un-masked region
                # so only the inpainted blob differs — making it behave like a true
                # splice instead of a whole-image fingerprint. Gated on real_path,
                # so this is a no-op for genuine CASIA/IMD splices.
                # When augment is on, a (1 - paste_frac) fraction skips the
                # paste and stays a full-VAE frame (full-AE positive). Eval
                # (augment=False) always pastes for a stable, comparable set.
                real_path = item.get('real_path')
                _do_paste = (not self.augment) or (random.random() < self.paste_frac)
                if real_path and mask is not None and _do_paste:
                    real = Image.open(real_path).convert('RGB')
                    if real.size != img.size:
                        real = real.resize(img.size, Image.BICUBIC)
                    m = mask if mask.size == img.size else mask.resize(img.size, Image.NEAREST)
                    img = Image.composite(img, real, m)
                sample = self._build_splice_sample(item, img, mask)
            elif kind in ('imd_real', 'indoor_real'):
                img    = Image.open(item['img']).convert('RGB')
                sample = self._build_real_sample(item, img)
            else:
                raise DataError(
                    f"LabDataset: unknown item kind {kind!r} for {item.get('img')!r}"
                )

            if sample is None:
                raise DataError(
                    f"LabDataset: sample builder returned None for {item.get('img')!r}"
                )

            # ── Shape contract ────────────────────────────────────────────
            img_t = sample['img']
            C, H, W = img_t.shape
            if C != 3 or H != self.res.image_size or W != self.res.image_size:
                raise DataError(
                    f"LabDataset shape contract violated: expected "
                    f"(3, {self.res.image_size}, {self.res.image_size}), "
                    f"got {tuple(img_t.shape)} for item {item.get('img')!r}"
                )
            return sample

        except DataError:
            raise
        except Exception as exc:
            raise DataError(
                "LabDataset.__getitem__ failed "
                f"source={item.get('source', '')!r} "
                f"kind={item.get('kind', '')!r} "
                f"case_id={item.get('case_id', '')!r} "
                f"img={item.get('img', '')!r} "
                f"mask={item.get('mask', '')!r}: {exc}"
            ) from exc
        finally:
            if not self.augment:
                random.setstate(py_state)
                np.random.set_state(np_state)


# ── Collate ──────────────────────────────────────────────────────────────────

def lab_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Filter None samples and stack tensors.

    Raises:
        DataError: If the entire batch is None (indicates systematic failure).
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        raise DataError(
            "lab_collate_fn: received an empty batch after filtering None items. "
            "Check that your dataset items are loading correctly."
        )
    return {
        'img':                       torch.stack([b['img'] for b in batch]),
        'splice_labels':             torch.stack([b['splice_labels'] for b in batch]),
        'splice_patch_weights':      torch.stack([b['splice_patch_weights'] for b in batch]),
        'supervised':                torch.stack([b['supervised'] for b in batch]),
        'is_single':                 torch.stack([b['is_single'] for b in batch]),
        'splice_loss_weight':        torch.stack([b['splice_loss_weight'] for b in batch]),
        'degrade_labels':            torch.stack([b['degrade_labels'] for b in batch]),
        'degrade_supervised':        torch.stack([b['degrade_supervised'] for b in batch]),
        'is_single_degrade':         torch.stack([b['is_single_degrade'] for b in batch]),
        'degrade_loss_weight':       torch.stack([b['degrade_loss_weight'] for b in batch]),
        'degrade_is_global_negative': torch.stack([b['degrade_is_global_negative'] for b in batch]),
        'degrade_is_multi_region':   torch.stack([b['degrade_is_multi_region'] for b in batch]),
        'degrade_has_large_context': torch.stack([b['degrade_has_large_context'] for b in batch]),
        'degrade_num_regimes':       torch.stack([b['degrade_num_regimes'] for b in batch]),
        'degrade_num_treated_regions': torch.stack([b['degrade_num_treated_regions'] for b in batch]),
        'degrade_target_area_frac':  torch.stack([b['degrade_target_area_frac'] for b in batch]),
        'degrade_dominant_area_frac': torch.stack([b['degrade_dominant_area_frac'] for b in batch]),
        'invariance_clean':          torch.stack([b['invariance_clean'] for b in batch]),
        'invariance_aug':            torch.stack([b['invariance_aug'] for b in batch]),
        'invariance_active':         torch.stack([b['invariance_active'] for b in batch]),
        'meta':                      [b['meta'] for b in batch],
    }
