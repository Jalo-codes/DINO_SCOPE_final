"""lab_utils.eval.sliding_window — multi-scale sliding-window eval for image-level BCE.

For each image, runs the model on:
  - the full image (1.0x), AND
  - all overlapping sub-windows at the requested smaller scales,
each resized to the model's input resolution.

Two cropping modes:
  - **source_image=None** (legacy): crop the already-resized model-input tensor
    and bilinearly upsample back to model-input size. Lossy: at scale 0.5 you
    get a 2× upsample of pixels that were already 2-5× downsampled when the
    image was loaded into the loader. Tighter scales just amplify the loss.
  - **source_image=PIL** (preferred): crop the SOURCE-resolution PIL image at
    its native pixel coordinates, then resize each window to the model input
    size. Tight scales now preserve real forensic detail. A 0.5-scale window
    of a 1024×1024 source = 512 native pixels resized to 448 (slight
    downsample). 0.3-scale = 307 native pixels mildly upsampled.

Aggregates with max-logit (most suspicious window decides). Records both the
full-image logit and the max sliding-window logit so the caller can compare
operating points and FP behavior on real images directly.

The sliding window is mask-blind by design — it simulates real inference where
we don't know where the splice is. Use diagnose_zoom_splice for the mask-guided
diagnostic.
"""

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from lab_utils.logging.text import log_line


_DEFAULT_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_NORMALIZE_STD  = (0.229, 0.224, 0.225)


# ── geometry ─────────────────────────────────────────────────────────────────

def _crop_indices(h: int, w: int, scales: Sequence[float], stride_frac: float):
    """Return list of (top, left, win_h, win_w) for each sub-window.

    Skips scales >= 1.0 (the full image is added separately).
    Always covers the bottom/right edge by appending (h - win_h, w - win_w)
    if the regular grid stops short.
    """
    out = []
    for s in scales:
        if s >= 1.0:
            continue
        wh = max(1, int(h * s))
        ww = max(1, int(w * s))
        sh = max(1, int(wh * stride_frac))
        sw = max(1, int(ww * stride_frac))
        tops  = list(range(0, h - wh + 1, sh))
        lefts = list(range(0, w - ww + 1, sw))
        if not tops or tops[-1] + wh < h:
            tops.append(h - wh)
        if not lefts or lefts[-1] + ww < w:
            lefts.append(w - ww)
        for t in tops:
            for l in lefts:
                out.append((t, l, wh, ww))
    return out


def _crop_resize(img: torch.Tensor, t: int, l: int, h: int, w: int,
                 H: int, W: int) -> torch.Tensor:
    """Crop a normalized (3, H, W) tensor and resize back to (3, H, W).

    LEGACY tensor-mode cropping. Loses detail at tight scales because the
    input tensor is already at model resolution.
    """
    crop = img[:, t:t + h, l:l + w].unsqueeze(0)
    crop = F.interpolate(crop, size=(H, W), mode='bilinear', align_corners=False)
    return crop.squeeze(0)


def _crop_resize_source(
    source_image: Image.Image,
    t_tgt: int, l_tgt: int, wh_tgt: int, ww_tgt: int,
    H_tgt: int, W_tgt: int,
    normalize: transforms.Normalize,
) -> torch.Tensor:
    """Crop from SOURCE-resolution PIL at native pixels, resize to model input.

    Maps target-space (t, l, wh, ww) to source-space proportionally so the
    crop covers the same fraction of the image as the legacy mode would, but
    with native pixel resolution preserved.

    Args:
        source_image: PIL.Image (source resolution; can be any size).
        t_tgt, l_tgt, wh_tgt, ww_tgt: crop region in target-space coords (i.e.,
            in the model-input H_tgt × W_tgt space).
        H_tgt, W_tgt: model input spatial size (e.g., 448, 448).
        normalize: torchvision Normalize transform applied after to_tensor().

    Returns:
        Normalized tensor (3, H_tgt, W_tgt).
    """
    W_src, H_src = source_image.size  # PIL: (W, H)
    ratio_h = H_src / float(H_tgt)
    ratio_w = W_src / float(W_tgt)
    t_src  = int(round(t_tgt  * ratio_h))
    l_src  = int(round(l_tgt  * ratio_w))
    wh_src = max(1, int(round(wh_tgt * ratio_h)))
    ww_src = max(1, int(round(ww_tgt * ratio_w)))
    # Clamp to source bounds.
    t_src = max(0, min(t_src, H_src - wh_src))
    l_src = max(0, min(l_src, W_src - ww_src))
    crop_src = source_image.crop((l_src, t_src, l_src + ww_src, t_src + wh_src))
    crop_resized = crop_src.resize((W_tgt, H_tgt), Image.BILINEAR)
    return normalize(TF.to_tensor(crop_resized))


def _build_normalize(mean, std) -> transforms.Normalize:
    return transforms.Normalize(list(mean), list(std))


def _load_source_pil(path: str) -> Image.Image:
    """Open a source-resolution image as RGB. Caller-managed lifetime."""
    return Image.open(path).convert('RGB')


# ── square scale-ladder geometry ─────────────────────────────────────────────

def _square_crop_boxes(
    H: int, W: int, scales: Sequence[float], stride_frac: float,
) -> List[Tuple[int, int, int]]:
    """Square sub-window boxes as ``(top, left, side)`` in the coord space of
    ``(H, W)``.

    Unlike :func:`_crop_indices` (whose windows inherit the image aspect ratio),
    every window here is SQUARE — ``side = scale * min(H, W)`` — so a window
    resized square→square to the model input never distorts aspect. This
    matches the dataset, which presents square crops to the model. The full
    image (scale >= 1.0) is added separately by the caller and skipped here.
    Both axes are strided with explicit bottom/right edge coverage.
    """
    out: List[Tuple[int, int, int]] = []
    short = min(int(H), int(W))
    for s in scales:
        if s >= 1.0:
            continue
        side = max(1, int(round(short * float(s))))
        side = min(side, short)
        stride = max(1, int(round(side * float(stride_frac))))
        tops  = list(range(0, H - side + 1, stride)) or [0]
        lefts = list(range(0, W - side + 1, stride)) or [0]
        if tops[-1] + side < H:
            tops.append(H - side)
        if lefts[-1] + side < W:
            lefts.append(W - side)
        for t in tops:
            for l in lefts:
                out.append((int(t), int(l), int(side)))
    return out


def _top2_logit(logits: np.ndarray) -> float:
    """Mean of the two highest window logits (multiple-comparison-robust aggregator).

    Falls back to the single logit when only one window is present. Requiring
    two windows to agree raises far fewer false alarms on reals than raw max.
    """
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if arr.size >= 2:
        return float(np.sort(arr)[-2:].mean())
    return float(arr[0]) if arr.size else float('nan')


def _crop_resize_square_source(
    source_image: Image.Image,
    top: int, left: int, side: int,
    H_tgt: int, W_tgt: int,
    normalize: transforms.Normalize,
) -> torch.Tensor:
    """Crop a SQUARE ``side×side`` window from a source-resolution PIL at native
    pixels and resize square→square to the model input ``(H_tgt, W_tgt)``."""
    crop = source_image.crop((left, top, left + side, top + side))
    crop = crop.resize((W_tgt, H_tgt), Image.BILINEAR)
    return normalize(TF.to_tensor(crop))


# ── per-image inference ──────────────────────────────────────────────────────

@torch.no_grad()
def sliding_window_logits(
    model: torch.nn.Module,
    img: torch.Tensor,                       # (3, H, W) normalized
    device: torch.device,
    *,
    scales: Sequence[float] = (1.0, 0.7, 0.5),
    stride_frac: float = 0.5,
    inner_batch_size: int = 8,
    source_image: Optional[Image.Image] = None,
    square: bool = False,
    normalize_mean: Tuple[float, float, float] = _DEFAULT_NORMALIZE_MEAN,
    normalize_std:  Tuple[float, float, float] = _DEFAULT_NORMALIZE_STD,
) -> np.ndarray:
    """Returns numpy array of logits, [0] = full image, rest = sliding-window crops.

    Args:
        img: (3, H, W) normalized model-input tensor (mandatory; defines target
             spatial size and provides the full-image scale=1.0 forward).
        source_image: Optional PIL image at source resolution. When provided,
             sub-windows are cropped from THIS at native pixel coordinates and
             resized to (H, W). Strongly recommended — without it, tight scales
             just upsample already-downsampled content. Pass None to keep the
             legacy tensor-cropping mode (e.g. for the robustness sweep where
             the loader has already corrupted `img` and there is no clean
             source counterpart).
        square: When True, sub-windows are SQUARE (``side = scale * short_edge``)
             instead of inheriting the image aspect ratio, resized square→square
             to (H, W) with no aspect distortion — matching the square crops the
             model trains on. Strongly preferred for the BCE logit path.
    """
    C, H, W = img.shape
    # Full image (scale 1.0) is always the model-input tensor as-is. When we
    # crop sub-windows from a source PIL (CPU), keep the full-image tensor on
    # CPU too so torch.cat doesn't mix devices; the inner loop re-uploads.
    on_cpu = source_image is not None
    crops = [(img.detach().cpu() if on_cpu else img).unsqueeze(0)]
    normalize = _build_normalize(normalize_mean, normalize_std)
    if square:
        if source_image is not None:
            W_src, H_src = source_image.size  # PIL: (W, H)
            for (t, l, side) in _square_crop_boxes(H_src, W_src, scales, stride_frac):
                crops.append(_crop_resize_square_source(
                    source_image, t, l, side, H, W, normalize
                ).unsqueeze(0))
        else:
            # Tensor-mode square fallback (lossy): square boxes in target space.
            for (t, l, side) in _square_crop_boxes(H, W, scales, stride_frac):
                crops.append(_crop_resize(img, t, l, side, side, H, W).unsqueeze(0))
    elif source_image is not None:
        for (t, l, wh, ww) in _crop_indices(H, W, scales, stride_frac):
            crops.append(_crop_resize_source(
                source_image, t, l, wh, ww, H, W, normalize
            ).unsqueeze(0))
    else:
        for (t, l, wh, ww) in _crop_indices(H, W, scales, stride_frac):
            crops.append(_crop_resize(img, t, l, wh, ww, H, W).unsqueeze(0))
    all_crops = torch.cat(crops, dim=0)

    out = []
    for i in range(0, len(all_crops), inner_batch_size):
        batch = all_crops[i:i + inner_batch_size].to(device, non_blocking=True)
        out.append(model(batch).detach().cpu().float().numpy().reshape(-1))
    return np.concatenate(out)


# ── sliding-window localization (per-patch mask aggregation) ────────────────

@torch.no_grad()
def tile_window_contrastive_masks(
    multi_head: torch.nn.Module,
    source_image: Image.Image,
    device: torch.device,
    *,
    n_patch_per_side: int,
    scale: float = 0.7,
    stride_frac: float = 1.0,
    inner_batch_size: int = 8,
    kmeans_init: int = 4,
    bce_gate_threshold: Optional[float] = 0.0,
    max_windows: int = 8,
    normalize_mean: Tuple[float, float, float] = _DEFAULT_NORMALIZE_MEAN,
    normalize_std:  Tuple[float, float, float] = _DEFAULT_NORMALIZE_STD,
    target_image_size: int = 448,
) -> Tuple[np.ndarray, int, int]:
    """Run a small BCE-guided light-zoom localization pass.

    Hard rules:
      - SQUARE windows only — `crop_side = scale * min(H_src, W_src)`. Each
        window is square in source pixels, then resized square→square to the
        model input. No aspect distortion ever.
      - NO full image included. The full-image partition is reported separately
        by the caller; this function returns ONLY the windowed estimate.
      - At most `max_windows` source candidates are evaluated. For the default
        scale=0.7 this is usually 4 square crops, 6 for moderately elongated
        images, capped at 8 for extreme aspect ratios.
      - BCE is a literal boolean crop filter: a window contributes iff its
        image_logit >= `bce_gate_threshold`. The default threshold is 0.0,
        i.e. sigmoid(logit) >= 0.5.
      - Per-window cluster polarity from BCE attention — the cluster with
        higher mean attention is the splice. Smaller-cluster rule is broken
        for splice fractions >50% of patch grid (oracle: 78% inversion on
        CASIA large at full image).
      - OR-aggregate only BCE-positive window partitions. Negative windows are
        ignored entirely, so clean windows do not get to add k-means noise.

    Args:
        scale: Window side as a fraction of `min(H_src, W_src)`. 0.7 ≈ 50% area
               (the median sweet spot from the oracle-crop sweep).
        stride_frac: Stride as a fraction of crop_side. 1.0 = non-overlapping;
                     0.5 = half-overlap (~4× more windows).
        max_windows: Hard cap on candidate crop count.

    Returns:
        agg_mask: (n_patch_per_side, n_patch_per_side) bool. All-False if no
                  window is BCE-positive.
        n_pass: number of BCE-positive windows that contributed to the output.
        n_boundary: number of contributing windows whose predicted positive
                    region touches the crop boundary. High values suggest
                    more overlap/windows may help.
    """
    from lab_utils.eval.partition import spherical_kmeans2

    if source_image is None:
        raise ValueError('tile_window_contrastive_masks: source_image is required '
                         '(no tensor-mode squish fallback).')

    n = int(n_patch_per_side)
    T = int(target_image_size)
    if T % n != 0:
        raise ValueError(f'target_image_size={T} must be divisible by n_patch_per_side={n}')

    W_src, H_src = source_image.size              # PIL is (W, H)
    crop_side = max(n, int(round(min(H_src, W_src) * float(scale))))
    crop_side = min(crop_side, min(H_src, W_src))   # never exceed image

    max_windows = max(1, int(max_windows))

    def _axis_positions(length: int, side: int, stride_frac_: float) -> List[int]:
        max_start = max(0, int(length) - int(side))
        if max_start == 0:
            return [0]
        stride = max(1, int(round(side * float(stride_frac_))))
        pos = list(range(0, max_start + 1, stride))
        if pos[-1] != max_start:
            pos.append(max_start)
        return sorted(set(int(p) for p in pos))

    tops  = _axis_positions(H_src, crop_side, stride_frac)
    lefts = _axis_positions(W_src, crop_side, stride_frac)

    # Keep the candidate set small. When an aspect ratio would produce too many
    # edge-covering positions, keep evenly spaced positions along the longer
    # axis instead of turning this into dense sliding-window eval.
    while len(tops) * len(lefts) > max_windows:
        if len(lefts) >= len(tops) and len(lefts) > 1:
            keep = max(1, max_windows // len(tops))
            idx = np.linspace(0, len(lefts) - 1, num=keep)
            lefts = [lefts[int(round(x))] for x in idx]
        elif len(tops) > 1:
            keep = max(1, max_windows // len(lefts))
            idx = np.linspace(0, len(tops) - 1, num=keep)
            tops = [tops[int(round(x))] for x in idx]
        else:
            break
        tops = sorted(set(tops))
        lefts = sorted(set(lefts))

    crops_meta = [(t, l, crop_side, crop_side) for t in tops for l in lefts]

    # Build square crops, resize square→square to (T, T). No squish.
    normalize = _build_normalize(normalize_mean, normalize_std)
    crops = []
    for (t, l, s, _) in crops_meta:
        win_pil = source_image.crop((l, t, l + s, t + s))   # PIL is (left, top, right, bottom)
        win_pil = TF.resize(win_pil, [T, T], interpolation=Image.BILINEAR)
        win_t   = TF.to_tensor(win_pil)
        win_t   = normalize(win_t)
        crops.append(win_t.unsqueeze(0))
    all_crops = torch.cat(crops, dim=0)

    # Forward all windows (chunked).
    z_chunks, logit_chunks, att_chunks = [], [], []
    for i in range(0, len(all_crops), inner_batch_size):
        batch = all_crops[i:i + inner_batch_size].to(device, non_blocking=True)
        out = multi_head(batch)
        z = out['contrastive']
        if z is None:
            raise RuntimeError('tile_window_contrastive_masks: model has no contrastive head')
        z_chunks.append(z.detach().cpu().float().numpy())
        if out['image_logit'] is not None:
            logit_chunks.append(out['image_logit'].detach().cpu().float().numpy())
        elif bce_gate_threshold is not None:
            raise RuntimeError(
                'tile_window_contrastive_masks: bce_gate_threshold is set but model has no BCE head'
            )
        if out.get('pool_attention') is not None:
            att_chunks.append(out['pool_attention'].detach().cpu().float().numpy())
    z_all = np.concatenate(z_chunks, axis=0)
    logits_all = np.concatenate(logit_chunks, axis=0) if logit_chunks else None
    att_all    = np.concatenate(att_chunks,   axis=0) if att_chunks   else None

    # Per-window mask: kmeans → BCE-attention cluster polarity → (n, n) bool.
    win_masks: List[np.ndarray] = []
    for k in range(z_all.shape[0]):
        raw_labels, _ = spherical_kmeans2(z_all[k], n_init=int(kmeans_init))
        if att_all is not None:
            a = att_all[k]
            m0 = (raw_labels == 0); m1 = (raw_labels == 1)
            a0 = float(a[m0].mean()) if m0.any() else -np.inf
            a1 = float(a[m1].mean()) if m1.any() else -np.inf
            splice_label = 0 if a0 >= a1 else 1
        else:
            n0 = int((raw_labels == 0).sum()); n1 = int((raw_labels == 1).sum())
            splice_label = 0 if n0 <= n1 else 1
        win_masks.append((raw_labels == splice_label).astype(np.bool_).reshape(n, n))

    if bce_gate_threshold is not None and logits_all is None:
        raise RuntimeError(
            'tile_window_contrastive_masks: bce_gate_threshold is set but model has no BCE head'
        )
    logits_flat = logits_all.reshape(-1) if logits_all is not None else None
    if bce_gate_threshold is None:
        selected = list(range(len(crops_meta)))
    else:
        selected = [
            int(k) for k, logit in enumerate(logits_flat)
            if float(logit) >= float(bce_gate_threshold)
        ]

    if not selected:
        return np.zeros((n, n), dtype=np.bool_), 0, 0

    # Project BCE-positive window masks back to TARGET patch space via
    # source→target pixel mapping, OR-aggregating their positive patches.
    agg = np.zeros((n, n), dtype=np.bool_)
    target_patch_size = T // n   # e.g. 16 for T=448, n=28
    n_pass = 0
    n_boundary = 0

    for k in selected:
        top, left, s, _ = crops_meta[k]
        n_pass += 1
        wm = win_masks[k]                                 # (n, n) bool — window patch grid
        if wm[0, :].any() or wm[-1, :].any() or wm[:, 0].any() or wm[:, -1].any():
            n_boundary += 1
        win_patch_size_src = s / float(n)                 # source pixels per window patch

        # Map source coords to the square target image used by the dataset.
        tgt_top_pix    = top  * T / float(H_src)
        tgt_bot_pix    = (top  + s) * T / float(H_src)
        tgt_left_pix   = left * T / float(W_src)
        tgt_right_pix  = (left + s) * T / float(W_src)
        tgt_top_patch    = max(0, int(math.floor(tgt_top_pix    / target_patch_size)))
        tgt_bot_patch    = min(n, int(math.ceil (tgt_bot_pix    / target_patch_size)))
        tgt_left_patch   = max(0, int(math.floor(tgt_left_pix   / target_patch_size)))
        tgt_right_patch  = min(n, int(math.ceil (tgt_right_pix  / target_patch_size)))

        for ti in range(tgt_top_patch, tgt_bot_patch):
            # Map this target patch's center back to source pixel coords.
            src_y_pix = (ti + 0.5) * target_patch_size * H_src / float(T)
            # Find which window patch row covers it.
            wi = int((src_y_pix - top) / win_patch_size_src)
            if wi < 0 or wi >= n:
                continue
            for tj in range(tgt_left_patch, tgt_right_patch):
                src_x_pix = (tj + 0.5) * target_patch_size * W_src / float(T)
                wj = int((src_x_pix - left) / win_patch_size_src)
                if wj < 0 or wj >= n:
                    continue
                if wm[wi, wj]:
                    agg[ti, tj] = True

    return agg.astype(np.bool_), n_pass, n_boundary


def sliding_window_contrastive_masks(
    multi_head: torch.nn.Module,
    img: torch.Tensor,                       # (3, H, W) normalized
    device: torch.device,
    *,
    n_patch_per_side: int,
    scales: Sequence[float] = (1.0, 0.7, 0.5),
    stride_frac: float = 0.5,
    inner_batch_size: int = 8,
    kmeans_init: int = 4,
    bce_gate_threshold: float = None,
    source_image: Optional[Image.Image] = None,
    normalize_mean: Tuple[float, float, float] = _DEFAULT_NORMALIZE_MEAN,
    normalize_std:  Tuple[float, float, float] = _DEFAULT_NORMALIZE_STD,
):
    """OR-aggregate per-window contrastive masks into one image-level mask.

    For each crop:
      1. Encode → contrastive embeddings (1, N, d), and BCE image_logit if
         the model has the BCE head.
      2. Spherical k-means(2) on the patches → smaller-cluster mask (n×n).
      3. **If `bce_gate_threshold` is set and this window's `image_logit`
         falls below it, SKIP this window** (don't add to the OR buffer).
         This is essential to avoid OR-aggregating noise predictions from
         windows that don't contain a splice — k-means will always partition
         clean content into two semantic clusters, which would otherwise
         pollute the final mask.
      4. Upsample window mask to window pixel coords via block-replication.
      5. OR into a (H, W) pixel buffer at the window's offset (target space).
    Final: max-pool the pixel buffer back to the image patch grid (n×n).

    Cropping mode:
      - `source_image=None` (legacy): sub-windows are cropped from `img` and
         bilinearly resized back to (H, W). Lossy at tight scales.
      - `source_image=PIL`: sub-windows are cropped from the source-resolution
         PIL at native pixels and resized to (H, W). Crop INDICES stay in
         target space so the mask aggregation remains aligned with GT.

    Args:
        bce_gate_threshold: If not None, requires the model to expose an
            image-BCE head. Each window must have `image_logit > threshold`
            to contribute to the OR-aggregated mask. Use the cross-source
            calibrated `opt_thresh` (typically calibrated on imd_val).
        source_image: PIL.Image at source resolution. Passing this is strongly
            recommended for forensic detail at tight scales.

    Returns:
        (full_mask, swin_mask) — both (n_patch_per_side, n_patch_per_side) bool.
        full_mask is the partition of the full-image embeddings (un-aggregated).
        If all windows are gated out, swin_mask is all-zeros.
    """
    # Local import to avoid circular dep (eval.partition imports from elsewhere).
    from lab_utils.eval.partition import spherical_kmeans2

    C, H, W = img.shape
    n = int(n_patch_per_side)

    crops_meta = [(0, 0, H, W)]   # full image first — uses input tensor as-is
    for (t, l, wh, ww) in _crop_indices(H, W, scales, stride_frac):
        crops_meta.append((t, l, wh, ww))

    crops = []
    if source_image is not None:
        normalize = _build_normalize(normalize_mean, normalize_std)
        # Full image: use input tensor (already at target res, normalized).
        # Move to CPU so torch.cat below doesn't mix CUDA full-image with
        # CPU source-cropped sub-windows. The inner-batch loop re-uploads.
        crops.append(img.detach().cpu().unsqueeze(0))
        # Sub-windows: crop from source PIL at native pixels.
        for (t, l, wh, ww) in crops_meta[1:]:
            crops.append(_crop_resize_source(
                source_image, t, l, wh, ww, H, W, normalize
            ).unsqueeze(0))
    else:
        for (t, l, wh, ww) in crops_meta:
            crops.append(_crop_resize(img, t, l, wh, ww, H, W).unsqueeze(0))
    all_crops = torch.cat(crops, dim=0)

    # Forward in chunks; collect contrastive embeddings AND per-window logits
    # AND per-window patch attention (when BCE head present).
    z_chunks, logit_chunks, att_chunks = [], [], []
    for i in range(0, len(all_crops), inner_batch_size):
        batch = all_crops[i:i + inner_batch_size].to(device, non_blocking=True)
        out = multi_head(batch)
        z = out['contrastive']
        if z is None:
            raise RuntimeError(
                'sliding_window_contrastive_masks: model has no contrastive head'
            )
        z_chunks.append(z.detach().cpu().float().numpy())
        if out['image_logit'] is not None:
            logit_chunks.append(out['image_logit'].detach().cpu().float().numpy())
        elif bce_gate_threshold is not None:
            raise RuntimeError(
                'sliding_window_contrastive_masks: bce_gate_threshold is set '
                'but model has no image-BCE head'
            )
        if out.get('pool_attention') is not None:
            att_chunks.append(out['pool_attention'].detach().cpu().float().numpy())
    z_all = np.concatenate(z_chunks, axis=0)   # (n_crops, N, d)
    logits_all = (np.concatenate(logit_chunks, axis=0)
                  if logit_chunks else None)   # (n_crops,) or None
    att_all = (np.concatenate(att_chunks, axis=0)
               if att_chunks else None)        # (n_crops, N) or None

    # Per-crop: kmeans → cluster reshaped to (n, n). Cluster polarity comes
    # from BCE attention (if available) — cluster with HIGHER mean attention
    # is the splice. Falls back to smaller-cluster rule when no BCE head.
    win_masks = []
    for k in range(z_all.shape[0]):
        raw_labels, _ = spherical_kmeans2(z_all[k], n_init=int(kmeans_init))
        att_k = att_all[k] if att_all is not None else None
        if att_k is not None:
            m0 = (raw_labels == 0); m1 = (raw_labels == 1)
            a0 = float(att_k[m0].mean()) if m0.any() else -np.inf
            a1 = float(att_k[m1].mean()) if m1.any() else -np.inf
            splice_label = 0 if a0 >= a1 else 1
        else:
            n0 = int((raw_labels == 0).sum()); n1 = int((raw_labels == 1).sum())
            splice_label = 0 if n0 <= n1 else 1
        win_mask = (raw_labels == splice_label).astype(np.bool_).reshape(n, n)
        win_masks.append(win_mask)

    # Full-image (raw, un-aggregated) — first crop. Always returned, never
    # gated, so the caller can compare full vs. swin even when the BCE head
    # would have suppressed the full-image pass.
    full_mask = win_masks[0]

    # OR-aggregate windows whose BCE logit (if gating is enabled) clears the
    # threshold. Without this gate, windows that contain no splice still
    # produce kmeans masks (k-means always partitions); OR-ing them in is
    # destructive — verified empirically to drop F1_med from 0.78 → 0.34 on
    # IMD2020 small splices.
    pix_buf = np.zeros((H, W), dtype=np.bool_)
    n_pass = 0
    for k, (t, l, wh, ww) in enumerate(crops_meta):
        if bce_gate_threshold is not None and logits_all is not None:
            if float(logits_all[k]) <= float(bce_gate_threshold):
                continue   # this window does not look splicey; do not OR in
        n_pass += 1
        wm = win_masks[k]   # (n, n) bool
        # Upsample (n, n) → (wh, ww) via block-replication / nearest-neighbor.
        pix_per_h = wh // n; pix_per_w = ww // n
        if pix_per_h > 0 and pix_per_w > 0 and pix_per_h * n == wh and pix_per_w * n == ww:
            up = np.repeat(np.repeat(wm, pix_per_h, axis=0), pix_per_w, axis=1)
        else:
            # Generic fractional case: per-cell pixel slice via integer math.
            up = np.zeros((wh, ww), dtype=np.bool_)
            for wi in range(n):
                r0 = wi * wh // n; r1 = (wi + 1) * wh // n
                if r0 == r1:
                    continue
                for wj in range(n):
                    if wm[wi, wj]:
                        c0 = wj * ww // n; c1 = (wj + 1) * ww // n
                        up[r0:r1, c0:c1] = True
        pix_buf[t:t + wh, l:l + ww] |= up

    # Downsample (H, W) pixel buffer → (n, n) image patch grid by max-pool.
    ps_h = H // n; ps_w = W // n
    if ps_h * n == H and ps_w * n == W:
        swin_mask = pix_buf.reshape(n, ps_h, n, ps_w).any(axis=(1, 3))
    else:
        # Fallback: per-cell loop (rare; only if H/W not multiples of n).
        swin_mask = np.zeros((n, n), dtype=np.bool_)
        for i in range(n):
            r0 = i * H // n; r1 = (i + 1) * H // n
            for j in range(n):
                c0 = j * W // n; c1 = (j + 1) * W // n
                swin_mask[i, j] = bool(pix_buf[r0:r1, c0:c1].any())

    return full_mask, swin_mask


# ── loader-level eval ────────────────────────────────────────────────────────

def run_sliding_window_eval(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    scales: Sequence[float] = (1.0, 0.7, 0.5),
    stride_frac: float = 0.5,
    inner_batch_size: int = 8,
    real_kinds: Sequence[str] = ('imd_real', 'indoor_real', 'casia_real'),
    log_tag: str = '[swin]',
    tag: str = '',
    log_every: int = 200,
    use_source_resolution: bool = True,
    square: bool = False,
    normalize_mean: Tuple[float, float, float] = _DEFAULT_NORMALIZE_MEAN,
    normalize_std:  Tuple[float, float, float] = _DEFAULT_NORMALIZE_STD,
) -> List[Dict]:
    """Run sliding-window eval over loader. Returns one record per image.

    Each record carries several window aggregators so the caller can compare
    operating points: ``full_logit`` (scale 1.0 only), ``max_logit`` (most
    suspicious single window), ``top2_logit`` (mean of the two highest windows —
    a multiple-comparison-robust aggregator: two windows must agree, so reals
    raise far fewer false alarms than under raw max), and ``mean_logit``.

    Args:
        use_source_resolution: When True (default), sub-windows are cropped
            from each item's source-resolution image (re-loaded from
            meta['path']) at native pixels, then resized to model input.
            Strongly recommended — preserves forensic detail at tight scales.
            Set False when the loader applies image-modifying corruptions
            (e.g. robustness sweep) and there is no clean source counterpart
            on disk that matches what `imgs` contains.
        square: Use square sub-windows (no aspect distortion; matches training
            crops). Passed through to :func:`sliding_window_logits`.
    """
    model.eval()
    records: List[Dict] = []
    n = 0
    real_kinds_set = frozenset(real_kinds)

    for batch in loader:
        if batch is None:
            continue
        imgs = batch['img']
        meta_list = batch['meta'] if isinstance(batch['meta'], list) else [
            {k: v[i] for k, v in batch['meta'].items()} for i in range(imgs.shape[0])
        ]
        for i in range(imgs.shape[0]):
            source_pil = None
            if use_source_resolution:
                src_path = str(meta_list[i].get('path', '') or '')
                if src_path:
                    try:
                        source_pil = _load_source_pil(src_path)
                    except Exception as exc:
                        # Don't crash the whole eval over one bad path; fall
                        # back to legacy mode for this item and warn loudly.
                        log_line(f'{log_tag}{(" "+tag) if tag else ""} '
                                 f'WARN source load failed for {src_path!r}: {exc}; '
                                 f'falling back to tensor-mode swin')
                        source_pil = None
            logits = sliding_window_logits(
                model, imgs[i], device,
                scales=scales, stride_frac=stride_frac,
                inner_batch_size=inner_batch_size,
                source_image=source_pil,
                square=square,
                normalize_mean=normalize_mean,
                normalize_std=normalize_std,
            )
            kind = str(meta_list[i].get('kind', ''))
            top2 = _top2_logit(logits)
            records.append({
                'kind':       kind,
                'is_real':    kind in real_kinds_set,
                'area':       float(meta_list[i].get('blob_area_actual', 0.0)),
                'full_logit': float(logits[0]),
                'max_logit':  float(logits.max()),
                'top2_logit': top2,
                'mean_logit': float(logits.mean()),
                'n_windows':  int(len(logits)),
            })
            n += 1
            if n % log_every == 0:
                log_line(f'{log_tag}{(" "+tag) if tag else ""} processed {n} images')
    log_line(f'{log_tag}{(" "+tag) if tag else ""} done n={n} '
             f'n_windows={records[0]["n_windows"] if records else 0}')
    return records


# ── reporting ────────────────────────────────────────────────────────────────

# Area-fraction tiers for splice-size stratified reporting. Edges align to the
# pinned area-distribution analysis (tiny ≤0.05) and the production small/medium
# (0.15) and medium/large (0.30) boundaries. 'tiny'+'small' are the OOD regime
# that dies at full image and that the sliding window is meant to rescue.
_AREA_TIERS: Tuple[str, ...] = ('tiny', 'small', 'medium', 'large')


def _area_bucket(a: float) -> str:
    if a <= 0.05:
        return 'tiny'
    if a < 0.15:
        return 'small'
    if a < 0.30:
        return 'medium'
    return 'large'


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(-scores)
    sl = labels[order]
    tpr = np.cumsum(sl) / n_pos
    fpr = np.cumsum(1 - sl) / n_neg
    auc = float(np.trapezoid(tpr, fpr))
    return 1.0 + auc if auc < 0 else auc


def _opt_threshold(scores: np.ndarray, labels: np.ndarray):
    """Threshold maximising balanced accuracy."""
    best_t, best_b = float(scores[0] if len(scores) else 0.0), 0.5
    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return best_t, best_b, 0.0, 1.0
    for t in np.unique(scores):
        p = (scores >= t).astype(np.int32)
        tpr = float(((p == 1) & (labels == 1)).sum()) / n_pos
        tnr = float(((p == 0) & (labels == 0)).sum()) / n_neg
        b = 0.5 * (tpr + tnr)
        if b > best_b:
            best_b, best_t = b, float(t)
    p = (scores >= best_t).astype(np.int32)
    tpr = float(((p == 1) & (labels == 1)).sum()) / n_pos
    tnr = float(((p == 0) & (labels == 0)).sum()) / n_neg
    return best_t, best_b, tpr, tnr


def _threshold_at_tnr(scores: np.ndarray, labels: np.ndarray, target_tnr: float):
    """Lowest threshold where tnr >= target_tnr. Returns (threshold, tpr, tnr)."""
    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan'), float('nan'), float('nan')
    real_scores = scores[labels == 0]
    if len(real_scores) == 0:
        return float('nan'), float('nan'), float('nan')
    # Threshold = (1 - target_tnr) percentile of real scores from the top
    t = float(np.quantile(real_scores, target_tnr))
    p = (scores >= t).astype(np.int32)
    tpr = float(((p == 1) & (labels == 1)).sum()) / n_pos
    tnr = float(((p == 0) & (labels == 0)).sum()) / n_neg
    return t, tpr, tnr


# ── public calibrate-then-apply helpers (cross-split deployment) ──────────────

def _scores_labels(records: List[Dict], score_key: str) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.array([r[score_key] for r in records], dtype=np.float64)
    labels = np.array([0 if r['is_real'] else 1 for r in records], dtype=np.int32)
    return scores, labels


def calibrate_threshold_at_tnr(
    records: List[Dict], score_key: str, target_tnr: float,
) -> float:
    """Decision threshold whose TNR on the REAL records equals ``target_tnr``.

    This is the principled fix for the multiple-comparison FP inflation of
    window aggregation: calibrate ONE threshold on the aggregated statistic
    (e.g. ``max_logit``) over held-out reals, then apply it everywhere — never
    each split's own test-set optimum. Returns ``+inf`` if there are no reals.
    """
    scores, labels = _scores_labels(records, score_key)
    real = scores[labels == 0]
    if real.size == 0:
        return float('inf')
    return float(np.quantile(real, float(target_tnr)))


def metrics_at_threshold(
    records: List[Dict], score_key: str, threshold: float,
) -> Dict:
    """Apply a FIXED threshold and report overall TPR/TNR + per-area-tier TPR.

    Positive decision is ``score >= threshold``. Per-tier TPR is computed over
    splices only (reals carry area 0 and never enter a tier count). Use the
    threshold from :func:`calibrate_threshold_at_tnr` on a held-out split.
    """
    scores, labels = _scores_labels(records, score_key)
    areas = np.array([r['area'] for r in records], dtype=np.float64)
    pred = (scores >= float(threshold)).astype(np.int32)
    pos = labels == 1
    neg = labels == 0
    out: Dict = {
        'threshold': float(threshold),
        'tpr': float(pred[pos].mean()) if pos.any() else float('nan'),
        'tnr': float((1 - pred[neg]).mean()) if neg.any() else float('nan'),
        'n_splice': int(pos.sum()),
        'n_real': int(neg.sum()),
        'tiers': {},
    }
    for tier in _AREA_TIERS:
        m = pos & np.array([_area_bucket(a) == tier for a in areas])
        out['tiers'][tier] = {
            'n':   int(m.sum()),
            'tpr': float(pred[m].mean()) if m.any() else float('nan'),
        }
    return out


def format_sliding_window_report(
    records: List[Dict],
    *,
    log_tag: str = '[swin]',
    tag: str = '',
    fixed_tnr_targets: Sequence[float] = (0.95, 0.99),
):
    """Print full-image vs sliding-window comparison with FP analysis."""
    if not records:
        log_line(f'{log_tag} no records')
        return
    suffix = f' {tag}' if tag else ''

    full   = np.array([r['full_logit'] for r in records], dtype=np.float64)
    swin   = np.array([r['max_logit']  for r in records], dtype=np.float64)
    top2   = np.array([r.get('top2_logit', r['max_logit']) for r in records], dtype=np.float64)
    labels = np.array([0 if r['is_real'] else 1 for r in records], dtype=np.int32)
    areas  = np.array([r['area'] for r in records], dtype=np.float64)

    n_total  = len(records)
    n_splice = int(labels.sum())
    n_real   = n_total - n_splice
    n_win    = records[0]['n_windows']
    log_line(
        f'{log_tag}{suffix} n_total={n_total} n_splice={n_splice} '
        f'n_real={n_real} n_windows={n_win}'
    )

    # ── AUC + balanced-acc operating point
    full_auc = _auc(full, labels)
    swin_auc = _auc(swin, labels)
    top2_auc = _auc(top2, labels)
    f_t, f_b, f_tpr, f_tnr = _opt_threshold(full, labels)
    s_t, s_b, s_tpr, s_tnr = _opt_threshold(swin, labels)
    t_t, t_b, t_tpr, t_tnr = _opt_threshold(top2, labels)
    log_line(
        f'{log_tag}{suffix} FULL  auc={full_auc:.4f} '
        f'opt={f_t:.3f} bacc={f_b:.4f} tpr={f_tpr:.4f} tnr={f_tnr:.4f}'
    )
    log_line(
        f'{log_tag}{suffix} SWIN  auc={swin_auc:.4f} '
        f'opt={s_t:.3f} bacc={s_b:.4f} tpr={s_tpr:.4f} tnr={s_tnr:.4f}  '
        f'Δauc={swin_auc - full_auc:+.4f} Δbacc={s_b - f_b:+.4f}  (max-agg)'
    )
    log_line(
        f'{log_tag}{suffix} SWIN2 auc={top2_auc:.4f} '
        f'opt={t_t:.3f} bacc={t_b:.4f} tpr={t_tpr:.4f} tnr={t_tnr:.4f}  '
        f'Δauc={top2_auc - full_auc:+.4f} Δbacc={t_b - f_b:+.4f}  (top2-agg)'
    )

    # ── precision-controlled comparison: same FPR, who has higher TPR?
    for tnr_target in fixed_tnr_targets:
        ft, ftp, ftn = _threshold_at_tnr(full, labels, tnr_target)
        st, stp, stn = _threshold_at_tnr(swin, labels, tnr_target)
        tt, ttp, ttn = _threshold_at_tnr(top2, labels, tnr_target)
        log_line(
            f'{log_tag}{suffix}   @ tnr~={tnr_target:.2f}: '
            f'FULL tpr={ftp:.4f}  |  '
            f'SWIN tpr={stp:.4f} (Δ{stp - ftp:+.4f})  |  '
            f'SWIN2 tpr={ttp:.4f} (Δ{ttp - ftp:+.4f})'
        )

    # ── per-area-tier TPR (each method at its own optimal threshold)
    for bn in _AREA_TIERS:
        m = (labels == 1) & np.array([_area_bucket(a) == bn for a in areas])
        if m.sum() == 0:
            continue
        ftp_b = float((full[m] >= f_t).mean())
        stp_b = float((swin[m] >= s_t).mean())
        ttp_b = float((top2[m] >= t_t).mean())
        log_line(
            f'{log_tag}{suffix}   tier={bn} n={int(m.sum())} '
            f'area_med={float(np.median(areas[m])):.3f} '
            f'full_tpr={ftp_b:.4f} swin_tpr={stp_b:.4f} top2_tpr={ttp_b:.4f} '
            f'lift={stp_b - ftp_b:+.4f}'
        )

    # ── FP risk: real-image max-logit distribution (the smoking gun for swin)
    real_max   = swin[labels == 0]
    splice_max = swin[labels == 1]
    if len(real_max) and len(splice_max):
        log_line(
            f'{log_tag}{suffix} FP_RISK (swin max-logit on reals): '
            f'med={float(np.median(real_max)):+.3f} '
            f'p95={float(np.percentile(real_max, 95)):+.3f} '
            f'p99={float(np.percentile(real_max, 99)):+.3f}  |  '
            f'on splices: med={float(np.median(splice_max)):+.3f} '
            f'p5={float(np.percentile(splice_max, 5)):+.3f}'
        )
        # How much did each population shift from full → swin?
        real_full_med   = float(np.median(full[labels == 0]))
        splice_full_med = float(np.median(full[labels == 1]))
        real_shift   = float(np.median(real_max)) - real_full_med
        splice_shift = float(np.median(splice_max)) - splice_full_med
        log_line(
            f'{log_tag}{suffix} SHIFT swin-max vs full median: '
            f'reals={real_shift:+.3f} splices={splice_shift:+.3f}  '
            f'(want splice_shift > real_shift; '
            f'separation Δ={splice_shift - real_shift:+.3f})'
        )
