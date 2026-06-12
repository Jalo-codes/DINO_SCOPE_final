"""lab_utils.model.losses.contrastive — pairwise hinge contrastive loss.

Originally lifted from contrastive_test/core/losses.py + harness_losses.py.
The attract term is now margin-based (max(0, attract_margin - sim)) rather
than the legacy (1 - sim) point-attract; the legacy behavior is recovered
exactly with attract_margin=1.0.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# ── core pairwise loss ────────────────────────────────────────────────────────

def pairwise_contrastive_loss(
    z: torch.Tensor,                 # (B, N, D) L2-normalized
    labels: torch.Tensor,            # (B, N) int {0, 1}
    is_single_class: torch.Tensor,   # (B,) bool
    sample_weights: torch.Tensor = None,
    patch_weights: torch.Tensor = None,  # (B, N) float in [0,1]; None = all 1
    neg_margin: float = 0.3,
    lambda_repel: float = 1.0,
    single_class_weight: float = 0.0,
    attract_margin: float = 1.0,
    single_class_attract_margin: float = None,
    single_class_attract_squared: bool = False,
    single_class_topk: int = 0,
    temperature: float = 0.10,       # accepted but unused (back-compat)
) -> Tuple[torch.Tensor, Dict]:
    """Within-image pairwise hinge contrastive loss with margin-attract.

    For each image:
        L_attract = mean over same-label pairs of max(0, attract_margin - sim)
        L_repel   = mean over diff-label pairs of max(0, sim - neg_margin)
        L_image   = L_attract + lambda_repel * L_repel

    With attract_margin=1.0 the attract term reduces to (1 - sim) since cosine
    similarity is bounded above by 1 — i.e. point-attract, the legacy behavior.
    Smaller attract_margin opens a dead zone above the margin where same-label
    pairs receive zero gradient, preventing within-image representational
    collapse on single-class images.

    For splice images, the attract term is region-balanced: the label-0 and
    label-1 within-region attract means are averaged equally so a large clean
    region does not drown out a small splice region.

    Single-class images are down-weighted by single_class_weight.
    When single_class_topk > 0, single-class images use the mean of the top-k
    active violating same-label pairs instead of the full-image mean.

    Returns:
        (scalar loss, diagnostics_dict)
    """
    B, N, D = z.shape
    device   = z.device
    dtype    = z.dtype
    is_single_class = is_single_class.to(device=device).bool()
    if single_class_attract_margin is None:
        single_class_attract_margin = attract_margin
    single_class_attract_margin = float(single_class_attract_margin)

    sim = torch.bmm(z, z.transpose(1, 2))                          # (B, N, N)
    eye  = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    same = (labels.unsqueeze(2) == labels.unsqueeze(1)) & ~eye     # (B, N, N)
    diff = (labels.unsqueeze(2) != labels.unsqueeze(1)) & ~eye     # (B, N, N)

    same_f = same.to(dtype)
    diff_f = diff.to(dtype)

    # Per-patch weights: multiply each pair (i,j) by w_i * w_j. A zero-weight
    # patch contributes 0 to both attract and repel for that image. Weights of
    # all 1.0 reduce exactly to the legacy hard-mask behavior.
    if patch_weights is not None:
        pw = patch_weights.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)
        pair_w = pw.unsqueeze(2) * pw.unsqueeze(1)                  # (B, N, N)
        same_f = same_f * pair_w
        diff_f = diff_f * pair_w

    same_count = same_f.sum(dim=(1, 2)).clamp(min=1.0)
    diff_count = diff_f.sum(dim=(1, 2)).clamp(min=1.0)

    # Splice images get point-attract (margin=1.0); single-class reals get the
    # configurable floor so they don't collapse to a single point.
    per_img_margin = torch.where(
        is_single_class,
        torch.full((B,), single_class_attract_margin, dtype=dtype, device=device),
        torch.ones(B, dtype=dtype, device=device),
    )                                                                   # (B,)
    attract_hinge   = torch.clamp(per_img_margin[:, None, None] - sim, min=0.0)
    if bool(single_class_attract_squared):
        attract_hinge = torch.where(
            is_single_class[:, None, None],
            attract_hinge.pow(2),
            attract_hinge,
        )
    attract_per_img = (attract_hinge * same_f).sum(dim=(1, 2)) / same_count
    splice_idx = torch.nonzero(~is_single_class, as_tuple=True)[0].tolist()
    for idx in splice_idx:
        region_terms = []
        # Per-patch weight slice for this image (or all-ones).
        if patch_weights is not None:
            pw_i = patch_weights[idx].to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)
        else:
            pw_i = torch.ones(N, device=device, dtype=dtype)
        for label_val in (0, 1):
            region = labels[idx] == label_val
            if int(region.sum().item()) < 2:
                continue
            region_same = region[:, None] & region[None, :] & (~eye[0])
            region_same_f = region_same.to(dtype) * (pw_i.unsqueeze(1) * pw_i.unsqueeze(0))
            region_count = region_same_f.sum().clamp(min=1.0)
            # Skip a region whose entire weight collapsed to zero (e.g., all
            # ramped-out edge patches on a tiny splice).
            if float(region_same_f.sum().item()) <= 0.0:
                continue
            region_loss = (attract_hinge[idx] * region_same_f).sum() / region_count
            region_terms.append(region_loss)
        if region_terms:
            attract_per_img[idx] = torch.stack(region_terms).mean()
    single_class_topk = max(0, int(single_class_topk))
    if single_class_topk > 0:
        for idx in torch.nonzero(is_single_class, as_tuple=True)[0].tolist():
            active_vals = attract_hinge[idx][same[idx]]
            active_vals = active_vals[active_vals > 0]
            if active_vals.numel() == 0:
                attract_per_img[idx] = 0.0
                continue
            k = min(single_class_topk, int(active_vals.numel()))
            attract_per_img[idx] = active_vals.topk(k).values.mean()
    repel_per_img   = (torch.clamp(sim - neg_margin, min=0.0) * diff_f).sum(dim=(1, 2)) / diff_count

    diff_present   = (diff_f.sum(dim=(1, 2)) > 0).to(dtype)
    repel_per_img  = repel_per_img * diff_present
    per_img_loss   = attract_per_img + lambda_repel * repel_per_img

    weights = torch.where(
        is_single_class,
        torch.full_like(per_img_loss, single_class_weight),
        torch.ones_like(per_img_loss),
    )
    if sample_weights is not None:
        weights = weights * sample_weights.to(device=device, dtype=dtype)
    loss = (per_img_loss * weights).sum() / weights.sum().clamp(min=1.0)

    same_count_t = same_f.sum().clamp(min=1.0)
    diff_count_t = diff_f.sum().clamp(min=1.0)
    sim_pos_mean = (sim * same_f).sum() / same_count_t
    sim_neg_mean = (sim * diff_f).sum() / diff_count_t
    sim_pos_per  = (sim * same_f).sum(dim=(1, 2)) / same_count
    sim_neg_per  = (sim * diff_f).sum(dim=(1, 2)) / diff_count
    has_signal   = (diff_f.sum(dim=(1, 2)) > 0) & (~is_single_class)
    active_pairs = (attract_hinge > 0).to(dtype) * same_f          # (B, N, N)
    attract_active_frac         = active_pairs.sum() / same_count_t
    single_mask = is_single_class.to(dtype)[:, None, None]
    attract_active_frac_single  = (active_pairs * single_mask).sum() / (same_f * single_mask).sum().clamp(min=1.0)
    attract_active_frac_splice  = (active_pairs * (1 - single_mask)).sum() / (same_f * (1 - single_mask)).sum().clamp(min=1.0)

    # Per-patch-weight diagnostics (informative only when patch_weights given).
    if patch_weights is not None:
        pw_diag = patch_weights.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)
        patch_weight_mean    = float(pw_diag.mean().item())
        ignore_band_frac     = float((pw_diag == 0).to(dtype).mean().item())
        soft_pos_band_frac   = float(((pw_diag > 0) & (pw_diag < 1)).to(dtype).mean().item())
    else:
        patch_weight_mean    = 1.0
        ignore_band_frac     = 0.0
        soft_pos_band_frac   = 0.0

    diag = {
        'loss': loss.item(),
        'sim_pos_mean': sim_pos_mean.item(),
        'sim_neg_mean': sim_neg_mean.item(),
        'sim_gap':      (sim_pos_mean - sim_neg_mean).item(),
        'attract_mean': attract_per_img.mean().item(),
        'attract_active_frac': attract_active_frac.item(),
        'attract_active_frac_single': attract_active_frac_single.item(),
        'attract_active_frac_splice': attract_active_frac_splice.item(),
        'repel_mean':   repel_per_img.mean().item(),
        'frac_single_class':   is_single_class.to(dtype).mean().item(),
        'frac_image_has_signal': has_signal.to(dtype).mean().item(),
        'neg_margin':    float(neg_margin),
        'attract_margin': float(attract_margin),
        'single_class_attract_margin': float(single_class_attract_margin),
        'single_class_attract_squared': bool(single_class_attract_squared),
        'splice_region_balanced_attract': True,
        'single_class_topk': int(single_class_topk),
        'lambda_repel':  float(lambda_repel),
        'patch_weight_mean':  patch_weight_mean,
        'ignore_band_frac':   ignore_band_frac,
        'soft_pos_band_frac': soft_pos_band_frac,
        'per_image': {
            'loss':            per_img_loss.detach().cpu(),
            'sim_pos':         sim_pos_per.detach().cpu(),
            'sim_neg':         sim_neg_per.detach().cpu(),
            'has_signal':      has_signal.detach().cpu(),
            'is_single_class': is_single_class.detach().cpu(),
            'weight':          weights.detach().cpu(),
        },
    }
    return loss, diag


# ── selective wrappers ────────────────────────────────────────────────────────

def _empty_diag(device: torch.device) -> Dict:
    return {
        'loss': 0.0, 'sim_pos_mean': 0.0, 'sim_neg_mean': 0.0,
        'sim_gap': 0.0, 'attract_mean': 0.0,
        'attract_active_frac': 0.0, 'attract_active_frac_single': 0.0, 'attract_active_frac_splice': 0.0,
        'repel_mean': 0.0,
        'frac_single_class': 0.0, 'frac_image_has_signal': 0.0,
        'neg_margin': 0.0, 'attract_margin': 0.0,
        'single_class_attract_margin': 0.0, 'single_class_attract_squared': False,
        'splice_region_balanced_attract': True,
        'single_class_topk': 0, 'lambda_repel': 0.0,
        'patch_weight_mean': 1.0, 'ignore_band_frac': 0.0, 'soft_pos_band_frac': 0.0,
        'per_image': {
            'loss':            torch.empty(0, device=device).cpu(),
            'sim_pos':         torch.empty(0, device=device).cpu(),
            'sim_neg':         torch.empty(0, device=device).cpu(),
            'has_signal':      torch.empty(0, device=device, dtype=torch.bool).cpu(),
            'is_single_class': torch.empty(0, device=device, dtype=torch.bool).cpu(),
            'weight':          torch.empty(0, device=device).cpu(),
        },
    }


def selective_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    is_single_class: torch.Tensor,
    active_mask: torch.Tensor,
    neg_margin: float,
    lambda_repel: float,
    single_class_weight: float,
    sample_weights: torch.Tensor = None,
    patch_weights: torch.Tensor = None,
    attract_margin: float = 1.0,
    single_class_attract_margin: float = None,
    single_class_attract_squared: bool = False,
    single_class_topk: int = 0,
) -> Tuple[torch.Tensor, Dict]:
    """Apply pairwise_contrastive_loss only to active (supervised) samples."""
    active = active_mask.bool()
    if int(active.sum().item()) == 0:
        return z.sum() * 0.0, _empty_diag(z.device)
    weights_active = sample_weights[active] if sample_weights is not None else None
    pweights_active = patch_weights[active] if patch_weights is not None else None
    return pairwise_contrastive_loss(
        z[active], labels[active], is_single_class[active],
        sample_weights=weights_active,
        patch_weights=pweights_active,
        neg_margin=neg_margin,
        lambda_repel=lambda_repel,
        single_class_weight=single_class_weight,
        attract_margin=attract_margin,
        single_class_attract_margin=single_class_attract_margin,
        single_class_attract_squared=single_class_attract_squared,
        single_class_topk=single_class_topk,
    )


# ── symmetric pairwise loss (mirrored dead-point hinges) ──────────────────────

def symmetric_pairwise_contrastive_loss(
    z: torch.Tensor,                 # (B, N, D) L2-normalized
    labels: torch.Tensor,            # (B, N) int {0, 1}
    is_single_class: torch.Tensor,   # (B,) bool
    sample_weights: torch.Tensor = None,
    patch_weights: torch.Tensor = None,  # (B, N) float in [0,1]; None = all 1
    tau_pos: float = 0.55,
    tau_neg: float = 0.20,
    lambda_repel: float = 1.0,
    single_class_weight: float = 0.05,
    area_balance_power: float = 0.5,
    norm_power: float = 1.0,
    diversity_weight: float = 0.0,
    diversity_tau: float = 0.90,
    topk: int = 0,
) -> Tuple[torch.Tensor, Dict]:
    """Symmetric within-image pairwise contrastive loss with mirrored dead-point
    hinges.

    Target geometry (per image): every same-label pair should be at least
    ``tau_pos`` similar, every different-label pair at most ``tau_neg`` similar.
    Both sides are hinges with a hard dead zone (zero gradient once satisfied)
    and NO upper bound on cohesion — naturally-similar patches are never pushed
    apart::

        v_pos_ij = max(0, tau_pos - sim_ij)   for same-label pairs
        v_neg_ij = max(0, sim_ij - tau_neg)   for diff-label pairs

    Three region-balanced terms, each a mean over its *active* (violating) pairs
    so satisfied pairs do not dilute the signal::

        A_in  = mean v_pos over (both label 1)      # within-splice cohesion
        A_out = mean v_pos over (both label 0)      # within-clean cohesion
        R     = mean v_neg over (cross) pairs       # separation

    Cohesion is balanced by a tempered area weight (sqrt by default,
    ``area_balance_power=0.5``) so a tiny splice region is neither drowned (raw
    proportion, power=1) nor over-amplified (full balance, power=0)::

        p = spliced-patch fraction (patch-weight aware)
        w_in  = p**power / (p**power + (1-p)**power)
        w_out = (1-p)**power / (...)
        L = w_in * A_in + w_out * A_out + lambda_repel * R

    Single-class images (``is_single_class``) carry only one region, so R and the
    absent cohesion term vanish on their own; they are down-weighted hard by
    ``single_class_weight`` (kept very low — reals are localization-irrelevant,
    the BCE head owns detection).

    ``patch_weights`` (boundary soft/ignore band) multiply each pair (i, j) by
    w_i * w_j, exactly as in :func:`pairwise_contrastive_loss`.

    Deferred knobs (default OFF, gated on measurement): ``diversity_weight`` adds
    a per-region anti-collapse penalty ``max(0, mean_within_sim - diversity_tau)``;
    ``topk`` (>0) is reserved for per-term hard-offender mining and currently
    raises ``NotImplementedError``.

    Returns:
        (scalar loss, diagnostics_dict)
    """
    if int(topk) > 0:
        raise NotImplementedError(
            'symmetric_pairwise_contrastive_loss: per-term top-k offender mining '
            'is a deferred hook (topk must be 0 for now).'
        )
    B, N, D = z.shape
    device, dtype = z.device, z.dtype
    is_single_class = is_single_class.to(device=device).bool()
    power = float(area_balance_power)
    np_ = float(norm_power)

    sim  = torch.bmm(z, z.transpose(1, 2))                          # (B, N, N)
    eye  = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    same = (labels.unsqueeze(2) == labels.unsqueeze(1)) & ~eye
    diff = (labels.unsqueeze(2) != labels.unsqueeze(1)) & ~eye
    pos1 = (labels == 1)                                            # (B, N)
    same_in  = same & pos1.unsqueeze(2) & pos1.unsqueeze(1)         # both spliced
    same_out = same & (~pos1).unsqueeze(2) & (~pos1).unsqueeze(1)   # both clean

    # Per-pair weight w_i * w_j (boundary band). All-ones reduces to hard masks.
    if patch_weights is not None:
        pw = patch_weights.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        pair_w = pw.unsqueeze(2) * pw.unsqueeze(1)                  # (B, N, N)
    else:
        pw = torch.ones(B, N, device=device, dtype=dtype)
        pair_w = torch.ones(B, N, N, device=device, dtype=dtype)

    v_pos = torch.clamp(tau_pos - sim, min=0.0)
    v_neg = torch.clamp(sim - tau_neg, min=0.0)

    def _active_mean(viol, region_mask):
        """Region-balanced mean violation with a geometric blend between the
        active-pair and all-pair normalizers (controlled by ``norm_power``)::

            denom = act**np_ * total**(1 - np_)

        ``np_ = 1`` → mean over violators only: count-insensitive (an image with
        5 bad pairs is scored like one with 5000 — the old behavior). ``np_ = 0``
        → mean over *all* pairs: "more wrong = more loss", but per-violation
        gradient starves as violations get rare. ``np_ = 0.5`` keeps both — an
        image with 100× more violating pairs takes ~10× the loss, while a handful
        of violations still get a sqrt-boosted gradient instead of being drowned
        by the denominator. Violation *count* now moves the scalar."""
        w     = region_mask.to(dtype) * pair_w                     # (B, N, N)
        num   = (viol * w).sum(dim=(1, 2))
        act   = ((viol > 0).to(dtype) * w).sum(dim=(1, 2))         # weighted #active
        total = w.sum(dim=(1, 2))                                  # weighted #pairs
        denom = act.clamp(min=1.0) ** np_ * total.clamp(min=1.0) ** (1.0 - np_)
        return num / denom

    def _active_frac(viol, region_mask):
        """Weighted fraction of the region's pairs that are still violating."""
        w   = region_mask.to(dtype) * pair_w
        act = ((viol > 0).to(dtype) * w).sum(dim=(1, 2))
        tot = w.sum(dim=(1, 2)).clamp(min=1.0)
        return act / tot

    A_in  = _active_mean(v_pos, same_in)
    A_out = _active_mean(v_pos, same_out)
    R     = _active_mean(v_neg, diff)

    # Patch-weight-aware spliced fraction per image.
    pw_sum = pw.sum(dim=1).clamp(min=1e-6)
    p      = (pw * pos1.to(dtype)).sum(dim=1) / pw_sum             # (B,) in [0,1]
    wp_in  = p.clamp(0.0, 1.0) ** power
    wp_out = (1.0 - p).clamp(0.0, 1.0) ** power
    denom  = (wp_in + wp_out).clamp(min=1e-6)
    w_in, w_out = wp_in / denom, wp_out / denom

    cohesion     = w_in * A_in + w_out * A_out
    per_img_loss = cohesion + lambda_repel * R

    # Deferred anti-collapse diversity penalty (default OFF).
    if float(diversity_weight) > 0.0:
        def _region_mean_sim(region_mask):
            w = region_mask.to(dtype) * pair_w
            return (sim * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp(min=1.0)
        div_in  = torch.clamp(_region_mean_sim(same_in)  - diversity_tau, min=0.0)
        div_out = torch.clamp(_region_mean_sim(same_out) - diversity_tau, min=0.0)
        per_img_loss = per_img_loss + float(diversity_weight) * (
            w_in * div_in + w_out * div_out
        )

    weights = torch.where(
        is_single_class,
        torch.full_like(per_img_loss, float(single_class_weight)),
        torch.ones_like(per_img_loss),
    )
    if sample_weights is not None:
        weights = weights * sample_weights.to(device=device, dtype=dtype)
    loss = (per_img_loss * weights).sum() / weights.sum().clamp(min=1e-6)

    # ── diagnostics (median+SD; reals reported separately) ──
    splice  = ~is_single_class
    same_w  = same.to(dtype) * pair_w
    diff_w  = diff.to(dtype) * pair_w
    sim_pos_img = (sim * same_w).sum(dim=(1, 2)) / same_w.sum(dim=(1, 2)).clamp(min=1e-6)
    sim_neg_img = (sim * diff_w).sum(dim=(1, 2)) / diff_w.sum(dim=(1, 2)).clamp(min=1e-6)

    def _msd(vec, mask):
        if bool(mask.any()):
            vals = vec[mask]
            med  = float(vals.median().item())
            sd   = float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0
            return med, sd
        return float('nan'), float('nan')

    sp_med, sp_sd     = _msd(sim_pos_img, splice)
    sn_med, sn_sd     = _msd(sim_neg_img, splice)
    real_med, real_sd = _msd(sim_pos_img, is_single_class)   # within-image sim on reals

    # Active-pair fractions: the count-sensitive signal that R_mean alone hides.
    # These should fall as geometry improves even when the (active-normalized)
    # R_mean plateaus. Reported on splices only (reals have no cross pairs).
    repel_af_med, _   = _msd(_active_frac(v_neg, diff),    splice)
    cohes_af_med, _   = _msd(_active_frac(v_pos, same_in), splice)

    if patch_weights is not None:
        ignore_band_frac   = float((pw == 0).to(dtype).mean().item())
        soft_pos_band_frac = float(((pw > 0) & (pw < 1)).to(dtype).mean().item())
    else:
        ignore_band_frac, soft_pos_band_frac = 0.0, 0.0

    gap_med = (sp_med - sn_med) if (sp_med == sp_med and sn_med == sn_med) else float('nan')
    diag = {
        'loss': loss.item(),
        'mode': 'symmetric',
        'tau_pos': float(tau_pos), 'tau_neg': float(tau_neg),
        'lambda_repel': float(lambda_repel),
        'single_class_weight': float(single_class_weight),
        'area_balance_power': power,
        'norm_power': np_,
        'A_in_mean':  float(A_in[splice].mean().item())  if bool(splice.any()) else 0.0,
        'A_out_mean': float(A_out.mean().item()),
        'R_mean':     float(R[splice].mean().item())     if bool(splice.any()) else 0.0,
        'repel_active_frac_median':    repel_af_med,
        'cohesion_active_frac_median': cohes_af_med,
        'sim_pos_median': sp_med, 'sim_pos_std': sp_sd,
        'sim_neg_median': sn_med, 'sim_neg_std': sn_sd,
        'sim_gap_median': gap_med,
        'real_sim_median': real_med, 'real_sim_std': real_sd,
        'frac_single_class': float(is_single_class.to(dtype).mean().item()),
        'ignore_band_frac': ignore_band_frac,
        'soft_pos_band_frac': soft_pos_band_frac,
    }
    return loss, diag


def selective_symmetric_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    is_single_class: torch.Tensor,
    active_mask: torch.Tensor,
    tau_pos: float = 0.55,
    tau_neg: float = 0.20,
    lambda_repel: float = 1.0,
    single_class_weight: float = 0.05,
    area_balance_power: float = 0.5,
    norm_power: float = 1.0,
    sample_weights: torch.Tensor = None,
    patch_weights: torch.Tensor = None,
    diversity_weight: float = 0.0,
    diversity_tau: float = 0.90,
    topk: int = 0,
) -> Tuple[torch.Tensor, Dict]:
    """Apply symmetric_pairwise_contrastive_loss only to active samples."""
    active = active_mask.bool()
    if int(active.sum().item()) == 0:
        return z.sum() * 0.0, _empty_diag(z.device)
    sw  = sample_weights[active] if sample_weights is not None else None
    pwt = patch_weights[active]  if patch_weights  is not None else None
    return symmetric_pairwise_contrastive_loss(
        z[active], labels[active], is_single_class[active],
        sample_weights=sw, patch_weights=pwt,
        tau_pos=tau_pos, tau_neg=tau_neg, lambda_repel=lambda_repel,
        single_class_weight=single_class_weight, area_balance_power=area_balance_power,
        norm_power=norm_power,
        diversity_weight=diversity_weight, diversity_tau=diversity_tau, topk=topk,
    )


def embedding_invariance_loss(
    clean_z: torch.Tensor,
    aug_z: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """1 - mean cosine similarity between clean and augmented embeddings."""
    active = active_mask.bool()
    if int(active.sum().item()) == 0:
        return (clean_z.sum() + aug_z.sum()) * 0.0
    sim = (clean_z[active] * aug_z[active]).sum(dim=-1)
    return (1.0 - sim).mean()
