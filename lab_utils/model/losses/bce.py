"""lab_utils.model.losses.bce — binary cross-entropy losses for forensics head.

Lifted from contrastive_test/core/harness_losses.py (BCE section).
"""

import torch
import torch.nn.functional as F


def selective_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_mask: torch.Tensor,
    pos_weight: float,
    sample_weights: torch.Tensor = None,
) -> torch.Tensor:
    """BCE with logits applied only to active (supervised) samples.

    Args:
        logits:      (B, N) raw logits.
        labels:      (B, N) int {0, 1}.
        active_mask: (B,) bool — which batch items to include.
        pos_weight:  BCE positive-class weight.

    Returns:
        Scalar loss tensor.  Zero-gradient zero if no active items.
    """
    active = active_mask.bool()
    if int(active.sum().item()) == 0:
        return logits.sum() * 0.0
    pos_w = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss_per_patch = F.binary_cross_entropy_with_logits(
        logits[active],
        labels[active].to(logits.dtype),
        pos_weight=pos_w,
        reduction='none',
    )
    loss_per_img = loss_per_patch.mean(dim=1)
    if sample_weights is not None:
        weights = sample_weights[active].to(device=logits.device, dtype=logits.dtype)
        return (loss_per_img * weights).sum() / weights.sum().clamp(min=1.0)
    return loss_per_img.mean()


def selective_bce_loss_with_diag(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_mask: torch.Tensor,
    pos_weight: float,
    sample_weights: torch.Tensor = None,
) -> tuple[torch.Tensor, dict]:
    """BCE loss plus per-active-image diagnostics for bucketed logging."""
    active = active_mask.bool()
    if int(active.sum().item()) == 0:
        return logits.sum() * 0.0, {'per_image': {'loss': torch.empty(0), 'weight': torch.empty(0)}}
    pos_w = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss_per_patch = F.binary_cross_entropy_with_logits(
        logits[active],
        labels[active].to(logits.dtype),
        pos_weight=pos_w,
        reduction='none',
    )
    loss_per_img = loss_per_patch.mean(dim=1)
    if sample_weights is not None:
        weights = sample_weights[active].to(device=logits.device, dtype=logits.dtype)
    else:
        weights = torch.ones_like(loss_per_img)
    loss = (loss_per_img * weights).sum() / weights.sum().clamp(min=1.0)
    return loss, {
        'per_image': {
            'loss': loss_per_img.detach().cpu(),
            'weight': weights.detach().cpu(),
        }
    }


def selective_patch_bce_loss(
    logits: torch.Tensor,            # (B, N) per-patch logits
    labels: torch.Tensor,            # (B, N) {0, 1} per-patch splice labels
    active_mask: torch.Tensor,       # (B,) bool — images to supervise
    pos_weight: float,
    sample_weights: torch.Tensor = None,   # (B,)
    patch_weights: torch.Tensor = None,     # (B, N) in [0,1]; None = all 1
) -> tuple[torch.Tensor, dict]:
    """Dense per-patch BCE for the supervised splice-flagging baseline.

    Unlike :func:`selective_bce_loss`, this honors a per-patch weight map
    (``patch_weights``) so the boundary ignore/soft band zeroes out ambiguous
    edge patches exactly as the contrastive loss does — keeping the two methods'
    supervision masks identical. Each image's loss is the patch-weighted mean
    over its patches; ``pos_weight`` rebalances the rare positive (splice)
    patches against the abundant clean patches.

    ``active_mask`` should include reals (all-zero labels → negative
    supervision that trains specificity) and supervised splices, and exclude
    missed-splice crops whose splice fell below the in-frame threshold (no
    reliable patch labels). Returns (scalar loss, diagnostics).
    """
    active = active_mask.bool()
    device, dtype = logits.device, logits.dtype
    if int(active.sum().item()) == 0:
        return logits.sum() * 0.0, {'patch_pos_frac': 0.0, 'pred_pos_frac': 0.0, 'n_active_img': 0}
    lg = logits[active]
    tg = labels[active].to(dtype)
    pos_w = torch.tensor(float(pos_weight), device=device, dtype=dtype)
    per_patch = F.binary_cross_entropy_with_logits(
        lg, tg, pos_weight=pos_w, reduction='none',
    )                                                              # (b, N)
    if patch_weights is not None:
        pw = patch_weights[active].to(device=device, dtype=dtype).clamp(0.0, 1.0)
    else:
        pw = torch.ones_like(per_patch)
    num = (per_patch * pw).sum(dim=1)
    den = pw.sum(dim=1).clamp(min=1.0)
    per_img = num / den                                            # (b,)
    if sample_weights is not None:
        w = sample_weights[active].to(device=device, dtype=dtype)
    else:
        w = torch.ones_like(per_img)
    loss = (per_img * w).sum() / w.sum().clamp(min=1.0)
    with torch.no_grad():
        wsum = pw.sum().clamp(min=1.0)
        patch_pos_frac = float(((tg > 0.5).to(dtype) * pw).sum().item() / wsum.item())
        pred_pos_frac  = float(((lg >= 0).to(dtype) * pw).sum().item() / wsum.item())
    return loss, {
        'patch_pos_frac': patch_pos_frac,
        'pred_pos_frac':  pred_pos_frac,
        'n_active_img':   int(active.sum().item()),
    }


def logit_consistency_loss(
    clean_logits: torch.Tensor,
    aug_logits: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE between sigmoid probabilities on clean vs augmented views.

    Encourages the BCE head to be invariant to light augmentations.

    Args:
        clean_logits: (B, N) logits on clean image.
        aug_logits:   (B, N) logits on augmented image.
        active_mask:  (B,) bool.

    Returns:
        Scalar loss tensor.
    """
    active = active_mask.bool()
    if int(active.sum().item()) == 0:
        return clean_logits.sum() * 0.0
    clean_prob = torch.sigmoid(clean_logits[active])
    aug_prob   = torch.sigmoid(aug_logits[active])
    return F.mse_loss(clean_prob, aug_prob)
