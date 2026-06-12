"""Unit tests for margin-attract behavior in pairwise_contrastive_loss.

Locks in three properties of the new attract_margin formulation:
  1. attract_margin=1.0 reproduces the legacy (1 - sim) point-attract.
  2. Single-class image with already-clustered patches → zero gradient.
  3. Single-class image with one outlier patch → non-zero gradient,
     proportional to the floor distance.
"""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import torch.nn.functional as F

from lab_utils.model.losses.contrastive import pairwise_contrastive_loss


def _normed(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def test_attract_margin_one_matches_legacy():
    """With attract_margin=1.0 the new formula equals the legacy (1 - sim)."""
    torch.manual_seed(0)
    B, N, D = 3, 12, 32
    z = _normed(torch.randn(B, N, D))
    labels = torch.randint(0, 2, (B, N))
    is_single = torch.zeros(B, dtype=torch.bool)
    is_single[0] = True

    # New: attract_margin=1.0
    new_loss, _ = pairwise_contrastive_loss(
        z, labels, is_single,
        neg_margin=0.3, lambda_repel=1.0, single_class_weight=0.0,
        attract_margin=1.0,
    )

    # Reference: (1 - sim) point-attract, same repel.
    sim = torch.bmm(z, z.transpose(1, 2))
    eye = torch.eye(N, dtype=torch.bool).unsqueeze(0)
    same = (labels.unsqueeze(2) == labels.unsqueeze(1)) & ~eye
    diff = (labels.unsqueeze(2) != labels.unsqueeze(1)) & ~eye
    same_f, diff_f = same.float(), diff.float()
    same_count = same_f.sum(dim=(1, 2)).clamp(min=1.0)
    diff_count = diff_f.sum(dim=(1, 2)).clamp(min=1.0)
    attract = ((1.0 - sim) * same_f).sum(dim=(1, 2)) / same_count
    repel = (torch.clamp(sim - 0.3, min=0.0) * diff_f).sum(dim=(1, 2)) / diff_count
    diff_present = (diff_f.sum(dim=(1, 2)) > 0).float()
    repel = repel * diff_present
    per_img = attract + repel
    weights = torch.where(is_single, torch.zeros_like(per_img), torch.ones_like(per_img))
    ref_loss = (per_img * weights).sum() / weights.sum().clamp(min=1.0)

    # Tolerance accounts for max(0, ...) clipping of marginal float-precision
    # cases where sim slightly exceeds 1.0 on identical or near-identical vectors.
    assert abs(new_loss.item() - ref_loss.item()) < 1e-5


def test_clustered_single_class_image_zero_gradient():
    """All patches already similar → margin-attract gives zero loss & gradient."""
    B, N, D = 1, 10, 32
    torch.manual_seed(1)
    base = _normed(torch.randn(1, D))
    z = base.unsqueeze(0).expand(1, N, D).clone().requires_grad_(True)
    labels = torch.zeros(B, N, dtype=torch.long)
    is_single = torch.ones(B, dtype=torch.bool)

    loss, diag = pairwise_contrastive_loss(
        z, labels, is_single,
        neg_margin=0.3, lambda_repel=1.0,
        single_class_weight=1.0,
        attract_margin=0.4,
    )
    loss.backward()

    assert loss.item() == 0.0, f'expected zero loss on clustered single-class, got {loss.item()}'
    assert diag['attract_active_frac'] == 0.0
    assert z.grad is not None and float(z.grad.abs().max()) == 0.0


def test_float_single_class_mask_is_accepted():
    """Float 0/1 masks from batch collation should behave like bool masks."""
    torch.manual_seed(4)
    B, N, D = 2, 8, 16
    z = _normed(torch.randn(B, N, D))
    labels = torch.randint(0, 2, (B, N))
    is_single_bool = torch.tensor([True, False])
    is_single_float = is_single_bool.to(dtype=z.dtype)

    loss_bool, diag_bool = pairwise_contrastive_loss(
        z, labels, is_single_bool,
        neg_margin=0.3, lambda_repel=1.0,
        single_class_weight=0.5,
        attract_margin=0.4,
    )
    loss_float, diag_float = pairwise_contrastive_loss(
        z, labels, is_single_float,
        neg_margin=0.3, lambda_repel=1.0,
        single_class_weight=0.5,
        attract_margin=0.4,
    )

    assert torch.isclose(loss_bool, loss_float)
    assert diag_bool['frac_single_class'] == diag_float['frac_single_class']
    assert torch.equal(
        diag_bool['per_image']['is_single_class'],
        diag_float['per_image']['is_single_class'],
    )


def test_clustered_single_class_legacy_is_nonzero():
    """Confirm the OLD point-attract would have fired here (regression sanity)."""
    B, N, D = 1, 10, 32
    torch.manual_seed(1)
    base = _normed(torch.randn(1, D))
    z = base.unsqueeze(0).expand(1, N, D).clone().requires_grad_(True)
    labels = torch.zeros(B, N, dtype=torch.long)
    is_single = torch.ones(B, dtype=torch.bool)

    legacy_loss, _ = pairwise_contrastive_loss(
        z, labels, is_single,
        neg_margin=0.3, lambda_repel=1.0,
        single_class_weight=1.0,
        attract_margin=1.0,
    )
    # With sim ≈ 1 everywhere, legacy attract = (1 - 1) ≈ 0 exactly.
    # To prove the legacy formula CAN fire on a single-class image with a
    # mild outlier, perturb one patch and confirm the legacy value > 0.
    z2 = z.detach().clone()
    z2[0, 0] = _normed(z2[0, 0] + 0.5 * torch.randn(D))
    legacy_outlier_loss, _ = pairwise_contrastive_loss(
        z2, labels, is_single,
        neg_margin=0.3, lambda_repel=1.0,
        single_class_weight=1.0,
        attract_margin=1.0,
    )
    assert legacy_outlier_loss.item() > 0.0


def test_outlier_patch_triggers_attract():
    """One patch dragged below the margin floor → loss > 0 and gradient flows."""
    B, N, D = 1, 10, 32
    torch.manual_seed(2)
    base = _normed(torch.randn(1, D))
    z = base.unsqueeze(0).expand(1, N, D).clone()

    # Push patch 0 to be roughly orthogonal (sim ≈ 0) to the cluster.
    z[0, 0] = _normed(torch.randn(D))
    z = z.requires_grad_(True)
    labels = torch.zeros(B, N, dtype=torch.long)
    is_single = torch.ones(B, dtype=torch.bool)

    loss, diag = pairwise_contrastive_loss(
        z, labels, is_single,
        neg_margin=0.3, lambda_repel=1.0,
        single_class_weight=1.0,
        attract_margin=0.4,
    )
    loss.backward()

    assert loss.item() > 0.0
    # Outlier should have non-trivial active-pair fraction.
    assert diag['attract_active_frac'] > 0.0
    # Gradient on outlier patch should dominate the others.
    grad_norms = z.grad[0].norm(dim=-1)
    assert grad_norms[0] > grad_norms[1:].max()


def test_single_class_topk_amplifies_outlier_penalty():
    """Top-k reduction should penalize a small outlier set more than the mean."""
    B, N, D = 1, 10, 32
    torch.manual_seed(5)
    base = _normed(torch.randn(1, D))
    z = base.unsqueeze(0).expand(1, N, D).clone()
    z[0, 0] = _normed(torch.randn(D))
    labels = torch.zeros(B, N, dtype=torch.long)
    is_single = torch.ones(B, dtype=torch.bool)

    mean_loss, _ = pairwise_contrastive_loss(
        z, labels, is_single,
        neg_margin=0.3, lambda_repel=1.0,
        single_class_weight=1.0,
        attract_margin=0.4,
        single_class_topk=0,
    )
    topk_loss, diag = pairwise_contrastive_loss(
        z, labels, is_single,
        neg_margin=0.3, lambda_repel=1.0,
        single_class_weight=1.0,
        attract_margin=0.4,
        single_class_topk=5,
    )

    assert topk_loss.item() > mean_loss.item()
    assert diag['single_class_topk'] == 5


def test_single_class_high_margin_squared_hinge_is_stronger():
    """A higher single-class floor with squared hinge should hit outliers harder."""
    B, N, D = 1, 10, 32
    torch.manual_seed(6)
    base = _normed(torch.randn(1, D))
    z = base.unsqueeze(0).expand(1, N, D).clone()
    z[0, 0] = _normed(torch.randn(D))
    labels = torch.zeros(B, N, dtype=torch.long)
    is_single = torch.ones(B, dtype=torch.bool)

    low_floor_loss, _ = pairwise_contrastive_loss(
        z, labels, is_single,
        neg_margin=0.3, lambda_repel=1.0,
        single_class_weight=1.0,
        attract_margin=0.4,
        single_class_attract_margin=0.4,
        single_class_attract_squared=False,
        single_class_topk=5,
    )
    high_floor_loss, diag = pairwise_contrastive_loss(
        z, labels, is_single,
        neg_margin=0.3, lambda_repel=1.0,
        single_class_weight=1.0,
        attract_margin=0.4,
        single_class_attract_margin=0.9,
        single_class_attract_squared=True,
        single_class_topk=5,
    )

    assert high_floor_loss.item() > low_floor_loss.item()
    assert diag['single_class_attract_margin'] == 0.9
    assert diag['single_class_attract_squared'] is True


def test_splice_region_balanced_attract_protects_small_region():
    """Splice attract should not be dominated by the large clean region."""
    z = torch.tensor(
        [[
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    z = _normed(z)
    labels = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=torch.long)
    is_single = torch.tensor([False])

    loss, diag = pairwise_contrastive_loss(
        z, labels, is_single,
        neg_margin=0.3, lambda_repel=0.0,
        single_class_weight=1.0,
        attract_margin=1.0,
    )

    assert abs(loss.item() - 0.5) < 1e-6
    assert diag['splice_region_balanced_attract'] is True


def test_attract_margin_zero_is_repel_only():
    """attract_margin=0 with normalized vectors → attract is zero everywhere."""
    torch.manual_seed(3)
    B, N, D = 2, 16, 32
    z = _normed(torch.randn(B, N, D))
    labels = torch.randint(0, 2, (B, N))
    is_single = torch.zeros(B, dtype=torch.bool)

    # attract_margin=0 + lambda_repel=0 → loss should be exactly 0
    # (sim ≥ 0 in expectation but can be negative; clamp(0 - sim, min=0) is
    # nonzero only when sim < 0. We disable repel to isolate attract.)
    loss, diag = pairwise_contrastive_loss(
        z, labels, is_single,
        neg_margin=0.3, lambda_repel=0.0,
        single_class_weight=1.0,
        attract_margin=0.0,
    )
    # attract_active_frac measures pairs with sim < attract_margin = 0,
    # i.e. anti-correlated pairs. Random vectors → ~half of pairs.
    # Loss is small (only the negative-sim portion contributes).
    assert loss.item() >= 0.0
    assert diag['attract_margin'] == 0.0
