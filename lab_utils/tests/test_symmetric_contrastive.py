"""Unit tests for symmetric_pairwise_contrastive_loss.

Locks in the design properties of the mirrored dead-point loss:
  1. Dead zone: same-pairs ≥ τ_pos and diff-pairs ≤ τ_neg → loss & grad exactly 0.
  2. No upper bound: even cos≈1 same-pairs never get pushed apart.
  3. Separation: a cross pair above τ_neg drives R (loss > 0).
  4. Active-pair denominator: A_out = per-active-pair violation, NOT diluted by
     the count of satisfied same-pairs (independent of N).
  5. Area balance: weights sum to 1 ⇒ cohesion is invariant to region sizes.
  6. Reals are weighted very low via single_class_weight.
  7. top-k is a deferred hook and raises.
"""

import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pytest
import torch
import torch.nn.functional as F

from lab_utils.model.losses.contrastive import symmetric_pairwise_contrastive_loss


def _deg(*degrees):
    """Stack unit 2-D vectors from a list of angles in degrees → (len, 2)."""
    rows = [[math.cos(math.radians(d)), math.sin(math.radians(d))] for d in degrees]
    return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


def test_dead_zone_zero_loss_and_grad():
    """All same-pairs ≥ τ_pos and cross-pairs ≤ τ_neg ⇒ exactly zero everywhere."""
    # clean ≈ 0°±20° (within-sim cos40≈0.766 ≥ 0.55); splice ≈ 180°±20°.
    z = torch.cat([_deg(20, -20), _deg(160, 200)], dim=0).unsqueeze(0).clone()
    z = z.requires_grad_(True)
    labels = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    is_single = torch.tensor([False])

    loss, diag = symmetric_pairwise_contrastive_loss(
        z, labels, is_single, tau_pos=0.55, tau_neg=0.20, lambda_repel=1.0,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert float(z.grad.abs().max()) == 0.0
    assert diag['A_in_mean'] == 0.0 and diag['A_out_mean'] == 0.0 and diag['R_mean'] == 0.0


def test_no_upper_bound_on_cohesion():
    """Nearly-identical same-region patches get zero attract (no push-apart)."""
    clean = _deg(0, 1, 2)            # essentially coincident, sim ≈ 1
    splice = _deg(178, 180, 182)
    z = torch.cat([clean, splice], dim=0).unsqueeze(0).clone().requires_grad_(True)
    labels = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)
    is_single = torch.tensor([False])

    loss, diag = symmetric_pairwise_contrastive_loss(
        z, labels, is_single, tau_pos=0.55, tau_neg=0.20, lambda_repel=1.0,
    )
    # Cohesion satisfied (sim≈1 ≥ τ_pos) and regions separated ⇒ ~0 loss.
    assert loss.item() < 1e-5
    assert diag['A_in_mean'] == 0.0 and diag['A_out_mean'] == 0.0


def test_separation_violation_triggers_R():
    """A cross pair above τ_neg produces a positive separation term."""
    # clean at 0°, splice at 40° → cross sim cos40≈0.766 > τ_neg ⇒ R>0.
    z = torch.cat([_deg(0, 0), _deg(40, 40)], dim=0).unsqueeze(0)
    labels = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    is_single = torch.tensor([False])

    loss, diag = symmetric_pairwise_contrastive_loss(
        z, labels, is_single, tau_pos=0.55, tau_neg=0.20, lambda_repel=1.0,
    )
    assert diag['R_mean'] > 0.0
    # cross sim ≈ 0.766 ⇒ v_neg ≈ 0.766 - 0.20 = 0.566.
    assert abs(diag['R_mean'] - (math.cos(math.radians(40)) - 0.20)) < 1e-4


@pytest.mark.parametrize('n_cluster', [4, 49])
def test_active_denominator_not_diluted(n_cluster):
    """A_out equals the per-active-pair violation, independent of #satisfied pairs."""
    cluster = _deg(*([0] * n_cluster))      # all at (1,0), within-sim = 1 (satisfied)
    outlier = _deg(90)                       # (0,1): sim 0 with cluster ⇒ violates
    z = torch.cat([cluster, outlier], dim=0).unsqueeze(0)
    n = n_cluster + 1
    labels = torch.zeros(1, n, dtype=torch.long)
    is_single = torch.tensor([True])

    _, diag = symmetric_pairwise_contrastive_loss(
        z, labels, is_single, tau_pos=0.55, tau_neg=0.20,
        single_class_weight=1.0,
    )
    # Active pairs are only outlier↔cluster (sim 0 ⇒ v=0.55); cluster↔cluster
    # (v=0) are excluded from the denominator ⇒ A_out = 0.55 regardless of n.
    assert abs(diag['A_out_mean'] - 0.55) < 1e-5


def test_single_class_weighted_very_low():
    """In a MIXED batch, a real's contribution is scaled by single_class_weight.

    (With a batch of one image the weight cancels in the weighted-mean — the same
    relative-weight convention as the legacy loss — so this must be a mix.)
    """
    # Image 0: a perfectly-satisfied splice (L=0): tight clean +x, tight splice
    #          -x, antipodal ⇒ no cohesion or separation violation.
    splice = torch.cat([_deg(0, 0, 0), _deg(180, 180, 180)], dim=0)
    # Image 1: a real with one orthogonal outlier ⇒ A_out = 0.55.
    real = torch.cat([_deg(*([0] * 5)), _deg(90)], dim=0)
    z = torch.stack([splice, real], dim=0)                    # (2, 6, 2)
    labels = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]], dtype=torch.long)
    is_single = torch.tensor([False, True])

    def _loss(scw):
        loss, _ = symmetric_pairwise_contrastive_loss(
            z, labels, is_single, tau_pos=0.55, tau_neg=0.20, lambda_repel=1.0,
            single_class_weight=scw,
        )
        return loss.item()

    # loss(scw) = (0*1 + 0.55*scw) / (1 + scw).
    assert abs(_loss(1.0) - 0.55 * 1.0 / 2.0) < 1e-4
    assert abs(_loss(0.05) - 0.55 * 0.05 / 1.05) < 1e-4
    assert _loss(0.05) < _loss(1.0)


def _two_region_image(n_clean_cluster, n_splice_cluster):
    """Build a splice image: each region = cluster (sim 1) + one orthogonal
    outlier (sim 0 ⇒ within-region violation 0.55), regions antipodal (R=0)."""
    clean = torch.cat([_deg(*([0] * n_clean_cluster)), _deg(90)], dim=0)     # +x cluster, +y outlier
    splice = torch.cat([_deg(*([180] * n_splice_cluster)), _deg(270)], dim=0)  # -x cluster, -y outlier
    z = torch.cat([clean, splice], dim=0).unsqueeze(0)
    labels = torch.tensor(
        [[0] * (n_clean_cluster + 1) + [1] * (n_splice_cluster + 1)], dtype=torch.long
    )
    return z, labels


def test_area_balance_invariant_to_region_size():
    """w_in + w_out = 1 ⇒ cohesion (=0.55 here) is invariant to the split ratio."""
    is_single = torch.tensor([False])
    z_a, lab_a = _two_region_image(8, 2)     # p = 3/12
    z_b, lab_b = _two_region_image(2, 8)     # p = 9/12
    loss_a, diag_a = symmetric_pairwise_contrastive_loss(
        z_a, lab_a, is_single, tau_pos=0.55, tau_neg=0.20, lambda_repel=1.0,
    )
    loss_b, diag_b = symmetric_pairwise_contrastive_loss(
        z_b, lab_b, is_single, tau_pos=0.55, tau_neg=0.20, lambda_repel=1.0,
    )
    # Both regions violate by 0.55; R=0 (antipodal). Weights sum to 1 ⇒ loss≈0.55.
    assert diag_a['R_mean'] == 0.0 and diag_b['R_mean'] == 0.0
    assert abs(loss_a.item() - 0.55) < 1e-4
    assert abs(loss_b.item() - 0.55) < 1e-4
    assert abs(loss_a.item() - loss_b.item()) < 1e-5


def test_topk_deferred_raises():
    z = _deg(0, 90, 180).unsqueeze(0)
    labels = torch.tensor([[0, 0, 1]], dtype=torch.long)
    is_single = torch.tensor([False])
    with pytest.raises(NotImplementedError):
        symmetric_pairwise_contrastive_loss(z, labels, is_single, topk=4)
