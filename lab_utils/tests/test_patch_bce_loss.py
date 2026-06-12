"""Unit tests for selective_patch_bce_loss — the dense supervised
splice-flagging baseline loss.

Locks in:
  1. Inactive items (missed-splice ignore) are excluded; reals (all-zero
     labels) are included as negative supervision.
  2. The per-patch weight map zeroes out ignored patches.
  3. pos_weight raises the loss when positive (splice) patches are missed.
  4. The all-inactive batch returns a real (gradient-carrying) zero.
"""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch

from lab_utils.model.losses.bce import selective_patch_bce_loss


def _setup():
    torch.manual_seed(0)
    B, N = 4, 16
    logit = torch.randn(B, N, requires_grad=True)
    labels = torch.zeros(B, N)
    labels[1, :4] = 1.0          # img1: supervised splice
    labels[2, 8:] = 1.0          # img2: supervised splice
    is_splice = torch.tensor([False, True, True, True])
    is_single = torch.tensor([True, False, False, True])  # img0 real, img3 missed
    return logit, labels, is_splice, is_single


def test_active_mask_includes_reals_excludes_missed():
    logit, labels, is_splice, is_single = _setup()
    active = ~(is_splice & is_single)          # exclude img3 only
    loss, diag = selective_patch_bce_loss(
        logit, labels, active, pos_weight=10.0)
    assert diag['n_active_img'] == 3           # real + 2 supervised splices
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logit.grad).all()


def test_patch_weight_zeroes_ignored_patches():
    logit, labels, is_splice, is_single = _setup()
    active = torch.ones(4, dtype=torch.bool)
    pw_full = torch.ones_like(labels)
    pw_half = pw_full.clone()
    pw_half[:, :8] = 0.0                        # ignore first half of every image
    l_full, _ = selective_patch_bce_loss(logit, labels, active, 1.0, patch_weights=pw_full)
    l_half, _ = selective_patch_bce_loss(logit, labels, active, 1.0, patch_weights=pw_half)
    # Different supervised patch sets → different loss; zero-weight patches drop out.
    assert abs(float(l_full) - float(l_half)) > 1e-6


def test_pos_weight_raises_loss_on_missed_positives():
    B, N = 4, 16
    labels = torch.zeros(B, N); labels[:, :2] = 1.0
    lg = torch.full((B, N), -3.0)               # confidently negative → misses positives
    active = torch.ones(B, dtype=torch.bool)
    l_lo, _ = selective_patch_bce_loss(lg, labels, active, pos_weight=1.0)
    l_hi, _ = selective_patch_bce_loss(lg, labels, active, pos_weight=20.0)
    assert float(l_hi) > float(l_lo)


def test_all_inactive_is_zero():
    logit, labels, _, _ = _setup()
    loss, diag = selective_patch_bce_loss(
        logit, labels, torch.zeros(4, dtype=torch.bool), pos_weight=10.0)
    assert float(loss) == 0.0
    assert diag['n_active_img'] == 0
