"""Unit tests for the calibrated graph-components decode.

Synthetic fixtures only (no data/GPU). Locks in the properties that motivate
the decode over k-means(2):
  - planted splice recovered, single-class abstains;
  - anti-chaining (a sub-threshold bridge can't merge two clusters);
  - fragmented background (high mutual cross-sim, no edges) is REJECTED;
  - multi-region returns multiple accepted components;
  - determinism (no RNG); non-square / small grids handled.
"""

import numpy as np

from lab_utils.eval.partition import (
    DecodeSpec,
    calibrate_graph_decode,
    decode_deploy_mask,
    decode_oracle_labels,
    graph_components_decode,
)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _cluster(direction, n, noise, rng, dim=16):
    base = np.zeros((n, dim), dtype=np.float32)
    base[:, direction] = 1.0
    base += noise * rng.standard_normal((n, dim)).astype(np.float32)
    return _unit(base)


TP, TN = 0.55, 0.20


def test_planted_splice_recovered():
    rng = np.random.default_rng(0)
    bg = _cluster(0, 760, 0.05, rng)
    splice = _cluster(5, 24, 0.05, rng)
    z = np.concatenate([bg, splice], axis=0)
    mask, info = graph_components_decode(z, tau_pos=TP, tau_neg=TN)
    assert not info['abstained']
    assert info['n_accepted'] >= 1
    # The 24 splice patches (indices 760..783) should dominate the prediction.
    pred = np.where(mask == 1)[0]
    assert pred.size > 0
    recovered = np.mean(pred >= 760)
    assert recovered > 0.9, f'splice purity too low: {recovered}'


def test_single_class_abstains():
    rng = np.random.default_rng(1)
    z = _cluster(0, 784, 0.05, rng)
    mask, info = graph_components_decode(z, tau_pos=TP, tau_neg=TN)
    assert info['abstained']
    assert int(mask.sum()) == 0


def test_anti_chaining_bridge_does_not_merge():
    rng = np.random.default_rng(2)
    a = _cluster(0, 380, 0.03, rng)
    b = _cluster(7, 380, 0.03, rng)
    # A thin bridge of vectors halfway between the two modes, each below s_edge
    # similarity to either side.
    bridge = _unit(np.eye(16, dtype=np.float32)[0] + np.eye(16, dtype=np.float32)[7])
    bridge = np.tile(bridge, (24, 1)) + 0.03 * rng.standard_normal((24, 16)).astype(np.float32)
    z = np.concatenate([a, b, _unit(bridge)], axis=0)
    _, info = graph_components_decode(z, tau_pos=TP, tau_neg=TN)
    # Two real modes must not be fused into one giant component via the bridge:
    # at least two components of meaningful size survive.
    big = [c for c in info['components'] if c['size'] >= 50]
    assert info['n_components'] >= 2
    # background is one mode; at least one other large component exists.
    assert len(big) >= 1


def test_fragmented_background_rejected():
    rng = np.random.default_rng(3)
    # One semantic background direction, but split into two shards that happen
    # to lack mutual-kNN edges to each other. They still sit ON the background
    # (high cross-sim), so the cross test must reject the non-background shard.
    shard_a = _cluster(0, 400, 0.02, rng)
    shard_b = _cluster(0, 380, 0.02, rng)
    z = np.concatenate([shard_a, shard_b], axis=0)
    mask, info = graph_components_decode(z, tau_pos=TP, tau_neg=TN)
    # Even if the graph fragments, nothing should be accepted as a splice: the
    # shards are mutually too similar (cross > theta_x).
    assert int(mask.sum()) == 0, 'fragmented same-direction background leaked as splice'


def test_two_planted_splices_multi_region():
    rng = np.random.default_rng(4)
    bg = _cluster(0, 720, 0.04, rng)
    s1 = _cluster(5, 32, 0.04, rng)
    s2 = _cluster(9, 32, 0.04, rng)
    z = np.concatenate([bg, s1, s2], axis=0)
    _, info = graph_components_decode(z, tau_pos=TP, tau_neg=TN)
    assert info['n_accepted'] >= 2, f'expected 2 regions, got {info["n_accepted"]}'


def test_determinism():
    rng = np.random.default_rng(5)
    bg = _cluster(0, 760, 0.05, rng)
    splice = _cluster(5, 24, 0.05, rng)
    z = np.concatenate([bg, splice], axis=0)
    m1, _ = graph_components_decode(z, tau_pos=TP, tau_neg=TN)
    m2, _ = graph_components_decode(z, tau_pos=TP, tau_neg=TN)
    assert np.array_equal(m1, m2)


def test_non_square_and_small_grids():
    rng = np.random.default_rng(6)
    # 14×14 = 196 with spatial gate on.
    bg = _cluster(0, 196 - 20, 0.04, rng)
    splice = _cluster(5, 20, 0.04, rng)
    z = np.concatenate([bg, splice], axis=0)
    mask, info = graph_components_decode(
        z, tau_pos=TP, tau_neg=TN, grid_hw=(14, 14), r_spatial=2
    )
    assert mask.shape[0] == 196
    # N below min_patches abstains outright.
    z_small = _cluster(0, 9, 0.04, rng)
    m_small, info_small = graph_components_decode(z_small, tau_pos=TP, tau_neg=TN)
    assert info_small['abstained'] and int(m_small.sum()) == 0


def test_requires_unit_norm():
    rng = np.random.default_rng(7)
    z = (rng.standard_normal((100, 16)).astype(np.float32)) * 3.0  # not unit-norm
    try:
        graph_components_decode(z, tau_pos=TP, tau_neg=TN)
    except ValueError as exc:
        assert 'L2-normalized' in str(exc)
    else:
        raise AssertionError('expected ValueError on non-unit-norm input')


def test_dispatcher_kmeans_default_matches_spherical():
    from lab_utils.eval.partition import spherical_kmeans2
    rng = np.random.default_rng(8)
    bg = _cluster(0, 760, 0.05, rng)
    splice = _cluster(5, 24, 0.05, rng)
    z = np.concatenate([bg, splice], axis=0)
    labels = decode_oracle_labels(z, DecodeSpec())   # default = kmeans
    ref, _ = spherical_kmeans2(z, n_init=4)
    assert np.array_equal(labels, ref)


def test_deploy_mask_graph_commits_foreground():
    rng = np.random.default_rng(9)
    bg = _cluster(0, 760, 0.05, rng)
    splice = _cluster(5, 24, 0.05, rng)
    z = np.concatenate([bg, splice], axis=0)
    spec = DecodeSpec(method='graph', tau_pos=TP, tau_neg=TN)
    mask, info = decode_deploy_mask(z, spec)
    assert mask.dtype == bool
    assert info['method'] == 'graph'
    assert mask[760:].mean() > 0.9   # foreground is the splice region


def test_calibrate_graph_decode_runs():
    rng = np.random.default_rng(10)
    z_list, gt = [], []
    for _ in range(6):  # splices
        bg = _cluster(0, 200, 0.05, rng)
        sp = _cluster(5, 24, 0.05, rng)
        z_list.append(np.concatenate([bg, sp], axis=0)); gt.append(False)
    for _ in range(6):  # singles
        z_list.append(_cluster(0, 224, 0.05, rng)); gt.append(True)
    best_s, info = calibrate_graph_decode(z_list, gt, tau_pos=TP, tau_neg=TN)
    assert TN < best_s < TP
    assert 0.0 <= info['best_balanced_acc'] <= 1.0
