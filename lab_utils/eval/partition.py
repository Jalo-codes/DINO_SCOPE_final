"""lab_utils.eval.partition — patch decoders (k-means + calibrated graph) and gates.

Original content lifted from contrastive_test/core/partition.py (spherical
k-means + silhouette gate). Extended with ``graph_components_decode`` — a
connected-components decode over a thresholded similarity graph that uses the
*calibrated* contrastive geometry (same-region pairs ≥ tau_pos, cross-region
pairs ≤ tau_neg) instead of re-deriving a 2-way split from scratch. See
``GRAPH_DECODE_PLAN.md`` for the rationale and exact formulation.
"""

import dataclasses
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Spherical KMeans (k=2) ────────────────────────────────────────────────────

def _init_centroids(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n  = z.shape[0]
    i0 = int(rng.integers(0, n))
    c0 = z[i0]
    d  = np.clip(1.0 - z @ c0, a_min=0.0, a_max=None)
    if d.sum() <= 0:
        i1 = int(rng.integers(0, n))
    else:
        i1 = int(rng.choice(n, p=d / d.sum()))
    return np.stack([z[i0], z[i1]], axis=0)


def spherical_kmeans2(
    z: np.ndarray,
    n_init: int = 4,
    n_iters: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """Spherical k-means (k=2) on L2-normalised embeddings.

    Args:
        z:       (N, D) float32, L2-normalised.
        n_init:  Number of random restarts.
        n_iters: Maximum iterations per restart.
        seed:    Base seed for k-means++ initialisation.

    Returns:
        (labels, inertia) — labels ∈ {0,1}^N, inertia = sum(1 - cos to centroid).
    """
    z = np.ascontiguousarray(z, dtype=np.float32)
    best_labels  = None
    best_inertia = np.inf

    for run in range(n_init):
        rng       = np.random.default_rng(seed + run)
        centroids = _init_centroids(z, rng)
        labels    = np.zeros(z.shape[0], dtype=np.int64)

        for _ in range(n_iters):
            sim        = z @ centroids.T
            new_labels = np.argmax(sim, axis=1)
            if (new_labels == labels).all():
                labels = new_labels
                break
            labels        = new_labels
            new_centroids = np.zeros_like(centroids)
            for k in (0, 1):
                mask = labels == k
                if mask.sum() == 0:
                    other = centroids[1 - k]
                    far   = int(np.argmin(z @ other))
                    new_centroids[k] = z[far]
                else:
                    mean = z[mask].mean(axis=0)
                    n    = np.linalg.norm(mean) + 1e-12
                    new_centroids[k] = mean / n
            centroids = new_centroids

        sim     = z @ centroids.T
        labels  = np.argmax(sim, axis=1)
        inertia = float((1.0 - sim[np.arange(z.shape[0]), labels]).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels  = labels

    return best_labels, best_inertia


def _silhouette_from_dist(dist: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score given a precomputed N×N distance matrix.

    Vectorised binary-label specialisation (k=2 only).  Hot path for the
    silhouette-null gate, which calls this 32× per image per eval step.
    """
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    n0, n1 = idx0.size, idx1.size
    if n0 < 2 or n1 < 2:
        return -1.0

    # Sub-matrices
    d00 = dist[np.ix_(idx0, idx0)]   # (n0, n0)
    d11 = dist[np.ix_(idx1, idx1)]   # (n1, n1)
    d01 = dist[np.ix_(idx0, idx1)]   # (n0, n1)

    # Mean intra-cluster distance (exclude self on diagonal)
    a0 = (d00.sum(axis=1) - np.diag(d00)) / (n0 - 1)   # (n0,)
    a1 = (d11.sum(axis=1) - np.diag(d11)) / (n1 - 1)   # (n1,)

    # Mean inter-cluster distance
    b0 = d01.mean(axis=1)   # (n0,)
    b1 = d01.mean(axis=0)   # (n1,)

    # Silhouette per point, guarded against zero denominator
    denom0 = np.maximum(a0, b0)
    denom1 = np.maximum(a1, b1)
    s0 = np.where(denom0 > 0, (b0 - a0) / denom0, 0.0)
    s1 = np.where(denom1 > 0, (b1 - a1) / denom1, 0.0)

    return float(np.concatenate([s0, s1]).mean())


def silhouette_cosine(z: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette score in cosine distance space.

    Returns a value in [-1, 1].  Returns -1 when there are fewer than 2
    points or only one cluster is present.
    """
    z = np.ascontiguousarray(z, dtype=np.float32)
    sim  = z @ z.T
    dist = 1.0 - sim
    return _silhouette_from_dist(dist, labels)


def partition_image(
    z: np.ndarray,
    tau: float,
    n_init: int = 4,
) -> Tuple[np.ndarray, dict]:
    """Cluster z into 2 groups; gate with silhouette < tau → single-class.

    Args:
        z:      (N, D) L2-normalised embeddings for one image.
        tau:    Silhouette threshold.  Below → predicted single-class.
        n_init: k-means restarts.

    Returns:
        (mask, info) — mask is (N,) int {0,1}, all-zeros if single-class.
    """
    raw_labels, _ = spherical_kmeans2(z, n_init=n_init)
    sil           = silhouette_cosine(z, raw_labels)
    n0 = int((raw_labels == 0).sum())
    n1 = int((raw_labels == 1).sum())
    smaller = 0 if n0 <= n1 else 1

    if sil < tau:
        return (
            np.zeros(z.shape[0], dtype=np.int64),
            {'predicted_single': True, 'silhouette': sil,
             'smaller_count': min(n0, n1), 'larger_count': max(n0, n1),
             'raw_labels': raw_labels, 'smaller_cluster_label': smaller},
        )
    mask = (raw_labels == smaller).astype(np.int64)
    return (
        mask,
        {'predicted_single': False, 'silhouette': sil,
         'smaller_count': min(n0, n1), 'larger_count': max(n0, n1),
         'raw_labels': raw_labels, 'smaller_cluster_label': smaller},
    )


def calibrate_gate_tau(
    z_list: Sequence[np.ndarray],
    gt_is_single: Sequence[bool],
    tau_grid: Optional[np.ndarray] = None,
    fallback_tau: Optional[float] = None,
) -> Tuple[float, dict]:
    """Calibrate the silhouette gate threshold tau on a calibration set.

    Sweeps tau and picks the value that maximises balanced accuracy
    (TPR + TNR) / 2 on gt_is_single labels.

    Args:
        z_list:       List of (N, D) embedding arrays, one per image.
        gt_is_single: Ground-truth single-class flag per image.
        tau_grid:     Grid of tau candidates to sweep (auto-derived if None).
        fallback_tau: Used when z_list is empty (raises ValueError if also None).

    Returns:
        (best_tau, info_dict)
    """
    from lab_utils.errors import EvalError

    if len(z_list) == 0:
        if fallback_tau is None:
            raise EvalError("calibrate_gate_tau: calibration set is empty and fallback_tau is None.")
        return float(fallback_tau), {
            'silhouettes': [], 'gt_is_single': [], 'per_tau': [],
            'best_tau': float(fallback_tau), 'best_balanced_acc': float('nan'),
            'used_fallback': True, 'fallback_reason': 'empty_calibration_set',
        }

    sils = np.array(
        [silhouette_cosine(z, spherical_kmeans2(z, n_init=4)[0]) for z in z_list],
        dtype=np.float64,
    )
    gt = np.asarray(gt_is_single, dtype=bool)

    if tau_grid is None:
        lo = max(-0.10, float(sils.min()) - 0.05)
        hi = min(0.999, float(sils.max()) + 0.05)
        if hi <= lo:
            hi = min(0.999, lo + 0.05)
        tau_grid = np.linspace(lo, hi, 81)

    n_pos = int(gt.sum())
    n_neg = int((~gt).sum())
    best_tau, best_score = float(tau_grid[0]), -np.inf
    best_tpr = best_tnr = float('nan')
    per_tau  = []

    for tau in tau_grid:
        pred_single = sils < tau
        tpr = float((pred_single & gt).sum()) / n_pos if n_pos > 0 else 1.0
        tnr = float((~pred_single & ~gt).sum()) / n_neg if n_neg > 0 else 1.0
        bacc = 0.5 * (tpr + tnr)
        per_tau.append((float(tau), bacc, tpr, tnr))
        if bacc > best_score:
            best_score, best_tau, best_tpr, best_tnr = bacc, float(tau), tpr, tnr

    return best_tau, {
        'silhouettes': sils.tolist(),
        'gt_is_single': gt.tolist(),
        'per_tau': per_tau,
        'best_tau': best_tau,
        'best_balanced_acc': best_score,
        'best_tpr': best_tpr,
        'best_tnr': best_tnr,
        'n_single': n_pos,
        'n_two_class': n_neg,
        'tau_grid_min': float(np.min(tau_grid)),
        'tau_grid_max': float(np.max(tau_grid)),
        'single_silhouette_mean': float(np.mean(sils[gt])) if n_pos > 0 else float('nan'),
        'single_silhouette_median': float(np.median(sils[gt])) if n_pos > 0 else float('nan'),
        'single_silhouette_std': float(np.std(sils[gt])) if n_pos > 0 else float('nan'),
        'two_class_silhouette_mean': float(np.mean(sils[~gt])) if n_neg > 0 else float('nan'),
        'two_class_silhouette_median': float(np.median(sils[~gt])) if n_neg > 0 else float('nan'),
        'two_class_silhouette_std': float(np.std(sils[~gt])) if n_neg > 0 else float('nan'),
        'silhouette_gap_mean': (
            float(sils[~gt].mean() - sils[gt].mean())
            if n_pos > 0 and n_neg > 0 else float('nan')
        ),
    }


# ── Per-image silhouette null gate ────────────────────────────────────────────
#
# The silhouette-vs-tau gate (`partition_image`) compares an image's silhouette
# to a global threshold calibrated on a pooled population.  That threshold is
# content-prior-dependent: the silhouette distribution of "real" images shifts
# with the dataset, so a tau calibrated on IMD2020 reals over-suppresses
# CASIA splices.
#
# The null gate replaces the global threshold with a *per-image* null
# distribution: for each image, compare the k-means silhouette to the
# silhouette of `n_shuffles` random binary partitions of the SAME embeddings
# at the SAME cluster sizes.  Under "no two-cluster structure", k-means barely
# beats a random split; under a real splice, it beats it by many standard
# deviations.  The dataset's content prior cancels because the null is built
# from the image's own embeddings.

def silhouette_null(
    z: np.ndarray,
    cluster_sizes: Tuple[int, int],
    n_shuffles: int = 32,
    seed: int = 0,
    dist: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Null distribution of silhouette under random binary partitions.

    Each null partition assigns the embeddings to two clusters of the given
    sizes uniformly at random (matching the k-means cluster sizes keeps the
    silhouette directly comparable).

    Args:
        z:             (N, D) L2-normalised embeddings (ignored if dist given).
        cluster_sizes: (n0, n1) sizes for the random binary partition.
        n_shuffles:    Number of random partitions to draw.
        seed:          RNG seed for reproducibility.
        dist:          Optional precomputed (N, N) cosine-distance matrix.
                       Pass to avoid recomputing z @ z.T.

    Returns:
        (null_mean, null_std, null_silhouettes_array_of_length_n_shuffles)
    """
    n0, n1 = int(cluster_sizes[0]), int(cluster_sizes[1])
    n_total = n0 + n1
    if dist is None:
        z_arr = np.ascontiguousarray(z, dtype=np.float32)
        if z_arr.shape[0] != n_total:
            raise ValueError(
                f"silhouette_null: z has {z_arr.shape[0]} rows but cluster_sizes sum to {n_total}"
            )
        sim = z_arr @ z_arr.T
        dist = 1.0 - sim
    elif dist.shape[0] != n_total:
        raise ValueError(
            f"silhouette_null: dist is {dist.shape[0]}×{dist.shape[1]} but cluster_sizes sum to {n_total}"
        )

    if n0 < 2 or n1 < 2:
        return -1.0, 0.0, np.full(n_shuffles, -1.0, dtype=np.float64)

    rng = np.random.default_rng(seed)
    null_sils = np.empty(int(n_shuffles), dtype=np.float64)
    labels = np.empty(n_total, dtype=np.int64)
    for i in range(int(n_shuffles)):
        idx = rng.permutation(n_total)
        labels[idx[:n0]] = 0
        labels[idx[n0:]] = 1
        null_sils[i] = _silhouette_from_dist(dist, labels)
    return float(null_sils.mean()), float(null_sils.std()), null_sils


def partition_image_null(
    z: np.ndarray,
    tau_z: float,
    n_init: int = 4,
    n_shuffles: int = 32,
    seed: int = 0,
) -> Tuple[np.ndarray, dict]:
    """Cluster z into 2 groups; gate via z-score against per-image null.

    Mirrors `partition_image` but replaces the global silhouette threshold
    with a per-image standardisation: gate fires when the k-means silhouette
    is `tau_z` standard deviations or more above the random-partition null.

    Args:
        z:          (N, D) L2-normalised embeddings for one image.
        tau_z:      Z-score threshold.  At/above → predicted two-class.
        n_init:     k-means restarts.
        n_shuffles: Random partitions used to build the per-image null.
        seed:       RNG seed for reproducible nulls.

    Returns:
        (mask, info) — mask is (N,) int {0,1}, all-zeros if single-class.
        info adds 'silhouette_z', 'null_mean', 'null_std' on top of the
        fields returned by `partition_image`.
    """
    raw_labels, _ = spherical_kmeans2(z, n_init=n_init)
    n0 = int((raw_labels == 0).sum())
    n1 = int((raw_labels == 1).sum())
    smaller = 0 if n0 <= n1 else 1

    # Compute distance matrix once and reuse for the observed silhouette + null.
    z_arr = np.ascontiguousarray(z, dtype=np.float32)
    dist  = 1.0 - (z_arr @ z_arr.T)
    sil   = _silhouette_from_dist(dist, raw_labels)
    null_mean, null_std, _ = silhouette_null(
        z_arr, cluster_sizes=(n0, n1),
        n_shuffles=n_shuffles, seed=seed, dist=dist,
    )
    sil_z = (sil - null_mean) / max(null_std, 1e-6)

    info = {
        'silhouette': sil,
        'silhouette_z': float(sil_z),
        'null_mean': float(null_mean),
        'null_std': float(null_std),
        'smaller_count': min(n0, n1),
        'larger_count':  max(n0, n1),
        'raw_labels': raw_labels,
        'smaller_cluster_label': smaller,
    }
    if sil_z < tau_z:
        info['predicted_single'] = True
        return np.zeros(z.shape[0], dtype=np.int64), info
    info['predicted_single'] = False
    mask = (raw_labels == smaller).astype(np.int64)
    return mask, info


def calibrate_gate_tau_null(
    z_list: Sequence[np.ndarray],
    gt_is_single: Sequence[bool],
    tau_z_grid: Optional[np.ndarray] = None,
    fallback_tau_z: Optional[float] = None,
    n_shuffles: int = 32,
    seed: int = 0,
) -> Tuple[float, dict]:
    """Calibrate the silhouette-null z-score threshold on a calibration set.

    Mirrors `calibrate_gate_tau`'s structure (sweep, balanced-acc objective,
    deterministic) but operates on per-image silhouette z-scores.

    Args:
        z_list:        List of (N, D) embedding arrays, one per image.
        gt_is_single:  Ground-truth single-class flag per image.
        tau_z_grid:    Grid of tau_z candidates (auto-derived if None).
        fallback_tau_z: Used when z_list is empty.
        n_shuffles:    Random partitions used to build each per-image null.
        seed:          Base RNG seed; per-image seed is offset deterministically.

    Returns:
        (best_tau_z, info_dict)
    """
    from lab_utils.errors import EvalError

    if len(z_list) == 0:
        if fallback_tau_z is None:
            raise EvalError(
                "calibrate_gate_tau_null: calibration set is empty and fallback_tau_z is None."
            )
        return float(fallback_tau_z), {
            'silhouette_z': [], 'gt_is_single': [], 'per_tau': [],
            'best_tau_z': float(fallback_tau_z), 'best_balanced_acc': float('nan'),
            'used_fallback': True, 'fallback_reason': 'empty_calibration_set',
        }

    sil_zs = np.empty(len(z_list), dtype=np.float64)
    sils   = np.empty(len(z_list), dtype=np.float64)
    null_means = np.empty(len(z_list), dtype=np.float64)
    null_stds  = np.empty(len(z_list), dtype=np.float64)
    for i, z in enumerate(z_list):
        raw_labels, _ = spherical_kmeans2(z, n_init=4)
        n0 = int((raw_labels == 0).sum())
        n1 = int((raw_labels == 1).sum())
        z_arr = np.ascontiguousarray(z, dtype=np.float32)
        dist  = 1.0 - (z_arr @ z_arr.T)
        sil   = _silhouette_from_dist(dist, raw_labels)
        nm, ns, _ = silhouette_null(
            z_arr, cluster_sizes=(n0, n1),
            n_shuffles=n_shuffles, seed=seed + i, dist=dist,
        )
        sils[i] = sil
        sil_zs[i] = (sil - nm) / max(ns, 1e-6)
        null_means[i] = nm
        null_stds[i]  = ns

    gt = np.asarray(gt_is_single, dtype=bool)

    if tau_z_grid is None:
        lo = float(sil_zs.min()) - 0.5
        hi = float(sil_zs.max()) + 0.5
        if hi <= lo:
            hi = lo + 1.0
        tau_z_grid = np.linspace(lo, hi, 81)

    n_pos = int(gt.sum())
    n_neg = int((~gt).sum())
    best_tau_z, best_score = float(tau_z_grid[0]), -np.inf
    best_tpr = best_tnr = float('nan')
    per_tau  = []

    for tau_z in tau_z_grid:
        pred_single = sil_zs < tau_z
        tpr = float((pred_single & gt).sum()) / n_pos if n_pos > 0 else 1.0
        tnr = float((~pred_single & ~gt).sum()) / n_neg if n_neg > 0 else 1.0
        bacc = 0.5 * (tpr + tnr)
        per_tau.append((float(tau_z), bacc, tpr, tnr))
        if bacc > best_score:
            best_score, best_tau_z, best_tpr, best_tnr = bacc, float(tau_z), tpr, tnr

    return best_tau_z, {
        'silhouette_z': sil_zs.tolist(),
        'silhouettes': sils.tolist(),
        'null_means': null_means.tolist(),
        'null_stds': null_stds.tolist(),
        'gt_is_single': gt.tolist(),
        'per_tau': per_tau,
        'best_tau_z': best_tau_z,
        'best_balanced_acc': best_score,
        'best_tpr': best_tpr,
        'best_tnr': best_tnr,
        'n_single': n_pos,
        'n_two_class': n_neg,
        'tau_z_grid_min': float(np.min(tau_z_grid)),
        'tau_z_grid_max': float(np.max(tau_z_grid)),
        'single_z_mean':   float(np.mean(sil_zs[gt]))   if n_pos > 0 else float('nan'),
        'single_z_median': float(np.median(sil_zs[gt])) if n_pos > 0 else float('nan'),
        'single_z_std':    float(np.std(sil_zs[gt]))    if n_pos > 0 else float('nan'),
        'two_class_z_mean':   float(np.mean(sil_zs[~gt]))   if n_neg > 0 else float('nan'),
        'two_class_z_median': float(np.median(sil_zs[~gt])) if n_neg > 0 else float('nan'),
        'two_class_z_std':    float(np.std(sil_zs[~gt]))    if n_neg > 0 else float('nan'),
        'z_gap_mean': (
            float(sil_zs[~gt].mean() - sil_zs[gt].mean())
            if n_pos > 0 and n_neg > 0 else float('nan')
        ),
    }


# ── Calibrated graph-components decode ────────────────────────────────────────
#
# K-means(2) ignores the trained margins and is FORCED to split every image into
# two clusters, so when no margin-respecting split exists it partitions along
# semantics (large blobs swallowing background / surrounding objects; catastrophic
# output when the image is clean-ish). The graph decode thresholds pairwise
# similarity INSIDE the trained dead band [tau_neg, tau_pos] and takes connected
# components, so:
#   - joining the splice component needs an ABSOLUTE similarity bar (a pair the
#     training pushed to ≤ tau_neg can't clear it) — bleed-resistant by design;
#   - the number of components falls out automatically (multi-region for free);
#   - zero accepted components = natural abstention (replaces the silhouette gate);
#   - fully deterministic (no RNG, no n_init restarts, no prototype, no attention).


def _infer_grid_hw(n: int) -> Optional[Tuple[int, int]]:
    """Best-effort square grid (h, w) for N patches; None when N isn't square."""
    s = int(round(math.sqrt(n)))
    return (s, s) if s * s == n else None


def _union_find_components(adj: np.ndarray) -> np.ndarray:
    """Connected-component labels for a boolean (N, N) symmetric adjacency.

    Plain array-based union-find with path compression (numpy only — scipy is
    not a repo dependency). Isolated nodes form singleton components.
    """
    n = adj.shape[0]
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    # Only iterate the upper triangle of present edges.
    src, dst = np.where(np.triu(adj, k=1))
    for a, b in zip(src.tolist(), dst.tolist()):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    # Relabel roots to compact 0..k-1, ordered by first appearance (deterministic).
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def graph_components_decode(
    z: np.ndarray,                       # (N, D) L2-normalized
    *,
    tau_pos: float,
    tau_neg: float,
    grid_hw: Optional[Tuple[int, int]] = None,
    s_edge: Optional[float] = None,      # None → (tau_pos + tau_neg) / 2
    mutual_knn_k: int = 10,
    r_spatial: Optional[int] = None,     # None → feature graph only (no spatial gate)
    m_min: int = 4,
    theta_w: Optional[float] = None,     # None → tau_pos - 0.05
    theta_x: Optional[float] = None,     # None → (tau_pos + tau_neg) / 2
    attention: Optional[np.ndarray] = None,
    attention_polarity: bool = False,
    min_patches: int = 16,
) -> Tuple[np.ndarray, dict]:
    """Connected-components decode over a calibrated similarity graph.

    Args:
        z:            (N, D) L2-normalized per-patch contrastive embeddings — the
                      SAME array fed to ``spherical_kmeans2``.
        tau_pos/tau_neg: trained margins (from the run config). They set the
                      decode's default thresholds — do NOT freeze numeric values.
        grid_hw:      (H, W) patch grid; inferred as square when None. Required
                      only when ``r_spatial`` is set.
        s_edge:       absolute similarity bar for an edge. Default mid-band
                      (tau_pos + tau_neg) / 2.
        mutual_knn_k: an edge (i, j) also requires j ∈ kNN(i) AND i ∈ kNN(j)
                      (anti-chaining).
        r_spatial:    if set, an edge also requires Chebyshev grid distance ≤ r.
        m_min:        ignore components smaller than this (drops singletons/noise).
        theta_w:      accept a component iff internal cohesion ≥ theta_w.
                      Default tau_pos - 0.05.
        theta_x:      accept a component iff its mean similarity to background
                      ≤ theta_x (kills fragmented-background shards). Default
                      mid-band.
        attention:    (N,) per-patch BCE attention; only consulted when
                      ``attention_polarity`` is True.
        attention_polarity: pick background among large components by LOWEST mean
                      attention (handles the >50%-splice regime). Default OFF.
        min_patches:  images with fewer patches than this abstain outright.

    Returns:
        (mask, info) — mask is (N,) int {0,1}; 1 = accepted foreground (splice).
        all-zeros = abstain (predicted single-class). ``info`` carries per-component
        stats, the chosen parameters, background size, and ``abstained``.
    """
    z = np.ascontiguousarray(z, dtype=np.float32)
    n = z.shape[0]
    s_edge = float((tau_pos + tau_neg) / 2.0) if s_edge is None else float(s_edge)
    theta_w = float(tau_pos - 0.05) if theta_w is None else float(theta_w)
    theta_x = float((tau_pos + tau_neg) / 2.0) if theta_x is None else float(theta_x)

    base_info = {
        'method': 'graph', 'abstained': True, 'n_components': 0,
        'background_size': 0, 'components': [],
        's_edge': s_edge, 'theta_w': theta_w, 'theta_x': theta_x,
        'mutual_knn_k': int(mutual_knn_k), 'r_spatial': r_spatial,
        'm_min': int(m_min),
    }
    if n < int(min_patches):
        return np.zeros(n, dtype=np.int64), base_info

    # Rows should already be unit-norm; assert rather than silently renormalize.
    norms = np.linalg.norm(z, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            'graph_components_decode: z rows must be L2-normalized '
            f'(norm range [{norms.min():.4f}, {norms.max():.4f}]).'
        )

    sim = z @ z.T                                    # (N, N) cosine
    np.fill_diagonal(sim, -np.inf)                   # exclude self for kNN

    # Mutual-kNN mask.
    k = max(1, min(int(mutual_knn_k), n - 1))
    knn_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    knn = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    knn[rows, knn_idx.reshape(-1)] = True
    mutual = knn & knn.T

    np.fill_diagonal(sim, 1.0)                        # restore for thresholds/stats
    edges = (sim >= s_edge) & mutual
    np.fill_diagonal(edges, False)

    if r_spatial is not None:
        hw = grid_hw or _infer_grid_hw(n)
        if hw is None:
            raise ValueError(
                f'graph_components_decode: r_spatial set but grid_hw is None and '
                f'N={n} is not square — pass grid_hw explicitly.'
            )
        h, w = hw
        rr = np.repeat(np.arange(h), w)
        cc = np.tile(np.arange(w), h)
        cheb = np.maximum(np.abs(rr[:, None] - rr[None, :]),
                          np.abs(cc[:, None] - cc[None, :]))
        edges &= (cheb <= int(r_spatial))

    labels = _union_find_components(edges)
    comp_ids, comp_sizes = np.unique(labels, return_counts=True)

    # Background = largest component (deterministic tie-break: lowest min index).
    max_size = int(comp_sizes.max())
    tied = [int(c) for c, s in zip(comp_ids, comp_sizes) if int(s) == max_size]
    if len(tied) == 1:
        bg_id = tied[0]
    else:
        bg_id = min(tied, key=lambda c: int(np.where(labels == c)[0].min()))

    if attention_polarity and attention is not None:
        a = np.asarray(attention, dtype=np.float64).reshape(-1)
        if a.shape[0] == n:
            big = [int(c) for c, s in zip(comp_ids, comp_sizes)
                   if int(s) >= 0.2 * n]
            if len(big) >= 2:
                bg_id = min(big, key=lambda c: float(a[labels == c].mean()))

    bg_mask = labels == bg_id
    bg_size = int(bg_mask.sum())

    def _within(idx: np.ndarray) -> float:
        if idx.size < 2:
            return 1.0
        sub = sim[np.ix_(idx, idx)]
        iu = np.triu_indices(idx.size, k=1)
        return float(sub[iu].mean())

    components: List[Dict] = []
    accept = np.zeros(n, dtype=bool)
    bg_idx = np.where(bg_mask)[0]
    for c, sz in zip(comp_ids.tolist(), comp_sizes.tolist()):
        if c == bg_id:
            continue
        idx = np.where(labels == c)[0]
        if idx.size < int(m_min):
            continue
        within = _within(idx)
        cross = (float(sim[np.ix_(idx, bg_idx)].mean())
                 if bg_idx.size else 0.0)
        accepted = bool(within >= theta_w and cross <= theta_x)
        components.append({
            'comp_id': int(c),
            'size': int(idx.size), 'within': within, 'cross': cross,
            'margin': within - cross, 'accepted': accepted,
        })
        if accepted:
            accept[idx] = True

    mask = accept.astype(np.int64)
    info = dict(base_info)
    info.update({
        'abstained': bool(mask.sum() == 0),
        'n_components': int(comp_ids.size),
        'background_size': bg_size,
        'background_id': int(bg_id),
        'labels': labels.copy(),
        'components': components,
        'n_accepted': int(sum(c['accepted'] for c in components)),
    })
    return mask, info


# ── Decode dispatcher ─────────────────────────────────────────────────────────
#
# A single ``DecodeSpec`` threads through the eval suites. Default = k-means, so
# existing behavior is byte-identical unless a caller opts into the graph decode.

@dataclasses.dataclass(frozen=True)
class DecodeSpec:
    """Selects and parameterizes the patch decode used across the eval suites."""
    method: str = 'kmeans'          # 'kmeans' | 'graph'
    tau_pos: float = 0.55
    tau_neg: float = 0.20
    n_init: int = 4                 # k-means restarts
    s_edge: Optional[float] = None
    mutual_knn_k: int = 10
    r_spatial: Optional[int] = None
    m_min: int = 4
    theta_w: Optional[float] = None
    theta_x: Optional[float] = None
    attention_polarity: bool = False

    def _graph_kwargs(self) -> Dict:
        return dict(
            tau_pos=self.tau_pos, tau_neg=self.tau_neg, s_edge=self.s_edge,
            mutual_knn_k=self.mutual_knn_k, r_spatial=self.r_spatial,
            m_min=self.m_min, theta_w=self.theta_w, theta_x=self.theta_x,
            attention_polarity=self.attention_polarity,
        )


def decode_oracle_labels(
    z: np.ndarray,
    spec: DecodeSpec = DecodeSpec(),
    *,
    grid_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """2-labeling for the ORACLE-polarity metric path.

    The headline localization metric scores ``_oracle_polarity(labels, gt)`` —
    the labeling and its complement, best-by-F1 — so absolute label identity
    doesn't matter, only the partition. Returns (N,) int in {0, 1}.

    - kmeans: ``spherical_kmeans2`` labels (arbitrary polarity).
    - graph:  1 = accepted foreground, 0 = everything else. An abstain is all-0,
              which oracle-polarity scores as an empty prediction (costs recall).
    """
    if spec.method == 'graph':
        mask, _ = graph_components_decode(z, grid_hw=grid_hw, **spec._graph_kwargs())
        return mask.astype(np.int64)
    raw_labels, _ = spherical_kmeans2(z, n_init=spec.n_init)
    return raw_labels


def decode_deploy_mask(
    z: np.ndarray,
    spec: DecodeSpec = DecodeSpec(),
    *,
    attention: Optional[np.ndarray] = None,
    grid_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, dict]:
    """Committed foreground (splice) mask for DEPLOYMENT — no GT available.

    Returns (mask_bool (N,), info).

    - kmeans: polarity from BCE attention (higher-attention cluster = splice);
      falls back to the smaller-cluster rule when ``attention`` is None. This
      reproduces the existing ``_select_cluster`` deployment heuristic.
    - graph:  the accepted-components mask, already committed (no re-polarization
      — re-applying an attention rule here could invert it).
    """
    if spec.method == 'graph':
        mask, info = graph_components_decode(
            z, grid_hw=grid_hw, attention=attention, **spec._graph_kwargs()
        )
        return mask.astype(bool), info
    raw_labels, _ = spherical_kmeans2(z, n_init=spec.n_init)
    if attention is not None:
        a = np.asarray(attention, dtype=np.float64).reshape(-1)
        m0 = (raw_labels == 0); m1 = (raw_labels == 1)
        a0 = float(a[m0].mean()) if m0.any() else -np.inf
        a1 = float(a[m1].mean()) if m1.any() else -np.inf
        splice_label = 0 if a0 >= a1 else 1
    else:
        n0 = int((raw_labels == 0).sum()); n1 = int((raw_labels == 1).sum())
        splice_label = 0 if n0 <= n1 else 1
    mask = (raw_labels == splice_label)
    return mask, {'method': 'kmeans', 'splice_label': int(splice_label)}


def calibrate_graph_decode(
    z_list: Sequence[np.ndarray],
    gt_is_single: Sequence[bool],
    *,
    tau_pos: float,
    tau_neg: float,
    s_edge_grid: Optional[np.ndarray] = None,
    base_spec: Optional[DecodeSpec] = None,
    fallback_s_edge: Optional[float] = None,
) -> Tuple[float, dict]:
    """Calibrate the graph decode's ``s_edge`` on a calibration set.

    Mirrors ``calibrate_gate_tau``: sweep ``s_edge`` and pick the value that
    maximizes balanced accuracy of abstain-vs-single (abstain = empty mask) on
    ``gt_is_single``. Returns (best_s_edge, info).
    """
    from lab_utils.errors import EvalError

    if len(z_list) == 0:
        if fallback_s_edge is None:
            raise EvalError(
                'calibrate_graph_decode: calibration set is empty and '
                'fallback_s_edge is None.'
            )
        return float(fallback_s_edge), {
            'per_s_edge': [], 'best_s_edge': float(fallback_s_edge),
            'best_balanced_acc': float('nan'), 'used_fallback': True,
        }

    if s_edge_grid is None:
        s_edge_grid = np.linspace(tau_neg + 0.05, tau_pos - 0.05, 21)
    base = base_spec or DecodeSpec(method='graph', tau_pos=tau_pos, tau_neg=tau_neg)
    gt = np.asarray(gt_is_single, dtype=bool)
    n_pos = int(gt.sum())
    n_neg = int((~gt).sum())

    best_s, best_score = float(s_edge_grid[0]), -np.inf
    best_tpr = best_tnr = float('nan')
    per_s_edge = []
    for s_edge in s_edge_grid:
        spec = dataclasses.replace(base, method='graph', s_edge=float(s_edge))
        pred_single = np.array([
            graph_components_decode(z, grid_hw=_infer_grid_hw(z.shape[0]),
                                    **spec._graph_kwargs())[0].sum() == 0
            for z in z_list
        ], dtype=bool)
        tpr = float((pred_single & gt).sum()) / n_pos if n_pos > 0 else 1.0
        tnr = float((~pred_single & ~gt).sum()) / n_neg if n_neg > 0 else 1.0
        bacc = 0.5 * (tpr + tnr)
        per_s_edge.append((float(s_edge), bacc, tpr, tnr))
        if bacc > best_score:
            best_score, best_s, best_tpr, best_tnr = bacc, float(s_edge), tpr, tnr

    return best_s, {
        'per_s_edge': per_s_edge,
        'best_s_edge': best_s,
        'best_balanced_acc': best_score,
        'best_tpr': best_tpr,
        'best_tnr': best_tnr,
        'n_single': n_pos,
        'n_two_class': n_neg,
        's_edge_grid_min': float(np.min(s_edge_grid)),
        's_edge_grid_max': float(np.max(s_edge_grid)),
    }


def polarity_attn(raw_labels: np.ndarray, attention: Optional[np.ndarray]) -> np.ndarray:
    """Cluster with higher mean attention is splice.

    If attention is None, falls back to the smaller cluster (legacy default).
    """
    raw = np.asarray(raw_labels).reshape(-1)
    n0 = int((raw == 0).sum())
    n1 = int((raw == 1).sum())
    if attention is None:
        chosen = 0 if n0 <= n1 else 1
        return (raw == chosen).astype(bool)
    att = np.asarray(attention).reshape(-1)
    mean0 = float(att[raw == 0].mean()) if n0 else float("-inf")
    mean1 = float(att[raw == 1].mean()) if n1 else float("-inf")
    chosen = 0 if mean0 >= mean1 else 1
    return (raw == chosen).astype(bool)

