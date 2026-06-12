"""lab_utils.eval.partition — spherical k-means and silhouette gate.

Lifted from contrastive_test/core/partition.py (no functional changes).
"""

from typing import Optional, Sequence, Tuple

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
