# Calibrated-Graph Decode — implementation plan

Replace k-means(2) patch decoding with a connected-components decode over a
thresholded similarity graph. The contrastive loss trains an explicit, calibrated
geometry — same-region pairs ≥ `TAU_POS`, cross-region pairs ≤ `TAU_NEG` — leaving
a wide trained dead band between the two pair populations. K-means ignores those
margins and *must* split every image into 2 clusters, so when no margin-respecting
split exists it partitions along semantics (the observed failure: large blobs
swallowing background / surrounding objects, catastrophic output when uncertain).
The graph decode thresholds pairwise similarity *inside the trained band*, so:

- joining the splice component requires an **absolute** similarity bar (a pair the
  training pushed to ≤ TAU_NEG can't clear it) — bleed-resistant by construction;
- the number of components falls out automatically (multi-region for free);
- zero foreign components = natural abstention (replaces the silhouette gate);
- fully deterministic — no RNG, no n_init restarts, no prototypes, no attention.

**Explicit non-goals for this change:** no training/loss changes (the `TAU_POS`
0.55→0.60 bump is a separate retrain via the existing `--tau_pos` flag); no
attention-derived background prototypes; do not remove or modify the existing
k-means path — the new decode is added *alongside* it for honest comparison.

---

## 1. Exact formulation

Inputs, per image (or per window):

- `Z ∈ R^{N×D}` — L2-normalized per-patch contrastive embeddings; the **same
  array** currently passed to `spherical_kmeans2` (D = 128, N = H·W patches;
  28×28 = 784 at 448px / patch 16). Rows MUST be unit-norm (assert, don't renorm).
- Grid coords `g_i = (i // W, i % W)`.
- Trained margins `tau_pos` (currently 0.55), `tau_neg` (0.20) — read from config,
  never hardcoded.

**Step 1 — similarity matrix**

    S = Z @ Z.T            # (N, N), S_ij = cosine similarity

**Step 2 — edge set.** Let `N_k(i)` = indices of the k largest `S_ij, j ≠ i`.
Undirected edge (i, j) exists iff ALL of:

    (a) S_ij >= s_edge                          # calibrated threshold in the dead band
    (b) j in N_k(i)  AND  i in N_k(j)           # mutual-kNN (anti-chaining)
    (c) max(|g_i.r - g_j.r|, |g_i.c - g_j.c|) <= r_spatial    # OPTIONAL, off by default

Defaults: `s_edge = (tau_pos + tau_neg) / 2` (= 0.375 at current margins; this is
the prior — calibration sweeps it, §4), `k = 10`, `r_spatial = None` (off; the
spatial-constrained variant `r_spatial ∈ {1, 2}` is evaluated as a separate
strategy in the comparison script).

**Step 3 — components.** `{C_1 … C_m}` = connected components of the edge graph.
Pure-python/numpy union-find (scipy is NOT a repo dependency — do not add it).
Isolated nodes are singleton components.

**Step 4 — background.** `B = argmax_t |C_t|` (tie → lowest patch index;
deterministic). Optional flag `attention_polarity` for the >50%-splice regime:
among components with `|C_t| >= 0.2·N`, take the one with the LOWEST mean
per-patch BCE attention as background (same rationale as `_select_cluster` in
`lab_utils/eval/localization.py:42`). Default OFF.

**Step 5 — per-component statistics**, for every `C_t ≠ B` with `|C_t| >= m_min`:

    within(C_t) = mean_{i<j in C_t} S_ij                    # internal cohesion
    cross(C_t)  = mean_{i in C_t, j in B} S_ij              # similarity to background
    margin(C_t) = within(C_t) - cross(C_t)

**Step 6 — acceptance.** `C_t` is a foreign-region prediction iff:

    within(C_t) >= theta_w        # default: tau_pos - 0.05
    cross(C_t)  <= theta_x        # default: (tau_pos + tau_neg) / 2

The cross test is what kills "fragmented background masquerading as a splice":
two components can lack edges (mutual-kNN starvation) yet still have high mean
cross-similarity — those are background shards, rejected here.

**Step 7 — output.** `mask[i] = 1` iff `i ∈ union of accepted components`;
all-zeros = predicted single-class (abstain). Return `(mask, info)` where `info`
carries: per-component `(size, within, cross, margin, accepted)`, background
size, `n_components`, `abstained`, and the parameters used.

Complexity: one N×N matmul + O(N²) masking ≈ trivial at N = 784. No RNG.

Edge cases: `N < 16` → abstain. Variable N (zoom/window grids) must work — take
`grid_hw=(H, W)` as an argument, never assume 28×28. Singletons excluded by
`m_min` (default 4 ≈ 0.5% of a 784-grid).

---

## 2. Phase 1 — core decode (`lab_utils/eval/partition.py`)

New function, signature mirroring `partition_image` (same module, same
`(mask, info)` contract so call sites can swap):

```
def graph_components_decode(
    z: np.ndarray,                # (N, D) L2-normalized
    *,
    grid_hw: Tuple[int, int],
    tau_pos: float,
    tau_neg: float,
    s_edge: Optional[float] = None,      # None → (tau_pos + tau_neg) / 2
    mutual_knn_k: int = 10,
    r_spatial: Optional[int] = None,
    m_min: int = 4,
    theta_w: Optional[float] = None,     # None → tau_pos - 0.05
    theta_x: Optional[float] = None,     # None → (tau_pos + tau_neg) / 2
    attention: Optional[np.ndarray] = None,
    attention_polarity: bool = False,
) -> Tuple[np.ndarray, dict]
```

Implementation notes for the implementer:

- Union-find: plain array-based with path compression, ~20 lines, numpy indices.
- Mutual-kNN via `np.argpartition` per row; build a boolean kNN mask and AND with
  its transpose.
- Match the module's existing style (numpy-only, docstring with the math,
  diagnostics dict).

**Unit tests** (new file `lab_utils/tests/test_graph_decode.py`, synthetic
fixtures, no data/GPU markers):

1. *Planted splice*: 760 background vectors near one direction + 24 splice
   vectors near an orthogonal direction (unit-normalized, small noise) → decode
   recovers exactly the planted indices; `info` shows 1 accepted component.
2. *Single-class*: all 784 vectors from one mode → abstains (all-zero mask).
3. *Anti-chaining*: two tight clusters plus a thin bridge of intermediate
   vectors below `s_edge` to each side → still 2 components, no merge.
4. *Fragmented background*: background split into two shards with high mutual
   cross-sim (> theta_x) but no edges → shard is REJECTED by the cross test.
5. *Two planted splices* → two accepted components (multi-region).
6. *Determinism*: two calls, identical output; no RNG consumed.
7. *Non-square / small grids*: works at (14, 14) and (7, 10); N < 16 abstains.

---

## 3. Phase 2 — comparison eval

**Script** `contrastive_inpainting_v1/scripts/eval_graph_decode.py`, modeled
directly on `swin_outlier_decode.py` (same checkpoint loading, item building,
deterministic subsampling, views):

- Strategies compared on identical samples: `kmeans` (reference =
  `spherical_kmeans2` + silhouette gate, exactly the current path), `graph`
  (defaults above), `graph_spatial` (`r_spatial=2`), and `graph` at ±0.05
  `s_edge` offsets (sensitivity row).
- Views: `FULL_FRAME` at minimum; `SWIN_IMAGE` / `BEST_CAP_WIN` reuse the
  window plumbing from `swin_outlier_decode.py` if time allows (flag-gated).
- Polarity: report both as-emitted (background = largest component) and
  oracle-flip (`_oracle_polarity` convention, `localization.py:84`) so numbers
  are comparable with every existing k-means table.
- Metrics per (split, area-tier, strategy): pixel-level IoU / F1 / precision /
  recall. **Reporting style (required):** median-led with mean alongside, full
  percentiles (p5/q1/med/q3/p95), reals pooled separately from splices. Plus an
  abstention table: false-abstain rate on splices, correct-abstain rate on
  reals (this is the gate-replacement story — compare against the silhouette
  gate at its calibrated tau).

**Calibration** — `calibrate_graph_decode(z_list, gt_is_single, gt_masks, ...)`
next to `calibrate_gate_tau` (`partition.py:164`), same idiom: sweep `s_edge`
over `np.linspace(tau_neg + 0.05, tau_pos - 0.05, 21)` (optionally a small
`theta_x` grid), objective = balanced accuracy of abstain-vs-single (as today)
tie-broken by median splice F1. Uses the existing calibration split
(`CALIBRATION_FRAC` etc. in `configs/base.py`).

---

## 4. Phase 3 — zoom-verify ratchet (proposal/verifier loop)

For each accepted component (these are the proposals):

1. Split into spatial 8-connected pixel blobs; drop blobs < `m_min`.
2. Blob → padded bbox crop (reuse the `_minority_bbox` + crop idiom from
   `localization.py:1050` / `collect_coarse_to_fine_samples`).
3. Re-embed the crop; get image-head probability `p_zoom`.
4. **Ratchet rule:** keep the component iff `p_zoom >= p_full - eps`
   (`eps = 0.0` default). If every component is dropped, fall back to the
   full-frame decode mask unchanged — refinement may never make the output
   worse than the coarse pass by the detector's own judgment.

Integrate as a new mode in the coarse-to-fine path
(`collect_coarse_to_fine_samples` / `report_coarse_to_fine`) so the existing
report directly shows kmeans-c2f vs graph-c2f.

## 5. Phase 4 (later, separate PR) — swin integration

Per-window `graph_components_decode` inside the sliding-window pass
(`diagnose/passes/swin.py`, `lab_utils/eval/sliding_window.py`), OR-composited
as today. Windows abstain individually — this is the property k-means lacked
that made swin noisy. Do not start until Phase 2 numbers justify it.

---

## 6. Acceptance criteria

On the val comparison (Phase 2, same checkpoint, same samples):

1. Median pixel **precision** on splices improves vs k-means on small + medium
   area tiers (this is the large-blob/bleed failure being targeted).
2. Median IoU overall not worse than k-means by more than noise (graph decode
   must not trade recall catastrophically for precision).
3. Correct-abstain rate on reals ≥ silhouette gate at its calibrated tau, at
   equal-or-lower false-abstain on splices.
4. All unit tests green; `pytest` from repo root passes; no new dependencies.
5. Runtime: full-frame decode < 50 ms/image on CPU at N=784 (it's one matmul —
   anything slower means something is wrong).

## 7. Hand-off notes / gotchas

- `z` must be the **contrastive head** output already L2-normalized — same
  tensor the `spherical_kmeans2` call sites use (`localization.py:487`,
  `collect_coarse_to_fine_samples`). Assert unit norms; don't silently renorm.
- `tau_pos`/`tau_neg` come from the run config. After the planned `--tau_pos
  0.60` retrain the decode's defaults shift automatically — that's intended
  (training and decode share the same margins; do not freeze 0.375 anywhere).
- Largest-component-as-background fails for >50% splices; that's what the
  `attention_polarity` flag is for. Keep it OFF in headline numbers; report it
  as a variant row.
- No scipy, no sklearn. Numpy only, matching the module.
- Keep the k-means path byte-identical. Every existing report must reproduce.
