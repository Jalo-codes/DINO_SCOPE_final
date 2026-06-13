# graph_lab — a compartmentalized bench for the graph-components decode

Twist the knobs on the **calibrated-graph decode** (`graph_components_decode`,
the connected-components-over-similarity-graph replacement for k-means(2)) and
*see* where it helps — without re-running the model every time.

The whole point: **dump once, sweep forever.** One GPU pass freezes real
embeddings to a `.npz`; after that every experiment is pure numpy + PIL on the
cache (no model, no dataset, instant).

```
graph_lab/
  dump_embeddings.py   # run ONCE: model → cache/<run>.npz  (z, attention, GT, thumbnails)
  sandbox.py           # run ANY number of times: cache → labelled PNG composites
  cache/               # the .npz dumps live here (gitignored-friendly)
```

## 1. Dump (once per checkpoint, needs GPU + data)

```bash
python -m graph_lab.dump_embeddings \
    --ckpt /content/drive/MyDrive/DINO_SCOPE_RUNS/<run>/epoch_006.pt \
    --imd2020_root /content/IMD2020 --casia_root /content/casia \
    --casia_train --imd_val_only \
    --tau_pos 0.55 --tau_neg 0.20 \
    --n_items 20 --out graph_lab/cache/e006.npz
```

Caches `z` (L2-normalized contrastive embeddings — the exact decode input),
per-patch BCE attention, the GT mask, and a square thumbnail, plus the run's
`tau_pos/tau_neg` so the sandbox defaults match the training margins.

## 2. Sandbox (instant, numpy + PIL only)

**Single setting** — one composite per image,
`Original | GT | K-means | Graph | Graph+spatial`. Graph panels are coloured per
component (green=accepted, red=rejected, gray=sub-`m_min`) and labelled with the
decode's own reasoning + IoU vs GT. Prints a k-means-vs-graph median-IoU line and
a win count.

```bash
python -m graph_lab.sandbox --cache graph_lab/cache/e006.npz \
    --out graph_lab/out/baseline --s_edge 0.375 --knn 10 --spatial 2
```

**Sweep one knob** — `Original | GT | <a panel per value>` per image, plus a
stdout table of median/mean IoU and abstain-rate at each setting so the knee is
obvious. Sweepable: `s_edge`, `mutual_knn_k`, `r_spatial`, `m_min`, `theta_w`,
`theta_x`, `tau_pos`, `tau_neg`.

```bash
python -m graph_lab.sandbox --cache graph_lab/cache/e006.npz \
    --out graph_lab/out/sweep_sedge --sweep s_edge \
    --sweep_vals 0.30 0.34 0.38 0.42 0.46
```

## Knobs (all map to `graph_components_decode` / `DecodeSpec`)

| flag | meaning | default |
|------|---------|---------|
| `--s_edge` | absolute cosine bar for an edge | mid-band `(tau_pos+tau_neg)/2` |
| `--knn` | mutual-kNN k (anti-chaining) | 10 |
| `--spatial` | Chebyshev radius for the `Graph+spatial` panel | 2 (0=skip) |
| `--m_min` | min component size to score | 4 |
| `--theta_w` | component acceptance: cohesion floor | `tau_pos - 0.05` |
| `--theta_x` | component acceptance: sim-to-background ceiling | mid-band |
| `--tau_pos/--tau_neg` | trained margins | from cache |

`None`/unset → the decode's own calibrated default, so leaving a flag off is the
honest baseline. See `GRAPH_DECODE_PLAN.md` for the full formulation.

> Same heavy deps as `scripts/viz_decode.py` (torch only for the dump; PIL+numpy
> for the sandbox). Runs on Colab where the checkpoints live.
