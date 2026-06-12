"""diagnose — clean rewrite of splice-localization diagnostics.

Design contract (single source of truth, read this before changing anything):

  1. ONE F1 DEFINITION. `metrics.f1_pixel(pred_HxW, gt_HxW)` is called by every
     pass.  Predictions are 28x28 patch grids projected to original-resolution
     pixel masks via NN-expansion.  GT is the original-resolution mask.  Pixels
     outside the pass's operating region are forced False (so any GT splice
     missed by a crop is FN, no silent inflation).

  2. APPLES-TO-APPLES ROWS. Every image (splice OR real) gets the same ROW_KEYS
     populated.  Missing values are explicit NaN, never absent.  `schema._
     validate_row(row)` enforces this before any summary runs.

  3. EXPLICIT ORACLE SCOPING. Each metric column carries its oracle-knowledge
     suffix:
       _pure   = no GT used
       _ceil   = k-means cluster identity picked by F1-max vs GT (polarity oracle)

  4. PASSES PRODUCE ROW UPDATES, NOT METRICS DIRECTLY. A pass writes its
     prediction (as a full-image pixel mask) and any pass-local diagnostics
     (e.g. swin per-window category counts) into the row.  F1s are computed
     centrally from the pixel mask.

  5. NO CAPPING OF WINDOWS. `max_windows_grid` is removed from swin.

  6. STRIDE/SCALE BUG SAFEGUARD. The header emits a hash of the generated
     window-coord set for each (scale, stride) combo on a probe image and
     asserts all hashes distinct.

  7. LOG TRANSPARENCY. Header is self-describing: every pass id, every metric
     definition, every threshold, every area_tier boundary, every crop area,
     every window-set hash.  Someone reading the log alone should be able to
     reconstruct the experiment.
"""

__all__ = []
