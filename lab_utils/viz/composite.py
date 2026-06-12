"""Visualization helpers: colormaps, heatmap overlays, composite grids.

PIL-only — no matplotlib dependency. All functions operate on numpy arrays
and PIL Images.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Colormap ─────────────────────────────────────────────────────────────────

def cmap_jet(v: float) -> Tuple[int, int, int]:
    """Map a scalar in [0, 1] to an RGB tuple.  Blue → Cyan → Green → Yellow → Red."""
    v = max(0.0, min(1.0, v))
    if v < 0.25:
        t = v / 0.25
        return (0, int(t * 255), 255)
    elif v < 0.5:
        t = (v - 0.25) / 0.25
        return (0, 255, int((1 - t) * 255))
    elif v < 0.75:
        t = (v - 0.5) / 0.25
        return (int(t * 255), 255, 0)
    else:
        t = (v - 0.75) / 0.25
        return (255, int((1 - t) * 255), 0)


# ── Heatmap / overlay helpers ────────────────────────────────────────────────

def heatmap_rgb(grid: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """Convert a 2-D float grid to an RGB heatmap, nearest-upscaled.

    Args:
        grid: (rows, cols) float array.  Normalised to [0, 1] internally.
        size_hw: target (height, width) for the output.

    Returns:
        (H, W, 3) uint8 numpy array.
    """
    g = np.asarray(grid, dtype=np.float64)
    lo, hi = float(g.min()), float(g.max())
    g = (g - lo) / (hi - lo + 1e-12)
    rows, cols = g.shape
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            rgb[r, c] = cmap_jet(float(g[r, c]))
    return np.asarray(
        Image.fromarray(rgb).resize((size_hw[1], size_hw[0]), Image.NEAREST),
        dtype=np.uint8,
    )


def overlay_blend(
    base: np.ndarray, heat: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Alpha-blend a heatmap onto a base image.  Both must be (H, W, 3) uint8."""
    return np.clip(
        (1 - alpha) * base.astype(np.float32) + alpha * heat.astype(np.float32),
        0, 255,
    ).astype(np.uint8)


def mask_tint(
    base: np.ndarray,
    mask_2d: np.ndarray,
    size_hw: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a boolean 2-D mask as a coloured tint on the original image.

    Positive pixels get ``color`` blended at ``alpha``; negative pixels pass
    through unchanged.
    """
    mask_up = (
        np.asarray(
            Image.fromarray(mask_2d.astype(np.uint8) * 255).resize(
                (size_hw[1], size_hw[0]), Image.NEAREST,
            ),
            dtype=np.uint8,
        )
        > 127
    )
    out = base.copy()
    c = np.array(color, dtype=np.float32)
    out[mask_up] = np.clip(
        (1 - alpha) * base[mask_up].astype(np.float32) + alpha * c, 0, 255
    ).astype(np.uint8)
    return out


# ── Composite grid saving ───────────────────────────────────────────────────

def _load_font(size: int = 13) -> ImageFont.ImageFont:
    """Try common Linux font paths; fall back to the built-in default."""
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeMono.ttf',
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def save_composite(
    panels: List[Tuple[str, np.ndarray]],
    path: str,
    *,
    panel_size: int = 280,
    cols: int = 4,
) -> None:
    """Save a labelled grid of image panels as a single PNG.

    Args:
        panels: list of ``(label_str, rgb_numpy_uint8)`` tuples.
        path: output file path (directories created automatically).
        panel_size: pixel size of each square tile.
        cols: number of columns in the grid.
    """
    n = len(panels)
    rows = (n + cols - 1) // cols
    has_newline = any('\n' in label for label, _ in panels)
    label_h = 36 if has_newline else 22
    cell_h = panel_size + label_h
    cell_w = panel_size
    canvas = Image.new('RGB', (cols * cell_w, rows * cell_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font = _load_font()

    for i, (label, px) in enumerate(panels):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, r * cell_h
        draw.text((x0 + 4, y0 + 3), label, fill=(230, 230, 230), font=font)
        tile = Image.fromarray(px).resize(
            (panel_size, panel_size), Image.BILINEAR
        )
        canvas.paste(tile, (x0, y0 + label_h))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    canvas.save(path, quality=95)
