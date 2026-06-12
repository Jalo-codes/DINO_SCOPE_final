"""lab_utils.data.paste — regional compositing for splice simulation.

Two entry points:
    paste_regional_ae          — paste AE-reconstructed pixels into a region
    paste_regional_degradation — paste algorithmically-degraded pixels into a region

Both accept a soft float mask (values in [0, 1]) so experiments can use the
raw output of generate_blob_mask without thresholding, preserving edge
realism.  A hard binary PIL 'L' mask is also accepted (auto-converted).
"""

from typing import Callable, Union
import numpy as np
from PIL import Image

from lab_utils.errors import DataError
from lab_utils.data.blob import paste_soft_alpha


def _to_float_alpha(mask: Union[Image.Image, np.ndarray],
                    expected_hw: tuple) -> np.ndarray:
    """Normalise mask to float32 (H, W) array in [0, 1]."""
    if isinstance(mask, Image.Image):
        arr = np.array(mask, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
    else:
        arr = np.asarray(mask, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

    if arr.ndim == 3:
        arr = arr[..., 0]  # (H, W, 1) → (H, W)

    if arr.shape != expected_hw:
        raise DataError(
            f"paste: mask shape {arr.shape} does not match image H×W {expected_hw}."
        )
    return arr


def paste_regional_ae(
    img: Image.Image,
    ae_recon: Image.Image,
    mask: Union[Image.Image, np.ndarray],
) -> Image.Image:
    """Composite AE-reconstructed pixels into img within the mask region.

    Pixels where mask=1 come from ae_recon; mask=0 keeps the original img.
    Soft mask values (0 < alpha < 1) produce a smooth boundary blend.

    Args:
        img:      Original clean PIL RGB image (H × W).
        ae_recon: AE reconstruction PIL RGB image — must be same size as img.
        mask:     Soft float mask (H × W) in [0, 1] or hard PIL 'L' mask.

    Returns:
        Composited PIL RGB image.

    Raises:
        DataError: If img and ae_recon sizes differ, or mask has wrong shape.
    """
    if img.size != ae_recon.size:
        raise DataError(
            f"paste_regional_ae: img.size={img.size} != ae_recon.size={ae_recon.size}."
        )
    H, W = img.size[1], img.size[0]
    alpha = _to_float_alpha(mask, (H, W))

    composited = paste_soft_alpha(
        background=np.array(img,      dtype=np.uint8),
        foreground=np.array(ae_recon, dtype=np.uint8),
        alpha=alpha,
    )
    return Image.fromarray(composited)


def paste_regional_degradation(
    img: Image.Image,
    mask: Union[Image.Image, np.ndarray],
    degradation_fn: Callable[[Image.Image], Image.Image],
) -> Image.Image:
    """Apply degradation_fn to img then composite result into img within mask.

    degradation_fn is applied to the *entire* image first (so global
    frequency statistics are intact), then only the masked region is kept.

    Args:
        img:             Original PIL RGB image.
        mask:            Soft float mask (H × W) in [0, 1] or hard PIL 'L' mask.
        degradation_fn:  Callable: PIL Image → PIL Image.  Must preserve size.

    Returns:
        Composited PIL RGB image.

    Raises:
        DataError: If degradation_fn changes the image size, or mask has
                   wrong shape.
    """
    H, W = img.size[1], img.size[0]
    alpha = _to_float_alpha(mask, (H, W))

    degraded = degradation_fn(img)
    if degraded.size != img.size:
        raise DataError(
            f"paste_regional_degradation: degradation_fn changed size "
            f"{img.size} → {degraded.size}.  Degradation must preserve size."
        )

    composited = paste_soft_alpha(
        background=np.array(img,      dtype=np.uint8),
        foreground=np.array(degraded, dtype=np.uint8),
        alpha=alpha,
    )
    return Image.fromarray(composited)
