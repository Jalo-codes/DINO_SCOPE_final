"""lab_utils.model — detector backbones (ImageBCEDetector,
MultiHeadDetector) and loss functions."""

from lab_utils.model.image_bce_detector import ImageBCEDetector
from lab_utils.model.multi_head_detector import (
    MultiHeadDetector,
    build_multi_head_detector,
)

__all__ = [
    'ImageBCEDetector',
    'MultiHeadDetector', 'build_multi_head_detector',
]
