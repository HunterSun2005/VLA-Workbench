"""Camera image geometry shared by model and ROS processes; no Torch imports."""
from __future__ import annotations

import numpy as np


def resize_rgb_with_pad(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Fit the whole RGB frame into H×W, preserving aspect ratio with black padding.

    Downsample with INTER_AREA; upsample with INTER_LINEAR. Center the image,
    placing an odd extra padding pixel on the bottom/right. Return owned,
    contiguous HWC uint8 data without changing the input.
    """
    import cv2

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected HWC uint8 RGB image, got {rgb.shape}/{rgb.dtype}")
    source_height, source_width = rgb.shape[:2]
    if source_height == 0 or source_width == 0:
        raise ValueError("camera image dimensions must be positive")
    if any(isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size <= 0
           for size in (height, width)):
        raise ValueError("target image dimensions must be positive integers")
    scale = min(height / source_height, width / source_width)
    resized_height = max(1, min(height, round(source_height * scale)))
    resized_width = max(1, min(width, round(source_width * scale)))
    if (resized_height, resized_width) != (source_height, source_width):
        rgb = cv2.resize(rgb, (resized_width, resized_height),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - resized_height) // 2, (width - resized_width) // 2
    output[top:top + resized_height, left:left + resized_width] = rgb
    return output
