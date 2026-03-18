"""
Stage 3 — Mask Generation

Creates a binary mask for each colour cluster and applies morphological
cleanup to remove noise caused by JPEG compression artefacts.

Why morphological cleanup?
  JPEG block boundaries create isolated "wrong-cluster" pixels at region
  edges.  A closing operation (dilation then erosion) seals small gaps;
  an opening operation (erosion then dilation) removes small isolated blobs.
  Together they give clean, contiguous masks suitable for contour tracing.
"""

from typing import List, Tuple

import cv2
import numpy as np


def generate_masks(
    labels: np.ndarray,
    centers_rgb: np.ndarray,
    min_area_ratio: float = 0.0001,
    morph_kernel_size: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Generate and clean binary masks for every colour cluster.

    Args:
        labels:           (H, W) int32 cluster label array
        centers_rgb:      (N, 3) uint8 cluster colours in RGB
        min_area_ratio:   discard clusters covering less than this fraction
                          of the total image area
        morph_kernel_size: size of the elliptical structuring element

    Returns:
        List of (mask, color, cluster_id) tuples sorted by covered area,
        largest region first (important for correct painter-algorithm layering).
    """
    h, w = labels.shape
    total_area = h * w
    min_area = max(int(total_area * min_area_ratio), 4)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morph_kernel_size, morph_kernel_size),
    )

    results: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for cluster_id in range(len(centers_rgb)):
        # Binary mask: 255 where this cluster, 0 elsewhere
        mask = (labels == cluster_id).astype(np.uint8) * 255

        # Morphological cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        covered = int(np.count_nonzero(mask))
        if covered < min_area:
            continue

        results.append((mask, centers_rgb[cluster_id].copy(), cluster_id))

    # Sort: largest area first so background layers paint behind foreground
    results.sort(key=lambda x: int(np.count_nonzero(x[0])), reverse=True)
    return results
