"""
Stage 4 — Contour Extraction

Finds the boundaries of each colour region and simplifies them using the
Ramer-Douglas-Peucker (RDP) algorithm before vectorization.

Why RDP?
  Raw contours from cv2.findContours can have thousands of vertices per
  region due to pixel-level detail and JPEG noise.  RDP removes vertices
  that lie within ε of the straight line between their neighbours, reducing
  point count by 80-95 % while preserving the visual boundary shape.
  A sensible ε is chosen proportional to the contour's arc length so it
  scales with image size automatically.
"""

from pathlib import Path
import sys

import cv2
import numpy as np

# Ensure project root is importable from within the stages/ sub-package
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Tuple
from utils.bezier import rdp_simplify


# ─────────────────────────────────────────────────────────────────────────────
# Single-mask contour extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_contours(
    mask: np.ndarray,
    min_area: int = 50,
    epsilon_ratio: float = 0.002,
) -> List[np.ndarray]:
    """
    Extract simplified outer contours from a binary mask.

    Args:
        mask:          uint8 binary mask (0 / 255)
        min_area:      discard contours with pixel area < this value
        epsilon_ratio: RDP ε as a fraction of the contour's arc length

    Returns:
        List of (N, 2) float32 arrays — one per contour.
    """
    raw_contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    result: List[np.ndarray] = []

    for cnt in raw_contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        # Squeeze OpenCV's (N,1,2) layout → (N,2)
        pts = cnt.squeeze()
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        if len(pts) < 4:
            continue

        # RDP simplification via cv2.approxPolyDP for speed
        if epsilon_ratio > 0:
            arc = cv2.arcLength(cnt, closed=True)
            eps = epsilon_ratio * arc
            approx = cv2.approxPolyDP(cnt, eps, closed=True)
            pts = approx.squeeze()
            if pts.ndim == 1:
                pts = pts.reshape(1, 2)

        if len(pts) < 3:
            continue

        result.append(pts.astype(np.float32))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline step: all masks → contour lists
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_contours(
    masks: List[Tuple[np.ndarray, np.ndarray, int]],
    image_area: int,
    min_area_ratio: float = 0.00005,
    epsilon_ratio: float = 0.002,
) -> List[Tuple[List[np.ndarray], np.ndarray, int]]:
    """
    Run contour extraction for every mask produced by stage 3.

    Args:
        masks:           output of generate_masks()
        image_area:      H × W (for relative minimum area threshold)
        min_area_ratio:  minimum contour area relative to image area
        epsilon_ratio:   RDP simplification aggressiveness

    Returns:
        List of (contours, color, cluster_id) — preserving the
        largest-first ordering from the mask stage.
    """
    min_area = max(int(image_area * min_area_ratio), 50)
    results: List[Tuple[List[np.ndarray], np.ndarray, int]] = []

    for mask, color, cluster_id in masks:
        contours = extract_contours(
            mask,
            min_area=min_area,
            epsilon_ratio=epsilon_ratio,
        )
        if contours:
            results.append((contours, color, cluster_id))

    return results
