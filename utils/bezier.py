"""
Bezier curve math utilities for contour vectorization.

Implements:
  - Catmull-Rom spline → cubic Bezier conversion
  - RDP (Ramer-Douglas-Peucker) polyline simplification
  - SVG path data generation
"""

import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# RDP Simplification
# ─────────────────────────────────────────────────────────────────────────────

def _perp_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from point to line segment a→b."""
    ab = b - a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab)) / ab_sq
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker polyline simplification.

    Args:
        points: (N, 2) array of polyline vertices
        epsilon: maximum allowed perpendicular deviation

    Returns:
        Simplified (M, 2) array (M ≤ N).
    """
    if len(points) < 3:
        return points

    # Find the point farthest from the chord start→end
    dists = np.array([
        _perp_distance(points[i], points[0], points[-1])
        for i in range(1, len(points) - 1)
    ])
    max_idx = int(np.argmax(dists)) + 1
    max_dist = dists[max_idx - 1]

    if max_dist > epsilon:
        left = rdp_simplify(points[: max_idx + 1], epsilon)
        right = rdp_simplify(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([points[0], points[-1]])


# ─────────────────────────────────────────────────────────────────────────────
# Catmull-Rom → Cubic Bezier
# ─────────────────────────────────────────────────────────────────────────────

def catmull_rom_to_bezier(
    points: np.ndarray,
    closed: bool = True,
    tension: float = 1.0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convert a Catmull-Rom spline to a list of cubic Bezier segments.

    For each segment from P[i] to P[i+1], the Bezier control points are:
        CP1 = P[i]   + (P[i+1] - P[i-1]) / 6 * tension
        CP2 = P[i+1] - (P[i+2] - P[i]  ) / 6 * tension

    Args:
        points:  (N, 2) array of spline knots
        closed:  True if the spline forms a closed loop
        tension: scaling factor for tangent length (1.0 = standard)

    Returns:
        List of (P0, CP1, CP2, P1) tuples for each segment.
    """
    n = len(points)
    if n < 2:
        return []

    segments: List[Tuple] = []
    count = n if closed else n - 1

    for i in range(count):
        p0 = points[(i - 1) % n]
        p1 = points[i % n]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]

        cp1 = p1 + (p2 - p0) / 6.0 * tension
        cp2 = p2 - (p3 - p1) / 6.0 * tension

        segments.append((p1.copy(), cp1.copy(), cp2.copy(), p2.copy()))

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# SVG Path Data Generation
# ─────────────────────────────────────────────────────────────────────────────

def segments_to_svg_path(
    segments: List[Tuple],
    precision: int = 2,
) -> str:
    """
    Convert a list of cubic Bezier segments to an SVG path data string.

    Format: "M x0,y0 C cx1,cy1 cx2,cy2 x1,y1 C ... Z"
    """
    if not segments:
        return ""

    def fmt(v: np.ndarray) -> str:
        return f"{round(float(v[0]), precision)},{round(float(v[1]), precision)}"

    parts = [f"M {fmt(segments[0][0])}"]
    for _, cp1, cp2, end in segments:
        parts.append(f"C {fmt(cp1)} {fmt(cp2)} {fmt(end)}")
    parts.append("Z")

    return " ".join(parts)


def polyline_to_svg_path(points: np.ndarray) -> str:
    """
    Convert a polyline point array to an SVG path data string (L commands).
    Used as fallback when curve generation is disabled.
    """
    if len(points) == 0:
        return ""
    parts = [f"M {points[0][0]},{points[0][1]}"]
    for pt in points[1:]:
        parts.append(f"L {pt[0]},{pt[1]}")
    parts.append("Z")
    return " ".join(parts)
