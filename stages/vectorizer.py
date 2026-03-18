"""
Stage 5 — Vectorization

Converts the simplified contour polygons into a scalable SVG file using
cubic Bezier curves derived from Catmull-Rom splines.

Why Catmull-Rom → Bezier?
  A polyline drawn through the RDP-simplified vertices produces visible
  "kinks" at each control point.  The Catmull-Rom spline passes smoothly
  through every knot and its conversion to cubic Bezier form is exact and
  cheap.  SVG's cubic Bezier command (C) gives browser/renderer/Inkscape
  full anti-aliased curve rendering with no extra libraries required.

SVG layer order:
  Regions are drawn in the order supplied (largest first).  Each region's
  colour is used as the fill; no stroke is applied so adjacent regions
  share clean boundaries without visible outlines.
"""

from pathlib import Path
import sys

import numpy as np
import svgwrite

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Tuple
from utils.bezier import (
    catmull_rom_to_bezier,
    segments_to_svg_path,
    polyline_to_svg_path,
)


def contours_to_svg(
    contour_data: List[Tuple[List[np.ndarray], np.ndarray, int]],
    width: int,
    height: int,
    output_path: str,
    use_curves: bool = True,
    tension: float = 1.0,
    precision: int = 2,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Write all colour regions to an SVG file.

    Args:
        contour_data:     output of extract_all_contours()
        width, height:    canvas dimensions in pixels
        output_path:      destination .svg file path
        use_curves:       True → Catmull-Rom Bezier; False → straight polylines
        tension:          Catmull-Rom tension (1.0 = natural smoothness)
        precision:        decimal places in SVG coordinates
        background_color: canvas background (R, G, B)
    """
    bg_r, bg_g, bg_b = background_color
    bg_hex = f"rgb({bg_r},{bg_g},{bg_b})"

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{width}px", f"{height}px"),
        viewBox=f"0 0 {width} {height}",
        profile="full",
    )

    # Background rectangle
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=bg_hex))

    # Draw each colour region (largest → smallest, painter's algorithm)
    for contours, color, _ in contour_data:
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        fill_color = f"rgb({r},{g},{b})"

        for pts in contours:
            if len(pts) < 3:
                continue

            if use_curves and len(pts) >= 4:
                pts_f = pts.astype(np.float64)
                segments = catmull_rom_to_bezier(pts_f, closed=True, tension=tension)
                path_data = segments_to_svg_path(segments, precision=precision)
            else:
                path_data = polyline_to_svg_path(pts)

            if path_data:
                dwg.add(
                    dwg.path(
                        d=path_data,
                        fill=fill_color,
                        stroke="none",
                        fill_rule="evenodd",
                    )
                )

    dwg.save(pretty=True)
