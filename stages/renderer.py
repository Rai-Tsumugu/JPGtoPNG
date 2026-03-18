"""
Stage 6 — PNG Re-rendering

Converts the vector representation back to a raster PNG image.

Two rendering paths are provided:

  1. Direct OpenCV render (default, no extra dependencies)
     Draws filled polygons on a canvas using cv2.fillPoly.
     Supersampling (render_scale > 1) renders at N× resolution then
     downsamples with INTER_AREA for high-quality anti-aliasing.

  2. cairosvg render (optional, highest quality)
     Uses the Cairo 2D graphics library through the cairosvg Python
     binding to rasterize the SVG produced by stage 5.
     Requires: pip install cairosvg  AND  native Cairo library.

The two methods produce visually identical results; cairosvg is
marginally smoother on very thin features.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Direct OpenCV renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_direct(
    contour_data: List[Tuple[List[np.ndarray], np.ndarray, int]],
    width: int,
    height: int,
    scale: int = 2,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Rasterize vector regions directly using OpenCV filled polygons.

    Args:
        contour_data:  output of extract_all_contours()
        width, height: final output dimensions
        scale:         supersampling factor (1–4 recommended)
        background:    canvas fill colour (R, G, B)

    Returns:
        (H, W, 3) uint8 RGB image array.
    """
    scale = max(1, scale)
    sw, sh = width * scale, height * scale

    # OpenCV uses BGR internally
    bg_bgr = (int(background[2]), int(background[1]), int(background[0]))
    canvas = np.full((sh, sw, 3), bg_bgr, dtype=np.uint8)

    # Draw in reverse order: smallest region on top (foreground), largest behind
    # contour_data is already sorted largest-first, so we draw in that order
    # (each new polygon paints over what came before — painter's algorithm)
    for contours, color, _ in contour_data:
        bgr = (int(color[2]), int(color[1]), int(color[0]))  # RGB → BGR

        for pts in contours:
            if len(pts) < 3:
                continue
            scaled = (pts * scale).astype(np.int32)
            cv2.fillPoly(canvas, [scaled], bgr)

    # Downsample with INTER_AREA for high-quality anti-aliasing
    if scale > 1:
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# cairosvg renderer (optional)
# ─────────────────────────────────────────────────────────────────────────────

def render_from_svg(
    svg_path: str,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """
    Rasterize an SVG file to a numpy array using cairosvg.

    Returns None if cairosvg is not installed, so callers can fall back
    to render_direct without crashing.
    """
    try:
        import cairosvg
        import io

        png_bytes = cairosvg.svg2png(
            url=str(Path(svg_path).resolve()),
            output_width=width,
            output_height=height,
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def save_png(
    image: np.ndarray,
    output_path: str,
    optimize: bool = True,
) -> None:
    """Save an RGB uint8 array as a lossless PNG."""
    img = Image.fromarray(image.astype(np.uint8))
    save_kwargs = {"format": "PNG"}
    if optimize:
        save_kwargs["optimize"] = True
    img.save(output_path, **save_kwargs)
    print(f"    -> saved: {output_path}")
