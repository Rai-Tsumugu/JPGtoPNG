#!/usr/bin/env python3
"""
JPG -> SVG Vector Converter
============================

Converts a JPEG logo/graphic to an SVG by bitmap tracing.

Backend priority (first available is used):
  1. vtracer    - pip install vtracer  (recommended, Rust-based, fast)
  2. Potrace    - system binary `potrace` must be on PATH
  3. OpenCV     - contour-based fallback (always available)

Optional --split-text mode
--------------------------
For logos that contain a symbol mark on top and text on the bottom:
  - The top portion is traced as vector paths.
  - The bottom portion is replaced with an SVG <text> element rendered
    with a web font (Montserrat Light by default), which gives pixel-perfect
    sharp text that scales without degradation.

Usage examples
--------------
  python convert_svg.py logo.jpg
  python convert_svg.py logo.jpg -o logo.svg
  python convert_svg.py logo.jpg --colormode color
  python convert_svg.py logo.jpg --split-text "Hokkaido University Metaverse Club"
  python convert_svg.py logo.jpg --split-text "My Club" --text-ratio 0.65
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def binarize(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to a binary (0/255) grayscale array via Otsu threshold,
    after mild sharpening to strengthen edges before tracing.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(gray, -1, kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ─────────────────────────────────────────────────────────────────────────────
# Backend: vtracer
# ─────────────────────────────────────────────────────────────────────────────

def _try_vtracer(
    input_path: str,
    output_path: str,
    colormode: str = "binary",
) -> bool:
    """Try vtracer backend. Returns True on success."""
    try:
        import vtracer  # noqa: PLC0415
    except ImportError:
        return False
    try:
        vtracer.convert_image_to_svg_py(
            input_path,
            output_path,
            colormode=colormode,
            hierarchical="stacked",
            mode="spline",
            filter_speckle=4,
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3,
        )
        return True
    except Exception as exc:
        print(f"  [!] vtracer failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Backend: Potrace
# ─────────────────────────────────────────────────────────────────────────────

def _try_potrace(binary_img: np.ndarray, output_path: str) -> bool:
    """
    Write the binary image as a PBM temp file and call `potrace -s`.
    Returns True on success.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".pbm", delete=False) as tf:
            pbm_path = tf.name
        # PBM: P4 binary format - Potrace works best with PBM
        Image.fromarray(binary_img).save(pbm_path)
        result = subprocess.run(
            ["potrace", "-s", "-o", output_path, pbm_path],
            capture_output=True,
            timeout=60,
        )
        Path(pbm_path).unlink(missing_ok=True)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    except Exception as exc:
        print(f"  [!] Potrace failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Backend: OpenCV contour fallback
# ─────────────────────────────────────────────────────────────────────────────

def _opencv_to_svg(rgb: np.ndarray, output_path: str) -> None:
    """
    Contour-based SVG fallback using OpenCV + svgwrite.
    Traces the binary-thresholded image to polyline paths.
    """
    import svgwrite  # noqa: PLC0415

    h, w = rgb.shape[:2]
    binary = binarize(rgb)

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{w}px", f"{h}px"),
        viewBox=f"0 0 {w} {h}",
    )
    dwg.add(dwg.rect((0, 0), (w, h), fill="black"))

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue
        eps = 0.002 * cv2.arcLength(cnt, closed=True)
        simplified = cv2.approxPolyDP(cnt, eps, closed=True)
        pts = simplified.reshape(-1, 2)
        if len(pts) < 3:
            continue
        path_d = "M " + " L ".join(f"{x},{y}" for x, y in pts) + " Z"
        dwg.add(dwg.path(d=path_d, fill="white"))

    dwg.save()


# ─────────────────────────────────────────────────────────────────────────────
# Split-text SVG helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_split_text_svg(
    mark_svg_path: str,
    text: str,
    total_width: int,
    total_height: int,
    mark_height: int,
    output_path: str,
    font_family: str = "Montserrat, 'Open Sans', sans-serif",
    font_weight: str = "300",
    text_color: str = "white",
    bg_color: str = "black",
) -> None:
    """
    Combine the traced mark SVG with an SVG <text> element for the text portion.

    The mark SVG paths are read, clipped to [0, mark_height], and embedded.
    A centered <text> element is placed in the lower region.
    """
    # Read the mark SVG and extract the path content
    mark_svg_content = Path(mark_svg_path).read_text(encoding="utf-8")

    # Extract the inner content of the mark SVG (between <svg …> and </svg>)
    import re
    inner = re.sub(r"<\?xml[^>]*\?>", "", mark_svg_content)
    inner = re.sub(r"<svg[^>]*>", "", inner, count=1)
    inner = re.sub(r"</svg>", "", inner)

    # Scale mark paths to fit the mark_height portion of the full canvas
    text_region_h = total_height - mark_height
    text_y_center = mark_height + text_region_h * 0.55
    font_size = max(10, int(text_region_h * 0.38))

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{total_width}" height="{total_height}"
     viewBox="0 0 {total_width} {total_height}">
  <defs>
    <clipPath id="mark-clip">
      <rect x="0" y="0" width="{total_width}" height="{mark_height}"/>
    </clipPath>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300&amp;display=swap');
    </style>
  </defs>
  <!-- Background -->
  <rect x="0" y="0" width="{total_width}" height="{total_height}" fill="{bg_color}"/>
  <!-- Symbol mark (traced vector) -->
  <g clip-path="url(#mark-clip)">
    {inner.strip()}
  </g>
  <!-- Text -->
  <text
    x="{total_width // 2}"
    y="{text_y_center:.1f}"
    font-family="{font_family}"
    font-weight="{font_weight}"
    font-size="{font_size}"
    fill="{text_color}"
    text-anchor="middle"
    dominant-baseline="middle"
    letter-spacing="2"
  >{text}</text>
</svg>
"""
    Path(output_path).write_text(svg, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def convert(
    input_path: str,
    output_path: str,
    colormode: str = "binary",
    split_text: str | None = None,
    text_ratio: float = 0.70,
    verbose: bool = True,
) -> str:
    """
    Convert a JPEG to SVG using the best available backend.

    Args:
        input_path:  Source JPEG.
        output_path: Destination SVG.
        colormode:   "binary" (default) or "color" - passed to vtracer.
        split_text:  If set, the bottom (1-text_ratio) fraction of the image
                     is replaced with a <text> SVG element containing this string.
                     The top portion is traced as vector paths.
        text_ratio:  Fraction of image height used for the symbol mark (default 0.70).
        verbose:     Print progress messages.

    Returns:
        Name of the backend used: "vtracer", "potrace", or "opencv".
    """
    img = Image.open(input_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    h, w = rgb.shape[:2]

    if verbose:
        print(f"  Input : {input_path}  ({w}x{h})")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Split-text mode: crop just the mark portion for tracing ───────────────
    trace_input = input_path
    mark_height = h
    mark_tmp_path: str | None = None

    if split_text is not None:
        mark_height = int(h * text_ratio)
        mark_rgb = rgb[:mark_height, :]
        mark_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        mark_tmp_path = mark_tmp.name
        mark_tmp.close()
        Image.fromarray(mark_rgb).save(mark_tmp_path)
        trace_input = mark_tmp_path
        if verbose:
            print(f"  Split-text mode: mark region = top {mark_height}px, "
                  f"text = \"{split_text}\"")

    # ── Try backends ──────────────────────────────────────────────────────────
    # For split-text we need an intermediate SVG for the mark, then combine.
    if split_text is not None:
        mark_svg_tmp = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
        mark_svg_path = mark_svg_tmp.name
        mark_svg_tmp.close()
    else:
        mark_svg_path = output_path

    backend = "opencv"

    if verbose:
        print("  Trying vtracer …")
    if _try_vtracer(trace_input, mark_svg_path, colormode=colormode):
        backend = "vtracer"
    else:
        mark_rgb_arr = np.array(Image.open(trace_input).convert("RGB"), dtype=np.uint8)
        binary = binarize(mark_rgb_arr)

        if verbose:
            print("  Trying Potrace …")
        if _try_potrace(binary, mark_svg_path):
            backend = "potrace"
        else:
            if verbose:
                print("  Using OpenCV contour fallback …")
            _opencv_to_svg(mark_rgb_arr, mark_svg_path)
            backend = "opencv"

    # ── Combine mark SVG + text element ──────────────────────────────────────
    if split_text is not None:
        _build_split_text_svg(
            mark_svg_path=mark_svg_path,
            text=split_text,
            total_width=w,
            total_height=h,
            mark_height=mark_height,
            output_path=output_path,
        )
        Path(mark_svg_path).unlink(missing_ok=True)
        if mark_tmp_path:
            Path(mark_tmp_path).unlink(missing_ok=True)

    if verbose:
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"  Output: {output_path}  ({size_kb:.1f} KB)  [backend: {backend}]")

    return backend


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_svg",
        description="JPG -> SVG vector conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Input JPEG file")
    p.add_argument("-o", "--output", default=None,
                   help="Output SVG path (default: <stem>.svg)")

    g = p.add_argument_group("Tracing options")
    g.add_argument(
        "--colormode", choices=["binary", "color"], default="binary",
        help="Tracing mode: binary=black-and-white (default), color=multi-color",
    )

    gt = p.add_argument_group("Split-text options (for mark+text logos)")
    gt.add_argument(
        "--split-text", default=None, metavar="TEXT",
        help="Replace the bottom text region with an SVG <text> element. "
             "Supply the text string, e.g. \"Hokkaido University Metaverse Club\"",
    )
    gt.add_argument(
        "--text-ratio", type=float, default=0.70, metavar="R",
        help="Fraction of image height dedicated to the symbol mark (default: 0.70). "
             "Remainder becomes the text region.",
    )

    p.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: file not found - {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output or str(
        Path(args.input).parent / (Path(args.input).stem + ".svg")
    )

    convert(
        args.input, output,
        colormode=args.colormode,
        split_text=args.split_text,
        text_ratio=args.text_ratio,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
