#!/usr/bin/env python3
"""
Faithful JPG -> PNG Converter with Background Transparency
==========================================================

Converts a JPEG to PNG without any AI reconstruction, filtering, or
supersampling - only a plain JPEG decode followed by optional background
removal via flood-fill from the image corners.

Usage examples
--------------
  python convert_faithful.py logo.jpg
  python convert_faithful.py logo.jpg -o logo_clean.png
  python convert_faithful.py logo.jpg --tolerance 40
  python convert_faithful.py logo.jpg --bg-mode pixel --bg-color 45,45,45
  python convert_faithful.py logo.jpg --no-bg          # no transparency
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Background removal helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_background_color(rgb: np.ndarray, patch: int = 8) -> np.ndarray:
    """
    Estimate background color by averaging pixels in all 4 corners.
    Returns an (3,) uint8 array.
    """
    h, w = rgb.shape[:2]
    p = max(1, min(patch, h // 4, w // 4))
    patches = [
        rgb[:p, :p],
        rgb[:p, w - p:],
        rgb[h - p:, :p],
        rgb[h - p:, w - p:],
    ]
    colors = np.vstack([pa.reshape(-1, 3) for pa in patches])
    return np.round(colors.mean(axis=0)).astype(np.uint8)


def remove_background_flood(rgb: np.ndarray, tolerance: int = 30) -> np.ndarray:
    """
    Flood-fill from all 4 corners to build a background mask, then set those
    pixels to alpha=0.

    This preserves interior pixels that happen to share the background color
    (e.g. dark fill inside a logo shape) because they are not reachable from
    the corners through a connected flood fill.

    Returns an RGBA uint8 array.
    """
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    combined = np.zeros((h, w), dtype=np.uint8)

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    flags = cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

    for cy, cx in corners:
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        temp = bgr.copy()
        cv2.floodFill(
            temp, mask, (cx, cy), 0,
            loDiff=(tolerance,) * 3,
            upDiff=(tolerance,) * 3,
            flags=flags,
        )
        combined = np.maximum(combined, mask[1:-1, 1:-1])

    alpha = np.where(combined > 0, 0, 255).astype(np.uint8)
    return np.dstack([rgb, alpha])


def remove_background_pixel(
    rgb: np.ndarray,
    bg_color: np.ndarray,
    tolerance: int = 30,
) -> np.ndarray:
    """
    Per-pixel background removal: every pixel whose max channel distance
    from bg_color is within `tolerance` is made transparent.

    Simpler than flood fill but may accidentally erase interior pixels that
    match the background color.

    Returns an RGBA uint8 array.
    """
    dist = np.abs(rgb.astype(np.int32) - bg_color.astype(np.int32)).max(axis=2)
    alpha = np.where(dist <= tolerance, 0, 255).astype(np.uint8)
    return np.dstack([rgb, alpha])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def convert(
    input_path: str,
    output_path: str,
    tolerance: int = 30,
    mode: str = "flood",
    bg_color: tuple | None = None,
    verbose: bool = True,
) -> None:
    """
    Convert a JPEG to a transparent-background PNG without any filtering.

    Args:
        input_path:  Source JPEG.
        output_path: Destination PNG.
        tolerance:   Color tolerance for background matching (0–255).
        mode:        "flood"  - flood fill from corners (recommended)
                     "pixel"  - per-pixel distance to bg_color
                     "none"   - no background removal (opaque PNG)
        bg_color:    (R, G, B) tuple to override auto-detected background.
                     Only used when mode="pixel".
        verbose:     Print progress messages.
    """
    # Load with Pillow - no OpenCV decompression artifacts, no color shifts
    img = Image.open(input_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)

    if verbose:
        print(f"  Input : {input_path}  ({img.width}x{img.height})")

    if mode == "none":
        out_img = img  # keep as RGB, no alpha
    elif mode == "pixel":
        if bg_color is None:
            bg = sample_background_color(rgb)
            if verbose:
                print(f"  Background (auto-sampled) : RGB{tuple(bg.tolist())}")
        else:
            bg = np.array(bg_color, dtype=np.uint8)
            if verbose:
                print(f"  Background (manual)       : RGB{tuple(bg)}")
        rgba = remove_background_pixel(rgb, bg, tolerance)
        out_img = Image.fromarray(rgba, mode="RGBA")
    else:  # flood (default)
        if verbose:
            print(f"  Mode: flood-fill from corners  (tolerance={tolerance})")
        rgba = remove_background_flood(rgb, tolerance)
        out_img = Image.fromarray(rgba, mode="RGBA")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path, "PNG")

    if verbose:
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"  Output: {output_path}  ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_faithful",
        description="Faithful JPG -> PNG with optional background transparency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Input JPEG file")
    p.add_argument("-o", "--output", default=None,
                   help="Output PNG path (default: <stem>_faithful.png)")

    g = p.add_argument_group("Background removal")
    g.add_argument(
        "--tolerance", type=int, default=30, metavar="T",
        help="Color tolerance 0-255 for background matching (default: 30)",
    )
    g.add_argument(
        "--bg-mode", choices=["flood", "pixel", "none"], default="flood",
        dest="bg_mode",
        help="Removal method: flood=flood-fill from corners (default), "
             "pixel=per-pixel distance, none=keep opaque",
    )
    g.add_argument(
        "--no-bg", action="store_true",
        help="Shorthand for --bg-mode none",
    )
    g.add_argument(
        "--bg-color", default=None, metavar="R,G,B",
        help="Force background color, e.g. 45,45,45 (only used with --bg-mode pixel)",
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
        Path(args.input).parent / (Path(args.input).stem + "_faithful.png")
    )

    bg_color = None
    if args.bg_color:
        try:
            parts = [int(x.strip()) for x in args.bg_color.split(",")]
            if len(parts) != 3:
                raise ValueError
            bg_color = tuple(parts)
        except ValueError:
            print("Error: --bg-color must be three integers: R,G,B  e.g. 45,45,45",
                  file=sys.stderr)
            sys.exit(1)

    mode = "none" if args.no_bg else args.bg_mode

    convert(
        args.input, output,
        tolerance=args.tolerance,
        mode=mode,
        bg_color=bg_color,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
