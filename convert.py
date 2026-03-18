#!/usr/bin/env python3
"""
Faithful JPG -> PNG/SVG Converter
===================================

Two-mode replacement for the AI-heavy pipeline.py:

  Mode 1 - Faithful PNG
    Straight JPEG decode (no bilateral filter, no K-means, no supersampling).
    Optionally removes the background using flood-fill from image corners.
    Output is a transparent-background RGBA PNG.

  Mode 2 - SVG Vector
    High-quality bitmap tracing for scalable logo output.
    Backend priority: vtracer (pip install vtracer) -> Potrace -> OpenCV.
    Optional --split-text mode embeds the text portion as a crisp SVG <text>
    element with a web font instead of tracing low-resolution pixels.

Processing flow
---------------
  [Input JPG]
      |
      +---> [PNG module] background flood-fill only -> output_faithful.png
      |
      +---> [SVG module] binarize -> trace -> optional text inject -> output.svg

Usage examples
--------------
  python convert.py logo.jpg                        # both PNG and SVG
  python convert.py logo.jpg --png-only             # PNG only
  python convert.py logo.jpg --svg-only             # SVG only
  python convert.py logo.jpg --tolerance 40         # wider BG tolerance
  python convert.py logo.jpg --no-bg                # PNG, no transparency
  python convert.py logo.jpg --colormode color      # color SVG trace
  python convert.py logo.jpg \\
      --split-text "Hokkaido University Metaverse Club" \\
      --text-ratio 0.68                             # mark+text SVG
  python convert.py logo.jpg -o out/               # write to out/ directory
"""

import argparse
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_output_dir(input_path: Path, output_arg: str | None) -> Path:
    if output_arg is None:
        return input_path.parent
    p = Path(output_arg)
    # If it looks like a directory (no suffix, or ends with / or \)
    if not p.suffix or str(output_arg).endswith(("/", "\\")):
        return p
    # Treat as directory anyway (user probably meant -o out/)
    return p if p.is_dir() else p.parent if p.suffix else p


def _parse_bg_color(value: str) -> tuple:
    try:
        parts = [int(x.strip()) for x in value.split(",")]
        if len(parts) != 3:
            raise ValueError
        return tuple(parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Background color must be R,G,B integers e.g. 45,45,45 - got: {value!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Error: file not found - {args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = _resolve_output_dir(inp, args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = inp.stem
    verbose = not args.quiet

    run_png = not args.svg_only
    run_svg = not args.png_only

    sep = "=" * 54
    if verbose:
        print(f"\n{sep}")
        print(f"  Faithful JPG -> PNG/SVG Converter")
        print(f"  Input  : {inp}")
        print(f"  Output : {out_dir}/")
        print(f"{sep}")

    # ── PNG ───────────────────────────────────────────────────────────────────
    if run_png:
        if verbose:
            print("\n[1/2] Faithful PNG - background-transparent conversion")
            print("-" * 54)

        from convert_faithful import convert as convert_png  # noqa: PLC0415

        png_path = str(out_dir / f"{stem}_faithful.png")
        bg_color = _parse_bg_color(args.bg_color) if args.bg_color else None
        mode = "none" if args.no_bg else args.bg_mode

        convert_png(
            str(inp), png_path,
            tolerance=args.tolerance,
            mode=mode,
            bg_color=bg_color,
            verbose=verbose,
        )

    # ── SVG ───────────────────────────────────────────────────────────────────
    if run_svg:
        if verbose:
            step = "2/2" if run_png else "1/1"
            print(f"\n[{step}] SVG - vector tracing")
            print("-" * 54)

        from convert_svg import convert as convert_svg  # noqa: PLC0415

        svg_path = str(out_dir / f"{stem}.svg")
        backend = convert_svg(
            str(inp), svg_path,
            colormode=args.colormode,
            split_text=args.split_text,
            text_ratio=args.text_ratio,
            verbose=verbose,
        )
        if verbose:
            print(f"  Backend: {backend}")

    if verbose:
        print(f"\n{sep}")
        print("  Done.")
        print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert",
        description="Faithful JPG -> PNG/SVG Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Input JPEG file")
    p.add_argument(
        "-o", "--output", default=None,
        help="Output directory (default: same directory as input)",
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    gm = p.add_argument_group("Output mode")
    ex = gm.add_mutually_exclusive_group()
    ex.add_argument("--png-only", action="store_true",
                    help="Produce only the faithful PNG (skip SVG)")
    ex.add_argument("--svg-only", action="store_true",
                    help="Produce only the SVG (skip PNG)")

    # ── PNG options ───────────────────────────────────────────────────────────
    gp = p.add_argument_group("PNG background-removal options")
    gp.add_argument(
        "--tolerance", type=int, default=30, metavar="T",
        help="Color tolerance 0-255 for background detection (default: 30)",
    )
    gp.add_argument(
        "--bg-mode", choices=["flood", "pixel"], default="flood", dest="bg_mode",
        help="flood=flood-fill from corners (default), pixel=per-pixel distance",
    )
    gp.add_argument(
        "--no-bg", action="store_true",
        help="Skip background removal - output opaque PNG",
    )
    gp.add_argument(
        "--bg-color", default=None, metavar="R,G,B",
        help="Override background color detection, e.g. 45,45,45 "
             "(only effective with --bg-mode pixel)",
    )

    # ── SVG options ───────────────────────────────────────────────────────────
    gs = p.add_argument_group("SVG tracing options")
    gs.add_argument(
        "--colormode", choices=["binary", "color"], default="binary",
        help="binary=black-and-white trace (default), color=multi-color trace",
    )

    gt = p.add_argument_group("SVG split-text options (for mark+text logos)")
    gt.add_argument(
        "--split-text", default=None, metavar="TEXT",
        help="Replace the lower text region with a crisp SVG <text> element. "
             "Example: --split-text \"Hokkaido University Metaverse Club\"",
    )
    gt.add_argument(
        "--text-ratio", type=float, default=0.70, metavar="R",
        help="Fraction of image height for the symbol mark (default: 0.70). "
             "The remaining fraction becomes the text region.",
    )

    # ── Global ────────────────────────────────────────────────────────────────
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    return p


if __name__ == "__main__":
    main()
