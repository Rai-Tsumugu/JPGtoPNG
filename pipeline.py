#!/usr/bin/env python3
"""
JPEG → Complete PNG Restoration Pipeline
=========================================

Research-level pipeline for reconstructing a clean, artefact-free PNG from
a JPEG source image.  Every stage is parameterizable and produces an
intermediate output for inspection.

Pipeline stages
---------------
  1  Preprocessing       Bilateral + median denoising to reduce DCT artefacts
  2  Color Clustering    MiniBatch K-means in CIE Lab space → flat-colour labels
  3  Mask Generation     Per-cluster binary masks + morphological cleanup
  4  Contour Extraction  cv2.findContours + RDP simplification
  5  Vectorization       Catmull-Rom Bezier curves → SVG
  6  Re-rendering        SVG back to PNG via supersampled OpenCV fill

Usage examples
--------------
  python pipeline.py photo.jpg
  python pipeline.py photo.jpg -o clean.png --clusters 32
  python pipeline.py photo.jpg --clusters 16 --epsilon 0.001 --scale 3
  python pipeline.py photo.jpg --no-curves            # straight polylines
  python pipeline.py photo.jpg --no-intermediates -q  # silent, minimal output
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── project imports ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import PipelineConfig
from utils.image_utils import (
    load_image,
    save_image,
    preprocess_jpeg,
    compute_ssim,
    compute_psnr,
    resize_to_match,
)
from stages.clustering import cluster_colors, merge_similar_clusters, quantized_image
from stages.masking import generate_masks
from stages.contour_extractor import extract_all_contours
from stages.vectorizer import contours_to_svg
from stages.renderer import render_direct, render_from_svg, save_png


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def _stage(name: str, step: int, total: int = 6, verbose: bool = True) -> None:
    if verbose:
        print(f"  [{step}/{total}] {name} …")


def run_pipeline(
    input_path: str,
    output_path: str,
    config: PipelineConfig,
    verbose: bool = True,
) -> dict:
    """
    Execute the full JPEG → PNG pipeline.

    Returns:
        dict with timing (seconds) and quality metrics for each stage.
    """
    t_start = time.perf_counter()
    metrics: dict = {}

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    def _save_inter(tag: str, arr: np.ndarray) -> None:
        if config.save_intermediates:
            path = str(out_dir / f"{stem}_{tag}.png")
            save_image(arr, path)
            if verbose:
                print(f"         -> {path}")

    if verbose:
        sep = "-" * 52
        print(f"\n{sep}")
        print(f"  JPEG -> PNG Pipeline")
        print(f"  input : {input_path}")
        print(f"  output: {output_path}")
        print(f"{sep}")

    # ── Load ──────────────────────────────────────────────────────────────────
    original = load_image(input_path)
    h, w = original.shape[:2]
    out_w = config.output_width or w
    out_h = config.output_height or h

    if verbose:
        print(f"  Image size: {w}x{h}  ->  output: {out_w}x{out_h}\n")

    # ── Stage 1: Preprocessing ────────────────────────────────────────────────
    t0 = time.perf_counter()
    _stage("Preprocessing (JPEG denoising)", 1, verbose=verbose)

    image = preprocess_jpeg(original, denoise_strength=config.denoise_strength)
    _save_inter("1_preprocessed", image)

    metrics["t_preprocess"] = time.perf_counter() - t0

    # ── Stage 2: Color Clustering ─────────────────────────────────────────────
    t0 = time.perf_counter()
    _stage(f"Color clustering  k={config.n_clusters}", 2, verbose=verbose)

    labels, centers = cluster_colors(image, n_clusters=config.n_clusters)

    if config.merge_threshold > 0:
        labels, centers = merge_similar_clusters(
            labels, centers, distance_threshold=config.merge_threshold
        )

    n_actual = len(centers)
    if verbose:
        print(f"       {n_actual} clusters after merge (threshold={config.merge_threshold})")

    q_img = quantized_image(labels, centers)
    _save_inter("2_quantized", q_img)

    metrics["t_clustering"] = time.perf_counter() - t0
    metrics["n_clusters"] = n_actual

    # ── Stage 3: Mask Generation ──────────────────────────────────────────────
    t0 = time.perf_counter()
    _stage("Mask generation", 3, verbose=verbose)

    masks = generate_masks(
        labels,
        centers,
        min_area_ratio=config.min_area_ratio,
        morph_kernel_size=config.morph_kernel_size,
    )

    if verbose:
        print(f"       {len(masks)} significant colour regions")

    # Save a composite mask overview (optional debug image)
    if config.save_intermediates:
        debug_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        import cv2
        for i, (m, col, _) in enumerate(masks):
            debug_canvas[m > 0] = col
        _save_inter("3_masks_composite", debug_canvas)

    metrics["t_masking"] = time.perf_counter() - t0
    metrics["n_regions"] = len(masks)

    # ── Stage 4: Contour Extraction ───────────────────────────────────────────
    t0 = time.perf_counter()
    _stage("Contour extraction + RDP simplification", 4, verbose=verbose)

    contour_data = extract_all_contours(
        masks,
        image_area=h * w,
        epsilon_ratio=config.epsilon_ratio,
    )

    n_total_contours = sum(len(c[0]) for c in contour_data)
    if verbose:
        print(f"       {n_total_contours} contours across {len(contour_data)} regions")

    metrics["t_contours"] = time.perf_counter() - t0
    metrics["n_contours"] = n_total_contours

    # ── Stage 5: Vectorization ────────────────────────────────────────────────
    t0 = time.perf_counter()
    _stage("Vectorization -> SVG", 5, verbose=verbose)

    svg_path = str(out_dir / f"{stem}_vector.svg")
    contours_to_svg(
        contour_data,
        width=out_w,
        height=out_h,
        output_path=svg_path,
        use_curves=config.use_curves,
        tension=config.tension,
        precision=config.svg_precision,
        background_color=config.background_color,
    )

    svg_size_kb = Path(svg_path).stat().st_size / 1024
    if verbose:
        print(f"       -> {svg_path}  ({svg_size_kb:.1f} KB)")

    metrics["t_vectorize"] = time.perf_counter() - t0
    metrics["svg_size_kb"] = svg_size_kb

    # ── Stage 6: Re-rendering ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    _stage("Re-rendering -> PNG", 6, verbose=verbose)

    output_image: np.ndarray | None = None

    if config.use_cairosvg:
        output_image = render_from_svg(svg_path, out_w, out_h)
        if output_image is None and verbose:
            print("       [!] cairosvg not available, falling back to direct render")

    if output_image is None:
        method = "supersampled OpenCV"
        if verbose:
            print(f"       method: {method}  scale={config.render_scale}x")
        output_image = render_direct(
            contour_data,
            width=out_w,
            height=out_h,
            scale=config.render_scale,
            background=config.background_color,
        )

    save_png(output_image, output_path)
    metrics["t_rendering"] = time.perf_counter() - t0

    # ── Quality Metrics ───────────────────────────────────────────────────────
    if config.compute_metrics:
        try:
            ref = resize_to_match(original, out_w, out_h)
            metrics["ssim"] = compute_ssim(ref, output_image)
            metrics["psnr"] = compute_psnr(ref, output_image)
        except Exception as e:
            if verbose:
                print(f"  [!] metric computation failed: {e}")

    metrics["t_total"] = time.perf_counter() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    if verbose:
        sep = "-" * 52
        print(f"\n{sep}")
        print(f"  Done in {metrics['t_total']:.2f}s")
        print(f"  Clusters : {metrics.get('n_clusters', '?')}")
        print(f"  Regions  : {metrics.get('n_regions', '?')}")
        print(f"  Contours : {metrics.get('n_contours', '?')}")
        if "ssim" in metrics:
            print(f"  SSIM     : {metrics['ssim']:.4f}  (1.0 = identical)")
        if "psnr" in metrics:
            psnr_val = metrics["psnr"]
            psnr_str = f"{psnr_val:.2f} dB" if psnr_val != float("inf") else "inf"
            print(f"  PSNR     : {psnr_str}")
        print(f"{sep}\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="JPEG -> Complete PNG Restoration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("input", help="Input JPEG file path")
    p.add_argument(
        "-o", "--output", default=None,
        help="Output PNG path  (default: <input stem>_restored.png)",
    )

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    g2 = p.add_argument_group("Stage 2 — Color Clustering")
    g2.add_argument(
        "-k", "--clusters", type=int, default=24, metavar="N",
        help="Initial number of colour clusters  (default: 24)",
    )
    g2.add_argument(
        "--merge-threshold", type=float, default=12.0, metavar="D",
        help="CIE76 distance threshold for merging similar clusters  (default: 12.0)",
    )

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    g3 = p.add_argument_group("Stage 3 — Mask Generation")
    g3.add_argument(
        "--morph-kernel", type=int, default=3, metavar="K",
        help="Morphological kernel size  (default: 3)",
    )

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    g4 = p.add_argument_group("Stage 4 — Contour Extraction")
    g4.add_argument(
        "--epsilon", type=float, default=0.002, metavar="E",
        help="RDP simplification ratio  (default: 0.002 → ~0.2%% of arc length)",
    )

    # ── Stage 5 ───────────────────────────────────────────────────────────────
    g5 = p.add_argument_group("Stage 5 — Vectorization")
    g5.add_argument(
        "--no-curves", action="store_true",
        help="Use straight polylines instead of Catmull-Rom Bezier curves",
    )
    g5.add_argument(
        "--tension", type=float, default=1.0, metavar="T",
        help="Catmull-Rom tension  (default: 1.0)",
    )

    # ── Stage 6 ───────────────────────────────────────────────────────────────
    g6 = p.add_argument_group("Stage 6 — Rendering")
    g6.add_argument(
        "--scale", type=int, default=2, metavar="S",
        help="Supersampling scale for anti-aliasing 1–4  (default: 2)",
    )
    g6.add_argument(
        "--cairosvg", action="store_true",
        help="Use cairosvg for SVG→PNG rendering  (requires Cairo library)",
    )
    g6.add_argument(
        "--width", type=int, default=0, metavar="W",
        help="Output width override  (0 = same as input)",
    )
    g6.add_argument(
        "--height", type=int, default=0, metavar="H",
        help="Output height override  (0 = same as input)",
    )

    # ── Global ────────────────────────────────────────────────────────────────
    g0 = p.add_argument_group("Global options")
    g0.add_argument(
        "--denoise", type=int, default=5, metavar="D",
        help="Bilateral filter strength for JPEG denoising  (0 = off, default: 5)",
    )
    g0.add_argument(
        "--no-intermediates", action="store_true",
        help="Skip saving intermediate stage outputs",
    )
    g0.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress all progress output",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: file not found — {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output or str(
        Path(args.input).parent / (Path(args.input).stem + "_restored.png")
    )

    cfg = PipelineConfig(
        denoise_strength=args.denoise,
        n_clusters=args.clusters,
        merge_threshold=args.merge_threshold,
        morph_kernel_size=args.morph_kernel,
        epsilon_ratio=args.epsilon,
        use_curves=not args.no_curves,
        tension=args.tension,
        render_scale=args.scale,
        use_cairosvg=args.cairosvg,
        output_width=args.width,
        output_height=args.height,
        save_intermediates=not args.no_intermediates,
    )

    run_pipeline(args.input, output, cfg, verbose=not args.quiet)


if __name__ == "__main__":
    main()
