"""
Pipeline configuration dataclass.
All parameters for the JPEG → PNG reconstruction pipeline.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PipelineConfig:
    # ── Stage 1: Preprocessing ────────────────────────────────────────────────
    denoise_strength: int = 5
    """Bilateral filter strength for JPEG artifact removal (0 = disabled)."""

    # ── Stage 2: Color Clustering ─────────────────────────────────────────────
    n_clusters: int = 24
    """Initial number of K-means color clusters."""

    merge_threshold: float = 12.0
    """CIE76 perceptual distance threshold for merging similar clusters.
    Set to 0 to disable merging."""

    # ── Stage 3: Mask Generation ──────────────────────────────────────────────
    min_area_ratio: float = 0.0001
    """Minimum region area as fraction of total image area (filters noise)."""

    morph_kernel_size: int = 3
    """Morphological kernel size for mask cleanup (close/open operations)."""

    # ── Stage 4: Contour Extraction ───────────────────────────────────────────
    epsilon_ratio: float = 0.002
    """RDP epsilon as fraction of contour arc length.
    Higher → more aggressive simplification → fewer control points."""

    min_contour_area: int = 50
    """Absolute minimum contour area in pixels."""

    # ── Stage 5: Vectorization ────────────────────────────────────────────────
    use_curves: bool = True
    """Use Catmull-Rom Bezier curves (True) or straight polylines (False)."""

    tension: float = 1.0
    """Catmull-Rom tension. 1.0 = standard smooth, <1.0 = looser curves."""

    svg_precision: int = 2
    """Decimal places for SVG coordinate values."""

    # ── Stage 6: Rendering ────────────────────────────────────────────────────
    render_scale: int = 2
    """Supersampling scale for anti-aliasing (renders at N×, then downscales).
    Recommended: 2–4. Higher = better AA but more memory."""

    output_width: int = 0
    """Output width override. 0 = match input image width."""

    output_height: int = 0
    """Output height override. 0 = match input image height."""

    use_cairosvg: bool = False
    """Use cairosvg for SVG→PNG rendering (better quality, needs Cairo installed)."""

    background_color: Tuple[int, int, int] = field(
        default_factory=lambda: (255, 255, 255)
    )
    """Background fill color (R, G, B)."""

    # ── Output Options ────────────────────────────────────────────────────────
    save_intermediates: bool = True
    """Save intermediate outputs for each pipeline stage."""

    compute_metrics: bool = True
    """Compute SSIM / PSNR quality metrics vs. the original JPEG."""
