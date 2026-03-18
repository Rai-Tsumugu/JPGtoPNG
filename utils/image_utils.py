"""
Image I/O, preprocessing, and quality metric utilities.
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load any image file and return an (H, W, 3) uint8 RGB array."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(array: np.ndarray, path: str) -> None:
    """Save an (H, W, 3) uint8 RGB array to disk."""
    Image.fromarray(array.astype(np.uint8)).save(path)


# ─────────────────────────────────────────────────────────────────────────────
# JPEG Artifact Reduction
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_jpeg(image: np.ndarray, denoise_strength: int = 5) -> np.ndarray:
    """
    Reduce JPEG compression artifacts before clustering.

    Two-pass approach:
      1. Bilateral filter — smooths flat regions while preserving edges,
         reducing DCT-block noise without blurring true boundaries.
      2. 3×3 median filter — removes isolated pixel outliers (ringing artefacts).

    Args:
        image:            RGB uint8 array
        denoise_strength: 0 = disabled; higher values = stronger smoothing

    Returns:
        Cleaned RGB uint8 array.
    """
    if denoise_strength == 0:
        return image.copy()

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Bilateral: d is the neighbourhood diameter
    d = denoise_strength * 2 + 1
    sigma = 25 + denoise_strength * 10
    filtered = cv2.bilateralFilter(bgr, d, sigmaColor=sigma, sigmaSpace=sigma)

    # Median: removes salt-and-pepper / ringing artefacts
    filtered = cv2.medianBlur(filtered, 3)

    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) between two RGB images.
    Both images must have the same dimensions.
    Returns a value in [0, 1]; 1.0 = identical.
    """
    try:
        from skimage.metrics import structural_similarity
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        return float(structural_similarity(gray1, gray2, data_range=255))
    except ImportError:
        # Fallback: simplified SSIM approximation
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        f1, f2 = img1.astype(float), img2.astype(float)
        mu1, mu2 = f1.mean(), f2.mean()
        s1 = f1.std() ** 2
        s2 = f2.std() ** 2
        s12 = float(np.mean((f1 - mu1) * (f2 - mu2)))
        return float(
            ((2 * mu1 * mu2 + c1) * (2 * s12 + c2))
            / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
        )


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB between two uint8 images.
    Higher is better; ∞ = identical.
    """
    mse = float(np.mean((img1.astype(float) - img2.astype(float)) ** 2))
    if mse < 1e-10:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def resize_to_match(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image to (target_w × target_h) using high-quality Lanczos."""
    if img.shape[1] == target_w and img.shape[0] == target_h:
        return img
    return np.array(
        Image.fromarray(img).resize((target_w, target_h), Image.LANCZOS)
    )
