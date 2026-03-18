"""
Stage 2 — Color Clustering

Quantizes the image into N perceptually uniform color clusters using
MiniBatch K-means in CIE Lab color space.

Why Lab?
  JPEG stores colour information in a YCbCr space that exploits the eye's
  lower sensitivity to chroma detail. After decompression, clustering in
  RGB conflates luminance and chroma differences. CIE Lab (L*a*b*) is
  designed so that equal Euclidean distances correspond to equal perceived
  colour differences (CIE76 metric), giving more stable cluster boundaries.
"""

import warnings
from typing import Tuple

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


# ─────────────────────────────────────────────────────────────────────────────
# Primary clustering
# ─────────────────────────────────────────────────────────────────────────────

def cluster_colors(
    image: np.ndarray,
    n_clusters: int = 24,
    batch_size: int = 20_000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster image pixels into N colour groups using MiniBatch K-means in Lab space.

    Args:
        image:        RGB uint8 (H, W, 3)
        n_clusters:   number of desired colour clusters
        batch_size:   mini-batch size (controls speed vs. quality trade-off)
        random_state: reproducibility seed

    Returns:
        labels:      (H, W) int32 — cluster index per pixel
        centers_rgb: (n_clusters, 3) uint8 — cluster representative colour in RGB
    """
    h, w = image.shape[:2]

    # Convert to Lab — OpenCV Lab ranges: L [0,255], a [0,255], b [0,255]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab.reshape(-1, 3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(batch_size, len(pixels)),
            random_state=random_state,
            n_init=5,
            max_iter=300,
            reassignment_ratio=0.01,
        )
        kmeans.fit(pixels)

    labels = kmeans.labels_.reshape(h, w).astype(np.int32)

    # Convert cluster centers back to RGB
    centers_lab = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)
    centers_lab_img = centers_lab.reshape(1, -1, 3)
    centers_rgb = cv2.cvtColor(centers_lab_img, cv2.COLOR_LAB2RGB).reshape(-1, 3)

    return labels, centers_rgb


# ─────────────────────────────────────────────────────────────────────────────
# Cluster merging
# ─────────────────────────────────────────────────────────────────────────────

def merge_similar_clusters(
    labels: np.ndarray,
    centers_rgb: np.ndarray,
    distance_threshold: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge colour clusters whose CIE76 Lab distance falls below a threshold.

    Uses Union-Find to identify connected groups, then averages the member
    colours and remaps pixel labels.

    Args:
        labels:             (H, W) cluster label array
        centers_rgb:        (N, 3) cluster centre colours in RGB
        distance_threshold: max CIE76 distance to trigger a merge

    Returns:
        Tuple of updated (labels, centers_rgb) with fewer clusters.
    """
    n = len(centers_rgb)
    if n <= 1 or distance_threshold <= 0:
        return labels, centers_rgb

    # Convert centers to Lab for distance computation
    centers_lab = cv2.cvtColor(
        centers_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(float)

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # O(N²) pairwise comparison — fine for typical cluster counts (≤64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(centers_lab[i] - centers_lab[j]))
            if dist < distance_threshold:
                union(i, j)

    # Build group → new index mapping
    groups: dict = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    old_to_new = {}
    new_centers = []
    for new_idx, (root, members) in enumerate(groups.items()):
        for old_idx in members:
            old_to_new[old_idx] = new_idx
        avg_color = centers_rgb[members].mean(axis=0).astype(np.uint8)
        new_centers.append(avg_color)

    # Remap pixel labels
    remap = np.array([old_to_new[i] for i in range(n)], dtype=np.int32)
    new_labels = remap[labels]
    new_centers_rgb = np.array(new_centers, dtype=np.uint8)

    return new_labels, new_centers_rgb


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction helper
# ─────────────────────────────────────────────────────────────────────────────

def quantized_image(labels: np.ndarray, centers_rgb: np.ndarray) -> np.ndarray:
    """
    Reconstruct a flat-colour image from cluster labels and centre colours.

    Returns: (H, W, 3) uint8 RGB array.
    """
    return centers_rgb[labels].astype(np.uint8)
