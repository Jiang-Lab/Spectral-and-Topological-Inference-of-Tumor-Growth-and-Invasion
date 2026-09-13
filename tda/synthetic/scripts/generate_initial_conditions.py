#!/usr/bin/env python3
"""Regenerate the sealed four-IC realization from seed 42. For audit only."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data"
DOMAIN_UM = 2500.0
GRID = 1000
PIXEL_UM = DOMAIN_UM / GRID
CELL_RADIUS_PX = int(10.0 / PIXEL_UM)
N_CELLS = 6000
IC_NAMES = (
    "Random spaced",
    "One central cluster",
    "Many clusters",
    "Clusters + singles",
)


def draw_cells(points, radius_px=CELL_RADIUS_PX):
    image = np.zeros((GRID, GRID), dtype=np.uint8)
    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    disk = (xx**2 + yy**2) <= radius_px**2
    for x, y in points.astype(int):
        x0, x1 = max(0, x - radius_px), min(GRID, x + radius_px + 1)
        y0, y1 = max(0, y - radius_px), min(GRID, y + radius_px + 1)
        kx0, kx1 = radius_px - (x - x0), radius_px + (x1 - x)
        ky0, ky1 = radius_px - (y - y0), radius_px + (y1 - y)
        image[y0:y1, x0:x1] |= disk[ky0:ky1, kx0:kx1]
    return image.astype(float)


def random_spaced_cells(n_cells=N_CELLS, min_dist_um=18.0):
    min_dist_px = min_dist_um / PIXEL_UM
    points = []
    attempts = 0
    while len(points) < n_cells and attempts < n_cells * 500:
        point = np.array([np.random.uniform(0, GRID), np.random.uniform(0, GRID)])
        if not points or cKDTree(np.array(points)).query(point)[0] >= min_dist_px:
            points.append(point)
        attempts += 1
    return np.array(points)


def one_central_cluster(n_cells=N_CELLS, sigma_um=350.0):
    sigma_px = sigma_um / PIXEL_UM
    x = np.clip(np.random.normal(GRID / 2, sigma_px, n_cells), 0, GRID - 1)
    y = np.clip(np.random.normal(GRID / 2, sigma_px, n_cells), 0, GRID - 1)
    return np.column_stack([x, y])


def many_clusters(n_clusters=20, cells_per_cluster=300, sigma_um=120.0):
    sigma_px = sigma_um / PIXEL_UM
    centres_x = np.random.uniform(150, GRID - 150, n_clusters)
    centres_y = np.random.uniform(150, GRID - 150, n_clusters)
    groups = []
    for x0, y0 in zip(centres_x, centres_y):
        x = np.clip(np.random.normal(x0, sigma_px, cells_per_cluster), 0, GRID - 1)
        y = np.clip(np.random.normal(y0, sigma_px, cells_per_cluster), 0, GRID - 1)
        groups.append(np.column_stack([x, y]))
    return np.vstack(groups)


def clusters_plus_singles(
    n_clusters=20, cells_per_cluster=250, n_singles=1000, sigma_um=120.0
):
    clustered = many_clusters(n_clusters, cells_per_cluster, sigma_um)
    singles = np.column_stack(
        [np.random.uniform(0, GRID, n_singles), np.random.uniform(0, GRID, n_singles)]
    )
    return np.vstack([clustered, singles])


def generate(seed=42):
    np.random.seed(seed)
    points = {
        "Random spaced": random_spaced_cells(N_CELLS, min_dist_um=18.0),
        "One central cluster": one_central_cluster(N_CELLS, sigma_um=350.0),
        "Many clusters": many_clusters(20, N_CELLS // 20, 120.0),
        "Clusters + singles": clusters_plus_singles(
            20, 250, N_CELLS - 20 * 250, 120.0
        ),
    }
    return {name: draw_cells(points[name]) for name in IC_NAMES}


def downsample(mask, target=125):
    factor = GRID // target
    return mask.reshape(target, factor, target, factor).mean(axis=(1, 3))


def main():
    masks = generate()
    fields = {name: downsample(mask, 125).astype(np.float32) for name, mask in masks.items()}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "somiya_ic_125_regenerated.npz"
    np.savez_compressed(output, **fields, order=np.array(IC_NAMES))
    for name, field in fields.items():
        digest = hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest()
        print(f"{name:22s} mean={field.mean():.8f} sha256={digest}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
