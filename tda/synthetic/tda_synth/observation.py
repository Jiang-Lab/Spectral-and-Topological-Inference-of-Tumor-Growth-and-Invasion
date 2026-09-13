"""Observation operator: periodic resample then Gaussian smoothing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def observe_field(
    field: np.ndarray,
    *,
    output_grid_size: int,
    sigma_hat: float,
) -> np.ndarray:
    """Resample to ``output_grid_size``, then smooth by ``sigma_hat`` of the domain."""
    if not 0.0 < sigma_hat < 0.5:
        raise ValueError("sigma_hat must be a domain fraction in (0, 0.5)")
    factor = output_grid_size / field.shape[0]
    observed = zoom(
        field,
        (factor, factor),
        order=1,
        mode="grid-wrap",
        prefilter=False,
        grid_mode=True,
    )
    return gaussian_filter(observed, sigma=sigma_hat * output_grid_size, mode="wrap")


def reference_normalize(field: np.ndarray, rho_ref: float) -> np.ndarray:
    """Divide by one frozen cross-sample reference; never fit to this field."""
    if rho_ref <= 0:
        raise ValueError("rho_ref must be positive")
    return np.asarray(field, dtype=float) / rho_ref
