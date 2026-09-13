"""Periodic cubical persistent homology for density fields named ``rho``."""

from __future__ import annotations

import gudhi as gd
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import RecoveryConfig
from .observation import observe_field, reference_normalize


Diagram = dict[int, np.ndarray]


def finite_diagrams(
    filtration: np.ndarray,
    *,
    dimensions: tuple[int, ...],
    cutoff: float,
) -> Diagram:
    """Compute periodic H0/H1 and cap essential deaths at the filtration endpoint."""
    complex_ = gd.PeriodicCubicalComplex(
        top_dimensional_cells=filtration,
        periodic_dimensions=[True, True],
    )
    complex_.compute_persistence()
    endpoint = float(np.max(filtration))
    result: Diagram = {}
    for dimension in dimensions:
        intervals = complex_.persistence_intervals_in_dimension(dimension)
        intervals = (
            intervals.reshape(-1, 2)
            if intervals.size
            else np.zeros((0, 2), dtype=float)
        )
        if len(intervals):
            intervals = np.where(np.isfinite(intervals), intervals, endpoint)
            intervals = intervals[(intervals[:, 1] - intervals[:, 0]) >= cutoff]
        result[dimension] = (
            intervals.astype(float, copy=False)
            if len(intervals)
            else np.zeros((0, 2), dtype=float)
        )
    return result


def persistence_descriptor(field: np.ndarray, config: RecoveryConfig) -> Diagram:
    """Observe, divide by ``rho_ref``, then superlevel persistence of the fixed reference."""
    observed = observe_field(
        field,
        output_grid_size=config.observation_grid_size,
        sigma_hat=config.observation_sigma_hat,
    )
    normalized = reference_normalize(observed, config.rho_ref)
    filtration = config.filtration_reference - normalized
    return finite_diagrams(
        filtration,
        dimensions=config.homology_dimensions,
        cutoff=config.persistence_cutoff,
    )


def wasserstein2(diagram_a: np.ndarray, diagram_b: np.ndarray) -> float:
    """Euclidean 2-Wasserstein distance with diagonal matching."""
    n, m = len(diagram_a), len(diagram_b)
    if n == 0 and m == 0:
        return 0.0
    blocked = 1e9
    cost = np.zeros((n + m, n + m), dtype=float)
    if n and m:
        delta = diagram_a[:, None, :] - diagram_b[None, :, :]
        cost[:n, :m] = np.sum(delta**2, axis=2)
    if n:
        block = np.full((n, n), blocked)
        np.fill_diagonal(block, (diagram_a[:, 0] - diagram_a[:, 1]) ** 2 / 2.0)
        cost[:n, m:] = block
    if m:
        block = np.full((m, m), blocked)
        np.fill_diagonal(block, (diagram_b[:, 0] - diagram_b[:, 1]) ** 2 / 2.0)
        cost[n:, :m] = block
    rows, columns = linear_sum_assignment(cost)
    return float(np.sqrt(max(float(cost[rows, columns].sum()), 0.0)))
