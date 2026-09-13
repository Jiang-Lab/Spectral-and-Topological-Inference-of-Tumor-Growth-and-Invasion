"""Forward solvers for Model 1 (linear) and Model 2 (logistic) on ``t_hat in [0,1]``."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def laplacian_symbol(grid_size: int) -> np.ndarray:
    """Return the squared Fourier wave-number on the unit square."""
    spacing_hat = 1.0 / grid_size
    k = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=spacing_hat)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return kx**2 + ky**2


def reaction_term(rho: np.ndarray, rhat: float, model: str) -> np.ndarray:
    """Reaction of ``model``; shared by every solver so they cannot diverge."""
    if model == "linear":
        return rhat * rho
    if model == "logistic":
        return rhat * rho * (1.0 - rho)
    raise ValueError(f"unknown model: {model}")


def solve_linear(
    initial: np.ndarray,
    rhat: float,
    dhat: float,
) -> np.ndarray:
    """Exact Fourier map of Model 1 to ``t_hat = 1``; not bounded above."""
    symbol = np.exp(rhat - dhat * laplacian_symbol(initial.shape[0]))
    field = np.real(np.fft.ifft2(np.fft.fft2(initial) * symbol))
    return np.clip(field, 0.0, None)


def solve_logistic(
    initial: np.ndarray,
    rhat: float,
    dhat: float,
    *,
    solver_step: float = 0.01,
) -> np.ndarray:
    """Spectral IMEX map of Model 2 to ``t_hat = 1``; K = 1 by the scaling."""
    if solver_step <= 0:
        raise ValueError("solver_step must be positive")
    rho = np.asarray(initial, dtype=float).copy()
    steps = max(int(np.ceil(1.0 / solver_step)), 1)
    step = 1.0 / steps
    denominator = 1.0 + step * dhat * laplacian_symbol(rho.shape[0])
    for _ in range(steps):
        reaction_state = rho + step * reaction_term(rho, rhat, "logistic")
        rho = np.real(np.fft.ifft2(np.fft.fft2(reaction_state) / denominator))
        np.clip(rho, 0.0, None, out=rho)
    return rho


def simulate_snapshot(
    initial: np.ndarray,
    rhat: float,
    dhat: float,
    *,
    model: str,
    solver_step: float = 0.01,
) -> np.ndarray:
    """Map an IC to the fixed endpoint ``t_hat = 1``; there is no time argument."""
    if model == "linear":
        return solve_linear(initial, rhat, dhat)
    if model == "logistic":
        return solve_logistic(
            initial,
            rhat,
            dhat,
            solver_step=solver_step,
        )
    raise ValueError(f"unknown model: {model}")
