"""Dimensionless groups.

    rho_hat = rho/K      x_hat = x/L0      t_hat = t/T,  t_hat in [0, 1]
    R_hat   = R*T        D_hat = D*T/L0**2
    tau     = R_hat      alpha = D_hat/R_hat = D/(R*L0**2)

One snapshot has no measured T, so only the ratio alpha is free of it. A movie
has measured frame intervals, which fixes T; see to_physical.
"""

from __future__ import annotations

from typing import NamedTuple


class Canonical(NamedTuple):
    """``alpha = D_hat/R_hat`` and ``tau = R_hat``. Both dimensionless."""

    alpha: float
    tau: float


class Dimensionless(NamedTuple):
    """Everything a recovery may report. No entry carries a unit."""

    rhat: float
    dhat: float
    tau: float
    alpha: float
    length_scale_ratio: float


class Physical(NamedTuple):
    """``R`` and ``D`` with units. Only a movie may produce this."""

    growth_rate: float
    diffusivity: float
    time_unit: str
    length_unit: str
    elapsed_time: float
    domain_length: float


def to_physical(
    rhat: float,
    dhat: float,
    *,
    elapsed_time: float,
    domain_length: float,
    time_unit: str,
    length_unit: str,
) -> Physical:
    """R = R_hat/T and D = D_hat*L0**2/T from a measured elapsed time."""
    if elapsed_time <= 0:
        raise ValueError("elapsed_time must be positive and measured, not assumed")
    if domain_length <= 0:
        raise ValueError("domain_length must be positive")
    return Physical(
        growth_rate=float(rhat) / float(elapsed_time),
        diffusivity=float(dhat) * float(domain_length) ** 2 / float(elapsed_time),
        time_unit=str(time_unit),
        length_unit=str(length_unit),
        elapsed_time=float(elapsed_time),
        domain_length=float(domain_length),
    )


def tau_alpha(rhat: float, dhat: float) -> tuple[float, float]:
    """``(tau, alpha)`` for the solver coefficients ``(R_hat, D_hat)``."""
    if rhat <= 0:
        raise ValueError("R_hat must be positive to form alpha = D_hat / R_hat")
    return float(rhat), float(dhat) / float(rhat)


def canonical(rhat: float, dhat: float) -> Canonical:
    tau, alpha = tau_alpha(rhat, dhat)
    return Canonical(alpha=alpha, tau=tau)


def rhat_dhat(tau: float, alpha: float) -> tuple[float, float]:
    """Inverse of :func:`canonical`: ``(R_hat, D_hat) = (tau, alpha*tau)``."""
    return float(tau), float(alpha) * float(tau)


def length_scale_ratio(rhat: float, dhat: float) -> float:
    """sqrt(alpha) = sqrt(D/R)/L0: the characteristic length, over the domain side."""
    _, alpha = tau_alpha(rhat, dhat)
    return float(alpha) ** 0.5


def report(rhat: float, dhat: float) -> Dimensionless:
    """Everything a recovery may report, from the solver coefficients."""
    tau, alpha = tau_alpha(rhat, dhat)
    return Dimensionless(
        rhat=float(rhat),
        dhat=float(dhat),
        tau=tau,
        alpha=alpha,
        length_scale_ratio=float(alpha) ** 0.5,
    )
