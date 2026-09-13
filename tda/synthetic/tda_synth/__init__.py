"""Dimensionless TDA recovery of reaction-diffusion parameters."""

from . import movie
from .config import RecoveryConfig, load_config
from .initial_conditions import IC_NAMES, load_initial_conditions
from .pde import simulate_snapshot
from .persistence import persistence_descriptor
from .recovery import (
    FEATURE_SETS,
    PRIMARY_FEATURE_SET,
    build_library,
    estimate_scales,
    make_offgrid_holdouts,
    recover_continuous,
    usable_dimensions,
)
from .scaling import (
    Canonical,
    Dimensionless,
    Physical,
    canonical,
    length_scale_ratio,
    report,
    rhat_dhat,
    to_physical,
    tau_alpha,
)

__all__ = [
    "movie",
    "RecoveryConfig",
    "load_config",
    "IC_NAMES",
    "load_initial_conditions",
    "simulate_snapshot",
    "persistence_descriptor",
    "build_library",
    "estimate_scales",
    "FEATURE_SETS",
    "PRIMARY_FEATURE_SET",
    "make_offgrid_holdouts",
    "recover_continuous",
    "usable_dimensions",
    "Canonical",
    "Dimensionless",
    "Physical",
    "canonical",
    "length_scale_ratio",
    "report",
    "rhat_dhat",
    "to_physical",
    "tau_alpha",
]
