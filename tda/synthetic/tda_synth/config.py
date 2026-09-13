"""Configuration dataclass and YAML loader. Every field is dimensionless."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


#: Config fields exempt from the unit-suffix check in the tests.
REPORTING_ONLY_FIELDS: frozenset[str] = frozenset()

#: Output tag per model, shared by every script.
MODEL_TAGS: dict[str, str] = {"linear": "model1", "logistic": "model2"}


def model_tag(config: "RecoveryConfig") -> str:
    """``model1`` or ``model2``, from the config's reaction term."""
    return MODEL_TAGS[config.model]


@dataclass(frozen=True)
class RecoveryConfig:
    # --- dimensionless analysis settings -------------------------------------
    pde_grid_size: int
    observation_grid_size: int
    observation_sigma_hat: float
    rho_ref: float
    filtration_reference: float
    persistence_cutoff: float
    homology_dimensions: tuple[int, ...]
    model: str
    candidate_solver_step: float
    target_solver_step: float
    rhat_grid: tuple[float, ...]
    dhat_grid: tuple[float, ...]
    rhat_bounds: tuple[float, float]
    dhat_bounds: tuple[float, float]
    refinement_n_starts: int
    refinement_maxfev_stage1: int
    refinement_maxfev_stage2: int
    refinement_xatol_normalized: float
    refinement_fatol: float
    holdout_seed: int
    scale_estimation_seed: int
    n_holdouts: int

    def validate(self) -> None:
        if self.pde_grid_size <= 1 or self.observation_grid_size <= 1:
            raise ValueError("grid sizes must exceed one")
        if not 0.0 < self.observation_sigma_hat < 0.5:
            raise ValueError(
                "observation_sigma_hat is a fraction of the domain side and must "
                "lie in (0, 0.5)"
            )
        if self.model not in {"linear", "logistic"}:
            raise ValueError("model must be 'linear' or 'logistic'")
        if self.rho_ref <= 0:
            raise ValueError("rho_ref must be positive")
        # The reaction term rhat*rho*(1-rho) fixes K = 1.
        if self.model == "logistic" and (
            self.rho_ref != 1.0 or self.filtration_reference != 1.0
        ):
            raise ValueError(
                "the logistic model fixes K = 1, so rho_ref and "
                "filtration_reference must both equal 1.0"
            )
        if self.persistence_cutoff < 0:
            raise ValueError("persistence_cutoff must be non-negative")
        if self.candidate_solver_step <= 0 or self.target_solver_step <= 0:
            raise ValueError("numerical solver steps must be positive")
        if not self.rhat_grid or not self.dhat_grid:
            raise ValueError("parameter grids cannot be empty")
        if min(self.rhat_grid) <= 0:
            raise ValueError("R_hat must be positive so that alpha = D_hat/R_hat exists")
        if min(self.dhat_grid) < 0:
            raise ValueError("D_hat must be non-negative")
        if set(self.homology_dimensions) - {0, 1}:
            raise ValueError("this 2-D pipeline supports H0 and H1")
        # Strictly increasing: parameter_to_unit divides by the span.
        if self.rhat_bounds[0] >= self.rhat_bounds[1]:
            raise ValueError("rhat_bounds must be strictly increasing")
        if self.dhat_bounds[0] >= self.dhat_bounds[1]:
            raise ValueError("dhat_bounds must be strictly increasing")
        if not (
            self.rhat_bounds[0] <= min(self.rhat_grid)
            and max(self.rhat_grid) <= self.rhat_bounds[1]
        ):
            raise ValueError("rhat_grid must lie inside rhat_bounds")
        if not (
            self.dhat_bounds[0] <= min(self.dhat_grid)
            and max(self.dhat_grid) <= self.dhat_bounds[1]
        ):
            raise ValueError("dhat_grid must lie inside dhat_bounds")
        if self.rhat_bounds[0] <= 0:
            raise ValueError("rhat_bounds must stay positive")
        if self.refinement_n_starts < 1:
            raise ValueError("refinement_n_starts must be positive")
        if self.refinement_maxfev_stage1 < 1 or self.refinement_maxfev_stage2 < 0:
            raise ValueError("refinement evaluation budgets are invalid")
        if self.n_holdouts < 1:
            raise ValueError("n_holdouts must be positive")

    @property
    def observation_sigma_pixels(self) -> float:
        """Smoothing width in observation-grid pixels."""
        return self.observation_sigma_hat * self.observation_grid_size

    def to_dict(self) -> dict[str, Any]:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.__dict__.items()
        }


def load_config(path: str | Path) -> RecoveryConfig:
    """Load a YAML configuration and freeze list-valued fields as tuples."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw["homology_dimensions"] = tuple(int(v) for v in raw["homology_dimensions"])
    raw["rhat_grid"] = tuple(float(v) for v in raw["rhat_grid"])
    raw["dhat_grid"] = tuple(float(v) for v in raw["dhat_grid"])
    raw["rhat_bounds"] = tuple(float(v) for v in raw["rhat_bounds"])
    raw["dhat_bounds"] = tuple(float(v) for v in raw["dhat_bounds"])
    cfg = RecoveryConfig(**raw)
    cfg.validate()
    return cfg
