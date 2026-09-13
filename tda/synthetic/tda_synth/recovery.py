"""Coarse-to-continuous recovery from one snapshot, per initial condition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import RecoveryConfig
from .pde import simulate_snapshot
from .persistence import Diagram, persistence_descriptor, wasserstein2


FEATURE_SETS: dict[str, tuple[int, ...]] = {
    "h0": (0,),
    "h0h1": (0, 1),
}
PRIMARY_FEATURE_SET = "h0h1"


@dataclass(frozen=True)
class Candidate:
    rhat: float
    dhat: float
    ic_name: str
    diagram: Diagram


Library = dict[str, dict[tuple[float, float], Candidate]]

#: Field to persistence diagrams. Injectable, so a control can swap it out.
Descriptor = Callable[[np.ndarray, "RecoveryConfig"], Diagram]


def build_library(
    initial_conditions: dict[str, np.ndarray],
    config: RecoveryConfig,
    descriptor: Descriptor = persistence_descriptor,
) -> Library:
    """Build the coarse ``rhat_grid`` x ``dhat_grid`` library for every IC."""
    library: Library = {}
    for ic_name, rho_initial in initial_conditions.items():
        expected_shape = (config.pde_grid_size, config.pde_grid_size)
        if rho_initial.shape != expected_shape:
            raise ValueError(
                f"{ic_name} has shape {rho_initial.shape}; expected {expected_shape}"
            )
        by_parameter: dict[tuple[float, float], Candidate] = {}
        for rhat in config.rhat_grid:
            for dhat in config.dhat_grid:
                rho = simulate_snapshot(
                    rho_initial,
                    rhat,
                    dhat,
                    model=config.model,
                    solver_step=config.candidate_solver_step,
                )
                by_parameter[(rhat, dhat)] = Candidate(
                    rhat=rhat,
                    dhat=dhat,
                    ic_name=ic_name,
                    diagram=descriptor(rho, config),
                )
        library[ic_name] = by_parameter
    return library


def estimate_scales(
    library: Library,
    dimensions: tuple[int, ...],
    *,
    config: RecoveryConfig | None = None,
    sample_size: int = 30,
    seed: int | None = None,
) -> dict[tuple[str, int], float]:
    """Median pairwise diagram distance, used to standardize the loss."""
    if seed is None:
        if config is None:
            raise ValueError("pass either config or an explicit seed")
        seed = config.scale_estimation_seed
    rng = np.random.default_rng(seed)
    parameter_points = list(next(iter(library.values())))
    chosen = rng.choice(
        len(parameter_points), min(sample_size, len(parameter_points)), replace=False
    )
    selected_points = [parameter_points[int(index)] for index in chosen]
    scales: dict[tuple[str, int], float] = {}
    for ic_name, candidates_by_parameter in library.items():
        subset = [candidates_by_parameter[point] for point in selected_points]
        for dimension in dimensions:
            distances = [
                wasserstein2(a.diagram[dimension], b.diagram[dimension])
                for a, b in combinations(subset, 2)
            ]
            scales[(ic_name, dimension)] = float(np.median(distances))
    return scales


def diagram_loss(
    target: Diagram,
    candidate: Diagram,
    scales: dict[tuple[str, int], float],
    ic_name: str,
    dimensions: tuple[int, ...],
) -> tuple[float, dict[int, float], dict[int, float]]:
    """Equal-weight standardized loss; degenerate dimensions are dropped."""
    components: dict[int, float] = {}
    distances: dict[int, float] = {}
    for dimension in dimensions:
        scale = scales[(ic_name, dimension)]
        if scale <= 1e-12:
            continue
        distance = wasserstein2(target[dimension], candidate[dimension])
        distances[dimension] = distance
        components[dimension] = (distance / scale) ** 2
    if not components:
        return float("inf"), components, distances
    return float(np.mean(list(components.values()))), components, distances


def usable_dimensions(
    scales: dict[tuple[str, int], float],
    ic_name: str,
    dimensions: tuple[int, ...],
) -> tuple[int, ...]:
    """Dimensions with a non-degenerate median scale, in the requested order."""
    return tuple(q for q in dimensions if scales[(ic_name, q)] > 1e-12)


def score_library(
    target: Diagram,
    library: Library,
    scales: dict[tuple[str, int], float],
    ic_name: str,
    dimensions: tuple[int, ...],
) -> pd.DataFrame:
    """Score one snapshot against the library built from the same IC."""
    rows: list[dict[str, float | str]] = []
    for candidate in library[ic_name].values():
        loss, components, distances = diagram_loss(
            target, candidate.diagram, scales, ic_name, dimensions
        )
        row: dict[str, float | str] = {
            "ic_name": ic_name,
            "rhat": candidate.rhat,
            "dhat": candidate.dhat,
            "tau": candidate.rhat,
            "alpha": candidate.dhat / candidate.rhat,
            "loss": loss,
        }
        row.update({f"h{q}_loss": value for q, value in components.items()})
        row.update({f"h{q}_distance": value for q, value in distances.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("loss", ignore_index=True)


def parameter_to_unit(
    rhat: float, dhat: float, config: RecoveryConfig
) -> np.ndarray:
    """Map dimensionless parameters to the optimizer's common [0,1] scale."""
    return np.asarray(
        [
            (rhat - config.rhat_bounds[0])
            / (config.rhat_bounds[1] - config.rhat_bounds[0]),
            (dhat - config.dhat_bounds[0])
            / (config.dhat_bounds[1] - config.dhat_bounds[0]),
        ]
    )


def unit_to_parameter(
    z: np.ndarray, config: RecoveryConfig
) -> tuple[float, float]:
    """Map the optimizer coordinates back to dimensionless parameters."""
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    rhat = config.rhat_bounds[0] + z[0] * (
        config.rhat_bounds[1] - config.rhat_bounds[0]
    )
    dhat = config.dhat_bounds[0] + z[1] * (
        config.dhat_bounds[1] - config.dhat_bounds[0]
    )
    return float(rhat), float(dhat)


def _snap(z: np.ndarray, quantum: float) -> np.ndarray:
    """Round optimizer coordinates onto the cache lattice actually evaluated."""
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    return np.round(z / quantum) * quantum


def recover_continuous(
    target_field: np.ndarray | None,
    rho_initial: np.ndarray,
    library: Library,
    scales: dict[tuple[str, int], float],
    config: RecoveryConfig,
    ic_name: str,
    dimensions: tuple[int, ...] = FEATURE_SETS[PRIMARY_FEATURE_SET],
    *,
    target_diagram: Diagram | None = None,
    descriptor: Descriptor = persistence_descriptor,
) -> tuple[dict, pd.DataFrame]:
    """Coarse grid search, then bounded multi-start Nelder-Mead."""
    if target_diagram is None:
        if target_field is None:
            raise ValueError("pass either target_field or target_diagram")
        target_diagram = descriptor(target_field, config)
    target = target_diagram
    used = usable_dimensions(scales, ic_name, dimensions)
    if not used:
        raise ValueError(f"no usable homology dimension for {ic_name}")
    surface = score_library(target, library, scales, ic_name, dimensions)
    starts = surface.head(config.refinement_n_starts)

    # Cache on the normalized coordinates, quantized below the Nelder-Mead
    # tolerance. Raw parameters never hit: D_hat ~ 1e-4 keeps eleven digits.
    cache_quantum = min(config.refinement_xatol_normalized / 20.0, 1e-4)

    @lru_cache(maxsize=4096)
    def candidate_diagram_quantized(z0_key: int, z1_key: int) -> Diagram:
        rhat_key, dhat_key = unit_to_parameter(
            np.array([z0_key * cache_quantum, z1_key * cache_quantum]), config
        )
        rho = simulate_snapshot(
            rho_initial,
            rhat_key,
            dhat_key,
            model=config.model,
            solver_step=config.candidate_solver_step,
        )
        return descriptor(rho, config)

    def candidate_diagram(z: np.ndarray) -> Diagram:
        z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
        return candidate_diagram_quantized(
            int(round(z[0] / cache_quantum)), int(round(z[1] / cache_quantum))
        )

    runs: list[dict] = []
    for _, coarse in starts.iterrows():
        coarse_parameter = (float(coarse["rhat"]), float(coarse["dhat"]))

        def objective(z: np.ndarray) -> float:
            value, _, _ = diagram_loss(
                target, candidate_diagram(z), scales, ic_name, dimensions
            )
            return value

        budgets = (
            config.refinement_maxfev_stage1,
            config.refinement_maxfev_stage2,
        )
        current = parameter_to_unit(*coarse_parameter, config)
        stages: list[dict] = []
        for stage_index, budget in enumerate(budgets, start=1):
            if budget <= 0:
                continue
            result = minimize(
                objective,
                current,
                method="Nelder-Mead",
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                options={
                    "maxfev": budget,
                    "xatol": config.refinement_xatol_normalized,
                    "fatol": config.refinement_fatol,
                    "adaptive": True,
                },
            )
            rhat, dhat = unit_to_parameter(_snap(result.x, cache_quantum), config)
            stages.append(
                {
                    "stage": stage_index,
                    "rhat": rhat,
                    "dhat": dhat,
                    "tau": rhat,
                    "alpha": dhat / rhat,
                    "loss": float(result.fun),
                    "success": bool(result.success),
                    "status": int(result.status),
                    "message": str(result.message),
                    "nfev": int(result.nfev),
                    "nit": int(result.nit),
                }
            )
            current = result.x
            if result.success:
                break
        best_stage = min(stages, key=lambda row: row["loss"])
        runs.append(
            {
                "coarse_rhat": coarse_parameter[0],
                "coarse_dhat": coarse_parameter[1],
                "coarse_loss": float(coarse["loss"]),
                "best": best_stage,
                "stages": stages,
            }
        )

    winner = min(runs, key=lambda row: row["best"]["loss"])
    best = dict(winner["best"])
    # Success of the run that produced the reported estimate.
    best["success"] = bool(winner["best"]["success"])
    best["any_start_succeeded"] = any(
        stage["success"] for run in runs for stage in run["stages"]
    )
    best["dimensions_used"] = list(used)
    best["dimensions_requested"] = list(dimensions)
    best["coarse_rhat"] = float(surface.iloc[0]["rhat"])
    best["coarse_dhat"] = float(surface.iloc[0]["dhat"])
    best["coarse_loss"] = float(surface.iloc[0]["loss"])
    best["all_starts"] = runs
    return best, surface


def _interior(grid) -> tuple[float, float]:
    """The grid's span, pulled in by one cell at each end."""
    values = sorted(float(value) for value in grid)
    if len(values) < 3:
        raise ValueError("a candidate grid needs at least three points")
    return values[1], values[-2]


def make_offgrid_holdouts(
    config: RecoveryConfig,
    count: int | None = None,
) -> list[dict[str, float | str]]:
    """Deterministic truths one grid cell inside each end of the library."""
    count = config.n_holdouts if count is None else count
    rhat_low, rhat_high = _interior(config.rhat_grid)
    dhat_low, dhat_high = _interior(config.dhat_grid)
    rng = np.random.default_rng(config.holdout_seed)
    rows: list[dict[str, float | str]] = []
    while len(rows) < count:
        rhat = float(np.round(rng.uniform(rhat_low, rhat_high), 3))
        dhat = float(np.round(rng.uniform(dhat_low, dhat_high), 9))
        r_on_grid = any(
            np.isclose(rhat, value, rtol=0.0, atol=1e-12)
            for value in config.rhat_grid
        )
        d_on_grid = any(
            np.isclose(dhat, value, rtol=0.0, atol=1e-12)
            for value in config.dhat_grid
        )
        if not r_on_grid and not d_on_grid:
            rows.append(
                {
                    "case_id": f"case_{len(rows):03d}",
                    "rhat": rhat,
                    "dhat": dhat,
                    "tau": rhat,
                    "alpha": dhat / rhat,
                }
            )
    return rows
