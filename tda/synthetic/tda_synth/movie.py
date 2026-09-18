"""Multi-frame recovery and the two observation modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import RecoveryConfig
from .pde import laplacian_symbol, reaction_term
from .persistence import Diagram, persistence_descriptor, wasserstein2
from .recovery import (
    FEATURE_SETS,
    PRIMARY_FEATURE_SET,
    parameter_to_unit,
    unit_to_parameter,
)

#: Diagrams for one candidate at every observed frame.
Frames = tuple[Diagram, ...]
MovieLibrary = dict[str, dict[tuple[float, float], Frames]]
MovieScales = dict[tuple[str, int, int], float]


def validate_frames(frames) -> tuple[float, ...]:
    """Frames must increase and end at 1.0, the final observation time."""
    frames = tuple(float(f) for f in frames)
    if not frames:
        raise ValueError("at least one frame is required")
    if frames[-1] != 1.0:
        raise ValueError("the last frame must be 1.0, the final observation time")
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError("frames must be strictly increasing")
    if frames[0] <= 0.0:
        raise ValueError("frames must be positive")
    return frames


def realized_frames(frames, solver_step: float) -> tuple[float, ...]:
    """The times the stepper can actually land on, given ``solver_step``."""
    frames = validate_frames(frames)
    steps = max(int(np.ceil(1.0 / solver_step)), 1)
    return tuple(max(int(round(f * steps)), 1) / steps for f in frames)


def simulate_movie(
    initial: np.ndarray,
    rhat: float,
    dhat: float,
    frames,
    *,
    model: str,
    solver_step: float,
) -> list[np.ndarray]:
    """The field at each frame, by the same propagator ``simulate_snapshot`` uses.

    Model 1 is exact in Fourier space at any time, so it is propagated rather
    than marched.
    """
    frames = validate_frames(frames)
    if solver_step <= 0:
        raise ValueError("solver_step must be positive")
    rho = np.asarray(initial, dtype=float)
    symbol = laplacian_symbol(initial.shape[0])

    if model == "linear":
        spectrum = np.fft.fft2(rho)
        out = []
        for frame in frames:
            propagator = np.exp(frame * (rhat - dhat * symbol))
            field = np.real(np.fft.ifft2(spectrum * propagator))
            out.append(np.clip(field, 0.0, None))
        return out

    steps = max(int(np.ceil(1.0 / solver_step)), 1)
    step = 1.0 / steps
    denominator = 1.0 + step * dhat * symbol
    rho = rho.copy()
    wanted = [max(int(round(f * steps)), 1) for f in frames]
    out = []
    cursor = 0
    for index in range(steps):
        reaction_state = rho + step * reaction_term(rho, rhat, model)
        rho = np.real(np.fft.ifft2(np.fft.fft2(reaction_state) / denominator))
        np.clip(rho, 0.0, None, out=rho)
        while cursor < len(wanted) and (index + 1) >= wanted[cursor]:
            out.append(rho.copy())
            cursor += 1
    while len(out) < len(frames):
        out.append(rho.copy())
    return out


def describe_movie(
    initial: np.ndarray,
    rhat: float,
    dhat: float,
    frames,
    config: RecoveryConfig,
    *,
    solver_step: float | None = None,
) -> Frames:
    """Observe and describe every frame with the project descriptor."""
    step = config.candidate_solver_step if solver_step is None else solver_step
    return tuple(
        persistence_descriptor(field, config)
        for field in simulate_movie(
            initial, rhat, dhat, frames, model=config.model, solver_step=step
        )
    )


def build_movie_library(
    initial_conditions: dict[str, np.ndarray], config: RecoveryConfig, frames
) -> MovieLibrary:
    """The coarse ``rhat_grid`` x ``dhat_grid`` per IC, one diagram set per frame."""
    frames = validate_frames(frames)
    return {
        ic_name: {
            (rhat, dhat): describe_movie(initial, rhat, dhat, frames, config)
            for rhat in config.rhat_grid
            for dhat in config.dhat_grid
        }
        for ic_name, initial in initial_conditions.items()
    }


def estimate_movie_scales(
    library: MovieLibrary,
    config: RecoveryConfig,
    frames,
    *,
    sample_size: int = 30,
) -> MovieScales:
    """Median pairwise diagram distance per (frame, homology dimension)."""
    frames = validate_frames(frames)
    rng = np.random.default_rng(config.scale_estimation_seed)
    points = list(next(iter(library.values())))
    chosen = rng.choice(len(points), min(sample_size, len(points)), replace=False)
    selected_points = [points[int(index)] for index in chosen]
    scales: MovieScales = {}
    for ic_name, frames_by_parameter in library.items():
        subset = [frames_by_parameter[point] for point in selected_points]
        for slot in range(len(frames)):
            for dimension in config.homology_dimensions:
                distances = [
                    wasserstein2(a[slot][dimension], b[slot][dimension])
                    for a, b in combinations(subset, 2)
                ]
                scales[(ic_name, slot, dimension)] = float(np.median(distances))
    return scales


def movie_loss(
    target: Frames,
    candidate: Frames,
    scales: MovieScales,
    ic_name: str,
    dimensions: tuple[int, ...],
) -> float:
    """Equal-weight standardized loss over frames and homology dimensions."""
    parts = []
    for slot in range(len(target)):
        for dimension in dimensions:
            scale = scales[(ic_name, slot, dimension)]
            if scale <= 1e-12:
                continue
            parts.append(
                (wasserstein2(target[slot][dimension], candidate[slot][dimension]) / scale) ** 2
            )
    return float(np.mean(parts)) if parts else float("inf")


def score_movie_library(
    target: Frames,
    library: MovieLibrary,
    scales: MovieScales,
    dimensions: tuple[int, ...],
) -> pd.DataFrame:
    rows = [
        {
            "ic_name": ic_name,
            "rhat": rhat,
            "dhat": dhat,
            "tau": rhat,
            "alpha": dhat / rhat,
            "loss": movie_loss(target, diagrams, scales, ic_name, dimensions),
        }
        for ic_name, frames_by_parameter in library.items()
        for (rhat, dhat), diagrams in frames_by_parameter.items()
    ]
    return pd.DataFrame(rows).sort_values("loss", ignore_index=True)


def recover_movie(
    target: Frames,
    initial_conditions: dict[str, np.ndarray],
    library: MovieLibrary,
    scales: MovieScales,
    config: RecoveryConfig,
    frames,
    dimensions: tuple[int, ...] = FEATURE_SETS[PRIMARY_FEATURE_SET],
) -> tuple[dict, pd.DataFrame]:
    """Coarse search, then bounded multi-start Nelder-Mead refinement.

    The initial condition is unknown: every IC in ``library`` is a candidate,
    and each refinement start marches its own IC.
    """
    frames = validate_frames(frames)
    surface = score_movie_library(target, library, scales, dimensions)
    quantum = min(config.refinement_xatol_normalized / 20.0, 1e-4)
    caches: dict[str, dict[tuple[int, int], Frames]] = {}

    def diagrams_at(ic_name: str, z: np.ndarray) -> Frames:
        z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
        key = (int(round(z[0] / quantum)), int(round(z[1] / quantum)))
        cache = caches.setdefault(ic_name, {})
        if key not in cache:
            rhat, dhat = unit_to_parameter(np.array(key, dtype=float) * quantum, config)
            cache[key] = describe_movie(
                initial_conditions[ic_name], rhat, dhat, frames, config
            )
        return cache[key]

    runs = []
    for _, row in surface.head(config.refinement_n_starts).iterrows():
        start_ic = str(row["ic_name"])

        def objective(z: np.ndarray, ic_name: str = start_ic) -> float:
            return movie_loss(target, diagrams_at(ic_name, z), scales, ic_name, dimensions)

        current = parameter_to_unit(float(row["rhat"]), float(row["dhat"]), config)
        stages = []
        for stage_index, budget in enumerate(
            (config.refinement_maxfev_stage1, config.refinement_maxfev_stage2), start=1
        ):
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
            snapped = np.round(np.clip(result.x, 0.0, 1.0) / quantum) * quantum
            rhat, dhat = unit_to_parameter(snapped, config)
            stages.append(
                {
                    "stage": stage_index,
                    "ic_name": start_ic,
                    "rhat": rhat,
                    "dhat": dhat,
                    "tau": rhat,
                    "alpha": dhat / rhat,
                    "loss": float(objective(snapped)),
                    "success": bool(result.success),
                    "nfev": int(result.nfev),
                }
            )
            current = result.x
            if result.success:
                break
        runs.append(min(stages, key=lambda row: row["loss"]))

    best = dict(min(runs, key=lambda row: row["loss"]))
    best["n_frames"] = len(frames)
    best["frames"] = list(frames)
    best["coarse_ic_name"] = str(surface.iloc[0]["ic_name"])
    best["coarse_rhat"] = float(surface.iloc[0]["rhat"])
    best["coarse_dhat"] = float(surface.iloc[0]["dhat"])
    best["coarse_loss"] = float(surface.iloc[0]["loss"])
    return best, surface


# --- observation modes ------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    name: str
    frames: tuple[float, ...]
    note: str

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["frames"] = list(self.frames)
        record["n_frames"] = len(self.frames)
        return record


#: The ``multi_frame`` schedule, as fractions of the final observation time.
#: Each value lands on the substep lattice of the shipped configs.
MULTI_FRAME_SCHEDULE = (0.3, 0.6, 1.0)

SCENARIOS: dict[str, Scenario] = {
    "single_snapshot": Scenario(
        name="single_snapshot",
        frames=(1.0,),
        note=(
            "one spatial field at an unmeasured elapsed time, the mode a "
            "biopsy slice supplies; it fixes D_hat/R_hat, not a rate"
        ),
    ),
    "multi_frame": Scenario(
        name="multi_frame",
        frames=MULTI_FRAME_SCHEDULE,
        note=(
            "three fields of one trajectory at known intervals, written as "
            "fractions of the last, the mode a time-lapse movie supplies"
        ),
    ),
}

DEFAULT_ORDER = ("single_snapshot", "multi_frame")


def frames_for(mode: str) -> tuple[float, ...]:
    return SCENARIOS[mode].frames


def table(order=DEFAULT_ORDER) -> list[dict[str, Any]]:
    return [SCENARIOS[name].as_dict() for name in order]
