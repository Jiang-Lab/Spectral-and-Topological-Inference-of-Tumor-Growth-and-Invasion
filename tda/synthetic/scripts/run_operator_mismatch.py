#!/usr/bin/env python3
"""Recovery when the target comes from a different forward operator."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _study_common import (
    DEFAULT_CONFIG,
    RESULTS,
    add_tag_argument,
    recover,
    setup,
    signed_errors,
    stem,
    summarize,
    truths,
    write_json,
)

from tda_synth.pde import reaction_term, simulate_snapshot


def five_point_symbol(grid: int) -> np.ndarray:
    """Fourier symbol of the periodic five-point Laplacian on the unit square."""
    spacing = 1.0 / grid
    k = 2.0 * np.pi * np.fft.fftfreq(grid, d=spacing)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return (
        -4.0
        / spacing**2
        * (np.sin(kx * spacing / 2.0) ** 2 + np.sin(ky * spacing / 2.0) ** 2)
    )


def lift(field: np.ndarray, factor: int) -> np.ndarray:
    """Piecewise-constant refinement; exactly reversed by :func:`coarsen`."""
    return np.kron(field, np.ones((factor, factor), dtype=float))


def coarsen(field: np.ndarray, target: int) -> np.ndarray:
    factor = field.shape[0] // target
    return field.reshape(target, factor, target, factor).mean(axis=(1, 3))


def mismatched_snapshot(
    initial_fine: np.ndarray,
    rhat: float,
    dhat: float,
    *,
    model: str,
    symbol: np.ndarray,
    step: float,
    output_grid: int,
) -> np.ndarray:
    """Crank-Nicolson diffusion on the five-point stencil, explicit reaction."""
    rho = initial_fine.copy()
    steps = max(int(np.ceil(1.0 / step)), 1)
    substep = 1.0 / steps
    gain = (1.0 + 0.5 * substep * dhat * symbol) / (
        1.0 - 0.5 * substep * dhat * symbol
    )
    for _ in range(steps):
        reacted = rho + substep * reaction_term(rho, rhat, model)
        rho = np.real(np.fft.ifft2(np.fft.fft2(reacted) * gain))
        np.clip(rho, 0.0, None, out=rho)
    return coarsen(rho, output_grid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--ics",
        nargs="+",
        default=["Random spaced", "One central cluster", "Many clusters", "Clusters + singles"],
    )
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--fine-grid", type=int, default=250)
    parser.add_argument("--fine-step", type=float, default=0.002)
    add_tag_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, initials, library, scales = setup(tuple(args.ics), args.config)
    if args.fine_grid % config.pde_grid_size:
        raise SystemExit(
            f"--fine-grid must be a multiple of {config.pde_grid_size}"
        )
    factor = args.fine_grid // config.pde_grid_size
    symbol = five_point_symbol(args.fine_grid)
    coarse_symbol = five_point_symbol(config.pde_grid_size)
    fine_initials = {name: lift(field, factor) for name, field in initials.items()}
    cases = truths(config, args.cases)

    def make_target(arm: str, ic_name: str, truth: dict) -> np.ndarray:
        """One factor per arm; everything not named stays at the config value."""
        rhat, dhat = float(truth["rhat"]), float(truth["dhat"])
        shipped = config.candidate_solver_step
        if arm == "inverse_crime":
            return simulate_snapshot(
                initials[ic_name], rhat, dhat,
                model=config.model, solver_step=shipped,
            )
        if arm == "substep_only":
            return simulate_snapshot(
                initials[ic_name], rhat, dhat,
                model=config.model, solver_step=args.fine_step,
            )
        if arm == "grid_only":
            # Shipped scheme and substep, fine grid, area-averaged back.
            fine = simulate_snapshot(
                fine_initials[ic_name], rhat, dhat,
                model=config.model, solver_step=shipped,
            )
            return coarsen(fine, config.pde_grid_size)
        if arm == "operator_only":
            # Five-point Laplacian and Crank-Nicolson, shipped grid and substep.
            return mismatched_snapshot(
                initials[ic_name], rhat, dhat,
                model=config.model,
                symbol=coarse_symbol,
                step=shipped,
                output_grid=config.pde_grid_size,
            )
        if arm == "full_mismatch":
            return mismatched_snapshot(
                fine_initials[ic_name], rhat, dhat,
                model=config.model,
                symbol=symbol,
                step=args.fine_step,
                output_grid=config.pde_grid_size,
            )
        raise ValueError(f"unknown arm: {arm}")

    # Model 1 ignores the substep, so that arm would be a duplicate.
    single_factor = (
        ("grid_only", "operator_only")
        if config.model == "linear"
        else ("substep_only", "grid_only", "operator_only")
    )
    arms = ("inverse_crime",) + single_factor + ("full_mismatch",)
    arm_descriptions = {
        "inverse_crime": (
            f"the shipped solver, {config.pde_grid_size} grid, substep "
            f"{config.candidate_solver_step} -- no mismatch at all"
        ),
        "substep_only": (
            f"the shipped solver and grid, substep {args.fine_step} "
            "-- time step alone"
        ),
        "grid_only": (
            f"the shipped solver and substep, {args.fine_grid} grid, "
            f"area-averaged back to {config.pde_grid_size} -- spatial grid alone"
        ),
        "operator_only": (
            f"five-point Laplacian with Crank-Nicolson diffusion, "
            f"{config.pde_grid_size} grid, substep "
            f"{config.candidate_solver_step} -- the spatial stencil and the "
            "diffusion integrator together, on the shipped grid and substep"
        ),
        "full_mismatch": (
            f"five-point Laplacian, Crank-Nicolson diffusion with explicit "
            f"reaction, {args.fine_grid} grid, substep {args.fine_step}, "
            f"area-averaged to {config.pde_grid_size} -- all of them at once"
        ),
    }
    rows = []
    total = len(cases) * len(args.ics) * len(arms)
    index = 0
    for truth in cases:
        for ic_name in args.ics:
            for arm in arms:
                index += 1
                estimate = recover(
                    make_target(arm, ic_name, truth),
                    initials[ic_name],
                    library,
                    scales,
                    config,
                    ic_name,
                )
                errors = signed_errors(estimate, truth)
                rows.append(
                    {
                        "arm": arm,
                        "case_id": truth["case_id"],
                        "ic_name": ic_name,
                        "true_rhat": truth["rhat"],
                        "true_dhat": truth["dhat"],
                        "true_alpha": truth["alpha"],
                        "estimated_rhat": estimate["rhat"],
                        "estimated_dhat": estimate["dhat"],
                        "estimated_alpha": estimate["alpha"],
                        "success": bool(estimate["success"]),
                        **errors,
                    }
                )
                print(
                    f"[{index}/{total}] {truth['case_id']} {ic_name:20s} "
                    f"{arm:14s} R_hat {errors['rhat_signed']:+.3%}  "
                    f"D_hat {errors['dhat_signed']:+.3%}",
                    flush=True,
                )

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / f"{stem('operator_mismatch', config, args.tag)}.csv", index=False)

    by_arm = {
        arm: summarize(group.to_dict("records"))
        for arm, group in table.groupby("arm")
    }
    write_json(
        {
            "study": "forward-operator mismatch",
            "model": config.model,
            "reaction_term_used_for_the_mismatched_targets": (
                "R_hat*rho" if config.model == "linear" else "R_hat*rho*(1-rho)"
            ),
            "arms": {name: arm_descriptions[name] for name in arms},
            "candidate_solver": (
                "exact Fourier propagator"
                if config.model == "linear"
                else (
                    "spectral backward-Euler diffusion with explicit reaction, "
                    f"{config.pde_grid_size} grid, substep "
                    f"{config.candidate_solver_step} of t_hat"
                )
            ),
            "initial_condition": (
                "piecewise-constant lift of the sealed 125-point IC, so the fine "
                "run starts from the identical field"
            ),
            "n_recoveries": len(rows),
            "by_arm": by_arm,
            "by_ic_full_mismatch": {
                name: summarize(group.to_dict("records"))
                for name, group in table[table.arm == "full_mismatch"].groupby("ic_name")
            },
        },
        RESULTS / f"{stem('operator_mismatch', config, args.tag)}.json",
    )


if __name__ == "__main__":
    main()
