#!/usr/bin/env python3
"""Dimensionless parameter recovery from one simulated snapshot at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import gudhi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tda_synth.config import load_config, model_tag  # noqa: E402
from tda_synth.initial_conditions import load_initial_conditions  # noqa: E402
from tda_synth.observation import observe_field  # noqa: E402
from tda_synth.pde import simulate_snapshot  # noqa: E402
from tda_synth.persistence import persistence_descriptor  # noqa: E402
from tda_synth.plotting import (  # noqa: E402
    PANEL,
    field_panel,
    plot_diagram,
    plot_loss_map,
    shared_colorbar,
)
from tda_synth.recovery import (  # noqa: E402
    FEATURE_SETS,
    PRIMARY_FEATURE_SET,
    build_library,
    diagram_loss,
    estimate_scales,
    make_offgrid_holdouts,
    recover_continuous,
)
from tda_synth.scaling import report  # noqa: E402

SHOWCASE_IC = "Many clusters"

MODEL_EQUATION = {
    "linear": "d(rho)/d(t_hat) = D_hat*lap_hat(rho) + R_hat*rho",
    "logistic": "d(rho)/d(t_hat) = D_hat*lap_hat(rho) + R_hat*rho*(1-rho)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model2.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "output suffix. Defaults to the config's model, so Model 1 and "
            "Model 2 results never overwrite each other."
        ),
    )
    return parser.parse_args()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def save_field_figure(
    initials: dict[str, np.ndarray],
    target: np.ndarray,
    estimate: np.ndarray,
    output_path: Path,
) -> None:
    """Initial conditions on a 0-1 scale; the evolved fields on their own."""
    fig = plt.figure(figsize=(7.2, 4.2), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.0])

    ic_axes = top.subplots(1, len(initials))
    for index, (ax, (name, rho)) in enumerate(zip(ic_axes, initials.items())):
        handle = field_panel(ax, rho, title=name, cmap="magma",
                             vmin=0, vmax=1, panel=PANEL[index])
    shared_colorbar(top, handle, ic_axes, r"$\rho$  initial")

    difference = np.abs(target - estimate)
    scale = max(float(np.max(target)), float(np.max(estimate)), 1.0)
    ev_axes = bottom.subplots(1, 3)
    for index, (ax, title, rho, cmap, vmax) in enumerate(zip(
        ev_axes,
        ["Simulated target", "TDA estimate", "Absolute difference"],
        [target, estimate, difference],
        ["viridis", "viridis", "magma"],
        [scale, scale, float(np.max(difference)) or 1.0],
    )):
        h = field_panel(ax, rho, title=title, cmap=cmap, vmin=0, vmax=vmax,
                        panel=PANEL[len(initials) + index])
        if index == 1:
            shared_colorbar(bottom, h, ev_axes[:2], r"$\rho$  observed")
        if index == 2:
            shared_colorbar(bottom, h, [ax], r"$|\Delta\rho|$")
    fig.savefig(output_path)
    plt.close(fig)


def save_diagram_csv(diagram: dict[int, np.ndarray], path: Path) -> None:
    rows: list[tuple[int, float, float, float]] = []
    for dimension, intervals in sorted(diagram.items()):
        for birth, death in intervals:
            rows.append((dimension, float(birth), float(death), float(death - birth)))
    table = pd.DataFrame(
        rows, columns=["homology_dimension", "birth", "death", "persistence"]
    )
    table.to_csv(path, index=False)


def summarize(table: pd.DataFrame) -> dict:
    output: dict[str, dict] = {}
    for feature_name, group in table.groupby("feature_set", sort=False):
        output[str(feature_name)] = {
            "n_single_snapshot_recoveries": int(len(group)),
            "median_relative_error": {
                "R_hat": float(group["rhat_relative_error"].median()),
                "D_hat": float(group["dhat_relative_error"].median()),
                "alpha": float(group["alpha_relative_error"].median()),
            },
            "p90_relative_error": {
                "R_hat": float(group["rhat_relative_error"].quantile(0.90)),
                "D_hat": float(group["dhat_relative_error"].quantile(0.90)),
                "alpha": float(group["alpha_relative_error"].quantile(0.90)),
            },
            "maximum_relative_error": {
                "R_hat": float(group["rhat_relative_error"].max()),
                "D_hat": float(group["dhat_relative_error"].max()),
                "alpha": float(group["alpha_relative_error"].max()),
            },
            "both_parameters_under_10_percent": float(
                np.mean(
                    (group["rhat_relative_error"].to_numpy() < 0.10)
                    & (group["dhat_relative_error"].to_numpy() < 0.10)
                )
            ),
            "median_signed_relative_error": {
                "R_hat": float(group["rhat_signed_relative_error"].median()),
                "D_hat": float(group["dhat_signed_relative_error"].median()),
            },
            "fraction_overestimated": {
                "R_hat": float((group["rhat_signed_relative_error"] > 0).mean()),
                "D_hat": float((group["dhat_signed_relative_error"] > 0).mean()),
            },
            "formal_optimizer_success_fraction": float(group["success"].mean()),
            "median_loss_at_true_parameters": float(
                group["loss_at_true_parameters"].median()
            ),
            "estimate_loss_not_above_truth_fraction": float(
                group["estimate_loss_not_above_truth"].mean()
            ),
        }
    return output


def run(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    tag = args.tag or model_tag(config)
    result_dir = PROJECT_ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    checkpoint_path = result_dir / f"recovery_checkpoint_{tag}.json"

    config_payload = config.to_dict()
    config_sha = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    # An edit to any of these invalidates a checkpoint that --resume would reuse.
    method_files = tuple(
        PROJECT_ROOT / "tda_synth" / name
        for name in (
            "config.py",
            "initial_conditions.py",
            "movie.py",
            "observation.py",
            "pde.py",
            "persistence.py",
            "recovery.py",
            "scaling.py",
        )
    )
    method_digest = hashlib.sha256()
    for method_file in method_files:
        method_digest.update(method_file.name.encode("utf-8"))
        method_digest.update(method_file.read_bytes())
    method_sha = method_digest.hexdigest()
    initials = load_initial_conditions(
        PROJECT_ROOT / "data" / "somiya_ic_125.npz"
    )
    print(f"Building {len(initials)} separate IC libraries...", flush=True)
    library = build_library(initials, config)
    scales = estimate_scales(library, config.homology_dimensions, config=config)

    truths = make_offgrid_holdouts(config, config.n_holdouts)
    state: dict = {
        "config_sha256": config_sha,
        "method_sha256": method_sha,
        "completed": [],
        "records": [],
    }
    if args.resume and checkpoint_path.exists():
        loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if loaded.get("config_sha256") != config_sha:
            raise RuntimeError("checkpoint configuration does not match")
        if loaded.get("method_sha256") != method_sha:
            raise RuntimeError("checkpoint recovery code does not match")
        state = loaded

    completed = set(state["completed"])
    showcase_surface: pd.DataFrame | None = None
    showcase: dict | None = None
    total = len(truths) * len(initials) * len(FEATURE_SETS)
    sequence = 0
    for truth in truths:
        for ic_name, rho_initial in initials.items():
            pending = [
                name
                for name in FEATURE_SETS
                if f"{truth['case_id']}|{ic_name}|{name}" not in completed
            ]
            if not pending:
                sequence += len(FEATURE_SETS)
                continue
            target_field = simulate_snapshot(
                rho_initial,
                float(truth["rhat"]),
                float(truth["dhat"]),
                model=config.model,
                solver_step=config.target_solver_step,
            )
            target_diagram = persistence_descriptor(target_field, config)
            for feature_name, dimensions in FEATURE_SETS.items():
                sequence += 1
                record_id = f"{truth['case_id']}|{ic_name}|{feature_name}"
                if record_id in completed:
                    continue
                print(f"[{sequence}/{total}] {record_id}", flush=True)
                estimate, surface = recover_continuous(
                    target_field,
                    rho_initial,
                    library,
                    scales,
                    config,
                    ic_name,
                    dimensions,
                )
                candidate_at_truth = simulate_snapshot(
                    rho_initial,
                    float(truth["rhat"]),
                    float(truth["dhat"]),
                    model=config.model,
                    solver_step=config.candidate_solver_step,
                )
                truth_loss, _, _ = diagram_loss(
                    target_diagram,
                    persistence_descriptor(candidate_at_truth, config),
                    scales,
                    ic_name,
                    dimensions,
                )
                true_rhat = float(truth["rhat"])
                true_dhat = float(truth["dhat"])
                true_alpha = true_dhat / true_rhat
                estimated_rhat = float(estimate["rhat"])
                estimated_dhat = float(estimate["dhat"])
                estimated_alpha = float(estimate["alpha"])
                recovered = report(estimated_rhat, estimated_dhat)
                row = {
                    "record_id": record_id,
                    "case_id": str(truth["case_id"]),
                    "ic_name": ic_name,
                    "feature_set": feature_name,
                    "single_snapshot": True,
                    # Absent from Model 1's CSV rather than present and blank.
                    **({"k_fixed": 1.0} if config.model == "logistic" else {}),
                    "true_rhat": true_rhat,
                    "true_dhat": true_dhat,
                    "true_alpha": true_alpha,
                    "true_tau": true_rhat,
                    "estimated_rhat": estimated_rhat,
                    "estimated_dhat": estimated_dhat,
                    "estimated_alpha": estimated_alpha,
                    "estimated_dhat_over_rhat": estimated_alpha,
                    "estimated_tau": estimated_rhat,
                    "rhat_relative_error": abs(estimated_rhat - true_rhat) / true_rhat,
                    "dhat_relative_error": abs(estimated_dhat - true_dhat) / true_dhat,
                    "alpha_relative_error": abs(estimated_alpha - true_alpha)
                    / true_alpha,
                    "rhat_signed_relative_error": (estimated_rhat - true_rhat)
                    / true_rhat,
                    "dhat_signed_relative_error": (estimated_dhat - true_dhat)
                    / true_dhat,
                    "dimensions_used": "".join(
                        f"h{q}" for q in estimate["dimensions_used"]
                    ),
                    "estimated_length_scale_ratio": recovered.length_scale_ratio,
                    "loss": float(estimate["loss"]),
                    "loss_at_true_parameters": float(truth_loss),
                    "estimate_loss_not_above_truth": bool(
                        float(estimate["loss"]) <= float(truth_loss) + 1e-12
                    ),
                    "success": bool(estimate["success"]),
                    "any_start_succeeded": bool(estimate["any_start_succeeded"]),
                    "coarse_rhat": float(estimate["coarse_rhat"]),
                    "coarse_dhat": float(estimate["coarse_dhat"]),
                    "optimizer": estimate["all_starts"],
                }
                state["records"].append(row)
                state["completed"].append(record_id)
                completed.add(record_id)
                atomic_json(state, checkpoint_path)
                if (
                    showcase is None
                    and feature_name == PRIMARY_FEATURE_SET
                    and ic_name == SHOWCASE_IC
                ):
                    showcase = {
                        "truth": truth,
                        "ic_name": ic_name,
                        "target_field": target_field,
                        "target_diagram": target_diagram,
                        "estimate": estimate,
                    }
                    showcase_surface = surface

    table = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "optimizer"}
            for row in state["records"]
        ]
    )
    table.to_csv(result_dir / f"recovery_cases_{tag}.csv", index=False)
    summary = {
        "analysis": (
            f"dimensionless {config.model} recovery from one spatial snapshot"
        ),
        "model": config.model,
        "equation": MODEL_EQUATION[config.model],
        "one_snapshot_per_recovery": True,
        "density_symbol": "rho",
        **(
            {
                "carrying_capacity_definition": (
                    "rho=1 is the local maximum capacity; K is absorbed into "
                    "rho_hat = rho/K, which is why it equals one here"
                ),
                "K": 1.0,
                "K_is_recovered": False,
            }
            if config.model == "logistic"
            else {
                "carrying_capacity_definition": (
                    "none: the linear model has no carrying capacity and rho "
                    "is not bounded above"
                ),
                "K": None,
            }
        ),
        "parameters": ["R_hat", "D_hat", "tau", "alpha", "length_scale_ratio"],
        "parameter_definition": {
            "equation": MODEL_EQUATION[config.model],
            "R_hat": "R * T",
            "D_hat": "D * T / L0**2",
            "tau": "R_hat, written as tau in the (tau, alpha) pair",
            "alpha": "D_hat / R_hat, the ratio; equals D / (R * L0**2)",
            "T": "the reference time; not measured from one snapshot",
            "reference": "tda_synth/scaling.py",
        },
        "units": {
            "all": "dimensionless",
        },
        "four_ICs_are_separate_benchmarks": list(initials),
        "target_and_candidate_solver": (
            "exact Fourier propagator"
            if config.model == "linear"
            else "same spectral IMEX solver"
        ),
        "target_solver_step": config.target_solver_step,
        "candidate_solver_step": config.candidate_solver_step,
        "truths_are_strictly_off_grid": True,
        "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
        "metrics": summarize(table),
        "config": config_payload,
        "config_sha256": config_sha,
        "method_sha256": method_sha,
        "distance_scales": {
            f"{ic_name}|H{dimension}": value
            for (ic_name, dimension), value in scales.items()
        },
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "gudhi": gudhi.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    atomic_json(summary, result_dir / f"recovery_summary_{tag}.json")

    if showcase is None:
        primary_rows = sorted(
            (
                row
                for row in state["records"]
                if row["feature_set"] == PRIMARY_FEATURE_SET
                and row["ic_name"] == SHOWCASE_IC
            ),
            key=lambda row: row["case_id"],
        ) or sorted(
            (
                row
                for row in state["records"]
                if row["feature_set"] == PRIMARY_FEATURE_SET
            ),
            key=lambda row: (row["ic_name"], row["case_id"]),
        )
        if not primary_rows:
            raise RuntimeError("no primary recovery is available for figures")
        # Same criterion as the live path, so --resume picks the same case.
        selected = primary_rows[0]
        truth = next(row for row in truths if row["case_id"] == selected["case_id"])
        ic_name = str(selected["ic_name"])
        rho_initial = initials[ic_name]
        target_field = simulate_snapshot(
            rho_initial,
            float(truth["rhat"]),
            float(truth["dhat"]),
            model=config.model,
            solver_step=config.target_solver_step,
        )
        estimate, showcase_surface = recover_continuous(
            target_field,
            rho_initial,
            library,
            scales,
            config,
            ic_name,
            FEATURE_SETS[PRIMARY_FEATURE_SET],
        )
        showcase = {
            "truth": truth,
            "ic_name": ic_name,
            "target_field": target_field,
            "target_diagram": persistence_descriptor(target_field, config),
            "estimate": estimate,
        }

    assert showcase_surface is not None
    showcase_surface.to_csv(result_dir / f"recovery_surface_{tag}.csv", index=False)
    best_field = simulate_snapshot(
        initials[str(showcase["ic_name"])],
        float(showcase["estimate"]["rhat"]),
        float(showcase["estimate"]["dhat"]),
        model=config.model,
        solver_step=config.candidate_solver_step,
    )
    target_observed = observe_field(
        showcase["target_field"],
        output_grid_size=config.observation_grid_size,
        sigma_hat=config.observation_sigma_hat,
    )
    best_observed = observe_field(
        best_field,
        output_grid_size=config.observation_grid_size,
        sigma_hat=config.observation_sigma_hat,
    )
    np.savez_compressed(
        result_dir / f"recovery_fields_{tag}.npz",
        target=showcase["target_field"],
        estimate=best_field,
        target_observed=target_observed,
        estimate_observed=best_observed,
        **{f"ic_{index}": value for index, value in enumerate(initials.values())},
    )
    save_diagram_csv(
        showcase["target_diagram"],
        result_dir / f"recovery_target_persistence_{tag}.csv",
    )
    figure, _ = plot_loss_map(
        showcase_surface, result_dir / f"recovery_loss_map_{tag}.png"
    )
    plt.close(figure)
    figure, _ = plot_diagram(
        showcase["target_diagram"],
        result_dir / f"recovery_target_persistence_{tag}.png",
    )
    plt.close(figure)
    save_field_figure(
        initials,
        showcase["target_field"],
        best_field,
        result_dir / f"recovery_fields_{tag}.png",
    )
    print(json.dumps(summary["metrics"], indent=2), flush=True)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
