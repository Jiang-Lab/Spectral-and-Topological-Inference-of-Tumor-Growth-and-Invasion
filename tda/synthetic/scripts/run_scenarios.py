#!/usr/bin/env python3
"""Recovery accuracy of each observation mode, on the same targets.

``single_snapshot`` is the mode a biopsy slice supplies; ``multi_frame``
is the mode a time-lapse movie supplies. Both are validated here.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _study_common import (
    DEFAULT_CONFIG,
    DIMENSIONS,
    IC_PATH,
    RESULTS,
    add_tag_argument,
    noisy_diagram,
    stem,
    truths,
    write_json,
)

from tda_synth import movie
from tda_synth.config import load_config
from tda_synth.initial_conditions import load_initial_conditions
from tda_synth.movie import (
    build_movie_library,
    estimate_movie_scales,
    recover_movie,
    simulate_movie,
)
from tda_synth.plotting import (
    DOUBLE,
    COLUMN,
    PANEL,
    field_panel,
    grouped_bars,
    shared_colorbar,
)
from tda_synth.scaling import canonical, report, to_physical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--ics",
        nargs="+",
        default=[
            "Random spaced",
            "One central cluster",
            "Many clusters",
            "Clusters + singles",
        ],
    )
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--modes", nargs="+", default=list(movie.DEFAULT_ORDER))
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--noise-seed", type=int, default=20260911)
    parser.add_argument(
        "--elapsed-time",
        type=float,
        default=None,
        help=(
            "MULTI-FRAME ONLY: the measured time of the final frame, in "
            "--time-unit. Supplying it converts R_hat and D_hat into R and D "
            "with units. A single snapshot never gets this conversion, because "
            "its elapsed time is not measured."
        ),
    )
    parser.add_argument("--domain-length", type=float, default=None,
                        help="domain side in --length-unit, needed with --elapsed-time")
    parser.add_argument("--time-unit", default="day")
    parser.add_argument("--length-unit", default="um")
    add_tag_argument(parser)
    return parser.parse_args()


SHOWCASE_IC = "Many clusters"


def save_frame_figure(config, initials, truth, args) -> None:
    """The multi_frame trajectory: target and recovered field at every frame."""
    ic_name = SHOWCASE_IC if SHOWCASE_IC in args.ics else args.ics[0]
    initial = initials[ic_name]
    frames = movie.frames_for("multi_frame")
    truth_fields = simulate_movie(
        initial, float(truth["rhat"]), float(truth["dhat"]), frames,
        model=config.model, solver_step=config.target_solver_step,
    )
    library = build_movie_library(initial, config, frames)
    scales = estimate_movie_scales(library, config, frames)
    target = tuple(
        noisy_diagram(field, config, args.noise,
                      args.noise_seed + int(round(frame * 1000)) * 131)
        for frame, field in zip(frames, truth_fields)
    )
    estimate, _ = recover_movie(
        target, initial, library, scales, config, frames, DIMENSIONS
    )
    found = simulate_movie(
        initial, estimate["rhat"], estimate["dhat"], frames,
        model=config.model, solver_step=config.candidate_solver_step,
    )
    gaps = [np.abs(a - b) for a, b in zip(truth_fields, found)]

    top = max(float(np.max(f)) for f in truth_fields + found)
    gap_top = max(float(np.max(g)) for g in gaps) or 1.0
    fig, axes = plt.subplots(3, len(frames),
                             figsize=(DOUBLE * 0.78, DOUBLE * 0.80),
                             constrained_layout=True)
    rows = (
        ("Simulated target", truth_fields, "magma", (0.0, top)),
        ("Recovered", found, "magma", (0.0, top)),
        ("Absolute difference", gaps, "viridis", (0.0, gap_top)),
    )
    letters = iter(PANEL)
    for row, (label, fields, cmap, limits) in zip(axes, rows):
        for index, (ax, frame, field) in enumerate(zip(row, frames, fields)):
            handle = field_panel(
                ax, field, title=f"$\\hat t$ = {frame:g}", cmap=cmap,
                vmin=limits[0], vmax=limits[1], panel=next(letters),
            )
            if index == 0:
                ax.set_ylabel(label, fontsize=8)
        bar = r"$|\Delta\rho|$" if cmap == "viridis" else r"$\rho$"
        shared_colorbar(fig, handle, row, bar)
    path = RESULTS / f"{stem('scenario_frames', config, args.tag)}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    if (args.elapsed_time is None) != (args.domain_length is None):
        raise SystemExit("--elapsed-time and --domain-length must be given together")
    config = load_config(args.config)
    initials = load_initial_conditions(IC_PATH)
    cases = truths(config, args.cases)

    rows = []
    for mode in args.modes:
        frames = movie.frames_for(mode)
        print(f"[{mode}] {len(frames)} field(s) at {list(frames)}", flush=True)
        for ic_name in args.ics:
            initial = initials[ic_name]
            library = build_movie_library(initial, config, frames)
            scales = estimate_movie_scales(library, config, frames)
            for case_index, truth in enumerate(cases):
                fields = simulate_movie(
                    initial,
                    float(truth["rhat"]),
                    float(truth["dhat"]),
                    frames,
                    model=config.model,
                    solver_step=config.target_solver_step,
                )
                # Seeded on the frame time, so t_hat = 1 draws the same noise
                # in every mode.
                target = tuple(
                    noisy_diagram(
                        field,
                        config,
                        args.noise,
                        args.noise_seed
                        + case_index * 7919
                        + int(round(frame * 1000)) * 131,
                    )
                    for frame, field in zip(frames, fields)
                )
                estimate, _ = recover_movie(
                    target, initial, library, scales, config, frames, DIMENSIONS
                )
                pair = canonical(estimate["rhat"], estimate["dhat"])
                found = report(estimate["rhat"], estimate["dhat"])
                true_pair = canonical(float(truth["rhat"]), float(truth["dhat"]))
                true_values = report(float(truth["rhat"]), float(truth["dhat"]))
                rows.append(
                    {
                        "mode": mode,
                        "n_frames": len(frames),
                        "ic_name": ic_name,
                        "case_id": truth["case_id"],
                        "true_rhat": true_values.rhat,
                        "true_dhat": true_values.dhat,
                        "true_tau": true_pair.tau,
                        "true_alpha": true_pair.alpha,
                        "estimated_rhat": found.rhat,
                        "estimated_dhat": found.dhat,
                        "estimated_tau": pair.tau,
                        "estimated_alpha": pair.alpha,
                        "estimated_dhat_over_rhat": pair.alpha,
                        "estimated_length_scale_ratio": found.length_scale_ratio,
                        "rhat_relative_error": abs(found.rhat - true_values.rhat)
                        / true_values.rhat,
                        "dhat_relative_error": abs(found.dhat - true_values.dhat)
                        / true_values.dhat,
                        "alpha_relative_error": abs(pair.alpha - true_pair.alpha)
                        / true_pair.alpha,
                    }
                )
                # Units only for a movie, and only after the search.
                if args.elapsed_time is not None and len(frames) > 1:
                    phys = to_physical(
                        found.rhat, found.dhat,
                        elapsed_time=args.elapsed_time,
                        domain_length=args.domain_length,
                        time_unit=args.time_unit,
                        length_unit=args.length_unit,
                    )
                    true_phys = to_physical(
                        true_values.rhat, true_values.dhat,
                        elapsed_time=args.elapsed_time,
                        domain_length=args.domain_length,
                        time_unit=args.time_unit,
                        length_unit=args.length_unit,
                    )
                    rows[-1].update(
                        {
                            "estimated_R": phys.growth_rate,
                            "estimated_D": phys.diffusivity,
                            "true_R": true_phys.growth_rate,
                            "true_D": true_phys.diffusivity,
                            "R_unit": f"1/{args.time_unit}",
                            "D_unit": f"{args.length_unit}^2/{args.time_unit}",
                        }
                    )
        done = [r for r in rows if r["mode"] == mode]
        print(
            f"    median |R_hat| {np.median([r['rhat_relative_error'] for r in done]):.4%}"
            f"   median |D_hat| {np.median([r['dhat_relative_error'] for r in done]):.4%}",
            flush=True,
        )

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / f"{stem('scenarios', config, args.tag)}.csv", index=False)

    def block(group):
        return {
            "n": int(len(group)),
            "n_frames": int(group["n_frames"].iloc[0]),
            "median_relative_error": {
                "R_hat": float(group["rhat_relative_error"].median()),
                "D_hat": float(group["dhat_relative_error"].median()),
                "alpha": float(group["alpha_relative_error"].median()),
            },
            "p90_relative_error": {
                "R_hat": float(group["rhat_relative_error"].quantile(0.90)),
                "D_hat": float(group["dhat_relative_error"].quantile(0.90)),
            },
        }

    write_json(
        {
            "study": "observation modes",
            "noise_level": args.noise,
            "n_cases": len(cases),
            "initial_conditions": list(args.ics),
            "declared": movie.table(tuple(args.modes)),
            "units": {
                "search": "dimensionless",
                "single_snapshot": "dimensionless; ratio = D_hat/R_hat",
                "multi_frame": (
                    f"R in 1/{args.time_unit}, D in {args.length_unit}^2/"
                    f"{args.time_unit}"
                    if args.elapsed_time is not None
                    else "dimensionless"
                ),
            },
            **(
                {
                    "physical_scales": {
                        "elapsed_time": args.elapsed_time,
                        "time_unit": args.time_unit,
                        "domain_length": args.domain_length,
                        "length_unit": args.length_unit,
                        "applies_to": "multi_frame only",
                    }
                }
                if args.elapsed_time is not None
                else {}
            ),
            "overall": {mode: block(g) for mode, g in table.groupby("mode")},
            "by_ic": {
                f"{mode}|{ic}": block(g)
                for (mode, ic), g in table.groupby(["mode", "ic_name"])
            },
        },
        RESULTS / f"{stem('scenarios', config, args.tag)}.json",
    )

    if "multi_frame" in args.modes:
        save_frame_figure(config, initials, cases[0], args)

    modes = [m for m in args.modes if m in set(table["mode"])]
    if len(modes) > 1:
        fig, ax = plt.subplots(figsize=(COLUMN * 1.25, COLUMN * 0.8),
                               constrained_layout=True)
        grouped_bars(
            ax,
            [m.replace("_", " ") for m in modes],
            {
                r"$\widehat{R}$": [
                    100 * table[table["mode"] == m]["rhat_relative_error"].median()
                    for m in modes
                ],
                r"$\widehat{D}$": [
                    100 * table[table["mode"] == m]["dhat_relative_error"].median()
                    for m in modes
                ],
            },
            ylabel="median absolute error (%)",
            title=f"Accuracy of each observation mode, noise {args.noise:g}",
        )
        fig.savefig(RESULTS / f"{stem('scenarios', config, args.tag)}.png")
        plt.close(fig)
        print(f"wrote {RESULTS / (stem('scenarios', config, args.tag) + '.png')}")


if __name__ == "__main__":
    main()
