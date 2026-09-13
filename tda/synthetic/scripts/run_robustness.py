#!/usr/bin/env python3
"""Recovery under measurement noise, and under a smoothing-width mismatch."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _study_common import (
    DEFAULT_CONFIG,
    RESULTS,
    add_tag_argument,
    noisy_diagram,
    recover,
    setup,
    signed_errors,
    stem,
    summarize,
    truths,
    write_json,
)

from tda_synth.observation import observe_field, reference_normalize
from tda_synth.pde import simulate_snapshot
from tda_synth.persistence import finite_diagrams
from tda_synth.plotting import sensitivity_plot
from tda_synth.scaling import length_scale_ratio

NOISE_SEED_BASE = 20260910


def diagram_with_sigma(field, config, sigma_hat):
    observed = observe_field(
        field,
        output_grid_size=config.observation_grid_size,
        sigma_hat=sigma_hat,
    )
    normalized = reference_normalize(observed, config.rho_ref)
    return finite_diagrams(
        config.filtration_reference - normalized,
        dimensions=config.homology_dimensions,
        cutoff=config.persistence_cutoff,
    )


def noise_study(args, config, initials, library, scales):
    rho_initial = initials[args.ic]
    cases = truths(config, args.cases)

    clean = {
        truth["case_id"]: simulate_snapshot(
            rho_initial,
            float(truth["rhat"]),
            float(truth["dhat"]),
            model=config.model,
            solver_step=config.target_solver_step,
        )
        for truth in cases
    }

    rows = []
    for level_index, eta in enumerate(args.noise_levels):
        reps = 1 if eta == 0.0 else args.reps
        for case_index, truth in enumerate(cases):
            for rep in range(reps):
                # The case index enters the seed, so repetitions are independent.
                seed = (
                    NOISE_SEED_BASE
                    + level_index * 100_003
                    + rep * 997
                    + case_index * 7919
                )
                estimate = recover(
                    noisy_diagram(
                        clean[truth["case_id"]],
                        config,
                        eta,
                        seed,
                        after_smoothing=args.after_smoothing,
                    ),
                    rho_initial,
                    library,
                    scales,
                    config,
                    args.ic,
                    diagram=True,
                )
                rows.append(
                    {
                        "eta": eta,
                        "case_id": truth["case_id"],
                        "rep": rep,
                        **signed_errors(estimate, truth),
                    }
                )
        done = [r for r in rows if r["eta"] == eta]
        print(
            f"eta={eta:<5g} n={len(done):3d} "
            f"median|R_hat| {np.median([abs(r['rhat_signed']) for r in done]):.3%}  "
            f"median|D_hat| {np.median([abs(r['dhat_signed']) for r in done]):.3%}",
            flush=True,
        )

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / f"{stem('noise_study', config, args.tag)}.csv", index=False)
    by_eta = {
        eta: summarize(group.to_dict("records"))
        for eta, group in table.groupby("eta")
    }
    levels = sorted(by_eta)
    write_json(
        {
            "study": "measurement noise",
            "noise_model": (
                "additive Gaussian, standard deviation eta * (p99 - p01) of the "
                + (
                    f"observed {config.observation_grid_size}-point image"
                    if args.after_smoothing
                    else f"{config.pde_grid_size}-point density field"
                )
                + ", clipped at zero"
            ),
            "applied_after_smoothing": bool(args.after_smoothing),
            "ic_name": args.ic,
            "n_cases": len(cases),
            "reps_per_case": args.reps,
            "by_eta": {str(eta): by_eta[eta] for eta in levels},
        },
        RESULTS / f"{stem('noise_study', config, args.tag)}.json",
    )

    figure_path = RESULTS / f"{stem('noise_study', config, args.tag)}.png"
    figure, _ = sensitivity_plot(
        levels,
        {
            r"$\widehat R$": [by_eta[e]["rhat"]["median_absolute"] for e in levels],
            r"$\widehat D$": [by_eta[e]["dhat"]["median_absolute"] for e in levels],
            r"$\alpha$": [by_eta[e]["alpha"]["median_absolute"] for e in levels],
        },
        xlabel=r"relative noise level $\eta$",
        ylabel="median absolute error (%)",
        title="Measurement noise",
        output_path=figure_path,
    )
    plt.close(figure)
    print(f"wrote {figure_path}")


def sigma_sensitivity(args, config, initials, library, scales):
    rho_initial = initials[args.ic]
    cases = truths(config, args.cases)

    rows = []
    for factor in args.factors:
        sigma_true = factor * config.observation_sigma_hat
        for truth in cases:
            field = simulate_snapshot(
                rho_initial,
                float(truth["rhat"]),
                float(truth["dhat"]),
                model=config.model,
                solver_step=config.target_solver_step,
            )
            estimate = recover(
                diagram_with_sigma(field, config, sigma_true),
                rho_initial,
                library,
                scales,
                config,
                args.ic,
                diagram=True,
            )
            rows.append(
                {
                    "sigma_factor": factor,
                    "sigma_hat_true": sigma_true,
                    "sigma_hat_assumed": config.observation_sigma_hat,
                    "case_id": truth["case_id"],
                    "true_length_scale_ratio": length_scale_ratio(
                        float(truth["rhat"]), float(truth["dhat"])
                    ),
                    **signed_errors(estimate, truth),
                }
            )
        done = [r for r in rows if r["sigma_factor"] == factor]
        print(
            f"sigma x{factor:<5g}  median R_hat {np.median([r['rhat_signed'] for r in done]):+.3%}"
            f"  median D_hat {np.median([r['dhat_signed'] for r in done]):+.3%}",
            flush=True,
        )

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / f"{stem('sigma_sensitivity', config, args.tag)}.csv", index=False)
    by_factor = {
        factor: summarize(group.to_dict("records"))
        for factor, group in table.groupby("sigma_factor")
    }
    factors = sorted(by_factor)

    # Least-squares slope of the D_hat bias against the sigma error.
    sigma_error = np.array(factors, dtype=float) - 1.0
    dhat_bias = np.array(
        [by_factor[f]["dhat"]["median_signed"] for f in factors], dtype=float
    )
    slope = float(np.polyfit(sigma_error, dhat_bias, 1)[0])

    median_length_scale = float(table["true_length_scale_ratio"].median())
    write_json(
        {
            "study": "observation-width misspecification",
            "ic_name": args.ic,
            "n_cases": len(cases),
            "sigma_hat_assumed": config.observation_sigma_hat,
            "median_true_length_scale_ratio": median_length_scale,
            "length_scale_to_sigma_ratio": median_length_scale
            / config.observation_sigma_hat,
            "dhat_bias_per_unit_sigma_error": slope,
            "by_sigma_factor": {str(f): by_factor[f] for f in factors},
        },
        RESULTS / f"{stem('sigma_sensitivity', config, args.tag)}.json",
    )

    figure_path = RESULTS / f"{stem('sigma_sensitivity', config, args.tag)}.png"
    figure, _ = sensitivity_plot(
        factors,
        {
            r"$\widehat R$": [by_factor[f]["rhat"]["median_signed"] for f in factors],
            r"$\widehat D$": [by_factor[f]["dhat"]["median_signed"] for f in factors],
            r"$\alpha$": [by_factor[f]["alpha"]["median_signed"] for f in factors],
        },
        xlabel=r"true $\sigma$ as a multiple of the assumed $\sigma$",
        ylabel="median signed error (%)",
        title="Smoothing-width mismatch",
        reference_x=1.0,
        output_path=figure_path,
    )
    plt.close(figure)
    print(f"wrote {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ic", default="Many clusters")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10, 0.15]
    )
    parser.add_argument(
        "--after-smoothing",
        action="store_true",
        help="add the noise to the smoothed image instead of to the density field",
    )
    parser.add_argument(
        "--factors", type=float, nargs="+", default=[0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2],
        help="true sigma as a multiple of the sigma the candidates assume",
    )
    add_tag_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, initials, library, scales = setup((args.ic,), args.config)
    noise_study(args, config, initials, library, scales)
    sigma_sensitivity(args, config, initials, library, scales)


if __name__ == "__main__":
    main()
