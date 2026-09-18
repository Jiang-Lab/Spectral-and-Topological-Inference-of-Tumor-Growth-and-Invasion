#!/usr/bin/env python3
"""Recovery when the true initial condition is withheld from the candidates."""

from __future__ import annotations

import argparse

import pandas as pd

from _study_common import (
    DEFAULT_CONFIG,
    DIMENSIONS,
    RESULTS,
    add_tag_argument,
    setup,
    signed_errors,
    stem,
    summarize,
    truths,
    write_json,
)

from tda_synth.pde import simulate_snapshot
from tda_synth.recovery import recover_continuous


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
    parser.add_argument("--n-holdouts", type=int, default=None)
    add_tag_argument(parser)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    config, initials, library, scales = setup(tuple(args.ics), args.config)
    count = args.n_holdouts or config.n_holdouts

    rows: list[dict] = []
    errors: list[dict[str, float]] = []
    cases = truths(config, count)
    total = len(cases) * len(initials)
    index = 0
    for truth in cases:
        for true_ic, rho_initial in initials.items():
            index += 1
            target = simulate_snapshot(
                rho_initial,
                float(truth["rhat"]),
                float(truth["dhat"]),
                model=config.model,
                solver_step=config.target_solver_step,
            )
            held_out = {name: field for name, field in initials.items() if name != true_ic}
            held_out_library = {name: library[name] for name in held_out}
            estimate, _ = recover_continuous(
                target, held_out, held_out_library, scales, config, DIMENSIONS
            )
            error = signed_errors(estimate, truth)
            errors.append(error)
            rows.append(
                {
                    "case_id": str(truth["case_id"]),
                    "true_ic_name": true_ic,
                    "recovered_ic_name": str(estimate["ic_name"]),
                    "true_rhat": float(truth["rhat"]),
                    "true_dhat": float(truth["dhat"]),
                    "true_alpha": float(truth["dhat"]) / float(truth["rhat"]),
                    "estimated_rhat": float(estimate["rhat"]),
                    "estimated_dhat": float(estimate["dhat"]),
                    "estimated_alpha": float(estimate["alpha"]),
                    **error,
                    "loss": float(estimate["loss"]),
                }
            )
            print(
                f"[{index}/{total}] {truth['case_id']} {true_ic:20s} "
                f"-> {estimate['ic_name']:20s} "
                f"alpha_err={error['alpha_signed']:+.3f}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / f"{stem('ic_mismatch', config, args.tag)}.csv", index=False)
    write_json(
        {
            "n_recoveries": int(len(rows)),
            "ic_recovered_correctly_fraction": float(
                (table["true_ic_name"] == table["recovered_ic_name"]).mean()
            ),
            "errors": summarize(errors),
        },
        RESULTS / f"{stem('ic_mismatch', config, args.tag)}.json",
    )


if __name__ == "__main__":
    run(parse_args())
