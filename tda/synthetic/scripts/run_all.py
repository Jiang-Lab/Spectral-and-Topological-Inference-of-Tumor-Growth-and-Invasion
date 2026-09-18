#!/usr/bin/env python3
"""Run the main recovery, the controls and the studies for one model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model2.yaml"

MAIN = (("run_recovery.py", ["--resume"]),)
CONTROLS = (("run_phase_control.py", []),)
STUDIES = (
    ("run_operator_mismatch.py", []),
    ("run_ic_mismatch.py", []),
    ("run_robustness.py", []),
    ("run_scenarios.py", []),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="model config, forwarded to every step (default: configs/model2.yaml)",
    )
    parser.add_argument(
        "--skip-studies",
        action="store_true",
        help="run only the main recovery and the control",
    )
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help="skip run_recovery.py entirely (its results already exist)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands without running them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Path(args.config)
    if not config.is_file():
        raise SystemExit(f"config not found: {config}")

    # Read the one field directly, so this runs without the numerical stack.
    model = yaml.safe_load(config.read_text(encoding="utf-8"))["model"]
    steps = (
        (() if args.skip_main else MAIN)
        + CONTROLS
        + (() if args.skip_studies else STUDIES)
    )
    print(f"model: {model}", flush=True)
    for index, (name, extra) in enumerate(steps, start=1):
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / name),
            "--config",
            str(config),
            *extra,
        ]
        print(f"\n=== [{index}/{len(steps)}] {name} {' '.join(extra)} ===", flush=True)
        if args.dry_run:
            print("    " + " ".join(command[1:]), flush=True)
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
