#!/usr/bin/env python3
"""Negative control: a Fourier-phase scramble with the power spectrum kept."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from _study_common import add_tag_argument, model_tag, stem  # noqa: E402
from tda_synth.config import load_config  # noqa: E402
from tda_synth.initial_conditions import load_initial_conditions  # noqa: E402
from tda_synth.observation import reference_normalize  # noqa: E402
from tda_synth.persistence import (  # noqa: E402
    finite_diagrams,
    wasserstein2,
)
from tda_synth.plotting import (  # noqa: E402
    DOUBLE,
    GRID,
    PANEL,
    field_panel,
    series_color,
    shared_colorbar,
)
from tda_synth.recovery import build_library, estimate_scales  # noqa: E402

IC_NAME = "Many clusters"


def phase_scramble_real(field: np.ndarray, seed: int) -> np.ndarray:
    """Randomize phase while preserving the Fourier magnitude. Not clipped, so
    the result may go negative."""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fft2(field)
    noise_spectrum = np.fft.fft2(rng.standard_normal(field.shape))
    phase = np.ones_like(noise_spectrum, dtype=complex)
    nonzero = np.abs(noise_spectrum) > 0
    phase[nonzero] = noise_spectrum[nonzero] / np.abs(noise_spectrum[nonzero])
    if abs(spectrum[0, 0]) > 0:
        phase[0, 0] = spectrum[0, 0] / abs(spectrum[0, 0])
    return np.real(np.fft.ifft2(np.abs(spectrum) * phase))


def radial_psd(field: np.ndarray):
    shifted = np.fft.fftshift(np.abs(np.fft.fft2(field)) ** 2)
    yy, xx = np.indices(field.shape)
    cy, cx = (np.asarray(field.shape) - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    sums = np.bincount(radius.ravel(), weights=shifted.ravel())
    counts = np.bincount(radius.ravel())
    return np.arange(len(sums)), sums / np.maximum(counts, 1)


def descriptor_from_observed(field: np.ndarray, config):
    normalized = reference_normalize(field, config.rho_ref)
    return finite_diagrams(
        config.filtration_reference - normalized,
        dimensions=config.homology_dimensions,
        cutoff=config.persistence_cutoff,
    )


def phase_scramble_control(args):
    config = load_config(args.config)
    tag = args.tag or model_tag(config)
    fields_path = PROJECT_ROOT / "results" / f"recovery_fields_{tag}.npz"
    if not fields_path.exists():
        raise SystemExit(
            f"{fields_path.name} not found. Run\n"
            f"    python scripts/run_recovery.py --config {args.config}\n"
            "first: this control reads the target field that run produced."
        )
    with np.load(fields_path) as archive:
        original = archive["target_observed"].astype(float)
    scrambled = phase_scramble_real(original, args.seed)

    original_psd = np.abs(np.fft.fft2(original)) ** 2
    scrambled_psd = np.abs(np.fft.fft2(scrambled)) ** 2
    psd_relative_error = float(
        np.linalg.norm(scrambled_psd - original_psd) / np.linalg.norm(original_psd)
    )
    original_diagram = descriptor_from_observed(original, config)
    scrambled_diagram = descriptor_from_observed(scrambled, config)
    tda_distances = {
        f"H{dimension}": wasserstein2(
            original_diagram[dimension], scrambled_diagram[dimension]
        )
        for dimension in config.homology_dimensions
    }

    # In units of the median pairwise distance inside the candidate library.
    initials = load_initial_conditions(
        PROJECT_ROOT / "data" / "somiya_ic_125.npz"
    )
    library = build_library({IC_NAME: initials[IC_NAME]}, config)
    scales = estimate_scales(library, config.homology_dimensions, config=config)
    library_scale = {
        f"H{dimension}": scales[(IC_NAME, dimension)]
        for dimension in config.homology_dimensions
    }
    # A zero scale would make the ratios inf.
    in_scale_units = {
        key: (
            tda_distances[key] / library_scale[key]
            if library_scale[key] > 1e-30
            else float("nan")
        )
        for key in tda_distances
    }
    if any(value <= 1e-30 for value in library_scale.values()):
        print(
            "WARNING: a library diagram scale is zero, so the ratios below "
            "are not meaningful; check the candidate library.",
            flush=True,
        )

    output = PROJECT_ROOT / "results"
    np.savez_compressed(
        output / f"{stem('phase_scramble_fields', config, tag)}.npz",
        original=original,
        scrambled=scrambled,
    )
    summary = {
        "control": "Fourier phase scramble",
        "single_spatial_snapshot": True,
        "fourier_power_relative_l2_error": psd_relative_error,
        "tda_wasserstein": tda_distances,
        "library_median_diagram_scale": library_scale,
        "tda_wasserstein_in_units_of_library_scale": in_scale_units,
        "scrambled_field_range": [float(scrambled.min()), float(scrambled.max())],
        "original_field_range": [float(original.min()), float(original.max())],
    }
    (output / f"{stem('phase_scramble_control', config, tag)}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    radii_a, radial_a = radial_psd(original)
    radii_b, radial_b = radial_psd(scrambled)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, DOUBLE * 0.34), constrained_layout=True)
    top = max(float(original.max()), float(scrambled.max()))
    bottom = min(float(original.min()), float(scrambled.min()))
    for index, (ax, image, title) in enumerate(
        zip(axes[:2], (original, scrambled), ("Original", "Phase-scrambled"))
    ):
        handle = field_panel(
            ax, image, title=title, cmap="magma", vmin=bottom, vmax=top, panel=PANEL[index]
        )
    shared_colorbar(fig, handle, axes[:2], r"$\rho$")
    spectrum = axes[2]
    spectrum.semilogy(radii_a[1:], radial_a[1:], color=series_color(0))
    spectrum.semilogy(
        radii_b[1:], radial_b[1:], color=series_color(1), linestyle=(0, (3, 2))
    )
    spectrum.set_xlabel("radial frequency bin")
    spectrum.set_ylabel("power")
    spectrum.set_title(f"({PANEL[2]})  Radial PSD", loc="left", pad=4)
    spectrum.legend(
        spectrum.get_lines(), ["original", "phase-scrambled"], loc="lower left"
    )
    spectrum.grid(color=GRID, linewidth=0.5)
    spectrum.set_axisbelow(True)
    fig.savefig(output / f"{stem('phase_scramble_control', config, tag)}.png")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "model2.yaml")
    parser.add_argument("--seed", type=int, default=20260909)
    add_tag_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_scramble_control(args)


if __name__ == "__main__":
    main()
