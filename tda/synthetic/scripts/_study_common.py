"""Shared setup for the study scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tda_synth.config import (  # noqa: E402
    RecoveryConfig,
    load_config,
    model_tag,
)
from tda_synth.initial_conditions import load_initial_conditions  # noqa: E402
from tda_synth.observation import (  # noqa: E402
    observe_field,
    reference_normalize,
)
from tda_synth.persistence import finite_diagrams  # noqa: E402
from tda_synth.recovery import (  # noqa: E402
    FEATURE_SETS,
    PRIMARY_FEATURE_SET,
    build_library,
    estimate_scales,
    make_offgrid_holdouts,
    recover_continuous,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model2.yaml"

IC_PATH = PROJECT_ROOT / "data" / "somiya_ic_125.npz"
RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(exist_ok=True)  # a study may write a CSV before its first JSON
DIMENSIONS = FEATURE_SETS[PRIMARY_FEATURE_SET]


def stem(base: str, config, tag: str | None = None) -> str:
    """Output stem for ``base``, tagged by model unless ``tag`` overrides it."""
    return f"{base}_{tag or model_tag(config)}"


def add_tag_argument(parser) -> None:
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "output suffix. Defaults to the config's model, so Model 1 and "
            "Model 2 results never overwrite each other. Pass something else "
            "for an exploratory run."
        ),
    )


def setup(
    ic_names: tuple[str, ...],
    config_path: Path = DEFAULT_CONFIG,
    initials: dict[str, np.ndarray] | None = None,
):
    """Load config and ICs and build the shared candidate library."""
    config = load_config(config_path)
    if initials is None:
        initials = load_initial_conditions(IC_PATH)
    selected = {name: initials[name] for name in ic_names}
    library = build_library(selected, config)
    scales = estimate_scales(library, config.homology_dimensions, config=config)
    return config, selected, library, scales


def recover(target, rho_initial, library, scales, config, ic_name, *, diagram=False):
    """Thin wrapper returning only the estimate dict."""
    kwargs = {"target_diagram": target} if diagram else {}
    field = None if diagram else target
    estimate, _ = recover_continuous(
        field,
        {ic_name: rho_initial},
        {ic_name: library[ic_name]},
        scales,
        config,
        DIMENSIONS,
        **kwargs,
    )
    return estimate


def signed_errors(estimate: dict, truth: dict) -> dict[str, float]:
    true_rhat = float(truth["rhat"])
    true_dhat = float(truth["dhat"])
    true_alpha = true_dhat / true_rhat
    return {
        "rhat_signed": (float(estimate["rhat"]) - true_rhat) / true_rhat,
        "dhat_signed": (float(estimate["dhat"]) - true_dhat) / true_dhat,
        "alpha_signed": (float(estimate["alpha"]) - true_alpha) / true_alpha,
    }


def summarize(errors: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Median absolute, p90 absolute, median signed and the overestimate share."""
    out: dict[str, dict[str, float]] = {}
    for key in ("rhat_signed", "dhat_signed", "alpha_signed"):
        values = np.array([row[key] for row in errors], dtype=float)
        out[key.replace("_signed", "")] = {
            "median_absolute": float(np.median(np.abs(values))),
            "p90_absolute": float(np.quantile(np.abs(values), 0.90)),
            "max_absolute": float(np.max(np.abs(values))),
            "median_signed": float(np.median(values)),
            "fraction_overestimated": float(np.mean(values > 0)),
        }
    return out


def write_json(payload: dict, path: Path) -> None:
    RESULTS.mkdir(exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    print(f"wrote {path}")


def truths(config: RecoveryConfig, count: int) -> list[dict]:
    return make_offgrid_holdouts(config, count)


def noisy_field(field: np.ndarray, eta: float, seed: int) -> np.ndarray:
    """Additive Gaussian at ``eta`` of the field's p01-p99 span, clipped at zero."""
    if eta <= 0:
        return field
    low, high = np.quantile(field, (0.01, 0.99))
    rng = np.random.default_rng(seed)
    noise = eta * (high - low) * rng.standard_normal(field.shape)
    return np.clip(field + noise, 0.0, None)


def noisy_diagram(field, config, eta: float, seed: int, *, after_smoothing=False):
    """Descriptor of ``field``, with the noise added to the density field, or to
    the smoothed image when ``after_smoothing``."""
    if not after_smoothing:
        field = noisy_field(field, eta, seed)
    observed = observe_field(
        field,
        output_grid_size=config.observation_grid_size,
        sigma_hat=config.observation_sigma_hat,
    )
    if after_smoothing:
        observed = noisy_field(observed, eta, seed)
    normalized = reference_normalize(observed, config.rho_ref)
    return finite_diagrams(
        config.filtration_reference - normalized,
        dimensions=config.homology_dimensions,
        cutoff=config.persistence_cutoff,
    )
