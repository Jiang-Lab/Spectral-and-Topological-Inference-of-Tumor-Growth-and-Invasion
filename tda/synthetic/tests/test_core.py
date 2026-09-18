"""Invariants and numerical checks."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from tda_synth.config import REPORTING_ONLY_FIELDS, RecoveryConfig, load_config
from tda_synth.initial_conditions import IC_NAMES, load_initial_conditions
from tda_synth.observation import observe_field, reference_normalize
from tda_synth.pde import simulate_snapshot, solve_linear, solve_logistic
from tda_synth.persistence import persistence_descriptor
from tda_synth.recovery import make_offgrid_holdouts
from tda_synth.scaling import length_scale_ratio, report, rhat_dhat, tau_alpha


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def cfg() -> RecoveryConfig:
    return load_config(ROOT / "configs" / "model2.yaml")


# --------------------------------------------------------------------------
# the nondimensionalization itself
# --------------------------------------------------------------------------


def physical_units_reference(
    rho_initial: np.ndarray,
    growth_rate: float,
    diffusivity: float,
    *,
    domain_length: float,
    elapsed_time: float,
    steps: int,
) -> np.ndarray:
    """The same spectral IMEX scheme, written out in micrometres and days."""
    grid = rho_initial.shape[0]
    spacing = domain_length / grid
    wave = 2.0 * np.pi * np.fft.fftfreq(grid, d=spacing)
    kx, ky = np.meshgrid(wave, wave, indexing="ij")
    symbol = kx**2 + ky**2
    step = elapsed_time / steps
    rho = np.asarray(rho_initial, dtype=float).copy()
    denominator = 1.0 + step * diffusivity * symbol
    for _ in range(steps):
        reacted = rho + step * growth_rate * rho * (1.0 - rho)
        rho = np.real(np.fft.ifft2(np.fft.fft2(reacted) / denominator))
        np.clip(rho, 0.0, None, out=rho)
    return rho


@pytest.mark.parametrize(
    "growth_rate, diffusivity, domain_length, elapsed_time",
    [
        (0.75, 600.0, 2500.0, 1.0),   # the reference scales used for reporting
        (0.40, 900.0, 2500.0, 3.0),   # a different observation interval
        (1.10, 300.0, 1200.0, 0.5),   # a different tissue window as well
    ],
)
def test_dimensionless_solver_matches_the_same_scheme_in_physical_units(
    growth_rate, diffusivity, domain_length, elapsed_time
):
    """rho(x, t=T) in physical units equals the t_hat = 1 dimensionless map."""
    rng = np.random.default_rng(3)
    rho_initial = 0.6 * rng.random((32, 32))
    steps = 50

    # The relation under test, written out rather than imported, so the test
    # does not depend on the module it is checking.
    rhat = growth_rate * elapsed_time
    dhat = diffusivity * elapsed_time / domain_length**2

    dimensionless = solve_logistic(rho_initial, rhat, dhat, solver_step=1.0 / steps)
    physical = physical_units_reference(
        rho_initial,
        growth_rate,
        diffusivity,
        domain_length=domain_length,
        elapsed_time=elapsed_time,
        steps=steps,
    )
    np.testing.assert_allclose(dimensionless, physical, rtol=0, atol=1e-12)


def test_only_the_two_dimensionless_groups_matter():
    """Same (R_hat, D_hat) from different (R, D, L0, T) gives the same field."""
    rng = np.random.default_rng(5)
    rho_initial = 0.5 * rng.random((24, 24))
    first = (0.80 * 1.0, 640.0 * 1.0 / 2500.0**2)
    second = (0.40 * 2.0, 320.0 * 2.0 / 2500.0**2)
    assert first == pytest.approx(second)
    np.testing.assert_allclose(
        solve_logistic(rho_initial, *first, solver_step=0.02),
        solve_logistic(rho_initial, *second, solver_step=0.02),
        atol=1e-14,
    )


def test_tau_alpha_is_an_exact_reparametrization():
    rhat, dhat = 0.748, 9.5759e-05
    tau, alpha = tau_alpha(rhat, dhat)
    assert tau == pytest.approx(rhat)
    assert alpha == pytest.approx(dhat / rhat)
    np.testing.assert_allclose(rhat_dhat(tau, alpha), (rhat, dhat), rtol=1e-15)


def test_a_recovery_reports_nothing_with_a_unit(cfg):
    """Only dimensionless quantities may leave a recovery."""
    rhat, dhat = 0.748, 9.5759e-05
    found = report(rhat, dhat)
    assert found._fields == (
        "rhat",
        "dhat",
        "tau",
        "alpha",
        "length_scale_ratio",
    )
    assert found.rhat == rhat and found.dhat == dhat
    assert found.tau == pytest.approx(rhat)
    assert found.alpha == pytest.approx(dhat / rhat)
    assert found.length_scale_ratio == pytest.approx(
        length_scale_ratio(rhat, dhat)
    )
    np.testing.assert_allclose(
        rhat_dhat(found.tau, found.alpha), (rhat, dhat), rtol=1e-14
    )


def test_physical_conversion_needs_an_explicitly_measured_time():
    """R and D with units need a measured elapsed time, passed explicitly."""
    from tda_synth.scaling import to_physical

    with pytest.raises(TypeError):
        to_physical(0.7, 1.2e-4)  # no scales: must not be guessable

    phys = to_physical(
        0.7,
        1.2e-4,
        elapsed_time=5.0,
        domain_length=2500.0,
        time_unit="day",
        length_unit="um",
    )
    assert phys.growth_rate == pytest.approx(0.7 / 5.0)
    assert phys.diffusivity == pytest.approx(1.2e-4 * 2500.0**2 / 5.0)
    for bad in ({"elapsed_time": 0.0}, {"domain_length": -1.0}):
        kwargs = {
            "elapsed_time": 5.0,
            "domain_length": 2500.0,
            "time_unit": "day",
            "length_unit": "um",
        }
        kwargs.update(bad)
        with pytest.raises(ValueError):
            to_physical(0.7, 1.2e-4, **kwargs)


def test_the_dimensionless_report_is_still_unit_free():
    """report() is what a snapshot uses, and it must stay dimensionless."""
    found = report(0.7, 1.2e-4)
    assert "growth_rate" not in found._fields
    assert "diffusivity" not in found._fields


def test_no_dimensional_quantity_enters_the_calculation(cfg):
    """No configuration field carries a unit, and no solver takes a time."""
    unit_suffixes = ("_um", "_um2", "_day", "_days", "_hour", "_hours", "_cm", "_mm")
    offenders = [
        name
        for name in cfg.__dict__
        if name not in REPORTING_ONLY_FIELDS and name.endswith(unit_suffixes)
    ]
    assert offenders == []
    assert not any(
        "time" in name or "schedule" in name
        for name in inspect.signature(simulate_snapshot).parameters
    )


def test_smoothing_width_is_a_domain_fraction(cfg):
    """0.012 of the domain side: a ratio, carrying no unit."""
    assert cfg.observation_sigma_hat == pytest.approx(0.012)
    assert cfg.observation_sigma_pixels == pytest.approx(
        cfg.observation_sigma_hat * cfg.observation_grid_size
    )


def test_carrying_capacity_is_fixed_and_not_an_argument(cfg):
    assert not any(name.lower() in {"k", "k_hat", "khat"} for name in cfg.__dict__)
    assert not any(
        name.lower() in {"k", "k_hat", "khat"}
        for name in inspect.signature(solve_logistic).parameters
    )
    rho_full = np.ones((12, 12))
    np.testing.assert_allclose(
        solve_logistic(rho_full, 0.8, 0.00012, solver_step=0.02),
        rho_full,
        atol=1e-12,
    )


def test_density_scale_must_agree_with_the_fixed_carrying_capacity(cfg):
    """rho_ref != 1 would silently contradict the (1 - rho) reaction term."""
    from dataclasses import replace

    with pytest.raises(ValueError, match="K = 1"):
        replace(cfg, rho_ref=2.0).validate()
    with pytest.raises(ValueError, match="K = 1"):
        replace(cfg, filtration_reference=0.5).validate()


def test_target_and_candidate_solver_steps_agree(cfg):
    """Unequal substeps leak splitting error into the recovered parameters."""
    assert cfg.target_solver_step == cfg.candidate_solver_step


# --------------------------------------------------------------------------
# numerics and data
# --------------------------------------------------------------------------


def test_four_sealed_initial_conditions_are_unchanged():
    fields = load_initial_conditions(
        ROOT / "data" / "somiya_ic_125.npz"
    )
    assert tuple(fields) == IC_NAMES
    assert all(field.shape == (125, 125) for field in fields.values())
    assert all(float(field.min()) >= 0.0 for field in fields.values())
    assert all(float(field.max()) <= 1.0 for field in fields.values())


def test_observation_operator_matches_the_dimensional_formula(cfg):
    """sigma_hat * N must equal sigma_um / (l0_um / N) in pixels."""
    from scipy.ndimage import gaussian_filter, zoom

    rng = np.random.default_rng(17)
    field = rng.random((125, 125))
    factor = cfg.observation_grid_size / field.shape[0]
    expected = gaussian_filter(
        zoom(field, (factor, factor), order=1, mode="grid-wrap", prefilter=False, grid_mode=True),
        sigma=30.0 / (2500.0 / cfg.observation_grid_size),
        mode="wrap",
    )
    np.testing.assert_allclose(
        observe_field(
            field,
            output_grid_size=cfg.observation_grid_size,
            sigma_hat=cfg.observation_sigma_hat,
        ),
        expected,
        atol=1e-12,
    )


def test_reference_normalization_is_not_per_field_minmax():
    field = np.array([[0.0, 1.0], [2.0, 3.0]])
    normalized = reference_normalize(field, rho_ref=2.0)
    doubled = reference_normalize(2.0 * field, rho_ref=2.0)
    np.testing.assert_allclose(normalized, field / 2.0)
    np.testing.assert_allclose(doubled, 2.0 * normalized)
    assert doubled.max() == 3.0


def test_dimensionless_linear_map_matches_direct_formula():
    rng = np.random.default_rng(7)
    rho_initial = rng.random((24, 24))
    rhat = 0.65
    dhat = 0.00012
    k = 2.0 * np.pi * np.fft.fftfreq(24, d=1.0 / 24)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    expected = np.real(
        np.fft.ifft2(
            np.fft.fft2(rho_initial) * np.exp(rhat - dhat * (kx**2 + ky**2))
        )
    )
    np.testing.assert_allclose(
        solve_linear(rho_initial, rhat, dhat), expected, atol=1e-12
    )


def test_solver_step_error_is_measurable_at_the_configured_step(cfg):
    """Quartering the configured substep moves the field, and by how much."""
    fields = load_initial_conditions(
        ROOT / "data" / "somiya_ic_125.npz"
    )
    initial = fields["Many clusters"]
    coarse = solve_logistic(initial, 0.75, 0.00012, solver_step=cfg.candidate_solver_step)
    fine = solve_logistic(initial, 0.75, 0.00012, solver_step=cfg.candidate_solver_step / 4)
    difference = float(np.max(np.abs(coarse - fine)))
    assert 0.0 < difference < 1e-2, difference


def test_holdout_truths_are_strictly_off_both_grid_axes(cfg):
    for truth in make_offgrid_holdouts(cfg):
        assert not any(
            np.isclose(truth["rhat"], value, rtol=0.0, atol=1e-12)
            for value in cfg.rhat_grid
        )
        assert not any(
            np.isclose(truth["dhat"], value, rtol=0.0, atol=1e-12)
            for value in cfg.dhat_grid
        )
        assert truth["tau"] == pytest.approx(truth["rhat"])
        assert truth["alpha"] == pytest.approx(truth["dhat"] / truth["rhat"])


# --------------------------------------------------------------------------
# what the descriptor can and cannot see
# --------------------------------------------------------------------------


def _observed_diagram(field, cfg):
    from tda_synth.observation import reference_normalize
    from tda_synth.persistence import finite_diagrams

    return finite_diagrams(
        cfg.filtration_reference - reference_normalize(field, cfg.rho_ref),
        dimensions=cfg.homology_dimensions,
        cutoff=cfg.persistence_cutoff,
    )


@pytest.mark.parametrize(
    "name, transform",
    [
        ("translation", lambda f: np.roll(f, (7, 13), axis=(0, 1))),
        ("reflection", lambda f: f[::-1, :]),
        ("rotation", lambda f: np.rot90(f)),
    ],
)
def test_descriptor_is_exactly_blind_to_rigid_motions(cfg, name, transform):
    """Periodic cubical persistence sees no rigid motion, so no advection."""
    from tda_synth.persistence import wasserstein2
    from tda_synth.observation import observe_field

    fields = load_initial_conditions(
        ROOT / "data" / "somiya_ic_125.npz"
    )
    observed = observe_field(
        fields["Many clusters"],
        output_grid_size=cfg.observation_grid_size,
        sigma_hat=cfg.observation_sigma_hat,
    )
    base = _observed_diagram(observed, cfg)
    moved = _observed_diagram(transform(observed), cfg)
    for q in cfg.homology_dimensions:
        assert wasserstein2(base[q], moved[q]) == pytest.approx(0.0, abs=1e-12), name


def test_descriptor_does_see_a_global_amplitude_change(cfg):
    """The fixed reference keeps absolute density, so a global gain is visible."""
    from tda_synth.persistence import wasserstein2
    from tda_synth.observation import observe_field

    fields = load_initial_conditions(
        ROOT / "data" / "somiya_ic_125.npz"
    )
    observed = observe_field(
        fields["Many clusters"],
        output_grid_size=cfg.observation_grid_size,
        sigma_hat=cfg.observation_sigma_hat,
    )
    base = _observed_diagram(observed, cfg)
    gained = _observed_diagram(1.3 * observed, cfg)
    assert wasserstein2(base[0], gained[0]) > 1e-3


# --------------------------------------------------------------------------
# the (tau, alpha) reparametrization
# --------------------------------------------------------------------------


def test_canonical_symbols_round_trip_through_the_solver_pair():
    """(alpha, tau) and (R_hat, D_hat) are the same information."""
    from tda_synth.scaling import canonical, rhat_dhat

    rhat, dhat = 0.91, 0.000168
    pair = canonical(rhat, dhat)
    assert pair.tau == pytest.approx(rhat)
    assert pair.alpha == pytest.approx(dhat / rhat)
    np.testing.assert_allclose(
        rhat_dhat(pair.tau, pair.alpha), (rhat, dhat), rtol=1e-14
    )


# --------------------------------------------------------------------------
# the multi-frame layer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frames",
    [(1.0,), (0.5, 1.0), (1 / 3, 2 / 3, 1.0), tuple((i + 1) / 6 for i in range(6))],
)
@pytest.mark.parametrize("model", ["linear", "logistic"])
def test_movie_final_frame_equals_the_single_frame_solver(cfg, frames, model):
    """The movie and snapshot propagators must agree exactly, for both models.

    Otherwise a multi-frame result is not comparable with the main benchmark.
    """
    from tda_synth.movie import simulate_movie

    fields = load_initial_conditions(
        ROOT / "data" / "somiya_ic_125.npz"
    )
    initial = fields["Many clusters"]
    movie = simulate_movie(
        initial,
        0.75,
        0.00012,
        frames,
        model=model,
        solver_step=cfg.candidate_solver_step,
    )
    reference = simulate_snapshot(
        initial, 0.75, 0.00012, model=model, solver_step=cfg.candidate_solver_step
    )
    assert len(movie) == len(frames)
    np.testing.assert_allclose(movie[-1], reference, rtol=0, atol=1e-12)
    masses = [float(frame.mean()) for frame in movie]
    assert masses == sorted(masses)


def test_movie_reaction_follows_the_model_not_the_default():
    """A movie must use its own model's reaction, pinned in both directions."""
    from tda_synth.movie import simulate_movie

    fields = load_initial_conditions(
        ROOT / "data" / "somiya_ic_125.npz"
    )
    initial = fields["Many clusters"]
    rhat, dhat, step = 0.75, 0.00012, 0.02

    linear = simulate_movie(
        initial, rhat, dhat, (1.0,), model="linear", solver_step=step
    )[-1]
    logistic = simulate_movie(
        initial, rhat, dhat, (1.0,), model="logistic", solver_step=step
    )[-1]
    # The two models are genuinely different maps of the same input.
    assert not np.allclose(linear, logistic)

    # Model 1 is propagated exactly, so its movie is the exact map itself, and
    # it exceeds one where that map does.
    exact = simulate_snapshot(initial, rhat, dhat, model="linear")
    assert exact.max() > 1.0
    np.testing.assert_allclose(linear, exact, rtol=0, atol=1e-12)
    assert np.abs(logistic - exact).max() / exact.max() > 0.05


def test_simulate_movie_requires_an_explicit_model():
    """No default: a stepper that picks a model implicitly can pick the wrong one."""
    from tda_synth.movie import simulate_movie

    with pytest.raises(TypeError):
        simulate_movie(np.zeros((8, 8)), 0.5, 0.001, (1.0,), solver_step=0.1)


@pytest.mark.parametrize("config_name", ["model1.yaml", "model2.yaml"])
def test_declared_frame_times_are_the_times_actually_sampled(config_name):
    """A label of 0.3 must not mean a field taken at 0.28."""
    from tda_synth.config import load_config
    from tda_synth.movie import MULTI_FRAME_SCHEDULE, realized_frames

    cfg_ = load_config(ROOT / "configs" / config_name)
    for step in (cfg_.candidate_solver_step, cfg_.target_solver_step):
        assert realized_frames(MULTI_FRAME_SCHEDULE, step) == pytest.approx(
            MULTI_FRAME_SCHEDULE
        )


def test_movie_frames_are_validated():
    from tda_synth.movie import validate_frames

    assert validate_frames([0.5, 1.0]) == (0.5, 1.0)
    for bad in ([], [0.5], [1.0, 0.5], [0.5, 0.5, 1.0], [0.0, 1.0], [-0.5, 1.0]):
        with pytest.raises(ValueError):
            validate_frames(bad)


def test_single_frame_movie_recovery_matches_the_single_frame_pipeline(cfg):
    """One frame through the movie layer must reproduce the shipped result."""
    from tda_synth.movie import (
        build_movie_library,
        describe_movie,
        estimate_movie_scales,
        recover_movie,
    )
    from tda_synth.recovery import (
        build_library,
        estimate_scales,
        make_offgrid_holdouts,
        recover_continuous,
    )

    fields = load_initial_conditions(
        ROOT / "data" / "somiya_ic_125.npz"
    )
    name = "Many clusters"
    initial = fields[name]
    truth = make_offgrid_holdouts(cfg, 1)[0]
    target_field = simulate_snapshot(
        initial,
        truth["rhat"],
        truth["dhat"],
        model=cfg.model,
        solver_step=cfg.target_solver_step,
    )

    library = build_library({name: initial}, cfg)
    scales = estimate_scales(library, cfg.homology_dimensions, config=cfg)
    single, _ = recover_continuous(
        target_field, {name: initial}, library, scales, cfg
    )

    frames = (1.0,)
    movie_library = build_movie_library({name: initial}, cfg, frames)
    movie_scales = estimate_movie_scales(movie_library, cfg, frames)
    movie_target = describe_movie(
        initial,
        truth["rhat"],
        truth["dhat"],
        frames,
        cfg,
        solver_step=cfg.target_solver_step,
    )
    multi, _ = recover_movie(
        movie_target, {name: initial}, movie_library, movie_scales, cfg, frames
    )

    assert multi["rhat"] == pytest.approx(single["rhat"], rel=1e-9)
    assert multi["dhat"] == pytest.approx(single["dhat"], rel=1e-9)


def test_the_two_recovery_modes(cfg):
    """Exactly two modes, with the declared multi_frame schedule."""
    from tda_synth import movie

    assert movie.DEFAULT_ORDER == ("single_snapshot", "multi_frame")
    assert movie.frames_for("single_snapshot") == (1.0,)
    # Pinned, so a change to the schedule cannot pass silently.
    frames = movie.frames_for("multi_frame")
    assert frames == pytest.approx((0.3, 0.6, 1.0))
    for row in movie.table():
        assert row["frames"][-1] == 1.0
        assert row["frames"] == sorted(row["frames"])


# --------------------------------------------------------------------------
# the observation operator does not resample
# --------------------------------------------------------------------------


def test_observation_grid_equals_the_pde_grid(cfg):
    """Equal grids, so the operator smooths and never resamples."""
    assert cfg.observation_grid_size == cfg.pde_grid_size


def test_observation_operator_is_identity_before_smoothing(cfg):
    """With equal grids the resampling step must be exactly the identity."""
    from scipy.ndimage import zoom

    rng = np.random.default_rng(0)
    field = rng.random((cfg.pde_grid_size, cfg.pde_grid_size))
    factor = cfg.observation_grid_size / field.shape[0]
    resampled = zoom(
        field,
        (factor, factor),
        order=1,
        mode="grid-wrap",
        prefilter=False,
        grid_mode=True,
    )
    np.testing.assert_array_equal(resampled, field)


@pytest.mark.parametrize("shift", [1, 2, 3, 7])
def test_descriptor_is_invariant_to_whole_pixel_input_shifts(cfg, shift):
    """A whole-pixel shift of the input leaves every diagram unchanged."""
    from tda_synth.persistence import wasserstein2

    fields = load_initial_conditions(
        ROOT / "data" / "somiya_ic_125.npz"
    )
    initial = fields["Many clusters"]
    field = simulate_snapshot(
        initial, 0.75, 0.00012, model=cfg.model, solver_step=cfg.candidate_solver_step
    )
    base = persistence_descriptor(field, cfg)
    moved = persistence_descriptor(np.roll(field, (shift, shift), axis=(0, 1)), cfg)
    for q in cfg.homology_dimensions:
        assert wasserstein2(base[q], moved[q]) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Model 1: R is an amplitude, not a topological quantity
# --------------------------------------------------------------------------


def test_model1_growth_rate_only_rescales_the_field():
    """Model 1: exp(R_hat) is a global scalar, so the ratio is constant."""
    fields = load_initial_conditions(
        ROOT / "data" / "somiya_ic_125.npz"
    )
    initial = fields["Many clusters"]
    low = simulate_snapshot(initial, 0.35, 0.00012, model="linear")
    high = simulate_snapshot(initial, 1.05, 0.00012, model="linear")
    mask = low > 1e-9
    ratio = high[mask] / low[mask]
    assert float(np.std(ratio) / np.mean(ratio)) < 1e-4
    assert float(np.mean(ratio)) == pytest.approx(np.exp(1.05 - 0.35), rel=1e-3)


def test_both_models_are_configured_and_runnable():
    """Model 1 and Model 2 each have a config and each solves."""
    from tda_synth.config import load_config

    model1 = load_config(ROOT / "configs" / "model1.yaml")
    model2 = load_config(ROOT / "configs" / "model2.yaml")
    assert model1.model == "linear"
    assert model2.model == "logistic"
    # the observation operator and the descriptor settings are shared, so the
    # two models differ only in the reaction term
    for field in (
        "pde_grid_size",
        "observation_grid_size",
        "observation_sigma_hat",
        "persistence_cutoff",
        "homology_dimensions",
    ):
        assert getattr(model1, field) == getattr(model2, field), field

    fields = load_initial_conditions(
        ROOT / "data" / "somiya_ic_125.npz"
    )
    initial = fields["Many clusters"]
    for cfg_ in (model1, model2):
        out = simulate_snapshot(
            initial, 0.75, 0.00012, model=cfg_.model, solver_step=0.02
        )
        assert np.all(np.isfinite(out)) and out.min() >= 0.0


# --------------------------------------------------------------------------
# the different-solver check must vary the solver, not the equation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [("linear", 0.8 * 0.5), ("logistic", 0.8 * 0.5 * (1 - 0.5))],
)
def test_mismatch_reaction_follows_the_model(model, expected):
    """The mismatched targets solve the same equation as the candidates."""
    from run_operator_mismatch import reaction_term

    value = reaction_term(np.array([0.5]), 0.8, model)
    assert float(value[0]) == pytest.approx(expected)
    with pytest.raises(ValueError):
        reaction_term(np.array([0.5]), 0.8, "quadratic")


def test_mismatch_target_differs_from_the_candidate_solver(cfg):
    """The mismatched solver must actually produce a different field."""
    from run_operator_mismatch import five_point_symbol, lift, mismatched_snapshot

    fields = load_initial_conditions(
        ROOT / "data" / "somiya_ic_125.npz"
    )
    initial = fields["Many clusters"]
    reference = simulate_snapshot(
        initial, 0.75, 0.00012, model=cfg.model, solver_step=cfg.candidate_solver_step
    )
    mismatched = mismatched_snapshot(
        lift(initial, 2),
        0.75,
        0.00012,
        model=cfg.model,
        symbol=five_point_symbol(250),
        step=0.002,
        output_grid=cfg.pde_grid_size,
    )
    assert mismatched.shape == reference.shape
    difference = float(np.max(np.abs(mismatched - reference)))
    assert 1e-6 < difference < 0.5, difference


def test_coarsen_inverts_lift_exactly():
    """The fine run starts from the identical field, so no structure is added."""
    from run_operator_mismatch import coarsen, lift

    rng = np.random.default_rng(3)
    field = rng.random((25, 25))
    np.testing.assert_allclose(coarsen(lift(field, 2), 25), field, atol=1e-15)
