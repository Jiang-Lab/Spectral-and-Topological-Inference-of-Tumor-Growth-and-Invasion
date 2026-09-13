# Topological Inference of Tumour Growth and Invasion

| folder | contents |
|---|---|
| `synthetic/` | parameter recovery on simulated density fields |
| `movie/` | time-lapse data |
| `biopsy/` | biopsy point clouds |

---

## synthetic

Simulate density fields from known parameters, then re-estimate those
parameters from the persistence diagrams of the simulated images.

## Models and modes

| model | equation | config |
|---|---|---|
| Model 1 | `d(rho)/dt = D lap(rho) + R rho` | `configs/model1.yaml` |
| Model 2 | `d(rho)/dt = D lap(rho) + R rho (1 - rho/K)` | `configs/model2.yaml` |

| mode | fields given to the estimator | stands in for | what it fixes |
|---|---|---|---|
| `single_snapshot` | one field, at an unmeasured elapsed time | a biopsy slice | the ratio `D_hat/R_hat` |
| `multi_frame` | three fields of one trajectory, at known intervals | a time-lapse movie | `R` and `D`, given a measured elapsed time |

Each mode is validated on its own: they are two applications, not two candidates
for the same job. A single image carries no time information, so it fixes the
ratio and not a rate; `t_hat` runs over `[0, 1]` inside the solver as a
coordinate, which measures nothing.

## Parameters

```text
rho_hat = rho/K      x_hat = x/L0      t_hat = t/T,  t_hat in [0, 1]
R_hat = R*T          D_hat = D*T/L0**2
tau = R_hat          alpha = D_hat/R_hat = D/(R*L0**2)
sqrt(alpha)          characteristic length / L0
```

Model 2 fixes `K = 1` by the scaling; Model 1 has no carrying capacity and
carries no `K` field. `single_snapshot` reports `D_hat/R_hat`.
`run_scenarios.py --elapsed-time --domain-length` converts a `multi_frame`
result to `R` and `D` with units, through `scaling.to_physical`.

`synthetic/tda_synth/scaling.py` is the authoritative statement.

## Pipeline

| script | writes |
|---|---|
| `run_recovery.py` | `recovery_*` — the main sweep, loss surface, target diagram, field figure |
| `run_phase_control.py` | `phase_scramble_control_*` |
| `run_operator_mismatch.py` | `operator_mismatch_*` — five one-factor arms |
| `run_robustness.py` | `noise_study_*`, `sigma_sensitivity_*` |
| `run_scenarios.py` | `scenarios_*`, `scenario_frames_*` — accuracy of each observation mode |

The descriptor, the loss and the search are described in
`synthetic/TDA_recovery_results_model1.ipynb` and
`synthetic/TDA_recovery_results_model2.ipynb`, section 2. Every numerical setting lives in the two config files.

## Run

```bash
python -m pip install -e ".[test,notebook]"     # once, from the repo root
python -m pytest -q

cd synthetic
python scripts/run_all.py --config configs/model2.yaml
python scripts/run_all.py --config configs/model1.yaml
```

`run_all.py` forwards `--config` to every step, and every output file is tagged
with that config's model, so a Model 1 sweep cannot overwrite a Model 2 sweep.
Each script takes `--help`.

`TDA_recovery_results_model1.ipynb` and `TDA_recovery_results_model2.ipynb` read
`results/` and show nothing until the scripts have run. They are the same notebook
with a different `MODEL` in the first code cell.

## Layout

```text
synthetic/
  tda_synth/                implementation (start at scaling.py)
  scripts/                  reproducible calculations
  configs/                  model1.yaml, model2.yaml
  data/                     sealed four-IC input, SHA-256 verified on load
  results/                  generated tables and figures
  tests/                    invariants and numerical checks
  TDA_recovery_results_model1.ipynb
  TDA_recovery_results_model2.ipynb
```

The four initial conditions are separate benchmark families and are reported
separately.
