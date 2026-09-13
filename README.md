# Spectral and Topological Inference of Tumor Growth and Invasion

Recover growth (R) and invasion (D) parameters of tumor PDE models from spatial data
with two independent methods, validated on synthetic fields and applied to in vitro
movies and biopsy slides.

## Layout

```text
.
├── psd/        — spectral inference (power spectral density)
│   ├── synthetic/  — 53-model linear PDE library, 1D and 3D recovery
│   ├── movie/
│   └── biopsy/
├── tda/        — topological inference (persistence diagrams)
│   ├── synthetic/  — Model 1 (linear) and Model 2 (logistic) recovery
│   ├── movie/
│   └── biopsy/
└── data/       — (to add)
```

## Setup

```bash
conda create -n tumor python=3.13 -y
conda activate tumor
pip install -e psd/
pip install -e "tda/[test,notebook]"
```

## Run

```bash
cd psd/synthetic
python scripts/make_baseline.py
python scripts/run_all.py
pytest tests
```

```bash
cd tda
python -m pytest -q
cd synthetic
python scripts/run_all.py --config configs/model1.yaml
python scripts/run_all.py --config configs/model2.yaml
```

See `psd/README.md` and `tda/README.md`.
