```
psd/
├── README.md
├── pyproject.toml
├── .gitignore
├── synthetic/
│   ├── original/
│   │   ├── PSD_All_PDEs_Master2_1D_COMPLETE_With_All_Plots.ipynb
│   │   ├── PSD_All_PDEs_Master_3D_COMPLETE_With_Evolution_Plots_FIXED.ipynb
│   │   └── figures_1D/                 67 png
│   ├── PSD_recovery_results_1D.ipynb
│   ├── PSD_recovery_results_3D.ipynb
│   ├── psd_synth/     __init__  psd_1d  psd_3d
│   ├── scripts/       run_1d  run_3d  run_all  make_baseline  _run_notebook
│   ├── configs/       default_1d.yaml  default_3d.yaml
│   ├── results/
│   │   ├── 1d/        14 csv + figures/ (67 png)
│   │   └── 3d/        11 csv + figures/ (148 png)
│   └── tests/         test_1d  test_3d  _compare  baseline/ (25 csv)
├── movie/
└── biopsy/
```

```
conda create -n psd python=3.13 -y
conda activate psd
pip install -e .
cd synthetic
python scripts/make_baseline.py
python scripts/run_1d.py
python scripts/run_3d.py
pytest tests
```
