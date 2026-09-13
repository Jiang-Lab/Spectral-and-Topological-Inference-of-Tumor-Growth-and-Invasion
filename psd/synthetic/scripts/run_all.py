import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _run_notebook import run
if __name__=="__main__":
    run("PSD_recovery_results_1D.ipynb")
    run("PSD_recovery_results_3D.ipynb")
