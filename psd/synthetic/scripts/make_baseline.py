import os, sys, glob, shutil
import nbformat
from nbclient import NotebookClient

SYNTHETIC=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL=os.path.join(SYNTHETIC, "original")
BASELINE=os.path.join(SYNTHETIC, "tests", "baseline")
NOTEBOOKS=[
    "PSD_All_PDEs_Master2_1D_COMPLETE_With_All_Plots.ipynb",
    "PSD_All_PDEs_Master_3D_COMPLETE_With_Evolution_Plots_FIXED.ipynb",
]

if __name__=="__main__":
    os.makedirs(BASELINE, exist_ok=True)
    for f in glob.glob(os.path.join(BASELINE, "*.csv")):
        os.remove(f)
    for name in NOTEBOOKS:
        nb=nbformat.read(os.path.join(ORIGINAL, name), as_version=4)
        os.chdir(BASELINE)
        NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata":{"path":BASELINE}}).execute()
    for d in glob.glob(os.path.join(BASELINE, "master*_figures")):
        shutil.rmtree(d)
    print(sorted(os.listdir(BASELINE)))
