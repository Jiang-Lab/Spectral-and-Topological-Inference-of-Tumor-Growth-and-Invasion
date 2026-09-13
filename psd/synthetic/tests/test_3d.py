import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _compare import run_and_compare

def test_3d_matches_baseline():
    run_and_compare("run_3d.py", "3d", "MASTER_3D_")
