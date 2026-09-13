import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _compare import run_and_compare

def test_1d_matches_baseline():
    run_and_compare("run_1d.py", "1d", "MASTER2_1D_")
