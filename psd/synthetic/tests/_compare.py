import glob, os, subprocess, sys, tempfile
import pandas as pd
from pandas.testing import assert_frame_equal

SYNTHETIC=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE=os.path.join(SYNTHETIC, "tests", "baseline")

def run_and_compare(script, dim, prefix):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, os.path.join(SYNTHETIC, "scripts", script), "--results", tmp], check=True)
        out=os.path.join(tmp, dim)
        base=sorted(os.path.basename(f) for f in glob.glob(os.path.join(BASELINE, prefix+"*.csv")))
        new=sorted(os.path.basename(f) for f in glob.glob(os.path.join(out, prefix+"*.csv")))
        assert base==new, (base, new)
        for f in base:
            assert_frame_equal(pd.read_csv(os.path.join(BASELINE, f)), pd.read_csv(os.path.join(out, f)),
                               check_exact=False, rtol=1e-12, atol=1e-14)
