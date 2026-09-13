import argparse, os, sys
import nbformat
from nbclient import NotebookClient

def run(notebook, results_dir=None, timeout=7200):
    synthetic_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(synthetic_dir)
    if results_dir is not None:
        os.environ["PSD_RESULTS_DIR"]=os.path.abspath(results_dir)
    nb=nbformat.read(notebook, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata":{"path":synthetic_dir}}).execute()

def main(notebook):
    p=argparse.ArgumentParser()
    p.add_argument("--results", default=None)
    p.add_argument("--timeout", type=int, default=7200)
    a=p.parse_args()
    run(notebook, a.results, a.timeout)

if __name__=="__main__":
    sys.exit(main(sys.argv[1]))
