"""Run LMD with a given data path. Called by run_multiseed.py.
LMD is deterministic; variation across seeds comes from the recomputed PCA stored in the adata."""
import argparse, os, sys
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True)
parser.add_argument("--out_path", required=True)
args = parser.parse_args()

os.environ["R_HOME"] = "LMD_R_HOME"
os.environ["R_DEFAULT_PACKAGES"] = "base,utils,stats,graphics,grDevices,methods"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util, scanpy as sc

_spec = importlib.util.spec_from_file_location(
    "run_lmd", str(Path(__file__).resolve().parents[4] / "tools/run_lmd.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
lmd_scores_from_adata = _mod.lmd_scores_from_adata

from rpy2.robjects import conversion, default_converter, numpy2ri
conversion.set_conversion(default_converter + numpy2ri.converter)

adata = sc.read_h5ad(args.data_path)
print(f"Loaded: {adata}", flush=True)

scores = lmd_scores_from_adata(
    adata,
    feature_space_key="X_pca",
    min_cells=5,
    max_time=2**10,
    knn=5,
    assume_counts=False,
    center_scale_pcs=False,
    normalize_names=False,
    debug=True,
)
print("LMD done", flush=True)

np.savez(args.out_path, var_names=np.array(scores.index), lmd_score=np.array(scores.values))
print(f"Saved to {args.out_path}", flush=True)
