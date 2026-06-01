"""Run LMD on 3k PBMC restricted to 1838 HVGs."""
import os, sys, importlib.util, numpy as np

os.environ["R_HOME"] = "/banach2/wes/envs/lmd_rpy2/lib/R"
os.environ["R_DEFAULT_PACKAGES"] = "base,utils,stats,graphics,grDevices,methods"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
SEED = 13
np.random.seed(SEED)

import scanpy as sc
_spec = importlib.util.spec_from_file_location("run_lmd", "/banach2/wes/Locat-paper-repro-private/tools/run_lmd.py")
_mod  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
lmd_scores_from_adata = _mod.lmd_scores_from_adata

from rpy2.robjects import conversion, default_converter, numpy2ri
conversion.set_conversion(default_converter + numpy2ri.converter)

DATA_PATH = "/banach2/wes/Locat-paper-repro-private/data/pbmc3k_hvg_lognorm.h5ad"
OUT_PATH  = "/banach2/wes/Locat-paper-repro-private/notebooks/figures/Perturb_PBMC/celltype_specificity_comparison/lmd_scores_hvg.npz"

adata = sc.read_h5ad(DATA_PATH)
print(f"Loaded: {adata}", flush=True)

scores = lmd_scores_from_adata(adata, feature_space_key="X_pca", min_cells=5,
                                max_time=2**10, knn=5, assume_counts=False,
                                center_scale_pcs=False, normalize_names=False, debug=True)
print("LMD done", flush=True)

np.savez(OUT_PATH, var_names=np.array(scores.index), lmd_score=np.array(scores.values))
print(f"Saved {OUT_PATH}", flush=True)
