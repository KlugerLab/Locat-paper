"""
Run all 5 methods once on the full original Kang dataset (stim or ctrl)
using n_cells>=100 gene filter — matching the gene set used in the
bootstrap replacement runs.

Usage:
    python run_kang_real_dataset.py --condition stim [--gpu 0]
    python run_kang_real_dataset.py --condition ctrl [--gpu 0]
"""
import argparse, os, sys, subprocess, time, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

parser = argparse.ArgumentParser()
parser.add_argument("--condition", required=True, choices=["stim", "ctrl"])
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

HERE      = Path(__file__).parent
LOCAT_SRC = Path("/banach2/wes/locat-0.1")
DATA_PATH = Path("/banach2/wes/Locat-paper-repro-private/data/kang_counts_25k.h5ad")
GSPA_PYTHON = "/banach2/wes/.conda/envs/gspa-env/bin/python"
LMD_PYTHON  = "/banach2/wes/envs/lmd_rpy2/bin/python"
LMD_SCRIPT  = Path("/banach2/wes/Locat-paper-repro-private/notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison/run_lmd_seeded.py")
GSPA_SCRIPT = Path("/banach2/wes/Locat-paper-repro-private/notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison/run_gspa_seeded.py")

OUT_DIR = HERE / "Perturb_PBMC/celltype_specificity_comparison" / f"real_dataset_n100_{args.condition}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(LOCAT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAT_SRC))

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Load and preprocess ───────────────────────────────────────────────────────
log(f"Loading Kang {args.condition}...")
raw = sc.read_h5ad(DATA_PATH)
adata = raw[raw.obs["label"] == args.condition].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

expr = (adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)) > 0
adata = adata[:, expr.sum(axis=0) >= 100].copy()
log(f"  {adata.n_obs} cells × {adata.n_vars} genes after norm + n_cells≥100 filter")

adata.obs["cell_type"] = pd.Categorical(adata.obs["cell_type"])
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)

tmp = Path(tempfile.mktemp(suffix=".h5ad"))
adata.write_h5ad(tmp)

# ── 1. LOCAT ──────────────────────────────────────────────────────────────────
log("Running LOCAT...")
t0 = time.time()
from locat.locat import LOCAT
embedding = adata.obsm["X_pca"].astype(np.float64)[:, :8]
model = LOCAT(
    adata=adata, cell_embedding=embedding, k=20, n_bootstrap_inits=50,
    show_progress=True, knn=adata.obsp["connectivities"], knn_mode="connectivity",
)
model._reg_covar = 1e-6
results = model.gmm_scan(
    weights_transform=lambda x: np.clip(np.asarray(x), 0.0, np.inf),
    max_freq=0.9, include_depletion_scan=True,
    rc_lambda_values=np.linspace(1.0, 2.0, 8),
)
rows = [{"gene": g, **{k: v for k, v in r._asdict().items()}} for g, r in results.items()]
df = pd.DataFrame(rows).set_index("gene").sort_values("pval")
np.savez(OUT_DIR / "locat_scores.npz", gene_names=df.index.values, pval=df["pval"].values)
log(f"  LOCAT done in {(time.time()-t0)/60:.1f}m")

# ── 2. GSPA ───────────────────────────────────────────────────────────────────
log("Running GSPA...")
t0 = time.time()
subprocess.run(
    [GSPA_PYTHON, str(GSPA_SCRIPT), "--data_path", str(tmp),
     "--out_path", str(OUT_DIR / "gspa_scores.npz"), "--seed", "0", "--gpu", args.gpu],
    check=True,
)
log(f"  GSPA done in {(time.time()-t0)/60:.1f}m")

# ── 3. LMD ────────────────────────────────────────────────────────────────────
log("Running LMD...")
t0 = time.time()
subprocess.run(
    [LMD_PYTHON, str(LMD_SCRIPT), "--data_path", str(tmp),
     "--out_path", str(OUT_DIR / "lmd_scores.npz")],
    env={**os.environ, "R_HOME": "/banach2/wes/envs/lmd_rpy2/lib/R",
         "R_DEFAULT_PACKAGES": "base,utils,stats,graphics,grDevices,methods",
         "CUDA_VISIBLE_DEVICES": ""},
    check=True,
)
log(f"  LMD done in {(time.time()-t0)/60:.1f}m")

# ── 4. Hotspot ────────────────────────────────────────────────────────────────
log("Running Hotspot...")
t0 = time.time()
import hotspot as hs_pkg
hs = hs_pkg.Hotspot(adata, layer_key=None, model="normal",
                    latent_obsm_key="X_pca", umi_counts_obs_key=None)
hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
hs.compute_autocorrelations()
hotspot_scores = hs.results["FDR"]
np.savez(OUT_DIR / "hotspot_scores.npz",
         var_names=hotspot_scores.index.values, fdr=hotspot_scores.values)
log(f"  Hotspot done in {(time.time()-t0)/60:.1f}m")

# ── 5. Haystack ───────────────────────────────────────────────────────────────
log("Running Haystack...")
t0 = time.time()
import singleCellHaystack as sch
res = sch.haystack(adata, coord="pca")
haystack_scores = res.result.set_index("gene")["logpval"]
np.savez(OUT_DIR / "haystack_scores.npz",
         var_names=haystack_scores.index.values, logpval=haystack_scores.values)
log(f"  Haystack done in {(time.time()-t0)/60:.1f}m")

tmp.unlink(missing_ok=True)
log(f"All scores saved to {OUT_DIR}")
