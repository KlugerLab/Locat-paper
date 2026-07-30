"""
Run all 5 methods for one PCA seed and save results to scores/seed_{N}/.

Usage:
    /banach2/wes/.conda/envs/mulde_jax/bin/python run_multiseed.py --seed 0 [--gpu 0]

Emits machine-parseable progress lines for monitor_seeds.py:
    STEP_START|ts=...|seed=N|step=K|name=METHOD
    STEP_END|ts=...|seed=N|step=K|name=METHOD|duration=S
    DONE|ts=...|seed=N|total=S
    FAILED|ts=...|seed=N|step=K|name=METHOD|error=...
"""
import argparse, os, sys, random, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--gpu",  type=str, default="0")
args = parser.parse_args()

SEED = args.seed
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

HERE      = Path(__file__).parent
LOCAT_SRC = Path("/banach2/wes/locat-0.1")
DATA_PATH = Path("/banach2/wes/Locat-paper-repro-private/data/pbmc3k_9543_lognorm.h5ad")
OUT_DIR   = HERE / "scores" / f"seed_{SEED}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GSPA_PYTHON = "/banach2/wes/.conda/envs/gspa-env/bin/python"
LMD_PYTHON  = "/banach2/wes/envs/lmd_rpy2/bin/python"

if str(LOCAT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAT_SRC))

TOTAL_STEPS = 6
_run_start = int(time.time())

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def step_start(step_num, name):
    ts = int(time.time())
    print(f"STEP_START|ts={ts}|seed={SEED}|step={step_num}|name={name}", flush=True)
    log(f"[{step_num}/{TOTAL_STEPS}] {name} started")
    return ts

def step_end(step_num, name, t0):
    ts = int(time.time())
    dur = ts - t0
    print(f"STEP_END|ts={ts}|seed={SEED}|step={step_num}|name={name}|duration={dur}", flush=True)
    log(f"[{step_num}/{TOTAL_STEPS}] {name} done ({dur//60}m{dur%60:02d}s)")
    return dur

log(f"Seed {SEED} starting  |  GPU={args.gpu}  |  out={OUT_DIR}")

# ── 1. Recompute PCA on HVGs ─────────────────────────────────────────────────
t0 = step_start(1, "PCA")
adata = sc.read_h5ad(DATA_PATH)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
hvg_names = adata.var_names[adata.var["highly_variable"]].tolist()
log(f"HVGs computed from 9543-gene adata: {len(hvg_names)} genes")
adata_for_pca = adata[:, hvg_names].copy()
sc.pp.scale(adata_for_pca, max_value=10)
sc.tl.pca(adata_for_pca, n_comps=50, random_state=SEED)
adata.obsm["X_pca"] = adata_for_pca.obsm["X_pca"]
tmp_adata = OUT_DIR / "adata_tmp.h5ad"
adata.write_h5ad(tmp_adata)
step_end(1, "PCA", t0)

# ── 2. LOCAT (first 8 PCs) ───────────────────────────────────────────────────
t0 = step_start(2, "LOCAT")
from locat.locat import LOCAT
embedding = adata.obsm["X_pca"].astype(np.float64)[:, :8]
model = LOCAT(
    adata=adata,
    cell_embedding=embedding,
    k=20,
    show_progress=True,
    knn=adata.obsp["connectivities"],
    knn_mode="connectivity",
)
model._reg_covar = 1e-6
results = model.gmm_scan(
    weights_transform=lambda x: np.clip(np.asarray(x), 0.0, np.inf),
    max_freq=0.9,
    include_depletion_scan=True,
    rc_lambda_values=np.linspace(1.0, 2.0, 8),
)
rows = [{"gene": g, **{k: v for k, v in r._asdict().items()}} for g, r in results.items()]
locat_df = pd.DataFrame(rows).set_index("gene").sort_values("pval")
np.savez(OUT_DIR / "locat_scores.npz",
         gene_names=locat_df.index.values,
         pval=locat_df["pval"].values,
         concentration_pval=locat_df["concentration_pval"].values,
         depletion_pval=locat_df["depletion_pval"].values,
         zscore=locat_df["zscore"].values)
step_end(2, "LOCAT", t0)

# ── 3. GSPA (subprocess; own PHATE seed, does not use PCA) ───────────────────
t0 = step_start(3, "GSPA")
subprocess.run(
    [GSPA_PYTHON, str(HERE / "run_gspa_seeded.py"),
     "--data_path", str(DATA_PATH),
     "--out_path",  str(OUT_DIR / "gspa_scores.npz"),
     "--seed",      str(SEED),
     "--gpu",       args.gpu],
    check=True,
)
step_end(3, "GSPA", t0)

# ── 4. LMD (subprocess; deterministic given PCA) ─────────────────────────────
t0 = step_start(4, "LMD")
subprocess.run(
    [LMD_PYTHON, str(HERE / "run_lmd_seeded.py"),
     "--data_path", str(tmp_adata),
     "--out_path",  str(OUT_DIR / "lmd_scores.npz")],
    env={**os.environ,
         "R_HOME": "/banach2/wes/envs/lmd_rpy2/lib/R",
         "R_DEFAULT_PACKAGES": "base,utils,stats,graphics,grDevices,methods",
         "CUDA_VISIBLE_DEVICES": ""},
    check=True,
)
step_end(4, "LMD", t0)

# ── 5. Hotspot (all 50 PCs) ───────────────────────────────────────────────────
t0 = step_start(5, "Hotspot")
import hotspot as hs_pkg
np.random.seed(SEED)
hs = hs_pkg.Hotspot(adata, layer_key=None, model="normal",
                    latent_obsm_key="X_pca", umi_counts_obs_key=None)
hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
hs.compute_autocorrelations()
hotspot_scores = hs.results["FDR"]
np.savez(OUT_DIR / "hotspot_scores.npz",
         var_names=hotspot_scores.index.values, fdr=hotspot_scores.values)
step_end(5, "Hotspot", t0)

# ── 6. Haystack (all PCs via coord="pca") ────────────────────────────────────
t0 = step_start(6, "Haystack")
import singleCellHaystack as sch
np.random.seed(SEED)
res = sch.haystack(adata, coord="pca")
haystack_scores = res.result.set_index("gene")["logpval"]
np.savez(OUT_DIR / "haystack_scores.npz",
         var_names=haystack_scores.index.values, logpval=haystack_scores.values)
step_end(6, "Haystack", t0)

# ── Done ──────────────────────────────────────────────────────────────────────
tmp_adata.unlink(missing_ok=True)
total = int(time.time()) - _run_start
print(f"DONE|ts={int(time.time())}|seed={SEED}|total={total}", flush=True)
log(f"Seed {SEED} complete in {total//60}m{total%60:02d}s")
