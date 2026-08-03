"""
Run all 6 gene-specificity methods on Tabula Muris Senis bone marrow (10x) data
and compare using the tau cell-type-specificity metric.

Methods: Locat, GSPA, LMD, Hotspot, Haystack, Scanpy (Leiden + Wilcoxon)

Usage:
    python run_tabulamuris_comparison.py [--gpu 0]
"""
import argparse, os, sys, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

HERE      = Path(__file__).parent
LOCAT_SRC = Path("LOCAT01_PATH")
DATA_H5AD = HERE / "data" / "tabula_muris_marrow_proc.h5ad"
SCORES    = HERE / "scores"
SCORES.mkdir(exist_ok=True)

GSPA_PYTHON = "GSPA_PYTHON"
LMD_PYTHON  = "LMD_PYTHON"
LMD_SCRIPT  = Path(__file__).resolve().parents[3] / "notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison/run_lmd_seeded.py"
GSPA_SCRIPT = Path(__file__).resolve().parents[3] / "notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison/run_gspa_seeded.py"

if str(LOCAT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAT_SRC))

CELLTYPE_COL = "cell_type"
PCT_THRESH   = 0.05
LEIDEN_RES   = 0.5

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def compute_tau(adata, celltype_col=CELLTYPE_COL, pct_thresh=PCT_THRESH):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    cts = adata.obs[celltype_col].cat.categories.tolist()
    mean = np.array([X[adata.obs[celltype_col] == ct].mean(axis=0) for ct in cts])
    pct  = (X > 0).mean(axis=0)
    rs   = mean.sum(axis=0)
    mask = (rs > 0) & (pct >= pct_thresh)
    tau  = np.where(mask, mean.max(axis=0) / np.where(rs > 0, rs, 1.0), np.nan)
    return pd.Series(tau, index=adata.var_names)

# ── Load data ─────────────────────────────────────────────────────────────────
log("Loading Tabula Muris Marrow adata...")
adata = sc.read_h5ad(DATA_H5AD)
adata.obs[CELLTYPE_COL] = pd.Categorical(adata.obs[CELLTYPE_COL].astype(str))
log(f"  {adata.n_obs} cells x {adata.n_vars} genes")
log(f"  Cell types: {adata.obs[CELLTYPE_COL].value_counts().to_dict()}")

tau = compute_tau(adata)
log(f"  Genes with valid tau: {tau.notna().sum()}, median={np.nanmedian(tau.values):.4f}")

# ── 1. LOCAT ──────────────────────────────────────────────────────────────────
locat_path = SCORES / "locat_scores.npz"
log("Running LOCAT...")
t0 = time.time()
from locat.locat import LOCAT
embedding = adata.obsm["X_pca"].astype(np.float64)[:, :8]
model = LOCAT(
    adata=adata,
    cell_embedding=embedding,
    k=20,
    n_bootstrap_inits=50,
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
np.savez(locat_path, gene_names=locat_df.index.values, pval=locat_df["pval"].values)
log(f"  LOCAT done in {(time.time()-t0)/60:.1f}m")

# ── 2. GSPA ───────────────────────────────────────────────────────────────────
gspa_path = SCORES / "gspa_scores.npz"
log("Running GSPA...")
t0 = time.time()
try:
    subprocess.run(
        [GSPA_PYTHON, str(GSPA_SCRIPT),
         "--data_path", str(DATA_H5AD),
         "--out_path",  str(gspa_path),
         "--seed", "0", "--gpu", args.gpu],
        check=True, timeout=3600,
    )
    log(f"  GSPA done in {(time.time()-t0)/60:.1f}m")
except Exception as e:
    log(f"  GSPA failed: {e}")

# ── 3. LMD ────────────────────────────────────────────────────────────────────
lmd_path = SCORES / "lmd_scores.npz"
log("Running LMD...")
t0 = time.time()
try:
    subprocess.run(
        [LMD_PYTHON, str(LMD_SCRIPT),
         "--data_path", str(DATA_H5AD),
         "--out_path",  str(lmd_path)],
        env={**os.environ,
             "R_HOME": "LMD_R_HOME",
             "R_DEFAULT_PACKAGES": "base,utils,stats,graphics,grDevices,methods",
             "CUDA_VISIBLE_DEVICES": ""},
        check=True, timeout=3600,
    )
    log(f"  LMD done in {(time.time()-t0)/60:.1f}m")
except Exception as e:
    log(f"  LMD failed: {e}")

# ── 4. Hotspot ────────────────────────────────────────────────────────────────
hotspot_path = SCORES / "hotspot_scores.npz"
log("Running Hotspot...")
t0 = time.time()
try:
    import hotspot as hs_pkg
    hs = hs_pkg.Hotspot(adata, layer_key=None, model="normal",
                        latent_obsm_key="X_pca", umi_counts_obs_key=None)
    hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
    hs.compute_autocorrelations()
    hotspot_scores = hs.results["FDR"]
    np.savez(hotspot_path, var_names=hotspot_scores.index.values, fdr=hotspot_scores.values)
    log(f"  Hotspot done in {(time.time()-t0)/60:.1f}m")
except Exception as e:
    log(f"  Hotspot failed: {e}")

# ── 5. Haystack ───────────────────────────────────────────────────────────────
haystack_path = SCORES / "haystack_scores.npz"
log("Running Haystack...")
t0 = time.time()
try:
    import singleCellHaystack as sch
    res = sch.haystack(adata, coord="pca")
    haystack_scores = res.result.set_index("gene")["logpval"]
    np.savez(haystack_path, var_names=haystack_scores.index.values, logpval=haystack_scores.values)
    log(f"  Haystack done in {(time.time()-t0)/60:.1f}m")
except Exception as e:
    log(f"  Haystack failed: {e}")

# ── 6. Scanpy Leiden + Wilcoxon ───────────────────────────────────────────────
scanpy_path = SCORES / "scanpy_scores.npz"
log("Running Scanpy Leiden+Wilcoxon...")
t0 = time.time()
try:
    sc.tl.leiden(adata, resolution=LEIDEN_RES, key_added="leiden_scanpy",
                 flavor="igraph", n_iterations=2, directed=False)
    sc.tl.rank_genes_groups(adata, groupby="leiden_scanpy", method="wilcoxon",
                            key_added="rank_genes_scanpy", use_raw=False, pts=False)
    rgg = adata.uns["rank_genes_scanpy"]
    gene_pval_min = {}
    for grp in rgg["names"].dtype.names:
        for g, p in zip(rgg["names"][grp], rgg["pvals_adj"][grp]):
            if g not in gene_pval_min or p < gene_pval_min[g]:
                gene_pval_min[g] = p
    var_names = np.array(list(gene_pval_min.keys()))
    pval_min  = np.array([gene_pval_min[g] for g in var_names])
    np.savez(scanpy_path, var_names=var_names, pval_min=pval_min)
    log(f"  Scanpy done in {(time.time()-t0)/60:.1f}m, {len(var_names)} genes, "
        f"{adata.obs['leiden_scanpy'].nunique()} Leiden clusters")
except Exception as e:
    log(f"  Scanpy failed: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
log("\n=== Loading rankings for summary ===")
rankings = {}
COLORS = {"Locat": "#e6194b", "GSPA": "#3cb44b", "LMD": "#4363d8",
          "Hotspot": "#f58231", "Haystack": "#911eb4", "Scanpy": "#f032e6"}
METHOD_ORDER = ["Locat", "GSPA", "LMD", "Haystack", "Hotspot", "Scanpy"]

def try_load(path, key, ascending):
    if path.exists():
        x = np.load(path, allow_pickle=True)
        return pd.Series(x[key], index=x["var_names"] if "var_names" in x else x["gene_names"]).sort_values(ascending=ascending).index.tolist()
    return []

rankings["Locat"]    = try_load(locat_path, "pval", True)
rankings["GSPA"]     = try_load(gspa_path, "gene_localization", False)
rankings["LMD"]      = try_load(lmd_path, "lmd_score", True)
rankings["Hotspot"]  = try_load(hotspot_path, "fdr", True)
rankings["Haystack"] = try_load(haystack_path, "logpval", True)
rankings["Scanpy"]   = try_load(scanpy_path, "pval_min", True)

log("\nTop-50 mean tau per method:")
for m in METHOD_ORDER:
    if not rankings.get(m):
        log(f"  {m}: N/A")
        continue
    vals = tau[[g for g in rankings[m] if g in tau.index]][:50].dropna()
    log(f"  {m}: {vals.mean():.4f} +/- {vals.std():.4f} (n={len(vals)})")

log("\nAll methods complete.")
