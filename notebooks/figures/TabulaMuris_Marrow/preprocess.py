"""
Preprocess Tabula Muris Senis bone marrow (10x) data: QC, normalize, gene
filter (>=5% expressing cells), standardize cell type column, PCA/neighbors/UMAP.

Usage:
    python preprocess.py
"""
import scanpy as sc, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path

HERE = Path(__file__).parent
PROC_PATH = HERE / "data/tabula_muris_marrow_proc.h5ad"

raw_path = HERE / "data/tabula_muris_marrow_raw.h5ad"
adata = sc.read_h5ad(raw_path)
print(f"Raw: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)

# use gene symbols as var_names (feature_name), keep ensembl id as backup
if "feature_name" in adata.var.columns:
    adata.var_names = adata.var["feature_name"].astype(str)
    adata.var_names_make_unique()
    adata.var = adata.var.drop(columns=["feature_name"])
    adata.var.index.name = None

# ── QC ────────────────────────────────────────────────────────────────────────
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_cells(adata, max_genes=6000)   # rough doublet proxy
sc.pp.filter_genes(adata, min_cells=3)
print(f"After QC filter: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)

# ── Normalize + log1p ─────────────────────────────────────────────────────────
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ── Gene filter >=5% ────────────────────────────────────────────────────────────
pct = (adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)) > 0
adata = adata[:, pct.mean(axis=0) >= 0.05].copy()
print(f"After >=5%% expressed filter: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)

# ── Single donor: filter to 30-M-2 (3307 cells, 15 cell types) ───────────────
# Multi-donor data produces fragmented UMAPs (one sub-cluster per mouse per
# cell type). Using one donor avoids batch effects without needing correction.
adata = adata[adata.obs["donor_id"] == "30-M-2"].copy()
print(f"After single-donor filter (30-M-2): {adata.n_obs} cells x {adata.n_vars} genes", flush=True)

# ── Cell type column ──────────────────────────────────────────────────────────
CELLTYPE_COL = "cell_type"
adata.obs[CELLTYPE_COL] = pd.Categorical(adata.obs[CELLTYPE_COL].astype(str))

# Drop cell types with fewer than 20 cells (too small for meaningful tau)
ct_counts = adata.obs[CELLTYPE_COL].value_counts()
keep_cts = ct_counts[ct_counts >= 20].index
adata = adata[adata.obs[CELLTYPE_COL].isin(keep_cts)].copy()
adata.obs[CELLTYPE_COL] = pd.Categorical(adata.obs[CELLTYPE_COL].astype(str))

print(f"After QC: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
print(adata.obs[CELLTYPE_COL].value_counts(), flush=True)

# ── PCA + neighbors + UMAP ─────────────────────────────────────────────────────
sc.pp.pca(adata, n_comps=50, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)
sc.tl.umap(adata)

# keep obs slim -- drop columns that are all-empty/constant categoricals from Census
# (avoids h5ad write issues with huge unused category lists)
for col in list(adata.obs.columns):
    if pd.api.types.is_categorical_dtype(adata.obs[col]):
        adata.obs[col] = adata.obs[col].astype(str).astype("category")

adata.write_h5ad(PROC_PATH)
print(f"Saved processed h5ad to {PROC_PATH}.", flush=True)
