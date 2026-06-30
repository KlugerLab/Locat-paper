"""
Run Scanpy Leiden clustering + Wilcoxon rank_genes_groups on all 4 real
datasets and save scanpy_scores.npz into each dataset's score directory.

Preprocessing mirrors the other methods (see run_giniclust_real_dataset.py):
  - same normalization / gene-filter logic
  - if X_pca exists in adata.obsm, re-use it; otherwise run PCA (50 PCs)
  - compute neighbors on PCA (n_neighbors=30), run Leiden (resolution=0.5)
  - rank_genes_groups with method='wilcoxon', groupby='leiden'

Ranking key: pval_min — minimum adjusted p-value across all Leiden clusters
for each gene.  Lower pval_min = gene distinguishes at least one cluster more
cleanly = higher in ranking (sort ascending).

Usage:
    /banach2/wes/.conda/envs/mulde_jax/bin/python run_scanpy_wilcoxon_real_dataset.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

HERE = Path(__file__).parent

DATASETS = [
    dict(
        name="PBMC3k",
        data_path=Path("/banach2/wes/Locat-paper-repro-private/data/pbmc3k_9543_lognorm.h5ad"),
        scores_dir=HERE / "FigS1_3kPBMC/celltype_specificity_comparison/scores/seed_0",
        celltype_col="louvain",
        normalize=False,
        condition=None,
        gene_filter=None,
    ),
    dict(
        name="Dermal Condensates",
        data_path=Path("/banach2/wes/Locat/data/E145_dermal_erez_2026/dc_adata_proc.h5ad"),
        scores_dir=HERE / "Fig2_Dermal_Condensate/celltype_specificity_comparison/scores",
        celltype_col="celltype",
        normalize=False,
        condition=None,
        gene_filter=None,
    ),
    dict(
        name="Kang Stim",
        data_path=Path("/banach2/wes/Locat-paper-repro-private/data/kang_counts_25k.h5ad"),
        scores_dir=HERE / "Perturb_PBMC/celltype_specificity_comparison/real_dataset_n100_stim",
        celltype_col="cell_type",
        normalize=True,
        condition="stim",
        gene_filter="n100",
    ),
    dict(
        name="Kang Ctrl",
        data_path=Path("/banach2/wes/Locat-paper-repro-private/data/kang_counts_25k.h5ad"),
        scores_dir=HERE / "Perturb_PBMC/celltype_specificity_comparison/real_dataset_n100_ctrl",
        celltype_col="cell_type",
        normalize=True,
        condition="ctrl",
        gene_filter="n100",
    ),
]

LEIDEN_RESOLUTION = 0.5
N_NEIGHBORS = 30
N_PCS = 50

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

for ds in DATASETS:
    log(f"=== {ds['name']} ===")
    out_path = ds["scores_dir"] / "scanpy_scores.npz"
    if out_path.exists():
        log(f"  Already exists, skipping: {out_path}")
        continue

    adata = sc.read_h5ad(ds["data_path"])
    if ds["condition"]:
        adata = adata[adata.obs["label"] == ds["condition"]].copy()
    if ds["normalize"]:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if ds["gene_filter"] == "n100":
        expr = (adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)) > 0
        adata = adata[:, expr.sum(axis=0) >= 100].copy()
    adata.obs[ds["celltype_col"]] = pd.Categorical(adata.obs[ds["celltype_col"]])
    log(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    # PCA — re-use existing embedding if present, otherwise compute
    if "X_pca" not in adata.obsm:
        log(f"  Running PCA (n_comps={N_PCS})...")
        sc.pp.pca(adata, n_comps=N_PCS, svd_solver="arpack")
    else:
        log(f"  Re-using existing X_pca ({adata.obsm['X_pca'].shape[1]} PCs)")

    log(f"  Computing neighbors (n_neighbors={N_NEIGHBORS})...")
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=min(N_PCS, adata.obsm["X_pca"].shape[1]))

    log(f"  Running Leiden (resolution={LEIDEN_RESOLUTION})...")
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added="leiden_scanpy_wilcoxon")
    n_clusters = adata.obs["leiden_scanpy_wilcoxon"].nunique()
    log(f"  Found {n_clusters} Leiden clusters")

    log("  Running rank_genes_groups (Wilcoxon)...")
    t0 = time.time()
    sc.tl.rank_genes_groups(
        adata,
        groupby="leiden_scanpy_wilcoxon",
        method="wilcoxon",
        key_added="rank_genes_scanpy_wilcoxon",
        use_raw=False,
        pts=False,
    )
    log(f"  Done in {(time.time()-t0)/60:.1f}m")

    # Aggregate per-gene minimum pval_adj across all clusters
    rgg = adata.uns["rank_genes_scanpy_wilcoxon"]
    gene_names_arr = rgg["names"]   # structured array: shape (n_genes, n_groups)
    pvals_adj_arr  = rgg["pvals_adj"]

    # Build dict: gene -> min pval_adj across groups
    gene_pval_min = {}
    n_genes_per_group = gene_names_arr.shape[0]
    for group_idx, group_name in enumerate(gene_names_arr.dtype.names):
        genes  = gene_names_arr[group_name]
        pvals  = pvals_adj_arr[group_name]
        for g, p in zip(genes, pvals):
            if g not in gene_pval_min or p < gene_pval_min[g]:
                gene_pval_min[g] = p

    var_names = np.array(list(gene_pval_min.keys()))
    pval_min  = np.array([gene_pval_min[g] for g in var_names])

    np.savez(
        out_path,
        var_names=var_names,
        pval_min=pval_min,
    )
    log(f"  Saved {out_path}  ({len(var_names)} genes)")

log("All done.")
