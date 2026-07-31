"""
Run GiniClust3 on all 4 real datasets and save giniclust_scores.npz
into each dataset's existing score directory, matching the same
preprocessing/gene-filtering used for the other methods (see
run_spectralrh_real_dataset.py / plot_real_dataset_comparison.py).

Ranking key: gini_score (standard Gini coefficient of each gene's
expression across cells). Higher gini_score = more unequal expression
across cells = more specific/localized (rank descending, i.e. highest
gini_score first).

Usage:
    LOCAT_PYTHON run_giniclust_real_dataset.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

from giniclust3.gini import giniIndex

HERE = Path(__file__).parent

DATASETS = [
    dict(
        name="PBMC3k",
        data_path=Path(__file__).resolve().parents[2] / "data/pbmc3k_9543_lognorm.h5ad",
        scores_dir=HERE / "FigS1_3kPBMC/celltype_specificity_comparison/scores/seed_0",
        celltype_col="louvain",
        normalize=False,
        condition=None,
        gene_filter=None,
    ),
    dict(
        name="Dermal Condensates",
        data_path=Path(__file__).resolve().parents[2] / "data/E145_dermal_erez_2026/dc_adata_proc.h5ad",
        scores_dir=HERE / "Fig2_Dermal_Condensate/celltype_specificity_comparison/scores",
        celltype_col="celltype",
        normalize=False,
        condition=None,
        gene_filter=None,
    ),
    dict(
        name="Kang Stim",
        data_path=Path(__file__).resolve().parents[2] / "data/kang_counts_25k.h5ad",
        scores_dir=HERE / "Perturb_PBMC/celltype_specificity_comparison/real_dataset_n100_stim",
        celltype_col="cell_type",
        normalize=True,
        condition="stim",
        gene_filter="n100",
    ),
    dict(
        name="Kang Ctrl",
        data_path=Path(__file__).resolve().parents[2] / "data/kang_counts_25k.h5ad",
        scores_dir=HERE / "Perturb_PBMC/celltype_specificity_comparison/real_dataset_n100_ctrl",
        celltype_col="cell_type",
        normalize=True,
        condition="ctrl",
        gene_filter="n100",
    ),
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

for ds in DATASETS:
    log(f"=== {ds['name']} ===")
    out_path = ds["scores_dir"] / "giniclust_scores.npz"
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

    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float64)

    log("  Computing Gini index per gene...")
    t0 = time.time()
    gene_gini, gene_max = giniIndex(X.T)
    log(f"  Done in {(time.time()-t0)/60:.1f}m")

    np.savez(
        out_path,
        var_names=adata.var_names.values,
        gini_score=np.asarray(gene_gini),
    )
    log(f"  Saved {out_path}")

log("All done.")
