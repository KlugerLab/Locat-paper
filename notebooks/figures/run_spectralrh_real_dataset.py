"""
Run SpectralRH on all 4 real datasets and save spectralrh_scores.npz
into each dataset's existing score directory.

Ranking key: HR_avg_sizeNull = mean(p_entropy_sizeNull, pct_rayleigh_sizeNull)
Lower HR_avg_sizeNull = more structured/localized (rank ascending).

Usage:
    python run_spectralrh_real_dataset.py
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

LOCAT_SRC = Path("/banach2/wes/Locat")
if str(LOCAT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAT_SRC))

from locat.spectralrh import build_size_nulls, size_matched_pvalues, score_genes_with_graph_metrics

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

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

for ds in DATASETS:
    log(f"=== {ds['name']} ===")
    out_path = ds["scores_dir"] / "spectralrh_scores.npz"
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
    log(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    # PCA + neighbors (needed for Laplacian)
    log("  Computing PCA + neighbors...")
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)

    # SpectralRH scores
    log("  Running score_genes_with_graph_metrics...")
    t0 = time.time()
    scores = score_genes_with_graph_metrics(
        adata, use_connectivities=True, normalized_laplacian=True, m_eigs=96,
    )
    log(f"  Building null bank (P=256)...")
    bank = build_size_nulls(
        adata, W_key="connectivities", layer=None,
        normalized_laplacian=True, m_eigs=96, P=256, seed=42,
    )
    scores_sz = size_matched_pvalues(scores, bank, two_sided=False)
    scores_sz["HR_avg_sizeNull"] = scores_sz[["p_entropy_sizeNull", "pct_rayleigh_sizeNull"]].mean(axis=1)
    log(f"  Done in {(time.time()-t0)/60:.1f}m")

    np.savez(
        out_path,
        var_names=scores_sz.index.values,
        hr_avg_sizenull=scores_sz["HR_avg_sizeNull"].values,
    )
    log(f"  Saved {out_path}")

log("All done.")
