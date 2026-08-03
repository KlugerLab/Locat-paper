"""
Run LOCAT on Chlamydomonas Fe_neg and Fe_pos samples (run1, strain CC5390).

Preprocessing mirrors the Kang/DermalC pipelines:
  - Load CSV (genes x cells), transpose to AnnData (cells x genes)
  - Standard QC: min 200 genes/cell, max 6000 genes/cell (doublet proxy),
    genes expressed in ≥3 cells
  - normalize_total (1e4) + log1p
  - Filter to genes expressed in ≥5% of cells
  - PCA (50 PCs) + neighbors (n_neighbors=20)
  - LOCAT: first 8 PCs, k=20, n_bootstrap_inits=50, _reg_covar=1e-6,
    gmm_scan with max_freq=0.9, include_depletion_scan=True,
    rc_lambda_values=linspace(1.0, 2.0, 8)

Saves per-sample:
  - scores/<cond>/locat_scores.npz  (gene_names, pval)
  - scores/<cond>/adata_proc.h5ad   (processed adata for downstream use)

Usage:
    python run_chlamydomonas_locat.py [--gpu 0]
"""
import argparse, os, sys, time
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
DATA_DIR  = Path(__file__).resolve().parents[4] / "data/chlamydomonas"
SCORES    = HERE / "scores"

if str(LOCAT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAT_SRC))

from locat.locat import LOCAT

SAMPLES = [
    dict(
        name="fe_neg",
        csv=DATA_DIR / "GSM4770979_run1_CC5390_Fe_neg.csv.gz",
        label="Fe− (iron-deficient)",
    ),
    dict(
        name="fe_pos",
        csv=DATA_DIR / "GSM4770980_run1_CC5390_Fe_pos.csv.gz",
        label="Fe+ (iron-replete)",
    ),
]

# QC thresholds — chosen to match ~3k post-QC cells per sample as in scPrisma
MIN_GENES   = 200
MAX_GENES   = 6000   # rough doublet proxy
MIN_CELLS   = 3      # gene must be in ≥3 cells to be kept
PCT_THRESH  = 0.05   # ≥5% cells expressing gene, after QC

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

for ds in SAMPLES:
    cond_dir = SCORES / ds["name"]
    out_locat = cond_dir / "locat_scores.npz"
    out_adata = cond_dir / "adata_proc.h5ad"

    if out_locat.exists():
        log(f"=== {ds['label']}: already done, skipping ===")
        continue

    log(f"=== {ds['label']} ===")

    # ── Load ──────────────────────────────────────────────────────────────────
    log(f"  Loading {ds['csv'].name}...")
    t0 = time.time()
    df = pd.read_csv(ds["csv"], index_col=0)  # genes x cells
    adata = sc.AnnData(X=sp.csr_matrix(df.values.T.astype(np.float32)),
                       obs=pd.DataFrame(index=df.columns),
                       var=pd.DataFrame(index=df.index))
    log(f"  Raw: {adata.n_obs} cells × {adata.n_vars} genes  ({(time.time()-t0):.1f}s)")

    # ── QC ────────────────────────────────────────────────────────────────────
    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    sc.pp.filter_cells(adata, max_genes=MAX_GENES)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    log(f"  After QC: {adata.n_obs} cells × {adata.n_vars} genes")

    # ── Normalize + log1p ─────────────────────────────────────────────────────
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ── Gene filter ≥5% ───────────────────────────────────────────────────────
    pct = (adata.X.toarray() > 0).mean(axis=0)
    adata = adata[:, pct >= PCT_THRESH].copy()
    log(f"  After ≥5% filter: {adata.n_obs} cells × {adata.n_vars} genes")

    # ── PCA + neighbors ───────────────────────────────────────────────────────
    log("  Computing PCA (50 PCs) + neighbors...")
    sc.pp.pca(adata, n_comps=50, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)

    # save processed adata
    adata.write_h5ad(out_adata)
    log(f"  Saved processed adata → {out_adata.name}")

    # ── LOCAT ─────────────────────────────────────────────────────────────────
    log("  Running LOCAT...")
    t0 = time.time()
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
    np.savez(out_locat,
             gene_names=locat_df.index.values,
             pval=locat_df["pval"].values)
    log(f"  LOCAT done in {(time.time()-t0)/60:.1f}m  →  {out_locat.name}")
    log(f"  Top 20 genes: {locat_df.index[:20].tolist()}")

log("All done.")
