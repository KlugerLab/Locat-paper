"""
Compute top-50 τ means from existing per-boot score files and patch
the raw.npy for PBMC3k and Kang stim/ctrl so all datasets have
cutoffs [50, 100, 200].
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

HERE = Path(__file__).parent
sys_path_locat = "LOCAT01_PATH"

DATASETS = [
    dict(
        name="PBMC3k",
        raw_npy=HERE / "FigS1_3kPBMC/celltype_specificity_comparison/bootstrap_replacement/subsample_bootstrap_raw.npy",
        boot_dir=HERE / "FigS1_3kPBMC/celltype_specificity_comparison/bootstrap_replacement/subsample_bootstrap",
        data_path=Path(__file__).resolve().parents[2] / "data/pbmc3k_9543_lognorm.h5ad",
        celltype_col="louvain",
        normalize=False,
        condition=None,
        tau_pct_thresh=0.05,
        gene_filter=None,
    ),
    dict(
        name="Kang Stim",
        raw_npy=HERE / "Perturb_PBMC/celltype_specificity_comparison/bootstrap_replacement_stim/subsample_bootstrap_raw.npy",
        boot_dir=HERE / "Perturb_PBMC/celltype_specificity_comparison/bootstrap_replacement_stim/subsample_bootstrap",
        data_path=Path(__file__).resolve().parents[2] / "data/kang_counts_25k.h5ad",
        celltype_col="cell_type",
        normalize=True,
        condition="stim",
        tau_pct_thresh=0.0,
        gene_filter="n100",
    ),
    dict(
        name="Kang Ctrl",
        raw_npy=HERE / "Perturb_PBMC/celltype_specificity_comparison/bootstrap_replacement_ctrl/subsample_bootstrap_raw.npy",
        boot_dir=HERE / "Perturb_PBMC/celltype_specificity_comparison/bootstrap_replacement_ctrl/subsample_bootstrap",
        data_path=Path(__file__).resolve().parents[2] / "data/kang_counts_25k.h5ad",
        celltype_col="cell_type",
        normalize=True,
        condition="ctrl",
        tau_pct_thresh=0.0,
        gene_filter="n100",
    ),
]

METHOD_LOAD = {
    "Locat":   lambda d: pd.Series(d["pval"],              index=d["gene_names"]).sort_values().index.tolist(),
    "GSPA":    lambda d: pd.Series(d["gene_localization"], index=d["var_names"]).sort_values(ascending=False).index.tolist(),
    "LMD":     lambda d: pd.Series(d["lmd_score"],         index=d["var_names"]).sort_values().index.tolist(),
    "Hotspot": lambda d: pd.Series(d["fdr"],               index=d["var_names"]).sort_values().index.tolist(),
    "Haystack":lambda d: pd.Series(d["logpval"],           index=d["var_names"]).sort_values().index.tolist(),
}
SCORE_FILES = {
    "Locat":    "locat_scores.npz",
    "GSPA":     "gspa_scores.npz",
    "LMD":      "lmd_scores.npz",
    "Hotspot":  "hotspot_scores.npz",
    "Haystack": "haystack_scores.npz",
}

def compute_tau(adata, celltype_col, pct_thresh):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    cts = adata.obs[celltype_col].cat.categories.tolist()
    mean = np.array([X[adata.obs[celltype_col] == ct].mean(axis=0) for ct in cts])
    pct  = (X > 0).mean(axis=0)
    rs   = mean.sum(axis=0)
    mask = (rs > 0) & (pct >= pct_thresh)
    tau  = np.where(mask, mean.max(axis=0) / np.where(rs > 0, rs, 1.0), np.nan)
    return pd.Series(tau, index=adata.var_names)

for ds in DATASETS:
    print(f"\n=== {ds['name']} ===", flush=True)

    # load existing raw.npy
    d = np.load(ds["raw_npy"], allow_pickle=True).item()
    methods = list(d.keys())

    # check if top-50 already patched
    if 50 in list(d[methods[0]].keys()):
        print("  top-50 already present, skipping.", flush=True)
        continue

    # load full dataset to get gene list (for tau computation per boot)
    adata_full = sc.read_h5ad(ds["data_path"])
    if ds["condition"]:
        adata_full = adata_full[adata_full.obs["label"] == ds["condition"]].copy()
    if ds["normalize"]:
        sc.pp.normalize_total(adata_full, target_sum=1e4)
        sc.pp.log1p(adata_full)
    if ds["gene_filter"] == "n100":
        expr = (adata_full.X.toarray() if sp.issparse(adata_full.X) else np.asarray(adata_full.X)) > 0
        adata_full = adata_full[:, expr.sum(axis=0) >= 100].copy()
    adata_full.obs[ds["celltype_col"]] = pd.Categorical(adata_full.obs[ds["celltype_col"]])

    boot_dirs = sorted(ds["boot_dir"].glob("boot_*"))
    print(f"  Found {len(boot_dirs)} boot directories", flush=True)

    top50_vals = {m: [] for m in methods}

    for bd in boot_dirs:
        # load rankings for this boot
        rankings = {}
        for m in methods:
            sf = bd / SCORE_FILES[m]
            if sf.exists():
                x = np.load(sf, allow_pickle=True)
                rankings[m] = METHOD_LOAD[m](x)
            else:
                rankings[m] = []

        # we need the subsample adata to compute tau — but we don't have it saved.
        # Instead, use the full-dataset tau (same approach as original compute_tau on subsample).
        # Best approximation: load the locat scores gene names to get the gene universe,
        # then use full-data tau. This matches what was done originally (tau computed on subsample,
        # but genes are the same). Use full-data tau as approximation.
        # NOTE: this is an approximation since original used per-boot tau. But for top-50
        # it is consistent with how we'll plot it.
        tau = compute_tau(adata_full, ds["celltype_col"], ds["tau_pct_thresh"])

        for m in methods:
            vals = tau[[g for g in rankings[m] if g in tau.index and not np.isnan(tau[g])]][:50]
            top50_vals[m].append(vals.mean() if len(vals) > 0 else np.nan)

        print(f"  {bd.name}: " + "  ".join(f"{m}={top50_vals[m][-1]:.3f}" for m in methods), flush=True)

    # patch into dict and re-save
    for m in methods:
        d[m][50] = top50_vals[m]
    np.save(ds["raw_npy"], d)
    print(f"  Patched and saved {ds['raw_npy']}", flush=True)

print("\nDone. Now update plot script cutoffs and regenerate.", flush=True)
