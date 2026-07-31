"""
Plot τ distribution comparison across datasets:
  - All genes passing ≥5% expression filter
  - Locat top-100 genes
  - Rare genes: expressed in < 1/(2*K) fraction of cells (K = n cell types)

One panel per dataset (PBMC3k, DermalC, Kang stim, Kang ctrl).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

HERE = Path(__file__).parent

DATASETS = [
    dict(
        name="PBMC3k",
        data_path=Path(__file__).resolve().parents[2] / "data/pbmc3k_9543_lognorm.h5ad",
        celltype_col="louvain",
        locat_scores=HERE / "FigS1_3kPBMC/celltype_specificity_comparison/scores/seed_0/locat_scores.npz",
        normalize=False,
        condition=None,
    ),
    dict(
        name="DermalC",
        data_path=Path(__file__).resolve().parents[2] / "data/E145_dermal_erez_2026/dc_adata_proc.h5ad",
        celltype_col="celltype",
        locat_scores=HERE / "Fig2_Dermal_Condensate/celltype_specificity_comparison/scores/locat_scores.npz",
        normalize=False,
        condition=None,
    ),
    dict(
        name="Kang Stim",
        data_path=Path(__file__).resolve().parents[2] / "data/kang_counts_25k.h5ad",
        celltype_col="cell_type",
        locat_scores=HERE / "Perturb_PBMC/celltype_specificity_comparison/scores/stim/locat_scores.npz",
        normalize=True,
        condition="stim",
    ),
    dict(
        name="Kang Ctrl",
        data_path=Path(__file__).resolve().parents[2] / "data/kang_counts_25k.h5ad",
        celltype_col="cell_type",
        locat_scores=HERE / "Perturb_PBMC/celltype_specificity_comparison/scores/ctrl/locat_scores.npz",
        normalize=True,
        condition="ctrl",
    ),
]

def compute_tau_series(adata, celltype_col):
    """Return tau for ALL genes (no pct filter), plus pct_expr."""
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    cts = adata.obs[celltype_col].cat.categories.tolist()
    mean = np.array([X[adata.obs[celltype_col] == ct].mean(axis=0) for ct in cts])
    pct  = (X > 0).mean(axis=0)
    rs   = mean.sum(axis=0)
    tau  = np.where(rs > 0, mean.max(axis=0) / rs, np.nan)
    return pd.Series(tau, index=adata.var_names), pd.Series(pct, index=adata.var_names), len(cts)

fig, axes = plt.subplots(1, len(DATASETS), figsize=(4 * len(DATASETS), 5), sharey=False)

for ax, ds in zip(axes, DATASETS):
    print(f"Processing {ds['name']}...", flush=True)

    adata = sc.read_h5ad(ds["data_path"])
    if ds["condition"]:
        adata = adata[adata.obs["label"] == ds["condition"]].copy()
    if ds["normalize"]:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    adata.obs[ds["celltype_col"]] = pd.Categorical(adata.obs[ds["celltype_col"]])

    tau, pct, K = compute_tau_series(adata, ds["celltype_col"])
    rare_thresh = 1.0 / (2 * K)
    print(f"  {adata.n_obs} cells, {K} cell types, rare threshold = {rare_thresh:.3f} ({rare_thresh*100:.1f}%)", flush=True)

    # ── three gene sets ────────────────────────────────────────────────────────
    # 1. All genes passing ≥5% filter
    mask_5pct = pct >= 0.05
    tau_all = tau[mask_5pct].dropna().values
    print(f"  ≥5% genes: {mask_5pct.sum()} → {len(tau_all)} with valid τ", flush=True)

    # 2. Locat top-100 (filter NaN, same as subsample bootstrap)
    x = np.load(ds["locat_scores"], allow_pickle=True)
    locat_ranked = pd.Series(x["pval"], index=x["gene_names"]).sort_values().index.tolist()
    top100 = [g for g in locat_ranked if g in tau.index and not np.isnan(tau[g]) and pct[g] >= 0.05][:100]
    tau_top100 = tau[top100].values
    print(f"  Locat top-100 (≥5%): {len(top100)} genes", flush=True)

    # 3. Rare genes: expressed in < 1/(2K) of cells, τ computed without pct filter
    mask_rare = pct < rare_thresh
    tau_rare = tau[mask_rare].dropna().values
    print(f"  Rare genes (<{rare_thresh:.3f}): {mask_rare.sum()} → {len(tau_rare)} with valid τ", flush=True)

    # ── boxplot ───────────────────────────────────────────────────────────────
    groups = [tau_all, tau_top100, tau_rare]
    labels = [
        f"All genes\n(≥5%, n={len(tau_all)})",
        f"Locat top-100\n(n={len(tau_top100)})",
        f"Rare genes\n(<{rare_thresh*100:.0f}%, n={len(tau_rare)})",
    ]
    colors_box = ["#aaaaaa", "#e6194b", "#4488cc"]

    bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=2, alpha=0.3, linestyle="none"))
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(ds["name"], fontsize=11, fontweight="bold")
    ax.set_ylabel("τ (cell-type specificity)" if ax == axes[0] else "")
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax)

    # median annotations
    for i, vals in enumerate(groups, start=1):
        med = np.median(vals)
        ax.text(i, ax.get_ylim()[1] * 0.97, f"med={med:.3f}",
                ha="center", va="top", fontsize=7, color="black")

fig.suptitle("τ distribution: all expressed genes vs Locat top-100 vs rare genes",
             y=1.02, fontsize=12)
plt.tight_layout()
out = HERE / "tau_distribution_comparison.svg"
plt.savefig(out, bbox_inches="tight")
print(f"\nSaved {out}", flush=True)
