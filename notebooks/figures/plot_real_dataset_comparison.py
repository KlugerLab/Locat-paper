"""
Single-run (real dataset) comparison: each method run once on original data.
For each method × cutoff, shows mean τ of top-k genes ± 95% CI.

CI = mean ± 1.96 * SE,  SE = std(τ_top_k_genes) / sqrt(k)
This is the standard error of the mean, treating the k genes as
independent samples — appropriate since we care about the mean τ
as a summary statistic and want uncertainty around that estimate.

NOTE: Kang scores here used the >=5% expression filter (2579/2629 genes),
while the bootstrap runs used n_cells>=100 (~7k genes). Values are not
directly comparable across those two experiment types for Kang.

Output goes in real_dataset/ subdirectory within each figure folder.
"""
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

METHOD_ORDER = ["Locat", "GSPA", "LMD", "Haystack", "Hotspot"]
COLORS = {"Locat": "#e6194b", "GSPA": "#3cb44b", "LMD": "#4363d8",
          "Hotspot": "#f58231", "Haystack": "#911eb4"}
ALL_COLOR = "#cccccc"
CUTOFFS = [50, 100, 200]

DATASETS = [
    dict(
        name="PBMC3k",
        scores_dir=HERE / "FigS1_3kPBMC/celltype_specificity_comparison/scores/seed_0",
        out_dir=HERE / "FigS1_3kPBMC/celltype_specificity_comparison/real_dataset",
        data_path=Path("/banach2/wes/Locat-paper-repro-private/data/pbmc3k_9543_lognorm.h5ad"),
        celltype_col="louvain",
        normalize=False,
        condition=None,
        tau_pct_thresh=0.05,
        gene_filter=None,
    ),
    dict(
        name="Dermal Condensates",
        scores_dir=HERE / "Fig2_Dermal_Condensate/celltype_specificity_comparison/scores",
        out_dir=HERE / "Fig2_Dermal_Condensate/celltype_specificity_comparison/real_dataset",
        data_path=Path("/banach2/wes/Locat/data/E145_dermal_erez_2026/dc_adata_proc.h5ad"),
        celltype_col="celltype",
        normalize=False,
        condition=None,
        tau_pct_thresh=0.05,
        gene_filter=None,
    ),
    dict(
        name="Kang Stim",
        scores_dir=HERE / "Perturb_PBMC/celltype_specificity_comparison/real_dataset_n100_stim",
        out_dir=HERE / "Perturb_PBMC/celltype_specificity_comparison/real_dataset_stim",
        data_path=Path("/banach2/wes/Locat-paper-repro-private/data/kang_counts_25k.h5ad"),
        celltype_col="cell_type",
        normalize=True,
        condition="stim",
        tau_pct_thresh=0.0,
        gene_filter="n100",
    ),
    dict(
        name="Kang Ctrl",
        scores_dir=HERE / "Perturb_PBMC/celltype_specificity_comparison/real_dataset_n100_ctrl",
        out_dir=HERE / "Perturb_PBMC/celltype_specificity_comparison/real_dataset_ctrl",
        data_path=Path("/banach2/wes/Locat-paper-repro-private/data/kang_counts_25k.h5ad"),
        celltype_col="cell_type",
        normalize=True,
        condition="ctrl",
        tau_pct_thresh=0.0,
        gene_filter="n100",
    ),
]

def load_rankings(scores_dir):
    rankings = {}
    x = np.load(scores_dir / "locat_scores.npz",    allow_pickle=True)
    rankings["Locat"]    = pd.Series(x["pval"],              index=x["gene_names"]).sort_values().index.tolist()
    x = np.load(scores_dir / "gspa_scores.npz",     allow_pickle=True)
    rankings["GSPA"]     = pd.Series(x["gene_localization"], index=x["var_names"]).sort_values(ascending=False).index.tolist()
    x = np.load(scores_dir / "lmd_scores.npz",      allow_pickle=True)
    rankings["LMD"]      = pd.Series(x["lmd_score"],         index=x["var_names"]).sort_values().index.tolist()
    x = np.load(scores_dir / "hotspot_scores.npz",  allow_pickle=True)
    rankings["Hotspot"]  = pd.Series(x["fdr"],               index=x["var_names"]).sort_values().index.tolist()
    x = np.load(scores_dir / "haystack_scores.npz", allow_pickle=True)
    rankings["Haystack"] = pd.Series(x["logpval"],           index=x["var_names"]).sort_values().index.tolist()
    return rankings

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
    ds["out_dir"].mkdir(parents=True, exist_ok=True)

    # load data
    adata = sc.read_h5ad(ds["data_path"])
    if ds["condition"]:
        adata = adata[adata.obs["label"] == ds["condition"]].copy()
    if ds["normalize"]:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    adata.obs[ds["celltype_col"]] = pd.Categorical(adata.obs[ds["celltype_col"]])

    tau = compute_tau(adata, ds["celltype_col"], ds["tau_pct_thresh"])
    tau_all = tau.dropna().values
    print(f"  {adata.n_obs} cells, {(~tau.isna()).sum()} genes with valid τ, median={np.median(tau_all):.3f}", flush=True)

    rankings = load_rankings(ds["scores_dir"])

    for k in CUTOFFS:
        rows = []
        for m in METHOD_ORDER:
            gene_tau = tau[[g for g in rankings[m] if g in tau.index and not np.isnan(tau[g])]][:k]
            vals = gene_tau.values
            n = len(vals)
            mean_tau = vals.mean() if n > 0 else np.nan
            se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            ci = 1.96 * se
            rows.append({"method": m, "mean": mean_tau, "se": se, "ci": ci, "n": n})
            print(f"  top-{k} {m}: mean={mean_tau:.3f}  SE={se:.4f}  n={n}", flush=True)

        df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)

        # ── dot plot ──────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4.5))

        # grey band for all-genes median ± IQR
        q25, med, q75 = np.percentile(tau_all, [25, 50, 75])
        ax.axhspan(q25, q75, color=ALL_COLOR, alpha=0.25, zorder=0, label=f"All-genes IQR (med={med:.3f})")
        ax.axhline(med, color=ALL_COLOR, linewidth=1.5, linestyle="--", zorder=1)

        x_pos = np.arange(len(df))
        for i, row in df.iterrows():
            color = COLORS[row["method"]]
            ax.errorbar(i, row["mean"], yerr=row["ci"],
                        fmt="o", color=color, markersize=9,
                        capsize=5, capthick=1.5, linewidth=1.5, zorder=3)
            ax.text(i, row["mean"] + row["ci"] + 0.008, f"{row['mean']:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="black")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(df["method"], fontsize=9)
        ax.set_ylabel("Mean S (cell-type specificity)")
        ax.set_title(f"{ds['name']} — Top-{k} genes (single run)\n"
                     f"Mean S ± 95% CI  (CI = 1.96 × SD/√k,  k={k})")

        # legend for all-genes band
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=ALL_COLOR, alpha=0.4, label=f"All-genes IQR (median={med:.3f})")]
        ax.legend(handles=handles, fontsize=8, loc="lower right")

        sns.despine(ax=ax)
        plt.tight_layout()
        out = ds["out_dir"] / f"real_dataset_dotplot_top{k}.svg"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}", flush=True)

print("\nAll done.", flush=True)
