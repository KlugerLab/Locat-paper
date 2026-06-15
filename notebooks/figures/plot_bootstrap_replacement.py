"""
Boxplots of bootstrap-replacement results (10 boots, w/ replacement, full dataset size).
For each dataset × cutoff:
  - Box 0: τ distribution over ALL genes (original data, no outliers shown)
  - Boxes 1-5: per-boot mean τ for each method, sorted by mean descending
  - Significance bracket: Locat vs next-best (paired one-sided t-test)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scanpy as sc

HERE = Path(__file__).parent

METHOD_ORDER = ["Locat", "GSPA", "LMD", "Haystack", "Hotspot"]
COLORS = {"Locat": "#e6194b", "GSPA": "#3cb44b", "LMD": "#4363d8",
          "Hotspot": "#f58231", "Haystack": "#911eb4"}
ALL_COLOR = "#cccccc"

DATASETS = [
    dict(
        name="PBMC3k",
        raw_npy=HERE / "FigS1_3kPBMC/celltype_specificity_comparison/bootstrap_replacement/subsample_bootstrap_raw.npy",
        out_dir=HERE / "FigS1_3kPBMC/celltype_specificity_comparison/bootstrap_replacement",
        data_path=Path("/banach2/wes/Locat-paper-repro-private/data/pbmc3k_9543_lognorm.h5ad"),
        celltype_col="louvain",
        cutoffs=[50, 100, 200],
        normalize=False,
        condition=None,
        tau_pct_thresh=0.05,
    ),
    dict(
        name="Dermal Condensates",
        raw_npy=HERE / "Fig2_Dermal_Condensate/celltype_specificity_comparison/bootstrap_replacement/subsample_bootstrap_raw.npy",
        out_dir=HERE / "Fig2_Dermal_Condensate/celltype_specificity_comparison/bootstrap_replacement",
        data_path=Path("/banach2/wes/Locat/data/E145_dermal_erez_2026/dc_adata_proc.h5ad"),
        celltype_col="celltype",
        cutoffs=[50, 100, 200],
        normalize=False,
        condition=None,
        tau_pct_thresh=0.05,
    ),
    dict(
        name="Kang Stim",
        raw_npy=HERE / "Perturb_PBMC/celltype_specificity_comparison/bootstrap_replacement_stim/subsample_bootstrap_raw.npy",
        out_dir=HERE / "Perturb_PBMC/celltype_specificity_comparison/bootstrap_replacement_stim",
        data_path=Path("/banach2/wes/Locat-paper-repro-private/data/kang_counts_25k.h5ad"),
        celltype_col="cell_type",
        cutoffs=[50, 100, 200],
        normalize=True,
        condition="stim",
        tau_pct_thresh=0.0,
        gene_filter="n100",
    ),
    dict(
        name="Kang Ctrl",
        raw_npy=HERE / "Perturb_PBMC/celltype_specificity_comparison/bootstrap_replacement_ctrl/subsample_bootstrap_raw.npy",
        out_dir=HERE / "Perturb_PBMC/celltype_specificity_comparison/bootstrap_replacement_ctrl",
        data_path=Path("/banach2/wes/Locat-paper-repro-private/data/kang_counts_25k.h5ad"),
        celltype_col="cell_type",
        cutoffs=[50, 100, 200],
        normalize=True,
        condition="ctrl",
        tau_pct_thresh=0.0,
        gene_filter="n100",
    ),
]

def compute_tau_all(adata, celltype_col, pct_thresh):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    cts = adata.obs[celltype_col].cat.categories.tolist()
    mean = np.array([X[adata.obs[celltype_col] == ct].mean(axis=0) for ct in cts])
    pct  = (X > 0).mean(axis=0)
    rs   = mean.sum(axis=0)
    mask = (rs > 0) & (pct >= pct_thresh)
    tau  = np.where(mask, mean.max(axis=0) / np.where(rs > 0, rs, 1.0), np.nan)
    return tau[~np.isnan(tau)]

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

for ds in DATASETS:
    print(f"\nProcessing {ds['name']}...", flush=True)

    # ── Load original data for all-genes τ ───────────────────────────────────
    adata = sc.read_h5ad(ds["data_path"])
    if ds["condition"]:
        adata = adata[adata.obs["label"] == ds["condition"]].copy()
    if ds["normalize"]:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if ds.get("gene_filter") == "n100":
        expr = (adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)) > 0
        adata = adata[:, expr.sum(axis=0) >= 100].copy()
    adata.obs[ds["celltype_col"]] = pd.Categorical(adata.obs[ds["celltype_col"]])

    tau_all = compute_tau_all(adata, ds["celltype_col"], ds["tau_pct_thresh"])
    print(f"  All-genes τ: n={len(tau_all)}, median={np.median(tau_all):.3f}", flush=True)

    # ── Load bootstrap raw ────────────────────────────────────────────────────
    d = np.load(ds["raw_npy"], allow_pickle=True).item()

    for k in ds["cutoffs"]:
        boot_vals = {m: np.array([v for v in d[m][k] if not np.isnan(v)]) for m in METHOD_ORDER}

        # sort methods by mean descending
        sorted_methods = sorted(METHOD_ORDER, key=lambda m: boot_vals[m].mean() if len(boot_vals[m]) > 0 else 0, reverse=True)
        best_other = next(m for m in sorted_methods if m != "Locat")

        # paired t-test Locat vs next-best
        lv = boot_vals["Locat"]
        bv = boot_vals[best_other]
        n_paired = min(len(lv), len(bv))
        if n_paired >= 2:
            _, p = stats.ttest_rel(lv[:n_paired], bv[:n_paired], alternative="greater")
        else:
            p = np.nan
        sig = sig_label(p) if not np.isnan(p) else "n.d."
        print(f"  top-{k}: Locat={lv.mean():.3f} vs {best_other}={bv.mean():.3f}  p={p:.4f} {sig}", flush=True)

        # ── Plot ─────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 5))

        # methods first, all-genes last
        positions  = list(range(len(sorted_methods) + 1))
        all_data   = [boot_vals[m] for m in sorted_methods] + [tau_all]
        all_colors = [COLORS[m] for m in sorted_methods] + [ALL_COLOR]
        all_labels = sorted_methods + [f"All genes\n(n={len(tau_all)})"]

        bp = ax.boxplot(
            all_data,
            positions=positions,
            patch_artist=True,
            widths=0.55,
            showfliers=False,   # never show outlier circles
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch, c in zip(bp["boxes"], all_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(all_labels, fontsize=8.5)
        ax.set_ylabel("τ (cell-type specificity)")
        ax.set_title(f"{ds['name']} — Top-{k} genes\n"
                     f"Bootstrap w/ replacement (n=10 boots)")

        # ── significance bracket: Locat vs next-best ──────────────────────
        locat_pos  = positions[sorted_methods.index("Locat")]
        other_pos  = positions[sorted_methods.index(best_other)]
        locat_top  = np.max(boot_vals["Locat"]) if len(boot_vals["Locat"]) else lv.mean()
        other_top  = np.max(boot_vals[best_other]) if len(boot_vals[best_other]) else bv.mean()
        y_bracket  = max(locat_top, other_top) + 0.03
        y_text     = y_bracket + 0.015

        ax.plot([locat_pos, locat_pos, other_pos, other_pos],
                [y_bracket - 0.01, y_bracket, y_bracket, y_bracket - 0.01],
                color="black", linewidth=1.2)
        ax.text((locat_pos + other_pos) / 2, y_text, sig,
                ha="center", va="bottom", fontsize=11,
                fontweight="bold" if sig != "n.s." else "normal")

        # mean labels on method boxes
        for i, m in enumerate(sorted_methods):
            v = boot_vals[m]
            if len(v) > 0:
                ax.text(i, v.mean(), f"{v.mean():.3f}",
                        ha="center", va="bottom", fontsize=7, color="black",
                        bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.7))

        sns.despine(ax=ax)
        plt.tight_layout()
        out = ds["out_dir"] / f"bootstrap_replacement_boxplot_top{k}.svg"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}", flush=True)

print("\nAll done.", flush=True)
