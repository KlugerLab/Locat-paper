"""
Generate real-dataset-style comparison plots for Chlamydomonas Fe- and Fe+
using Leiden res=0.3 clusters as pseudo-cell-types.

Produces per condition:
  - real_dataset_dotplot_top{k}.svg  (dot plot with all-genes IQR band)
  - top10_genes_per_method.svg       (horizontal bar chart: τ of top-10 per method)

Usage:
    python plot_chlamydomonas_comparison.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scanpy as sc

HERE   = Path(__file__).parent
SCORES = HERE / "locat_run" / "scores"
COMP   = HERE / "comparison_results"

LEIDEN_RES   = 0.3
CUTOFFS      = [50, 100, 200]
PCT_THRESH   = 0.05
ALL_COLOR    = "#cccccc"

METHOD_ORDER = ["Locat", "GSPA", "LMD", "Haystack", "Hotspot", "Scanpy"]
COLORS = {
    "Locat":    "#e6194b",
    "GSPA":     "#3cb44b",
    "LMD":      "#4363d8",
    "Hotspot":  "#f58231",
    "Haystack": "#911eb4",
    "Scanpy":   "#f032e6",
}

SAMPLES = [
    dict(name="fe_neg", label="Chlamydomonas Fe−", short="Fe−"),
    dict(name="fe_pos", label="Chlamydomonas Fe+", short="Fe+"),
]

def compute_tau(adata, group_col):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    groups = adata.obs[group_col].cat.categories.tolist()
    pct  = (X > 0).mean(axis=0)
    mean = np.array([X[adata.obs[group_col] == g].mean(axis=0) for g in groups])
    rs   = mean.sum(axis=0)
    mask = (rs > 0) & (pct >= PCT_THRESH)
    tau  = np.where(mask, mean.max(axis=0) / np.where(rs > 0, rs, 1.0), np.nan)
    return pd.Series(tau, index=adata.var_names)

def load_rankings(name):
    locat_path = SCORES / name / "locat_scores.npz"
    comp_dir   = COMP / name
    rankings = {}
    x = np.load(locat_path, allow_pickle=True)
    rankings["Locat"]    = pd.Series(x["pval"],              index=x["gene_names"]).sort_values().index.tolist()
    x = np.load(comp_dir / "gspa_scores.npz",    allow_pickle=True)
    rankings["GSPA"]     = pd.Series(x["gene_localization"], index=x["var_names"]).sort_values(ascending=False).index.tolist()
    x = np.load(comp_dir / "lmd_scores.npz",     allow_pickle=True)
    rankings["LMD"]      = pd.Series(x["lmd_score"],         index=x["var_names"]).sort_values().index.tolist()
    x = np.load(comp_dir / "hotspot_scores.npz", allow_pickle=True)
    rankings["Hotspot"]  = pd.Series(x["fdr"],               index=x["var_names"]).sort_values().index.tolist()
    x = np.load(comp_dir / "haystack_scores.npz",allow_pickle=True)
    rankings["Haystack"] = pd.Series(x["logpval"],           index=x["var_names"]).sort_values().index.tolist()
    x = np.load(comp_dir / "scanpy_scores.npz",  allow_pickle=True)
    rankings["Scanpy"]   = pd.Series(x["pval_min"],          index=x["var_names"]).sort_values().index.tolist()
    return rankings

def plot_dotplot(tau, tau_all, rankings, k, out_path, title):
    rows = []
    for m in METHOD_ORDER:
        gene_tau = tau[[g for g in rankings[m] if g in tau.index and not np.isnan(tau[g])]][:k]
        vals = gene_tau.values
        n = len(vals)
        mean_tau = vals.mean() if n > 0 else np.nan
        se  = vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        ci  = 1.96 * se
        rows.append({"method": m, "mean": mean_tau, "se": se, "ci": ci, "n": n})

    df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    q25, med, q75 = np.percentile(tau_all, [25, 50, 75])
    ax.axhspan(q25, q75, color=ALL_COLOR, alpha=0.25, zorder=0)
    ax.axhline(med, color=ALL_COLOR, linewidth=1.5, linestyle="--", zorder=1)

    for i, row in df.iterrows():
        color = COLORS[row["method"]]
        ax.errorbar(i, row["mean"], yerr=row["ci"],
                    fmt="o", color=color, markersize=9,
                    capsize=5, capthick=1.5, linewidth=1.5, zorder=3)
        ax.text(i, row["mean"] + row["ci"] + 0.008, f"{row['mean']:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="black")

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["method"], fontsize=9)
    ax.set_ylabel("Mean S (cell-type specificity)")
    ax.set_title(f"{title} — Top-{k} genes (single run)\n"
                 f"Mean S ± 95% CI  (CI = 1.96 × SD/√k,  k={k})")

    handles = [mpatches.Patch(facecolor=ALL_COLOR, alpha=0.4,
                              label=f"All-genes IQR (median={med:.3f})")]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")

def plot_top10_genes(tau, rankings, out_path, title, n_groups):
    """Horizontal bar chart of τ for the top-10 genes from each method."""
    n_methods = len(METHOD_ORDER)
    fig, axes = plt.subplots(1, n_methods, figsize=(3.2 * n_methods, 5), sharey=False)

    tau_all = tau.dropna().values
    q25, med, q75 = np.percentile(tau_all, [25, 50, 75])

    for ax, m in zip(axes, METHOD_ORDER):
        top10 = [g for g in rankings[m] if g in tau.index and not np.isnan(tau[g])][:10]
        vals  = tau[top10].values

        # short gene labels (strip .v5.5 suffix)
        labels = [g.replace(".v5.5", "").replace("Cre", "Cre") for g in top10]

        y = np.arange(len(vals))
        ax.barh(y, vals, color=COLORS[m], alpha=0.85, edgecolor="white", linewidth=0.5)

        # IQR band as vertical spans
        ax.axvspan(q25, q75, color=ALL_COLOR, alpha=0.2, zorder=0)
        ax.axvline(med, color=ALL_COLOR, linewidth=1.2, linestyle="--", zorder=1)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("τ", fontsize=8)
        ax.set_title(m, fontsize=9, color=COLORS[m], fontweight="bold")
        ax.set_xlim(0, 1.0)
        # annotate each bar with τ value
        for i, v in enumerate(vals):
            ax.text(min(v + 0.01, 0.97), i, f"{v:.2f}", va="center", fontsize=6.5)
        sns.despine(ax=ax)

    fig.suptitle(f"{title}\nTop-10 genes per method  (τ, Leiden {n_groups} clusters)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")

# ── Main ──────────────────────────────────────────────────────────────────────
for ds in SAMPLES:
    name  = ds["name"]
    label = ds["label"]
    print(f"\n=== {label} ===")

    out_dir = COMP / name / "real_dataset"
    out_dir.mkdir(exist_ok=True)

    # Load processed adata and assign Leiden clusters
    adata = sc.read_h5ad(SCORES / name / "adata_phases.h5ad")
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)
    sc.tl.leiden(adata, resolution=LEIDEN_RES, key_added="leiden_coarse",
                 flavor="igraph", n_iterations=2, directed=False)
    adata.obs["leiden_coarse"] = pd.Categorical(adata.obs["leiden_coarse"])
    n_groups = adata.obs["leiden_coarse"].nunique()
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes, {n_groups} Leiden clusters")
    print(f"  Cluster sizes: {adata.obs['leiden_coarse'].value_counts().sort_index().to_dict()}")

    # τ
    tau     = compute_tau(adata, "leiden_coarse")
    tau_all = tau.dropna().values
    print(f"  Genes with valid τ: {len(tau_all)}, median={np.median(tau_all):.3f}, "
          f"IQR=[{np.percentile(tau_all,25):.3f}, {np.percentile(tau_all,75):.3f}]")

    rankings = load_rankings(name)

    # Dot plots for each cutoff
    for k in CUTOFFS:
        plot_dotplot(tau, tau_all, rankings, k,
                     out_dir / f"real_dataset_dotplot_top{k}.svg",
                     label)

    # Top-10 genes per method
    plot_top10_genes(tau, rankings,
                     out_dir / "top10_genes_per_method.svg",
                     label, n_groups)

    # Print summary table
    print(f"\n  Top-50 summary (Leiden {n_groups} clusters):")
    rows = []
    for m in METHOD_ORDER:
        gene_tau = tau[[g for g in rankings[m] if g in tau.index and not np.isnan(tau[g])]][:50]
        rows.append({"method": m, "mean": gene_tau.mean(), "sd": gene_tau.std(), "n": len(gene_tau)})
    summary = pd.DataFrame(rows).sort_values("mean", ascending=False)
    for _, r in summary.iterrows():
        print(f"    {r['method']:<12} {r['mean']:.4f} ± {r['sd']:.4f}  (n={r['n']})")

print("\nAll done.")
