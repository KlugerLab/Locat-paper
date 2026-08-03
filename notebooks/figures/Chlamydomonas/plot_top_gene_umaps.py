"""
For each Chlamydomonas condition, generate:
  1. umap_context.svg    — UMAP colored by Leiden cluster and scPrisma cycle angle
  2. top10_umap_<cond>.svg — one UMAP panel per method, colored by mean z-scored
     expression of that method's top-10 genes (shows spatial concentration/depletion)

Usage:
    python plot_top_gene_umaps.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scanpy as sc

HERE   = Path(__file__).parent
DATA   = Path("/data/wes/Locat-paper-repro-private/notebooks/figures/Chlamydomonas")
SCORES = DATA / "locat_run" / "scores"
COMP   = DATA / "comparison_results"

LEIDEN_RES   = 0.3
METHOD_ORDER = ["Locat", "GSPA", "LMD", "Haystack", "Hotspot", "Scanpy"]
COLORS = {
    "Locat":    "#e6194b", "GSPA": "#3cb44b", "LMD": "#4363d8",
    "Hotspot":  "#f58231", "Haystack": "#911eb4", "Scanpy": "#f032e6",
}
SAMPLES = [
    dict(name="fe_neg", label="Fe−"),
    dict(name="fe_pos", label="Fe+"),
]
PT_SIZE = 1.0   # scatter point size (small — many cells)
ALPHA   = 0.3

def load_rankings(name):
    rankings = {}
    x = np.load(SCORES / name / "locat_scores.npz", allow_pickle=True)
    rankings["Locat"]    = pd.Series(x["pval"],              index=x["gene_names"]).sort_values().index.tolist()
    x = np.load(COMP / name / "gspa_scores.npz",    allow_pickle=True)
    rankings["GSPA"]     = pd.Series(x["gene_localization"], index=x["var_names"]).sort_values(ascending=False).index.tolist()
    x = np.load(COMP / name / "lmd_scores.npz",     allow_pickle=True)
    rankings["LMD"]      = pd.Series(x["lmd_score"],         index=x["var_names"]).sort_values().index.tolist()
    x = np.load(COMP / name / "hotspot_scores.npz", allow_pickle=True)
    rankings["Hotspot"]  = pd.Series(x["fdr"],               index=x["var_names"]).sort_values().index.tolist()
    x = np.load(COMP / name / "haystack_scores.npz",allow_pickle=True)
    rankings["Haystack"] = pd.Series(x["logpval"],           index=x["var_names"]).sort_values().index.tolist()
    x = np.load(COMP / name / "scanpy_scores.npz",  allow_pickle=True)
    rankings["Scanpy"]   = pd.Series(x["pval_min"],          index=x["var_names"]).sort_values().index.tolist()
    return rankings


for ds in SAMPLES:
    name  = ds["name"]
    label = ds["label"]
    print(f"\n=== {label} ===")

    out_dir = HERE / "comparison_results" / name / "real_dataset"
    out_dir.mkdir(exist_ok=True)

    adata = sc.read_h5ad(SCORES / name / "adata_phases.h5ad")
    if "connectivities" not in adata.obsp:
        sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)
    sc.tl.leiden(adata, resolution=LEIDEN_RES, key_added="leiden_coarse",
                 flavor="igraph", n_iterations=2, directed=False)

    umap = adata.obsm["X_umap"]
    X    = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    genes = adata.var_names.tolist()
    g2i  = {g: i for i, g in enumerate(genes)}

    rankings = load_rankings(name)
    n_clusters = adata.obs["leiden_coarse"].nunique()

    # ── 1. Context panel: Leiden clusters + scPrisma angle ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Leiden clusters
    ax = axes[0]
    cluster_ids = adata.obs["leiden_coarse"].astype(int).values
    cmap_cat = plt.cm.get_cmap("tab10", n_clusters)
    for cl in range(n_clusters):
        mask = cluster_ids == cl
        ax.scatter(umap[mask, 0], umap[mask, 1], s=PT_SIZE, alpha=ALPHA,
                   color=cmap_cat(cl), label=f"C{cl} (n={mask.sum()})", rasterized=True)
    ax.legend(markerscale=5, fontsize=28, loc="upper right", framealpha=0.7)
    ax.set_title(f"{label} — Leiden clusters (res={LEIDEN_RES})", fontsize=40)
    ax.set_xlabel("UMAP 1", fontsize=40); ax.set_ylabel("UMAP 2", fontsize=40)
    ax.set_xticks([]); ax.set_yticks([])

    # scPrisma cycle angle (if present)
    ax = axes[1]
    if "umap_angle" in adata.obs.columns:
        angle = adata.obs["umap_angle"].values
        sc_im = ax.scatter(umap[:, 0], umap[:, 1], s=PT_SIZE, alpha=ALPHA,
                           c=angle, cmap="hsv", rasterized=True)
        cb = plt.colorbar(sc_im, ax=ax, shrink=0.8)
        cb.set_label("scPrisma cycle angle (rad)", fontsize=28)
        cb.ax.tick_params(labelsize=24)
        ax.set_title(f"{label} — scPrisma cycle angle", fontsize=40)
    elif "phase" in adata.obs.columns:
        phases = adata.obs["phase"].values
        phase_colors = {"G1": "#4363d8", "S": "#3cb44b", "G2M": "#e6194b"}
        for ph, col in phase_colors.items():
            mask = phases == ph
            ax.scatter(umap[mask, 0], umap[mask, 1], s=PT_SIZE, alpha=ALPHA,
                       color=col, label=ph, rasterized=True)
        ax.legend(markerscale=5, fontsize=32)
        ax.set_title(f"{label} — Cell cycle phase", fontsize=40)
    ax.set_xlabel("UMAP 1", fontsize=40); ax.set_ylabel("UMAP 2", fontsize=40)
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out = out_dir / f"umap_context_{name}.svg"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {out.name}")

    # ── 2. Top-10 gene UMAPs per method — one panel per gene ────────────────
    # One SVG per method: 2 rows × 5 cols, each panel = one gene's expression
    for m in METHOD_ORDER:
        top10_genes = [g for g in rankings[m] if g in g2i][:10]

        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        axes = axes.flatten()

        for ax, gene in zip(axes, top10_genes):
            expr = X[:, g2i[gene]].astype(np.float32)
            vmax_gene = np.percentile(expr[expr > 0], 95) if (expr > 0).any() else 1.0
            # grey background (zero-expressing cells), then color expressing cells on top
            zero = expr == 0
            ax.scatter(umap[zero, 0], umap[zero, 1], s=PT_SIZE * 0.5, alpha=0.15,
                       color="#dddddd", rasterized=True)
            sc_im = ax.scatter(umap[~zero, 0], umap[~zero, 1],
                               c=expr[~zero], s=PT_SIZE, alpha=0.6,
                               cmap="YlOrRd", vmin=0, vmax=vmax_gene, rasterized=True)
            cb = plt.colorbar(sc_im, ax=ax, shrink=0.8, pad=0.02)
            cb.ax.tick_params(labelsize=14)
            short = gene.replace(".v5.5", "")
            pct_expr = (~zero).mean() * 100
            ax.set_title(f"{short}\n({pct_expr:.0f}% cells)", fontsize=20)
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(f"Chlamydomonas {label} — {m} top-10 genes (individual expression on UMAP)\n"
                     f"Leiden res={LEIDEN_RES}, {n_clusters} clusters",
                     fontsize=22, color=COLORS[m], y=1.01)
        plt.tight_layout()
        out = out_dir / f"top10_umap_{name}_{m.lower()}.svg"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  Saved {out.name}")

    # Print top-10 genes per method for reference
    print(f"\n  Top-10 genes per method ({label}):")
    for m in METHOD_ORDER:
        top10_genes = [g for g in rankings[m] if g in g2i][:10]
        short = [g.replace(".v5.5", "") for g in top10_genes]
        print(f"    {m:<12}: {', '.join(short)}")

print("\nAll done.")
