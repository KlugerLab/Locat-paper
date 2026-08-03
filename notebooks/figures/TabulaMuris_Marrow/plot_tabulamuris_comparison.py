"""
Generate real-dataset-style comparison plots for Tabula Muris Senis bone
marrow (10x): dot plots, context UMAP, and per-method top-10 gene UMAPs.

Usage:
    python plot_tabulamuris_comparison.py
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

HERE      = Path(__file__).parent
DATA_H5AD = HERE / "data" / "tabula_muris_marrow_proc.h5ad"
SCORES    = HERE / "scores"
OUT       = HERE / "real_dataset"
OUT.mkdir(exist_ok=True)

# Census's Tabula Muris Senis ontology labels are fine-grained differentiation
# stages (18 types forming overlapping continua, e.g. "granulocytopoietic cell"
# -> "granulocyte"), which mechanically depresses tau relative to the plan's
# assumed ~9 broad cell types. We use the coarser "cell_type_coarse" column
# (added during preprocessing) to match the plan's expected granularity and
# the style of the other real-dataset comparisons (DermalC, Kang, PBMC3k).
CELLTYPE_COL = "cell_type"
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
PT_SIZE = 2.0
ALPHA   = 0.4

def log(msg):
    print(msg, flush=True)

def compute_tau(adata, celltype_col=CELLTYPE_COL, pct_thresh=PCT_THRESH):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    cts = adata.obs[celltype_col].cat.categories.tolist()
    mean = np.array([X[adata.obs[celltype_col] == ct].mean(axis=0) for ct in cts])
    pct  = (X > 0).mean(axis=0)
    rs   = mean.sum(axis=0)
    mask = (rs > 0) & (pct >= pct_thresh)
    tau  = np.where(mask, mean.max(axis=0) / np.where(rs > 0, rs, 1.0), np.nan)
    return pd.Series(tau, index=adata.var_names)

def load_rankings():
    rankings = {}
    x = np.load(SCORES / "locat_scores.npz", allow_pickle=True)
    rankings["Locat"] = pd.Series(x["pval"], index=x["gene_names"]).sort_values().index.tolist()
    x = np.load(SCORES / "gspa_scores.npz", allow_pickle=True)
    rankings["GSPA"] = pd.Series(x["gene_localization"], index=x["var_names"]).sort_values(ascending=False).index.tolist()
    x = np.load(SCORES / "lmd_scores.npz", allow_pickle=True)
    rankings["LMD"] = pd.Series(x["lmd_score"], index=x["var_names"]).sort_values().index.tolist()
    x = np.load(SCORES / "hotspot_scores.npz", allow_pickle=True)
    rankings["Hotspot"] = pd.Series(x["fdr"], index=x["var_names"]).sort_values().index.tolist()
    x = np.load(SCORES / "haystack_scores.npz", allow_pickle=True)
    rankings["Haystack"] = pd.Series(x["logpval"], index=x["var_names"]).sort_values().index.tolist()
    x = np.load(SCORES / "scanpy_scores.npz", allow_pickle=True)
    rankings["Scanpy"] = pd.Series(x["pval_min"], index=x["var_names"]).sort_values().index.tolist()
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
    log(f"  Saved {out_path.name}")

# ── Main ──────────────────────────────────────────────────────────────────────
log("Loading processed adata...")
adata = sc.read_h5ad(DATA_H5AD)
adata.obs[CELLTYPE_COL] = pd.Categorical(adata.obs[CELLTYPE_COL].astype(str))
n_cts = adata.obs[CELLTYPE_COL].nunique()
log(f"  {adata.n_obs} cells x {adata.n_vars} genes, {n_cts} cell types")
log(f"  Cell type counts:\n{adata.obs[CELLTYPE_COL].value_counts().to_string()}")

tau = compute_tau(adata)
tau_all = tau.dropna().values
log(f"  Genes with valid tau: {len(tau_all)}, median={np.median(tau_all):.4f}, "
    f"IQR=[{np.percentile(tau_all,25):.4f}, {np.percentile(tau_all,75):.4f}]")

rankings = load_rankings()

# ── 1. Dot plots ──────────────────────────────────────────────────────────────
for k in CUTOFFS:
    plot_dotplot(tau, tau_all, rankings, k,
                 OUT / f"real_dataset_dotplot_top{k}.svg",
                 "Tabula Muris Marrow (10x)")

# ── 2. Context UMAP ───────────────────────────────────────────────────────────
umap = adata.obsm["X_umap"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: cell type (categorical, tab20)
ax = axes[0]
cts = adata.obs[CELLTYPE_COL].cat.categories.tolist()
cmap_cat = plt.cm.get_cmap("tab20", max(len(cts), 1))
ct_counts = adata.obs[CELLTYPE_COL].value_counts()
for i, ct in enumerate(cts):
    mask = (adata.obs[CELLTYPE_COL] == ct).values
    ax.scatter(umap[mask, 0], umap[mask, 1], s=PT_SIZE, alpha=ALPHA,
               color=cmap_cat(i), label=f"{ct} (n={ct_counts[ct]})", rasterized=True)
ax.legend(markerscale=5, fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5),
          framealpha=0.7, ncol=1)
ax.set_title("Tabula Muris Marrow — Cell type", fontsize=10)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_xticks([]); ax.set_yticks([])

# Right: continuous marker (n_genes_by_counts if present, else total_counts)
ax = axes[1]
cont_col = None
for cand in ["n_genes_by_counts", "total_counts", "n_genes"]:
    if cand in adata.obs.columns:
        cont_col = cand
        break
if cont_col is not None:
    vals = pd.to_numeric(adata.obs[cont_col], errors="coerce").values
    sc_im = ax.scatter(umap[:, 0], umap[:, 1], s=PT_SIZE, alpha=ALPHA,
                       c=vals, cmap="viridis", rasterized=True)
    plt.colorbar(sc_im, ax=ax, label=cont_col, shrink=0.8)
    ax.set_title(f"Tabula Muris Marrow — {cont_col}", fontsize=10)
else:
    dev_col = "development_stage" if "development_stage" in adata.obs.columns else None
    if dev_col:
        stages = adata.obs[dev_col].astype(str)
        stage_cats = stages.unique().tolist()
        cmap2 = plt.cm.get_cmap("tab10", max(len(stage_cats), 1))
        for i, s in enumerate(stage_cats):
            mask = (stages == s).values
            ax.scatter(umap[mask, 0], umap[mask, 1], s=PT_SIZE, alpha=ALPHA,
                       color=cmap2(i), label=s, rasterized=True)
        ax.legend(markerscale=5, fontsize=7)
        ax.set_title("Tabula Muris Marrow — Development stage", fontsize=10)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
out = OUT / "umap_context.svg"
plt.savefig(out, bbox_inches="tight", dpi=150)
plt.close()
log(f"Saved {out.name}")

# ── 3. Top-10 gene UMAPs per method ───────────────────────────────────────────
X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
genes = adata.var_names.tolist()
g2i = {g: i for i, g in enumerate(genes)}

for m in METHOD_ORDER:
    top10_genes = [g for g in rankings[m] if g in g2i][:10]
    if not top10_genes:
        log(f"  Skipping {m}: no genes")
        continue

    fig, axes_grid = plt.subplots(2, 5, figsize=(18, 7))
    axes_grid = axes_grid.flatten()

    for ax, gene in zip(axes_grid, top10_genes):
        expr = X[:, g2i[gene]].astype(np.float32)
        zero = expr == 0
        vmax_gene = np.percentile(expr[~zero], 95) if (~zero).any() else 1.0
        ax.scatter(umap[zero, 0], umap[zero, 1], s=0.5, alpha=0.15,
                   color="#dddddd", rasterized=True)
        sc_im = ax.scatter(umap[~zero, 0], umap[~zero, 1],
                           c=expr[~zero], s=PT_SIZE, alpha=0.6,
                           cmap="YlOrRd", vmin=0, vmax=vmax_gene, rasterized=True)
        plt.colorbar(sc_im, ax=ax, shrink=0.8, pad=0.02)
        pct_expr = (~zero).mean() * 100
        ax.set_title(f"{gene}\n({pct_expr:.0f}% cells)", fontsize=20)
        ax.set_xticks([]); ax.set_yticks([])

    # hide any unused panels
    for ax in axes_grid[len(top10_genes):]:
        ax.axis("off")

    fig.suptitle(f"Tabula Muris Marrow — {m} top-10 genes (individual expression on UMAP)",
                 fontsize=22, color=COLORS[m], y=1.02)
    plt.tight_layout()
    out = OUT / f"top10_umap_{m.lower()}.svg"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log(f"  Saved {out.name}")

# ── Summary + verification ────────────────────────────────────────────────────
log("\n=== Verification ===")
log(f"Cell types (>=100 cells): "
    f"{(ct_counts >= 100).sum()} / {len(ct_counts)}")
log(f"All-genes median tau: {np.median(tau_all):.4f}")

log("\nTop-50 mean tau per method:")
for m in METHOD_ORDER:
    gene_tau = tau[[g for g in rankings[m] if g in tau.index and not np.isnan(tau[g])]][:50]
    log(f"  {m:<10} {gene_tau.mean():.4f} +/- {gene_tau.std():.4f}  (n={len(gene_tau)})")

MARKERS = {
    "B cell":            ["Cd19", "Ms4a1", "Cd79a", "Cd79b", "Ebf1"],
    "T/NK":              ["Cd3e", "Cd8a", "Cd4", "Nkg7", "Gzma"],
    "Monocyte/Macro":    ["Lyz2", "Csf1r", "Cd68", "S100a8"],
    "Erythroid":         ["Hba-a1", "Hbb-bt", "Gypa"],
}
locat_top50 = set(rankings["Locat"][:50])
log("\nMarker gene overlap in Locat top-50:")
n_hit_total = 0
for group, genes_ in MARKERS.items():
    hits = [g for g in genes_ if g in locat_top50]
    n_hit_total += len(hits)
    log(f"  {group}: {hits}")
log(f"Total marker genes found in Locat top-50: {n_hit_total}")

log("\nAll done.")
