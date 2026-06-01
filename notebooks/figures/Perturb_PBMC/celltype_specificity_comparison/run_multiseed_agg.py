"""Aggregate multi-seed results and generate bar chart with 95% CI."""
import os, sys
os.chdir("/banach2/wes/Locat-paper-repro-private/notebooks/figures/Perturb_PBMC/celltype_specificity_comparison")
sys.path.insert(0, "/banach2/wes/locat-0.1")
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

DATA_PATH = Path("/banach2/wes/Locat-paper-repro-private/data/pbmc3k_9543_lognorm.h5ad")
OUT_DIR   = Path(".")
SCORES_DIR = Path("scores")
CUTOFFS = [100, 200, 400]
THRESH = 0.5

# τ computation (Option B: ≥5% expressing genes)
adata_ms = sc.read_h5ad(DATA_PATH)
X_ms = adata_ms.X.toarray() if sp.issparse(adata_ms.X) else np.asarray(adata_ms.X)
pct_expr_ms = (X_ms > 0).mean(axis=0)
cell_types_ms = adata_ms.obs["louvain"].cat.categories.tolist()
mean_ms = np.array([X_ms[adata_ms.obs["louvain"] == ct].mean(axis=0) for ct in cell_types_ms])
rs_ms   = mean_ms.sum(axis=0)
expr_mask_ms = (rs_ms > 0) & (pct_expr_ms >= 0.05)
tau_b_arr_ms = np.where(expr_mask_ms, mean_ms.max(axis=0) / np.where(rs_ms > 0, rs_ms, 1), np.nan)
specificity_ms = pd.Series(tau_b_arr_ms, index=adata_ms.var_names)

def load_seed(seed_dir):
    d = {}
    f = seed_dir / "locat_scores.npz"
    if f.exists():
        x = np.load(f, allow_pickle=True)
        d["Locat"] = pd.Series(x["pval"], index=x["gene_names"]).sort_values().index.tolist()
    f = seed_dir / "gspa_scores.npz"
    if f.exists():
        x = np.load(f, allow_pickle=True)
        d["GSPA"] = pd.Series(x["gene_localization"], index=x["var_names"]).sort_values(ascending=False).index.tolist()
    f = seed_dir / "lmd_scores.npz"
    if f.exists():
        x = np.load(f, allow_pickle=True)
        d["LMD"] = pd.Series(x["lmd_score"], index=x["var_names"]).sort_values(ascending=True).index.tolist()
    f = seed_dir / "hotspot_scores.npz"
    if f.exists():
        x = np.load(f, allow_pickle=True)
        d["Hotspot"] = pd.Series(x["fdr"], index=x["var_names"]).sort_values(ascending=True).index.tolist()
    f = seed_dir / "haystack_scores.npz"
    if f.exists():
        x = np.load(f, allow_pickle=True)
        d["Haystack"] = pd.Series(x["logpval"], index=x["var_names"]).sort_values(ascending=True).index.tolist()
    return d

def tau_at_k(ranked, spec, k):
    valid = [g for g in ranked if g in spec.index and not np.isnan(spec[g])][:k]
    return np.mean([spec[g] for g in valid]) if valid else np.nan

seed_dirs = sorted(SCORES_DIR.glob("seed_*"))
print(f"Found {len(seed_dirs)} seed directories: {[s.name for s in seed_dirs]}")

records = []
for sd in seed_dirs:
    rankings = load_seed(sd)
    for method, ranked in rankings.items():
        for k in CUTOFFS:
            records.append({"seed": sd.name, "method": method, "k": k,
                            "mean_tau": tau_at_k(ranked, specificity_ms, k)})

df_ms = pd.DataFrame(records)
print(f"Loaded {len(df_ms)} records, {df_ms['seed'].nunique()} seeds, {df_ms['method'].nunique()} methods")

print("\n=== Multi-seed summary (mean ± sem) ===")
for k in CUTOFFS:
    sub = df_ms[df_ms["k"] == k]
    agg = sub.groupby("method")["mean_tau"].agg(["mean", "sem"]).sort_values("mean", ascending=False)
    agg.columns = ["Mean τ", "SEM"]
    print(f"\nTop-{k}:")
    print(agg.round(4).to_string())

colors = {
    "Locat":    "#e6194b",
    "GSPA":     "#3cb44b",
    "LMD":      "#4363d8",
    "Hotspot":  "#f58231",
    "Haystack": "#911eb4",
}

np.random.seed(42)
fig, axes = plt.subplots(1, len(CUTOFFS), figsize=(5 * len(CUTOFFS), 5), sharey=False)

for ax, k in zip(axes, CUTOFFS):
    sub = df_ms[df_ms["k"] == k]
    agg = sub.groupby("method")["mean_tau"].agg(["mean", "sem"]).reset_index()
    agg = agg.sort_values("mean", ascending=False)

    bar_colors = [colors.get(m, "gray") for m in agg["method"]]
    bars = ax.bar(agg["method"], agg["mean"], color=bar_colors,
                  edgecolor="white", linewidth=0.5, zorder=2)
    ax.errorbar(range(len(agg)), agg["mean"], yerr=agg["sem"] * 1.96,
                fmt="none", color="black", capsize=4, linewidth=1.5, zorder=3)

    for i, method in enumerate(agg["method"]):
        pts = sub[sub["method"] == method]["mean_tau"].values
        ax.scatter(np.full(len(pts), i) + np.random.uniform(-0.15, 0.15, len(pts)),
                   pts, color="black", s=18, alpha=0.5, zorder=4)

    ax.set_title(f"Top-{k} genes")
    ax.set_ylabel("Mean τ (Option B: ≥5% expressed)" if ax == axes[0] else "")
    ax.tick_params(axis="x", rotation=30)
    for bar, row in zip(bars, agg.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, row.mean + row.sem * 1.96 + 0.005,
                f"{row.mean:.3f}", ha="center", va="bottom", fontsize=8)
    sns.despine(ax=ax)

fig.suptitle(
    f"Multi-seed comparison (n={df_ms['seed'].nunique()} seeds)\n"
    "Mean τ ± 95% CI, Option B (≥5% expressing genes)",
    y=1.02, fontsize=12
)
plt.tight_layout()
out = OUT_DIR / "multiseed_mean_tau.svg"
plt.savefig(out, bbox_inches="tight")
print(f"\nSaved: {out}")
