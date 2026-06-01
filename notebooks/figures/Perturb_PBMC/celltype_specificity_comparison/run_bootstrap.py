"""
Bootstrap confidence intervals for cell-type specificity comparison.

Resamples cells with replacement, recomputes tau on each bootstrap sample
using fixed gene rankings from a single method run (seed_0).

Usage:
    python run_bootstrap.py [--n_boot 5000] [--seed 0]
"""
import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

parser = argparse.ArgumentParser()
parser.add_argument("--n_boot", type=int, default=5000)
parser.add_argument("--seed",   type=int, default=0)
args = parser.parse_args()

HERE      = Path(__file__).parent
DATA_PATH = Path("/banach2/wes/Locat-paper-repro-private/data/pbmc3k_9543_lognorm.h5ad")
SCORES_DIR = HERE / "scores" / "seed_0"
CUTOFFS   = [100, 200, 400]

rng = np.random.default_rng(args.seed)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading adata...", flush=True)
adata = sc.read_h5ad(DATA_PATH)
X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
louvain = adata.obs["louvain"].values
cell_types = adata.obs["louvain"].cat.categories.tolist()
n_cells, n_genes = X.shape
print(f"  {n_cells} cells × {n_genes} genes, {len(cell_types)} cell types", flush=True)

# ── Load gene rankings (fixed, from seed_0) ───────────────────────────────────
print("Loading rankings...", flush=True)
rankings = {}

x = np.load(SCORES_DIR / "locat_scores.npz", allow_pickle=True)
rankings["LOCAT"] = pd.Series(x["pval"], index=x["gene_names"]).sort_values().index.tolist()

x = np.load(SCORES_DIR / "gspa_scores.npz", allow_pickle=True)
rankings["GSPA"] = pd.Series(x["gene_localization"], index=x["var_names"]).sort_values(ascending=False).index.tolist()

x = np.load(SCORES_DIR / "lmd_scores.npz", allow_pickle=True)
rankings["LMD"] = pd.Series(x["lmd_score"], index=x["var_names"]).sort_values(ascending=True).index.tolist()

x = np.load(SCORES_DIR / "hotspot_scores.npz", allow_pickle=True)
rankings["Hotspot"] = pd.Series(x["fdr"], index=x["var_names"]).sort_values(ascending=True).index.tolist()

x = np.load(SCORES_DIR / "haystack_scores.npz", allow_pickle=True)
rankings["Haystack"] = pd.Series(x["logpval"], index=x["var_names"]).sort_values(ascending=True).index.tolist()

# Pre-compute top-k gene indices for each method/cutoff (fixed across bootstraps)
# Map gene names -> column indices in X
gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
top_k_idx = {}  # (method, k) -> array of gene column indices
for method, ranked in rankings.items():
    for k in CUTOFFS:
        # Take top-k genes that exist in adata
        valid = [g for g in ranked if g in gene_to_idx][:k]
        top_k_idx[(method, k)] = np.array([gene_to_idx[g] for g in valid])

# Pre-compute per-cell-type boolean masks
ct_masks = [louvain == ct for ct in cell_types]

# ── Bootstrap ─────────────────────────────────────────────────────────────────
print(f"Running {args.n_boot} bootstrap iterations...", flush=True)
t0 = time.time()

# results[method][k] = list of mean_tau values across bootstraps
results = {m: {k: [] for k in CUTOFFS} for m in rankings}

for b in range(args.n_boot):
    idx = rng.integers(0, n_cells, size=n_cells)  # resample with replacement
    X_boot = X[idx]
    louvain_boot = louvain[idx]

    # Recompute per-cell-type means on bootstrap sample
    ct_masks_boot = [louvain_boot == ct for ct in cell_types]
    # mean_boot: shape (n_cell_types, n_genes)
    mean_boot = np.array([
        X_boot[m].mean(axis=0) if m.any() else np.zeros(n_genes)
        for m in ct_masks_boot
    ])  # (K_ct, n_genes)

    # Option B filter: ≥5% expressing in bootstrap sample
    pct_expr_boot = (X_boot > 0).mean(axis=0)  # (n_genes,)
    rs_boot = mean_boot.sum(axis=0)             # (n_genes,)
    expr_mask_boot = (rs_boot > 0) & (pct_expr_boot >= 0.05)

    # tau: max_ct / sum_ct where expressed, else nan
    tau_boot = np.where(
        expr_mask_boot,
        mean_boot.max(axis=0) / np.where(rs_boot > 0, rs_boot, 1.0),
        np.nan,
    )  # (n_genes,)

    # Evaluate mean tau at each cutoff for each method
    for method in rankings:
        for k in CUTOFFS:
            gene_idx = top_k_idx[(method, k)]
            vals = tau_boot[gene_idx]
            valid = vals[~np.isnan(vals)]
            results[method][k].append(np.mean(valid) if len(valid) > 0 else np.nan)

    if (b + 1) % 500 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (b + 1) * (args.n_boot - b - 1)
        print(f"  {b+1}/{args.n_boot}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

elapsed = time.time() - t0
print(f"Bootstrap done in {elapsed:.1f}s ({elapsed/args.n_boot*1000:.1f}ms/iter)", flush=True)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Bootstrap summary (mean ± 95% CI) ===")
records = []
for method in rankings:
    for k in CUTOFFS:
        vals = np.array(results[method][k])
        vals = vals[~np.isnan(vals)]
        mean  = vals.mean()
        ci_lo = np.percentile(vals, 2.5)
        ci_hi = np.percentile(vals, 97.5)
        records.append({"method": method, "k": k, "mean": mean,
                        "ci_lo": ci_lo, "ci_hi": ci_hi, "vals": vals})

df = pd.DataFrame([{k: v for k, v in r.items() if k != "vals"} for r in records])
for k in CUTOFFS:
    sub = df[df["k"] == k].sort_values("mean", ascending=False)
    print(f"\nTop-{k}:")
    for _, row in sub.iterrows():
        print(f"  {row['method']:<10}  {row['mean']:.4f}  [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]")

# Save results
np.save(HERE / "bootstrap_results.npy", results)
df.to_csv(HERE / "bootstrap_summary.csv", index=False)
print(f"\nSaved bootstrap_results.npy and bootstrap_summary.csv", flush=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
colors = {
    "LOCAT":    "#e6194b",
    "GSPA":     "#3cb44b",
    "LMD":      "#4363d8",
    "Hotspot":  "#f58231",
    "Haystack": "#911eb4",
}

fig, axes = plt.subplots(1, len(CUTOFFS), figsize=(5 * len(CUTOFFS), 5), sharey=False)

for ax, k in zip(axes, CUTOFFS):
    sub = df[df["k"] == k].sort_values("mean", ascending=False).reset_index(drop=True)
    bar_colors = [colors.get(m, "gray") for m in sub["method"]]

    bars = ax.bar(sub["method"], sub["mean"], color=bar_colors,
                  edgecolor="white", linewidth=0.5, zorder=2)
    # 95% CI as error bars
    yerr_lo = sub["mean"] - sub["ci_lo"]
    yerr_hi = sub["ci_hi"] - sub["mean"]
    ax.errorbar(range(len(sub)), sub["mean"],
                yerr=[yerr_lo, yerr_hi],
                fmt="none", color="black", capsize=4, linewidth=1.5, zorder=3)

    ax.set_title(f"Top-{k} genes")
    ax.set_ylabel("Mean τ (Option B: ≥5% expressed)" if ax == axes[0] else "")
    ax.tick_params(axis="x", rotation=30)
    for bar, row in zip(bars, sub.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                row.ci_hi + 0.003,
                f"{row.mean:.3f}", ha="center", va="bottom", fontsize=8)
    sns.despine(ax=ax)

fig.suptitle(
    f"Bootstrap comparison (n={args.n_boot} resamples)\n"
    "Mean τ ± 95% CI, Option B (≥5% expressing genes)",
    y=1.02, fontsize=12
)
plt.tight_layout()
out = HERE / "bootstrap_mean_tau.svg"
plt.savefig(out, bbox_inches="tight")
print(f"Saved: {out}", flush=True)
