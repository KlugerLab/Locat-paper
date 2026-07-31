"""
Run all 5 methods on DermalC data, then bootstrap tau CIs.

Usage:
    python run_dermalc_comparison.py [--n_boot 5000] [--gpu 0]
"""
import argparse, os, sys, subprocess, time
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
parser.add_argument("--gpu",    type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

HERE      = Path(__file__).parent
LOCAT_SRC = Path("LOCAT01_PATH")
DATA_PATH = Path(__file__).resolve().parents[4] / "data/E145_dermal_erez_2026/dc_adata_proc.h5ad"
SCORES    = HERE / "scores"
SCORES.mkdir(exist_ok=True)

GSPA_PYTHON = "GSPA_PYTHON"
LMD_PYTHON  = "LMD_PYTHON"
LMD_SCRIPT  = Path(__file__).resolve().parents[4] / "notebooks/figures/Perturb_PBMC/celltype_specificity_comparison/run_lmd_seeded.py"
GSPA_SCRIPT = Path(__file__).resolve().parents[4] / "notebooks/figures/Perturb_PBMC/celltype_specificity_comparison/run_gspa_seeded.py"

if str(LOCAT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAT_SRC))

CUTOFFS = [50, 100, 200]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Load data ─────────────────────────────────────────────────────────────────
log("Loading DermalC adata...")
adata = sc.read_h5ad(DATA_PATH)
log(f"  {adata.n_obs} cells × {adata.n_vars} genes | cell types: {adata.obs['celltype'].value_counts().to_dict()}")

# ── 1. LOCAT ──────────────────────────────────────────────────────────────────
log("Running LOCAT...")
t0 = time.time()
from locat.locat import LOCAT
embedding = adata.obsm["X_pca"].astype(np.float64)[:, :8]
model = LOCAT(
    adata=adata,
    cell_embedding=embedding,
    k=20,
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
np.savez(SCORES / "locat_scores.npz",
         gene_names=locat_df.index.values, pval=locat_df["pval"].values)
log(f"  LOCAT done in {(time.time()-t0)/60:.1f}m")

# ── 2. GSPA ───────────────────────────────────────────────────────────────────
log("Running GSPA...")
t0 = time.time()
subprocess.run(
    [GSPA_PYTHON, str(GSPA_SCRIPT),
     "--data_path", str(DATA_PATH),
     "--out_path",  str(SCORES / "gspa_scores.npz"),
     "--seed",      "0",
     "--gpu",       args.gpu],
    check=True,
)
log(f"  GSPA done in {(time.time()-t0)/60:.1f}m")

# ── 3. LMD ────────────────────────────────────────────────────────────────────
log("Running LMD...")
t0 = time.time()
subprocess.run(
    [LMD_PYTHON, str(LMD_SCRIPT),
     "--data_path", str(DATA_PATH),
     "--out_path",  str(SCORES / "lmd_scores.npz")],
    env={**os.environ,
         "R_HOME": "LMD_R_HOME",
         "R_DEFAULT_PACKAGES": "base,utils,stats,graphics,grDevices,methods",
         "CUDA_VISIBLE_DEVICES": ""},
    check=True,
)
log(f"  LMD done in {(time.time()-t0)/60:.1f}m")

# ── 4. Hotspot ────────────────────────────────────────────────────────────────
log("Running Hotspot...")
t0 = time.time()
import hotspot as hs_pkg
hs = hs_pkg.Hotspot(adata, layer_key=None, model="normal",
                    latent_obsm_key="X_pca", umi_counts_obs_key=None)
hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
hs.compute_autocorrelations()
hotspot_scores = hs.results["FDR"]
np.savez(SCORES / "hotspot_scores.npz",
         var_names=hotspot_scores.index.values, fdr=hotspot_scores.values)
log(f"  Hotspot done in {(time.time()-t0)/60:.1f}m")

# ── 5. Haystack ───────────────────────────────────────────────────────────────
log("Running Haystack...")
t0 = time.time()
import singleCellHaystack as sch
res = sch.haystack(adata, coord="pca")
haystack_scores = res.result.set_index("gene")["logpval"]
np.savez(SCORES / "haystack_scores.npz",
         var_names=haystack_scores.index.values, logpval=haystack_scores.values)
log(f"  Haystack done in {(time.time()-t0)/60:.1f}m")

# ── Load rankings ─────────────────────────────────────────────────────────────
log("Loading rankings for bootstrap...")
gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

rankings = {}
x = np.load(SCORES / "locat_scores.npz", allow_pickle=True)
rankings["Locat"] = pd.Series(x["pval"], index=x["gene_names"]).sort_values().index.tolist()
x = np.load(SCORES / "gspa_scores.npz", allow_pickle=True)
rankings["GSPA"] = pd.Series(x["gene_localization"], index=x["var_names"]).sort_values(ascending=False).index.tolist()
x = np.load(SCORES / "lmd_scores.npz", allow_pickle=True)
rankings["LMD"] = pd.Series(x["lmd_score"], index=x["var_names"]).sort_values(ascending=True).index.tolist()
x = np.load(SCORES / "hotspot_scores.npz", allow_pickle=True)
rankings["Hotspot"] = pd.Series(x["fdr"], index=x["var_names"]).sort_values(ascending=True).index.tolist()
x = np.load(SCORES / "haystack_scores.npz", allow_pickle=True)
rankings["Haystack"] = pd.Series(x["logpval"], index=x["var_names"]).sort_values(ascending=True).index.tolist()

top_k_idx = {}
for method, ranked in rankings.items():
    for k in CUTOFFS:
        valid = [g for g in ranked if g in gene_to_idx][:k]
        top_k_idx[(method, k)] = np.array([gene_to_idx[g] for g in valid])

# ── Bootstrap ─────────────────────────────────────────────────────────────────
log(f"Running {args.n_boot} bootstrap iterations...")
X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
celltype = adata.obs["celltype"].values
cell_types = adata.obs["celltype"].cat.categories.tolist()
n_cells, n_genes = X.shape

rng = np.random.default_rng(42)
t0 = time.time()
boot_results = {m: {k: [] for k in CUTOFFS} for m in rankings}

for b in range(args.n_boot):
    idx = rng.integers(0, n_cells, size=n_cells)
    X_boot = X[idx]
    ct_boot = celltype[idx]

    mean_boot = np.array([
        X_boot[ct_boot == ct].mean(axis=0) if (ct_boot == ct).any() else np.zeros(n_genes)
        for ct in cell_types
    ])
    pct_expr = (X_boot > 0).mean(axis=0)
    rs = mean_boot.sum(axis=0)
    mask = (rs > 0) & (pct_expr >= 0.05)
    tau = np.where(mask, mean_boot.max(axis=0) / np.where(rs > 0, rs, 1.0), np.nan)

    for method in rankings:
        for k in CUTOFFS:
            vals = tau[top_k_idx[(method, k)]]
            valid = vals[~np.isnan(vals)]
            boot_results[method][k].append(np.mean(valid) if len(valid) > 0 else np.nan)

    if (b + 1) % 1000 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (b + 1) * (args.n_boot - b - 1)
        log(f"  {b+1}/{args.n_boot}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

log(f"Bootstrap done in {time.time()-t0:.1f}s")

# ── Results ───────────────────────────────────────────────────────────────────
records = []
for method in rankings:
    for k in CUTOFFS:
        vals = np.array(boot_results[method][k])
        vals = vals[~np.isnan(vals)]
        records.append({"method": method, "k": k,
                        "mean": vals.mean(),
                        "ci_lo": np.percentile(vals, 2.5),
                        "ci_hi": np.percentile(vals, 97.5)})

df = pd.DataFrame(records)
df.to_csv(HERE / "bootstrap_summary.csv", index=False)

print("\n=== DermalC bootstrap summary (mean ± 95% CI) ===")
for k in CUTOFFS:
    sub = df[df["k"] == k].sort_values("mean", ascending=False)
    print(f"\nTop-{k}:")
    for _, row in sub.iterrows():
        print(f"  {row['method']:<10}  {row['mean']:.4f}  [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]")

# ── Plot ──────────────────────────────────────────────────────────────────────
colors = {"Locat": "#e6194b", "GSPA": "#3cb44b", "LMD": "#4363d8",
          "Hotspot": "#f58231", "Haystack": "#911eb4"}

fig, axes = plt.subplots(1, len(CUTOFFS), figsize=(5 * len(CUTOFFS), 5), sharey=False)
for ax, k in zip(axes, CUTOFFS):
    sub = df[df["k"] == k].sort_values("mean", ascending=False).reset_index(drop=True)
    bars = ax.bar(sub["method"], sub["mean"],
                  color=[colors.get(m, "gray") for m in sub["method"]],
                  edgecolor="white", linewidth=0.5, zorder=2)
    ax.errorbar(range(len(sub)), sub["mean"],
                yerr=[sub["mean"] - sub["ci_lo"], sub["ci_hi"] - sub["mean"]],
                fmt="none", color="black", capsize=4, linewidth=1.5, zorder=3)
    ax.set_title(f"Top-{k} genes")
    ax.set_ylabel("Mean τ (≥5% expressed)" if ax == axes[0] else "")
    ax.tick_params(axis="x", rotation=30)
    for bar, row in zip(bars, sub.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, row.ci_hi + 0.003,
                f"{row.mean:.3f}", ha="center", va="bottom", fontsize=8)
    sns.despine(ax=ax)

fig.suptitle(f"DermalC — Bootstrap comparison (n={args.n_boot} resamples)\n"
             "Mean τ ± 95% CI (≥5% expressing genes)",
             y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(HERE / "bootstrap_mean_tau.svg", bbox_inches="tight")
log(f"Saved bootstrap_mean_tau.svg")
