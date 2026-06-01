"""
Run all 5 methods on Kang stim and ctrl subsets separately, then
generate barplot (mean±SD tau) and boxplot for each condition.

Usage:
    python run_kang_comparison.py [--gpu 0]
"""
import argparse, os, sys, subprocess, time, tempfile
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
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

HERE      = Path(__file__).parent
LOCAT_SRC = Path("/banach2/wes/locat-0.1")
DATA_PATH = Path("/banach2/wes/Locat-paper-repro-private/data/kang_counts_25k.h5ad")
SCORES    = HERE / "scores"
SCORES.mkdir(exist_ok=True)

GSPA_PYTHON = "/banach2/wes/.conda/envs/gspa-env/bin/python"
LMD_PYTHON  = "/banach2/wes/envs/lmd_rpy2/bin/python"
LMD_SCRIPT  = Path("/banach2/wes/Locat-paper-repro-private/notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison/run_lmd_seeded.py")
GSPA_SCRIPT = Path("/banach2/wes/Locat-paper-repro-private/notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison/run_gspa_seeded.py")

if str(LOCAT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAT_SRC))

CUTOFFS = [100, 200, 400]
CELLTYPE_COL = "cell_type"
CONDITION_COL = "label"

colors = {"Locat": "#e6194b", "GSPA": "#3cb44b", "LMD": "#4363d8",
          "Hotspot": "#f58231", "Haystack": "#911eb4"}
METHOD_ORDER = ["Locat", "GSPA", "LMD", "Haystack", "Hotspot"]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def compute_tau(adata):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    cts = adata.obs[CELLTYPE_COL].cat.categories.tolist()
    mean = np.array([X[adata.obs[CELLTYPE_COL] == ct].mean(axis=0) for ct in cts])
    pct  = (X > 0).mean(axis=0)
    rs   = mean.sum(axis=0)
    mask = (rs > 0) & (pct >= 0.05)
    tau  = np.where(mask, mean.max(axis=0) / np.where(rs > 0, rs, 1.0), np.nan)
    return pd.Series(tau, index=adata.var_names)

def make_barplot(tau, rankings, cutoffs, svg_path, title):
    fig, axes = plt.subplots(1, len(cutoffs), figsize=(5*len(cutoffs), 5), sharey=False)
    for ax, k in zip(axes, cutoffs):
        rows = []
        for m in METHOD_ORDER:
            vals = tau[[g for g in rankings[m] if g in tau.index]][:k].dropna()
            rows.append({"method": m, "mean": vals.mean(), "sd": vals.std()})
        sub = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
        bars = ax.bar(sub["method"], sub["mean"],
                      color=[colors[m] for m in sub["method"]],
                      edgecolor="white", linewidth=0.5, zorder=2)
        ax.errorbar(range(len(sub)), sub["mean"], yerr=sub["sd"],
                    fmt="none", color="black", capsize=4, linewidth=1.5, zorder=3)
        ax.set_title(f"Top-{k} genes")
        ax.set_ylabel("Mean τ (≥5% expressed)" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=30)
        for bar, row in zip(bars, sub.itertuples()):
            ax.text(bar.get_x() + bar.get_width()/2, row.mean + row.sd + 0.005,
                    f"{row.mean:.3f}", ha="center", va="bottom", fontsize=8)
        sns.despine(ax=ax)
    fig.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    log(f"Saved {svg_path}")

def make_boxplot(tau, rankings, cutoffs, svg_path, title):
    fig, axes = plt.subplots(1, len(cutoffs), figsize=(5*len(cutoffs), 5), sharey=False)
    for ax, k in zip(axes, cutoffs):
        medians = {m: np.nanmedian(tau[[g for g in rankings[m] if g in tau.index]][:k].dropna()) for m in METHOD_ORDER}
        sorted_methods = sorted(METHOD_ORDER, key=lambda m: medians[m], reverse=True)
        data = [tau[[g for g in rankings[m] if g in tau.index]][:k].dropna().values for m in sorted_methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.4, linestyle="none"))
        for patch, m in zip(bp["boxes"], sorted_methods):
            patch.set_facecolor(colors[m]); patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(sorted_methods)+1))
        ax.set_xticklabels(sorted_methods, rotation=30, ha="right")
        ax.set_title(f"Top-{k} genes")
        ax.set_ylabel("τ (≥5% expressed)" if ax == axes[0] else "")
        sns.despine(ax=ax)
    fig.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    log(f"Saved {svg_path}")

def run_condition(condition):
    log(f"{'='*60}")
    log(f"Processing condition: {condition}")
    log(f"{'='*60}")

    cond_dir = SCORES / condition
    cond_dir.mkdir(exist_ok=True)

    # ── Load and subset ───────────────────────────────────────────────────────
    log("Loading and subsetting adata...")
    adata_full = sc.read_h5ad(DATA_PATH)
    adata = adata_full[adata_full.obs[CONDITION_COL] == condition].copy()

    # filter genes ≥5% expressed
    pct = (adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)) > 0
    gene_mask = pct.mean(axis=0) >= 0.05
    adata = adata[:, gene_mask].copy()
    log(f"  {adata.n_obs} cells × {adata.n_vars} genes after ≥5% filter")
    log(f"  Cell types: {adata.obs[CELLTYPE_COL].value_counts().to_dict()}")

    # ensure cell_type is categorical
    adata.obs[CELLTYPE_COL] = pd.Categorical(adata.obs[CELLTYPE_COL])

    # recompute PCA and KNN (needed for LOCAT and Hotspot)
    log("  Computing PCA + neighbors...")
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)

    # save temp h5ad for subprocess scripts
    tmp_h5ad = Path(tempfile.mktemp(suffix=".h5ad"))
    adata.write_h5ad(tmp_h5ad)
    log(f"  Saved temp adata to {tmp_h5ad}")

    # ── 1. LOCAT ──────────────────────────────────────────────────────────────
    log("Running LOCAT...")
    t0 = time.time()
    from locat.locat import LOCAT
    embedding = adata.obsm["X_pca"].astype(np.float64)[:, :8]
    model = LOCAT(
        adata=adata,
        cell_embedding=embedding,
        k=20,
        n_bootstrap_inits=50,
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
    np.savez(cond_dir / "locat_scores.npz",
             gene_names=locat_df.index.values, pval=locat_df["pval"].values)
    log(f"  LOCAT done in {(time.time()-t0)/60:.1f}m")

    # ── 2. GSPA ───────────────────────────────────────────────────────────────
    log("Running GSPA...")
    t0 = time.time()
    subprocess.run(
        [GSPA_PYTHON, str(GSPA_SCRIPT),
         "--data_path", str(tmp_h5ad),
         "--out_path",  str(cond_dir / "gspa_scores.npz"),
         "--seed", "0", "--gpu", args.gpu],
        check=True,
    )
    log(f"  GSPA done in {(time.time()-t0)/60:.1f}m")

    # ── 3. LMD ────────────────────────────────────────────────────────────────
    log("Running LMD...")
    t0 = time.time()
    subprocess.run(
        [LMD_PYTHON, str(LMD_SCRIPT),
         "--data_path", str(tmp_h5ad),
         "--out_path",  str(cond_dir / "lmd_scores.npz")],
        env={**os.environ,
             "R_HOME": "/banach2/wes/envs/lmd_rpy2/lib/R",
             "R_DEFAULT_PACKAGES": "base,utils,stats,graphics,grDevices,methods",
             "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )
    log(f"  LMD done in {(time.time()-t0)/60:.1f}m")

    # ── 4. Hotspot ────────────────────────────────────────────────────────────
    log("Running Hotspot...")
    t0 = time.time()
    import hotspot as hs_pkg
    hs = hs_pkg.Hotspot(adata, layer_key=None, model="normal",
                        latent_obsm_key="X_pca", umi_counts_obs_key=None)
    hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
    hs.compute_autocorrelations()
    hotspot_scores = hs.results["FDR"]
    np.savez(cond_dir / "hotspot_scores.npz",
             var_names=hotspot_scores.index.values, fdr=hotspot_scores.values)
    log(f"  Hotspot done in {(time.time()-t0)/60:.1f}m")

    # ── 5. Haystack ───────────────────────────────────────────────────────────
    log("Running Haystack...")
    t0 = time.time()
    import singleCellHaystack as sch
    res = sch.haystack(adata, coord="pca")
    haystack_scores = res.result.set_index("gene")["logpval"]
    np.savez(cond_dir / "haystack_scores.npz",
             var_names=haystack_scores.index.values, logpval=haystack_scores.values)
    log(f"  Haystack done in {(time.time()-t0)/60:.1f}m")

    tmp_h5ad.unlink(missing_ok=True)

    # ── Load rankings and compute tau ─────────────────────────────────────────
    log("Computing τ and generating plots...")
    tau = compute_tau(adata)

    rankings = {}
    x = np.load(cond_dir / "locat_scores.npz",   allow_pickle=True); rankings["Locat"]   = pd.Series(x["pval"],              index=x["gene_names"]).sort_values().index.tolist()
    x = np.load(cond_dir / "gspa_scores.npz",    allow_pickle=True); rankings["GSPA"]    = pd.Series(x["gene_localization"], index=x["var_names"]).sort_values(ascending=False).index.tolist()
    x = np.load(cond_dir / "lmd_scores.npz",     allow_pickle=True); rankings["LMD"]     = pd.Series(x["lmd_score"],         index=x["var_names"]).sort_values().index.tolist()
    x = np.load(cond_dir / "hotspot_scores.npz", allow_pickle=True); rankings["Hotspot"] = pd.Series(x["fdr"],               index=x["var_names"]).sort_values().index.tolist()
    x = np.load(cond_dir / "haystack_scores.npz",allow_pickle=True); rankings["Haystack"]= pd.Series(x["logpval"],           index=x["var_names"]).sort_values().index.tolist()

    label = condition.capitalize()
    make_barplot(tau, rankings, CUTOFFS,
                 HERE / f"barplot_tau_{condition}.svg",
                 f"Kang {label} PBMC — Mean τ ± SD of top-k genes per method")
    make_boxplot(tau, rankings, CUTOFFS,
                 HERE / f"boxplot_tau_{condition}.svg",
                 f"Kang {label} PBMC — τ distribution of top-k genes per method")

    log(f"Condition '{condition}' complete.")
    return rankings, tau

# ── Main ──────────────────────────────────────────────────────────────────────
t_total = time.time()
run_condition("stim")
run_condition("ctrl")
log(f"All done in {(time.time()-t_total)/60:.1f}m total.")
