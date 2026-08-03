"""
Run all methods on Chlamydomonas Fe- and Fe+ samples and compare using τ metric.

Pseudo-cell-types = diurnal phase labels (G1/S/G2M) from Zones 2015 bulk
time-course, assigned via gene module scores (assign_cell_cycle_phases.py).

Methods compared:
  Locat    — scores already computed: locat_run/scores/<cond>/locat_scores.npz
  Hotspot  — run here
  Haystack — run here
  LMD      — run via subprocess (R/rpy2 env)
  GSPA     — run via subprocess (gspa-env)
  Scanpy   — Leiden (res=0.5) + Wilcoxon rank_genes_groups

τ = max_phase_mean / sum_phase_means  (requiring ≥5% expressing cells)

Usage:
    python run_chlamydomonas_comparison.py [--gpu 0]
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
LOCAT_SRC = Path("LOCAT01_PATH")
SCORES    = HERE / "locat_run" / "scores"
OUT       = HERE / "comparison_results"
OUT.mkdir(exist_ok=True)

GSPA_PYTHON = "GSPA_PYTHON"
LMD_PYTHON  = "LMD_PYTHON"
LMD_SCRIPT  = Path(__file__).resolve().parents[3] / "notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison/run_lmd_seeded.py"
GSPA_SCRIPT = Path(__file__).resolve().parents[3] / "notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison/run_gspa_seeded.py"

if str(LOCAT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAT_SRC))

CELLTYPE_COL = "phase"
CUTOFFS      = [50, 100, 200]
LEIDEN_RES   = 0.5

COLORS = {
    "Locat":    "#e6194b",
    "GSPA":     "#3cb44b",
    "LMD":      "#4363d8",
    "Hotspot":  "#f58231",
    "Haystack": "#911eb4",
    "Scanpy":   "#f032e6",
}
METHOD_ORDER = ["Locat", "GSPA", "LMD", "Haystack", "Hotspot", "Scanpy"]

SAMPLES = [
    dict(name="fe_neg", label="Fe−"),
    dict(name="fe_pos", label="Fe+"),
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def compute_tau(adata):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    phases = adata.obs[CELLTYPE_COL].cat.categories.tolist()
    # ≥5% expressing filter
    pct_expr = (X > 0).mean(axis=0)
    expressed = pct_expr >= 0.05
    mean = np.array([X[adata.obs[CELLTYPE_COL] == ph].mean(axis=0) for ph in phases])
    rs   = mean.sum(axis=0)
    tau  = np.where((rs > 0) & expressed, mean.max(axis=0) / rs, np.nan)
    return pd.Series(tau, index=adata.var_names)

def make_barplot(tau, rankings, cutoffs, svg_path, title):
    fig, axes = plt.subplots(1, len(cutoffs), figsize=(5*len(cutoffs), 5), sharey=False)
    for ax, k in zip(axes, cutoffs):
        rows = []
        for m in METHOD_ORDER:
            if not rankings.get(m):
                continue
            vals = tau[[g for g in rankings[m] if g in tau.index]][:k].dropna()
            rows.append({"method": m, "mean": vals.mean(), "sd": vals.std()})
        sub = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
        bars = ax.bar(sub["method"], sub["mean"],
                      color=[COLORS[m] for m in sub["method"]],
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
    log(f"Saved {svg_path.name}")

def make_boxplot(tau, rankings, cutoffs, svg_path, title):
    fig, axes = plt.subplots(1, len(cutoffs), figsize=(5*len(cutoffs), 5), sharey=False)
    for ax, k in zip(axes, cutoffs):
        present = [m for m in METHOD_ORDER if rankings.get(m)]
        medians = {m: np.nanmedian(tau[[g for g in rankings[m] if g in tau.index]][:k].dropna()) for m in present}
        sorted_methods = sorted(present, key=lambda m: medians[m], reverse=True)
        data = [tau[[g for g in rankings[m] if g in tau.index]][:k].dropna().values for m in sorted_methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.4, linestyle="none"))
        for patch, m in zip(bp["boxes"], sorted_methods):
            patch.set_facecolor(COLORS[m]); patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(sorted_methods)+1))
        ax.set_xticklabels(sorted_methods, rotation=30, ha="right")
        ax.set_title(f"Top-{k} genes")
        ax.set_ylabel("τ (≥5% expressed)" if ax == axes[0] else "")
        sns.despine(ax=ax)
    fig.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    log(f"Saved {svg_path.name}")

def run_sample(ds):
    name     = ds["name"]
    label    = ds["label"]
    cond_dir = SCORES / name
    out_dir  = OUT / name
    out_dir.mkdir(exist_ok=True)

    log(f"\n{'='*60}")
    log(f"Processing: {label}")
    log(f"{'='*60}")

    # Load adata with phase labels
    adata = sc.read_h5ad(cond_dir / "adata_phases.h5ad")
    adata.obs[CELLTYPE_COL] = pd.Categorical(adata.obs[CELLTYPE_COL])
    log(f"  {adata.n_obs} cells × {adata.n_vars} genes")
    log(f"  Phase distribution: {adata.obs[CELLTYPE_COL].value_counts().to_dict()}")

    tau = compute_tau(adata)
    log(f"  Genes with valid τ: {tau.notna().sum()}")

    # Save tmp h5ad for subprocess scripts
    tmp_h5ad = Path(tempfile.mktemp(suffix=".h5ad"))
    adata.write_h5ad(tmp_h5ad)

    rankings = {}

    # ── Locat (pre-computed) ──────────────────────────────────────────────────
    locat_path = cond_dir / "locat_scores.npz"
    x = np.load(locat_path, allow_pickle=True)
    rankings["Locat"] = pd.Series(x["pval"], index=x["gene_names"]).sort_values().index.tolist()
    log(f"  Locat: loaded {len(rankings['Locat'])} genes")

    # ── GSPA ─────────────────────────────────────────────────────────────────
    gspa_path = out_dir / "gspa_scores.npz"
    if not gspa_path.exists():
        log("  Running GSPA...")
        t0 = time.time()
        try:
            subprocess.run(
                [GSPA_PYTHON, str(GSPA_SCRIPT),
                 "--data_path", str(tmp_h5ad),
                 "--out_path",  str(gspa_path),
                 "--seed", "0", "--gpu", args.gpu],
                check=True, timeout=3600,
            )
            log(f"  GSPA done in {(time.time()-t0)/60:.1f}m")
        except Exception as e:
            log(f"  GSPA failed: {e}")
    if gspa_path.exists():
        x = np.load(gspa_path, allow_pickle=True)
        rankings["GSPA"] = pd.Series(x["gene_localization"], index=x["var_names"]).sort_values(ascending=False).index.tolist()
    else:
        rankings["GSPA"] = []

    # ── LMD ──────────────────────────────────────────────────────────────────
    lmd_path = out_dir / "lmd_scores.npz"
    if not lmd_path.exists():
        log("  Running LMD...")
        t0 = time.time()
        try:
            subprocess.run(
                [LMD_PYTHON, str(LMD_SCRIPT),
                 "--data_path", str(tmp_h5ad),
                 "--out_path",  str(lmd_path)],
                env={**os.environ,
                     "R_HOME": "LMD_R_HOME",
                     "R_DEFAULT_PACKAGES": "base,utils,stats,graphics,grDevices,methods",
                     "CUDA_VISIBLE_DEVICES": ""},
                check=True, timeout=3600,
            )
            log(f"  LMD done in {(time.time()-t0)/60:.1f}m")
        except Exception as e:
            log(f"  LMD failed: {e}")
    if lmd_path.exists():
        x = np.load(lmd_path, allow_pickle=True)
        rankings["LMD"] = pd.Series(x["lmd_score"], index=x["var_names"]).sort_values().index.tolist()
    else:
        rankings["LMD"] = []

    # ── Hotspot ───────────────────────────────────────────────────────────────
    hotspot_path = out_dir / "hotspot_scores.npz"
    if not hotspot_path.exists():
        log("  Running Hotspot...")
        t0 = time.time()
        try:
            import hotspot as hs_pkg
            hs = hs_pkg.Hotspot(adata, layer_key=None, model="normal",
                                latent_obsm_key="X_pca", umi_counts_obs_key=None)
            hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
            hs.compute_autocorrelations()
            fdr = hs.results["FDR"]
            np.savez(hotspot_path, var_names=fdr.index.values, fdr=fdr.values)
            log(f"  Hotspot done in {(time.time()-t0)/60:.1f}m")
        except Exception as e:
            log(f"  Hotspot failed: {e}")
    if hotspot_path.exists():
        x = np.load(hotspot_path, allow_pickle=True)
        rankings["Hotspot"] = pd.Series(x["fdr"], index=x["var_names"]).sort_values().index.tolist()
    else:
        rankings["Hotspot"] = []

    # ── Haystack ──────────────────────────────────────────────────────────────
    haystack_path = out_dir / "haystack_scores.npz"
    if not haystack_path.exists():
        log("  Running Haystack...")
        t0 = time.time()
        try:
            import singleCellHaystack as sch
            res = sch.haystack(adata, coord="pca")
            hay = res.result.set_index("gene")["logpval"]
            np.savez(haystack_path, var_names=hay.index.values, logpval=hay.values)
            log(f"  Haystack done in {(time.time()-t0)/60:.1f}m")
        except Exception as e:
            log(f"  Haystack failed: {e}")
    if haystack_path.exists():
        x = np.load(haystack_path, allow_pickle=True)
        rankings["Haystack"] = pd.Series(x["logpval"], index=x["var_names"]).sort_values().index.tolist()
    else:
        rankings["Haystack"] = []

    # ── Scanpy Leiden + Wilcoxon ──────────────────────────────────────────────
    scanpy_path = out_dir / "scanpy_scores.npz"
    if not scanpy_path.exists():
        log("  Running Scanpy Leiden+Wilcoxon...")
        t0 = time.time()
        sc.pp.neighbors(adata, n_neighbors=30, n_pcs=min(50, adata.obsm["X_pca"].shape[1]))
        sc.tl.leiden(adata, resolution=LEIDEN_RES, key_added="leiden_scanpy")
        sc.tl.rank_genes_groups(adata, groupby="leiden_scanpy", method="wilcoxon",
                                key_added="rank_genes_scanpy", use_raw=False, pts=False)
        rgg = adata.uns["rank_genes_scanpy"]
        gene_pval_min = {}
        for grp in rgg["names"].dtype.names:
            for g, p in zip(rgg["names"][grp], rgg["pvals_adj"][grp]):
                if g not in gene_pval_min or p < gene_pval_min[g]:
                    gene_pval_min[g] = p
        var_names = np.array(list(gene_pval_min.keys()))
        pval_min  = np.array([gene_pval_min[g] for g in var_names])
        np.savez(scanpy_path, var_names=var_names, pval_min=pval_min)
        log(f"  Scanpy done in {(time.time()-t0)/60:.1f}m, {len(var_names)} genes")
    if scanpy_path.exists():
        x = np.load(scanpy_path, allow_pickle=True)
        rankings["Scanpy"] = pd.Series(x["pval_min"], index=x["var_names"]).sort_values().index.tolist()
    else:
        rankings["Scanpy"] = []

    tmp_h5ad.unlink(missing_ok=True)

    # ── Plots ─────────────────────────────────────────────────────────────────
    make_barplot(tau, rankings, CUTOFFS,
                 OUT / f"barplot_tau_{name}.svg",
                 f"Chlamydomonas {label} — Mean τ ± SD of top-k genes (G1/S/G2M phases)")
    make_boxplot(tau, rankings, CUTOFFS,
                 OUT / f"boxplot_tau_{name}.svg",
                 f"Chlamydomonas {label} — τ distribution of top-k genes (G1/S/G2M phases)")

    # Print top-50 τ summary per method
    log(f"\n  Top-50 mean τ per method:")
    for m in METHOD_ORDER:
        if not rankings.get(m):
            log(f"    {m}: N/A")
            continue
        vals = tau[[g for g in rankings[m] if g in tau.index]][:50].dropna()
        log(f"    {m}: {vals.mean():.4f} ± {vals.std():.4f} (n={len(vals)})")

    return rankings, tau

t_total = time.time()
for ds in SAMPLES:
    run_sample(ds)
log(f"\nAll done in {(time.time()-t_total)/60:.1f}m total.")
