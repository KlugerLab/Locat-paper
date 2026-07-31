"""
Resampling comparison of 5 gene localization methods.

Two modes:
  Subsampling (default): 70% of cells without replacement
  Bootstrap (--replace):  full dataset with replacement (true bootstrap)

Usage:
    python run_subsample_bootstrap.py --dataset pbmc3k [--n_boot 8] [--frac 0.7] [--gpu 0]
    python run_subsample_bootstrap.py --dataset pbmc3k --replace [--n_boot 10] [--gpu 0]
"""
import argparse, os, sys, subprocess, time, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["pbmc3k", "dermalc", "kang_stim", "kang_ctrl"])
parser.add_argument("--n_boot", type=int, default=None)
parser.add_argument("--frac",   type=float, default=0.7)
parser.add_argument("--replace", action="store_true", help="Bootstrap with replacement at full dataset size")
parser.add_argument("--gpu",    type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

LOCAT_SRC   = Path("LOCAT01_PATH")
GSPA_PYTHON = "GSPA_PYTHON"
LMD_PYTHON  = "LMD_PYTHON"

# ── Dataset config ─────────────────────────────────────────────────────────────
GSPA_SCRIPT = Path(__file__).resolve().parents[2] / "notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison/run_gspa_seeded.py"
LMD_SCRIPT  = Path(__file__).resolve().parents[2] / "notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison/run_lmd_seeded.py"
KANG_RAW    = Path(__file__).resolve().parents[2] / "data/kang_counts_25k.h5ad"
KANG_OUTDIR = Path(__file__).resolve().parents[2] / "notebooks/figures/Perturb_PBMC/celltype_specificity_comparison"
NORMALIZE   = False  # set True for datasets that need normalization

_boot_tag = "bootstrap_replacement" if args.replace else "subsample_bootstrap"

if args.dataset == "pbmc3k":
    DATA_PATH      = Path(__file__).resolve().parents[2] / "data/pbmc3k_9543_lognorm.h5ad"
    CELLTYPE_COL   = "louvain"
    CUTOFFS        = [100, 200, 400]
    OUT_DIR        = Path(__file__).resolve().parents[2] / "notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison" / _boot_tag
    N_BOOT         = args.n_boot or (10 if args.replace else 8)
    TITLE_PREFIX   = "PBMC3k"
    TAU_PCT_THRESH = 0.05   # ≥5% expressing for τ
elif args.dataset == "dermalc":
    DATA_PATH      = Path(__file__).resolve().parents[2] / "data/E145_dermal_erez_2026/dc_adata_proc.h5ad"
    CELLTYPE_COL   = "celltype"
    CUTOFFS        = [50, 100, 200]
    OUT_DIR        = Path(__file__).resolve().parents[2] / "notebooks/figures/Fig2_Dermal_Condensate/celltype_specificity_comparison" / _boot_tag
    N_BOOT         = args.n_boot or (10 if args.replace else 6)
    TITLE_PREFIX   = "DermalC"
    TAU_PCT_THRESH = 0.05   # ≥5% expressing for τ
elif args.dataset in ("kang_stim", "kang_ctrl"):
    CONDITION      = args.dataset.split("_")[1]   # "stim" or "ctrl"
    DATA_PATH      = None   # loaded specially below
    CELLTYPE_COL   = "cell_type"
    CUTOFFS        = [100, 200, 400]
    OUT_DIR        = KANG_OUTDIR / f"{_boot_tag}_{CONDITION}"
    N_BOOT         = args.n_boot or (10 if args.replace else 6)
    NORMALIZE      = True
    TITLE_PREFIX   = f"Kang {CONDITION.capitalize()}"
    TAU_PCT_THRESH = 0.0    # genes already pre-filtered to n_cells≥100; no extra pct cut

if str(LOCAT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAT_SRC))

METHOD_ORDER = ["Locat", "GSPA", "LMD", "Haystack", "Hotspot"]
colors = {"Locat": "#e6194b", "GSPA": "#3cb44b", "LMD": "#4363d8",
          "Hotspot": "#f58231", "Haystack": "#911eb4"}

OUT_DIR.mkdir(parents=True, exist_ok=True)
BOOT_DIR = OUT_DIR / "subsample_bootstrap"
BOOT_DIR.mkdir(exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def compute_tau(adata):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    cts = adata.obs[CELLTYPE_COL].cat.categories.tolist()
    mean = np.array([X[adata.obs[CELLTYPE_COL] == ct].mean(axis=0) for ct in cts])
    pct  = (X > 0).mean(axis=0)
    rs   = mean.sum(axis=0)
    mask = (rs > 0) & (pct >= TAU_PCT_THRESH)
    tau  = np.where(mask, mean.max(axis=0) / np.where(rs > 0, rs, 1.0), np.nan)
    return pd.Series(tau, index=adata.var_names)

def run_one_boot(adata_sub, boot_idx, seed):
    """Run all 5 methods on adata_sub; return dict method -> ranked gene list."""
    boot_dir = BOOT_DIR / f"boot_{boot_idx:02d}"
    boot_dir.mkdir(exist_ok=True)

    # save temp h5ad for subprocess scripts
    tmp = Path(tempfile.mktemp(suffix=".h5ad"))
    adata_sub.write_h5ad(tmp)

    rankings = {}

    # ── LOCAT ──────────────────────────────────────────────────────────────────
    from locat.locat import LOCAT
    embedding = adata_sub.obsm["X_pca"].astype(np.float64)[:, :8]
    model = LOCAT(
        adata=adata_sub,
        cell_embedding=embedding,
        k=20,
        n_bootstrap_inits=50,
        show_progress=True,
        knn=adata_sub.obsp["connectivities"],
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
    df = pd.DataFrame(rows).set_index("gene")
    np.savez(boot_dir / "locat_scores.npz", gene_names=df.index.values, pval=df["pval"].values)
    rankings["Locat"] = df.sort_values("pval").index.tolist()

    # ── GSPA ───────────────────────────────────────────────────────────────────
    gspa_out = boot_dir / "gspa_scores.npz"
    subprocess.run(
        [GSPA_PYTHON, str(GSPA_SCRIPT),
         "--data_path", str(tmp), "--out_path", str(gspa_out),
         "--seed", str(seed), "--gpu", args.gpu],
        check=True,
    )
    x = np.load(gspa_out, allow_pickle=True)
    rankings["GSPA"] = pd.Series(x["gene_localization"], index=x["var_names"]).sort_values(ascending=False).index.tolist()

    # ── LMD ────────────────────────────────────────────────────────────────────
    # With-replacement bootstrap can create duplicate-cell KNN artifacts that crash LMD's
    # igraph traversal ("Unknown vertex selected"). Skip and record NaN for that boot.
    lmd_out = boot_dir / "lmd_scores.npz"
    try:
        subprocess.run(
            [LMD_PYTHON, str(LMD_SCRIPT),
             "--data_path", str(tmp), "--out_path", str(lmd_out)],
            env={**os.environ,
                 "R_HOME": "LMD_R_HOME",
                 "R_DEFAULT_PACKAGES": "base,utils,stats,graphics,grDevices,methods",
                 "CUDA_VISIBLE_DEVICES": ""},
            check=True,
        )
        x = np.load(lmd_out, allow_pickle=True)
        rankings["LMD"] = pd.Series(x["lmd_score"], index=x["var_names"]).sort_values().index.tolist()
    except subprocess.CalledProcessError as e:
        log(f"  WARNING: LMD failed (boot {boot_idx}), skipping — {e}")
        rankings["LMD"] = []

    # ── Hotspot ────────────────────────────────────────────────────────────────
    import hotspot as hs_pkg
    hs = hs_pkg.Hotspot(adata_sub, layer_key=None, model="normal",
                        latent_obsm_key="X_pca", umi_counts_obs_key=None)
    hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
    hs.compute_autocorrelations()
    hs_scores = hs.results["FDR"]
    np.savez(boot_dir / "hotspot_scores.npz", var_names=hs_scores.index.values, fdr=hs_scores.values)
    rankings["Hotspot"] = hs_scores.sort_values().index.tolist()

    # ── Haystack ───────────────────────────────────────────────────────────────
    import singleCellHaystack as sch
    res = sch.haystack(adata_sub, coord="pca")
    hay_scores = res.result.set_index("gene")["logpval"]
    np.savez(boot_dir / "haystack_scores.npz", var_names=hay_scores.index.values, logpval=hay_scores.values)
    rankings["Haystack"] = hay_scores.sort_values().index.tolist()

    tmp.unlink(missing_ok=True)
    return rankings

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def make_plots(all_tau_means, cutoffs, title_prefix):
    # all_tau_means: dict method -> dict k -> list of mean_tau across boots
    for k in cutoffs:
        vals = {m: np.array(all_tau_means[m][k]) for m in METHOD_ORDER}
        locat_vals = vals["Locat"]

        rows = []
        for m in METHOD_ORDER:
            v = vals[m]
            n = len(v)
            mean = v.mean()
            se   = v.std(ddof=1) / np.sqrt(n)
            ci   = se * stats.t.ppf(0.975, df=n-1)
            if m != "Locat":
                _, p_one = stats.ttest_rel(locat_vals, v, alternative="greater")
            else:
                p_one = np.nan
            rows.append({"method": m, "mean": mean, "ci": ci, "se": se, "p": p_one, "vals": v})

        df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(df["method"], df["mean"],
                      color=[colors[m] for m in df["method"]],
                      edgecolor="white", linewidth=0.5, zorder=2)
        ax.errorbar(range(len(df)), df["mean"], yerr=df["ci"],
                    fmt="none", color="black", capsize=5, linewidth=1.5, zorder=3)

        # significance annotations vs Locat bar
        locat_idx = df.index[df["method"] == "Locat"][0]
        locat_top = df.loc[locat_idx, "mean"] + df.loc[locat_idx, "ci"]
        y_ann = locat_top + 0.02

        for i, row in df.iterrows():
            if row["method"] == "Locat":
                continue
            label = sig_label(row["p"])
            ax.text(i, row["mean"] + row["ci"] + 0.005, label,
                    ha="center", va="bottom", fontsize=10, color="black")

        # value labels
        for bar, row in zip(bars, df.itertuples()):
            ax.text(bar.get_x() + bar.get_width()/2,
                    row.mean + row.ci + 0.02,
                    f"{row.mean:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Mean τ (≥5% expressed)")
        _ci_desc = (f"{N_BOOT} bootstrap resamples, w/ replacement"
                    if args.replace else
                    f"{N_BOOT} subsamples, {int(args.frac*100)}% cells w/o replacement")
        ax.set_title(f"{title_prefix} — Top-{k} genes\nMean τ ± 95% CI ({_ci_desc})")
        ax.tick_params(axis="x", rotation=30)
        sns.despine(ax=ax)
        plt.tight_layout()
        out = OUT_DIR / f"subsample_bootstrap_barplot_top{k}.svg"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        log(f"Saved {out}")

        # also save summary csv
        df.drop(columns=["vals"]).to_csv(OUT_DIR / f"subsample_bootstrap_summary_top{k}.csv", index=False)

# ── Main ───────────────────────────────────────────────────────────────────────
_mode_str = "bootstrap w/ replacement (n=N)" if args.replace else f"subsampling {int(args.frac*100)}% w/o replacement"
log(f"Dataset: {args.dataset}  n_boot={N_BOOT}  mode={_mode_str}")
if args.dataset in ("kang_stim", "kang_ctrl"):
    raw = sc.read_h5ad(KANG_RAW)
    adata_full = raw[raw.obs["label"] == CONDITION].copy()
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)
    expr_all = (adata_full.X.toarray() if sp.issparse(adata_full.X) else np.asarray(adata_full.X)) > 0
    adata_full = adata_full[:, expr_all.sum(axis=0) >= 100].copy()
    log(f"  Kang {CONDITION}: {adata_full.n_obs} cells × {adata_full.n_vars} genes after norm + n_cells≥100 filter")
else:
    adata_full = sc.read_h5ad(DATA_PATH)
adata_full.obs[CELLTYPE_COL] = pd.Categorical(adata_full.obs[CELLTYPE_COL])
n_cells = adata_full.n_obs

rng = np.random.default_rng(42)
all_tau_means = {m: {k: [] for k in CUTOFFS} for m in METHOD_ORDER}

n_sub = n_cells if args.replace else int(n_cells * args.frac)
log(f"Full dataset: {n_cells} cells × {adata_full.n_vars} genes → drawing {n_sub} cells each run (replace={args.replace})")

for b in range(N_BOOT):
    log(f"\n--- Bootstrap {b+1}/{N_BOOT} ---")
    t0 = time.time()

    idx = rng.choice(n_cells, size=n_sub, replace=args.replace)
    adata_sub = adata_full[idx].copy()
    adata_sub.obs[CELLTYPE_COL] = pd.Categorical(adata_sub.obs[CELLTYPE_COL])

    # recompute PCA + KNN on subsample
    sc.pp.pca(adata_sub, n_comps=50)
    sc.pp.neighbors(adata_sub, n_neighbors=20, n_pcs=50)

    tau = compute_tau(adata_sub)
    rankings = run_one_boot(adata_sub, b, seed=b)

    for m in METHOD_ORDER:
        for k in CUTOFFS:
            vals = tau[[g for g in rankings[m] if g in tau.index and not np.isnan(tau[g])]][:k]
            all_tau_means[m][k].append(vals.mean() if len(vals) > 0 else np.nan)

    log(f"  Boot {b+1} done in {(time.time()-t0)/60:.1f}m")
    # print current means
    for k in CUTOFFS:
        line = f"  top-{k}: " + "  ".join(f"{m}={np.nanmean(all_tau_means[m][k]):.3f}" for m in METHOD_ORDER)
        log(line)

np.save(OUT_DIR / "subsample_bootstrap_raw.npy", all_tau_means)
log("Making plots...")
make_plots(all_tau_means, CUTOFFS, TITLE_PREFIX)
log("All done.")
