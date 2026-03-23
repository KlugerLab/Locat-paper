"""
Assess LOCAT multiple testing impact and null p-value calibration.

Outputs:
  1) Panel: raw vs BH-adjusted discoveries on simulated mixed signal/null genes.
  2) Panel: QQ plots for combined LOCAT p-values under several null scenarios.
  3) CSV tables with calibration metrics and threshold summaries.

Design note:
  This script is intentionally conservative and transparent: it reports every tested
  null scenario and ranks them by distance to Uniform(0,1), rather than showing only
  one preferred simulation.
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys
from dataclasses import dataclass

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import kstest
from sklearn.datasets import make_blobs

sys.path.insert(0, "../../../locat-0.1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from locat.locat import LOCAT


OUTDIR = "/banach2/wes/Locat/notebooks/figures/Simulations"
P_FLOOR = 1e-300

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 17,
        "axes.labelsize": 17,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def cauchy_combine(pvals, p_floor=P_FLOOR):
    def _safe_float(x):
        x = float(pd.to_numeric(x, errors="coerce"))
        if not np.isfinite(x):
            return np.nan
        return float(np.clip(x, p_floor, 1.0))

    ps = np.array([_safe_float(p) for p in pvals], dtype=float)
    w = np.ones(len(ps), dtype=float) / float(len(ps))
    t = np.sum(w * np.tan((0.5 - ps) * np.pi))
    p = 0.5 - np.arctan(t) / np.pi
    return float(np.clip(p, p_floor, 1.0))


def _safe_p(x, p_floor=P_FLOOR):
    x = float(pd.to_numeric(x, errors="coerce"))
    if not np.isfinite(x):
        return 1.0
    return float(np.clip(x, p_floor, 1.0))


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def run_locat(gene_matrix, coords, knn_neighbors=30, n_components=20, n_bootstrap_inits=80):
    adata = ad.AnnData(gene_matrix.astype(np.float64))
    adata.obs_names = [f"Cell_{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i}" for i in range(adata.n_vars)]
    adata.obsm["coords"] = coords.astype(np.float64)
    sc.pp.neighbors(adata, use_rep="coords", n_neighbors=knn_neighbors)

    model = LOCAT(
        adata,
        adata.obsm["coords"].astype(np.float64),
        n_components,
        show_progress=True,
        n_bootstrap_inits=n_bootstrap_inits,
        knn=adata.obsp["connectivities"],
    )
    model.background_pdf(weights_transform=None)
    return model.gmm_scan(
        zscore_thresh=-np.inf,
        max_freq=1.0,
        rc_min_abs_deficit=0.0,
        rc_min_expected=0.0,
        rc_min_p0_abs=0.0,
        rc_n_trials_cap=np.sqrt(adata.shape[0]),
        rc_n_eff_scale=0.9,
    )


def extract_combined_pvals(sres):
    genes = list(sres.keys())
    loc = np.array([float(np.clip(getattr(sres[g], "depletion_pval"), P_FLOOR, 1.0)) for g in genes], dtype=float)
    con = np.array([float(np.clip(getattr(sres[g], "concentration_pval"), P_FLOOR, 1.0)) for g in genes], dtype=float)
    p_cauchy = np.array([cauchy_combine([loc[i], con[i]]) for i in range(len(genes))], dtype=float)
    sample_size = np.array([float(getattr(sres[g], "sample_size", np.nan)) for g in genes], dtype=float)
    sens_score = np.array([float(np.clip(getattr(sres[g], "sens_score", 1.0), 1e-9, 1.0)) for g in genes], dtype=float)
    p_size = np.array([_safe_p(1.0 - np.exp(-1.0 / (n + 1.0))) if np.isfinite(n) else 1.0 for n in sample_size], dtype=float)
    p_sens = np.array([_safe_p(1.0 - (s + 1e-9)) for s in sens_score], dtype=float)
    p_final_model = np.array([float(np.clip(getattr(sres[g], "pval", np.nan), P_FLOOR, 1.0)) for g in genes], dtype=float)
    p_final = np.where(np.isfinite(p_final_model), p_final_model, 1.0 - (1.0 - p_cauchy) * (1.0 - 0.05 * p_size) * (1.0 - 0.12 * p_sens))
    p_final = np.clip(p_final, P_FLOOR, 1.0)
    return pd.DataFrame(
        {
            "gene": genes,
            "depletion_pval": loc,
            "concentration_pval": con,
            "p_combined": p_cauchy,
            "p_size": p_size,
            "p_sens": p_sens,
            "p_final": p_final,
            "sample_size": sample_size,
            "sens_score": sens_score,
        }
    )


def qq_points(pvals):
    p = np.sort(np.clip(np.asarray(pvals, dtype=float), P_FLOOR, 1.0))
    n = p.size
    expected = np.arange(1, n + 1, dtype=float) / (n + 1.0)
    return expected, p


def calibration_metrics(pvals):
    p = np.clip(np.asarray(pvals, dtype=float), 0.0, 1.0)
    ks_stat, ks_p = kstest(p, "uniform")
    m = p.size
    grid = np.array([0.001, 0.01, 0.05, 0.10], dtype=float)
    frac_below = np.array([(p <= a).mean() for a in grid], dtype=float)
    return {
        "n_genes": int(m),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "frac_below_0.001": float(frac_below[0]),
        "frac_below_0.01": float(frac_below[1]),
        "frac_below_0.05": float(frac_below[2]),
        "frac_below_0.10": float(frac_below[3]),
    }


@dataclass
class NullScenario:
    name: str
    n_cells: int
    n_genes: int
    sample_size_mode: str
    sample_size_min: int
    sample_size_max: int
    cluster_std: float


def simulate_null_gene_matrix(coords, scenario, seed):
    rng = np.random.default_rng(seed)
    n_cells = coords.shape[0]
    genes = np.zeros((n_cells, scenario.n_genes), dtype=np.uint8)
    dists_to_center = np.sqrt(np.sum((coords - coords.mean(axis=0, keepdims=True)) ** 2, axis=1))
    w_center = np.exp(-(dists_to_center ** 2) / (2.0 * scenario.cluster_std ** 2))
    w_center /= w_center.sum()

    for j in range(scenario.n_genes):
        if scenario.sample_size_mode == "fixed":
            n_expr = scenario.sample_size_min
        elif scenario.sample_size_mode == "uniform":
            n_expr = int(rng.integers(scenario.sample_size_min, scenario.sample_size_max + 1))
        elif scenario.sample_size_mode == "geom":
            u = rng.random()
            lo = np.log(max(scenario.sample_size_min, 1))
            hi = np.log(max(scenario.sample_size_max, scenario.sample_size_min + 1))
            n_expr = int(np.clip(np.round(np.exp(lo + u * (hi - lo))), scenario.sample_size_min, scenario.sample_size_max))
        else:
            raise ValueError(f"Unknown sample_size_mode: {scenario.sample_size_mode}")

        if scenario.name.endswith("center_weighted"):
            idx = rng.choice(n_cells, size=n_expr, replace=False, p=w_center)
        else:
            idx = rng.choice(n_cells, size=n_expr, replace=False)
        genes[idx, j] = 1
    return genes


def simulate_mixed_signal_matrix(coords, n_genes=350, signal_fraction=0.20, seed=7):
    rng = np.random.default_rng(seed)
    n_cells = coords.shape[0]
    n_signal = int(round(n_genes * signal_fraction))
    n_null = n_genes - n_signal
    genes = np.zeros((n_cells, n_genes), dtype=np.uint8)
    truth = np.zeros(n_genes, dtype=np.uint8)

    center = coords.mean(axis=0)
    d = np.sqrt(np.sum((coords - center) ** 2, axis=1))
    in_r = np.flatnonzero(d < np.quantile(d, 0.35))
    out_r = np.flatnonzero(d >= np.quantile(d, 0.35))

    for j in range(n_signal):
        n_expr = int(rng.integers(25, 90))
        n_in = int(np.clip(np.round(n_expr * rng.uniform(0.75, 0.95)), 1, n_expr))
        n_out = n_expr - n_in
        idx_in = rng.choice(in_r, size=n_in, replace=False) if n_in > 0 else np.array([], dtype=int)
        idx_out = rng.choice(out_r, size=n_out, replace=False) if n_out > 0 else np.array([], dtype=int)
        idx = np.unique(np.concatenate([idx_in, idx_out]))
        if idx.size < n_expr:
            rem = np.setdiff1d(np.arange(n_cells), idx)
            extra = rng.choice(rem, size=(n_expr - idx.size), replace=False)
            idx = np.concatenate([idx, extra])
        genes[idx, j] = 1
        truth[j] = 1

    for j in range(n_signal, n_signal + n_null):
        n_expr = int(rng.integers(25, 90))
        idx = rng.choice(n_cells, size=n_expr, replace=False)
        genes[idx, j] = 1
    return genes, truth


def summarize_thresholds(df, alpha=0.05):
    m = df.shape[0]
    df = df.copy()
    df["q_bh"] = benjamini_hochberg(df["p_final"].to_numpy())
    raw_hits = int((df["p_final"] <= alpha).sum())
    bh_hits = int((df["q_bh"] <= alpha).sum())

    # If all tested genes were null, this is expected false positives.
    exp_fp_raw_under_global_null = alpha * m
    return {
        "n_tests": int(m),
        "alpha": float(alpha),
        "raw_hits": raw_hits,
        "bh_hits_q05": bh_hits,
        "raw_hit_rate": float(raw_hits / m),
        "bh_hit_rate": float(bh_hits / m),
        "expected_fp_raw_if_all_null": float(exp_fp_raw_under_global_null),
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("=== LOCAT multiple testing + calibration report ===")

    coords, _, _ = make_blobs(
        n_samples=[4500],
        n_features=2,
        centers=None,
        return_centers=True,
        random_state=0,
        cluster_std=[1.25],
    )

    # 1) Mixed signal simulation: compare raw p vs BH q-values.
    print("Running mixed signal/null simulation...")
    genes_mix, truth_mix = simulate_mixed_signal_matrix(coords, n_genes=350, signal_fraction=0.20, seed=7)
    sres_mix = run_locat(genes_mix, coords)
    df_mix = extract_combined_pvals(sres_mix)
    df_mix["is_signal_sim"] = truth_mix.astype(int)
    df_mix["q_bh"] = benjamini_hochberg(df_mix["p_final"].to_numpy())
    df_mix.to_csv(f"{OUTDIR}/locat_multitest_mixed_signal_results.csv", index=False)

    summary_mix = summarize_thresholds(df_mix, alpha=0.05)
    summary_mix["raw_true_positives_sim"] = int(((df_mix["p_final"] <= 0.05) & (df_mix["is_signal_sim"] == 1)).sum())
    summary_mix["raw_false_positives_sim"] = int(((df_mix["p_final"] <= 0.05) & (df_mix["is_signal_sim"] == 0)).sum())
    summary_mix["bh_true_positives_sim"] = int(((df_mix["q_bh"] <= 0.05) & (df_mix["is_signal_sim"] == 1)).sum())
    summary_mix["bh_false_positives_sim"] = int(((df_mix["q_bh"] <= 0.05) & (df_mix["is_signal_sim"] == 0)).sum())

    # 2) Null simulations for calibration / QQ plots.
    null_scenarios = [
        NullScenario("null_fixed_n40_uniform", n_cells=4500, n_genes=260, sample_size_mode="fixed", sample_size_min=40, sample_size_max=40, cluster_std=1.25),
        NullScenario("null_uniform_n20_120_uniform", n_cells=4500, n_genes=260, sample_size_mode="uniform", sample_size_min=20, sample_size_max=120, cluster_std=1.25),
        NullScenario("null_geom_n10_150_uniform", n_cells=4500, n_genes=260, sample_size_mode="geom", sample_size_min=10, sample_size_max=150, cluster_std=1.25),
        NullScenario("null_uniform_n20_120_center_weighted", n_cells=4500, n_genes=260, sample_size_mode="uniform", sample_size_min=20, sample_size_max=120, cluster_std=1.25),
    ]

    calibration_rows = []
    qq_data_by_scenario = {}
    scenario_examples = {}
    for i, scen in enumerate(null_scenarios):
        print(f"Running calibration scenario: {scen.name}")
        genes_null = simulate_null_gene_matrix(coords, scen, seed=100 + i)
        sres_null = run_locat(genes_null, coords)
        df_null = extract_combined_pvals(sres_null)
        df_null["q_bh"] = benjamini_hochberg(df_null["p_final"].to_numpy())
        df_null.to_csv(f"{OUTDIR}/{scen.name}_locat_results.csv", index=False)

        m = calibration_metrics(df_null["p_final"].to_numpy())
        m["scenario"] = scen.name
        m["raw_hits_p005"] = int((df_null["p_final"] <= 0.05).sum())
        m["bh_hits_q005"] = int((df_null["q_bh"] <= 0.05).sum())
        calibration_rows.append(m)
        qq_data_by_scenario[scen.name] = qq_points(df_null["p_final"].to_numpy())
        scenario_examples[scen.name] = {"genes_matrix": genes_null, "df": df_null.copy()}

    cal_df = pd.DataFrame(calibration_rows).sort_values(["ks_stat", "frac_below_0.05"], ascending=[True, True])
    cal_df.to_csv(f"{OUTDIR}/locat_null_calibration_summary.csv", index=False)

    # 3) Figure: raw vs BH and QQ calibration.
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.32, hspace=0.35)

    ax0 = fig.add_subplot(gs[:, 0])
    bins = np.linspace(0.0, 0.12, 30)
    ax0.hist(df_mix.loc[df_mix["is_signal_sim"] == 0, "p_final"], bins=bins, alpha=0.65, color="#4C78A8", label="sim null genes")
    ax0.hist(df_mix.loc[df_mix["is_signal_sim"] == 1, "p_final"], bins=bins, alpha=0.65, color="#E45756", label="sim signal genes")
    ax0.axvline(0.05, color="black", lw=1.0, linestyle="--")
    ax0.set_title("Mixed simulation: p-value separation")
    ax0.set_xlabel("Final LOCAT p-value")
    ax0.set_ylabel("Gene count")
    ax0.legend(frameon=False, fontsize=12)
    txt = (
        f"n={summary_mix['n_tests']}\n"
        f"raw hits @0.05: {summary_mix['raw_hits']}\n"
        f"BH hits @0.05: {summary_mix['bh_hits_q05']}\n"
        f"expected FP if all null: {summary_mix['expected_fp_raw_if_all_null']:.1f}\n"
        f"raw FP (sim): {summary_mix['raw_false_positives_sim']}\n"
        f"BH FP (sim): {summary_mix['bh_false_positives_sim']}"
    )
    ax0.text(0.98, 0.98, txt, transform=ax0.transAxes, va="top", ha="right", fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))

    top_for_plot = cal_df.head(4).copy()
    scenario_label = {
        "null_fixed_n40_uniform": "Null: fixed n=40, spatially uniform",
        "null_uniform_n20_120_uniform": "Null: n~Uniform[20,120], spatially uniform",
        "null_geom_n10_150_uniform": "Null: n~log-uniform [10,150], spatially uniform",
        "null_uniform_n20_120_center_weighted": "Stress null: n~Uniform[20,120], center-weighted",
    }
    axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    for ax, (_, row) in zip(axes, top_for_plot.iterrows()):
        scen = row["scenario"]
        exp_p, obs_p = qq_data_by_scenario[scen]
        ax.plot(-np.log10(exp_p), -np.log10(obs_p), "o", ms=2.8, alpha=0.65, color="#1f77b4", rasterized=True)
        lim = max((-np.log10(exp_p)).max(), (-np.log10(obs_p)).max())
        lim = min(max(lim, 1.5), 6.0)
        ax.plot([0, lim], [0, lim], "--", lw=0.9, color="0.45")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        pretty = scenario_label.get(scen, scen)
        ax.set_title(pretty, fontsize=11.0, pad=8)
        ax.set_xlabel(r"Expected $-\log_{10}(p)$")
        ax.set_ylabel(r"Observed $-\log_{10}(p)$")
        ax.text(
            0.04,
            0.96,
            f"KS={row['ks_stat']:.3f}\nfrac(p<=0.05)={row['frac_below_0.05']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85),
        )

    fig.suptitle("LOCAT: Multiple-testing impact and null p-value calibration", fontsize=20, y=0.99, fontweight="bold")
    fig.savefig(f"{OUTDIR}/suppfig_multitest_calibration.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    # 3b) Figure: representative gene patterns for each null scenario.
    # Columns are lower-tail, median, and upper-tail p_final genes.
    fig_e, axes_e = plt.subplots(4, 3, figsize=(12.5, 12.5))
    scenario_order = [r["scenario"] for _, r in top_for_plot.iterrows()]
    col_labels = ["Low p_final", "Median p_final", "High p_final"]
    for c, lab in enumerate(col_labels):
        axes_e[0, c].set_title(lab, fontsize=11, pad=8)

    for r, scen in enumerate(scenario_order):
        genes_null = scenario_examples[scen]["genes_matrix"]
        df_null = scenario_examples[scen]["df"].sort_values("p_final").reset_index(drop=True)
        idx_low = 0
        idx_mid = int(0.50 * (len(df_null) - 1))
        idx_high = len(df_null) - 1
        pick = [idx_low, idx_mid, idx_high]

        for c, ridx in enumerate(pick):
            ax = axes_e[r, c]
            gene_name = df_null.loc[ridx, "gene"]
            gene_col = int(gene_name.split("_")[-1])
            expr = genes_null[:, gene_col] > 0
            pval = float(df_null.loc[ridx, "p_final"])
            n_expr = int(expr.sum())

            ax.scatter(coords[~expr, 0], coords[~expr, 1], c="#808080", s=3, alpha=0.35, linewidths=0, rasterized=True)
            ax.scatter(coords[expr, 0], coords[expr, 1], c="#d62728", s=10, alpha=0.85, linewidths=0, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if c == 0:
                row_lab = scenario_label.get(scen, scen)
                ax.set_ylabel(row_lab, fontsize=9, rotation=90)
            ax.text(
                0.02,
                0.98,
                f"n={n_expr}\np={pval:.3g}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9),
            )
            for sp in ax.spines.values():
                sp.set_visible(False)

    fig_e.suptitle("Representative null genes per QQ scenario", fontsize=16, y=0.995, fontweight="bold")
    fig_e.tight_layout(rect=[0.02, 0.02, 1.0, 0.975])
    fig_e.savefig(f"{OUTDIR}/suppfig_multitest_calibration_gene_examples.svg", format="svg", bbox_inches="tight")
    plt.close(fig_e)

    # 4) Short text report for manuscript/advisor discussion.
    with open(f"{OUTDIR}/locat_multitest_calibration_report.txt", "w", encoding="utf-8") as f:
        f.write("LOCAT multiple-testing + calibration summary\n")
        f.write("===========================================\n\n")
        f.write("[Mixed signal/null simulation]\n")
        for k, v in summary_mix.items():
            f.write(f"{k}: {v}\n")
        f.write("\n[Null calibration scenarios, sorted by best KS]\n")
        f.write(cal_df.to_string(index=False))
        f.write("\n\nInterpretation guide:\n")
        f.write("- If frac_below_0.05 is close to 0.05 under null, combined p-values are near-calibrated.\n")
        f.write("- If raw hits and BH hits are similar, BH correction minimally changes conclusions.\n")
        f.write("- If BH materially reduces hits, report both and keep BH-adjusted calls as primary.\n")

    print(f"Saved outputs to: {OUTDIR}")
    print("  - suppfig_multitest_calibration.svg")
    print("  - suppfig_multitest_calibration_gene_examples.svg")
    print("  - locat_multitest_mixed_signal_results.csv")
    print("  - locat_null_calibration_summary.csv")
    print("  - locat_multitest_calibration_report.txt")


if __name__ == "__main__":
    main()
