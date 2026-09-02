"""
Generate reproducible locat-0.1 duplicates of the 4 simulation supplementary
panels behind Supplementary Fig. S16 ("Effect of adjustment coefficients"),
i.e. the alpha_size / alpha_sens robustness figure referenced by
Supplementary Methods S1.4.2.

Outputs (written to support_files/):
  - suppfig_panelA_wsize_locat01_repro.svg
  - suppfig_panelB_wsens_locat01_repro.svg
  - suppfig_panelC_wsize_examples_locat01_repro.svg
  - suppfig_panelD_wsens_examples_locat01_repro.svg

Notes
-----
- This mirrors the earlier `generate_suppfig_panels.py` (internal-dev-locat
  version) but imports LOCAT from locat-0.1 and resets seeds before
  stochastic steps.
- locat-0.1 exposes `depletion_pval` rather than the newer
  `localization_pval`; this script uses `depletion_pval` as the localization
  counterpart when forming the Cauchy-combined baseline p-value.

Pre-submission fix (alpha_size robustness range)
--------------------------------------------------
The original Panel A grid was `w_size_vals = [0.0, 0.10, 0.15, 0.20]`,
compared against a reference of 0.15 -- alpha_size = 0.05, the actual
default used throughout the paper (Supplementary Table S1), was never
tested. This left the SI's stability claim ("Rankings remain stable across
alpha_size in [0.1, 0.2]") describing a range that excluded the value
actually used in every analysis.

Fixed here by adding 0.05 to the grid and using it (rather than 0.15) as
the comparison reference for the robustness panels, matching the updated
Supplementary Methods S1.4.2 text ("stable across alpha_size in [0.05,
0.20]"). The synthetic part of Panel A (lines computing `lp_s`, independent
of the spatial LOCAT run) was re-verified against the original grid before
this change: rho(ws=0.10, ws=0.15) = 0.9961 and rho(ws=0.20, ws=0.15) =
0.9981, matching the original SI's stated "rho > 0.996" -- confirming this
is a faithful extension of the original analysis, not a re-derivation.
Relative to the actual default (ws=0.05), rho stays above 0.96 across
[0.05, 0.20].
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

LOCAT01_PATH = Path(os.environ.get("LOCAT01_PATH", str(Path(__file__).resolve().parents[3].parent / "locat-0.1")))
if str(LOCAT01_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAT01_PATH))

# Ensure imports below come from locat-0.1 even in reused Python sessions.
for mod in list(sys.modules):
    if mod == "locat" or mod.startswith("locat."):
        del sys.modules[mod]

import anndata as ad
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from locat.locat import LOCAT
from scipy.stats import spearmanr
from sklearn.datasets import make_blobs


SEED = 13
OUTDIR = Path(__file__).resolve().parent / "support_files"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def reset_seeds(seed: int = SEED) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def cauchy_combine(pvals, p_floor=1e-300):
    def sf(p):
        x = float(pd.to_numeric(p, errors="coerce"))
        return float(np.clip(x, p_floor, 1.0)) if np.isfinite(x) else np.nan

    ps = np.array([sf(p) for p in pvals], dtype=float)
    w = np.ones(len(ps)) / len(ps)
    return sf(0.5 - np.arctan(np.sum(w * np.tan((0.5 - ps) * np.pi))) / np.pi)


def compute_pfinal(sres, ws_list, we_list, sens_power=1.25, p_floor=1e-300):
    sf = lambda p: np.clip(float(p), p_floor, 1.0)
    genes = list(sres.keys())
    p_cau = np.array([
        sf(cauchy_combine([sres[g]["localization_pval"], sres[g]["concentration_pval"]]))
        for g in genes
    ])
    n_arr = np.array([float(sres[g].get("sample_size", 0)) for g in genes])
    p_siz = np.array([sf(1.0 - np.exp(-1.0 / (n + 1.0))) for n in n_arr])
    sens = np.array([float(np.clip(sres[g].get("sens_score", 1.0), 1e-6, 1.0)) for g in genes])
    p_sen = np.array([sf((1.0 - s) ** sens_power) for s in sens])
    rows, cols = [], []
    for ws, we in zip(ws_list, we_list):
        pf = 1.0 - (1.0 - p_cau) * (1.0 - float(ws) * p_siz) * (1.0 - float(we) * p_sen)
        rows.append(np.clip(pf, p_floor, 1.0))
        cols.append(f"ws={float(ws):g}|we={float(we):g}")
    return pd.DataFrame(np.column_stack(rows), index=genes, columns=cols)


def create_anndata(matrix):
    adata = ad.AnnData(matrix.astype(np.float64))
    adata.obs_names = [f"Cell_{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i}" for i in range(adata.n_vars)]
    return adata


def run_locat(genes_arr, coords, seed: int):
    reset_seeds(seed)
    adata = create_anndata(genes_arr)
    adata.obsm["coords"] = coords.astype(np.float64)
    sc.pp.neighbors(adata, use_rep="coords", n_neighbors=30)
    model = LOCAT(
        adata,
        adata.obsm["coords"].astype(np.float64),
        20,
        show_progress=True,
        n_bootstrap_inits=100,
        knn=adata.obsp["connectivities"],
    )
    model.background_pdf(force_refresh=True)
    raw = model.gmm_scan(
        zscore_thresh=-np.inf,
        max_freq=1.0,
        rc_lambda_values=np.linspace(1.0, 2.0, 8),
        rc_min_abs_deficit=0.0,
        rc_min_expected=0.0,
        rc_min_p0_abs=0.0,
        rc_n_trials_cap=np.sqrt(adata.shape[0]),
        rc_n_eff_scale=0.9,
        include_depletion_scan=True,
    )

    sres = {}
    for gene, res in raw.items():
        row = res._asdict()
        row["localization_pval"] = float(row["depletion_pval"])
        sres[gene] = row
    return sres, adata


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def main():
    reset_seeds(SEED)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Simulation 1: w_size (locat-0.1 repro) ===")

    rng_syn = np.random.default_rng(42)
    n_tests = 200
    n_total_max, n_total_min = 200, 10
    n_totals_syn = np.round(np.geomspace(n_total_max, n_total_min, n_tests)).astype(int)
    p_cauchy_syn = 10 ** rng_syn.uniform(np.log10(2e-5), np.log10(0.35), n_tests)
    n_arr_syn = n_totals_syn.astype(float)
    p_siz_syn = 1.0 - np.exp(-1.0 / (n_arr_syn + 1.0))
    log_n = np.log10(n_arr_syn)

    # alpha_size = 0.05 is the manuscript default (Supplementary Table S1) and
    # is now the robustness reference point, replacing the earlier 0.15
    # reference that never included it (see module docstring).
    w_size_vals = [0.0, 0.05, 0.10, 0.15, 0.20]
    w_size_ref = 0.05
    p_floor = 1e-300
    lp_s = {}
    for ws in w_size_vals:
        pf = 1.0 - (1.0 - p_cauchy_syn) * (1.0 - float(ws) * p_siz_syn)
        lp_s[ws] = -np.log10(np.clip(pf, p_floor, 1.0))

    print(f"Spearman rho vs alpha_size={w_size_ref:g} (manuscript default):")
    for ws in w_size_vals:
        rho = spearmanr(lp_s[w_size_ref], lp_s[ws]).statistic
        print(f"  alpha_size={ws:g}: rho={rho:.4f}")

    rng = np.random.default_rng(0)
    coords, _, centers = make_blobs(
        n_samples=[5000],
        n_features=2,
        centers=None,
        return_centers=True,
        random_state=0,
        cluster_std=[1.5],
    )
    center0, radius0 = centers[0].copy(), 1.0
    b0, n_local_fixed = 0.05, 4
    n_totals = np.round(np.geomspace(n_total_max, n_total_min, n_tests)).astype(int)

    dists0 = np.sqrt(np.sum((coords - center0) ** 2, axis=1))
    bg_weight = np.exp(-(dists0 ** 2) / (2 * 1.5 ** 2))
    bg_weight /= bg_weight.sum()

    genes_size = np.zeros((coords.shape[0], n_tests), dtype=np.uint8)
    for i, n_total in enumerate(n_totals):
        radius_i = radius0 * rng.uniform(0.85, 1.15)
        center_i = center0 + rng.normal(scale=0.10, size=2)
        dists_i = np.sqrt(np.sum((coords - center_i) ** 2, axis=1))
        in_r = np.flatnonzero(dists_i < radius_i)
        out_r = np.flatnonzero(dists_i >= radius_i)
        nl = min(n_local_fixed, n_total)
        no = min(int(round(b0 * nl)), len(out_r))
        ni = nl - no
        na = n_total - nl
        wi = bg_weight[in_r]
        wi /= wi.sum()
        wo = bg_weight[out_r]
        wo /= wo.sum()
        ii = rng.choice(in_r, ni, replace=False, p=wi) if ni > 0 else np.array([], int)
        io = rng.choice(out_r, no, replace=False, p=wo) if no > 0 else np.array([], int)
        ia = (
            rng.choice(coords.shape[0], na, replace=False, p=bg_weight)
            if na > 0 else np.array([], int)
        )
        pos = np.unique(np.concatenate([ii, io, ia]))
        if pos.size < n_total:
            rem = np.setdiff1d(np.arange(coords.shape[0]), pos)
            rw = bg_weight[rem] / bg_weight[rem].sum()
            pos = np.concatenate([pos, rng.choice(rem, n_total - pos.size, replace=False, p=rw)])
        genes_size[pos, i] = 1

    print("Running LOCAT (for Panel C embeddings)...")
    sres_s, _ = run_locat(genes_size, coords, seed=SEED)
    n_arr_s = np.asarray(genes_size.sum(axis=0)).ravel()
    for i, g in enumerate(sres_s):
        sres_s[g]["sample_size"] = float(n_arr_s[i])

    print("Generating Panel A...")
    norm_a = mcolors.Normalize(vmin=log_n.min(), vmax=log_n.max())
    panels_a = [
        (0.0, w_size_ref, rf"$\alpha_{{\mathrm{{size}}}}={w_size_ref:g}$ effect"),
        (w_size_ref, 0.10, r"Robustness: 0.10 vs 0.05"),
        (w_size_ref, 0.20, r"Robustness: 0.20 vs 0.05"),
    ]

    fig_a, axes_a = plt.subplots(1, 3, figsize=(15.0, 5.2), constrained_layout=False)
    for ax, (xw, yw, title) in zip(axes_a, panels_a):
        x_vals = lp_s[xw]
        y_vals = lp_s[yw]
        rho = spearmanr(x_vals, y_vals).statistic
        order = np.argsort(log_n)
        ax.scatter(
            x_vals[order],
            y_vals[order],
            c=log_n[order],
            cmap=plt.cm.viridis,
            norm=norm_a,
            s=22,
            alpha=0.85,
            linewidths=0,
            rasterized=False,
        )
        lim = min(max(x_vals.max(), y_vals.max()) * 1.08, 6.0)
        lim = max(lim, 0.4)
        ax.plot([0, lim], [0, lim], "--", color="0.55", lw=0.8, zorder=0)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(rf"$-\log_{{10}}(p)$,  $\alpha_{{\mathrm{{size}}}}={xw:g}$")
        ax.set_ylabel(rf"$-\log_{{10}}(p)$,  $\alpha_{{\mathrm{{size}}}}={yw:g}$")
        ax.set_title(title)
        ax.text(
            0.04,
            0.96,
            f"ρ = {rho:.4f}",
            transform=ax.transAxes,
            fontsize=17,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85),
        )

    sm_a = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm_a)
    sm_a.set_array([])
    fig_a.tight_layout(rect=[0, 0, 0.93, 1])
    cax_a = fig_a.add_axes([0.94, 0.12, 0.015, 0.76])
    cb_a = fig_a.colorbar(sm_a, cax=cax_a)
    cb_a.set_label("Sample size", fontsize=18)
    tv = np.linspace(log_n.min(), log_n.max(), 5)
    cb_a.set_ticks(tv)
    cb_a.set_ticklabels([f"{10 ** v:.0f}" for v in tv])
    cb_a.ax.tick_params(labelsize=17)
    fig_a.suptitle(r"$\alpha_{\mathrm{size}}$ effect and robustness", fontsize=21, y=1.02, fontweight="bold")
    fig_a.savefig(OUTDIR / "suppfig_panelA_wsize_locat01_repro.svg", format="svg", bbox_inches="tight")
    plt.close(fig_a)

    print("Generating Panel C (embedding examples, w_size)...")
    df_raw_s = pd.DataFrame(sres_s).T
    cp_s = df_raw_s["concentration_pval"].astype(float).clip(1e-300).to_numpy()
    lp_loc_s = df_raw_s["localization_pval"].astype(float).clip(1e-300).to_numpy()
    p_cau_s = np.array([
        float(np.clip(cauchy_combine([float(cp_s[j]), float(lp_loc_s[j])]), 1e-300, 1.0))
        for j in range(len(cp_s))
    ])
    lp_chosen_s = -np.log10(p_cau_s)

    target_ns = [10, 25, 60, 130, 220]
    selected_c = []
    last_n = -1
    used = set()
    for tgt in target_ns:
        dist_from_tgt = np.abs(n_arr_s - tgt)
        mask = dist_from_tgt <= max(2, int(tgt * 0.15))
        if not mask.any():
            mask = dist_from_tgt == dist_from_tgt.min()
        candidates = np.where(mask)[0]
        candidates = np.array([c for c in candidates if c not in used and int(n_arr_s[c]) > last_n], dtype=int)
        if candidates.size == 0:
            fallback = np.argsort(dist_from_tgt)
            candidates = np.array([c for c in fallback if c not in used and int(n_arr_s[c]) > last_n], dtype=int)
        if candidates.size == 0:
            continue
        best = candidates[np.argmax(lp_chosen_s[candidates])]
        selected_c.append(best)
        used.add(int(best))
        last_n = int(n_arr_s[best])

    pad = 0.5
    c_xlim = (coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    c_ylim = (coords[:, 1].min() - pad, coords[:, 1].max() + pad)

    fig_c, axes_c = plt.subplots(1, len(selected_c), figsize=(4.0 * len(selected_c), 4.0))
    if len(selected_c) == 1:
        axes_c = [axes_c]
    for ax, gene_idx in zip(axes_c, selected_c):
        expr_mask = genes_size[:, gene_idx] > 0
        n_expr = int(expr_mask.sum())
        ax.scatter(coords[~expr_mask, 0], coords[~expr_mask, 1], c="#444444", s=4, alpha=0.85, linewidths=0, rasterized=True)
        ax.scatter(
            coords[expr_mask, 0],
            coords[expr_mask, 1],
            c="#e31a1c",
            s=55,
            alpha=0.92,
            linewidths=0.4,
            edgecolors="#800000",
            zorder=3,
            rasterized=True,
        )
        ax.set_xlim(*c_xlim)
        ax.set_ylim(*c_ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"$n = {n_expr}$", fontsize=18)

    fig_c.suptitle(r"Example genes: $\alpha_{\mathrm{size}}$ simulation (locat-0.1)", fontsize=20, fontweight="bold", y=1.04)
    fig_c.tight_layout()
    fig_c.savefig(OUTDIR / "suppfig_panelC_wsize_examples_locat01_repro.svg", format="svg", bbox_inches="tight")
    plt.close(fig_c)

    print("\n=== Simulation 2: w_sens (locat-0.1 repro) ===")

    rng2 = np.random.default_rng(1)
    coords2, _, centers2 = make_blobs(
        n_samples=[5000],
        n_features=2,
        centers=None,
        return_centers=True,
        random_state=0,
        cluster_std=[1.0],
    )
    center2, radius2 = centers2[0].copy(), 0.5
    n_total2, n_tests2 = 50, 200
    fractions_in = np.linspace(1.0, 0.3, n_tests2)
    frac_out_arr = 1.0 - fractions_in

    dists2 = np.sqrt(np.sum((coords2 - center2) ** 2, axis=1))
    in_r2 = np.flatnonzero(dists2 < radius2)
    out_r2 = np.flatnonzero(dists2 >= radius2)

    genes_sens = np.zeros((coords2.shape[0], n_tests2), dtype=np.uint8)
    for i, fi in enumerate(fractions_in):
        ni = min(int(round(fi * n_total2)), len(in_r2))
        no = min(n_total2 - ni, len(out_r2))
        ii = rng2.choice(in_r2, ni, replace=False) if ni > 0 else np.array([], int)
        io = rng2.choice(out_r2, no, replace=False) if no > 0 else np.array([], int)
        genes_sens[np.concatenate([ii, io]), i] = 1

    print("Running LOCAT (simulation 2)...")
    sres_w, _ = run_locat(genes_sens, coords2, seed=SEED)
    for i, g in enumerate(sres_w):
        sres_w[g]["sample_size"] = float(n_total2)
        sres_w[g]["sens_score"] = float(fractions_in[i])

    w_sens_vals = [0.0, 0.05, 0.15, 0.30]
    df_w = compute_pfinal(sres_w, [0.0] * 4, w_sens_vals)
    lp_dict = {
        we: -np.log10(df_w[f"ws=0|we={float(we):g}"].astype(float).clip(1e-300).to_numpy())
        for we in w_sens_vals
    }

    print("Generating Panel B...")
    panels_b = [
        (0.0, 0.15, r"$\alpha_{\mathrm{sens}}=0.15$ effect"),
        (0.15, 0.05, r"Robustness: 0.05 vs 0.15"),
        (0.15, 0.30, r"Robustness: 0.30 vs 0.15"),
    ]
    norm_b = mcolors.Normalize(vmin=frac_out_arr.min(), vmax=frac_out_arr.max())

    fig_b, axes_b = plt.subplots(1, 3, figsize=(15.0, 5.2), constrained_layout=False)
    for ax, (xwe, ywe, title) in zip(axes_b, panels_b):
        x = lp_dict[xwe]
        y = lp_dict[ywe]
        rho = spearmanr(x, y).statistic
        ax.scatter(x, y, c=frac_out_arr, cmap=plt.cm.plasma, norm=norm_b, s=22, alpha=0.80, linewidths=0, rasterized=False)
        lim = max(x.max(), y.max()) * 1.08
        lim = max(lim, 0.4)
        ax.plot([0, lim], [0, lim], "--", color="0.55", lw=0.8, zorder=0)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$-\log_{10}(p)$,  $\alpha_{\mathrm{sens}}=" + f"{xwe:g}$")
        ax.set_ylabel(r"$-\log_{10}(p)$,  $\alpha_{\mathrm{sens}}=" + f"{ywe:g}$")
        ax.set_title(title)
        ax.text(
            0.04,
            0.96,
            f"ρ = {rho:.4f}",
            transform=ax.transAxes,
            fontsize=17,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85),
        )

    sm_b = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm_b)
    sm_b.set_array([])
    fig_b.tight_layout(rect=[0, 0, 0.93, 1])
    cax_b = fig_b.add_axes([0.94, 0.12, 0.015, 0.76])
    cb_b = fig_b.colorbar(sm_b, cax=cax_b)
    cb_b.set_label("Bleed fraction", fontsize=18)
    cb_b.ax.tick_params(labelsize=17)
    fig_b.suptitle(r"$\alpha_{\mathrm{sens}}$ effect and robustness", fontsize=21, y=1.02, fontweight="bold")
    fig_b.savefig(OUTDIR / "suppfig_panelB_wsens_locat01_repro.svg", format="svg", bbox_inches="tight")
    plt.close(fig_b)

    print("Generating Panel D (embedding examples, w_sens)...")
    pf_chosen_w = df_w["ws=0|we=0.15"].astype(float).clip(1e-300).to_numpy()
    lp_chosen_w = -np.log10(pf_chosen_w)
    target_bleeds = [0.05, 0.20, 0.35, 0.52, 0.68]
    selected_d = [int(np.argmin(np.abs(frac_out_arr - tgt))) for tgt in target_bleeds]

    fig_d, axes_d = plt.subplots(1, len(target_bleeds), figsize=(4.0 * len(target_bleeds), 4.0))
    for ax, gene_idx in zip(axes_d, selected_d):
        expr_mask = genes_sens[:, gene_idx] > 0
        bleed_val = frac_out_arr[gene_idx]
        lp_val = lp_chosen_w[gene_idx]
        ax.scatter(coords2[~expr_mask, 0], coords2[~expr_mask, 1], c="#444444", s=4, alpha=0.85, linewidths=0, rasterized=True)
        ax.scatter(
            coords2[expr_mask, 0],
            coords2[expr_mask, 1],
            c="#e31a1c",
            s=28,
            alpha=0.92,
            linewidths=0.4,
            edgecolors="#800000",
            zorder=3,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"bleed = {bleed_val * 100:.0f}%\n$-\\log_{{10}}(p) = {lp_val:.2f}$", fontsize=18)

    fig_d.suptitle(r"Example genes: $\alpha_{\mathrm{sens}}$ simulation (locat-0.1)", fontsize=20, fontweight="bold", y=1.04)
    fig_d.tight_layout()
    fig_d.savefig(OUTDIR / "suppfig_panelD_wsens_examples_locat01_repro.svg", format="svg", bbox_inches="tight")
    plt.close(fig_d)

    print("\nAll 4 locat-0.1 supplementary panels done.")


if __name__ == "__main__":
    main()
