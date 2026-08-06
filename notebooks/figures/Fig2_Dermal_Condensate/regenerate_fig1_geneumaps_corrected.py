"""Regenerates the Fig. 2.1 / Fig. 1B gene-UMAP panel with the genes the
caption and body text actually describe and cite p-values for.

Context: the figure previously committed to the dissertation and locat
manuscript (figures/localization_example_new_genes.png) showed a different
gene panel (Postn, Mllt3, Kif2c, Hist1h2bn) than the one the surrounding
text discusses (H19, Nnat, Sox2, Hist1h2bb), a stale-link bug rather than a
labeling typo. This script reproduces the correct panel from data and code
already tracked in this repo:

  - Gene UMAP feature-plot helper: identical to
    DermalC_pca-runlocat_2_cauchy-021826_locat01_repro.ipynb cell 38
    (plot_gene_umap_expressing_on_top).
  - Score table: notebooks/figures/Fig2_Dermal_Condensate/support_files/
    locat_df_DC14_5_2_locat01_repro.pkl (already committed, no Locat re-fit
    needed).
  - Processed AnnData: data/E145_dermal_erez_2026/dc_adata_proc_rep.h5ad.
    This is gitignored (*.h5ad) like all raw/processed data in this repo;
    it must be present locally to re-run this script but is not itself
    tracked. Source: the same file used throughout the DermalC_pca-*
    notebooks.

The dissertation prose cites these exact p-values for this panel
(Sec. 2.3.1): Sox2 depletion p < 1e-12, localization p = 9.89e-3;
Hist1h2bb localization p = 2.48e-2; H19 depletion p = 0.140, localization
p = 0.089; and (after fixing a "Nfib" -> "Nnat" mislabel found alongside
this figure bug) Nnat depletion p = 0.103, localization p = 0.082. Running
this script reprints those numbers from the committed score table so they
can be checked against the dissertation text directly.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "../../../data/E145_dermal_erez_2026/"
SCORE_PKL = "support_files/locat_df_DC14_5_2_locat01_repro.pkl"
GENES = ["H19", "Nnat", "Sox2", "Hist1h2bb"]


def get_gene_x(adata, gene, layer=None):
    x = adata[:, gene].X if layer is None else adata[:, gene].layers[layer]
    if sp.issparse(x):
        x = x.toarray().ravel()
    else:
        x = np.asarray(x).ravel()
    return x


def plot_gene_umap_expressing_on_top(
    adata, gene, ax,
    expr_thresh=0.0, q_vmax=50, layer=None,
    s_bg=3, s_expr=6, alpha_bg=0.6, alpha_expr=0.9,
    use_log1p=False, title_fs=22, cmap_genes="Reds",
):
    X = adata.obsm["X_umap"]
    x = get_gene_x(adata, gene, layer=layer)
    x_plot = np.clip(x, 0, None)
    expr = x_plot > expr_thresh
    ax.scatter(X[:, 0], X[:, 1], c="gray", s=s_bg, alpha=alpha_bg, linewidths=0)
    xe = x_plot[expr]
    vmax = np.percentile(xe, q_vmax) if xe.size > 0 else 1.0
    ax.scatter(
        X[expr, 0], X[expr, 1], c=np.clip(xe, 0, vmax), s=s_expr,
        alpha=alpha_expr, cmap=cmap_genes, vmin=0, vmax=vmax, linewidths=0,
    )
    ax.set_title(gene, fontsize=title_fs, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    adata_train = sc.read_h5ad(DATA_DIR + "dc_adata_proc_rep.h5ad")
    locat_df = pd.read_pickle(SCORE_PKL)

    for g in GENES:
        row = locat_df.loc[g]
        print(f"{g}: depletion_pval={row['depletion_pval']:.6g}  localization_pval={row['pval']:.6g}")

    fig = plt.figure(figsize=(16, 4.3))
    gs = fig.add_gridspec(1, len(GENES) + 1, width_ratios=[0.35] + [1] * len(GENES), wspace=0.05)

    ax_label = fig.add_subplot(gs[0, 0])
    ax_label.axis("off")
    ax_label.text(0.0, 0.5, "B.", fontsize=40, fontweight="bold", va="center", ha="left")

    for i, gene in enumerate(GENES):
        ax = fig.add_subplot(gs[0, i + 1])
        plot_gene_umap_expressing_on_top(
            adata_train, gene, ax=ax,
            expr_thresh=0.0, q_vmax=60, s_bg=10, s_expr=15, cmap_genes="Reds",
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    out_path = "panelB_geneumaps_corrected.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
