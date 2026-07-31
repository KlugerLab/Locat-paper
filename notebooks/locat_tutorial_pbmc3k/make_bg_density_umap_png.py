from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import anndata as ad

from locat.locat import LOCAT
from locat.preprocessing import filter_genes, get_embedding


def main():
    repo = Path(__file__).resolve().parents[2]
    data_dir = repo / 'data'
    out_png = repo / 'notebooks' / 'locat_tutorial_pbmc3k' / 'pbmc3k_umap_locat_bg_density_pca8.png'

    adata_processed = ad.read_h5ad(data_dir / 'pbmc3k_processed.h5ad')
    adata = adata_processed.raw.to_adata()

    adata.obsm['X_pca'] = adata_processed.obsm['X_pca'].copy()
    adata.obsm['X_umap'] = adata_processed.obsm['X_umap'].copy()

    adata_locat = filter_genes(adata)

    model = LOCAT(
        adata=adata_locat,
        cell_embedding=get_embedding(adata_locat, n_dims=8),
        n_bootstrap_inits=10,
        show_progress=True,
        knn_mode='connectivity',
        reg_covar=1e-6,
    )

    f0 = model.background_pdf(weights_transform=None)
    f0 = np.asarray(f0).ravel()
    log_f0 = np.log10(f0 + 1e-12)

    umap = np.asarray(adata_locat.obsm['X_umap'])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    sca = ax.scatter(
        umap[:, 0],
        umap[:, 1],
        c=log_f0,
        s=7,
        cmap='viridis',
        linewidths=0,
        alpha=0.95,
    )
    ax.set_title('PBMC3K UMAP with LOCAT background density (fit in PCA-8)')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log10 background density f0(x)')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    print(f'SAVED_PNG={out_png}')


if __name__ == '__main__':
    main()
