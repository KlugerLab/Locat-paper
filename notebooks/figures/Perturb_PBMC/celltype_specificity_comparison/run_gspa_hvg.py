"""Run GSPA on 3k PBMC restricted to 1838 HVGs."""
import os, sys, numpy as np, scanpy as sc, scipy.sparse as sp

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
SEED = 13
np.random.seed(SEED)

import gspa, phate

DATA_PATH = "/banach2/wes/Locat-paper-repro-private/data/pbmc3k_hvg_lognorm.h5ad"
OUT_PATH  = "/banach2/wes/Locat-paper-repro-private/notebooks/figures/Perturb_PBMC/celltype_specificity_comparison/gspa_scores_hvg.npz"

adata = sc.read_h5ad(DATA_PATH)
X = adata.X.toarray() if sp.issparse(adata.X) else adata.X.astype("float64")
print(f"Loaded: {adata}", flush=True)

phate_op = phate.PHATE(n_components=2, random_state=SEED, verbose=0)
embedding = phate_op.fit_transform(X)
print("PHATE done", flush=True)

gspa_op = gspa.GSPA()
gspa_op.construct_graph(X)
gspa_op.build_diffusion_operator()
gspa_op.build_wavelet_dictionary()
gene_ae, gene_pc = gspa_op.get_gene_embeddings(X.T)
gene_localization = gspa_op.calculate_localization()
print("GSPA done", flush=True)

np.savez(OUT_PATH, var_names=np.array(adata.var_names), gene_localization=gene_localization)
print(f"Saved {OUT_PATH}", flush=True)
