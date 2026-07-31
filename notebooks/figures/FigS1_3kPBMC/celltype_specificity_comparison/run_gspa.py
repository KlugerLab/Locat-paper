"""Run GSPA on 3k PBMC (9543-gene AUC benchmark gene set) and save gene localization scores."""
import os, sys, numpy as np, scanpy as sc
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
SEED = 13
np.random.seed(SEED)

import gspa, phate

DATA_PATH = str(Path(__file__).resolve().parents[4] / "data/pbmc3k_9543_lognorm.h5ad")
OUT_PATH  = str(Path(__file__).resolve().parents[4] / "notebooks/figures/Perturb_PBMC/celltype_specificity_comparison/gspa_scores.npz")

adata = sc.read_h5ad(DATA_PATH)
print(f"Loaded: {adata}", flush=True)

import scipy.sparse as sp
X = adata.X.toarray() if sp.issparse(adata.X) else adata.X.astype("float64")

phate_op = phate.PHATE(n_components=2, random_state=SEED, verbose=0)
embedding = phate_op.fit_transform(X)
print("PHATE done", flush=True)

gspa_op = gspa.GSPA()
gspa_op.construct_graph(X)
gspa_op.build_diffusion_operator()
gspa_op.build_wavelet_dictionary()

gene_signals = X.T
gene_ae, gene_pc = gspa_op.get_gene_embeddings(gene_signals)
gene_localization = gspa_op.calculate_localization()
print("GSPA done", flush=True)

np.savez(OUT_PATH, var_names=np.array(adata.var_names), gene_localization=gene_localization)
print(f"Saved to {OUT_PATH}", flush=True)
