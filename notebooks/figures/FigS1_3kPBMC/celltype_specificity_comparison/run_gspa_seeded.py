"""Run GSPA with a given seed and data path. Called by run_multiseed.py."""
import argparse, os, sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True)
parser.add_argument("--out_path", required=True)
parser.add_argument("--seed", type=int, default=13)
parser.add_argument("--gpu", type=str, default="2")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
np.random.seed(args.seed)

import scanpy as sc, gspa, scipy.sparse as sp

adata = sc.read_h5ad(args.data_path)
print(f"Loaded: {adata}", flush=True)

X = adata.X.toarray() if sp.issparse(adata.X) else adata.X.astype("float64")

# random_state seeds the internal PCA used by graphtools.Graph (n_pca=100)
gspa_op = gspa.GSPA(random_state=args.seed)
gspa_op.construct_graph(X)
gspa_op.build_diffusion_operator()
gspa_op.build_wavelet_dictionary()

gene_signals = X.T
gene_ae, gene_pc = gspa_op.get_gene_embeddings(gene_signals)
gene_localization = gspa_op.calculate_localization()
print("GSPA done", flush=True)

np.savez(args.out_path, var_names=np.array(adata.var_names), gene_localization=gene_localization)
print(f"Saved to {args.out_path}", flush=True)
