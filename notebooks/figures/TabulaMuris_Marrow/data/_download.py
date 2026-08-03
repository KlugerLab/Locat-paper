import time
from pathlib import Path
t0 = time.time()
import cellxgene_census
print("importing done", time.time()-t0, flush=True)

with cellxgene_census.open_soma() as census:
    print("opened soma", time.time()-t0, flush=True)
    adata = cellxgene_census.get_anndata(
        census,
        organism="Mus musculus",
        obs_value_filter="dataset_id == '0bd1a1de-3aee-40e0-b2ec-86c7a30c7149'",
        obs_column_names=["cell_type", "development_stage", "sex", "donor_id"],
        var_column_names=["feature_id", "feature_name"],
    )
print("fetched anndata", time.time()-t0, flush=True)
print(adata, flush=True)
import scipy.sparse as sp
print("X type:", type(adata.X), "dtype:", adata.X.dtype, flush=True)
if sp.issparse(adata.X):
    print("X nnz:", adata.X.nnz, "shape:", adata.X.shape, flush=True)
OUT_PATH = Path(__file__).parent / "tabula_muris_marrow_raw.h5ad"
adata.write_h5ad(OUT_PATH)
print(f"saved to {OUT_PATH}", time.time()-t0, flush=True)
