# Tabula Muris Bone Marrow — Locat Cell-Type Specificity Comparison

Add Tabula Muris bone marrow as a fifth real-dataset comparison in the Locat paper,
matching the style of DermalC, Kang Stim/Ctrl, and PBMC3k.

---

## 0. Directory layout to create

```
Locat-paper-repro-private/notebooks/figures/TabulaMuris_Marrow/
├── data/                         ← raw/processed h5ad lives here
├── scores/                       ← all method score npz files
│   ├── locat_scores.npz
│   ├── gspa_scores.npz
│   ├── lmd_scores.npz
│   ├── hotspot_scores.npz
│   ├── haystack_scores.npz
│   └── scanpy_scores.npz
└── real_dataset/                 ← all output figures
    ├── real_dataset_dotplot_top50.svg
    ├── real_dataset_dotplot_top100.svg
    ├── real_dataset_dotplot_top200.svg
    ├── umap_context.svg
    └── top10_umap_<method>.svg   (one per method, 6 total)
```

---

## 1. Data acquisition

### Option A — CZI cellxgene Census (recommended, simplest)

The CZI Census Python API gives direct h5ad access to Tabula Muris with
cell-type annotations already included.

```bash
/banach2/wes/.conda/envs/mulde_jax/bin/pip install cellxgene-census
```

```python
import cellxgene_census
with cellxgene_census.open_soma() as census:
    adata = cellxgene_census.get_anndata(
        census,
        organism="Mus musculus",
        obs_value_filter="tissue == 'bone marrow' and dataset_title == 'Tabula Muris'",
        var_value_filter="feature_biotype == 'gene'",
    )
adata.write_h5ad("data/tabula_muris_marrow_raw.h5ad")
```

Cell type column in Census output: `cell_type` (free-text label from the original
publication, e.g. "B cell", "T cell", "monocyte", "macrophage", "erythrocyte",
"natural killer cell", "granulocyte", "hematopoietic precursor cell",
"megakaryocyte").

### Option B — Figshare processed Seurat object → h5ad (fallback)

The Tabula Muris paper deposited processed Seurat `.rds` objects on figshare
(DOI 10.6084/m9.figshare.5821263). Download `marrow_10X_P7_8.rds` and convert:

```python
import rpy2.robjects as ro
ro.r('library(Seurat); obj <- readRDS("marrow_10X_P7_8.rds")')
# export counts + metadata via rpy2, then build AnnData manually
```

### Option C — GEO GSE109774 raw 10x matrices

GEO hosts per-tissue tar.gz files of 10x Cell Ranger output (matrix.mtx +
barcodes + genes). The Marrow tar is at:
```
ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE109nnn/GSE109774/suppl/GSE109774_Marrow.tar.gz
```
**Note**: This GEO submission is actually SMART-seq2 FACS plate data (individual
cell CSVs, one per cell). The 10x matrices are in the RAW tar or in a separate
accession (GSE132042 for Tabula Muris Senis, which also has bone marrow with
10x and includes cell type labels). Prefer Option A or B.

### Expected cell counts / composition
Tabula Muris bone marrow 10x data has ~25,000 cells from multiple mice.
After QC expect ~15,000–20,000 cells, ~15,000–20,000 genes before filtering.
Key cell types (≥100 cells each): B cell, T cell, NK cell, monocyte,
macrophage, erythrocyte, granulocyte, HSC, megakaryocyte.
This gives well-separated, known-biology clusters — good for τ comparison.

---

## 2. Preprocessing

Mirror the DermalC/Kang pipeline exactly. Python env:
`/banach2/wes/.conda/envs/mulde_jax/bin/python`

```python
import scanpy as sc, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path

HERE = Path("Locat-paper-repro-private/notebooks/figures/TabulaMuris_Marrow")

adata = sc.read_h5ad(HERE / "data/tabula_muris_marrow_raw.h5ad")

# ── QC ────────────────────────────────────────────────────────────────────────
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_cells(adata, max_genes=6000)   # rough doublet proxy
sc.pp.filter_genes(adata, min_cells=3)

# ── Normalize + log1p ─────────────────────────────────────────────────────────
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ── Gene filter ≥5% ───────────────────────────────────────────────────────────
pct = (adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)) > 0
adata = adata[:, pct.mean(axis=0) >= 0.05].copy()

# ── Cell type column ──────────────────────────────────────────────────────────
# Column name depends on source:
#   Census: "cell_type"
#   Figshare Seurat: "cell_ontology_class"
CELLTYPE_COL = "cell_type"   # adjust if needed
adata.obs[CELLTYPE_COL] = pd.Categorical(adata.obs[CELLTYPE_COL])

# Drop cell types with fewer than 20 cells (too small for meaningful τ)
ct_counts = adata.obs[CELLTYPE_COL].value_counts()
keep_cts = ct_counts[ct_counts >= 20].index
adata = adata[adata.obs[CELLTYPE_COL].isin(keep_cts)].copy()
adata.obs[CELLTYPE_COL] = pd.Categorical(adata.obs[CELLTYPE_COL])

print(f"After QC: {adata.n_obs} cells × {adata.n_vars} genes")
print(adata.obs[CELLTYPE_COL].value_counts())

# ── PCA + neighbors ───────────────────────────────────────────────────────────
sc.pp.pca(adata, n_comps=50, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)
sc.tl.umap(adata)

adata.write_h5ad(HERE / "data/tabula_muris_marrow_proc.h5ad")
```

---

## 3. Run all 6 methods

Write `run_tabulamuris_comparison.py` in
`Locat-paper-repro-private/notebooks/figures/TabulaMuris_Marrow/`.
Model it on `Perturb_PBMC/celltype_specificity_comparison/run_kang_comparison.py`
and `Chlamydomonas/run_chlamydomonas_comparison.py`.

Key constants:
```python
LOCAT_SRC    = Path("/banach2/wes/locat-0.1")
GSPA_PYTHON  = "/banach2/wes/.conda/envs/gspa-env/bin/python"
LMD_PYTHON   = "/banach2/wes/envs/lmd_rpy2/bin/python"
LMD_SCRIPT   = Path("/banach2/wes/Locat-paper-repro-private/notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison/run_lmd_seeded.py")
GSPA_SCRIPT  = Path("/banach2/wes/Locat-paper-repro-private/notebooks/figures/FigS1_3kPBMC/celltype_specificity_comparison/run_gspa_seeded.py")
CELLTYPE_COL = "cell_type"   # match preprocessing above
DATA_H5AD    = HERE / "data/tabula_muris_marrow_proc.h5ad"
SCORES_DIR   = HERE / "scores"
PCT_THRESH   = 0.05
```

Method parameters matching the rest of the paper:
- **Locat**: `embedding = X_pca[:, :8]`, `k=20`, `n_bootstrap_inits=50`,
  `_reg_covar=1e-6`, `max_freq=0.9`, `include_depletion_scan=True`,
  `rc_lambda_values=np.linspace(1.0, 2.0, 8)`
- **GSPA**: via subprocess with `--seed 0 --gpu 0`
- **LMD**: via subprocess with `R_HOME=/banach2/wes/envs/lmd_rpy2/lib/R`
- **Hotspot**: `model="normal"`, `latent_obsm_key="X_pca"`, `n_neighbors=30`
- **Haystack**: `sc.tl.haystack(adata, coord="pca")` via singleCellHaystack
- **Scanpy**: Leiden `resolution=0.5`, `rank_genes_groups(method="wilcoxon", use_raw=False)`,
  rank by min adjusted p-value across groups

τ computation (≥5% expressing cells):
```python
def compute_tau(adata, celltype_col, pct_thresh=0.05):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    cts = adata.obs[celltype_col].cat.categories.tolist()
    mean = np.array([X[adata.obs[celltype_col] == ct].mean(axis=0) for ct in cts])
    pct  = (X > 0).mean(axis=0)
    rs   = mean.sum(axis=0)
    mask = (rs > 0) & (pct >= pct_thresh)
    tau  = np.where(mask, mean.max(axis=0) / np.where(rs > 0, rs, 1.0), np.nan)
    return pd.Series(tau, index=adata.var_names)
```

Save all scores as `.npz` into `scores/` using the same key names as all
other datasets:
- Locat: `gene_names`, `pval`
- GSPA: `var_names`, `gene_localization`
- LMD: `var_names`, `lmd_score`
- Hotspot: `var_names`, `fdr`
- Haystack: `var_names`, `logpval`
- Scanpy: `var_names`, `pval_min`

---

## 4. Generate figures

Write `plot_tabulamuris_comparison.py` in the same folder.
Model it on `Chlamydomonas/plot_chlamydomonas_comparison.py` and
`plot_real_dataset_comparison.py`.

### 4a. Dot plots (matching existing paper style)

For cutoffs `[50, 100, 200]`, produce `real_dataset/real_dataset_dotplot_top{k}.svg`:
- Y axis: mean τ of top-k genes ± 95% CI (1.96 × SD / √k)
- Grey band: all-genes IQR (25th–75th percentile of τ across all genes)
- Dashed grey line: all-genes median τ
- Methods sorted descending by mean τ
- Dot color from `COLORS` dict (Locat=#e6194b, GSPA=#3cb44b, LMD=#4363d8,
  Hotspot=#f58231, Haystack=#911eb4, Scanpy=#f032e6)
- Figure size: (7, 4.5)

### 4b. Context UMAP

`real_dataset/umap_context.svg` — two panels side by side:
- Left: UMAP colored by cell type (categorical, tab20 palette), legend showing
  all cell types with counts
- Right: UMAP colored by a continuous marker if available, else a second
  interesting obs column

### 4c. Top-10 gene UMAPs per method

For each of the 6 methods, produce `real_dataset/top10_umap_<method>.svg`:
- 2 rows × 5 columns grid (figsize ~18 × 7)
- Each panel: UMAP colored by log-normalized expression of one gene
  - Zero-expressing cells in light grey (s=0.5, alpha=0.15)
  - Expressing cells colored by YlOrRd, vmax = 95th percentile of non-zero
    expression
  - Panel title: short gene name + (XX% cells)
- Figure title: method name in method color

---

## 5. Verify output quality

Before finishing, print and check:
1. Cell type distribution — confirm ≥5 cell types with ≥100 cells each
2. τ all-genes median — expect ~0.3–0.5 (bone marrow has well-separated types,
   so baseline should be higher than Chlamydomonas ~0.25)
3. Locat top-50 mean τ — expect ≥0.6 if the dataset is good
4. Overlap of Locat top-50 with known marker genes:
   - B cells: Cd19, Ms4a1, Cd79a, Cd79b, Ebf1
   - T/NK: Cd3e, Cd8a, Cd4, Nkg7, Gzma
   - Monocyte/Macro: Lyz2, Csf1r, Cd68, S100a8
   - Erythroid: Hba-a1, Hbb-bt, Gypa
   If ≥3 of these appear in Locat's top-50, the pipeline is working correctly.

---

## 6. Execution order

```bash
cd /banach2/wes
/banach2/wes/.conda/envs/mulde_jax/bin/python \
  Locat-paper-repro-private/notebooks/figures/TabulaMuris_Marrow/run_tabulamuris_comparison.py \
  --gpu 0 2>&1 | tee TabulaMuris_Marrow/run.log

/banach2/wes/.conda/envs/mulde_jax/bin/python \
  Locat-paper-repro-private/notebooks/figures/TabulaMuris_Marrow/plot_tabulamuris_comparison.py \
  2>&1 | tee TabulaMuris_Marrow/plot.log
```

Expected runtime: ~25–40 min total (GSPA dominates at ~20 min for ~15k cells).

---

## 7. Git

After confirming figures look good:
```bash
git -C Locat-paper-repro-private add notebooks/figures/TabulaMuris_Marrow/
git -C Locat-paper-repro-private commit -m "Add Tabula Muris bone marrow cell-type specificity comparison"
```

Do NOT commit the raw/processed h5ad files (they are large); add to `.gitignore`:
```
notebooks/figures/TabulaMuris_Marrow/data/
```
