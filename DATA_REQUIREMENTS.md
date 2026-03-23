# Data Requirements

This repo includes the updated `locat-0.1` notebooks, scripts, and smaller supporting files needed for review and code inspection. Some large inputs and derived objects were not committed because they are too large for a normal GitHub repository.

The full machine-readable list is in `manifests/repro_bundle_manifest.csv`.

## Included directly
- Small and medium-sized summary CSVs, figures, and helper outputs
- `pbmc3k_lognorm.h5ad` and `pbmc3k_processed.h5ad`
- `kang_counts_25k.h5ad`
- selected PBMC CV outputs and simulation benchmark outputs
- bundled `locat-0.1` source snapshot

## External large files

### Dermal Condensate
These notebooks still require large dermal data objects that were not committed:
- `dc_adata_proc_rep.h5ad`
- `dc_adata_proc.h5ad`
- `adatasigsubfiltered.h5ad`

The repo does include smaller dermal support files such as:
- `clusters.csv`
- `E145_wls_dermal_FP_results.npy`
- selected locat result pickles used by the repro notebooks

### ESC Timecourse
These notebooks depend on several large ESC/embryoid inputs or derived objects that were not committed:
- `GSE227320_Aggregated_all_filtered_feature_bc_matrix.h5`
- `EMB_TC_datas_dict2.pkl`
- `EMB_TC_fulldatas_dict.pkl`
- some large intermediate result pickles

The repo does include the smaller downstream CSV outputs used in the analysis notebooks.

### Perturb PBMC
The main count matrix `kang_counts_25k.h5ad` is included.
A larger derived embedding object was not committed:
- `kang_merged_LGembedding.h5ad`

The repo does include the cross-validation summaries, AUROC tables, and held-out figure outputs.

### Simulations
The repo includes the main simulation notebooks/scripts and several smaller simulation inputs/outputs.
Two large derived result pickles were not committed:
- `dermalc_depletion_sres.pkl`
- `dermalc_subs_sres.pkl`

## Recommendation
If someone needs to rerun every notebook end-to-end rather than inspect the code and included summaries, start with `manifests/repro_bundle_manifest.csv` and restore the rows marked `excluded_large` into the corresponding paths.
