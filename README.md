# Locat-paper-repro-private

This private staging repo starts from the public `KlugerLab/Locat-paper` repository and adds the updated reproducibility materials that use `locat-0.1`.

## Included
- `locat-0.1` is expected as an external sibling repo at `/banach2/wes/locat-0.1`
- Updated locat-0.1 reproducibility notebooks under `notebooks/figures/`
- Updated simulation/runtime scripts under `notebooks/figures/Simulations/`
- Small and medium support files, result summaries, and selected figure outputs
- Dependency audit in `manifests/repro_bundle_manifest.csv`

## Not Included
Some required data/derived objects are too large for a normal GitHub repo (hundreds of MB to multiple GB). These are listed in `manifests/repro_bundle_manifest.csv` with status `excluded_large`.

## Folder Layout
The bundle mirrors the working folder structure used in the main LOCAT repo:
- `notebooks/figures/FigS1_3kPBMC`
- `notebooks/figures/Fig2_Dermal_Condensate`
- `notebooks/figures/Fig3_ESC_Timecourse`
- `notebooks/figures/Perturb_PBMC`
- `notebooks/figures/Simulations`
- `data/` for smaller input and summary files

## Path Note
Copied notebooks/scripts now expect `locat-0.1` to be available separately at `/banach2/wes/locat-0.1`. Notebook-adjacent support files were moved into `support_files/` subfolders to keep the analytical notebooks and scripts visually clean.
