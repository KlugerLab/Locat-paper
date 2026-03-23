# Locat-paper-repro-private

This private staging repo starts from the public `KlugerLab/Locat-paper` repository and adds the updated reproducibility materials that use `locat-0.1`.

## Included
- Snapshot of `locat-0.1` source in `locat-0.1/`
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
Copied notebooks/scripts were patched so references to `/banach2/wes/locat-0.1` now point to the bundled `locat-0.1` snapshot via a relative path. Other large absolute data paths were not fully rewritten because the corresponding large files are not bundled in this repo.
