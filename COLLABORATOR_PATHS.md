# Collaborator Path Guide

This repo was cleaned so it does not depend on Wes's personal absolute paths.

The notebooks and scripts now expect only:
- paths relative to this repo, and
- a separate `locat-0.1` checkout, provided either by `LOCAT01_PATH` or by placing `locat-0.1` next to this repo.

## Recommended Local Layout

The simplest setup is to clone both repos side by side:

```text
<your-work-dir>/
  Locat-paper-repro-private/
  locat-0.1/
```

Example:

```text
/home/yourname/projects/
  Locat-paper-repro-private/
  locat-0.1/
```

In that layout, most notebooks/scripts should work without further path edits.

## LOCAT Source

The reproducibility notebooks and scripts resolve `locat-0.1` like this:

1. If `LOCAT01_PATH` is set, they use that.
2. Otherwise, they look for a sibling folder named `locat-0.1`.

Example:

```bash
export LOCAT01_PATH=/home/yourname/projects/locat-0.1
```

## Where Files Live Inside This Repo

Main analysis notebooks/scripts:
- `notebooks/figures/FigS1_3kPBMC`
- `notebooks/figures/Fig2_Dermal_Condensate`
- `notebooks/figures/Fig3_ESC_Timecourse`
- `notebooks/figures/Perturb_PBMC`
- `notebooks/figures/Simulations`

Notebook-adjacent smaller support files:
- `notebooks/figures/Fig2_Dermal_Condensate/support_files`
- `notebooks/figures/Fig3_ESC_Timecourse/support_files`
- `notebooks/figures/Simulations/support_files`

Shared smaller data objects and summary tables:
- `data/`

Dependency audit:
- `manifests/repro_bundle_manifest.csv`

## Mapping to the Original Working Tree

If a collaborator wants to understand how this corresponds to the original LOCAT working tree:

- original `Locat/notebooks/figures/...`
  maps to
  `Locat-paper-repro-private/notebooks/figures/...`

- original `Locat/data/...`
  maps to
  `Locat-paper-repro-private/data/...`

The manifest keeps the original source locations in its first column for provenance, but the files in this repo should be used via the repo-relative paths in the second column.

## Large Files Not Included

Some notebooks still require larger source or intermediate files that are not committed here.

To see those:
- open `DATA_REQUIREMENTS.md`
- or inspect `manifests/repro_bundle_manifest.csv` and look for rows with `excluded_large` or `missing_in_source`

## Practical Advice for Collaborators

- Open notebooks from within this repo rather than copying them elsewhere.
- Keep `support_files/` folders next to their corresponding notebooks.
- Set `LOCAT01_PATH` explicitly if `locat-0.1` is not cloned as a sibling directory.
- Use `manifests/repro_bundle_manifest.csv` as the source of truth for what is bundled versus external.
