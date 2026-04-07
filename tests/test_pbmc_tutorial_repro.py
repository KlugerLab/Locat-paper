"""
Regression tests for the PBMC3K LOCAT tutorial.

Fast (default): load PBMC_locat_results.csv from the notebook directory and
check counts, rankings, and numerical values against the reference fixture.
These run in < 1 s and are useful after any code change to catch regressions.

Full re-execution (--run-notebook): execute the tutorial notebook from
scratch using nbconvert, which overwrites the CSV, then run the same checks.
This takes ~2 minutes and requires a working Jupyter kernel + all LOCAT dependencies.

  Run fast checks only:  pytest tests/test_pbmc_tutorial_repro.py
  Run full end-to-end:   pytest tests/test_pbmc_tutorial_repro.py --run-notebook

Regenerate fixtures only after an intentional result change:
    cp notebooks/figures/FigS1_3kPBMC/PBMC_locat_results.csv tests/fixtures/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

NB_DIR   = Path("notebooks/locat_tutorial_pbmc3k")
NOTEBOOK = NB_DIR / "locat_tutorial_pbmc3k.ipynb"
NB_RESULTS = NB_DIR / "PBMC_locat_results.csv"

FIX_DIR     = Path("tests/fixtures")
FIX_RESULTS = FIX_DIR / "PBMC_locat_results.csv"

ALPHA = 0.05
N_TOP = 6

# ---------------------------------------------------------------------------
# Notebook execution (triggered by --run-notebook)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def maybe_run_notebook(request):
    if not request.config.getoption("--run-notebook"):
        return
    if not NOTEBOOK.exists():
        pytest.fail(f"Notebook not found: {NOTEBOOK}")
    print(f"\nExecuting notebook (~2 min): {NOTEBOOK}")
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=600",
            "--inplace",
            str(NOTEBOOK.resolve()),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Notebook execution failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-3000:]}\n"
            f"STDERR:\n{result.stderr[-3000:]}"
        )


# ---------------------------------------------------------------------------
# Helpers — derive summary counts and top genes from a results DataFrame
# ---------------------------------------------------------------------------

def _summary(df):
    conc = df["concentration_pval"] < ALPHA
    depl = df["depletion_pval"] < ALPHA
    return {
        "genes_scored": len(df),
        "conc_sig":  int(conc.sum()),
        "depl_sig":  int(depl.sum()),
        "joint_sig": int((df["pval"] < ALPHA).sum()),
        "conc_only": int((conc & ~depl).sum()),
        "conc_depl": int((conc & depl).sum()),
    }


def _top_conc_only(df):
    return (df[(df["concentration_pval"] < ALPHA) & (df["depletion_pval"] >= ALPHA)]
            .sort_values(["zscore", "max_deficit", "concentration_pval"],
                         ascending=[False, False, True])
            .index[:N_TOP].tolist())


def _top_conc_depl(df):
    return (df[(df["concentration_pval"] < ALPHA) & (df["depletion_pval"] < ALPHA)]
            .sort_values(["max_deficit", "depletion_pval", "concentration_pval"],
                         ascending=[False, True, True])
            .index[:N_TOP].tolist())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _require(path):
    if not path.exists():
        pytest.skip(f"File not found (run the notebook first): {path}")
    return path


@pytest.fixture(scope="module")
def nb_results():
    return pd.read_csv(_require(NB_RESULTS), index_col="gene")


@pytest.fixture(scope="module")
def fix_results():
    return pd.read_csv(_require(FIX_RESULTS), index_col="gene")


# ---------------------------------------------------------------------------
# Summary count tests
# ---------------------------------------------------------------------------

def test_gene_count(nb_results, fix_results):
    assert _summary(nb_results)["genes_scored"] == _summary(fix_results)["genes_scored"]


def test_conc_sig_count(nb_results, fix_results):
    assert _summary(nb_results)["conc_sig"] == _summary(fix_results)["conc_sig"]


def test_depl_sig_count(nb_results, fix_results):
    assert _summary(nb_results)["depl_sig"] == _summary(fix_results)["depl_sig"]


def test_joint_sig_count(nb_results, fix_results):
    assert _summary(nb_results)["joint_sig"] == _summary(fix_results)["joint_sig"]


def test_conc_only_count(nb_results, fix_results):
    assert _summary(nb_results)["conc_only"] == _summary(fix_results)["conc_only"]


def test_conc_depl_count(nb_results, fix_results):
    assert _summary(nb_results)["conc_depl"] == _summary(fix_results)["conc_depl"]


# ---------------------------------------------------------------------------
# Top gene ranking tests
# ---------------------------------------------------------------------------

def test_top_conc_only_genes(nb_results, fix_results):
    got, exp = _top_conc_only(nb_results), _top_conc_only(fix_results)
    assert got == exp, f"Top conc-only genes mismatch:\n  got:      {got}\n  expected: {exp}"


def test_top_conc_depl_genes(nb_results, fix_results):
    got, exp = _top_conc_depl(nb_results), _top_conc_depl(fix_results)
    assert got == exp, f"Top conc+depl genes mismatch:\n  got:      {got}\n  expected: {exp}"


# ---------------------------------------------------------------------------
# Numerical value tests
# ---------------------------------------------------------------------------

def test_pval_values(nb_results, fix_results):
    common = nb_results.index.intersection(fix_results.index)
    assert len(common) == len(fix_results), "Gene index mismatch"
    np.testing.assert_allclose(
        nb_results.loc[common, "pval"].astype(float).values,
        fix_results.loc[common, "pval"].astype(float).values,
        rtol=1e-5, err_msg="pval column differs from reference",
    )


def test_concentration_pval_values(nb_results, fix_results):
    common = nb_results.index.intersection(fix_results.index)
    np.testing.assert_allclose(
        nb_results.loc[common, "concentration_pval"].astype(float).values,
        fix_results.loc[common, "concentration_pval"].astype(float).values,
        rtol=1e-5, err_msg="concentration_pval column differs from reference",
    )


def test_zscore_values(nb_results, fix_results):
    common = nb_results.index.intersection(fix_results.index)
    np.testing.assert_allclose(
        nb_results.loc[common, "zscore"].astype(float).values,
        fix_results.loc[common, "zscore"].astype(float).values,
        rtol=1e-4, err_msg="zscore column differs from reference",
    )


def test_max_deficit_values(nb_results, fix_results):
    common = nb_results.index.intersection(fix_results.index)
    valid = ~fix_results.loc[common, "max_deficit"].isna()
    np.testing.assert_allclose(
        nb_results.loc[common[valid], "max_deficit"].astype(float).values,
        fix_results.loc[common[valid], "max_deficit"].astype(float).values,
        rtol=1e-4, err_msg="max_deficit column differs from reference",
    )
