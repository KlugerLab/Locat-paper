"""
Assign G1/S/G2M phase labels to Chlamydomonas cells using Zones 2015 bulk
diurnal time-course data.

Strategy (mirrors Seurat cell_cycle_scoring approach):
  1. Load Zones 2015 RPKM time-course (GSE71469) — 17,736 genes × 28 time points
  2. Average two replicates; compute peak ZT for each gene
  3. Define phase gene sets by peak ZT:
       S      genes: peak ZT  9-13  (late light / DNA replication onset)
       G2M    genes: peak ZT 13-21  (dark period / mitosis)
       G1     genes: everything else (low-amplitude or early-light peak)
     — Genes must also pass a rhythmicity filter: ratio of max/mean > 3
  4. Intersect with genes present in each processed adata
  5. Compute per-cell module scores (mean log-expr of phase genes, minus
     background control gene set matched by expression level)
  6. Assign phase: if both scores < 0 → G1; else argmax(S, G2M)
  7. Save phase labels to adata and write updated h5ad

Zones 2015 time points (light:dark 12h:12h, ZT = hours after light on):
  ZT 1-12  = light period (G1 growth)
  ZT 12-24 = dark period (S-phase replication + M-phase divisions)

Usage:
    python assign_cell_cycle_phases.py
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from pathlib import Path
import time

HERE      = Path(__file__).parent
SCORES    = HERE / "locat_run" / "scores"
ZONES_TSV = Path("/tmp/zones2015/zones_rpkm.txt")

SAMPLES = [
    dict(name="fe_neg", label="Fe−"),
    dict(name="fe_pos", label="Fe+"),
]

# ZT bin definitions (inclusive on both ends)
# Based on Zones 2015 cluster timing and Chlamydomonas cell cycle literature:
#   G1  : ZT0-8  (light phase growth)
#   S   : ZT9-13 (DNA replication, peaks at ~ZT11 for MCM/PCNA genes)
#   G2M : ZT14-22 (mitosis + cytokinesis, dark phase)
S_ZT_MIN, S_ZT_MAX     = 9,  13
G2M_ZT_MIN, G2M_ZT_MAX = 14, 22
AMPLITUDE_RATIO_MIN     = 3.0   # max/mean RPKM must exceed this
MIN_PHASE_GENES         = 5     # require at least this many genes per phase in adata

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── 1. Load and process Zones 2015 bulk data ─────────────────────────────────
log("Loading Zones 2015 RPKM data...")
zones = pd.read_csv(ZONES_TSV, sep="\t", index_col=0)
zones = zones.drop(columns=["Locus ID (v5.3.1)"], errors="ignore")

# Column names like "1_1", "11.5_2" → extract ZT float
col_zt = {}
for col in zones.columns:
    zt_str, rep = col.rsplit("_", 1)
    col_zt[col] = float(zt_str)

# Average across the two replicates at each ZT
zt_values = sorted(set(col_zt.values()))
rep_groups = {}
for col, zt in col_zt.items():
    rep_groups.setdefault(zt, []).append(col)

avg_expr = pd.DataFrame(
    {zt: zones[cols].mean(axis=1) for zt, cols in rep_groups.items()},
    index=zones.index,
)
avg_expr = avg_expr[sorted(avg_expr.columns)]  # sort by ZT

log(f"  {avg_expr.shape[0]} genes × {avg_expr.shape[1]} time points (ZT averaged)")

# Peak ZT per gene
peak_zt = avg_expr.idxmax(axis=1)

# Rhythmicity filter: max / mean > threshold
gene_max  = avg_expr.max(axis=1)
gene_mean = avg_expr.mean(axis=1).replace(0, np.nan)
amplitude = gene_max / gene_mean
rhythmic  = amplitude >= AMPLITUDE_RATIO_MIN

log(f"  Rhythmic genes (max/mean ≥ {AMPLITUDE_RATIO_MIN}): {rhythmic.sum()}")

# Phase gene sets (Cre ##.g######  without .v5.5 suffix in Zones data)
rhythmic_genes = avg_expr.index[rhythmic]
s_genes_zones   = set(avg_expr.index[rhythmic & (peak_zt >= S_ZT_MIN)   & (peak_zt <= S_ZT_MAX)])
g2m_genes_zones = set(avg_expr.index[rhythmic & (peak_zt >= G2M_ZT_MIN) & (peak_zt <= G2M_ZT_MAX)])

log(f"  S genes (ZT {S_ZT_MIN}-{S_ZT_MAX}): {len(s_genes_zones)}")
log(f"  G2M genes (ZT {G2M_ZT_MIN}-{G2M_ZT_MAX}): {len(g2m_genes_zones)}")

# Print a few well-known expected genes for sanity check
known = {"Cre10.g465900": "CDKA1", "Cre08.g370401": "CYCB1"}
for cre_id, name in known.items():
    if cre_id in peak_zt.index:
        zt = peak_zt[cre_id]
        amp = amplitude.get(cre_id, np.nan)
        log(f"  Known gene {name} ({cre_id}): peak ZT={zt}, amplitude={amp:.1f}")

# ── 2. Process each sample ────────────────────────────────────────────────────
for ds in SAMPLES:
    cond_dir  = SCORES / ds["name"]
    h5ad_path = cond_dir / "adata_proc.h5ad"
    out_path  = cond_dir / "adata_phases.h5ad"

    log(f"\n=== {ds['label']} ===")
    adata = sc.read_h5ad(h5ad_path)
    log(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    # Zones IDs are like Cre01.g000050 (no .v5.5)
    # adata IDs are like Cre01.g000050.v5.5
    # Build mapping
    adata_genes_set = set(adata.var_names.tolist())
    def zones_to_adata(gid):
        return gid + ".v5.5"

    s_genes   = [zones_to_adata(g) for g in s_genes_zones   if zones_to_adata(g) in adata_genes_set]
    g2m_genes = [zones_to_adata(g) for g in g2m_genes_zones if zones_to_adata(g) in adata_genes_set]

    log(f"  S genes in adata: {len(s_genes)} / {len(s_genes_zones)}")
    log(f"  G2M genes in adata: {len(g2m_genes)} / {len(g2m_genes_zones)}")

    if len(s_genes) < MIN_PHASE_GENES or len(g2m_genes) < MIN_PHASE_GENES:
        log(f"  WARNING: too few phase genes; skipping phase assignment")
        continue

    # ── 3. Score cells with sc.tl.score_genes ────────────────────────────────
    # score_genes computes (mean expression of gene set) - (mean of control genes
    # randomly sampled from genes with similar expression levels)
    sc.tl.score_genes(adata, gene_list=s_genes,   score_name="S_score",   use_raw=False)
    sc.tl.score_genes(adata, gene_list=g2m_genes, score_name="G2M_score", use_raw=False)

    s_score   = adata.obs["S_score"].values
    g2m_score = adata.obs["G2M_score"].values

    # ── 4. Assign phase ───────────────────────────────────────────────────────
    # Seurat convention: G1 if both scores ≤ 0; else argmax
    phase = np.where(
        (s_score <= 0) & (g2m_score <= 0),
        "G1",
        np.where(s_score >= g2m_score, "S", "G2M"),
    )
    adata.obs["phase"] = pd.Categorical(phase, categories=["G1", "S", "G2M"])

    counts = adata.obs["phase"].value_counts()
    log(f"  Phase counts: {counts.to_dict()}")
    log(f"  S score   range: [{s_score.min():.3f}, {s_score.max():.3f}]")
    log(f"  G2M score range: [{g2m_score.min():.3f}, {g2m_score.max():.3f}]")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    adata.write_h5ad(out_path)
    log(f"  Saved → {out_path.name}")

    # Also save a simple CSV for downstream use
    phase_df = adata.obs[["S_score", "G2M_score", "phase"]].copy()
    phase_df.to_csv(cond_dir / "cell_phases.csv")
    log(f"  Saved → cell_phases.csv")

log("\nAll done.")

# ── Summary of phase gene lists ───────────────────────────────────────────────
log(f"\nGene set summary:")
log(f"  S phase genes (Zones peak ZT {S_ZT_MIN}-{S_ZT_MAX}):   {len(s_genes_zones)} total, mapped to adata")
log(f"  G2M genes   (Zones peak ZT {G2M_ZT_MIN}-{G2M_ZT_MAX}): {len(g2m_genes_zones)} total, mapped to adata")
