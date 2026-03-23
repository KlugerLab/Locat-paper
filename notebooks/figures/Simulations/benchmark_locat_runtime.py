#!/usr/bin/env python
"""Benchmark LOCAT runtime scaling over cell and gene subsampling grids."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
from anndata import read_h5ad


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def choose_embedding_key(adata, requested_key: str | None) -> str:
    keys = list(adata.obsm.keys())
    if requested_key and requested_key in keys:
        return requested_key
    for k in ("X_umap", "X_pca", "X_tsne"):
        if k in keys:
            return k
    if not keys:
        raise ValueError("No embedding found in adata.obsm; provide an AnnData with precomputed embedding.")
    return keys[0]


def patch_null_calibration(locat_obj, null_reps: int, null_fracs: np.ndarray) -> None:
    from locat.locat import LOCAT

    base_method = LOCAT.estimate_null_parameters

    def _patched(self, fractions=None, n_reps=50):
        if fractions is None:
            min_frac = max(10.0 / float(self.n_cells), 1.0 / float(self.n_cells))
            use_fracs = np.clip(null_fracs, min_frac, 0.999)
            use_fracs = np.unique(use_fracs)
            if use_fracs.size == 0:
                use_fracs = np.array([min_frac, min(0.1, 0.999), min(0.5, 0.999)], dtype=float)
        else:
            use_fracs = fractions
        use_reps = int(null_reps) if n_reps == 50 else int(n_reps)
        return base_method(self, fractions=use_fracs, n_reps=use_reps)

    locat_obj.estimate_null_parameters = types.MethodType(_patched, locat_obj)


def run_once(
    adata,
    embedding_key: str,
    n_cells: int,
    n_genes: int,
    rng: np.random.Generator,
    args,
) -> dict:
    from locat.locat import LOCAT

    cell_idx = rng.choice(adata.n_obs, size=n_cells, replace=False)
    gene_idx = rng.choice(adata.n_vars, size=n_genes, replace=False)
    ad_sub = adata[cell_idx, gene_idx].copy()

    emb = np.asarray(ad_sub.obsm[embedding_key], dtype=float)
    if args.embedding_dims > 0 and emb.shape[1] > args.embedding_dims:
        emb = emb[:, : args.embedding_dims]

    loc = LOCAT(
        adata=ad_sub,
        cell_embedding=emb,
        k=args.k,
        show_progress=False,
    )
    patch_null_calibration(
        locat_obj=loc,
        null_reps=args.null_reps,
        null_fracs=np.asarray(args.null_fracs, dtype=float),
    )

    t0 = time.perf_counter()
    loc.background_pdf(reps=args.background_reps, force_refresh=True)
    setup_sec = time.perf_counter() - t0

    genes = ad_sub.var_names.tolist()
    t1 = time.perf_counter()
    res = loc.gmm_scan(
        genes=genes,
        max_freq=args.max_freq,
        verbose=False,
    )
    scan_sec = time.perf_counter() - t1

    return {
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "setup_sec": float(setup_sec),
        "scan_sec": float(scan_sec),
        "total_sec": float(setup_sec + scan_sec),
        "returned_genes": int(len(res)),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LOCAT runtime scaling.")
    parser.add_argument(
        "--h5ad",
        type=str,
        default=str(Path(__file__).with_name("sim_subs.h5ad")),
        help="Input AnnData file.",
    )
    parser.add_argument(
        "--locat-repo",
        type=str,
        default="../../../locat-0.1",
        help="Path containing the locat package source.",
    )
    parser.add_argument("--embedding-key", type=str, default="X_umap")
    parser.add_argument("--embedding-dims", type=int, default=0, help="0 keeps all embedding dims.")
    parser.add_argument("--k", type=int, default=30, help="LOCAT k neighborhood parameter.")
    parser.add_argument("--cells-grid", type=str, default="500,1000,2000")
    parser.add_argument("--genes-grid", type=str, default="250,500,1000")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-freq", type=float, default=0.9)
    parser.add_argument("--background-reps", type=int, default=3)
    parser.add_argument("--null-reps", type=int, default=8)
    parser.add_argument("--null-fracs", type=str, default="0.01,0.03,0.1,0.3,0.6")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=str(Path(__file__).with_name("locat_runtime_benchmark")),
    )
    args = parser.parse_args()

    args.cells_grid = parse_int_list(args.cells_grid)
    args.genes_grid = parse_int_list(args.genes_grid)
    args.null_fracs = parse_float_list(args.null_fracs)

    sys.path.insert(0, args.locat_repo)

    adata = read_h5ad(args.h5ad)
    emb_key = choose_embedding_key(adata, args.embedding_key)

    cells_grid = [n for n in args.cells_grid if n <= adata.n_obs]
    genes_grid = [g for g in args.genes_grid if g <= adata.n_vars]
    if not cells_grid:
        raise ValueError(f"No valid cell sizes in grid for dataset with n_obs={adata.n_obs}.")
    if not genes_grid:
        raise ValueError(f"No valid gene sizes in grid for dataset with n_vars={adata.n_vars}.")

    rng = np.random.default_rng(args.seed)
    rows = []

    if args.warmup:
        _ = run_once(
            adata=adata,
            embedding_key=emb_key,
            n_cells=min(cells_grid),
            n_genes=min(genes_grid),
            rng=rng,
            args=args,
        )

    for n_cells in cells_grid:
        for n_genes in genes_grid:
            for rep in range(args.reps):
                out = run_once(
                    adata=adata,
                    embedding_key=emb_key,
                    n_cells=n_cells,
                    n_genes=n_genes,
                    rng=rng,
                    args=args,
                )
                out["rep"] = int(rep)
                rows.append(out)
                print(
                    f"[rep {rep}] n_cells={n_cells} n_genes={n_genes} "
                    f"setup={out['setup_sec']:.2f}s scan={out['scan_sec']:.2f}s total={out['total_sec']:.2f}s",
                    flush=True,
                )

    summary_map: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: {"setup_sec": [], "scan_sec": [], "total_sec": [], "returned_genes": []}
    )
    for row in rows:
        key = (int(row["n_cells"]), int(row["n_genes"]))
        summary_map[key]["setup_sec"].append(float(row["setup_sec"]))
        summary_map[key]["scan_sec"].append(float(row["scan_sec"]))
        summary_map[key]["total_sec"].append(float(row["total_sec"]))
        summary_map[key]["returned_genes"].append(float(row["returned_genes"]))

    summary_rows = []
    for (n_cells, n_genes), vals in sorted(summary_map.items()):
        summary_rows.append(
            {
                "n_cells": n_cells,
                "n_genes": n_genes,
                "median_setup_sec": float(np.median(vals["setup_sec"])),
                "median_scan_sec": float(np.median(vals["scan_sec"])),
                "median_total_sec": float(np.median(vals["total_sec"])),
                "median_returned_genes": float(np.median(vals["returned_genes"])),
            }
        )

    out_prefix = Path(args.out_prefix)
    raw_csv = out_prefix.with_name(out_prefix.name + "_raw.csv")
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    meta_json = out_prefix.with_name(out_prefix.name + "_meta.json")

    write_csv(
        raw_csv,
        rows,
        fieldnames=[
            "n_cells",
            "n_genes",
            "rep",
            "setup_sec",
            "scan_sec",
            "total_sec",
            "returned_genes",
        ],
    )
    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "n_cells",
            "n_genes",
            "median_setup_sec",
            "median_scan_sec",
            "median_total_sec",
            "median_returned_genes",
        ],
    )

    meta = {
        "h5ad": str(Path(args.h5ad).resolve()),
        "embedding_key_used": emb_key,
        "cells_grid": cells_grid,
        "genes_grid": genes_grid,
        "reps": args.reps,
        "seed": args.seed,
        "k": args.k,
        "background_reps": args.background_reps,
        "null_reps": args.null_reps,
        "null_fracs": args.null_fracs,
        "locat_repo": str(Path(args.locat_repo).resolve()),
        "locat_api": "locat.locat.LOCAT.gmm_scan",
    }
    meta_json.write_text(json.dumps(meta, indent=2))

    print(f"Wrote raw benchmark rows: {raw_csv}")
    print(f"Wrote summary benchmark rows: {summary_csv}")
    print(f"Wrote benchmark metadata: {meta_json}")


if __name__ == "__main__":
    main()
