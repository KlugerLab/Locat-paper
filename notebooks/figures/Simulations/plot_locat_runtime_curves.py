#!/usr/bin/env python
"""Generate runtime scaling curves from LOCAT benchmark outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


def _plain_number(x, _pos=None):
    if x >= 1000:
        return f"{int(round(x)):,}"
    if x >= 1:
        return f"{x:.0f}"
    return f"{x:.2g}"


def _set_dense_log_y_ticks(ax) -> None:
    ymin, ymax = ax.get_ylim()
    if ymin <= 0 or ymax <= 0:
        return
    kmin = int(math.floor(math.log10(ymin)))
    kmax = int(math.ceil(math.log10(ymax)))
    ticks = []
    for k in range(kmin, kmax + 1):
        for m in (1, 2, 5):
            v = m * (10 ** k)
            if ymin <= v <= ymax:
                ticks.append(v)
    if ticks:
        ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_plain_number))
    ax.yaxis.set_minor_locator(NullLocator())


def add_curve_panel(ax, data: pd.DataFrame, metric: str, title: str) -> None:
    genes = sorted(data["n_genes"].unique())
    for g in genes:
        d = data[data["n_genes"] == g].sort_values("n_cells")
        ax.plot(d["n_cells"], d["median"], marker="o", linewidth=2, label=f"{g} genes")
        ax.fill_between(d["n_cells"], d["q25"], d["q75"], alpha=0.15)

    ax.set_xscale("log", base=10)
    ax.xaxis.set_major_locator(FixedLocator(sorted(data["n_cells"].unique())))
    ax.xaxis.set_major_formatter(FuncFormatter(_plain_number))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(title)
    _set_dense_log_y_ticks(ax)
    ax.grid(alpha=0.25, linestyle="--")


def add_curve_panel_by_cells(ax, data: pd.DataFrame, title: str) -> None:
    cells = sorted(data["n_cells"].unique())
    for n in cells:
        d = data[data["n_cells"] == n].sort_values("n_genes")
        ax.plot(d["n_genes"], d["median"], marker="o", linewidth=2, label=f"{n} cells")
        ax.fill_between(d["n_genes"], d["q25"], d["q75"], alpha=0.15)

    ax.set_xscale("log", base=10)
    ax.xaxis.set_major_locator(FixedLocator(sorted(data["n_genes"].unique())))
    ax.xaxis.set_major_formatter(FuncFormatter(_plain_number))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Number of genes")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(title)
    _set_dense_log_y_ticks(ax)
    ax.grid(alpha=0.25, linestyle="--")


def summarize(raw: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    out = (
        raw.groupby(["n_cells", "n_genes"], as_index=False)[metric_col]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .sort_values(["n_genes", "n_cells"])
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-csv",
        type=str,
        default="/banach2/wes/Locat-paper-repro-private/notebooks/figures/Simulations/support_files/locat_runtime_benchmark_16k_sparsegrid_r3_raw.csv",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="/banach2/wes/Locat-paper-repro-private/notebooks/figures/Simulations/support_files/locat_runtime_curves_r3",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_csv)
    out_prefix = Path(args.out_prefix)

    raw = pd.read_csv(raw_path)

    setup = summarize(raw, "setup_sec")
    scan = summarize(raw, "scan_sec")
    total = summarize(raw, "total_sec")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    add_curve_panel(axes[0], setup, "setup_sec", "Setup Time vs Cells")
    add_curve_panel(axes[1], scan, "scan_sec", "Scan Time vs Cells")
    add_curve_panel(axes[2], total, "total_sec", "Total Time vs Cells")
    axes[2].legend(frameon=False, loc="upper left", fontsize=9)

    svg = out_prefix.with_suffix(".svg")
    png = out_prefix.with_suffix(".png")
    fig.savefig(svg, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    add_curve_panel_by_cells(axes2[0], setup, "Setup Time vs Genes")
    add_curve_panel_by_cells(axes2[1], scan, "Scan Time vs Genes")
    add_curve_panel_by_cells(axes2[2], total, "Total Time vs Genes")
    axes2[2].legend(frameon=False, loc="upper left", fontsize=9)
    svg2 = out_prefix.with_name(out_prefix.name + "_by_genes.svg")
    png2 = out_prefix.with_name(out_prefix.name + "_by_genes.png")
    fig2.savefig(svg2, dpi=300)
    fig2.savefig(png2, dpi=300)
    plt.close(fig2)

    # Also write derived summary tables for manuscript plotting flexibility.
    setup.to_csv(out_prefix.with_name(out_prefix.name + "_setup_summary.csv"), index=False)
    scan.to_csv(out_prefix.with_name(out_prefix.name + "_scan_summary.csv"), index=False)
    total.to_csv(out_prefix.with_name(out_prefix.name + "_total_summary.csv"), index=False)

    print(f"Wrote figure: {svg}")
    print(f"Wrote figure: {png}")
    print(f"Wrote figure: {svg2}")
    print(f"Wrote figure: {png2}")
    print("Wrote per-metric summaries with median/IQR.")


if __name__ == "__main__":
    main()
