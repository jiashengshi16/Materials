#!/usr/bin/env python3
"""
python wanniererrorplots.py \
  "/Users/jshi/Downloads/Wannierisation_Gemini_Benchmark - DeepseekControlledChemSimExperimentIter3.csv" \
  "/Users/jshi/Downloads/Wannierisation_Gemini_Benchmark - DeepseekControlledChemSimExperimentIter4.csv" \
  -o part6_rerun_comparison_combined.png

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator

MATERIAL_COL = "material"
ORIGINAL_AVG_COL = "avg_original_run_error_ratio"
RERUN_AVG_COL = "avg_new_run_error_ratio"
ORIGINAL_RUN_COLS = [
    "original_run_1_error_ratio",
    "original_run_2_error_ratio",
    "original_run_3_error_ratio",
]
RERUN_RUN_COLS = [
    "new_run_1_error_ratio",
    "new_run_2_error_ratio",
    "new_run_3_error_ratio",
]

# Presentation colors. Change these hex values if needed.
BASELINE_GRAY = "#9A9A9A"
BRIGHT_GREEN = "#20D45A"
MEDIUM_GREEN = "#4F9D69"
WORSE_RED = "#D9534F"
MINIMAL_BEIGE = "#D8C3A5"
THRESHOLD_GREEN = "#168A45"


def classify_rerun(original: float, rerun: float, usable_threshold: float) -> tuple[str, str]:
    """Return category label and color, with usable status taking priority."""
    if rerun <= usable_threshold:
        return f"Usable after rerun (≤ {usable_threshold:g})", BRIGHT_GREEN
    if rerun <= 0.5 * original:
        return "Improved by ≥ 50%", MEDIUM_GREEN
    if rerun >= 1.5 * original:
        return "Worsened by ≥ 50%", WORSE_RED
    return "Minimal/moderate change", MINIMAL_BEIGE


def _numericify(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_and_prepare(csv_paths: list[Path], original_minimum: float, usable_threshold: float) -> pd.DataFrame:
    frames = []
    required = {MATERIAL_COL, *ORIGINAL_RUN_COLS, *RERUN_RUN_COLS}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

        df = _numericify(df, ORIGINAL_RUN_COLS + RERUN_RUN_COLS + [ORIGINAL_AVG_COL, RERUN_AVG_COL])
        df[MATERIAL_COL] = df[MATERIAL_COL].astype(str).str.strip()
        df["source_csv"] = csv_path.stem
        frames.append(df[[MATERIAL_COL, "source_csv", *ORIGINAL_RUN_COLS, *RERUN_RUN_COLS]])

    combined_raw = pd.concat(frames, ignore_index=True)
    combined_raw = combined_raw.dropna(subset=[MATERIAL_COL]).copy()
    combined_raw = combined_raw[combined_raw[MATERIAL_COL] != ""].copy()

    rows = []
    for material, grp in combined_raw.groupby(MATERIAL_COL, sort=False):
        original_values = grp[ORIGINAL_RUN_COLS].to_numpy(dtype=float).ravel()
        rerun_values = grp[RERUN_RUN_COLS].to_numpy(dtype=float).ravel()

        original_values = original_values[np.isfinite(original_values)]
        rerun_values = rerun_values[np.isfinite(rerun_values)]

        if len(original_values) == 0 or len(rerun_values) == 0:
            continue

        rows.append(
            {
                MATERIAL_COL: material,
                ORIGINAL_AVG_COL: float(np.mean(original_values)),
                RERUN_AVG_COL: float(np.mean(rerun_values)),
                "n_original_values": int(len(original_values)),
                "n_rerun_values": int(len(rerun_values)),
                "sources_used": ", ".join(sorted(grp["source_csv"].dropna().unique())),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No rows could be built from the supplied CSV files.")

    df = df[(df[ORIGINAL_AVG_COL] >= original_minimum) & (df[ORIGINAL_AVG_COL] > 0) & (df[RERUN_AVG_COL] > 0)].copy()

    if df.empty:
        raise ValueError(f"No valid rows have combined {ORIGINAL_AVG_COL} >= {original_minimum}.")

    # Positive means improvement; negative means worsening.
    df["relative_improvement"] = (df[ORIGINAL_AVG_COL] - df[RERUN_AVG_COL]) / df[ORIGINAL_AVG_COL]

    classifications = [
        classify_rerun(original, rerun, usable_threshold)
        for original, rerun in zip(df[ORIGINAL_AVG_COL], df[RERUN_AVG_COL])
    ]
    df["rerun_category"] = [item[0] for item in classifications]
    df["rerun_color"] = [item[1] for item in classifications]

    # Best relative improvement first; worsening cases naturally move to the end.
    df = df.sort_values(
        by=["relative_improvement", RERUN_AVG_COL],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    return df


def make_plot(
    df: pd.DataFrame,
    output_path: Path,
    usable_threshold: float,
    original_minimum: float,
    title: str,
    dpi: int,
) -> None:
    n = len(df)
    x = np.arange(n)

    fig_width = max(12.0, 0.42 * n + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 10.0))

    original = df[ORIGINAL_AVG_COL].to_numpy(float)
    rerun = df[RERUN_AVG_COL].to_numpy(float)
    rerun_colors = df["rerun_color"].tolist()

    positive_min = min(original.min(), rerun.min(), usable_threshold)
    lower_limit = max(0.1, positive_min / 1.8)
    upper_limit = max(original.max(), rerun.max()) * 1.35
    ax.axhspan(lower_limit, usable_threshold, color=BRIGHT_GREEN, alpha=0.07, zorder=0)

    for i, (orig, new, new_color) in enumerate(zip(original, rerun, rerun_colors)):
        if orig >= new:
            ax.bar(i, orig, width=0.84, color=BASELINE_GRAY, alpha=0.82,
                   edgecolor="white", linewidth=0.55, zorder=2)
            ax.bar(i, new, width=0.50, color=new_color,
                   edgecolor="white", linewidth=0.65, zorder=3)
        else:
            ax.bar(i, new, width=0.84, color=new_color, alpha=0.88,
                   edgecolor="white", linewidth=0.55, zorder=2)
            ax.bar(i, orig, width=0.50, color=BASELINE_GRAY,
                   edgecolor="white", linewidth=0.65, zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(lower_limit, upper_limit)

    ax.axhline(usable_threshold, color=THRESHOLD_GREEN, linewidth=2.0,
               linestyle="--", zorder=4)
    ax.annotate(
        f"Usable threshold ≤ {usable_threshold:g}",
        xy=(0.0, usable_threshold),
        xycoords=("axes fraction", "data"),
        xytext=(8, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=THRESHOLD_GREEN,
        fontsize=18,
        fontweight="bold",
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df[MATERIAL_COL], rotation=65, ha="right", fontsize=8)
    ax.set_ylabel(
        "Average error ratio (log scale)",
        fontsize=20,
        fontweight="bold",
        labelpad=14,
    )
    ax.set_xlabel(
        "Materials",
        fontsize=25,
        fontweight="bold",
        labelpad=16,
    )
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=14)
    ax.text(
        0,
        1.01,
            (
        f""
    ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#555555",
    )

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.8, alpha=0.45)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=BASELINE_GRAY, label="Original-run average"),
        Patch(facecolor=BRIGHT_GREEN, label=f"Rerun usable (≤ {usable_threshold:g})"),
        Patch(facecolor=MEDIUM_GREEN, label="Rerun improved by ≥ 50%"),
        Patch(facecolor=MINIMAL_BEIGE, label="Rerun minimal/moderate change"),
        Patch(facecolor=WORSE_RED, label="Rerun worsened by ≥ 50%"),
    ]
    ax.legend(
        handles=legend_handles,
        ncol=1,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=20,
    )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.87, bottom=0.34)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if output_path.suffix.lower() != ".svg":
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", type=Path, nargs="+", help="One or more input CSV files")
    parser.add_argument("-o", "--output", type=Path, default=Path("part6_rerun_comparison_combined.png"))
    parser.add_argument("--original-minimum", type=float, default=0.0)
    parser.add_argument("--usable-threshold", type=float, default=6.0)
    parser.add_argument("--title", default="")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_and_prepare(args.csvs, args.original_minimum, args.usable_threshold)
    make_plot(
        df=df,
        output_path=args.output,
        usable_threshold=args.usable_threshold,
        original_minimum=args.original_minimum,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"Plotted {len(df)} materials.")
    print(f"Saved: {args.output}")
    if args.output.suffix.lower() != ".svg":
        print(f"Saved: {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
