#!/usr/bin/env python3
"""Cluster Gemini calculations by process diagnostics in submitted outputs.

The process heatmap deliberately does not use interpolation error.  Error ratio
is shown as a row-colour annotation there, so enrichment of high-error
calculations in a cluster is an independent observation rather than a built-in
result.  A second heatmap includes interpolation error as a clustered feature.

Outputs:
  gemini_wout_cluster_heatmap_process.png/pdf     process-only heatmap
  gemini_wout_cluster_heatmap_all_columns.png/pdf heatmap including error ratio
  gemini_wout_cluster_assignments.csv  one traceable row per calculation
  gemini_wout_cluster_summary.csv      cluster-level error/feature summary
  gemini_wout_features.csv             all raw and transformed diagnostics
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

# Keep matplotlib/font caches inside a writable location on managed machines.
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import nan_euclidean_distances


DEFAULT_SUMMARY = ROOT / "jobs" / "num_wann_ordered_diagnostics_summary.json"
DEFAULT_JOBS = ROOT / "jobs"
DEFAULT_DATASET = ROOT / "harbor_datasets" / "wannier_200"
DEFAULT_OUTPUT = ROOT / "jobs" / "gemini_wout_clustering"
DEFAULT_PROJECTION_CATEGORIES = ROOT / "jobs" / "gemini_projection_categories_from_win.json"

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
WF_RE = re.compile(
    rf"WF\s+centre\s+and\s+spread\s+(\d+)\s+\([^\n]*\)\s+({FLOAT})",
    re.IGNORECASE,
)
OMEGA_RE = re.compile(
    rf"Omega\s+(I|D|OD|Total)\s*=\s*({FLOAT})", re.IGNORECASE
)
ITER_RE = re.compile(
    rf"^\s*(\d+)\s+{FLOAT}\s+{FLOAT}\s+{FLOAT}\s+{FLOAT}\s+<--\s*CONV",
    re.MULTILINE,
)
WIN_INT_RE = re.compile(r"^\s*(num_iter|dis_num_iter)\s*=\s*(\d+)", re.I | re.M)
WOUT_TOTAL_ITER_RE = re.compile(
    r"Total number of iterations\s*:\s*(\d+)", re.IGNORECASE
)


# Human-readable names appear as columns in the heatmap. The underlying raw
# values remain available in gemini_wout_features.csv. Missing values are kept
# blank in the heatmap and ignored pairwise when computing row distances.
PROCESS_FEATURES = {
    "log_final_spread_per_wf": "final spread/WF badness",
    "log_final_spread_per_wf_vs_ref": "spread/WF vs ref badness",
    "log_max_wf_spread": "worst-WF spread badness",
    "log_max_wf_spread_vs_ref": "worst WF vs ref badness",
    "log_max_to_median": "worst/median WF badness",
    "omega_I_fraction": "Omega I fraction badness",
    "omega_OD_fraction": "Omega OD fraction badness",
    "fractional_spread_reduction": "spread reduction deficit",
}
ALL_FEATURES = {
    **PROCESS_FEATURES,
    "log_error_ratio": "interpolation error badness",
}
FEATURE_DIRECTIONS = {
    # Values in the heatmap are oriented so red always means worse and blue
    # always means better.  Negating the reduction feature preserves pairwise
    # distances but fixes the visual sign convention.
    "fractional_spread_reduction": -1,
}

CLUSTER_CMAP = LinearSegmentedColormap.from_list(
    "cluster_orange_to_leafgreen",
    [
        "#F58518",  # orange
        "#FDBA12",  # yellow-orange
        "#FFE34D",  # yellow
        "#A8E05F",  # lime
        "#2E7D32",  # leaf green
    ],
)

ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error_pink_to_purple",
    ["#F7B6D2", "#6A00A8"],  # pink -> purple
)

MISSING_COLOR = "#FFFFFF"
MISSING_CLUSTER_COLOR = "#BDBDBD"
PROJECTION_COLORS = {
    "explicit_projection_runs": "#D9D9D9",
    "random_projection_runs": "#4A4A4A",
    "none_or_implicit_projection_runs": "#000000",
}
PROJECTION_LABELS = {
    "explicit_projection_runs": "Explicit projections",
    "random_projection_runs": "Random projections",
    "none_or_implicit_projection_runs": "None/implicit projections",
}
def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def parse_state_spreads(text: str, marker: str, last: bool) -> list[float]:
    start = text.rfind(marker) if last else text.find(marker)
    if start < 0:
        return []
    end = text.find("Sum of centres and spreads", start)
    if end < 0:
        return []
    return [float(value.replace("D", "E").replace("d", "e")) for _, value in WF_RE.findall(text[start:end])]


def parse_wout(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    initial = parse_state_spreads(text, "Initial State", last=False)
    final = parse_state_spreads(text, "Final State", last=True)
    omega: dict[str, float] = {}
    for key, value in OMEGA_RE.findall(text):
        omega[key.lower()] = float(value.replace("D", "E").replace("d", "e"))
    iterations = [int(value) for value in ITER_RE.findall(text)]
    lower = text.lower()
    if "disentanglement convergence criteria satisfied" in lower:
        dis_converged: float | None = 1.0
    elif "maximum number of disentanglement iterations reached" in lower:
        dis_converged = 0.0
    else:
        # No disentanglement is required when num_bands == num_wann.
        dis_converged = None
    return {
        "initial_wf_spreads": initial,
        "final_wf_spreads": final,
        "omega_I": omega.get("i"),
        "omega_D": omega.get("d"),
        "omega_OD": omega.get("od"),
        "omega_total": omega.get("total"),
        "localization_iterations": max(iterations) if iterations else None,
        # The first occurrence belongs to WANNIERISE; a later occurrence can
        # be the disentanglement limit.  Some submitted .win files omit the
        # default, while .wout always prints the resolved value.
        "localization_iteration_limit": (
            int(WOUT_TOTAL_ITER_RE.search(text).group(1))
            if WOUT_TOTAL_ITER_RE.search(text)
            else None
        ),
        "disentanglement_converged": dis_converged,
        "warning_count": sum("warning" in line.lower() for line in text.splitlines()),
    }


def find_trial(record: dict[str, Any], jobs_root: Path) -> Path:
    job = jobs_root / str(record["job_folder"])
    named = record.get("trial_folder")
    if isinstance(named, str) and (job / named).is_dir():
        return job / named
    candidates = [p for p in sorted(job.iterdir()) if p.is_dir() and (p / "verifier").is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one trial directory under {job}")
    return candidates[0]


def gemini_file(trial: Path, suffix: str) -> Path:
    candidates = [
        *sorted((trial / "artifacts" / "artifacts").glob(f"attempt_*/*{suffix}")),
        *sorted((trial / "artifacts").glob(f"attempt_*/*{suffix}")),
    ]
    if not candidates:
        raise FileNotFoundError(f"no Gemini *{suffix} below {trial}")
    # Highest attempt number sorts last and is normally the submitted result.
    return candidates[-1]


def parse_win_limits(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    values = {key.lower(): int(value) for key, value in WIN_INT_RE.findall(path.read_text(errors="replace"))}
    return values.get("num_iter"), values.get("dis_num_iter")


def safe_log(value: float | None) -> float | None:
    return math.log1p(value) if value is not None and value >= 0 else None


def safe_log_ratio(a: float | None, b: float | None) -> float | None:
    return math.log(a / b) if a is not None and b is not None and a > 0 and b > 0 else None


def spread_stats(parsed: dict[str, Any]) -> dict[str, float | None]:
    final = np.asarray(parsed["final_wf_spreads"], dtype=float)
    initial = np.asarray(parsed["initial_wf_spreads"], dtype=float)
    total = number(parsed["omega_total"])
    if total is None and final.size:
        total = float(final.sum())
    median = float(np.median(final)) if final.size else None
    maximum = float(np.max(final)) if final.size else None
    initial_total = float(initial.sum()) if initial.size else None
    reduction = (
        (initial_total - total) / initial_total
        if initial_total is not None and total is not None and initial_total > 0
        else None
    )
    return {
        "num_parsed_wf": int(final.size),
        "final_spread_per_wf": total / final.size if total is not None and final.size else None,
        "median_wf_spread": median,
        "max_wf_spread": maximum,
        "max_to_median": maximum / median if maximum is not None and median and median > 0 else None,
        "fractional_spread_reduction": reduction,
    }

def gradient_colors(values: list[int], cmap: LinearSegmentedColormap) -> dict[int, str]:
    """Map sorted integer labels to evenly spaced colours from a gradient."""
    labels = sorted(values)
    if len(labels) == 1:
        return {labels[0]: to_hex(cmap(0.5))}
    return {
        label: to_hex(cmap(i / (len(labels) - 1)))
        for i, label in enumerate(labels)
    }


def error_gradient_colors(error_ratio: pd.Series) -> pd.Series:
    """Map interpolation error ratios to a pink-purple gradient."""
    numeric = pd.to_numeric(error_ratio, errors="coerce")

    if numeric.notna().sum() == 0:
        return pd.Series(MISSING_COLOR, index=error_ratio.index)

    # Log scale is usually better because ratios like 2x, 10x, 100x otherwise compress.
    log_error = np.log10(numeric.clip(lower=1e-12))

    norm = Normalize(
        vmin=float(log_error.quantile(0.05)),
        vmax=float(log_error.quantile(0.95)),
        clip=True,
    )

    return log_error.map(
        lambda value: MISSING_COLOR if pd.isna(value) else to_hex(ERROR_CMAP(norm(value)))
    )

def error_category(ratio: float | None) -> str:
    if ratio is None:
        return "missing"
    if ratio < 2:
        return "<2x"
    if ratio < 5:
        return "2-5x"
    if ratio < 10:
        return "5-10x"
    return ">=10x"


def load_projection_categories(path: Path) -> dict[str, str]:
    """Load material -> projection category from classify_gemini_projection_modes.py JSON."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}

    for cohort in data.get("cohorts", {}).values():
        if not isinstance(cohort, dict):
            continue

        details = cohort.get("details", [])
        if isinstance(details, list):
            for item in details:
                if not isinstance(item, dict):
                    continue
                material = item.get("material")
                category = item.get("projection_category")
                if isinstance(material, str) and isinstance(category, str):
                    result[material] = category

        materials = cohort.get("materials", {})
        if isinstance(materials, dict):
            for category, names in materials.items():
                if not isinstance(category, str) or not isinstance(names, list):
                    continue
                for material in names:
                    if isinstance(material, str):
                        result.setdefault(material, category)

    return result


def projection_color(category: str | None) -> str:
    if not category:
        return MISSING_COLOR
    return PROJECTION_COLORS.get(category, MISSING_COLOR)


def record_row(record: dict[str, Any], jobs_root: Path, dataset: Path) -> dict[str, Any]:
    material = str(record.get("material") or record.get("material_from_folder"))
    trial = find_trial(record, jobs_root)
    wout = gemini_file(trial, ".wout")
    win_candidates = list(wout.parent.glob("*.win"))
    num_iter_limit, _ = parse_win_limits(win_candidates[0]) if win_candidates else (None, None)
    reference_wout = dataset / material / "tests/reference/wannier/output/wannier90/aiida.wout"
    gemini = parse_wout(wout)
    reference = parse_wout(reference_wout)
    gs = spread_stats(gemini)
    rs = spread_stats(reference)
    total = number(gemini["omega_total"])
    omega_i = number(gemini["omega_I"])
    omega_od = number(gemini["omega_OD"])
    iters = number(gemini["localization_iterations"])
    if num_iter_limit is None:
        num_iter_limit = gemini["localization_iteration_limit"]
    ratio = number(record.get("gemini_to_reference_rmse_ratio"))
    if ratio is None:
        gemini_error = number(record.get("rmse_eV"))
        reference_error = number(record.get("reference_offmesh_rmse_eV"))
        if gemini_error is not None and reference_error and reference_error > 0:
            ratio = gemini_error / reference_error
    row: dict[str, Any] = {
        "material": material,
        "job_folder": record.get("job_folder"),
        "trial_folder": trial.name,
        "wout_path": str(wout),
        "reference_wout_path": str(reference_wout),
        "num_wann": number(record.get("num_target_bands") or record.get("num_wann_from_folder")),
        "gemini_error_eV": number(record.get("rmse_eV")),
        "reference_error_eV": number(record.get("reference_offmesh_rmse_eV")),
        "error_ratio": ratio,
        "error_category": error_category(ratio),
        "warning_count": gemini["warning_count"],
        "disentanglement_converged": gemini["disentanglement_converged"],
        "localization_iterations": iters,
        "localization_iteration_limit": num_iter_limit,
        **{f"gemini_{key}": value for key, value in gs.items()},
        **{f"reference_{key}": value for key, value in rs.items()},
        "omega_I_A2": omega_i,
        "omega_D_A2": number(gemini["omega_D"]),
        "omega_OD_A2": omega_od,
        "omega_total_A2": total,
        "omega_I_fraction": omega_i / total if omega_i is not None and total and total > 0 else None,
        "omega_OD_fraction": omega_od / total if omega_od is not None and total and total > 0 else None,
    }
    row.update({
        "log_final_spread_per_wf": safe_log(gs["final_spread_per_wf"]),
        "log_final_spread_per_wf_vs_ref": safe_log_ratio(gs["final_spread_per_wf"], rs["final_spread_per_wf"]),
        "log_max_wf_spread": safe_log(gs["max_wf_spread"]),
        "log_max_wf_spread_vs_ref": safe_log_ratio(gs["max_wf_spread"], rs["max_wf_spread"]),
        "log_max_to_median": safe_log(gs["max_to_median"]),
        "fractional_spread_reduction": gs["fractional_spread_reduction"],
        "log_error_ratio": math.log10(ratio) if ratio is not None and ratio > 0 else None,
    })
    return row


def build_rows(summary: Path, jobs_root: Path, dataset: Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    data = json.loads(summary.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for record in data.get("results", []):
        if not isinstance(record, dict) or record.get("successful") is not True:
            continue
        try:
            rows.append(record_row(record, jobs_root, dataset))
        except Exception as exc:
            failures.append({
                "material": str(record.get("material") or record.get("material_from_folder")),
                "job_folder": str(record.get("job_folder")),
                "error": f"{type(exc).__name__}: {exc}",
            })
    return pd.DataFrame(rows), failures


def cluster(
    df: pd.DataFrame,
    features: dict[str, str],
    n_clusters: int,
    cluster_column: str,
    order_column: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    feature_df = df[list(features)].apply(pd.to_numeric, errors="coerce")
    usable = feature_df.columns[feature_df.notna().sum() >= 2]
    dropped = [name for name in feature_df.columns if name not in usable]
    if dropped:
        print("Dropped entirely/mostly missing clustering features:", ", ".join(dropped))
    feature_df = feature_df[usable]
    clustered_rows = feature_df.notna().any(axis=1)
    if (~clustered_rows).any():
        names = ", ".join(df.loc[~clustered_rows, "material"].astype(str))
        print(f"Left out rows with no usable {cluster_column} features: {names}")
    feature_df = feature_df.loc[clustered_rows]
    if len(feature_df) < 2:
        raise RuntimeError("need at least two parsed calculations to cluster")
    oriented = feature_df.copy()
    for feature, direction in FEATURE_DIRECTIONS.items():
        if feature in oriented:
            oriented[feature] = oriented[feature] * direction
    medians = oriented.median(skipna=True)
    q25 = oriented.quantile(0.25)
    q75 = oriented.quantile(0.75)
    scale = (q75 - q25).replace(0, 1).fillna(1)
    display_scaled_df = (oriented - medians) / scale
    # Constant columns can be displayed but cannot contribute to distances.
    keep = display_scaled_df.max(axis=0) > display_scaled_df.min(axis=0)
    distance_df = display_scaled_df.loc[:, keep]
    if distance_df.shape[1] == 0:
        raise RuntimeError("all clustering features are constant")
    distances = nan_euclidean_distances(distance_df.to_numpy())
    finite_distances = distances[np.isfinite(distances)]
    fill_distance = float(finite_distances.max()) if finite_distances.size else 0.0
    distances = np.nan_to_num(distances, nan=fill_distance, posinf=fill_distance, neginf=0.0)
    np.fill_diagonal(distances, 0.0)
    z = linkage(squareform(distances, checks=False), method="average", optimal_ordering=True)
    result = df.copy()
    result[cluster_column] = np.nan
    result.loc[feature_df.index, cluster_column] = fcluster(z, t=min(n_clusters, len(feature_df)), criterion="maxclust")
    result[order_column] = np.nan
    order = leaves_list(z)
    for position, row_index in enumerate(order, start=1):
        result.loc[feature_df.index[row_index], order_column] = position
    display_scaled_df = display_scaled_df.rename(columns={name: features[name] for name in display_scaled_df.columns})
    return result, z, display_scaled_df


def make_plot(
    df: pd.DataFrame,
    z: np.ndarray,
    scaled: pd.DataFrame,
    output: Path,
    cluster_column: str,
    title: str,
    xlabel: str,
) -> None:
    plot_df = df.loc[scaled.index]
    cluster_labels = sorted(plot_df[cluster_column].dropna().unique())
    cluster_color = gradient_colors(
        cluster_labels,
        CLUSTER_CMAP,
    )

    row_colors = pd.DataFrame({
        "Cluster": plot_df[cluster_column].map(cluster_color).fillna(MISSING_CLUSTER_COLOR),
        "Error ratio": error_gradient_colors(plot_df["error_ratio"]),
        "Projection": plot_df["projection_category"].map(projection_color),
    }, index=plot_df.index)
    sns.set_theme(style="white", font_scale=0.72)
    height = max(14.0, min(42.0, 0.19 * len(df)))
    cmap = plt.get_cmap("vlag").copy()
    cmap.set_bad(MISSING_COLOR)
    grid = sns.clustermap(
        scaled,
        row_linkage=z,
        col_cluster=False,
        row_colors=row_colors,
        cmap=cmap,
        center=0,
        vmin=-3,
        vmax=3,
        mask=scaled.isna(),
        linewidths=0,
        figsize=(16, height),
        dendrogram_ratio=(0.13, 0.02),
        colors_ratio=0.025,
        cbar_pos=(0.02, 0.82, 0.018, 0.12),
        cbar_kws={"label": "Robust-scaled diagnostic\n(clipped at +/-3)"},
        yticklabels=plot_df["material"].tolist(),
    )
    grid.ax_heatmap.set_xlabel(xlabel)
    grid.ax_heatmap.set_ylabel("Material")
    grid.ax_heatmap.tick_params(axis="y", labelsize=6)
    grid.ax_heatmap.set_xticklabels(grid.ax_heatmap.get_xticklabels(), rotation=35, ha="right")
    legend = [Patch(facecolor=color, label=f"Cluster {int(cid)}") for cid, color in cluster_color.items()]
    legend += [Patch(facecolor=MISSING_CLUSTER_COLOR, label="No process cluster")]
    legend += [
        Patch(facecolor=to_hex(ERROR_CMAP(0.0)), label="Lower interpolation error"),
        Patch(facecolor=to_hex(ERROR_CMAP(1.0)), label="Higher interpolation error"),
        Patch(facecolor=MISSING_COLOR, label="Missing error"),
    ]
    legend += [
        Patch(facecolor=PROJECTION_COLORS["explicit_projection_runs"], label="Explicit projections"),
        Patch(facecolor=PROJECTION_COLORS["random_projection_runs"], label="Random projections"),
        Patch(facecolor=PROJECTION_COLORS["none_or_implicit_projection_runs"], label="None/implicit projections"),
        Patch(facecolor=MISSING_COLOR, label="Missing projection category"),
    ]
    grid.ax_col_dendrogram.legend(
        handles=legend, title="Row annotations", loc="center", ncol=min(5, len(legend)), frameon=False
    )
    grid.fig.suptitle(title, y=0.997)
    grid.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    grid.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(grid.fig)


def write_summary(
    df: pd.DataFrame,
    features: dict[str, str],
    cluster_column: str,
    order_column: str,
    path: Path,
) -> None:
    rows = []
    clustered_df = df[df[cluster_column].notna()]
    for cluster_id, group in clustered_df.groupby(cluster_column, sort=True):
        ordered = group.sort_values(order_column)
        valid_error = group["error_ratio"].dropna()
        row: dict[str, Any] = {
            "cluster": int(cluster_id),
            "n_materials": len(group),
            "n_with_error_ratio": len(valid_error),
            "fraction_error_ge_2x": float((valid_error >= 2).mean()) if len(valid_error) else None,
            "median_error_ratio": float(valid_error.median()) if len(valid_error) else None,
            "materials_in_plot_order": "; ".join(ordered["material"]),
        }
        for feature in features:
            if feature in group:
                row[f"median_{feature}"] = group[feature].median(skipna=True)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--projection-categories",
        type=Path,
        default=DEFAULT_PROJECTION_CATEGORIES,
        help="JSON from scripts/classify_gemini_projection_modes.py used for projection-mode row colours",
    )
    parser.add_argument("--clusters", type=int, default=10, help="number of coloured dendrogram groups")
    args = parser.parse_args()
    if args.clusters < 2:
        raise SystemExit("--clusters must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, failures = build_rows(args.summary, args.jobs_root, args.dataset)
    projection_categories = load_projection_categories(args.projection_categories)
    df["projection_category"] = df["material"].map(projection_categories)
    df["projection_color_hex"] = df["projection_category"].map(projection_color)
    df = df.reset_index(drop=True)
    pd.DataFrame(failures, columns=["material", "job_folder", "error"]).to_csv(
        args.output_dir / "gemini_wout_parse_failures.csv", index=False
    )
    clustered, z_process, scaled_process = cluster(
        df,
        PROCESS_FEATURES,
        args.clusters,
        "process_cluster",
        "process_dendrogram_order_top_to_bottom",
    )
    clustered, z_all, scaled_all = cluster(
        clustered,
        ALL_FEATURES,
        args.clusters,
        "all_columns_cluster",
        "all_columns_dendrogram_order_top_to_bottom",
    )
    process_cluster_colors = gradient_colors(
        sorted(clustered["process_cluster"].dropna().unique()),
        CLUSTER_CMAP,
    )
    all_cluster_colors = gradient_colors(
        sorted(clustered["all_columns_cluster"].dropna().unique()),
        CLUSTER_CMAP,
    )

    clustered["cluster"] = clustered["process_cluster"]
    clustered["dendrogram_order_top_to_bottom"] = clustered["process_dendrogram_order_top_to_bottom"]
    clustered["cluster_color_hex"] = clustered["process_cluster"].map(process_cluster_colors).fillna(MISSING_CLUSTER_COLOR)
    clustered["process_cluster_color_hex"] = clustered["process_cluster"].map(process_cluster_colors).fillna(MISSING_CLUSTER_COLOR)
    clustered["all_columns_cluster_color_hex"] = clustered["all_columns_cluster"].map(all_cluster_colors).fillna(MISSING_CLUSTER_COLOR)
    clustered["error_color_hex"] = error_gradient_colors(clustered["error_ratio"])
    clustered["projection_color_hex"] = clustered["projection_category"].map(projection_color)
    clustered.sort_values("process_dendrogram_order_top_to_bottom").to_csv(
        args.output_dir / "gemini_wout_cluster_assignments.csv", index=False
    )
    clustered.to_csv(args.output_dir / "gemini_wout_features.csv", index=False)
    write_summary(
        clustered,
        PROCESS_FEATURES,
        "process_cluster",
        "process_dendrogram_order_top_to_bottom",
        args.output_dir / "gemini_wout_cluster_summary.csv",
    )
    write_summary(
        clustered,
        ALL_FEATURES,
        "all_columns_cluster",
        "all_columns_dendrogram_order_top_to_bottom",
        args.output_dir / "gemini_wout_cluster_summary_all_columns.csv",
    )
    make_plot(
        clustered,
        z_process,
        scaled_process,
        args.output_dir / "gemini_wout_cluster_heatmap_process",
        "process_cluster",
        (
            "Gemini Wannier90 process signatures\n"
            "Rows clustered without interpolation error; missing cells are blank"
        ),
        "Process diagnostic (interpolation error excluded from clustering)",
    )
    make_plot(
        clustered,
        z_all,
        scaled_all,
        args.output_dir / "gemini_wout_cluster_heatmap_all_columns",
        "all_columns_cluster",
        (
            "Gemini Wannier90 process and outcome signatures\n"
            "Rows clustered with interpolation error included; missing cells are blank"
        ),
        "Diagnostic including interpolation error",
    )

    print(f"Parsed calculations: {len(clustered)}")
    print(f"Parse failures: {len(failures)}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print("Wrote process and all-column heatmaps.")
    print("Trace any row via gemini_wout_cluster_assignments.csv (material, order, colours, and .wout path).")


if __name__ == "__main__":
    main()
