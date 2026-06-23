#!/usr/bin/env python3
"""Cluster Gemini calculations by localization/disentanglement symptoms in .wout.

The clustering deliberately does not use interpolation error.  Error ratio is
shown as a row-colour annotation, so enrichment of high-error calculations in
a cluster is an independent observation rather than a built-in result.

Outputs:
  gemini_wout_cluster_heatmap.png/pdf  labelled heatmap and dendrogram
  gemini_wout_cluster_assignments.csv  one traceable row per calculation
  gemini_wout_cluster_summary.csv      cluster-level error/feature summary
  gemini_wout_features.csv             all raw and transformed diagnostics
"""

from __future__ import annotations
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


DEFAULT_SUMMARY = ROOT / "jobs" / "num_wann_ordered_diagnostics_summary.json"
DEFAULT_JOBS = ROOT / "jobs"
DEFAULT_DATASET = ROOT / "harbor_datasets" / "wannier_200"
DEFAULT_OUTPUT = ROOT / "jobs" / "gemini_wout_clustering"

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


# Human-readable names appear as columns in the heatmap.  The underlying raw
# values remain available in gemini_wout_features.csv.
FEATURES = {
    "log_final_spread_per_wf": "log final spread/WF",
    "log_final_spread_per_wf_vs_ref": "log spread/WF vs ref",
    "log_max_wf_spread": "log worst-WF spread",
    "log_max_wf_spread_vs_ref": "log worst WF vs ref",
    "log_max_to_median": "log worst/median WF",
    "omega_I_fraction": "Omega I fraction",
    "omega_OD_fraction": "Omega OD fraction",
    "fractional_spread_reduction": "spread reduction",
    "localization_iteration_fraction": "localization iter fraction",
    "localization_reached_limit": "localization reached limit",
    "disentanglement_converged": "disentanglement converged",
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
    iter_fraction = iters / num_iter_limit if iters is not None and num_iter_limit else None
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
        "localization_iteration_fraction": iter_fraction,
        "localization_reached_limit": float(iters >= num_iter_limit) if iters is not None and num_iter_limit else None,
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


def cluster(df: pd.DataFrame, n_clusters: int) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    feature_df = df[list(FEATURES)].apply(pd.to_numeric, errors="coerce")
    usable = feature_df.columns[feature_df.notna().sum() >= 2]
    dropped = [name for name in feature_df.columns if name not in usable]
    if dropped:
        print("Dropped entirely/mostly missing clustering features:", ", ".join(dropped))
    feature_df = feature_df[usable]
    if len(feature_df) < 2:
        raise RuntimeError("need at least two parsed calculations to cluster")
    imputed = SimpleImputer(strategy="median").fit_transform(feature_df)
    scaled = RobustScaler().fit_transform(imputed)
    # Prevent a constant binary column from contributing NaN/meaningless values.
    keep = np.ptp(scaled, axis=0) > 0
    scaled = scaled[:, keep]
    names = [FEATURES[name] for name, use in zip(feature_df.columns, keep) if use]
    if scaled.shape[1] == 0:
        raise RuntimeError("all clustering features are constant")
    z = linkage(scaled, method="ward", metric="euclidean", optimal_ordering=True)
    result = df.copy()
    result["cluster"] = fcluster(z, t=min(n_clusters, len(result)), criterion="maxclust")
    result["dendrogram_order_top_to_bottom"] = np.nan
    order = leaves_list(z)
    for position, row_index in enumerate(order, start=1):
        result.loc[result.index[row_index], "dendrogram_order_top_to_bottom"] = position
    scaled_df = pd.DataFrame(scaled, index=result.index, columns=names)
    return result, z, scaled_df


def make_plot(df: pd.DataFrame, z: np.ndarray, scaled: pd.DataFrame, output: Path) -> None:
    cluster_color = gradient_colors(
        sorted(df["cluster"].unique()),
        CLUSTER_CMAP,
    )

    row_colors = pd.DataFrame({
        "Cluster": df["cluster"].map(cluster_color),
        "Error ratio": error_gradient_colors(df["error_ratio"]),
    }, index=df.index)
    sns.set_theme(style="white", font_scale=0.72)
    height = max(14.0, min(42.0, 0.19 * len(df)))
    grid = sns.clustermap(
        scaled,
        row_linkage=z,
        col_cluster=False,
        row_colors=row_colors,
        cmap="vlag",
        center=0,
        vmin=-3,
        vmax=3,
        linewidths=0,
        figsize=(16, height),
        dendrogram_ratio=(0.13, 0.02),
        colors_ratio=0.025,
        cbar_pos=(0.02, 0.82, 0.018, 0.12),
        cbar_kws={"label": "Robust-scaled diagnostic\n(clipped at +/-3)"},
        yticklabels=df["material"].tolist(),
    )
    grid.ax_heatmap.set_xlabel(".wout diagnostic (not interpolation error)")
    grid.ax_heatmap.set_ylabel("Material")
    grid.ax_heatmap.tick_params(axis="y", labelsize=6)
    grid.ax_heatmap.set_xticklabels(grid.ax_heatmap.get_xticklabels(), rotation=35, ha="right")
    legend = [Patch(facecolor=color, label=f"Cluster {cid}") for cid, color in cluster_color.items()]
    legend += [
        Patch(facecolor=to_hex(ERROR_CMAP(0.0)), label="Lower interpolation error"),
        Patch(facecolor=to_hex(ERROR_CMAP(1.0)), label="Higher interpolation error"),
        Patch(facecolor=MISSING_COLOR, label="Missing error"),
    ]
    grid.ax_col_dendrogram.legend(
        handles=legend, title="Row annotations", loc="center", ncol=min(5, len(legend)), frameon=False
    )
    grid.fig.suptitle(
        "Gemini Wannier90 failure signatures\nClustering uses localization/disentanglement diagnostics; error ratio is annotation only",
        y=0.997,
    )
    grid.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    grid.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(grid.fig)


def write_summary(df: pd.DataFrame, path: Path) -> None:
    rows = []
    for cluster_id, group in df.groupby("cluster", sort=True):
        ordered = group.sort_values("dendrogram_order_top_to_bottom")
        valid_error = group["error_ratio"].dropna()
        row: dict[str, Any] = {
            "cluster": cluster_id,
            "n_materials": len(group),
            "n_with_error_ratio": len(valid_error),
            "fraction_error_ge_2x": float((valid_error >= 2).mean()) if len(valid_error) else None,
            "median_error_ratio": float(valid_error.median()) if len(valid_error) else None,
            "materials_in_plot_order": "; ".join(ordered["material"]),
        }
        for feature in FEATURES:
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
    parser.add_argument("--clusters", type=int, default=5, help="number of coloured dendrogram groups")
    args = parser.parse_args()
    if args.clusters < 2:
        raise SystemExit("--clusters must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, failures = build_rows(args.summary, args.jobs_root, args.dataset)
    # A row with no final spread has no localization signature to cluster.
    # Preserve it in the failure ledger instead of median-imputing every
    # diagnostic and making it look artificially ordinary.
    unusable = df["log_final_spread_per_wf"].isna()
    for _, row in df.loc[unusable].iterrows():
        failures.append({
            "material": str(row["material"]),
            "job_folder": str(row["job_folder"]),
            "error": "No parseable Final State/WF spreads in .wout",
        })
    df = df.loc[~unusable].reset_index(drop=True)
    if failures:
        pd.DataFrame(failures).to_csv(args.output_dir / "gemini_wout_parse_failures.csv", index=False)
    clustered, z, scaled = cluster(df, args.clusters)
    cluster_colors = gradient_colors(
        sorted(clustered["cluster"].unique()),
        CLUSTER_CMAP,
    )

    clustered["cluster_color_hex"] = clustered["cluster"].map(cluster_colors)
    clustered["error_color_hex"] = error_gradient_colors(clustered["error_ratio"])
    clustered.sort_values("dendrogram_order_top_to_bottom").to_csv(
        args.output_dir / "gemini_wout_cluster_assignments.csv", index=False
    )
    clustered.to_csv(args.output_dir / "gemini_wout_features.csv", index=False)
    write_summary(clustered, args.output_dir / "gemini_wout_cluster_summary.csv")
    make_plot(clustered, z, scaled, args.output_dir / "gemini_wout_cluster_heatmap")

    print(f"Parsed calculations: {len(clustered)}")
    print(f"Parse failures: {len(failures)}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print("Trace any row via gemini_wout_cluster_assignments.csv (material, order, colours, and .wout path).")


if __name__ == "__main__":
    main()
