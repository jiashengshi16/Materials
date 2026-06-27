#!/usr/bin/env python3
"""Compare projection modes for rerun materials against their original jobs.

Hardcoded paths only, by request.  Projection categories use the same rule as
the existing Gemini projection analysis:

* begin projections containing only random -> random_projection_runs
* any other begin projections block -> explicit_projection_runs
* no projections block -> none_or_implicit_projection_runs
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgba
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from cluster_gemini_wout_failures import (
    PROCESS_FEATURES,
    FEATURE_DIRECTIONS,
    MISSING_COLOR,
    PROJECTION_COLORS,
    number,
    parse_wout,
    safe_log,
    safe_log_ratio,
    spread_stats,
)

JOBS_ROOT = ROOT / "jobs"
RERUNS_ROOT = ROOT / "reruns"
SUMMARY_PATH = JOBS_ROOT / "num_wann_ordered_diagnostics_summary.json"
DATASET_ROOT = ROOT / "harbor_datasets" / "wannier_200"
OUTPUT_CSV = RERUNS_ROOT / "projection_mode_comparison.csv"
OUTPUT_JSON = RERUNS_ROOT / "projection_mode_comparison_summary.json"
OUTPUT_ERROR_CSV = RERUNS_ROOT / "projection_error_ratio_comparison.csv"
OUTPUT_ERROR_JSON = RERUNS_ROOT / "projection_error_ratio_comparison.json"
OUTPUT_HEATMAP = RERUNS_ROOT / "projection_mode_delta_heatmap"

ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error_pink_to_purple",
    ["#F7B6D2", "#6A00A8"],
)
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "delta_blue_white_red",
    ["#2D70B8", "#FFFFFF", "#B83A3A"],
)
ERROR_RATIO_COLUMNS = ("jobs error ratio", "rerun error ratio")
PROJECTION_COLUMNS = ("jobs projection", "rerun projection")

CATEGORY_KEYS = (
    "random_projection_runs",
    "explicit_projection_runs",
    "none_or_implicit_projection_runs",
)

PROJECTION_BLOCK_RE = re.compile(
    r"^\s*begin\s+projections\s*$([\s\S]*?)^\s*end\s+projections\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def material_from_job_folder(path: Path) -> str:
    return path.name.rsplit("__", 1)[-1]


def projection_block(text: str) -> str | None:
    match = PROJECTION_BLOCK_RE.search(text)
    return match.group(1) if match else None


def classify_win(win_path: Path | None) -> str:
    if win_path is None or not win_path.exists():
        return "none_or_implicit_projection_runs"

    block = projection_block(win_path.read_text(errors="ignore"))
    if block is None:
        return "none_or_implicit_projection_runs"

    content_lines = []
    for line in block.splitlines():
        line = line.split("!", 1)[0].strip()
        if line:
            content_lines.append(line.lower())

    if content_lines == ["random"]:
        return "random_projection_runs"
    return "explicit_projection_runs"


def attempt_number(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"attempt_(\d+)", part)
        if match:
            return int(match.group(1))
    return -1


def sort_submitted_candidates(candidates: list[Path]) -> list[Path]:
    return sorted(
        candidates,
        key=lambda path: (
            attempt_number(path),
            "artifacts/artifacts" not in str(path),
            str(path),
        ),
    )


def find_trial(job_folder: Path) -> Path | None:
    trials = [
        path
        for path in sorted(job_folder.iterdir())
        if path.is_dir() and (path / "verifier").is_dir()
    ]
    if len(trials) == 1:
        return trials[0]
    return None


def find_submitted_file(job_folder: Path, material: str, suffix: str) -> Path | None:
    trial = find_trial(job_folder)
    if trial is None:
        return None

    candidates = [
        *sorted((trial / "artifacts" / "artifacts").glob(f"attempt_*/*{material}{suffix}")),
        *sorted((trial / "artifacts").glob(f"attempt_*/*{material}{suffix}")),
    ]
    if not candidates:
        candidates = [
            *sorted((trial / "artifacts" / "artifacts").glob(f"attempt_*/*{suffix}")),
            *sorted((trial / "artifacts").glob(f"attempt_*/*{suffix}")),
        ]
    if not candidates:
        return None

    return sort_submitted_candidates(candidates)[-1]


def find_submitted_win(job_folder: Path, material: str) -> Path | None:
    return find_submitted_file(job_folder, material, ".win")


def find_submitted_wout(job_folder: Path, material: str) -> Path | None:
    return find_submitted_file(job_folder, material, ".wout")


def job_folders_by_material(root: Path) -> dict[str, list[Path]]:
    by_material: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name.startswith("num_wann_ordered__"):
            by_material[material_from_job_folder(path)].append(path)
    return dict(by_material)


def category_counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts = Counter(row[key] for row in rows)
    return {category: counts.get(category, 0) for category in CATEGORY_KEYS}


def category_materials(rows: list[dict[str, object]], key: str) -> dict[str, list[str]]:
    materials: dict[str, list[str]] = {category: [] for category in CATEGORY_KEYS}
    for row in rows:
        category = row.get(key)
        material = row.get("material")
        if isinstance(category, str) and isinstance(material, str) and category in materials:
            materials[category].append(material)
    return {category: sorted(names) for category, names in materials.items()}


def finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def load_reference_rmse_by_material() -> dict[str, float]:
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    result: dict[str, float] = {}
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        material = item.get("material") or item.get("material_from_folder")
        reference_rmse = finite(item.get("reference_offmesh_rmse_eV"))
        if isinstance(material, str) and reference_rmse is not None:
            result[material] = reference_rmse
    return result


def load_result_metrics(job_folder: Path) -> dict[str, float]:
    result_path = job_folder / "result.json"
    if not result_path.is_file():
        return {}
    data = json.loads(result_path.read_text(encoding="utf-8"))
    evals = data.get("stats", {}).get("evals", {})
    if not isinstance(evals, dict):
        return {}
    for eval_data in evals.values():
        if not isinstance(eval_data, dict):
            continue
        metrics = eval_data.get("metrics", [])
        if isinstance(metrics, list) and metrics and isinstance(metrics[0], dict):
            return metrics[0]
    return {}


def error_ratio(job_folder: Path, reference_rmse: float | None) -> tuple[float | None, float | None]:
    metrics = load_result_metrics(job_folder)
    rmse = finite(metrics.get("rmse_eV"))
    ratio = rmse / reference_rmse if rmse is not None and reference_rmse and reference_rmse > 0 else None
    return rmse, ratio


def process_feature_row(wout_path: Path | None, reference_wout: Path) -> dict[str, float | None]:
    if wout_path is None or not wout_path.is_file() or not reference_wout.is_file():
        return {feature: None for feature in PROCESS_FEATURES}

    gemini = parse_wout(wout_path)
    reference = parse_wout(reference_wout)
    gs = spread_stats(gemini)
    rs = spread_stats(reference)
    total = number(gemini["omega_total"])
    omega_i = number(gemini["omega_I"])
    omega_od = number(gemini["omega_OD"])
    raw = {
        "log_final_spread_per_wf": safe_log(gs["final_spread_per_wf"]),
        "log_final_spread_per_wf_vs_ref": safe_log_ratio(gs["final_spread_per_wf"], rs["final_spread_per_wf"]),
        "log_max_wf_spread": safe_log(gs["max_wf_spread"]),
        "log_max_wf_spread_vs_ref": safe_log_ratio(gs["max_wf_spread"], rs["max_wf_spread"]),
        "log_max_to_median": safe_log(gs["max_to_median"]),
        "omega_I_fraction": omega_i / total if omega_i is not None and total and total > 0 else None,
        "omega_OD_fraction": omega_od / total if omega_od is not None and total and total > 0 else None,
        "fractional_spread_reduction": gs["fractional_spread_reduction"],
    }
    return raw


def oriented_feature_value(feature: str, value: float | None) -> float | None:
    if value is None:
        return None
    return value * FEATURE_DIRECTIONS.get(feature, 1)


def relative(path: Path | None) -> str:
    return str(path.relative_to(ROOT)) if path else ""


def error_ratio_color_values(values: pd.DataFrame) -> np.ndarray:
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if numeric.notna().sum().sum() == 0:
        return np.ones((*numeric.shape, 4))
    log_values = np.log10(numeric.clip(lower=1e-12))
    finite_values = log_values.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    norm = Normalize(
        vmin=float(np.quantile(finite_values, 0.05)),
        vmax=float(np.quantile(finite_values, 0.95)),
        clip=True,
    )
    rgba = np.ones((*numeric.shape, 4))
    for row_index in range(numeric.shape[0]):
        for col_index in range(numeric.shape[1]):
            value = log_values.iat[row_index, col_index]
            if pd.notna(value):
                rgba[row_index, col_index] = ERROR_CMAP(norm(float(value)))
    return rgba


def projection_color_values(values: pd.DataFrame) -> np.ndarray:
    rgba = np.ones((*values.shape, 4))
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            color = PROJECTION_COLORS.get(str(values.iat[row_index, col_index]), MISSING_COLOR)
            rgba[row_index, col_index] = to_rgba(color)
    return rgba


def projection_code(category: object) -> str:
    return {
        "explicit_projection_runs": "E",
        "random_projection_runs": "R",
        "none_or_implicit_projection_runs": "N",
    }.get(str(category), "?")


def make_delta_heatmap(rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    feature_columns = [f"delta_{feature}" for feature in PROCESS_FEATURES]
    labels = {f"delta_{key}": label for key, label in PROCESS_FEATURES.items()}
    df["delta_log_error_ratio"] = np.log10(pd.to_numeric(df["rerun_error_ratio"], errors="coerce")) - np.log10(
        pd.to_numeric(df["jobs_error_ratio"], errors="coerce")
    )
    feature_columns.append("delta_log_error_ratio")
    labels["delta_log_error_ratio"] = "interpolation error badness"

    df = df.sort_values(["delta_log_error_ratio", "material"], na_position="last").reset_index(drop=True)
    heatmap_df = df[feature_columns].apply(pd.to_numeric, errors="coerce").rename(columns=labels)
    ratio_df = df[["jobs_error_ratio", "rerun_error_ratio"]].rename(
        columns={
            "jobs_error_ratio": ERROR_RATIO_COLUMNS[0],
            "rerun_error_ratio": ERROR_RATIO_COLUMNS[1],
        }
    )
    projection_df = df[["jobs_projection_category", "rerun_projection_category"]].rename(
        columns={
            "jobs_projection_category": PROJECTION_COLUMNS[0],
            "rerun_projection_category": PROJECTION_COLUMNS[1],
        }
    )

    abs_values = np.abs(heatmap_df.to_numpy(dtype=float))
    finite_abs_values = abs_values[np.isfinite(abs_values)]
    vmax = float(np.quantile(finite_abs_values, 0.95)) if finite_abs_values.size else 1.0
    vmax = max(vmax, 1e-9)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    sns.set_theme(style="white", font_scale=0.82)
    height = max(8.5, 0.25 * len(df) + 2.0)
    fig = plt.figure(figsize=(17.0, height))
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[0.95, 0.95, 7.0, 0.35],
        left=0.08,
        right=0.92,
        top=0.90,
        bottom=0.16,
        wspace=0.04,
    )
    ratio_ax = fig.add_subplot(gs[0, 0])
    projection_ax = fig.add_subplot(gs[0, 1], sharey=ratio_ax)
    heatmap_ax = fig.add_subplot(gs[0, 2], sharey=ratio_ax)
    cbar_ax = fig.add_subplot(gs[0, 3])

    nrows = len(df)

    ratio_ax.imshow(
        error_ratio_color_values(ratio_df),
        aspect="auto",
        interpolation="nearest",
        extent=(0, len(ratio_df.columns), nrows, 0),
    )

    projection_ax.imshow(
        projection_color_values(projection_df),
        aspect="auto",
        interpolation="nearest",
        extent=(0, len(projection_df.columns), nrows, 0),
    )
    for row_index in range(len(projection_df)):
        for col_index in range(len(projection_df.columns)):
            category = projection_df.iat[row_index, col_index]
            color = "white" if category in {"random_projection_runs", "none_or_implicit_projection_runs"} else "black"
            projection_ax.text(
                col_index + 0.5,
                row_index + 0.5,
                projection_code(category),
                ha="center",
                va="center",
                fontsize=6,
                color=color,
                fontweight="bold",
            )

    sns.heatmap(
        heatmap_df,
        ax=heatmap_ax,
        cmap=DELTA_CMAP,
        norm=norm,
        mask=heatmap_df.isna(),
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"label": "Delta badness, rerun - jobs\nblue improves, red worsens"},
        linewidths=0,
        yticklabels=df["material"].tolist(),
    )

    for ax, columns in ((ratio_ax, ERROR_RATIO_COLUMNS), (projection_ax, PROJECTION_COLUMNS)):
        ax.set_xticks(np.arange(len(columns)) + 0.5)
        ax.set_xticklabels(columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(df)) + 0.5)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ratio_ax.set_yticklabels(df["material"].tolist(), fontsize=7)
    projection_ax.tick_params(labelleft=False)
    heatmap_ax.tick_params(axis="y", labelleft=False, labelright=True, right=False, length=0)
    heatmap_ax.set_yticklabels(df["material"].tolist(), rotation=0, fontsize=7)
    heatmap_ax.set_xlabel("Process diagnostic delta, reruns vs jobs")
    heatmap_ax.set_ylabel("Material")
    heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), rotation=35, ha="right")
    ratio_ax.set_ylabel("Material")

    legend = [
        Patch(facecolor=PROJECTION_COLORS["explicit_projection_runs"], label="Explicit projections"),
        Patch(facecolor=PROJECTION_COLORS["random_projection_runs"], label="Random projections"),
        Patch(facecolor=PROJECTION_COLORS["none_or_implicit_projection_runs"], label="None/implicit projections"),
        Patch(facecolor=MISSING_COLOR, label="Missing projection"),
        Patch(facecolor=ERROR_CMAP(0.15), label="Lower error ratio"),
        Patch(facecolor=ERROR_CMAP(0.95), label="Higher error ratio"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.52, 0.985))
    fig.suptitle("Rerun deltas for materials also present in jobs", y=0.995)
    fig.savefig(OUTPUT_HEATMAP.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_HEATMAP.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    jobs_by_material = job_folders_by_material(JOBS_ROOT)
    reruns_by_material = job_folders_by_material(RERUNS_ROOT)
    reference_rmse_by_material = load_reference_rmse_by_material()

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for material, rerun_folders in sorted(reruns_by_material.items()):
        jobs_folders = jobs_by_material.get(material, [])
        reference_rmse = reference_rmse_by_material.get(material)
        if len(jobs_folders) != 1 or len(rerun_folders) != 1 or reference_rmse is None:
            skipped.append(
                {
                    "material": material,
                    "jobs_matches": str(len(jobs_folders)),
                    "rerun_matches": str(len(rerun_folders)),
                    "has_reference_rmse": str(reference_rmse is not None),
                }
            )
            continue

        jobs_folder = jobs_folders[0]
        rerun_folder = rerun_folders[0]
        jobs_win = find_submitted_win(jobs_folder, material)
        rerun_win = find_submitted_win(rerun_folder, material)
        jobs_wout = find_submitted_wout(jobs_folder, material)
        rerun_wout = find_submitted_wout(rerun_folder, material)
        jobs_category = classify_win(jobs_win)
        rerun_category = classify_win(rerun_win)
        jobs_rmse, jobs_ratio = error_ratio(jobs_folder, reference_rmse)
        rerun_rmse, rerun_ratio = error_ratio(rerun_folder, reference_rmse)
        delta_ratio = (
            rerun_ratio - jobs_ratio
            if rerun_ratio is not None and jobs_ratio is not None
            else None
        )
        ratio_fold_change = (
            rerun_ratio / jobs_ratio
            if rerun_ratio is not None and jobs_ratio is not None and jobs_ratio > 0
            else None
        )
        delta_log10_ratio = (
            math.log10(rerun_ratio) - math.log10(jobs_ratio)
            if rerun_ratio is not None and jobs_ratio is not None and rerun_ratio > 0 and jobs_ratio > 0
            else None
        )
        reference_wout = DATASET_ROOT / material / "tests/reference/wannier/output/wannier90/aiida.wout"
        jobs_features = process_feature_row(jobs_wout, reference_wout)
        rerun_features = process_feature_row(rerun_wout, reference_wout)
        deltas = {}
        for feature in PROCESS_FEATURES:
            jobs_value = oriented_feature_value(feature, jobs_features.get(feature))
            rerun_value = oriented_feature_value(feature, rerun_features.get(feature))
            deltas[f"delta_{feature}"] = (
                rerun_value - jobs_value
                if rerun_value is not None and jobs_value is not None
                else None
            )

        row: dict[str, object] = {
            "material": material,
            "reference_rmse_eV": reference_rmse,
            "jobs_rmse_eV": jobs_rmse,
            "rerun_rmse_eV": rerun_rmse,
            "jobs_error_ratio": jobs_ratio,
            "rerun_error_ratio": rerun_ratio,
            "delta_error_ratio_rerun_minus_jobs": delta_ratio,
            "ratio_fold_change_rerun_over_jobs": ratio_fold_change,
            "delta_log10_error_ratio_rerun_minus_jobs": delta_log10_ratio,
            "jobs_projection_category": jobs_category,
            "rerun_projection_category": rerun_category,
            "changed_projection_category": str(jobs_category != rerun_category),
            "jobs_was_random": str(jobs_category == "random_projection_runs"),
            "rerun_still_random": str(rerun_category == "random_projection_runs"),
            "jobs_job_folder": jobs_folder.name,
            "rerun_job_folder": rerun_folder.name,
            "jobs_win_path": relative(jobs_win),
            "rerun_win_path": relative(rerun_win),
            "jobs_wout_path": relative(jobs_wout),
            "rerun_wout_path": relative(rerun_wout),
        }
        row.update(deltas)
        rows.append(row)

    transitions = {
        original: {rerun: 0 for rerun in CATEGORY_KEYS}
        for original in CATEGORY_KEYS
    }
    for row in rows:
        transitions[row["jobs_projection_category"]][row["rerun_projection_category"]] += 1

    jobs_random_rows = [
        row for row in rows
        if row["jobs_projection_category"] == "random_projection_runs"
    ]
    jobs_random_rerun_counts = category_counts(
        jobs_random_rows,
        "rerun_projection_category",
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    error_fields = [
        "material",
        "reference_rmse_eV",
        "jobs_rmse_eV",
        "rerun_rmse_eV",
        "jobs_error_ratio",
        "rerun_error_ratio",
        "delta_error_ratio_rerun_minus_jobs",
        "ratio_fold_change_rerun_over_jobs",
        "delta_log10_error_ratio_rerun_minus_jobs",
        "jobs_projection_category",
        "rerun_projection_category",
        "jobs_job_folder",
        "rerun_job_folder",
    ]
    with OUTPUT_ERROR_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in error_fields} for row in rows])
    OUTPUT_ERROR_JSON.write_text(
        json.dumps(
            [
                {field: row.get(field) for field in error_fields}
                for row in rows
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    make_delta_heatmap(rows)

    summary = {
        "paths": {
            "jobs_root": str(JOBS_ROOT.relative_to(ROOT)),
            "reruns_root": str(RERUNS_ROOT.relative_to(ROOT)),
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "error_ratio_csv": str(OUTPUT_ERROR_CSV.relative_to(ROOT)),
            "error_ratio_json": str(OUTPUT_ERROR_JSON.relative_to(ROOT)),
            "heatmap_png": str(OUTPUT_HEATMAP.with_suffix(".png").relative_to(ROOT)),
            "heatmap_pdf": str(OUTPUT_HEATMAP.with_suffix(".pdf").relative_to(ROOT)),
        },
        "classification": (
            "begin projections with only random => random_projection_runs; "
            "any other projections block => explicit_projection_runs; "
            "no block or missing .win => none_or_implicit_projection_runs"
        ),
        "compared_materials": len(rows),
        "skipped_materials": skipped,
        "jobs_counts_for_same_materials": category_counts(
            rows,
            "jobs_projection_category",
        ),
        "rerun_counts_for_same_materials": category_counts(
            rows,
            "rerun_projection_category",
        ),
        "jobs_materials_by_projection_category": category_materials(
            rows,
            "jobs_projection_category",
        ),
        "rerun_materials_by_projection_category": category_materials(
            rows,
            "rerun_projection_category",
        ),
        "transition_counts_jobs_to_rerun": transitions,
        "rerun_counts_for_jobs_random_materials": jobs_random_rerun_counts,
        "jobs_random_materials": sorted(
            row["material"] for row in jobs_random_rows
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Compared materials: {len(rows)}")
    print(f"Skipped materials: {len(skipped)}")
    print("Jobs counts for same materials:")
    print(json.dumps(summary["jobs_counts_for_same_materials"], indent=2))
    print("Rerun counts for same materials:")
    print(json.dumps(summary["rerun_counts_for_same_materials"], indent=2))
    print("Transition counts, jobs -> rerun:")
    print(json.dumps(transitions, indent=2))
    print("Rerun counts among materials whose jobs run used random projections:")
    print(json.dumps(jobs_random_rerun_counts, indent=2))
    print(f"Wrote {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_ERROR_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_ERROR_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_HEATMAP.with_suffix('.png').relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_HEATMAP.with_suffix('.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
