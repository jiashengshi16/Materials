#!/usr/bin/env python3
"""Compare .win choice similarity for self-debug review runs against prior best runs."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import types
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgba
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

try:
    import seaborn  # noqa: F401
except ModuleNotFoundError:
    sys.modules["seaborn"] = types.SimpleNamespace()

from cluster_gemini_wout_failures import (
    PROCESS_FEATURES,
    FEATURE_DIRECTIONS,
    MISSING_COLOR,
    number,
    parse_wout,
    safe_log,
    safe_log_ratio,
    spread_stats,
)
from compare_wannier_choices import (
    compare_window_masks,
    find_eig_near_win,
    multiset_jaccard,
    parse_win,
)

JOBS_ROOT = ROOT / "jobs"
RERUNS_ROOT = ROOT / "reruns"
REVIEWS_ROOT = JOBS_ROOT / "gemini_self_debug_reviews_self"
SUMMARY_PATH = JOBS_ROOT / "num_wann_ordered_diagnostics_summary.json"
ORIGINAL_BEST_CSV = JOBS_ROOT / "successful_run_errors.csv"
DATASET_ROOT = ROOT / "harbor_datasets" / "wannier_200"
OUTPUT_CSV = REVIEWS_ROOT / "projection_mode_comparison.csv"
OUTPUT_JSON = REVIEWS_ROOT / "projection_mode_comparison_summary.json"
OUTPUT_ERROR_CSV = REVIEWS_ROOT / "projection_error_ratio_comparison.csv"
OUTPUT_ERROR_JSON = REVIEWS_ROOT / "projection_error_ratio_comparison.json"
OUTPUT_ALL_RATIOS_CSV = REVIEWS_ROOT / "all_error_ratios_by_material.csv"
OUTPUT_HEATMAP = REVIEWS_ROOT / "projection_mode_delta_heatmap"
PROJECTION_SIMILARITY_CMAP = LinearSegmentedColormap.from_list(
    "projection_similarity_orange_to_green",
    ["#D95F02", "#2E8B57"],  # orange -> green
)

ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error_pink_to_purple",
    ["#F7B6D2", "#6A00A8"],
)
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "delta_blue_white_red",
    ["#2D70B8", "#FFFFFF", "#B83A3A"],
)
ERROR_RATIO_COLUMNS = ("original run BEST", "new run BEST")
CHOICE_COLUMNS = ("projection similarity", "window strict equal", "window similarity")


def material_from_job_folder(path: Path) -> str:
    return path.name.rsplit("__", 1)[-1]


def num_wann_from_job_folder(path: Path) -> int | None:
    match = re.search(r"__num_wann_(\d+)__", path.name)
    return int(match.group(1)) if match else None


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
        *sorted((trial / "artifacts" / "logs" / "artifacts").glob(f"attempt_*/*{material}{suffix}")),
        *sorted((trial / "artifacts" / "artifacts").glob(f"attempt_*/*{material}{suffix}")),
        *sorted((trial / "artifacts").glob(f"attempt_*/*{material}{suffix}")),
    ]
    if not candidates:
        candidates = [
            *sorted((trial / "artifacts" / "logs" / "artifacts").glob(f"attempt_*/*{suffix}")),
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


def load_original_best_rows() -> dict[tuple[str, int], dict[str, object]]:
    df = pd.read_csv(ORIGINAL_BEST_CSV)
    df["num_wann"] = pd.to_numeric(df["num_wann"], errors="coerce")
    df["gemini_to_reference_ratio"] = pd.to_numeric(df["gemini_to_reference_ratio"], errors="coerce")
    df = df.dropna(subset=["material", "num_wann", "gemini_to_reference_ratio"])
    result: dict[tuple[str, int], dict[str, object]] = {}
    for (_material, _num_wann), group in df.groupby(["material", "num_wann"], sort=True):
        best = group.sort_values(["gemini_to_reference_ratio", "gemini_error_eV"], na_position="last").iloc[0]
        material = str(best["material"])
        num_wann = int(best["num_wann"])
        result[(material, num_wann)] = best.to_dict()
    return result


def job_folder_from_run_id(run_id: object) -> Path | None:
    if not isinstance(run_id, str) or not run_id:
        return None
    parts = Path(run_id).parts
    if len(parts) < 2:
        return None
    candidate = ROOT / parts[0] / parts[1]
    return candidate if candidate.is_dir() else None


def write_all_error_ratios_csv(new_by_material: dict[str, list[Path]]) -> int:
    original_df = pd.read_csv(ORIGINAL_BEST_CSV)
    original_df["num_wann"] = pd.to_numeric(original_df["num_wann"], errors="coerce")
    original_df["gemini_to_reference_ratio"] = pd.to_numeric(
        original_df["gemini_to_reference_ratio"],
        errors="coerce",
    )
    original_df["reference_error_eV"] = pd.to_numeric(
        original_df["reference_error_eV"],
        errors="coerce",
    )
    original_df = original_df.dropna(subset=["material", "num_wann"])

    rows: list[dict[str, object]] = []
    max_original_runs = 0
    max_new_runs = 0

    for material, new_folders in sorted(new_by_material.items()):
        folders_by_num_wann: dict[int, list[Path]] = defaultdict(list)
        for folder in new_folders:
            num_wann = num_wann_from_job_folder(folder)
            if num_wann is not None:
                folders_by_num_wann[num_wann].append(folder)

        for num_wann, candidate_folders in sorted(folders_by_num_wann.items()):
            original_matches = original_df[
                (original_df["material"] == material)
                & (original_df["num_wann"] == num_wann)
            ].copy()
            if original_matches.empty:
                continue

            reference_values = original_matches["reference_error_eV"].dropna()
            if reference_values.empty:
                continue
            reference_rmse = float(reference_values.iloc[0])

            original_ratios = [
                float(value)
                for value in original_matches["gemini_to_reference_ratio"].dropna().tolist()
                if math.isfinite(float(value))
            ]
            original_ratios.sort()

            new_ratios: list[float] = []
            for folder in candidate_folders:
                _new_rmse, new_ratio = error_ratio(folder, reference_rmse)
                if new_ratio is not None:
                    new_ratios.append(new_ratio)
            new_ratios.sort()

            max_original_runs = max(max_original_runs, len(original_ratios))
            max_new_runs = max(max_new_runs, len(new_ratios))
            row: dict[str, object] = {
                "material": material,
                "num_wann": num_wann,
            }
            for index, ratio in enumerate(original_ratios, start=1):
                row[f"original_run_{index}_error_ratio"] = ratio
            for index, ratio in enumerate(new_ratios, start=1):
                row[f"new_run_{index}_error_ratio"] = ratio
            rows.append(row)

    fieldnames = (
        ["material", "num_wann"]
        + [f"original_run_{index}_error_ratio" for index in range(1, max_original_runs + 1)]
        + [f"new_run_{index}_error_ratio" for index in range(1, max_new_runs + 1)]
    )
    OUTPUT_ALL_RATIOS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_ALL_RATIOS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


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


def compare_win_choices_for_pair(
    original_win: Path | None,
    new_win: Path | None,
    material: str,
) -> dict[str, object]:
    if original_win is None or new_win is None or not original_win.is_file() or not new_win.is_file():
        return {
            "projection_similarity": "",
            "window_strict_equal": "",
            "outer_window_similarity": "",
            "frozen_window_similarity": "",
            "window_similarity": "",
        }

    original_data = parse_win(original_win)
    new_data = parse_win(new_win)

    original_params = original_data["params"]
    new_params = new_data["params"]

    original_eig = find_eig_near_win(original_win, material)
    new_eig = find_eig_near_win(new_win, material)

    mask_compare = compare_window_masks(
        original_params,
        new_params,
        original_eig,
        new_eig,
    )

    window_strict_equal = all(
        original_params.get(key) == new_params.get(key)
        for key in ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max")
    )

    def fmt(value: object) -> str:
        return "" if value is None else f"{float(value):.6f}"

    return {
        "projection_similarity": f"{multiset_jaccard(original_data['projections'], new_data['projections']):.6f}",
        "window_strict_equal": window_strict_equal,
        "outer_window_similarity": fmt(mask_compare["outer_similarity"]),
        "frozen_window_similarity": fmt(mask_compare["frozen_similarity"]),
        "window_similarity": fmt(mask_compare["combined_similarity"]),
    }


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

def choice_color_values(values: pd.DataFrame) -> np.ndarray:
    colors = {
        True: "#2E8B57",
        False: "#D95F02",
    }
    rgba = np.ones((*values.shape, 4))
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values.iat[row_index, col_index]
            if col_index in {0, 2}:
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iat[0]
                if pd.notna(numeric):
                    # projection similarity is in [0, 1]:
                    # 0 = orange (dissimilar), 1 = green (similar)
                    rgba[row_index, col_index] = PROJECTION_SIMILARITY_CMAP(
                        float(np.clip(numeric, 0.0, 1.0))
                    )
                else:
                    rgba[row_index, col_index] = to_rgba(MISSING_COLOR)
            elif value in colors:
                rgba[row_index, col_index] = to_rgba(colors[value])
            else:
                rgba[row_index, col_index] = to_rgba(MISSING_COLOR)
    return rgba

def choice_label(value: object, col_index: int) -> str:
    if col_index in {0, 2}:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iat[0]
        return f"{float(numeric):.2f}" if pd.notna(numeric) else ""
    if value is True:
        return "T"
    if value is False:
        return "F"
    return ""


def make_delta_heatmap(rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    feature_columns = [f"delta_{feature}" for feature in PROCESS_FEATURES]
    labels = {f"delta_{key}": label for key, label in PROCESS_FEATURES.items()}
    df["delta_log_error_ratio"] = np.log10(pd.to_numeric(df["new_error_ratio"], errors="coerce")) - np.log10(
        pd.to_numeric(df["original_error_ratio"], errors="coerce")
    )
    feature_columns.append("delta_log_error_ratio")
    labels["delta_log_error_ratio"] = "interpolation error badness"

    df = df.sort_values(["delta_log_error_ratio", "material"], na_position="last").reset_index(drop=True)
    heatmap_df = df[feature_columns].apply(pd.to_numeric, errors="coerce").rename(columns=labels)
    ratio_df = df[["original_error_ratio", "new_error_ratio"]].rename(
        columns={
            "original_error_ratio": ERROR_RATIO_COLUMNS[0],
            "new_error_ratio": ERROR_RATIO_COLUMNS[1],
        }
    )
    choice_df = df[["projection_similarity", "window_strict_equal", "window_similarity"]].rename(
        columns={
            "projection_similarity": CHOICE_COLUMNS[0],
            "window_strict_equal": CHOICE_COLUMNS[1],
            "window_similarity": CHOICE_COLUMNS[2],
        }
    )

    abs_values = np.abs(heatmap_df.to_numpy(dtype=float))
    finite_abs_values = abs_values[np.isfinite(abs_values)]
    vmax = float(np.quantile(finite_abs_values, 0.95)) if finite_abs_values.size else 1.0
    vmax = max(vmax, 1e-9)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    plt.rcParams.update(
        {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "font.size": 8.2,
        }
    )
    height = max(8.5, 0.25 * len(df) + 2.0)
    fig = plt.figure(figsize=(17.0, height))
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[0.95, 1.45, 7.0, 0.35],
        left=0.08,
        right=0.92,
        top=0.90,
        bottom=0.16,
        wspace=0.04,
    )
    ratio_ax = fig.add_subplot(gs[0, 0])
    choice_ax = fig.add_subplot(gs[0, 1], sharey=ratio_ax)
    heatmap_ax = fig.add_subplot(gs[0, 2], sharey=ratio_ax)
    cbar_ax = fig.add_subplot(gs[0, 3])

    nrows = len(df)

    ratio_ax.imshow(
        error_ratio_color_values(ratio_df),
        aspect="auto",
        interpolation="nearest",
        extent=(0, len(ratio_df.columns), nrows, 0),
    )

    choice_ax.imshow(
        choice_color_values(choice_df),
        aspect="auto",
        interpolation="nearest",
        extent=(0, len(choice_df.columns), nrows, 0),
    )
    for row_index in range(len(choice_df)):
        for col_index in range(len(choice_df.columns)):
            value = choice_df.iat[row_index, col_index]
            choice_ax.text(
                col_index + 0.5,
                row_index + 0.5,
                choice_label(value, col_index),
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                fontweight="bold",
            )

    heatmap_values = heatmap_df.to_numpy(dtype=float)
    masked_heatmap_values = np.ma.masked_invalid(heatmap_values)
    heatmap_cmap = DELTA_CMAP.copy()
    heatmap_cmap.set_bad(MISSING_COLOR)
    image = heatmap_ax.imshow(
        masked_heatmap_values,
        aspect="auto",
        interpolation="nearest",
        cmap=heatmap_cmap,
        norm=norm,
        extent=(0, len(heatmap_df.columns), nrows, 0),
    )
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Delta badness, new - original\nblue improves, red worsens")
    heatmap_ax.set_xticks(np.arange(len(heatmap_df.columns)) + 0.5)
    heatmap_ax.set_xticklabels(
        heatmap_df.columns.tolist(),
        rotation=35,
        ha="right",
    )

    for ax, columns in ((ratio_ax, ERROR_RATIO_COLUMNS), (choice_ax, CHOICE_COLUMNS)):
        ax.set_xticks(np.arange(len(columns)) + 0.5)
        ax.set_xticklabels(columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(df)) + 0.5)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ratio_ax.set_yticklabels(df["material"].tolist(), fontsize=7)
    choice_ax.tick_params(labelleft=False)
    heatmap_ax.set_yticks(np.arange(len(df)) + 0.5)
    heatmap_ax.tick_params(axis="y", labelleft=False, labelright=True, right=False, length=0)
    heatmap_ax.set_yticklabels(df["material"].tolist(), rotation=0, fontsize=7)
    heatmap_ax.set_xlabel("Process diagnostic delta, new best vs original best")
    heatmap_ax.set_ylabel("Material")
    heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), rotation=35, ha="right")
    ratio_ax.set_ylabel("Material")

    legend = [
        Patch(facecolor="#2E8B57", label="Window equal"),
        Patch(facecolor="#D95F02", label="Window different"),
        Patch(facecolor=DELTA_CMAP(1.0), label="Higher projection similarity"),
        Patch(facecolor=MISSING_COLOR, label="Missing .win choice"),
        Patch(facecolor=ERROR_CMAP(0.15), label="Lower error ratio"),
        Patch(facecolor=ERROR_CMAP(0.95), label="Higher error ratio"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.52, 0.985))
    fig.suptitle("New self-debug best deltas against original best runs", y=0.995)
    fig.savefig(OUTPUT_HEATMAP.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_HEATMAP.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    original_best_by_key = load_original_best_rows()
    new_by_material = job_folders_by_material(REVIEWS_ROOT)
    all_ratio_row_count = write_all_error_ratios_csv(new_by_material)

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for material, new_folders in sorted(new_by_material.items()):
        folders_by_num_wann: dict[int, list[Path]] = defaultdict(list)
        for folder in new_folders:
            num_wann = num_wann_from_job_folder(folder)
            if num_wann is not None:
                folders_by_num_wann[num_wann].append(folder)

        for num_wann, candidate_folders in sorted(folders_by_num_wann.items()):
            original_row = original_best_by_key.get((material, num_wann))
            if original_row is None:
                skipped.append(
                    {
                        "material": material,
                        "num_wann": str(num_wann),
                        "reason": "no original CSV row with matching material and num_wann",
                    }
                )
                continue

            reference_rmse = finite(original_row.get("reference_error_eV"))
            if reference_rmse is None:
                skipped.append(
                    {
                        "material": material,
                        "num_wann": str(num_wann),
                        "reason": "original CSV row has no finite reference_error_eV",
                    }
                )
                continue

            new_candidates: list[tuple[float, float | None, Path]] = []
            for folder in candidate_folders:
                new_rmse, new_ratio = error_ratio(folder, reference_rmse)
                if new_ratio is not None:
                    new_candidates.append((new_ratio, new_rmse, folder))

            if not new_candidates:
                skipped.append(
                    {
                        "material": material,
                        "num_wann": str(num_wann),
                        "reason": "no new candidate had finite rmse/reference ratio",
                    }
                )
                continue

            new_ratio, new_rmse, new_folder = sorted(new_candidates, key=lambda item: (item[0], item[2].name))[0]
            original_rmse = finite(original_row.get("gemini_error_eV"))
            original_ratio = finite(original_row.get("gemini_to_reference_ratio"))
            if original_ratio is None:
                skipped.append(
                    {
                        "material": material,
                        "num_wann": str(num_wann),
                        "reason": "original CSV row has no finite gemini_to_reference_ratio",
                    }
                )
                continue

            original_job_folder = job_folder_from_run_id(original_row.get("run_id"))
            original_win = find_submitted_win(original_job_folder, material) if original_job_folder else None
            original_wout = find_submitted_wout(original_job_folder, material) if original_job_folder else None
            new_win = find_submitted_win(new_folder, material)
            new_wout = find_submitted_wout(new_folder, material)
            win_choice_comparison = compare_win_choices_for_pair(original_win, new_win, material)
            delta_ratio = new_ratio - original_ratio
            ratio_fold_change = new_ratio / original_ratio if original_ratio > 0 else None
            delta_log10_ratio = (
                math.log10(new_ratio) - math.log10(original_ratio)
                if new_ratio > 0 and original_ratio > 0
                else None
            )
            reference_wout = DATASET_ROOT / material / "tests/reference/wannier/output/wannier90/aiida.wout"
            original_features = process_feature_row(original_wout, reference_wout)
            new_features = process_feature_row(new_wout, reference_wout)
            deltas = {}
            for feature in PROCESS_FEATURES:
                original_value = oriented_feature_value(feature, original_features.get(feature))
                new_value = oriented_feature_value(feature, new_features.get(feature))
                deltas[f"delta_{feature}"] = (
                    new_value - original_value
                    if new_value is not None and original_value is not None
                    else None
                )

            row: dict[str, object] = {
                "material": material,
                "num_wann": num_wann,
                "reference_rmse_eV": reference_rmse,
                "original_rmse_eV": original_rmse,
                "new_rmse_eV": new_rmse,
                "original_error_ratio": original_ratio,
                "new_error_ratio": new_ratio,
                "delta_error_ratio_new_minus_original": delta_ratio,
                "ratio_fold_change_new_over_original": ratio_fold_change,
                "delta_log10_error_ratio_new_minus_original": delta_log10_ratio,
                "projection_similarity": win_choice_comparison["projection_similarity"],
                "window_strict_equal": win_choice_comparison["window_strict_equal"],
                "outer_window_similarity": win_choice_comparison["outer_window_similarity"],
                "frozen_window_similarity": win_choice_comparison["frozen_window_similarity"],
                "window_similarity": win_choice_comparison["window_similarity"],
                "original_run_id": original_row.get("run_id"),
                "new_job_folder": new_folder.name,
                "new_candidate_count_for_material_num_wann": len(candidate_folders),
                "original_job_folder": original_job_folder.name if original_job_folder else "",
                "original_win_path": relative(original_win),
                "new_win_path": relative(new_win),
                "original_wout_path": relative(original_wout),
                "new_wout_path": relative(new_wout),
            }
            row.update(deltas)
            rows.append(row)

    if not rows:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_CSV.write_text("", encoding="utf-8")
        OUTPUT_ERROR_CSV.write_text("", encoding="utf-8")
        OUTPUT_ERROR_JSON.write_text("[]\n", encoding="utf-8")
        OUTPUT_JSON.write_text(
            json.dumps(
                {
                    "compared_materials": 0,
                    "skipped_materials": skipped,
                    "all_error_ratios_csv": str(OUTPUT_ALL_RATIOS_CSV.relative_to(ROOT)),
                    "all_error_ratios_rows": all_ratio_row_count,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("Compared materials: 0")
        print(f"Skipped materials: {len(skipped)}")
        print(f"Wrote {OUTPUT_ALL_RATIOS_CSV.relative_to(ROOT)}")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    error_fields = [
        "material",
        "num_wann",
        "reference_rmse_eV",
        "original_rmse_eV",
        "new_rmse_eV",
        "original_error_ratio",
        "new_error_ratio",
        "delta_error_ratio_new_minus_original",
        "ratio_fold_change_new_over_original",
        "delta_log10_error_ratio_new_minus_original",
        "projection_similarity",
        "window_strict_equal",
        "outer_window_similarity",
        "frozen_window_similarity",
        "window_similarity",
        "original_run_id",
        "new_job_folder",
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
            "new_runs_root": str(REVIEWS_ROOT.relative_to(ROOT)),
            "original_best_csv": str(ORIGINAL_BEST_CSV.relative_to(ROOT)),
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "error_ratio_csv": str(OUTPUT_ERROR_CSV.relative_to(ROOT)),
            "error_ratio_json": str(OUTPUT_ERROR_JSON.relative_to(ROOT)),
            "all_error_ratios_csv": str(OUTPUT_ALL_RATIOS_CSV.relative_to(ROOT)),
            "heatmap_png": str(OUTPUT_HEATMAP.with_suffix(".png").relative_to(ROOT)),
            "heatmap_pdf": str(OUTPUT_HEATMAP.with_suffix(".pdf").relative_to(ROOT)),
        },
        "selection": (
            "Original best is the lowest gemini_to_reference_ratio per material,num_wann "
            "from successful_run_errorsFROMACMINI.csv. New best is the lowest "
            "rmse_eV/reference_error_eV among gemini_self_debug_reviews runs with "
            "the same material and num_wann."
        ),
        "win_choice_comparison": (
            "projection_similarity is the multiset Jaccard similarity of normalized projections. "
            "window_strict_equal checks exact window parameter equality. window_similarity is the combined "
            "0-1 Jaccard similarity of actual outer and frozen window band masks computed from paired .eig files."
        ),
        "compared_materials": len(rows),
        "all_error_ratios_rows": all_ratio_row_count,
        "skipped_materials": skipped,
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Compared materials: {len(rows)}")
    print(f"Skipped materials: {len(skipped)}")
    print(f"Wrote {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_ERROR_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_ERROR_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_ALL_RATIOS_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_HEATMAP.with_suffix('.png').relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_HEATMAP.with_suffix('.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
