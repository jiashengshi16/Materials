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

JOBS_ROOT = ROOT / "jobsDeepseekProTerminus2Controlled"
#RERUNS_ROOT = ROOT / "reruns"
REVIEWS_ROOT = ROOT / "jobsGeminiReviewsDeepseek/ChemSimReruns"
REFERENCE_ERROR_CSV = ROOT / "jobs" / "successful_run_errors.csv"
DATASET_ROOT = ROOT / "harbor_datasets" / "wannier_200"
OUTPUT_CSV = REVIEWS_ROOT / "projection_mode_comparison.csv"
OUTPUT_JSON = REVIEWS_ROOT / "projection_mode_comparison_summary.json"
OUTPUT_ERROR_CSV = REVIEWS_ROOT / "projection_error_ratio_comparison.csv"
OUTPUT_ERROR_JSON = REVIEWS_ROOT / "projection_error_ratio_comparison.json"
OUTPUT_ALL_RATIOS_CSV = REVIEWS_ROOT / "all_error_ratios_by_material.csv"
OUTPUT_HEATMAP = REVIEWS_ROOT / "projection_mode_delta_heatmap"
OUTPUT_AVERAGE_HEATMAP = REVIEWS_ROOT / "projection_mode_pairwise_average_delta_heatmap"
PROJECTION_SIMILARITY_CMAP = LinearSegmentedColormap.from_list(
    "projection_similarity_orange_to_green",
    ["#D95F02", "#2E8B57"],  # orange -> green
)
MISSING_RMSE_ERROR_RATIO = 10_000.0
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error_pink_to_purple",
    ["#F7B6D2", "#6A00A8"],
)
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "delta_blue_white_red",
    ["#2D70B8", "#FFFFFF", "#B83A3A"],
)
ERROR_RATIO_COLUMNS = (
    "original run BEST",
    "new run BEST",
    "avg original run error ratio",
    "avg new run error ratio",
)
CHOICE_COLUMNS = ("projection similarity", "window strict equal", "window similarity")


def material_from_job_folder(path: Path) -> str:
    metadata_path = path / "case_files" / "case_metadata.json"
    if metadata_path.is_file():
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        material = data.get("material")
        if isinstance(material, str) and material:
            return material
    if path.parent != REVIEWS_ROOT and not path.parent.name.startswith("num_wann_ordered__"):
        return path.parent.name
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
    if (job_folder / "artifacts").is_dir() and (
        (job_folder / "verifier").is_dir() or any((job_folder / "artifacts").glob("attempt_*"))
    ):
        return job_folder
    trials = [
        path
        for path in sorted(job_folder.iterdir())
        if path.is_dir() and (path / "verifier").is_dir()
    ]
    if len(trials) == 1:
        return trials[0]
    return None


def find_submitted_file(job_folder: Path, material: str, suffix: str) -> Path | None:
    case_files = job_folder / "case_files"
    if case_files.is_dir():
        candidates = [
            *sorted((case_files / "artifacts").glob(f"attempt_*/*{material}{suffix}")),
            *sorted((case_files / "artifacts").glob(f"attempt_*/*{suffix}")),
        ]
        if candidates:
            return sort_submitted_candidates(candidates)[-1]

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

def load_original_rows() -> list[dict[str, object]]:
    reference_df = pd.read_csv(REFERENCE_ERROR_CSV)

    reference_df["num_wann"] = pd.to_numeric(
        reference_df["num_wann"], errors="coerce"
    )
    reference_df["reference_error_eV"] = pd.to_numeric(
        reference_df["reference_error_eV"], errors="coerce"
    )
    reference_df = reference_df.dropna(
        subset=[
            "material",
            "num_wann",
            "reference_error_eV",
        ]
    )

    reference_by_key = {
        (str(row["material"]), int(row["num_wann"])): float(row["reference_error_eV"])
        for _, row in reference_df.iterrows()
    }

    rows: list[dict[str, object]] = []
    for job_folder in sorted(JOBS_ROOT.iterdir()):
        if not job_folder.is_dir() or not job_folder.name.startswith("num_wann_ordered__"):
            continue

        material = job_folder.name.rsplit("__", 1)[-1]
        num_wann = num_wann_from_job_folder(job_folder)
        if num_wann is None:
            continue

        reference_rmse = reference_by_key.get((material, num_wann))
        if reference_rmse is None or reference_rmse <= 0:
            continue

        metrics = load_result_metrics(job_folder)
        rmse = finite(metrics.get("rmse_eV"))
        ratio = (
            rmse / reference_rmse
            if rmse is not None
            else MISSING_RMSE_ERROR_RATIO
        )

        rows.append(
            {
                "material": material,
                "run_id": relative(job_folder),
                "num_wann": num_wann,
                "reward": metrics.get("reward"),
                "gemini_error_eV": rmse,
                "reference_error_eV": reference_rmse,
                "gemini_to_reference_ratio": ratio,
                "original_source": str(JOBS_ROOT.relative_to(ROOT)),
            }
        )

    return rows

def load_original_best_rows() -> dict[tuple[str, int], dict[str, object]]:
    rows = load_original_rows()
    df = pd.DataFrame(rows)
    df["num_wann"] = pd.to_numeric(df["num_wann"], errors="coerce")
    df["gemini_to_reference_ratio"] = pd.to_numeric(df["gemini_to_reference_ratio"], errors="coerce")
    df["gemini_error_eV"] = pd.to_numeric(df["gemini_error_eV"], errors="coerce")
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
    direct = ROOT / run_id
    if direct.is_dir():
        return direct
    parts = Path(run_id).parts
    if len(parts) < 2:
        return None
    candidate = ROOT / parts[0] / parts[1]
    return candidate if candidate.is_dir() else None


def write_all_error_ratios_csv(new_by_material: dict[str, list[Path]]) -> tuple[int, dict[tuple[str, int], dict[str, float | None]]]:
    original_df = pd.DataFrame(load_original_rows())
    original_df["num_wann"] = pd.to_numeric(original_df["num_wann"], errors="coerce")
    original_df["gemini_to_reference_ratio"] = pd.to_numeric(original_df["gemini_to_reference_ratio"], errors="coerce")
    original_df["reference_error_eV"] = pd.to_numeric(original_df["reference_error_eV"], errors="coerce")
    original_df = original_df.dropna(subset=["material", "num_wann"])

    rows: list[dict[str, object]] = []
    average_by_key: dict[tuple[str, int], dict[str, float | None]] = {}
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
            avg_original = float(np.mean(original_ratios)) if original_ratios else None
            avg_new = float(np.mean(new_ratios)) if new_ratios else None
            average_by_key[(material, num_wann)] = {
                "avg_original_run_error_ratio": avg_original,
                "avg_new_run_error_ratio": avg_new,
            }

            row: dict[str, object] = {
                "material": material,
                "num_wann": num_wann,
                "avg_original_run_error_ratio": avg_original,
            }
            for index, ratio in enumerate(original_ratios, start=1):
                row[f"original_run_{index}_error_ratio"] = ratio
            row["avg_new_run_error_ratio"] = avg_new
            for index, ratio in enumerate(new_ratios, start=1):
                row[f"new_run_{index}_error_ratio"] = ratio
            rows.append(row)

    fieldnames = (
        ["material", "num_wann"]
        + [f"original_run_{index}_error_ratio" for index in range(1, max_original_runs + 1)]
        + ["avg_original_run_error_ratio"]
        + [f"new_run_{index}_error_ratio" for index in range(1, max_new_runs + 1)]
        + ["avg_new_run_error_ratio"]
    )
    OUTPUT_ALL_RATIOS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_ALL_RATIOS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), average_by_key


def finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        result = float(value)
        return result if math.isfinite(result) else None
    return None

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


def error_ratio(
    job_folder: Path,
    reference_rmse: float | None,
) -> tuple[float | None, float | None]:
    metrics = load_result_metrics(job_folder)
    rmse = finite(metrics.get("rmse_eV"))

    if reference_rmse is None or reference_rmse <= 0:
        return rmse, None

    if rmse is None:
        return None, MISSING_RMSE_ERROR_RATIO

    return rmse, rmse / reference_rmse


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

    vmin = float(np.quantile(finite_values, 0.05))

    norm = Normalize(
        vmin=min(vmin, math.log10(MISSING_RMSE_ERROR_RATIO) - 1e-9),
        vmax=math.log10(MISSING_RMSE_ERROR_RATIO),
        clip=True,
    )

    rgba = np.ones((*numeric.shape, 4))
    for row_index in range(numeric.shape[0]):
        for col_index in range(numeric.shape[1]):
            value = log_values.iat[row_index, col_index]
            if pd.notna(value):
                rgba[row_index, col_index] = ERROR_CMAP(norm(float(value)))
    return rgba

def numeric_metric(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def choice_color_values(values: pd.DataFrame, *, all_numeric: bool = False) -> np.ndarray:
    colors = {
        True: "#2E8B57",
        False: "#D95F02",
    }
    rgba = np.ones((*values.shape, 4))
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values.iat[row_index, col_index]
            if all_numeric or col_index in {0, 2}:
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iat[0]
                if pd.notna(numeric):
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


def choice_label(value: object, col_index: int, *, all_numeric: bool = False) -> str:
    if all_numeric or col_index in {0, 2}:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iat[0]
        return f"{float(numeric):.2f}" if pd.notna(numeric) else ""
    if value is True:
        return "T"
    if value is False:
        return "F"
    return ""


def build_pairwise_average_row(
    material: str,
    num_wann: int,
    original_rows: list[dict[str, object]],
    new_folders: list[Path],
    reference_rmse: float,
    averages: dict[str, float | None],
) -> dict[str, object] | None:
    reference_wout = DATASET_ROOT / material / "tests/reference/wannier/output/wannier90/aiida.wout"

    original_runs: list[dict[str, object]] = []
    for original_row in original_rows:
        original_ratio = finite(original_row.get("gemini_to_reference_ratio"))
        if original_ratio is None:
            continue
        original_folder = job_folder_from_run_id(original_row.get("run_id"))
        original_wout = find_submitted_wout(original_folder, material) if original_folder else None
        original_runs.append(
            {
                "ratio": original_ratio,
                "win": find_submitted_win(original_folder, material) if original_folder else None,
                "features": process_feature_row(original_wout, reference_wout),
            }
        )

    new_runs: list[dict[str, object]] = []
    for new_folder in new_folders:
        _new_rmse, new_ratio = error_ratio(new_folder, reference_rmse)
        if new_ratio is None:
            continue
        new_wout = find_submitted_wout(new_folder, material)
        new_runs.append(
            {
                "ratio": new_ratio,
                "win": find_submitted_win(new_folder, material),
                "features": process_feature_row(new_wout, reference_wout),
            }
        )

    if not original_runs or not new_runs:
        return None

    choice_values: dict[str, list[float]] = {
        "projection_similarity": [],
        "window_strict_equal": [],
        "window_similarity": [],
    }
    delta_values: dict[str, list[float]] = {
        f"delta_{feature}": [] for feature in PROCESS_FEATURES
    }
    interpolation_badness_values: list[float] = []

    for original_run in original_runs:
        original_ratio = float(original_run["ratio"])
        original_features = original_run["features"]

        for new_run in new_runs:
            new_ratio = float(new_run["ratio"])
            new_features = new_run["features"]

            comparison = compare_win_choices_for_pair(
                original_run["win"],
                new_run["win"],
                material,
            )
            for key in choice_values:
                value = numeric_metric(comparison.get(key))
                if value is not None:
                    choice_values[key].append(value)

            for feature in PROCESS_FEATURES:
                original_value = oriented_feature_value(feature, original_features.get(feature))
                new_value = oriented_feature_value(feature, new_features.get(feature))
                if original_value is not None and new_value is not None:
                    delta_values[f"delta_{feature}"].append(new_value - original_value)

            if original_ratio > 0 and new_ratio > 0:
                interpolation_badness_values.append(
                    math.log10(new_ratio) - math.log10(original_ratio)
                )

    row: dict[str, object] = {
        "material": material,
        "num_wann": num_wann,
        "avg_original_run_error_ratio": averages.get("avg_original_run_error_ratio"),
        "avg_new_run_error_ratio": averages.get("avg_new_run_error_ratio"),
        "projection_similarity": mean_or_none(choice_values["projection_similarity"]),
        "window_strict_equal": mean_or_none(choice_values["window_strict_equal"]),
        "window_similarity": mean_or_none(choice_values["window_similarity"]),
        "delta_log_error_ratio": mean_or_none(interpolation_badness_values),
        "pairwise_comparison_count": len(original_runs) * len(new_runs),
        "original_run_count": len(original_runs),
        "new_run_count": len(new_runs),
    }
    for key, values in delta_values.items():
        row[key] = mean_or_none(values)
    return row


def make_delta_heatmap(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
    average_mode: bool,
) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    feature_columns = [f"delta_{feature}" for feature in PROCESS_FEATURES]
    labels = {f"delta_{key}": label for key, label in PROCESS_FEATURES.items()}

    if "delta_log_error_ratio" not in df.columns:
        df["delta_log_error_ratio"] = (
            np.log10(pd.to_numeric(df["new_error_ratio"], errors="coerce"))
            - np.log10(pd.to_numeric(df["original_error_ratio"], errors="coerce"))
        )

    feature_columns.append("delta_log_error_ratio")
    labels["delta_log_error_ratio"] = "interpolation error badness"

    df = df.sort_values(["delta_log_error_ratio", "material"], na_position="last").reset_index(drop=True)
    heatmap_df = df[feature_columns].apply(pd.to_numeric, errors="coerce").rename(columns=labels)

    if average_mode:
        ratio_source_columns = [
            "avg_original_run_error_ratio",
            "avg_new_run_error_ratio",
        ]
        ratio_display_columns = ERROR_RATIO_COLUMNS[2:]
    else:
        ratio_source_columns = [
            "original_error_ratio",
            "new_error_ratio",
        ]
        ratio_display_columns = ERROR_RATIO_COLUMNS[:2]

    ratio_df = df[ratio_source_columns].copy()
    ratio_df.columns = ratio_display_columns

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
    fig = plt.figure(figsize=(16.0, height))
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.25, 1.45, 7.0, 0.35],
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
        choice_color_values(choice_df, all_numeric=average_mode),
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
                choice_label(value, col_index, all_numeric=average_mode),
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

    for ax, columns in ((ratio_ax, ratio_display_columns), (choice_ax, CHOICE_COLUMNS)):
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
    heatmap_ax.set_ylabel("Material")
    heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), rotation=35, ha="right")
    ratio_ax.set_ylabel("Material")

    if average_mode:
        heatmap_ax.set_xlabel("Mean process diagnostic delta across every original × new run pair")
        title = "All-pairs average deltas: every original run against every new run"
        choice_legend_label = "Higher average similarity / equality"
    else:
        heatmap_ax.set_xlabel("Process diagnostic delta, new best vs original best")
        title = "New self-debug best deltas against original best runs"
        choice_legend_label = "Higher choice similarity"

    legend = [
        Patch(facecolor=PROJECTION_SIMILARITY_CMAP(1.0), label=choice_legend_label),
        Patch(facecolor=PROJECTION_SIMILARITY_CMAP(0.0), label="Lower choice similarity"),
        Patch(facecolor=MISSING_COLOR, label="Missing comparison"),
        Patch(facecolor=ERROR_CMAP(0.15), label="Lower error ratio"),
        Patch(facecolor=ERROR_CMAP(0.95), label="Higher error ratio"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.52, 0.985))
    fig.suptitle(title, y=0.995)
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    original_best_by_key = load_original_best_rows()
    original_rows_by_key: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for original_run_row in load_original_rows():
        original_rows_by_key[(str(original_run_row["material"]), int(original_run_row["num_wann"]))].append(
            original_run_row
        )
    new_by_material = job_folders_by_material(REVIEWS_ROOT)
    all_ratio_row_count, average_error_ratios_by_key = write_all_error_ratios_csv(new_by_material)

    rows: list[dict[str, object]] = []
    average_rows: list[dict[str, object]] = []
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
                        "reason": "original DeepSeek row has no finite reference_error_eV",
                    }
                )
                continue

            averages = average_error_ratios_by_key.get((material, num_wann), {})
            average_row = build_pairwise_average_row(
                material,
                num_wann,
                original_rows_by_key.get((material, num_wann), []),
                candidate_folders,
                reference_rmse,
                averages,
            )
            if average_row is not None:
                average_rows.append(average_row)

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
                        "reason": "original DeepSeek row has no finite gemini_to_reference_ratio",
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
                "avg_original_run_error_ratio": averages.get("avg_original_run_error_ratio"),
                "new_error_ratio": new_ratio,
                "avg_new_run_error_ratio": averages.get("avg_new_run_error_ratio"),
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
        "avg_original_run_error_ratio",
        "new_error_ratio",
        "avg_new_run_error_ratio",
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
    make_delta_heatmap(rows, output_path=OUTPUT_HEATMAP, average_mode=False)
    make_delta_heatmap(average_rows, output_path=OUTPUT_AVERAGE_HEATMAP, average_mode=True)

    summary = {
        "paths": {
            "new_runs_root": str(REVIEWS_ROOT.relative_to(ROOT)),
            "original_runs_root": str(JOBS_ROOT.relative_to(ROOT)),
            "reference_error_csv": str(REFERENCE_ERROR_CSV.relative_to(ROOT)),
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "error_ratio_csv": str(OUTPUT_ERROR_CSV.relative_to(ROOT)),
            "error_ratio_json": str(OUTPUT_ERROR_JSON.relative_to(ROOT)),
            "all_error_ratios_csv": str(OUTPUT_ALL_RATIOS_CSV.relative_to(ROOT)),
            "heatmap_png": str(OUTPUT_HEATMAP.with_suffix(".png").relative_to(ROOT)),
            "heatmap_pdf": str(OUTPUT_HEATMAP.with_suffix(".pdf").relative_to(ROOT)),
            "average_heatmap_png": str(OUTPUT_AVERAGE_HEATMAP.with_suffix(".png").relative_to(ROOT)),
            "average_heatmap_pdf": str(OUTPUT_AVERAGE_HEATMAP.with_suffix(".pdf").relative_to(ROOT)),
        },
        "selection": (
            "Original best is the lowest rmse_eV/reference_error_eV per material,num_wann "
            f"from top-level {JOBS_ROOT.relative_to(ROOT)}/num_wann_ordered runs, "
            f"using reference_error_eV from {REFERENCE_ERROR_CSV.relative_to(ROOT)}. "
            "New best is the lowest rmse_eV/reference_error_eV among top-level "
            f"{REVIEWS_ROOT.relative_to(ROOT)}/num_wann_ordered runs with "
            "the same material and num_wann."
        ),
        "win_choice_comparison": (
            "projection_similarity is the multiset Jaccard similarity of normalized projections. "
            "window_strict_equal checks exact window parameter equality. window_similarity is the combined "
            "0-1 Jaccard similarity of actual outer and frozen window band masks computed from paired .eig files."
        ),
        "pairwise_average": (
            "Average heatmap metrics are means over every original-run × new-run pair for each "
            "material,num_wann. Similarity/equality metrics, process deltas, and log10 interpolation "
            "error badness are averaged over the valid pairwise values for that metric."
        ),
        "compared_materials": len(rows),
        "pairwise_average_materials": len(average_rows),
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
    print(f"Wrote {OUTPUT_AVERAGE_HEATMAP.with_suffix('.png').relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_AVERAGE_HEATMAP.with_suffix('.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
