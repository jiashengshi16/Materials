#!/usr/bin/env python3
"""Compare .win choice similarity for case1 lower-error runs against higher-error runs.

Hardcoded paths only, by request.  The case1 JSON values are interpreted as:

* value[0], value[2] -> lower-number run, treated as the "new" run
* value[1], value[3] -> higher-number run, treated as the "original" run

The heatmap compares the higher-number and lower-number submitted .win files
using the same orange/green choice columns as compare_self_critique:
projection similarity, strict window equality, and likely window equality.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import types
from collections import Counter
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
SUMMARY_PATH = JOBS_ROOT / "num_wann_ordered_diagnostics_summary.json"
CASE1_MATERIALS_JSON = JOBS_ROOT / "case1_materials.json"
DATASET_ROOT = ROOT / "harbor_datasets" / "wannier_200"
OUTPUT_DIR = JOBS_ROOT / "case1_projection_mode_comparison"
OUTPUT_CSV = OUTPUT_DIR / "projection_mode_comparison.csv"
OUTPUT_JSON = OUTPUT_DIR / "projection_mode_comparison_summary.json"
OUTPUT_ERROR_CSV = OUTPUT_DIR / "projection_error_ratio_comparison.csv"
OUTPUT_ERROR_JSON = OUTPUT_DIR / "projection_error_ratio_comparison.json"
OUTPUT_HEATMAP = OUTPUT_DIR / "projection_mode_delta_heatmap"
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
ERROR_RATIO_COLUMNS = ("higher-number run", "lower-number run")
CHOICE_COLUMNS = ("projection similarity", "window strict equal", "window similarity")

CATEGORY_KEYS = (
    "random_projection_runs",
    "explicit_projection_runs",
    "none_or_implicit_projection_runs",
)

PROJECTION_BLOCK_RE = re.compile(
    r"^\s*begin\s+projections\s*$([\s\S]*?)^\s*end\s+projections\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


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


def num_wann_from_case_path(path_value: object) -> int | None:
    if not isinstance(path_value, str):
        return None
    match = re.search(r"__num_wann_(\d+)__", path_value)
    return int(match.group(1)) if match else None


def job_folder_from_case_path(path_value: object) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir() and (path / "verifier").is_dir():
        return path.parent
    if path.is_dir():
        return path
    return None


def trial_name_from_case_path(path_value: object) -> str:
    if not isinstance(path_value, str) or not path_value:
        return ""
    return Path(path_value).name


def load_case1_rows() -> list[dict[str, object]]:
    data = json.loads(CASE1_MATERIALS_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{CASE1_MATERIALS_JSON} must contain a JSON object")

    rows: list[dict[str, object]] = []
    for material, value in sorted(data.items()):
        if not isinstance(material, str) or not isinstance(value, list) or len(value) < 4:
            continue
        rows.append(
            {
                "material": material,
                "new_input_error_ratio": finite(value[0]),
                "original_input_error_ratio": finite(value[1]),
                "new_case_path": value[2],
                "original_case_path": value[3],
                "new_job_folder": job_folder_from_case_path(value[2]),
                "original_job_folder": job_folder_from_case_path(value[3]),
                "new_trial_name": trial_name_from_case_path(value[2]),
                "original_trial_name": trial_name_from_case_path(value[3]),
                "new_num_wann": num_wann_from_case_path(value[2]),
                "original_num_wann": num_wann_from_case_path(value[3]),
            }
        )
    return rows


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


def load_reference_rmse_by_material_num_wann() -> dict[tuple[str, int], float]:
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    result: dict[tuple[str, int], float] = {}
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        material = item.get("material") or item.get("material_from_folder")
        num_wann = item.get("num_wann_from_folder")
        num_target_bands = item.get("num_target_bands")
        reference_rmse = finite(item.get("reference_offmesh_rmse_eV"))
        if (
            isinstance(material, str)
            and isinstance(num_wann, int)
            and isinstance(num_target_bands, int)
            and num_wann == num_target_bands
            and reference_rmse is not None
        ):
            result[(material, num_wann)] = reference_rmse
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


def error_ratio(
    job_folder: Path,
    reference_rmse: float | None,
    expected_num_wann: int | None = None,
) -> tuple[float | None, float | None]:
    metrics = load_result_metrics(job_folder)
    metric_num_wann = finite(metrics.get("num_target_bands"))
    if (
        expected_num_wann is not None
        and metric_num_wann is not None
        and int(metric_num_wann) != expected_num_wann
    ):
        return None, None
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
    if col_index == 0:
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
    cbar.set_label("Delta badness, lower-number - higher-number\nblue improves, red worsens")
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
    heatmap_ax.set_xlabel("Process diagnostic delta, lower-number run vs higher-number run")
    heatmap_ax.set_ylabel("Material")
    heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), rotation=35, ha="right")
    ratio_ax.set_ylabel("Material")

    legend = [
        Patch(facecolor="#2E8B57", label="Window equal"),
        Patch(facecolor="#D95F02", label="Window different"),
        Patch(facecolor=PROJECTION_SIMILARITY_CMAP(1.0), label="Higher projection similarity"),
        Patch(facecolor=MISSING_COLOR, label="Missing .win choice"),
        Patch(facecolor=ERROR_CMAP(0.15), label="Lower error ratio"),
        Patch(facecolor=ERROR_CMAP(0.95), label="Higher error ratio"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.52, 0.985))
    fig.suptitle("Case1 lower-number run deltas against higher-number runs", y=0.995)
    fig.savefig(OUTPUT_HEATMAP.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_HEATMAP.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    reference_rmse_by_key = load_reference_rmse_by_material_num_wann()
    case_rows = load_case1_rows()
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for case_row in case_rows:
        material = str(case_row["material"])
        new_num_wann = case_row.get("new_num_wann")
        original_num_wann = case_row.get("original_num_wann")
        if not isinstance(new_num_wann, int) or not isinstance(original_num_wann, int):
            skipped.append({"material": material, "num_wann": "", "reason": "missing num_wann in case paths"})
            continue
        if new_num_wann != original_num_wann:
            skipped.append(
                {
                    "material": material,
                    "num_wann": f"{new_num_wann}/{original_num_wann}",
                    "reason": "case paths disagree on num_wann",
                }
            )
            continue
        num_wann = new_num_wann

        reference_rmse = reference_rmse_by_key.get((material, num_wann))
        if reference_rmse is None:
            skipped.append(
                {
                    "material": material,
                    "num_wann": str(num_wann),
                    "reason": "no finite reference_offmesh_rmse_eV with matching material and num_wann",
                }
            )
            continue

        original_job_folder = case_row.get("original_job_folder")
        new_job_folder = case_row.get("new_job_folder")
        if not isinstance(original_job_folder, Path) or not isinstance(new_job_folder, Path):
            skipped.append(
                {
                    "material": material,
                    "num_wann": str(num_wann),
                    "reason": "could not resolve one or both case run folders",
                }
            )
            continue

        original_rmse, original_ratio = error_ratio(original_job_folder, reference_rmse, num_wann)
        new_rmse, new_ratio = error_ratio(new_job_folder, reference_rmse, num_wann)
        if original_ratio is None or new_ratio is None:
            skipped.append(
                {
                    "material": material,
                    "num_wann": str(num_wann),
                    "reason": "could not compute finite rmse/reference ratio for both case runs",
                }
            )
            continue

        original_win = find_submitted_win(original_job_folder, material)
        original_wout = find_submitted_wout(original_job_folder, material)
        new_win = find_submitted_win(new_job_folder, material)
        new_wout = find_submitted_wout(new_job_folder, material)
        original_category = classify_win(original_win)
        new_category = classify_win(new_win)
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
            "original_input_error_ratio": case_row.get("original_input_error_ratio"),
            "new_input_error_ratio": case_row.get("new_input_error_ratio"),
            "delta_error_ratio_new_minus_original": delta_ratio,
            "ratio_fold_change_new_over_original": ratio_fold_change,
            "delta_log10_error_ratio_new_minus_original": delta_log10_ratio,
            "projection_similarity": win_choice_comparison["projection_similarity"],
            "window_strict_equal": win_choice_comparison["window_strict_equal"],
            "outer_window_similarity": win_choice_comparison["outer_window_similarity"],
            "frozen_window_similarity": win_choice_comparison["frozen_window_similarity"],
            "window_similarity": win_choice_comparison["window_similarity"],
            "original_projection_category": original_category,
            "new_projection_category": new_category,
            "changed_projection_category": str(original_category != new_category),
            "original_was_random": str(original_category == "random_projection_runs"),
            "new_still_random": str(new_category == "random_projection_runs"),
            "original_case_path": case_row.get("original_case_path"),
            "new_case_path": case_row.get("new_case_path"),
            "original_trial_name": case_row.get("original_trial_name"),
            "new_trial_name": case_row.get("new_trial_name"),
            "original_job_folder": original_job_folder.name,
            "new_job_folder": new_job_folder.name,
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
                    "case1_materials_json": str(CASE1_MATERIALS_JSON.relative_to(ROOT)),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("Compared materials: 0")
        print(f"Skipped materials: {len(skipped)}")
        return

    transitions = {
        original: {new: 0 for new in CATEGORY_KEYS}
        for original in CATEGORY_KEYS
    }
    for row in rows:
        transitions[row["original_projection_category"]][row["new_projection_category"]] += 1

    original_random_rows = [
        row for row in rows
        if row["original_projection_category"] == "random_projection_runs"
    ]
    original_random_new_counts = category_counts(
        original_random_rows,
        "new_projection_category",
    )

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
        "original_input_error_ratio",
        "new_input_error_ratio",
        "delta_error_ratio_new_minus_original",
        "ratio_fold_change_new_over_original",
        "delta_log10_error_ratio_new_minus_original",
        "projection_similarity",
        "window_strict_equal",
        "outer_window_similarity",
        "frozen_window_similarity",
        "window_similarity",
        "original_projection_category",
        "new_projection_category",
        "original_case_path",
        "new_case_path",
        "new_job_folder",
        "original_job_folder",
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
            "case1_materials_json": str(CASE1_MATERIALS_JSON.relative_to(ROOT)),
            "summary_json": str(SUMMARY_PATH.relative_to(ROOT)),
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "error_ratio_csv": str(OUTPUT_ERROR_CSV.relative_to(ROOT)),
            "error_ratio_json": str(OUTPUT_ERROR_JSON.relative_to(ROOT)),
            "heatmap_png": str(OUTPUT_HEATMAP.with_suffix(".png").relative_to(ROOT)),
            "heatmap_pdf": str(OUTPUT_HEATMAP.with_suffix(".pdf").relative_to(ROOT)),
        },
        "selection": (
            "For each case1_materials.json value tuple, value[1]/value[3] is the "
            "higher-number run treated as original, and value[0]/value[2] is the "
            "lower-number run treated as new. Error ratios are recomputed from "
            "run rmse_eV divided by reference_offmesh_rmse_eV for matching "
            "material and num_wann."
        ),
        "win_choice_comparison": (
            "projection_similarity is the multiset Jaccard similarity of normalized projections between "
            "the higher-number and lower-number submitted .win files. window_strict_equal and "
            "window_similarity is the combined 0-1 Jaccard similarity of actual outer and frozen "
            "window band masks computed from the paired .eig files."
        ),
        "classification": (
            "Projection categories are still written to the CSV/JSON for audit: begin projections with "
            "only random => random_projection_runs; any other projections block => explicit_projection_runs; "
            "no block or missing .win => none_or_implicit_projection_runs"
        ),
        "compared_materials": len(rows),
        "case1_material_count": len(case_rows),
        "skipped_materials": skipped,
        "original_counts_for_same_materials": category_counts(
            rows,
            "original_projection_category",
        ),
        "new_counts_for_same_materials": category_counts(
            rows,
            "new_projection_category",
        ),
        "original_materials_by_projection_category": category_materials(
            rows,
            "original_projection_category",
        ),
        "new_materials_by_projection_category": category_materials(
            rows,
            "new_projection_category",
        ),
        "transition_counts_original_to_new": transitions,
        "new_counts_for_original_random_materials": original_random_new_counts,
        "original_random_materials": sorted(
            row["material"] for row in original_random_rows
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Compared materials: {len(rows)}")
    print(f"Skipped materials: {len(skipped)}")
    print("Original counts for same materials:")
    print(json.dumps(summary["original_counts_for_same_materials"], indent=2))
    print("New counts for same materials:")
    print(json.dumps(summary["new_counts_for_same_materials"], indent=2))
    print("Transition counts, original -> new:")
    print(json.dumps(transitions, indent=2))
    print("Lower-number counts among materials whose higher-number run used random projections:")
    print(json.dumps(original_random_new_counts, indent=2))
    print(f"Wrote {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_ERROR_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_ERROR_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_HEATMAP.with_suffix('.png').relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_HEATMAP.with_suffix('.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
