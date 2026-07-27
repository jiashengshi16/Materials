#!/usr/bin/env python3
"""Run Gemini CLI directly on prior-run Wannierisation evidence.

No argparse. Edit MATERIALS and GEMINI_BIN below, then run:

    scripts/run_gemini_self_debug_reviews.py
"""

from __future__ import annotations
import csv
import json
import os
import re
import shutil
import subprocess
import threading

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HARBOR_DATASET_ROOT = ROOT / "harbor_datasets" / "wannier_200"
CANDIDATE_RUN_ERROR_TABLE = ROOT / "include_only_candidates.csv"
RUN_QUALITY_ERROR_TABLE = ROOT / "successful_run_errors.csv"

# Hardcoded experiment controls.
# Choose either "chemically similar" or "list".
MATERIAL_SELECTION_MODE = "chemically similar"

# MATERIALS = [

# 'Al18Co4',
# 'Al4Sc2',
# 'Li4O6Si2',
# 'Si6Y10',
# 'Mg2O10Ti4',
# "Al4Mn2O8",
# "C4O12Sr4",
# 'B2Ta',
# 'RuTi',
# 'Ag2Y',
# 'NNb',
# 'C2Cu2O6'
# ]

MODEL = "gemini-3.5-flash"
GEMINI_BIN = "gemini"
MAX_CONCURRENT_GEMINI = 12
OUTPUT_ROOT = ROOT / "jobsGeminiReviewsDeepseekIter3" / "gemini_self_debug_reviews"
RUN_ROOTS = [
    ROOT / "jobsDeepseekProTerminus2ControlledIter2",
]
NUM_WANN_JOB_RE = re.compile(
    r"^num_wann_ordered__(?P<timestamp>.+?)__pid(?P<pid>\d+)__"
    r"(?P<middle>.+?)__num_wann_(?P<num_wann>\d+)__(?P<material>.+)$"
)

# Deterministic AMN diagnostics. Raw .amn/.mmn/.chk/_hr.dat files are not staged
# for Gemini; the complete .amn is reduced numerically before the review starts.
AMN_EFFECTIVE_RANK_RELATIVE_TOLS = (1.0e-6, 1.0e-8, 1.0e-10)
AMN_WORST_KPOINT_LIMIT = 8
MAX_CONCURRENT_AMN_SUMMARIES = 2
AMN_SUMMARY_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_AMN_SUMMARIES)


@dataclass(frozen=True)
class TrialCase:
    material: str
    job_dir: Path
    trial_dir: Path
    attempt_dir: Path
    case_id: str
    job_metadata: dict[str, Any]
    manifest: dict[str, Any]

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_present(path: Path) -> Any | None:
    """Return parsed JSON when available and valid; otherwise return None."""
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def read_json_object_if_present(path: Path) -> dict[str, Any]:
    data = read_json_if_present(path)
    return data if isinstance(data, dict) else {}


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_file(src: Path | str, dst: Path | str) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_file_if_present(src: Path, dst: Path) -> bool:
    """Copy a file when it exists. Missing files are recorded by the caller, not fatal."""
    if not src.is_file():
        return False
    try:
        copy_file(src, dst)
    except OSError:
        return False
    return True


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def find_attempt_file(attempt: Path, material: str, suffix: str) -> Path:
    exact = attempt / f"{material}{suffix}"
    if exact.exists():
        return exact
    matches = sorted(attempt.glob(f"*{suffix}"))
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"Could not resolve *{suffix} for {material} in {attempt}")


def has_attempt_file(attempt: Path, material: str, suffix: str) -> bool:
    exact = attempt / f"{material}{suffix}"
    if exact.exists():
        return True
    return bool(sorted(attempt.glob(f"*{suffix}")))


def optional_attempt_file(attempt: Path, material: str, suffix: str) -> Path | None:
    """Resolve an exact per-material artifact, falling back to one unambiguous suffix match."""
    exact = attempt / f"{material}{suffix}"
    if exact.is_file():
        return exact
    matches = sorted(path for path in attempt.glob(f"*{suffix}") if path.is_file())
    return matches[0] if len(matches) == 1 else None


def finite_json_float(value: float) -> float | None:
    """Return a JSON-safe finite float; represent infinities/NaNs as null."""
    value = float(value)
    return value if np.isfinite(value) else None


def fortran_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def parse_win_numeric_parameters(path: Path) -> dict[str, int | float]:
    """Read the small set of numeric .win parameters needed for AMN diagnostics."""
    wanted = {
        "num_wann",
        "num_bands",
        "dis_win_min",
        "dis_win_max",
        "dis_froz_min",
        "dis_froz_max",
    }
    result: dict[str, int | float] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if "=" not in line:
            continue
        key, raw_value = (part.strip() for part in line.split("=", 1))
        key = key.lower()
        if key not in wanted:
            continue
        token = raw_value.split()[0].rstrip(",")
        try:
            if key in {"num_wann", "num_bands"}:
                result[key] = int(token)
            else:
                result[key] = fortran_float(token)
        except ValueError:
            continue
    return result


def amn_recipe_parameters(case: TrialCase, win_path: Path | None) -> tuple[dict[str, Any], list[str]]:
    """Collect recipe/window parameters without depending on any hidden reference data."""
    parameters: dict[str, Any] = {}
    sources: list[str] = []

    if win_path is not None:
        parsed_win = parse_win_numeric_parameters(win_path)
        if parsed_win:
            parameters.update(parsed_win)
            sources.append(str(win_path))

    wannier_parameters = case.manifest.get("wannier_parameters")
    if isinstance(wannier_parameters, dict):
        for key in (
            "num_wann",
            "num_bands",
            "dis_win_min",
            "dis_win_max",
            "dis_froz_min",
            "dis_froz_max",
        ):
            if key not in parameters and key in wannier_parameters:
                parameters[key] = wannier_parameters[key]
        sources.append("run_manifest.json:wannier_parameters")

    for key in ("num_wann", "num_bands"):
        if key not in parameters and key in case.manifest:
            parameters[key] = case.manifest[key]

    recipe_paths = (
        case.trial_dir / "artifacts" / "app" / "workflow" / "LOCKED_RECIPE.json",
        case.trial_dir / "artifacts" / "app" / "workflow" / "recipe_request.json",
    )
    for recipe_path in recipe_paths:
        recipe = read_json_object_if_present(recipe_path)
        if not recipe:
            continue
        for key in ("num_wann", "num_bands"):
            if key not in parameters and key in recipe:
                parameters[key] = recipe[key]
        windows = recipe.get("windows")
        if isinstance(windows, dict):
            for key in ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max"):
                if key not in parameters and key in windows:
                    parameters[key] = windows[key]
        sources.append(str(recipe_path))

    if "num_wann" not in parameters:
        folder_num_wann = case.job_metadata.get("num_wann_from_folder")
        if isinstance(folder_num_wann, int):
            parameters["num_wann"] = folder_num_wann
            sources.append("job_folder:num_wann")

    return parameters, sources


def parse_amn_matrix(path: Path) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Parse a Wannier90 .amn file into A[k, band, projection]."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        header_lines: list[str] = []
        dimensions: tuple[int, int, int] | None = None

        # Standard .amn has one comment line followed by three dimensions. Scan a
        # few lines defensively so harmless extra header text does not break review.
        for _ in range(20):
            line = handle.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) == 3:
                try:
                    values = tuple(int(part) for part in parts)
                except ValueError:
                    values = ()
                if len(values) == 3 and all(value > 0 for value in values):
                    dimensions = values  # num_bands, num_kpoints, num_projections
                    break
            header_lines.append(line.rstrip("\n"))

        if dimensions is None:
            raise ValueError("could not locate the .amn dimension line")

        num_bands, num_kpoints, num_projections = dimensions
        matrix = np.zeros(
            (num_kpoints, num_bands, num_projections),
            dtype=np.complex128,
        )
        seen = np.zeros(
            (num_kpoints, num_bands, num_projections),
            dtype=np.bool_,
        )

        parsed_records = 0
        duplicate_records = 0
        malformed_records = 0
        out_of_range_records = 0
        nonfinite_records = 0
        nonblank_data_lines = 0

        for line in handle:
            if not line.strip():
                continue
            nonblank_data_lines += 1
            parts = line.split()
            if len(parts) < 5:
                malformed_records += 1
                continue
            try:
                band_index = int(parts[0])
                projection_index = int(parts[1])
                kpoint_index = int(parts[2])
                real = fortran_float(parts[3])
                imag = fortran_float(parts[4])
            except ValueError:
                malformed_records += 1
                continue

            if not (
                1 <= band_index <= num_bands
                and 1 <= projection_index <= num_projections
                and 1 <= kpoint_index <= num_kpoints
            ):
                out_of_range_records += 1
                continue
            if not (np.isfinite(real) and np.isfinite(imag)):
                nonfinite_records += 1
                continue

            parsed_records += 1
            target = (
                kpoint_index - 1,
                band_index - 1,
                projection_index - 1,
            )
            if seen[target]:
                duplicate_records += 1
                continue
            seen[target] = True
            matrix[target] = complex(real, imag)

    expected_records = num_bands * num_kpoints * num_projections
    unique_records = int(seen.sum())
    missing_records = expected_records - unique_records
    integrity = {
        "expected_records": expected_records,
        "nonblank_data_lines": nonblank_data_lines,
        "parsed_records": parsed_records,
        "unique_records": unique_records,
        "missing_records": missing_records,
        "duplicate_records": duplicate_records,
        "malformed_records": malformed_records,
        "out_of_range_records": out_of_range_records,
        "nonfinite_records": nonfinite_records,
        "complete_unique_coverage": missing_records == 0,
        "analysis_reliable": (
            missing_records == 0
            and malformed_records == 0
            and out_of_range_records == 0
            and nonfinite_records == 0
        ),
    }
    return matrix, integrity, header_lines


def parse_eig_matrix(
    path: Path,
    *,
    num_bands: int,
    num_kpoints: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse .eig into energies[k, band] for the AMN band pool."""
    energies = np.full((num_kpoints, num_bands), np.nan, dtype=np.float64)
    parsed_records = 0
    duplicate_records = 0
    malformed_records = 0
    out_of_range_records = 0
    nonfinite_records = 0

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            malformed_records += 1
            continue
        try:
            band_index = int(parts[0])
            kpoint_index = int(parts[1])
            energy = fortran_float(parts[2])
        except ValueError:
            malformed_records += 1
            continue
        if not (
            1 <= band_index <= num_bands
            and 1 <= kpoint_index <= num_kpoints
        ):
            out_of_range_records += 1
            continue
        if not np.isfinite(energy):
            nonfinite_records += 1
            continue

        target = (kpoint_index - 1, band_index - 1)
        if np.isfinite(energies[target]):
            duplicate_records += 1
            continue
        energies[target] = energy
        parsed_records += 1

    expected_records = num_bands * num_kpoints
    missing_records = int(np.isnan(energies).sum())
    return energies, {
        "expected_records_for_amn_band_pool": expected_records,
        "parsed_unique_records": parsed_records,
        "missing_records_for_amn_band_pool": missing_records,
        "duplicate_records": duplicate_records,
        "malformed_records": malformed_records,
        "out_of_range_records": out_of_range_records,
        "nonfinite_records": nonfinite_records,
        "complete_for_amn_band_pool": missing_records == 0,
    }

def matrix_singular_metrics(matrix: np.ndarray) -> dict[str, Any]:
    """Compute compact SVD conditioning and effective-rank sensitivity metrics."""
    num_columns = int(matrix.shape[1])

    if matrix.shape[0] == 0 or num_columns == 0:
        singular_values = np.zeros(num_columns, dtype=np.float64)
    else:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if singular_values.size < num_columns:
            singular_values = np.pad(
                singular_values,
                (0, num_columns - singular_values.size),
                mode="constant",
            )

    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
    normalized_smallest = sigma_min / sigma_max if sigma_max > 0.0 else 0.0

    effective_ranks: dict[str, int] = {}
    for tolerance in AMN_EFFECTIVE_RANK_RELATIVE_TOLS:
        threshold = tolerance * sigma_max
        effective_ranks[f"{tolerance:.0e}"] = int(
            np.count_nonzero(singular_values > threshold)
        )

    return {
        "effective_rank_by_relative_threshold": effective_ranks,
        "normalized_smallest_singular_value": normalized_smallest,
    }


def projection_pair_correlation(matrix: np.ndarray) -> tuple[float, tuple[int, int] | None]:
    """Return the largest normalized absolute column correlation at one k-point."""
    num_projections = matrix.shape[1]
    if num_projections < 2 or matrix.shape[0] == 0:
        return 0.0, None

    gram = matrix.conj().T @ matrix
    norms = np.sqrt(np.maximum(np.real(np.diag(gram)), 0.0))
    denominator = norms[:, None] * norms[None, :]
    correlation = np.zeros_like(np.real(gram), dtype=np.float64)
    valid = denominator > 0.0
    correlation[valid] = np.abs(gram[valid]) / denominator[valid]
    np.fill_diagonal(correlation, 0.0)

    flat_index = int(np.argmax(correlation))
    row, column = np.unravel_index(flat_index, correlation.shape)
    maximum = float(correlation[row, column])
    if maximum <= 0.0:
        return 0.0, None
    return maximum, (int(row) + 1, int(column) + 1)

def analyze_amn_pool(
    amn: np.ndarray,
    *,
    band_masks: np.ndarray | None = None,
) -> dict[str, Any]:
    """Aggregate compact AMN conditioning and effective-rank sensitivity diagnostics."""
    num_kpoints, _num_bands, num_projections = amn.shape

    normalized_smallest_values: list[float] = []
    worst_kpoints: list[dict[str, Any]] = []
    effective_rank_sensitivity_counts = {
        f"{tolerance:.0e}": 0
        for tolerance in AMN_EFFECTIVE_RANK_RELATIVE_TOLS
    }

    maximum_pair_correlation = 0.0
    maximum_pair: tuple[int, int] | None = None
    maximum_pair_kpoint: int | None = None
    selected_band_counts: list[int] = []

    for kpoint_index in range(num_kpoints):
        matrix = amn[kpoint_index]

        if band_masks is not None:
            mask = band_masks[kpoint_index]
            matrix = matrix[mask]

        selected_band_count = int(matrix.shape[0])
        selected_band_counts.append(selected_band_count)

        metrics = matrix_singular_metrics(matrix)

        normalized_smallest = float(
            metrics["normalized_smallest_singular_value"]
        )
        normalized_smallest_values.append(normalized_smallest)

        effective_ranks = metrics[
            "effective_rank_by_relative_threshold"
        ]

        for tolerance_key, effective_rank in effective_ranks.items():
            if int(effective_rank) < num_projections:
                effective_rank_sensitivity_counts[tolerance_key] += 1

        pair_correlation, pair = projection_pair_correlation(matrix)
        if pair_correlation > maximum_pair_correlation:
            maximum_pair_correlation = pair_correlation
            maximum_pair = pair
            maximum_pair_kpoint = kpoint_index + 1

        worst_kpoints.append(
            {
                "kpoint_index": kpoint_index + 1,
                "bands_in_analyzed_pool": selected_band_count,
                "normalized_smallest_singular_value": normalized_smallest,
                "effective_rank_by_relative_threshold": effective_ranks,
            }
        )

    # These are the k-points with the smallest relative singular value.
    # This is a numerical ordering only; it is not a categorical bad/good judgment.
    worst_kpoints.sort(
        key=lambda item: float(
            item["normalized_smallest_singular_value"]
        )
    )
    worst_kpoints = worst_kpoints[:AMN_WORST_KPOINT_LIMIT]

    return {
        "num_projection_columns": num_projections,

        "effective_rank_sensitivity": {
            tolerance: {
                "kpoints_below_projection_count": count,
                "fraction_of_kpoints": (
                    count / num_kpoints
                    if num_kpoints
                    else None
                ),
            }
            for tolerance, count
            in effective_rank_sensitivity_counts.items()
        },

        "minimum_normalized_smallest_singular_value": (
            min(normalized_smallest_values)
            if normalized_smallest_values
            else None
        ),

        "median_normalized_smallest_singular_value": (
            float(np.median(normalized_smallest_values))
            if normalized_smallest_values
            else None
        ),

        "minimum_bands_in_analyzed_pool": (
            min(selected_band_counts)
            if selected_band_counts
            else 0
        ),

        "maximum_bands_in_analyzed_pool": (
            max(selected_band_counts)
            if selected_band_counts
            else 0
        ),

        "worst_kpoints": worst_kpoints,

        # Retained as a raw numerical observation in the JSON.
        # Do not automatically promote a single maximum correlation
        # into a diagnosis in direct_observations.
        "projection_redundancy": {
            "maximum_normalized_pair_correlation":
                maximum_pair_correlation,
            "most_correlated_projection_pair": (
                list(maximum_pair)
                if maximum_pair is not None
                else None
            ),
            "kpoint_index": maximum_pair_kpoint,
        },
    }

def effective_rank_sensitivity_text(
    label: str,
    analysis: dict[str, Any],
    num_kpoints: int,
) -> str:
    parts = []

    for tolerance, values in analysis["effective_rank_sensitivity"].items():
        count = int(values["kpoints_below_projection_count"])
        parts.append(f"{tolerance}: {count}/{num_kpoints}")

    return (
        f"{label} effective-rank sensitivity "
        f"(k-points with effective rank below the projection count): "
        + ", ".join(parts)
        + ". These values are threshold-dependent numerical descriptors, "
        "not categorical proof of a defective projection recipe."
    )

def summarize_amn_case(case: TrialCase) -> dict[str, Any]:
    """Create deterministic projection diagnostics from the complete raw .amn file."""
    material = case.material
    amn_path = optional_attempt_file(case.attempt_dir, material, ".amn")
    win_path = optional_attempt_file(case.attempt_dir, material, ".win")
    eig_path = optional_attempt_file(case.attempt_dir, material, ".eig")

    parameters, parameter_sources = amn_recipe_parameters(case, win_path)
    expected_num_wann = parameters.get("num_wann")
    try:
        expected_num_wann = (
            int(expected_num_wann)
            if expected_num_wann is not None
            else None
        )
    except (TypeError, ValueError):
        expected_num_wann = None

    summary: dict[str, Any] = {
        "summary_schema_version": 2,
        "material": material,
        "status": "missing_amn",
        "source_file": amn_path.name if amn_path is not None else None,
        "source_file_size_bytes": (
            amn_path.stat().st_size
            if amn_path is not None
            else None
        ),
        "parameter_sources": parameter_sources,
        "expected_num_wann": expected_num_wann,
        "method": {
            "description": (
                "Deterministic numerical reduction of the complete Wannier90 "
                ".amn matrix before Gemini review."
            ),
            "matrix_convention": "A[kpoint, band, projection]",
            "effective_rank_relative_thresholds": list(
                    AMN_EFFECTIVE_RANK_RELATIVE_TOLS
                ),
            "outer_window_rule": "dis_win_min <= eigenvalue <= dis_win_max",
        },
        "interpretation_limits": [
            (
                "AMN dimensions, file-integrity counts, band counts, singular-value "
                "statistics, effective-rank sensitivity values, and projection-pair "
                "correlations are direct numerical observations."
            ),
            (
                "Effective-rank results are threshold-dependent numerical descriptors, "
                "not categorical definitions of mathematical rank."
            ),
            (
                "Small singular values or high pair correlations can indicate numerical "
                "conditioning or redundancy risk but do not by themselves establish a "
                "defective projection recipe or explain the final outcome."
            ),
            (
                "Causal interpretation must be checked against the old decision chain, "
                ".wout, execution logs, and verifier diagnostics."
            ),
        ],
    }

    if amn_path is None:
        summary["direct_observations"] = [
            "No raw .amn file was available in the source attempt, so deterministic AMN diagnostics could not be generated."
        ]
        return summary

    try:
        with AMN_SUMMARY_SEMAPHORE:
            amn, integrity, header_lines = parse_amn_matrix(amn_path)
            num_kpoints, num_bands, num_projections = amn.shape

            summary["status"] = (
                "parsed"
                if integrity["analysis_reliable"]
                else "parsed_with_integrity_warnings"
            )
            summary["header_lines"] = header_lines[:5]
            summary["dimensions"] = {
                "num_bands": num_bands,
                "num_kpoints": num_kpoints,
                "num_projections": num_projections,
                "expected_num_wann": expected_num_wann,
                "projection_count_matches_expected_num_wann": (
                    num_projections == expected_num_wann
                    if expected_num_wann is not None
                    else None
                ),
            }
            summary["file_integrity"] = integrity
            summary["full_band_pool"] = analyze_amn_pool(amn)

            outer_window_keys = ("dis_win_min", "dis_win_max")
            frozen_window_keys = ("dis_froz_min", "dis_froz_max")
            have_outer_window = all(key in parameters for key in outer_window_keys)
            have_frozen_window = all(key in parameters for key in frozen_window_keys)

            energies = None
            eig_integrity = None
            if eig_path is not None:
                energies, eig_integrity = parse_eig_matrix(
                    eig_path,
                    num_bands=num_bands,
                    num_kpoints=num_kpoints,
                )

            if energies is not None and have_outer_window:
                dis_win_min = float(parameters["dis_win_min"])
                dis_win_max = float(parameters["dis_win_max"])
                finite_energies = np.isfinite(energies)
                band_masks = (
                    finite_energies
                    & (energies >= dis_win_min)
                    & (energies <= dis_win_max)
                )
                outer = analyze_amn_pool(amn, band_masks=band_masks)
                outer.update(
                    {
                        "status": "available",
                        "eig_source_file": eig_path.name,
                        "dis_win_min_ev": dis_win_min,
                        "dis_win_max_ev": dis_win_max,
                        "eig_integrity": eig_integrity,
                    }
                )
                summary["outer_window_restricted"] = outer
            else:
                missing_reasons: list[str] = []
                if eig_path is None:
                    missing_reasons.append("missing .eig file")
                if not have_outer_window:
                    missing_reasons.append("missing dis_win_min/dis_win_max")
                summary["outer_window_restricted"] = {
                    "status": "unavailable",
                    "reasons": missing_reasons,
                }

            # Frozen-window diagnostics intentionally report only band counts.
            # Requiring full projection rank inside the frozen window would be
            # conceptually wrong whenever fewer than num_wann states are frozen.
            if energies is not None and have_frozen_window:
                dis_froz_min = float(parameters["dis_froz_min"])
                dis_froz_max = float(parameters["dis_froz_max"])
                finite_energies = np.isfinite(energies)
                frozen_masks = (
                    finite_energies
                    & (energies >= dis_froz_min)
                    & (energies <= dis_froz_max)
                )
                frozen_counts = np.count_nonzero(frozen_masks, axis=1)
                exceeds_num_wann = (
                    int(np.count_nonzero(frozen_counts > expected_num_wann))
                    if expected_num_wann is not None
                    else None
                )
                summary["frozen_window_band_counts"] = {
                    "status": "available",
                    "eig_source_file": eig_path.name,
                    "dis_froz_min_ev": dis_froz_min,
                    "dis_froz_max_ev": dis_froz_max,
                    "minimum_frozen_band_count": int(frozen_counts.min()) if frozen_counts.size else 0,
                    "maximum_frozen_band_count": int(frozen_counts.max()) if frozen_counts.size else 0,
                    "median_frozen_band_count": float(np.median(frozen_counts)) if frozen_counts.size else 0.0,
                    "kpoints_with_frozen_band_count_above_num_wann": exceeds_num_wann,
                    "frozen_band_count_exceeds_num_wann": (
                        exceeds_num_wann > 0 if exceeds_num_wann is not None else None
                    ),
                    "eig_integrity": eig_integrity,
                    "interpretation": (
                        "This checks the direct Wannier90 consistency condition that the number of frozen bands at any k-point should not exceed num_wann. No full-rank projection test is imposed on the frozen-only subspace."
                    ),
                }
            else:
                missing_reasons: list[str] = []
                if eig_path is None:
                    missing_reasons.append("missing .eig file")
                if not have_frozen_window:
                    missing_reasons.append("missing dis_froz_min/dis_froz_max")
                summary["frozen_window_band_counts"] = {
                    "status": "unavailable",
                    "reasons": missing_reasons,
                }

            observations: list[str] = []

            # File integrity is a genuinely direct observation.
            if integrity["complete_unique_coverage"]:
                observations.append(
                    "The AMN parser found complete unique coverage of the "
                    "expected matrix records."
                )
            else:
                observations.append(
                    "The AMN file does not have complete unique record coverage; "
                    "numerical diagnostics should be treated cautiously."
                )

            # Projection count versus num_wann is also a direct structural observation.
            if expected_num_wann is not None:
                if num_projections == expected_num_wann:
                    observations.append(
                        f"The AMN contains {num_projections} projection columns, "
                        f"matching num_wann={expected_num_wann}."
                    )
                else:
                    observations.append(
                        f"The AMN contains {num_projections} projection columns, "
                        f"while num_wann={expected_num_wann}."
                    )

            # Report threshold-dependent sensitivity without converting any
            # particular threshold into a categorical rank verdict.
            observations.append(
                effective_rank_sensitivity_text(
                    "Full-band AMN",
                    summary["full_band_pool"],
                    num_kpoints,
                )
            )

            # The outer-window version is usually the more directly relevant
            # AMN view for disentanglement, when it can be constructed.
            outer = summary["outer_window_restricted"]
            if outer.get("status") == "available":
                observations.append(
                    effective_rank_sensitivity_text(
                        "Outer-window AMN",
                        outer,
                        num_kpoints,
                    )
                )
                observations.append(
                    "The outer-window AMN statistics use this run's own .eig "
                    "energies and dis_win_min/dis_win_max values."
                )

            # Frozen-window consistency is a direct count-based observation.
            frozen = summary["frozen_window_band_counts"]
            if frozen.get("status") == "available":
                exceed_count = frozen.get(
                    "kpoints_with_frozen_band_count_above_num_wann"
                )
                if exceed_count is not None:
                    observations.append(
                        f"The frozen window contains more than num_wann bands "
                        f"at {exceed_count} of {num_kpoints} k-points."
                    )

            summary["direct_observations"] = observations
            return summary
    except Exception as exc:
        summary["status"] = "parse_error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["direct_observations"] = [
            "The raw .amn file existed, but deterministic preprocessing failed; do not infer projection quality from the missing summary."
        ]
        return summary


def optional_attempt_evidence_files(attempt: Path, material: str) -> list[Path]:
    """Return optional text-like artifacts that can improve forensic reviews."""
    patterns = (
        f"{material}.eig",
        f"{material}.nnkp",
        f"{material}.pp.log",
        "*.pp.log",
        "*pw2wan*.log",
        "*.werr",
        "*.err",
    )
    files: dict[Path, None] = {}
    for pattern in patterns:
        for path in sorted(attempt.glob(pattern)):
            if path.is_file():
                files[path] = None
    return list(files)


def optional_workflow_evidence_files(trial_dir: Path) -> list[tuple[Path, Path]]:
    """Return workflow-contract and runner artifacts for forensic review staging."""
    relative_paths = (
        "artifacts/app/workflow/recipe_request.json",
        "artifacts/app/workflow/compile_recipe_report.json",
        "artifacts/app/workflow/locked_runner.log",
        "artifacts/app/workflow/LOCKED_RECIPE.json",
        "artifacts/app/workflow/DECISIONS.md",
        "artifacts/app/workflow/locked_runner_state.json",
        "config.json",
        "exception.txt",
        "result.json",
        "trial.log",
        "verifier/test-stdout.txt",
        "verifier/test-stderr.txt",
    )
    files: list[tuple[Path, Path]] = []
    for relative in relative_paths:
        src = trial_dir / relative
        if src.is_file():
            files.append((src, Path(relative)))

    compile_reports_dir = trial_dir / "artifacts" / "app" / "workflow" / "compile_recipe_reports"
    for src in sorted(compile_reports_dir.glob("compile_attempt_*.json")):
        if src.is_file():
            files.append((
                src,
                Path("artifacts/app/workflow/compile_recipe_reports") / src.name,
            ))
    return files


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_job_name(job_dir: Path) -> dict[str, Any]:
    match = NUM_WANN_JOB_RE.match(job_dir.name)
    if not match:
        return {
            "job_folder": job_dir.name,
            "job_timestamp": None,
            "pid": None,
            "job_middle": None,
            "ordinal": None,
            "attempt_from_folder": None,
            "num_wann_from_folder": None,
            "material_from_folder": None,
        }

    groups = match.groupdict()
    middle = groups["middle"]
    ordinal = int(middle) if middle.isdigit() else None
    attempt_match = re.search(r"(?:^|__)attempt_(\d+)(?:__|$)", middle)
    attempt_from_folder = int(attempt_match.group(1)) if attempt_match else None

    return {
        "job_folder": job_dir.name,
        "job_timestamp": groups["timestamp"],
        "pid": int(groups["pid"]),
        "job_middle": middle,
        "ordinal": ordinal,
        "attempt_from_folder": attempt_from_folder,
        "num_wann_from_folder": int(groups["num_wann"]),
        "material_from_folder": groups["material"],
    }


def case_id_for(job_metadata: dict[str, Any], trial_dir: Path) -> str:
    run_root = job_metadata.get("run_root") or "unknown_root"
    timestamp = job_metadata.get("job_timestamp") or "unknown_time"
    pid = job_metadata.get("pid")
    ordinal = job_metadata.get("ordinal")
    middle = job_metadata.get("job_middle")
    num_wann = job_metadata.get("num_wann_from_folder")
    middle_label = (
        f"ordinal_{ordinal:04d}"
        if isinstance(ordinal, int)
        else str(middle or "ordinal_unknown")
    )

    parts = [
        str(run_root),
        str(timestamp),
        f"pid{pid}" if isinstance(pid, int) else "pid_unknown",
        middle_label,
        f"num_wann_{num_wann:03d}" if isinstance(num_wann, int) else "num_wann_unknown",
        trial_dir.name,
    ]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", "__".join(parts)).strip("_")


def candidate_trial_dirs(job_dir: Path) -> list[Path]:
    # If the job itself looks like a trial, use it.
    if any(
        (job_dir / name).exists()
        for name in ("artifacts", "agent", "verifier")
    ):
        return [job_dir]

    # Otherwise, look for immediate child trial directories.
    trials = [
        path
        for path in sorted(job_dir.iterdir())
        if path.is_dir()
        and any(
            (path / name).exists()
            for name in ("artifacts", "agent", "verifier")
        )
    ]

    # Absolute last resort: still return the job directory.
    return trials or [job_dir]


def trial_attempt_dir(trial_dir: Path) -> Path | None:
    """Find the best available artifact directory without requiring a fixed layout."""

    candidates = [
        trial_dir / "artifacts" / "attempt_1",
        trial_dir / "artifacts" / "logs" / "artifacts" / "attempt_1",
        trial_dir / "attempt_1",
        trial_dir / "artifacts",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    # Search recursively for any attempt_1 directory.
    matches = sorted(
        path for path in trial_dir.rglob("attempt_1")
        if path.is_dir()
    )
    if matches:
        return matches[0]

    # Last resort: use the trial directory itself.
    # This allows a case to exist even if no artifact directory was produced.
    return trial_dir


def read_json_object(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return data


def validate_manifest_material(manifest: dict[str, Any], material: str, manifest_path: Path) -> None:
    manifest_material = manifest.get("material_id") or manifest.get("material")
    if isinstance(manifest_material, str) and manifest_material == material:
        return

    seedname = manifest.get("seedname")
    if isinstance(seedname, str) and seedname == material:
        return

    if isinstance(manifest_material, str):
        raise SystemExit(
            f"Manifest material mismatch for {material}: {manifest_path} "
            f"contains {manifest_material!r}"
        )


def find_trial_cases(material: str) -> list[TrialCase]:
    cases: list[TrialCase] = []

    for run_root in RUN_ROOTS:
        if not run_root.is_dir():
            continue

        for job_dir in sorted(path for path in run_root.glob("num_wann_ordered*") if path.is_dir()):
            job_metadata = {
                **parse_job_name(job_dir),
                "run_root": run_root.name,
                "run_root_path": display_path(run_root),
            }
            if job_metadata.get("material_from_folder") != material:
                continue

            for trial_dir in candidate_trial_dirs(job_dir):
                attempt_dir = trial_attempt_dir(trial_dir)
                if attempt_dir is None:
                    continue
                manifest_path = attempt_dir / "run_manifest.json"
                manifest = read_json_object_if_present(manifest_path)
                if not manifest_path.is_file():
                    print(
                        f"Warning for {material}: missing run_manifest.json at {manifest_path}; "
                        "continuing with empty metadata."
                    )
                elif not manifest:
                    print(
                        f"Warning for {material}: run_manifest.json is invalid or not a JSON object "
                        f"at {manifest_path}; continuing with empty metadata."
                    )
                else:
                    try:
                        validate_manifest_material(manifest, material, manifest_path)
                    except SystemExit as exc:
                        print(f"Warning for {material}: {exc}; continuing anyway.")

                if not has_attempt_file(attempt_dir, material, ".win"):
                    print(
                        f"Warning for {material}: missing .win file in {attempt_dir}; "
                        "continuing with the remaining evidence."
                    )

                cases.append(
                    TrialCase(
                        material=material,
                        job_dir=job_dir,
                        trial_dir=trial_dir,
                        attempt_dir=attempt_dir,
                        case_id=case_id_for(job_metadata, trial_dir),
                        job_metadata=job_metadata,
                        manifest=manifest,
                    )
                )

    if not cases:
        print(
            f"No discoverable trial folders for {material}: "
            f"under {', '.join(display_path(root) for root in RUN_ROOTS)}"
        )
        return []

    return cases


def dataset_task_instruction_path(material: str) -> Path | None:
    material_dir = HARBOR_DATASET_ROOT / material

    candidates = [
        material_dir / "instruction.md",
        material_dir / "instructions.md",
        material_dir / "prompt.md",
        material_dir / "task.md",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = sorted(material_dir.glob("*instruction*.md"))
    if len(matches) == 1:
        return matches[0]

    matches = sorted(material_dir.glob("*instructions*.md"))
    if len(matches) == 1:
        return matches[0]

    return None


def first_user_message_from_trajectory(trial_dir: Path) -> str | None:
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    if not trajectory_path.is_file():
        return None

    try:
        data = read_json(trajectory_path)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    steps = data.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict) or step.get("source") != "user":
                continue
            message = step.get("message")
            if isinstance(message, str) and message.strip():
                return message.rstrip() + "\n"

    messages = data.get("messages")
    if isinstance(messages, list):
        for message_record in messages:
            if not isinstance(message_record, dict):
                continue
            if message_record.get("role") not in {"user", "human"}:
                continue
            message = message_record.get("content")
            if isinstance(message, str) and message.strip():
                return message.rstrip() + "\n"

    return None


def original_task_instructions(case: TrialCase) -> tuple[str, str]:
    trajectory_prompt = first_user_message_from_trajectory(case.trial_dir)
    if trajectory_prompt is not None:
        return trajectory_prompt, "agent/trajectory.json:first user message"

    instruction_path = dataset_task_instruction_path(case.material)
    if instruction_path is not None:
        try:
            return instruction_path.read_text(encoding="utf-8"), display_path(instruction_path)
        except OSError:
            pass

    return (
        "Original task instructions were not available for this case.\n",
        "not available",
    )


def manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "material_id",
        "seedname",
        "attempt",
        "status",
        "executed_successfully",
        "target_dft_band_start",
        "target_dft_band_end",
        "num_wann",
        "num_bands",
    )
    return {key: manifest.get(key) for key in keys if key in manifest}


def run_quality_metrics(case: TrialCase) -> dict[str, Any]:
    """Load this run's RMSE and the same-material reference RMSE from the CSV."""
    result: dict[str, Any] = {
        "status": "unavailable",
        "source_csv": display_path(RUN_QUALITY_ERROR_TABLE),
        "candidate_error_eV": None,
        "reference_error_eV": None,
        "candidate_to_reference_error_ratio": None,
        "matched_run_id": None,
    }

    if not RUN_QUALITY_ERROR_TABLE.is_file():
        result["reason"] = "run-quality CSV not found"
        return result

    required_columns = {
        "material",
        "run_id",
        "gemini_error_eV",
        "reference_error_eV",
    }
    case_suffix = f"/{case.job_dir.name}/{case.trial_dir.name}"
    material_rows: list[dict[str, str]] = []
    matching_rows: list[dict[str, str]] = []

    with RUN_QUALITY_ERROR_TABLE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            result["reason"] = (
                "run-quality CSV must contain material, run_id, gemini_error_eV, "
                "and reference_error_eV columns"
            )
            return result

        for row in reader:
            if (row.get("material") or "").strip() != case.material:
                continue
            material_rows.append(row)
            run_id = (row.get("run_id") or "").strip().replace("\\", "/")
            if run_id.endswith(case_suffix):
                matching_rows.append(row)

    reference_values: set[float] = set()
    for row in material_rows:
        try:
            reference_values.add(float((row.get("reference_error_eV") or "").strip()))
        except ValueError:
            continue

    if len(reference_values) == 1:
        result["reference_error_eV"] = reference_values.pop()
    elif len(reference_values) > 1:
        result["reason"] = "multiple reference_error_eV values found for material"
        return result

    if not matching_rows:
        result["reason"] = "no CSV row matched this case's job/trial path"
        return result

    matched = matching_rows[0]
    try:
        candidate_error = float((matched.get("gemini_error_eV") or "").strip())
    except ValueError:
        result["reason"] = "matched row has invalid gemini_error_eV"
        return result

    reference_error = result["reference_error_eV"]
    if not isinstance(reference_error, float) or reference_error <= 0.0:
        result["reason"] = "reference_error_eV is missing or not positive"
        return result

    result.update(
        {
            "status": "available",
            "candidate_error_eV": candidate_error,
            "candidate_to_reference_error_ratio": candidate_error / reference_error,
            "matched_run_id": (matched.get("run_id") or "").strip(),
        }
    )
    return result


def prompt_text(case: TrialCase) -> str:
    material = case.material
    num_wann = case.job_metadata.get("num_wann_from_folder")
    target_start = case.manifest.get("target_dft_band_start")
    target_end = case.manifest.get("target_dft_band_end")

    return f"""# Gemini Self-Debug Review: {material}

You are reviewing a Wannierisation trajectory for `{material}`.
This exact case is:

- case id: `{case.case_id}`
- source run root: `{case.job_metadata.get("run_root")}`
- source job folder: `{case.job_dir.name}`
- source trial folder: `{case.trial_dir.name}`
- `num_wann` from the job folder: `{num_wann}`
- target DFT bands from the run manifest: `{target_start}` to `{target_end}`

If another case directory exists for the same material, treat it as a separate
old run or separate Wannierisation. Diagnose only the current case directory.

This is forensic analysis only. Do not rerun QE, do not produce a new
Wannierisation, and do not browse the internet. Use only files in this case
directory. Do not read files outside this case directory, even if trajectory
logs mention outside paths.

Your job is to reconstruct, as closely as the logs allow, the old decision chain 
using the original task instructions and old run logs. Treat 
`case_files/original_task_instructions.md` as the task prompt that was available 
to the old model during the original run.

Evaluate the trajectory fairly. If the old run made scientifically reasonable
choices given the information available at the time, say so and do not force a
critique. Only identify avoidable mistakes when the task materials, trajectory, 
logs, or final diagnostics provide evidence that a specific choice was not ideal/poor, 
contradicted by later run output, or led to an avoidable failure or high RMSE.
For any such issue, explain the evidence-backed diagnosis using only information
that is present in the staged case files. 

Follow this evidence-reading order. The order is part of the forensic method:
first reconstruct what the old model knew and chose, then inspect objective
recipe diagnostics, and only then inspect later outcomes. Do not use Stage 3
outcomes to retroactively claim that the old model knew something it could not
have known at decision time. After completing all three stages, integrate evidence across
stages while preserving the distinction between what was knowable at decision
time, direct numerical diagnostics, and later outcomes.

## STAGE 1 — Reconstruct the old decision and contract

Read ALL OF THESE first, if present:

- `case_files/original_task_instructions.md`
- `case_files/agent/trajectory.json`
- `case_files/workflow_contract/artifacts/app/workflow/recipe_request.json`
- `case_files/workflow_contract/artifacts/app/workflow/compile_recipe_reports/compile_attempt_*.json`
- `case_files/workflow_contract/artifacts/app/workflow/compile_recipe_report.json`
- `case_files/workflow_contract/artifacts/app/workflow/LOCKED_RECIPE.json`
- `case_files/workflow_contract/artifacts/app/workflow/DECISIONS.md`
- `case_files/artifacts/attempt_1/{material}.win`

From Stage 1, establish exactly what the old model proposed, what evidence it
used at the time, which recipe fields it could control, what compile/preflight
feedback it received, and what the locked runner changed or hard-coded. 
You must inspect the workflow compiler reports before judging whether a recipe 
failed scientifically. The folder compile_recipe_reports/ contains 
per-compile-attempt feedback; compile_recipe_report.json is the final/summary report.

## STAGE 2 — Inspect objective recipe diagnostics

Only after Stage 1, read ALL OF THESE if present:

- `case_files/artifacts/attempt_1/{material}.amn_summary.json`
- `case_files/artifacts/attempt_1/{material}.eig`
- `case_files/artifacts/attempt_1/{material}.nnkp`
- `case_files/artifacts/attempt_1/{material}.pp.log`
- `case_files/artifacts/attempt_1/*pw2wan*.log`
- other compile/preprocessor error logs relevant to the recipe

`{material}.amn_summary.json` is generated deterministically from the complete
raw `.amn` before Gemini starts. Treat its dimensions, integrity counts,
singular-value statistics, threshold-dependent effective-rank sensitivity,
band counts, and outer-window-restricted statistics as DIRECT NUMERICAL
OBSERVATIONS.

Interpret the AMN diagnostics carefully:

- Effective-rank counts are numerical sensitivity measurements at explicitly
  stated relative SVD thresholds. They are not categorical definitions of
  mathematical rank and must not be rewritten as simply "rank deficient" or
  "full rank" without qualification.
- A small normalized singular value or high projection-pair correlation can
  identify possible conditioning or redundancy risk, but does not by itself
  show that the projection recipe is defective or explain the final outcome.
- Compare patterns across thresholds rather than privileging any single
  threshold.
- Projection-count mismatches and file-integrity problems are direct structural
  observations.
- Do not claim that an AMN observation caused the final outcome unless later
  run evidence supports that causal link. Otherwise label the explanation
  `plausible_but_unproven`.

The raw `.amn`, `.mmn`, `.chk`, and `_hr.dat` files are intentionally not
staged for Gemini. Do not request them, attempt to infer their raw contents, or
list their intentional absence as an evidence gap. The complete `.amn` has
already been reduced numerically. Raw `.mmn` and `_hr.dat` are not needed for
the default forensic review; `.chk` is binary.

## STAGE 3 — Inspect later outcomes

Only after Stages 1 and 2, read ALL OF THESE if present:

- `case_files/artifacts/attempt_1/{material}.wout`
- `case_files/artifacts/attempt_1/run_manifest.json`
- `case_files/case_metadata.json`
- `case_files/verifier/diagnostics.json`
- `case_files/run_quality_metrics.json`
- `case_files/workflow_contract/artifacts/app/workflow/locked_runner.log`
- `case_files/workflow_contract/artifacts/app/workflow/locked_runner_state.json`
- `case_files/workflow_contract/config.json`
- `case_files/workflow_contract/exception.txt`
- `case_files/workflow_contract/result.json`
- `case_files/workflow_contract/trial.log`
- `case_files/workflow_contract/verifier/test-stdout.txt`
- `case_files/workflow_contract/verifier/test-stderr.txt`
- `case_files/artifacts/attempt_1/*.werr`
- `case_files/artifacts/attempt_1/*.err`

Compare these later observations against the Stage 1 decision chain and the
Stage 2 objective diagnostics. Separate symptoms from causes and explicitly
distinguish old-model responsibility from locked-runner behavior.

The per-run verifier diagnostics and `case_files/run_quality_metrics.json` are
allowed only as scalar final outcome metrics for this specific run. The latter
contains the candidate RMSE, the same-material `reference_error_eV`, and their
ratio; it does not reveal the hidden reference recipe or reference-only methods.
Do not use either file to infer hidden reference settings. The diagnosis must
obey `case_files/original_task_instructions.md` when judging what the old model
could know or control.

Interpret the two error measures differently and use both:

- The raw candidate RMSE in eV is the absolute interpolation error for this run.
- The error ratio is `candidate RMSE / reference_error_eV` for another system on
  the same material. A ratio below 1 means the candidate beat that reference, a
  ratio near 1 means comparable performance, and a ratio above 1 means worse
  relative performance.
- Raw RMSE alone can mis-rank run quality across materials: a numerically small
  RMSE can still be poor relative to an easy same-material reference, while a
  larger RMSE can still be comparatively strong for a difficult material.
- Use the ratio to improve comparison of candidate-run quality, but never rank a
  candidate's importance or relevance to a new target material from the ratio
  alone. Also consider chemical similarity, transferability of the diagnosed
  decisions, evidence quality, and what controls were available to the old run.

A recorded `TimeoutError` is not by itself evidence that the candidate run was
unsuccessful. If the workflow ultimately completed successfully and produced a
low error ratio, the run may still be judged good and successful. Distinguish a
transient timeout from an incomplete run, missing outputs, or failed scoring.

Do not handwave from aggregate statistics. The core diagnosis must come from
this material's `.win`, `.wout`, deterministic `.amn_summary.json`, `.eig`,
`.nnkp`, preprocessor/pw2wannier/wannier90/error logs, workflow-contract files,
run manifest, trajectory, and per-run verifier diagnostics, if present.

Use the workflow-contract files to distinguish what the old model could control
from what the locked runner hard-coded. Do not recommend or criticize the old
model for failing to set a field unless `recipe_request.json`, the original
task instructions, or the runner logs show that field was actually available to
the old model.

Write exactly these two files:

- `self_debug_report.md`
- `self_debug_report.json`

The Markdown report must be step-by-step and specific. For each substantive
decision in the old trajectory, judge whether it was good, bad, mixed, or
uncertain, and explain why USING CONCRETE EVIDENCE from `.win`, `.wout`,
run manifest, workflow-contract files, trajectory reasoning, and final error
metrics. Cite the file paths you used, and include line numbers when you have
them from grep, rg, nl, or similar inspection. Cover at least:

1. projection choice
2. `num_wann` / target-band handling;
3. `num_bands` / band-pool handling;
4. disentanglement outer and frozen windows;
5. response to Wannier90 warnings or iteration caps;
6. localization quality from WF spreads and spread components;
7. whether the old run accepted a result it should have rejected, while
   explicitly separating old-model responsibility from locked-runner behavior;
8. raw RMSE versus same-material reference error ratio, including whether either
   metric changes the run-quality interpretation;
9. whether any `TimeoutError` was transient or represented an actual incomplete
   run, using final completion and error-ratio evidence.

If the run shows evidence of avoidable issues, also cover:

10. the most likely specific failure chain.

Do NOT produce next-run recommendations, future playbooks, anti-loop rules, or
operational retry plans in this review.
Your job is diagnosis: what failed, where it failed, why the evidence
supports that diagnosis, and what remains uncertain.

Do not recommend changing fields that the old model could not set in
recipe_request.json. So do not recommend increasing num_iter,
dis_num_iter, conv_tol, dis_conv_tol, changing runner behavior, rerunning DFT,
or inspecting final .wout during the old model phase unless the original task
instructions explicitly allowed that action.

If a potentially useful action is outside the recipe_request.json schema, label
it as "outside_locked_recipe_contract" and do not present it as actionable advice
for the next recipe proposer.

For every decision review, answer all of these forensic questions:

- What exactly did the old run decide or claim?
- What evidence did the old run use at the time?
- What later evidence in `.wout`, verifier diagnostics, or trajectory contradicts
  or weakens that decision?
- Was the mistake avoidable without seeing the hidden reference recipe?
- Which exact step failed, if any: recipe writing, compile/preflight (look into
each of compile_attempt_*.json and compile_recipe_report.json, not inferred from later .wout/RMSE),
  `wannier90.x -pp`, `pw2wannier90.x`, final `wannier90.x` disentanglement,
  final `wannier90.x` localization, artifact collection, verifier scoring, or
  old-model interpretation? 
- What is the confidence level for any causal claim: `proven`,
  `strongly_supported`, `plausible_but_unproven`, or `unsupported`?
- What remains uncertain because the logs do not contain enough information?

IMPORTANT: This report will be reused as context for chemically similar future
runs. Therefore, prioritize advice safety over sounding decisive.

Classify every finding into one of these categories:
- CONTRACT FACT: directly follows from original_task_instructions.md,
  recipe_request.json schema, compile_recipe_report.json, compile_recipe_reports/compile_attempt_*.json,
  locked_runner.log, or runner behavior.
- DIRECT OBSERVATION: directly observed in .win, .wout, .amn_summary.json,
  .eig, .nnkp, diagnostics.json, trajectory.json, or logs.
- PLAUSIBLE INTERPRETATION: chemically or numerically plausible, but not proven
  by a controlled comparison.
- DO NOT GENERALIZE: a tempting explanation or next step that future runs should
  not treat as reliable.

Never present a PLAUSIBLE INTERPRETATION as a root cause. If a future run could
be harmed by treating it as a fact, put it in plausible_but_unproven_causes or
unsupported_or_overreaching_claims_to_avoid, not root_causes_supported.

Be especially suspicious of:

- accepting a result after checking only that `<seed>_hr.dat` exists;
- judging localization by average/total spread while ignoring max WF spread,
  spread outliers, final gradient, or iteration limits;
- declaring projections "excellent" or windows "robust" merely because they are
  chemically plausible;
- padding `num_wann` with duplicate same-site, same-angular-momentum
  projections without evidence that the channels are linearly independent;
- abandoning a physically motivated projection after a syntax, stale-file, or
  workflow error and falling back to `random`;
- claiming a root cause merely because it is a common failure mode;
- using a chemically plausible story as proof when the logs only show
  correlation.

For each diagnosis claim, cite the specific evidence that supports it. If the
files only show a symptom (for example high RMSE, spread outliers, SVD warning,
or nonconvergence) but do not identify the root cause, say "symptom observed;
root cause not proven" rather than inventing one.

Before calling a projection or window decision "bad", do the relevant arithmetic
from the available files when possible:

- projection count must equal `num_wann`;
- when `.amn_summary.json` exists, inspect full-pool and outer-window rank,
  rank sensitivity, frozen-window band counts, and projection-pair correlation;
- outer window per-k-point count must be at least `num_wann`;
- frozen window per-k-point count must be at most `num_wann`;
- coordinate-center claims must distinguish `f=` fractional coordinates from
  `c=` Cartesian coordinates;
- any claim about allowed recipe controls must be checked against staged
  workflow-contract files.

Be explicit about uncertainty. Do not claim causal proof when the files only
support diagnostic correlation. But make concrete judgments where evidence is
strong: projections, unconverged disentanglement, huge WF spreads,
fragile windows, or mismatch between claimed rationale and observed output.
Use "plausible but unproven" rather than "good" when a choice is chemically
reasonable but the run output shows poor localization or band interpolation.
The goal is to find avoidable scientific decision errors, not to assign credit
for parameters that merely look conventional.

The JSON report must have this shape:

```json
{{
  "material": "{material}",
  "case_id": "{case.case_id}",
  "run_root": "{case.job_metadata.get("run_root")}",
  "job_folder": "{case.job_dir.name}",
  "trial_folder": "{case.trial_dir.name}",
  "num_wann_from_job_folder": {json.dumps(num_wann)},
  "candidate_error_eV": "number | null",
  "reference_error_eV": "number | null",
  "candidate_to_reference_error_ratio": "number | null",
  "verdict": "good | mixed | bad | uncertain",
  "projection_verdict": "good | not_used | bad | uncertain",
  "decision_reviews": [
    {{
      "decision": "short name",
      "verdict": "good | mixed | bad | uncertain",
      "evidence": ["specific file-backed evidence"],
      "old_claim_or_decision": "what the old run said or did",
      "observed_failure_signal": "what later output showed",
      "failed_step": "none | recipe_writing | compile_preflight | wannier90_pp | pw2wannier90 | final_disentanglement | final_localization | artifact_collection | verifier_scoring | old_model_interpretation | locked_runner_behavior | unknown",
      "causal_confidence": "proven | strongly_supported | plausible_but_unproven | unsupported",
      "why": "specific evidence-backed explanation",
      "old_model_responsibility": "avoidable | not_avoidable | mixed | unknown",
      "uncertainty": "what remains uncertain, or null"
    }}
  ],
  "failure_chain": [
    {{
      "step": "specific failed step",
      "claim": "specific causal claim",
      "causal_confidence": "proven | strongly_supported | plausible_but_unproven | unsupported",
      "evidence": ["specific file-backed evidence"]
    }}
  ],
  "symptoms_observed": ["directly observed symptoms, not inferred causes"],
  "root_causes_supported": ["root causes with proven or strongly_supported evidence"],
  "plausible_but_unproven_causes": ["possible causes that should not be treated as facts"],
  "unsupported_or_overreaching_claims_to_avoid": ["claims not supported by staged files"],
  "workflow_constraints_relevant_to_diagnosis": ["constraints found in original task or workflow-contract files"],
  "evidence_gaps": ["missing files or missing diagnostics that limit the diagnosis"]
}}
```

If you cannot prove the exact root cause from the evidence, say so explicitly.
Do not fill the gap with a recommended recipe or future playbook.

Your final response should be a JSON object pointing to
`self_debug_report.md` and `self_debug_report.json`.
"""

def build_case(case: TrialCase) -> Path:
    material = case.material
    case_dir = OUTPUT_ROOT / material / case.case_id
    clean_dir(case_dir)
    case_files = case_dir / "case_files"

    win_dst = case_files / "artifacts" / "attempt_1" / f"{material}.win"
    try:
        copy_file(find_attempt_file(case.attempt_dir, material, ".win"), win_dst)
        win_copied = True
    except (SystemExit, FileNotFoundError, OSError):
        win_copied = False
    wout_dst = case_files / "artifacts" / "attempt_1" / f"{material}.wout"
    try:
        copy_file(find_attempt_file(case.attempt_dir, material, ".wout"), wout_dst)
        wout_copied = True
    except (SystemExit, FileNotFoundError, OSError):
        write_text(
            wout_dst,
            "No .wout file was present in the source artifacts for this case.\n",
        )
        wout_copied = False
    manifest_copied = copy_file_if_present(
        case.attempt_dir / "run_manifest.json",
        case_files / "artifacts" / "attempt_1" / "run_manifest.json",
    )

    amn_summary = summarize_amn_case(case)
    amn_summary_dst = (
        case_files
        / "artifacts"
        / "attempt_1"
        / f"{material}.amn_summary.json"
    )
    write_text(
        amn_summary_dst,
        json.dumps(amn_summary, indent=2, sort_keys=True) + "\n",
    )

    staged_optional_artifacts: list[dict[str, str]] = []
    for src in optional_attempt_evidence_files(case.attempt_dir, material):
        dst = case_files / "artifacts" / "attempt_1" / src.name
        if not copy_file_if_present(src, dst):
            continue
        staged_optional_artifacts.append(
            {
                "source": display_path(src),
                "staged": display_path(dst),
            }
        )

    staged_workflow_contract_artifacts: list[dict[str, str]] = []
    for src, relative_dst in optional_workflow_evidence_files(case.trial_dir):
        dst = case_files / "workflow_contract" / relative_dst
        if not copy_file_if_present(src, dst):
            continue
        staged_workflow_contract_artifacts.append(
            {
                "source": display_path(src),
                "staged": display_path(dst),
            }
        )

    trajectory_copied = copy_file_if_present(
        case.trial_dir / "agent" / "trajectory.json",
        case_files / "agent" / "trajectory.json",
    )

    diagnostics_src = case.trial_dir / "verifier" / "diagnostics.json"
    diagnostics_copied = copy_file_if_present(
        diagnostics_src,
        case_files / "verifier" / "diagnostics.json",
    )

    task_text, task_source = original_task_instructions(case)
    write_text(case_files / "original_task_instructions.md", task_text)

    quality_metrics = run_quality_metrics(case)
    write_text(
        case_files / "run_quality_metrics.json",
        json.dumps(quality_metrics, indent=2, sort_keys=True) + "\n",
    )

    metadata = {
        "material": material,
        "case_id": case.case_id,
        "run_root": case.job_metadata.get("run_root"),
        "job_folder": case.job_dir.name,
        "trial_folder": case.trial_dir.name,
        "source_job_path": display_path(case.job_dir),
        "source_trial_path": display_path(case.trial_dir),
        "source_attempt_path": display_path(case.attempt_dir),
        "job_metadata": case.job_metadata,
        "manifest_summary": manifest_summary(case.manifest),
        "original_task_instructions_source": task_source,
        "run_quality_metrics": quality_metrics,
        "win_copied": win_copied,
        "wout_copied": wout_copied,
        "run_manifest_copied": manifest_copied,
        "trajectory_copied": trajectory_copied,
        "amn_summary": {
            "staged": str(amn_summary_dst.relative_to(case_dir)),
            "status": amn_summary.get("status"),
            "source_file": amn_summary.get("source_file"),
        },
        "raw_numeric_artifact_presence": {
            "amn": has_attempt_file(case.attempt_dir, material, ".amn"),
            "mmn": has_attempt_file(case.attempt_dir, material, ".mmn"),
            "chk": has_attempt_file(case.attempt_dir, material, ".chk"),
            "hr_dat": has_attempt_file(case.attempt_dir, material, "_hr.dat"),
        },
        "raw_numeric_artifacts_intentionally_not_staged": {
            ".amn": "complete file is reduced deterministically into <material>.amn_summary.json",
            ".mmn": "raw neighbor-overlap matrix is not needed for the default LLM forensic review",
            ".chk": "binary Wannier90 checkpoint is not suitable as direct LLM evidence",
            "_hr.dat": "raw real-space Hamiltonian is not needed for the default forensic diagnosis",
        },
        "staged_optional_artifacts": staged_optional_artifacts,
        "staged_workflow_contract_artifacts": staged_workflow_contract_artifacts,
        "verifier_diagnostics_copied": diagnostics_copied,
        "verifier_diagnostics_source": display_path(diagnostics_src) if diagnostics_copied else None,
        "aggregate_inputs_intentionally_not_copied": [
            "jobs/num_wann_ordered_diagnostics_summary.json",
            "jobs/gemini_vs_reference_errors.xlsx",
            "jobs/gemini_failure_modes/failure_modes.csv",
        ],
    }
    write_text(case_files / "case_metadata.json", json.dumps(metadata, indent=2) + "\n")

    write_text(case_dir / "prompt.md", prompt_text(case))
    return case_dir

def report_is_nonempty(case_dir: Path) -> bool:
    md_path = case_dir / "self_debug_report.md"
    json_path = case_dir / "self_debug_report.json"

    if not md_path.is_file() or not json_path.is_file():
        return False

    if not md_path.read_text(encoding="utf-8").strip():
        return False

    raw_json = json_path.read_text(encoding="utf-8").strip()
    if not raw_json:
        return False

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return False

    # Minimal sanity checks that this is actually the requested diagnosis.
    if not isinstance(data, dict):
        return False
    if not data.get("verdict"):
        return False
    if not data.get("decision_reviews"):
        return False
    if not isinstance(data.get("failure_chain"), list):
        return False
    if not isinstance(data.get("symptoms_observed"), list):
        return False
    if not isinstance(data.get("evidence_gaps"), list):
        return False

    return True


def run_gemini(case_dir: Path) -> None:
    prompt = (case_dir / "prompt.md").read_text(encoding="utf-8")
    combined_log_path = case_dir / "gemini_stdout_stderr.txt"
    status_path = case_dir / "run_status.json"

    command = [
        GEMINI_BIN,
        "--yolo",
        "--skip-trust",
        f"--model={MODEL}",
        f"--prompt={prompt}",
    ]

    attempts: list[dict[str, Any]] = []
    max_attempts = 10

    for attempt_index in range(1, max_attempts + 1):
        # Remove stale outputs so success must come from this attempt.
        for output_name in ("self_debug_report.md", "self_debug_report.json"):
            output_path = case_dir / output_name
            if output_path.exists():
                output_path.unlink()

        attempt_log_path = case_dir / f"gemini_attempt_{attempt_index:02d}_stdout_stderr.txt"

        completed = subprocess.run(
            command,
            cwd=case_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        stdout_stderr = completed.stdout or ""
        attempt_log_path.write_text(stdout_stderr, encoding="utf-8")

        produced_nonempty_diagnosis = report_is_nonempty(case_dir)

        attempt_record = {
            "attempt": attempt_index,
            "returncode": completed.returncode,
            "attempt_log_path": attempt_log_path.name,
            "self_debug_report_md_exists": (case_dir / "self_debug_report.md").is_file(),
            "self_debug_report_json_exists": (case_dir / "self_debug_report.json").is_file(),
            "produced_nonempty_diagnosis": produced_nonempty_diagnosis,
        }
        attempts.append(attempt_record)

        status = {
            "command": command,
            "model": MODEL,
            "max_attempts": max_attempts,
            "success": produced_nonempty_diagnosis,
            "attempts": attempts,
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

        if produced_nonempty_diagnosis:
            combined_log_path.write_text(stdout_stderr, encoding="utf-8")
            return

        print(
            f"Gemini attempt {attempt_index}/{max_attempts} did not produce "
            f"a valid diagnosis for {case_dir.name}; retrying..."
        )

    combined_log_path.write_text(
        "\n\n".join(
            [
                f"===== ATTEMPT {record['attempt']} "
                f"returncode={record['returncode']} =====\n"
                f"{(case_dir / record['attempt_log_path']).read_text(encoding='utf-8')}"
                for record in attempts
            ]
        ),
        encoding="utf-8",
    )

    raise SystemExit(
        f"Gemini review failed after {max_attempts} attempts or did not write "
        f"a valid diagnosis: {case_dir}"
    )


def output_dir_for_case(case: TrialCase) -> Path:
    return OUTPUT_ROOT / case.material / case.case_id


def run_case(case: TrialCase) -> Path:
    case_dir = build_case(case)
    print(
        "Running Gemini review for "
        f"{case.material} case={case.case_id} in {case_dir}"
    )
    run_gemini(case_dir)
    case_files = case_dir / "case_files"
    if case_files.exists():
        shutil.rmtree(case_files)
        print(f"Removed staged case files after successful review: {case_files}")
    return case_dir

def candidate_materials_from_include_only_csv(
    path: Path,
) -> dict[str, list[str]]:
    """Read target_material,candidate_material rows using the downstream contract."""
    if not path.is_file():
        raise SystemExit(
            f"candidate include-only CSV does not exist: {path}"
        )

    candidates_by_target: dict[str, list[str]] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if (
            reader.fieldnames is None
            or "candidate_material" not in reader.fieldnames
        ):
            raise SystemExit(
                f"{path} must contain a candidate_material column"
            )

        target_column = (
            "target_material"
            if "target_material" in reader.fieldnames
            else "material"
            if "material" in reader.fieldnames
            else None
        )

        if target_column is None:
            raise SystemExit(
                f"{path} must contain target_material or material column"
            )

        for row in reader:
            target = (row.get(target_column) or "").strip()
            candidate = (row.get("candidate_material") or "").strip()

            # Ignore blank/spacer/incomplete rows.
            if not target or not candidate:
                continue

            candidates = candidates_by_target.setdefault(target, [])
            if candidate not in candidates:
                candidates.append(candidate)

    return candidates_by_target

def selected_materials() -> list[str]:
    mode = MATERIAL_SELECTION_MODE.strip().lower()

    if mode == "list":
        materials = [material.strip() for material in MATERIALS if material.strip()]
        if not materials:
            raise SystemExit(
                'MATERIALS is empty. Add material names at the top of the script '
                'or set MATERIAL_SELECTION_MODE = "chemically similar".'
            )
        if len(materials) != len(set(materials)):
            raise SystemExit("MATERIALS contains duplicate entries.")
        return materials

    if mode == "chemically similar":
        candidates_by_target = candidate_materials_from_include_only_csv(
            CANDIDATE_RUN_ERROR_TABLE
        )

        # Diagnose every unique candidate material exactly once.
        materials = sorted(
            {
                candidate
                for candidates in candidates_by_target.values()
                for candidate in candidates
                if candidate.strip()
            }
        )

        if not materials:
            raise SystemExit(
                "No candidate materials were found in "
                f"{display_path(CANDIDATE_RUN_ERROR_TABLE)}."
            )

        print(
            f"Loaded {len(candidates_by_target)} target material(s) and "
            f"{len(materials)} unique candidate material(s) from "
            f"{display_path(CANDIDATE_RUN_ERROR_TABLE)}."
        )

        return materials

    raise SystemExit(
        'MATERIAL_SELECTION_MODE must be either "chemically similar" or "list".'
    )


def collect_cases() -> list[TrialCase]:
    all_cases: list[TrialCase] = []

    materials = selected_materials()
    print(f"Material selection mode: {MATERIAL_SELECTION_MODE!r} ({len(materials)} material(s)).")

    for material in materials:
        cases = find_trial_cases(material)
        if len(cases) > 1:
            print(f"Found {len(cases)} trial folders for {material}; reviewing all of them.")
        all_cases.extend(cases)

    unique_cases: list[TrialCase] = []
    output_dirs: dict[Path, TrialCase] = {}
    for case in all_cases:
        output_dir = output_dir_for_case(case)

        if output_dir.exists():
            print(
                f"Skipping existing review for {case.material} "
                f"case={case.case_id}"
            )
            continue
        previous = output_dirs.get(output_dir)
        if previous is not None:
            if previous.material == case.material and previous.trial_dir == case.trial_dir:
                print(f"Skipping duplicate trial folder for {case.material}: {display_path(case.trial_dir)}")
                continue
            raise SystemExit(
                "Two trial cases resolve to the same output directory: "
                f"{previous.case_id} and {case.case_id} -> {output_dir}"
            )
        output_dirs[output_dir] = case
        unique_cases.append(case)

    return unique_cases

def main() -> None:
    if shutil.which(GEMINI_BIN) is None:
        raise SystemExit(
            f"Could not find {GEMINI_BIN!r} on PATH. Edit GEMINI_BIN at the top of this script "
            "or run it in the same environment where Gemini CLI is installed."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = collect_cases()

    if not cases:
        print("No cases to review.")
        return

    max_workers = min(MAX_CONCURRENT_GEMINI, len(cases))
    print(f"Running {len(cases)} Gemini review(s) with concurrency={max_workers}.")

    failures: list[tuple[TrialCase, BaseException]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_case = {pool.submit(run_case, case): case for case in cases}

        for future in as_completed(future_to_case):
            case = future_to_case[future]
            try:
                case_dir = future.result()
            except BaseException as exc:
                failures.append((case, exc))
                print(f"FAILED Gemini review for {case.material} case={case.case_id}: {exc}")
                continue

            print(f"Wrote expected outputs in {case_dir}")

    if failures:
        details = "\n".join(
            f"- {case.material} case={case.case_id}: {exc}"
            for case, exc in failures
        )
        raise SystemExit(
            f"{len(failures)} Gemini review(s) failed out of {len(cases)}:\n{details}"
        )


if __name__ == "__main__":
    main()
