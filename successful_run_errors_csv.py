#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from summarize_num_wann_ordered_diagnostics import (
        DEFAULT_REFERENCE_ROOTS,
        compute_reference_offmesh_rmse,
    )
except ModuleNotFoundError:
    DEFAULT_REFERENCE_ROOTS = ()
    compute_reference_offmesh_rmse = None


# =========================
# ONLY INPUTS YOU EDIT
# =========================

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [ROOT / "reruns", ROOT / "jobs"]
REFERENCE_SUMMARY_JSON = ROOT / "jobs/num_wann_ordered_diagnostics_summary.json"
OUTPUT_CSV = ROOT / "jobs/successful_run_errors.csv"

# =========================


NUM_WANN_RE = re.compile(r"__num_wann_(?P<num_wann>\d+)__(?P<material>.+)$")
HEADERS = [
    "material",
    "run_id",
    "num_wann",
    "reward",
    "gemini_error_eV",
    "reference_error_eV",
    "gemini_to_reference_ratio",
]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def as_float(value: Any) -> float | None:
    return float(value) if is_number(value) else None


def as_int(value: Any) -> int | None:
    if is_number(value):
        return int(value)
    return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def job_dir_from_diagnostics(path: Path) -> Path:
    # diagnostics.json lives at JOB/TRIAL/verifier/diagnostics.json.
    return path.parents[2]


def trial_dir_from_diagnostics(path: Path) -> Path:
    return path.parents[1]


def parse_job_folder(job_dir: Path) -> dict[str, str | int | None]:
    match = NUM_WANN_RE.search(job_dir.name)
    if not match:
        return {"material": None, "num_wann": None}
    return {
        "material": match.group("material"),
        "num_wann": int(match.group("num_wann")),
    }


def load_reference_lookup(path: Path) -> dict[tuple[str, int | None], float]:
    lookup: dict[tuple[str, int | None], float] = {}
    if not path.is_file():
        return lookup

    data = read_json(path)
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue

        material = item.get("material") or item.get("material_from_folder")
        if not isinstance(material, str):
            continue

        reference_error = as_float(item.get("reference_offmesh_rmse_eV"))
        if reference_error is None:
            continue

        num_wann = as_int(item.get("num_wann") or item.get("num_wann_from_folder") or item.get("num_target_bands"))
        lookup[(material, num_wann)] = reference_error
        lookup[(material, None)] = reference_error

    return lookup


def reference_error_for(
    material: str,
    num_wann: int | None,
    diagnostics: dict[str, Any],
    lookup: dict[tuple[str, int | None], float],
) -> float | None:
    if (material, num_wann) in lookup:
        return lookup[(material, num_wann)]
    if (material, None) in lookup:
        return lookup[(material, None)]

    target_start = as_int(diagnostics.get("target_dft_band_start"))
    target_end = as_int(diagnostics.get("target_dft_band_end"))
    fermi = as_float(diagnostics.get("fermi_energy_eV"))
    if target_start is None or target_end is None or fermi is None:
        return None

    if compute_reference_offmesh_rmse is None:
        return None

    try:
        computed_reference_error, _source = compute_reference_offmesh_rmse(
            material,
            target_start,
            target_end,
            fermi,
            list(DEFAULT_REFERENCE_ROOTS),
        )
    except Exception:
        return None

    lookup[(material, num_wann)] = computed_reference_error
    lookup[(material, None)] = computed_reference_error
    return computed_reference_error


def collect_rows() -> list[dict[str, Any]]:
    reference_lookup = load_reference_lookup(REFERENCE_SUMMARY_JSON)
    rows: list[dict[str, Any]] = []

    for search_root in SEARCH_ROOTS:
        if search_root.name == "jobs":
            diagnostics_paths = sorted(search_root.glob("*/verifier/diagnostics.json"))
        else:
            diagnostics_paths = sorted(search_root.glob("**/verifier/diagnostics.json"))

        for diagnostics_path in diagnostics_paths:
            diagnostics = read_json(diagnostics_path)
            if not isinstance(diagnostics, dict) or diagnostics.get("status") != "success":
                continue

            job_dir = job_dir_from_diagnostics(diagnostics_path)
            trial_dir = trial_dir_from_diagnostics(diagnostics_path)
            parsed = parse_job_folder(job_dir)

            material = diagnostics.get("material") or parsed["material"]
            if not isinstance(material, str):
                continue

            num_wann = (
                as_int(diagnostics.get("num_wann"))
                or as_int(diagnostics.get("num_target_bands"))
                or parsed["num_wann"]
            )
            reference_error = reference_error_for(material, num_wann, diagnostics, reference_lookup)
            gemini_error = as_float(diagnostics.get("rmse_eV"))
            ratio = (
                gemini_error / reference_error
                if gemini_error is not None and reference_error is not None and reference_error > 0
                else None
            )

            relative_trial = trial_dir.relative_to(ROOT)
            rows.append(
                {
                    "material": material,
                    "run_id": str(relative_trial),
                    "num_wann": num_wann,
                    "reward": as_float(diagnostics.get("reward")),
                    "gemini_error_eV": gemini_error,
                    "reference_error_eV": reference_error,
                    "gemini_to_reference_ratio": ratio,
                }
            )

    rows.sort(key=lambda row: (str(row["material"]), str(row["run_id"])))
    return rows


def write_csv(rows: list[dict[str, Any]]) -> Path:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return OUTPUT_CSV


def main() -> None:
    rows = collect_rows()
    if not rows:
        raise RuntimeError("No successful diagnostics found.")

    output_csv = write_csv(rows)
    comparable_count = sum(is_number(row.get("gemini_to_reference_ratio")) for row in rows)
    print(f"Wrote: {output_csv}")
    print(f"Rows written: {len(rows)}")
    print(f"Rows with Gemini/reference ratio: {comparable_count}")


if __name__ == "__main__":
    main()
