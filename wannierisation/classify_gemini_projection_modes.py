#!/usr/bin/env python3
"""Classify Gemini submissions by projection mode from submitted .win files.

The classifier uses the same rule used for the ad hoc analysis:

* ``begin projections`` containing only ``random`` -> random_projection_runs
* any other ``begin projections`` block -> explicit_projection_runs
* no projections block -> none_or_implicit_projection_runs

It reads the 197-row Gemini-vs-reference spreadsheet and uses the bad-case
spreadsheet only to split the output into the ratio<=2 and ratio>2 cohorts.
The projection category itself is always determined from each submitted .win.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


CATEGORY_KEYS = (
    "random_projection_runs",
    "explicit_projection_runs",
    "none_or_implicit_projection_runs",
)


def empty_categories() -> dict[str, list[str]]:
    return {key: [] for key in CATEGORY_KEYS}


def find_submitted_win(repo_root: Path, job_folder: str, material: str) -> Path | None:
    """Return the submitted .win path for a material/job, if found."""
    job_root = repo_root / "jobs" / job_folder
    material_named = sorted(job_root.glob(f"*/artifacts/attempt_*/*{material}.win"))
    if material_named:
        return material_named[0]

    wins = sorted(job_root.glob("*/artifacts/attempt_*/*.win"))
    return wins[0] if wins else None


def projection_block(text: str) -> str | None:
    match = re.search(
        r"^\s*begin\s+projections\s*$([\s\S]*?)^\s*end\s+projections\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1) if match else None


def classify_win(win_path: Path | None) -> str:
    if win_path is None or not win_path.exists():
        return "none_or_implicit_projection_runs"

    text = win_path.read_text(errors="ignore")
    block = projection_block(text)
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


def add_row(
    categories: dict[str, list[str]],
    details: list[dict[str, Any]],
    repo_root: Path,
    row: pd.Series,
) -> None:
    material = str(row["material"])
    job_folder = str(row["job_folder"])
    win_path = find_submitted_win(repo_root, job_folder, material)
    category = classify_win(win_path)

    categories[category].append(material)
    details.append(
        {
            "material": material,
            "projection_category": category,
            "gemini_to_reference_ratio": float(row["gemini_to_reference_ratio"]),
            "gemini_error_eV": float(row["gemini_error_eV"]),
            "reference_error_eV": float(row["reference_error_eV"]),
            "job_folder": job_folder,
            "win_path": str(win_path) if win_path else None,
        }
    )


def sorted_categories(categories: dict[str, list[str]]) -> dict[str, list[str]]:
    return {key: sorted(categories[key]) for key in CATEGORY_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify Gemini projection modes by reading submitted .win files."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to this script's parent repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "jobs/gemini_projection_categories_from_win.json."
        ),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_path = (
        args.output
        if args.output is not None
        else repo_root / "jobs" / "gemini_projection_categories_from_win.json"
    )

    all_results_path = repo_root / "jobs" / "gemini_vs_reference_errors.xlsx"
    bad_cases_path = repo_root / "jobs" / "gemini_bad_case_breakdown.xlsx"

    all_df = pd.read_excel(all_results_path, sheet_name="Gemini vs Reference")
    bad_df = pd.read_excel(bad_cases_path, sheet_name="Per-material breakdown")
    bad_materials = set(bad_df["material"].astype(str))

    nonbad_df = all_df[~all_df["material"].astype(str).isin(bad_materials)].copy()
    bad_rows_df = all_df[all_df["material"].astype(str).isin(bad_materials)].copy()

    cohorts: dict[str, dict[str, Any]] = {}
    for cohort_name, description, df in (
        (
            "nonbad_ratio_le_2",
            "Materials in gemini_vs_reference_errors.xlsx excluding the 88 bad-case materials.",
            nonbad_df,
        ),
        (
            "bad_ratio_gt_2",
            "The 88 materials listed in gemini_bad_case_breakdown.xlsx.",
            bad_rows_df,
        ),
    ):
        categories = empty_categories()
        details: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            add_row(categories, details, repo_root, row)
        categories = sorted_categories(categories)
        details.sort(key=lambda item: item["material"])
        cohorts[cohort_name] = {
            "description": description,
            "counts": {key: len(categories[key]) for key in CATEGORY_KEYS},
            "materials": categories,
            "details": details,
        }

    combined_counts = {
        key: sum(cohort["counts"][key] for cohort in cohorts.values())
        for key in CATEGORY_KEYS
    }

    result = {
        "source": {
            "all_results": str(all_results_path.relative_to(repo_root)),
            "bad_case_breakdown": str(bad_cases_path.relative_to(repo_root)),
            "classification": (
                "Projection categories are determined by reading each submitted "
                ".win file, not by using the spreadsheet projection-mode column."
            ),
        },
        "cohorts": cohorts,
        "combined_counts": combined_counts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(output_path)
    print(json.dumps({name: c["counts"] for name, c in cohorts.items()}, indent=2))
    print(json.dumps({"combined_counts": combined_counts}, indent=2))


if __name__ == "__main__":
    main()
