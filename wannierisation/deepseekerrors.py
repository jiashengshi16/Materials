#!/usr/bin/env python3

import csv
import json
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path("/home/jiasheng/WannierisationBenchmarking")
INPUT_CSV = BASE_DIR / "material_similarity_candidates_detailed.csv"
REFERENCE_CSV = BASE_DIR / "jobs" / "successful_run_errors.csv"
CONTROLLED_DIR = BASE_DIR / "jobsDeepseekProTerminus2Controlled"
CANDIDATES_DIR = BASE_DIR / "jobsDeepseekProTerminus2Candidates"
OUTPUT_CSV = BASE_DIR / "deepseek_error_ratios.csv"


OUTPUT_COLUMNS = [
    "target material",
    "(target material rmse error/reference rmse error)",
    "candidate material",
    "(candidate material rmse error/reference rmse error)",
]


def load_target_candidates(path):
    target_candidates = defaultdict(list)
    target_order = []

    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            target = row.get("target") or row.get("material")
            candidate = row.get("candidate") or row.get("candidate_material")
            if not target or not candidate:
                continue

            if target not in target_candidates:
                target_order.append(target)
            if candidate not in target_candidates[target]:
                target_candidates[target].append(candidate)

    return target_order, target_candidates


def load_reference_errors(path):
    reference_errors = {}

    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            material = row["material"]
            reference_error = row["reference_error_eV"]
            if not reference_error:
                continue

            value = float(reference_error)
            if material in reference_errors and reference_errors[material] != value:
                raise ValueError(
                    f"Conflicting reference errors for {material}: "
                    f"{reference_errors[material]} and {value}"
                )
            reference_errors[material] = value

    return reference_errors


def diagnostics_material(path, diagnostics):
    material = diagnostics.get("material")
    if material:
        return material

    # Expected layout: .../<material>__<run_id>/verifier/diagnostics.json
    run_dir = path.parent.parent.name
    return run_dir.split("__", 1)[0]


def index_rmse_errors(*roots, skipped_missing_rmse=None):
    errors_by_material = defaultdict(list)
    if skipped_missing_rmse is None:
        skipped_missing_rmse = set()

    for root in roots:
        for diagnostics_path in sorted(root.glob("**/verifier/diagnostics.json")):
            with diagnostics_path.open() as jsonfile:
                diagnostics = json.load(jsonfile)

            rmse = diagnostics.get("rmse_eV")
            if rmse is None:
                if diagnostics_path not in skipped_missing_rmse:
                    print(f"Skipping {diagnostics_path}: missing rmse_eV")
                    skipped_missing_rmse.add(diagnostics_path)
                continue

            material = diagnostics_material(diagnostics_path, diagnostics)
            errors_by_material[material].append(
                {
                    "rmse_eV": float(rmse),
                    "path": diagnostics_path,
                }
            )

    return errors_by_material


def ratio_rows(
    material,
    errors_by_material,
    reference_errors,
    missing_references,
    zero_references,
    missing_diagnostics,
):
    if material not in reference_errors:
        if material not in missing_references:
            print(f"Missing reference error for {material}; skipping")
            missing_references.add(material)
        return []

    reference_error = reference_errors[material]
    if reference_error == 0:
        if material not in zero_references:
            print(f"Reference error is zero for {material}; skipping")
            zero_references.add(material)
        return []

    if material not in errors_by_material:
        if material not in missing_diagnostics:
            print(f"Missing diagnostics for {material}; skipping")
            missing_diagnostics.add(material)
        return []

    return [
        error["rmse_eV"] / reference_error
        for error in errors_by_material[material]
    ]


def main():
    target_order, target_candidates = load_target_candidates(INPUT_CSV)
    reference_errors = load_reference_errors(REFERENCE_CSV)
    skipped_missing_rmse = set()
    target_errors = index_rmse_errors(
        CONTROLLED_DIR,
        skipped_missing_rmse=skipped_missing_rmse,
    )
    candidate_errors = index_rmse_errors(
        CANDIDATES_DIR,
        CONTROLLED_DIR,
        skipped_missing_rmse=skipped_missing_rmse,
    )
    missing_references = set()
    zero_references = set()
    missing_diagnostics = set()

    with OUTPUT_CSV.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for target in target_order:
            for ratio in ratio_rows(
                target,
                target_errors,
                reference_errors,
                missing_references,
                zero_references,
                missing_diagnostics,
            ):
                writer.writerow(
                    {
                        "target material": target,
                        "(target material rmse error/reference rmse error)": ratio,
                        "candidate material": "",
                        "(candidate material rmse error/reference rmse error)": "",
                    }
                )

            for candidate in target_candidates[target]:
                for ratio in ratio_rows(
                    candidate,
                    candidate_errors,
                    reference_errors,
                    missing_references,
                    zero_references,
                    missing_diagnostics,
                ):
                    writer.writerow(
                        {
                            "target material": "",
                            "(target material rmse error/reference rmse error)": "",
                            "candidate material": candidate,
                            "(candidate material rmse error/reference rmse error)": ratio,
                        }
                    )

    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
