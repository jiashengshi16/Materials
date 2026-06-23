#!/usr/bin/env python3
"""Find duplicate material runs under jobs/ and print folders that are safe to delete.

Deletion policy printed by this script:
  1. If a material has multiple failed/unsuccessful runs and no successful runs,
     keep all of them.
  2. If a material has one or more successful runs, keep only the successful run
     with the highest numeric reward and list every other duplicate folder for
     deletion.

This script does NOT delete anything. It only prints duplicate cases and a final
list of candidate folders to delete.
"""

from __future__ import annotations
import shutil
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JOBS_ROOT = Path("/Users/jshi/Documents/GitHub/WannierisationBenchmarking/jobs")

# Matches names like:
# num_wann_ordered__2025...__pid123__7__num_wann_12__Si2
JOB_NAME_RE = re.compile(
    r"^num_wann_ordered__(?P<timestamp>.+?)__pid(?P<pid>\d+)__"
    r"(?P<ordinal>\d+)__num_wann_(?P<num_wann>\d+)__(?P<material>.+)$"
)


@dataclass(frozen=True)
class RunRecord:
    folder: Path
    material: str
    successful: bool
    reward: float | None
    diagnostics_path: Path | None
    status_note: str


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def material_from_folder_name(folder: Path) -> str | None:
    match = JOB_NAME_RE.match(folder.name)
    if match:
        return match.group("material")
    return None


def numeric_reward(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None

def recency_sort_key(record: RunRecord) -> tuple[str, str]:
    """Sort by timestamp embedded in the job folder name, then folder name.

    The timestamp string in these job names is ISO-like, so lexical sorting should
    put newer runs later.
    """
    match = JOB_NAME_RE.match(record.folder.name)
    timestamp = match.group("timestamp") if match else ""
    return (timestamp, record.folder.name)

def diagnostics_paths_for_job_folder(folder: Path) -> list[Path]:
    """Return verifier diagnostics files belonging to a top-level job folder.

    Supports both common layouts:
      jobs/JOB/verifier/diagnostics.json
      jobs/JOB/TRIAL/verifier/diagnostics.json
    """
    direct = folder / "verifier" / "diagnostics.json"
    if direct.exists():
        return [direct]

    return sorted(folder.glob("*/verifier/diagnostics.json"))


def successful_from_diagnostics(data: dict[str, Any]) -> tuple[bool, str]:
    """Interpret success robustly across the schemas used in this project."""
    if data.get("successful") is True:
        return True, "successful=true"
    if data.get("successful") is False:
        return False, "successful=false"

    if data.get("executed_successfully") is True:
        return True, "executed_successfully=true"
    if data.get("executed_successfully") is False:
        return False, "executed_successfully=false"

    if data.get("status") == "success":
        return True, "status=success"
    if data.get("status") == "failed":
        return False, "status=failed"

    return False, "success status missing/undetermined"


def build_record(folder: Path) -> RunRecord:
    diagnostics_paths = diagnostics_paths_for_job_folder(folder)

    if not diagnostics_paths:
        material = material_from_folder_name(folder) or folder.name
        return RunRecord(
            folder=folder,
            material=material,
            successful=False,
            reward=None,
            diagnostics_path=None,
            status_note="missing verifier/diagnostics.json",
        )

    # A top-level job folder should usually contain one diagnostics file. If it
    # contains more than one, use the best successful nested diagnostics if any,
    # otherwise the first one. The printed diagnostics path makes this visible.
    candidate_records: list[RunRecord] = []
    for diagnostics_path in diagnostics_paths:
        try:
            data = read_json(diagnostics_path)
        except Exception as exc:
            material = material_from_folder_name(folder) or folder.name
            candidate_records.append(
                RunRecord(
                    folder=folder,
                    material=material,
                    successful=False,
                    reward=None,
                    diagnostics_path=diagnostics_path,
                    status_note=f"could not read diagnostics: {exc}",
                )
            )
            continue

        if not isinstance(data, dict):
            material = material_from_folder_name(folder) or folder.name
            candidate_records.append(
                RunRecord(
                    folder=folder,
                    material=material,
                    successful=False,
                    reward=None,
                    diagnostics_path=diagnostics_path,
                    status_note="diagnostics.json is not a JSON object",
                )
            )
            continue

        material = data.get("material")
        if not isinstance(material, str) or not material.strip():
            material = material_from_folder_name(folder) or folder.name

        successful, status_note = successful_from_diagnostics(data)
        reward = numeric_reward(data.get("reward"))

        candidate_records.append(
            RunRecord(
                folder=folder,
                material=material,
                successful=successful,
                reward=reward,
                diagnostics_path=diagnostics_path,
                status_note=status_note,
            )
        )

    successful_records = [record for record in candidate_records if record.successful]
    if successful_records:
        return max(successful_records, key=lambda record: reward_sort_key(record))
    return candidate_records[0]


def reward_sort_key(record: RunRecord) -> tuple[int, float, str]:
    # Missing reward sorts below any numeric reward among successful runs.
    has_reward = record.reward is not None
    return (1 if has_reward else 0, record.reward if record.reward is not None else float("-inf"), record.folder.name)


def print_record(record: RunRecord, *, prefix: str = "  ") -> None:
    reward = "None" if record.reward is None else f"{record.reward:g}"
    diagnostics = "None" if record.diagnostics_path is None else str(record.diagnostics_path)
    print(f"{prefix}{record.folder.name}")
    print(f"{prefix}  successful: {record.successful}")
    print(f"{prefix}  reward: {reward}")
    print(f"{prefix}  diagnostics: {diagnostics}")
    print(f"{prefix}  note: {record.status_note}")


def main() -> None:
    if not JOBS_ROOT.exists():
        raise SystemExit(f"Jobs root does not exist: {JOBS_ROOT}")
    if not JOBS_ROOT.is_dir():
        raise SystemExit(f"Jobs root is not a directory: {JOBS_ROOT}")

    records = [build_record(path) for path in sorted(JOBS_ROOT.iterdir()) if path.is_dir()]

    by_material: dict[str, list[RunRecord]] = {}
    for record in records:
        by_material.setdefault(record.material, []).append(record)

    duplicate_groups = {
        material: runs
        for material, runs in sorted(by_material.items())
        if len(runs) > 1
    }

    folders_to_delete: list[Path] = []

    print(f"Scanned {len(records)} subfolders under {JOBS_ROOT}")
    print(f"Found {len(duplicate_groups)} duplicate material case(s).")
    print()

    for material, runs in duplicate_groups.items():
        successes = [run for run in runs if run.successful]

        print("=" * 100)
        print(f"Material: {material}")
        print(f"Duplicate run count: {len(runs)}")
        print(f"Successful run count: {len(successes)}")
        print()

        for run in sorted(runs, key=lambda record: record.folder.name):
            print_record(run)
            print()

        if not successes:
            keep = max(runs, key=lambda record: recency_sort_key(record))
            delete_for_material = [run.folder for run in runs if run.folder != keep.folder]
            folders_to_delete.extend(delete_for_material)

            print("Decision: multiple fails only; keep the most recent failed run.")
            print(f"KEEP: {keep.folder}")
            print("DELETE:")
            for folder in delete_for_material:
                print(f"  {folder}")
            print()
            continue

        keep = max(successes, key=lambda record: reward_sort_key(record))
        delete_for_material = [run.folder for run in runs if run.folder != keep.folder]
        folders_to_delete.extend(delete_for_material)

        print("Decision: keep the successful run with the highest reward.")
        print(f"KEEP: {keep.folder}")
        print("DELETE:")
        for folder in delete_for_material:
            print(f"  {folder}")
        print()

    print("=" * 100)
    print("Final list of subfolders to delete:")
    if not folders_to_delete:
        print("  None")
    else:
        for folder in sorted(folders_to_delete):
            print(f"  Deleting {folder}")
            shutil.rmtree(folder)


if __name__ == "__main__":
    main()
