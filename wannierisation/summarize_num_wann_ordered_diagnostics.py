#!/usr/bin/env python3
"""Summarize verifier diagnostics for all jobs/num_wann_ordered* runs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_ROOT = ROOT / "jobs"
DEFAULT_OUTPUT_JSON = DEFAULT_JOBS_ROOT / "num_wann_ordered_diagnostics_summary.json"
DEFAULT_OUTPUT_MD = DEFAULT_JOBS_ROOT / "num_wann_ordered_diagnostics_summary.md"

NUM_WANN_JOB_RE = re.compile(
    r"^num_wann_ordered__(?P<timestamp>.+?)__pid(?P<pid>\d+)__"
    r"(?P<ordinal>\d+)__num_wann_(?P<num_wann>\d+)__(?P<material>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize verifier diagnostics across jobs/num_wann_ordered* folders."
    )
    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=DEFAULT_JOBS_ROOT,
        help="Root directory containing num_wann_ordered* job folders.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path for the JSON summary.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Path for the Markdown summary.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean_or_none(values),
        "median": median_or_none(values),
        "min": min_or_none(values),
        "max": max_or_none(values),
    }


def classify_diagnostics(data: dict[str, Any]) -> tuple[bool | None, str, str | None]:
    status = data.get("status")
    reward = data.get("reward")

    if status == "success":
        if not isinstance(reward, int | float):
            return None, "undeterminable", "diagnostics.json success has no numeric reward"
        if float(reward) == 0.0:
            return False, "failed", "diagnostics.json status is success but reward is 0.0"
        return True, "success", None

    if status == "failed":
        reason = data.get("error")
        return False, "failed", str(reason) if reason else None

    if status is None:
        return None, "undeterminable", "diagnostics.json has no status field"

    return None, "undeterminable", f"unrecognized status: {status!r}"


def parse_job_name(job_dir: Path) -> dict[str, Any]:
    match = NUM_WANN_JOB_RE.match(job_dir.name)
    if not match:
        return {"job_folder": job_dir.name}
    groups = match.groupdict()
    return {
        "job_folder": job_dir.name,
        "job_timestamp": groups["timestamp"],
        "pid": int(groups["pid"]),
        "ordinal": int(groups["ordinal"]),
        "num_wann_from_folder": int(groups["num_wann"]),
        "material_from_folder": groups["material"],
    }


def trial_dirs(job_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in job_dir.iterdir()
        if path.is_dir() and (path / "verifier").exists()
    )


def summarize_trial(job_dir: Path, trial_dir: Path, root: Path) -> dict[str, Any]:
    diagnostics_path = trial_dir / "verifier/diagnostics.json"
    record: dict[str, Any] = {
        **parse_job_name(job_dir),
        "trial_folder": trial_dir.name,
        "successful": None,
        "status": "undeterminable",
        "diagnostics_path": str(diagnostics_path.relative_to(root)),
    }

    if not diagnostics_path.exists():
        record["reason"] = "missing verifier/diagnostics.json"
        return record

    try:
        diagnostics = read_json(diagnostics_path)
    except json.JSONDecodeError as exc:
        record["reason"] = f"invalid JSON: {exc.msg}"
        return record

    if not isinstance(diagnostics, dict):
        record["reason"] = "diagnostics.json does not contain a JSON object"
        return record

    successful, status, reason = classify_diagnostics(diagnostics)
    record["successful"] = successful
    record["status"] = status
    if reason:
        record["reason"] = reason

    for key in (
        "material",
        "target_dft_band_start",
        "target_dft_band_end",
        "num_target_bands",
        "fermi_energy_eV",
        "num_offmesh_kpoints",
        "num_below_fermi_points",
        "reward",
        "rmse_eV",
        "mae_eV",
        "max_abs_eV",
        "p95_abs_eV",
        "executed_successfully",
    ):
        value = diagnostics.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            record[key] = value

    if "material" not in record and isinstance(record.get("material_from_folder"), str):
        record["material"] = record["material_from_folder"]
    if "num_target_bands" in record and record.get("num_wann_from_folder") is not None:
        record["num_wann_matches_num_target_bands"] = (
            record["num_target_bands"] == record["num_wann_from_folder"]
        )

    manifest_schema_errors = diagnostics.get("manifest_schema_errors")
    if isinstance(manifest_schema_errors, list):
        record["manifest_schema_error_count"] = len(manifest_schema_errors)

    return record


def summarize_missing_job(job_dir: Path, root: Path) -> dict[str, Any]:
    return {
        **parse_job_name(job_dir),
        "trial_folder": None,
        "successful": None,
        "status": "undeterminable",
        "diagnostics_path": None,
        "reason": "no trial folder with verifier directory found",
    }


def collect_results(jobs_root: Path) -> list[dict[str, Any]]:
    job_dirs = sorted(path for path in jobs_root.glob("num_wann_ordered*") if path.is_dir())
    results: list[dict[str, Any]] = []
    for job_dir in job_dirs:
        trials = trial_dirs(job_dir)
        if not trials:
            results.append(summarize_missing_job(job_dir, ROOT))
            continue
        for trial_dir in trials:
            results.append(summarize_trial(job_dir, trial_dir, ROOT))
    return results


def count_statuses(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "success": sum(1 for item in results if item["status"] == "success"),
        "failed": sum(1 for item in results if item["status"] == "failed"),
        "undeterminable": sum(1 for item in results if item["status"] == "undeterminable"),
        "total": len(results),
    }


def numeric_values(results: list[dict[str, Any]], key: str, *, successes_only: bool = False) -> list[float]:
    values = []
    for item in results:
        if successes_only and item.get("status") != "success":
            continue
        value = item.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def group_by_num_wann(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        value = item.get("num_wann_from_folder")
        if isinstance(value, int):
            grouped[value].append(item)

    rows = []
    for num_wann in sorted(grouped):
        items = grouped[num_wann]
        rows.append(
            {
                "num_wann": num_wann,
                "counts": count_statuses(items),
                "reward": numeric_summary(numeric_values(items, "reward", successes_only=True)),
                "rmse_eV": numeric_summary(numeric_values(items, "rmse_eV", successes_only=True)),
                "max_abs_eV": numeric_summary(numeric_values(items, "max_abs_eV", successes_only=True)),
            }
        )
    return rows


def build_summary(jobs_root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in results if item["status"] == "success"]
    failed_or_unknown = [item for item in results if item["status"] != "success"]
    return {
        "jobs_root": str(jobs_root),
        "job_folder_pattern": "num_wann_ordered*",
        "counts": count_statuses(results),
        "success_reward_summary": numeric_summary(numeric_values(results, "reward", successes_only=True)),
        "success_rmse_summary_eV": numeric_summary(numeric_values(results, "rmse_eV", successes_only=True)),
        "success_mae_summary_eV": numeric_summary(numeric_values(results, "mae_eV", successes_only=True)),
        "success_max_abs_summary_eV": numeric_summary(numeric_values(results, "max_abs_eV", successes_only=True)),
        "by_num_wann": group_by_num_wann(results),
        "successful_materials": [item.get("material") for item in successful],
        "failed_or_unknown": failed_or_unknown,
        "results": results,
    }


def format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Num-Wann Ordered Diagnostics Summary",
        "",
        f"Jobs root: `{summary['jobs_root']}`",
        "",
        "## Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    counts = summary["counts"]
    for key in ("success", "failed", "undeterminable", "total"):
        lines.append(f"| {key} | {counts[key]} |")

    lines.extend(
        [
            "",
            "## Successful Run Metrics",
            "",
            "| Metric | Count | Mean | Median | Min | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("Reward", "success_reward_summary"),
        ("RMSE eV", "success_rmse_summary_eV"),
        ("MAE eV", "success_mae_summary_eV"),
        ("Max abs eV", "success_max_abs_summary_eV"),
    ):
        stats = summary[key]
        lines.append(
            f"| {label} | {stats['count']} | {format_number(stats['mean'])} | "
            f"{format_number(stats['median'])} | {format_number(stats['min'])} | "
            f"{format_number(stats['max'])} |"
        )

    lines.extend(
        [
            "",
            "## By num_wann",
            "",
            "| num_wann | Total | Success | Failed | Unknown | Mean Reward | Mean RMSE eV | Mean Max Abs eV |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["by_num_wann"]:
        counts = row["counts"]
        lines.append(
            f"| {row['num_wann']} | {counts['total']} | {counts['success']} | "
            f"{counts['failed']} | {counts['undeterminable']} | "
            f"{format_number(row['reward']['mean'])} | "
            f"{format_number(row['rmse_eV']['mean'])} | "
            f"{format_number(row['max_abs_eV']['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Ordinal | Material | num_wann | Status | Reward | RMSE eV | MAE eV | Max Abs eV | Reason |",
            "|---:|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in sorted(summary["results"], key=lambda value: (value.get("ordinal") or 10**9, value.get("trial_folder") or "")):
        lines.append(
            f"| {format_number(item.get('ordinal'))} | {item.get('material') or item.get('material_from_folder') or '-'} | "
            f"{format_number(item.get('num_wann_from_folder'))} | {item['status']} | "
            f"{format_number(item.get('reward'))} | {format_number(item.get('rmse_eV'))} | "
            f"{format_number(item.get('mae_eV'))} | {format_number(item.get('max_abs_eV'))} | "
            f"{item.get('reason') or ''} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results = collect_results(args.jobs_root)
    summary = build_summary(args.jobs_root, results)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, summary)

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps({"success_reward_summary": summary["success_reward_summary"]}, indent=2))


if __name__ == "__main__":
    main()
