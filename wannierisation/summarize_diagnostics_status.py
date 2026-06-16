#!/usr/bin/env python3
"""Summarize verifier diagnostics status for a hardcoded jobs folder."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "jobs" / "2026-06-15__17-20-19"
OUTPUT_JSON = JOBS_DIR / "diagnostics_status_summary.json"

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

def summarize_job(job_dir: Path) -> dict[str, Any]:
    diagnostics_path = job_dir / "verifier" / "diagnostics.json"
    record: dict[str, Any] = {
        "subfolder": job_dir.name,
        "successful": None,
        "status": "undeterminable",
        "diagnostics_path": str(diagnostics_path.relative_to(ROOT)),
    }

    if not diagnostics_path.exists():
        record["reason"] = "missing verifier/diagnostics.json"
        return record

    try:
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        record["reason"] = f"invalid JSON: {exc.msg}"
        return record

    if not isinstance(diagnostics, dict):
        record["reason"] = "diagnostics.json does not contain a JSON object"
        return record

    successful, status, reason = classify_diagnostics(diagnostics)
    record["successful"] = successful
    record["status"] = status

    material = diagnostics.get("material")
    if isinstance(material, str):
        record["material"] = material

    reward = diagnostics.get("reward")
    if isinstance(reward, int | float):
        record["reward"] = float(reward)

    if reason:
        record["reason"] = reason

    return record


def main() -> None:
    job_dirs = sorted(path for path in JOBS_DIR.iterdir() if path.is_dir())
    results = [summarize_job(job_dir) for job_dir in job_dirs]

    success_rewards = [
        record["reward"]
        for record in results
        if record["status"] == "success" and isinstance(record.get("reward"), int | float)
    ]

    reward_summary = {
        "count": len(success_rewards),
        "mean": statistics.mean(success_rewards) if success_rewards else None,
        "median": statistics.median(success_rewards) if success_rewards else None,
    }

    counts = {
        "success": sum(1 for record in results if record["status"] == "success"),
        "failed": sum(1 for record in results if record["status"] == "failed"),
        "undeterminable": sum(1 for record in results if record["status"] == "undeterminable"),
        "total": len(results),
    }

    summary = {
        "jobs_dir": str(JOBS_DIR),
        "counts": counts,
        "success_reward_summary": reward_summary,
        "results": results,
    }

    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(json.dumps(counts, indent=2))
    print(json.dumps({"success_reward_summary": reward_summary}, indent=2))


if __name__ == "__main__":
    main()