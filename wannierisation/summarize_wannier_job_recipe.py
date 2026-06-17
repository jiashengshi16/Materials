#!/usr/bin/env python3
"""Print the Wannier recipe and evaluation metrics for one Harbor job folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
JOBS_ROOT = ROOT / "jobs"
EVAL_ROOT = ROOT / "evaluation" / "conductor_target_bands" / "gemini_job_results"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_job_dir(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()

    candidate = JOBS_ROOT / value
    if candidate.exists():
        return candidate.resolve()

    matches = sorted(JOBS_ROOT.glob(value))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise SystemExit(f"Ambiguous job folder {value!r}; matched {len(matches)} paths")
    raise SystemExit(f"Could not find job folder: {value}")


def relative(path: Path | None) -> str:
    if path is None:
        return "not found"
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def find_manifest(job_dir: Path) -> Path:
    direct = sorted(job_dir.glob("*/artifacts/attempt_*/run_manifest.json"))
    if direct:
        return direct[0]

    found = sorted(job_dir.rglob("run_manifest.json"))
    found = [path for path in found if "artifacts/artifacts" not in str(path)]
    if found:
        return found[0]
    raise SystemExit(f"Could not find run_manifest.json under {job_dir}")


def find_trial_dir(manifest_path: Path) -> Path:
    # .../TRIAL/artifacts/attempt_1/run_manifest.json
    if manifest_path.parent.name.startswith("attempt_") and manifest_path.parent.parent.name == "artifacts":
        return manifest_path.parent.parent.parent
    return manifest_path.parents[2]


def read_win_summary(win_path: Path) -> dict[str, Any]:
    lines = win_path.read_text(encoding="utf-8").splitlines()
    keys = {
        "num_wann",
        "num_bands",
        "dis_win_min",
        "dis_win_max",
        "dis_froz_min",
        "dis_froz_max",
        "dis_num_iter",
        "num_iter",
        "mp_grid",
    }
    params: dict[str, str] = {}
    projections: list[str] = []
    in_projections = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        lowered = stripped.lower()
        if lowered == "begin projections":
            in_projections = True
            continue
        if lowered == "end projections":
            in_projections = False
            continue
        if in_projections:
            projections.append(stripped)
            continue
        if "=" in stripped:
            key, value = (part.strip() for part in stripped.split("=", 1))
            if key in keys:
                params[key] = value

    return {"params": params, "projections": projections}


def find_eval_summary(material: str, trial_name: str) -> Path | None:
    exact = EVAL_ROOT / material / trial_name / "evaluation_summary.json"
    if exact.exists():
        return exact

    matches = sorted((EVAL_ROOT / material).glob("*/evaluation_summary.json"))
    if len(matches) == 1:
        return matches[0]
    return None


def format_number(value: Any, digits: int = 6) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.{digits}g}"


def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(
        f"- {label}: RMSE={format_number(metrics.get('rmse_eV'))} eV, "
        f"MAE={format_number(metrics.get('mae_eV'))} eV, "
        f"Max={format_number(metrics.get('max_abs_eV'))} eV, "
        f"occupied mismatches={metrics.get('occupied_count_mismatch_kpoints', 'n/a')}"
    )


def summarize(job_dir: Path) -> None:
    manifest_path = find_manifest(job_dir)
    manifest = read_json(manifest_path)
    trial_dir = find_trial_dir(manifest_path)
    attempt_dir = manifest_path.parent

    material = str(manifest.get("material_id") or manifest.get("material") or trial_dir.name.split("__", 1)[0])
    seed = str(manifest.get("seedname") or material)
    win_name = manifest.get("files", {}).get("win") if isinstance(manifest.get("files"), dict) else None
    win_path = attempt_dir / str(win_name or f"{seed}.win")
    if not win_path.exists():
        wins = sorted(attempt_dir.glob("*.win"))
        win_path = wins[0] if wins else win_path

    report_path = trial_dir / "artifacts" / "report.json"
    diagnostics_path = trial_dir / "verifier" / "diagnostics.json"
    eval_path = find_eval_summary(material, trial_dir.name)

    win_summary = read_win_summary(win_path) if win_path.exists() else {"params": {}, "projections": []}
    report = read_json(report_path) if report_path.exists() else {}
    diagnostics = read_json(diagnostics_path) if diagnostics_path.exists() else {}
    evaluation = read_json(eval_path) if eval_path else {}

    target_start = manifest.get("target_dft_band_start")
    target_end = manifest.get("target_dft_band_end")
    num_bands = manifest.get("num_bands")
    num_wann = manifest.get("num_wann")
    workflow = f"{num_bands} -> {num_wann}" if num_bands is not None and num_wann is not None else "n/a"
    disentangled = "yes" if num_bands != num_wann else "no"

    print(f"{material} ({trial_dir.name})")
    print("")
    print("Paths")
    print(f"- manifest: {relative(manifest_path)}")
    print(f"- win: {relative(win_path if win_path.exists() else None)}")
    print(f"- report: {relative(report_path if report_path.exists() else None)}")
    print(f"- diagnostics: {relative(diagnostics_path if diagnostics_path.exists() else None)}")
    print(f"- evaluation: {relative(eval_path)}")
    print("")
    print("Workflow")
    print(f"- status: {manifest.get('status', 'n/a')}; executed_successfully={manifest.get('executed_successfully', 'n/a')}")
    print(f"- target bands: {target_start}-{target_end}")
    print(f"- model: num_bands -> num_wann = {workflow}; disentanglement={disentangled}")
    print(f"- commands: {'; '.join(str(command) for command in manifest.get('commands', [])) or 'n/a'}")

    params = win_summary["params"]
    manifest_params = manifest.get("wannier_parameters", {})
    print("")
    print("Windows / Parameters")
    for key in ("num_wann", "num_bands", "dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max", "dis_num_iter", "num_iter", "mp_grid"):
        value = params.get(key)
        if value is None and isinstance(manifest_params, dict):
            value = manifest_params.get(key)
        if value is not None:
            print(f"- {key}: {value}")

    projections = win_summary["projections"] or manifest.get("projections", [])
    print("")
    print("Projections")
    if projections:
        for projection in projections:
            print(f"- {projection}")
    else:
        print("- n/a")

    notes = []
    if isinstance(report.get("runtime_notes"), list):
        notes.extend(str(note) for note in report["runtime_notes"])
    if isinstance(manifest.get("notes"), list):
        notes.extend(str(note) for note in manifest["notes"])
    print("")
    print("Notes")
    if notes:
        for note in dict.fromkeys(notes):
            print(f"- {note}")
    else:
        print("- n/a")

    print("")
    print("Results")
    if evaluation:
        print_metrics("on-mesh", evaluation.get("onmesh", {}))
        print_metrics("off-mesh", evaluation.get("offmesh", {}))
    elif diagnostics:
        print_metrics("off-mesh verifier", diagnostics)
        print(f"- reward: {format_number(diagnostics.get('reward'))}")
    else:
        print("- n/a")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_folder", help="Job folder path or name under jobs/")
    args = parser.parse_args()
    summarize(resolve_job_dir(args.job_folder))


if __name__ == "__main__":
    main()
